"""
Kimi Delta Attention using Flash Linear Attention (FLA) kernel.

```python
    kda = KimiDeltaAttentionFLA(hidden_size=1024, num_attention_heads=8, head_dim=128).cuda()
    batch_size = 2
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 1024).cuda()
    output = kda(x)
    loss = output.sum()
    loss.backward()
```
"""

from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.kda import chunk_kda
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    chunk_kda = None

from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear


def _init_weights(weight):
    """Simple Xavier uniform initialization."""
    nn.init.xavier_uniform_(weight)


@dataclass
class KimiDeltaAttentionCache:
    """Cache for KDA inference state.

    Caching S_t:
    Token 1: S_1 = K_1^T V_1
    Token 2: S_2 = β_2 ⊙ S_1 + K_2^T V_2
    Token 3: S_3 = β_3 ⊙ S_2 + K_3^T V_3
    ...
    ...
    Token t: S_t = β_t ⊙ S_{t-1} + K_t^T V_t
    Token t: S_t = β_1 ⊙ β_2 ⊙   ...   ⊙ β_t ⊙ K_1^TV_1 + β_2 ⊙  ...   ⊙ β_t ⊙ K_2^TV_2 + ... + K_t^TV_t

    Caching S_t also requires caching conv_{q/k/v}:
    Token 1:
        1. Project: Q_1, K_1, V_1 = projections(X_1)
        2. Conv: Q'_1 = Conv([0,0,0,Q_1])  # Need padding
        3. Attention: S_1 = K'_1^T V'_1
        4. Cache: 
        - recurrent_state = S_1
        - conv_buffer_q = [0, 0, Q_1]  # Save raw Q_1!
    Token 2:
        1. Project: Q_2, K_2, V_2 = projections(X_2)
        2. Conv: Q'_2 = Conv([0,0,Q_1,Q_2])  # Need Q_1 from buffer!
            - Can't get Q_1 from S_1 because S_1 = K'_1^T V'_1 is in float.
            - Save Q1 in the previous state, get Q_1 from conv_buffer_q
        3. Attention: S_2 = β_2 ⊙ S_1 + K'_2^T V'_2
        4. Cache:
        - recurrent_state = S_2
        - conv_buffer_q = [0, Q_1, Q_2]  ← Update buffer
    Token 3:
        1. Project: Q_3, K_3, V_3 = projections(X_3)
        2. Conv: Q'_3 = Conv([0,Q_1,Q_2,Q_3])  ← Need Q_1, Q_2 from buffer!
            - Can't reconstruct Q_1, Q_2 from S_2
            - Save Q1, Q2 in the previous state, get them from conv_buffer_q = [0, Q_1, Q_2]
        3. Attention: S_3 = β_3 ⊙ S_2 + K'_3^T V'_3
        4. Cache:
        - recurrent_state = S_3
        - conv_buffer_q = [Q_1, Q_2, Q_3]  ← Slide window

    At Step t, 
        1. Input X_t
        2. Project → Q_t, K_t, V_t  ← Need to cache these!
        3. Conv (needs previous Q, K, V) → Q'_t, K'_t, V'_t
        4. Attention (uses recurrent state) → O_t
        5. Update recurrent state: S_t = β_t ⊙ S_{t-1} + K'_t^T V'_t
    """
    # local -> tp -> single gpu
    recurrent_state: Optional[torch.Tensor] = None  # [B, H_local, K, V]; 
    conv_buffer_q: Optional[torch.Tensor] = None    # [B, H_local*K, kernel_size-1]
    conv_buffer_k: Optional[torch.Tensor] = None    # [B, H_local*K, kernel_size-1]
    conv_buffer_v: Optional[torch.Tensor] = None    # [B, H_local*K, kernel_size-1]


class ShortConvolution1D(nn.Module):
    """Causal depthwise convolution"""
    
    def __init__(self, hidden_size: int, kernel_size: int = 4, activation: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation
        
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size, # short conv sequence length
            groups=hidden_size, # Depthwise condition: each channel independent, no mixing.
            bias=True,
        )
    
    def forward(
        self,
        x_BTD: torch.Tensor,
        cache_BDK: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_BTD: [B, T, D] - input tensor
            cache_BDK: [B, D, K-1] - cached previous time steps
        
        Returns:
            y_BTD: [B, T, D] - output tensor
            new_cache_BDK: [B, D, K-1] - updated cache
        """
        B, T, D = x_BTD.shape
        x_BDT = x_BTD.transpose(1, 2)  # [B, D, T] for Conv1d
        
        if cache_BDK is not None: x_BDT = torch.cat([cache_BDK, x_BDT], dim=2)
        else: x_BDT = F.pad(x_BDT, (self.kernel_size - 1, 0))
        
        y_BDT = self.conv(x_BDT)
        y_BDT = y_BDT[:, :, -T:] # Trim to seq length, may be conv. bug, need more triage.
        
        if self.activation == "silu": y_BDT = F.silu(y_BDT)
        
        y_BTD = y_BDT.transpose(1, 2)
        new_cache_BDK = x_BDT[:, :, -(self.kernel_size - 1):].detach() # no detach(), backward() inf loop, need more triage.
        
        return y_BTD, new_cache_BDK


class RMSNormGated(nn.Module):
    """Head-wise RMSNorm with sigmoid gating."""
    
    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))
    
    def forward(self, x_BTHK: torch.Tensor, gate_BTHK: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_BTHK: [B, T, H, K] - input tensor
            gate_BTHK: [B, T, H, K] - gate tensor
        
        Returns:
            output_BTHK: [B, T, H, K] - normalized and gated output
        """
        var = x_BTHK.pow(2).mean(dim=-1, keepdim=True)
        x_norm_BTHK = x_BTHK * torch.rsqrt(var + self.eps)
        x_norm_BTHK = x_norm_BTHK * self.weight
        g_BTHK = torch.sigmoid(gate_BTHK)
        return x_norm_BTHK * g_BTHK


class KimiDeltaAttentionFLA(nn.Module):
    """Kimi Delta Attention using FLA's optimized kernel.
    
    > Input [B, T, hidden_size]
     > Q, K, V Projections (TP-split) by ColumnParallelLinear()
      > ShortConvolution1D
       > Decay Gates (g) & Beta (β)
        > FLA Chunk KDA Kernel (triton kernel)
         > RMSNormGated
          > Output Projection by RowParallelLinear
           > Output [B, T, hidden_size]

    Args:
        hidden_size: Model hidden dimension
        num_attention_heads: Total number of attention heads
        head_dim: Dimension per head (typically 128)
        model_parallel_config: Megatron parallel configuration
        short_conv_kernel_size: Kernel size for convolutions
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = 128,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        short_conv_kernel_size: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        if not FLA_AVAILABLE:
            raise ImportError(
                "flash-linear-attention is not found!"
            )
        
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.eps = eps
        
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size() if \
            torch.distributed.is_available() and torch.distributed.is_initialized() else 1
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank() if \
            torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        
        assert num_attention_heads % self.tp_size == 0, \
            f"num_heads ({num_attention_heads}) must be divisible by TP size ({self.tp_size})"
        self.num_heads_local = num_attention_heads // self.tp_size
        
        projection_dim = num_attention_heads * head_dim
        projection_dim_local = self.num_heads_local * head_dim
        
        # Create config with initialization disabled to avoid RNG tracker issues
        if model_parallel_config is None:
            self.model_parallel_config = ModelParallelConfig()
            self.model_parallel_config.perform_initialization = False
        else:
            self.model_parallel_config = model_parallel_config
        self.use_tp = self.tp_size > 1
        
        if self.use_tp:
            self.q_proj = ColumnParallelLinear(
                hidden_size, projection_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=False,
                bias=False,
            )
            self.k_proj = ColumnParallelLinear(
                hidden_size, projection_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=False,
                bias=False,
            )
            self.v_proj = ColumnParallelLinear(
                hidden_size, projection_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=False,
                bias=False,
            )
        else:
            self.q_proj = nn.Linear(hidden_size, projection_dim, bias=False)
            self.k_proj = nn.Linear(hidden_size, projection_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, projection_dim, bias=False)
        
        self.q_conv = ShortConvolution1D(projection_dim_local, short_conv_kernel_size, "silu")
        self.k_conv = ShortConvolution1D(projection_dim_local, short_conv_kernel_size, "silu")
        self.v_conv = ShortConvolution1D(projection_dim_local, short_conv_kernel_size, "silu")
        
        if self.use_tp:
            self.f_a_proj = ColumnParallelLinear(
                hidden_size, head_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=True,
                bias=False,
            )
            self.f_b_proj = ColumnParallelLinear(
                head_dim, projection_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=True,
                bias=False,
            )
        else:
            self.f_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.f_b_proj = nn.Linear(head_dim, projection_dim, bias=False)
        
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_attention_heads).uniform_(1.0, 16.0))
        )
        self.dt_bias = nn.Parameter(torch.zeros(projection_dim))
        
        if self.use_tp:
            self.b_proj = ColumnParallelLinear(
                hidden_size, num_attention_heads,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=True,
                bias=False,
            )
        else:
            self.b_proj = nn.Linear(hidden_size, num_attention_heads, bias=False)
        
        if self.use_tp:
            self.g_a_proj = ColumnParallelLinear(
                hidden_size, head_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=True,
                bias=False,
            )
            self.g_b_proj = ColumnParallelLinear(
                head_dim, projection_dim,
                config=self.model_parallel_config,
                init_method=_init_weights,
                gather_output=True,
                bias=False,
            )
        else:
            self.g_a_proj = nn.Linear(hidden_size, head_dim, bias=False)
            self.g_b_proj = nn.Linear(head_dim, projection_dim, bias=False)
        
        self.o_norm = RMSNormGated(head_dim, eps=eps)
        
        if self.use_tp:
            self.o_proj = RowParallelLinear(
                projection_dim, hidden_size,
                config=self.model_parallel_config,
                init_method=_init_weights,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
            )
        else:
            self.o_proj = nn.Linear(projection_dim, hidden_size, bias=False)
        
        self.recurrent_state = None
        self.conv_cache_q = None
        self.conv_cache_k = None
        self.conv_cache_v = None
    
    def tp_project(self, module, x_BTD):
        """tensor parallel projection.
        
        Args:
            module: Linear projection module
            x_BTD: [B, T, D] - input tensor
        
        Returns:
            output_BTD: [B, T, D'] - projected tensor
        """
        if self.use_tp:
            out, bias = module(x_BTD)
            return out if bias is None else out + bias
        return module(x_BTD)
    
    def forward(
        self,
        hidden_BTD: torch.Tensor,
        inference_cache: Optional[KimiDeltaAttentionCache] = None,
    ) -> torch.Tensor:
        """
        Equations: 
        1.
        Q = W_Q @ X
        K = W_K @ X
        V = W_V @ X

        2.
        Q' = Conv1D_causal(Q)
        K' = Conv1D_causal(K)
        V' = Conv1D_causal(V)

        3.
        f_a = W_f_a @ X
        g_raw = W_f_b @ f_a
        g_linear = g_raw · exp(A_log) + dt_bias
        g = log σ(g_linear)

        4. β = σ(W_β @ X)

        5. S_t = β_t ⊙ S_{t-1} + K'_t^T @ V'_t

        6. O_t = (1/√d_k) Q'_t @ S_t

        7.
        g2_a = W_g2_a @ X
        g2_raw = W_g2_b @ g2_a
        O_norm = RMSNorm(O) · σ(g2)

        Output = W_O @ O_norm


        Args:
            hidden_BTD: [B, T, D] - input hidden states
            inference_cache: Optional cache for incremental decoding
            
        Returns:
            output_BTD: [B, T, D] - output hidden states
        """

        # 1.
        B, T, D = hidden_BTD.shape
        # local -> tp -> single gpu
        q_BTHlocalK = self.tp_project(self.q_proj, hidden_BTD)  # [B, T, H_local * K]
        k_BTHlocalK = self.tp_project(self.k_proj, hidden_BTD)
        v_BTHlocalK = self.tp_project(self.v_proj, hidden_BTD)
        
        # 2.
        cache = inference_cache
        q_BTHlocalK, cache_q = self.q_conv(q_BTHlocalK, cache.conv_buffer_q if cache is not None else None)
        k_BTHlocalK, cache_k = self.k_conv(k_BTHlocalK, cache.conv_buffer_k if cache is not None else None)
        v_BTHlocalK, cache_v = self.v_conv(v_BTHlocalK, cache.conv_buffer_v if cache is not None else None)
        
        # 3.
        f_a_BTK = self.tp_project(self.f_a_proj, hidden_BTD)
        g_raw_BTHK = self.tp_project(self.f_b_proj, f_a_BTK)  # [B, T, H * K] (full, all heads)
        
        g_full_BTHK = g_raw_BTHK.view(B, T, self.num_heads, self.head_dim)
        start_idx = self.tp_rank * self.num_heads_local # manual slice
        end_idx = start_idx + self.num_heads_local
        g_local_BTHlocalK = g_full_BTHK[:, :, start_idx:end_idx, :]  # # manual slice, [B, T, H_local, K]

        # Manual Slice, A_log: [num_heads] -> [num_heads_local]
        A_log_local = self.A_log[start_idx:end_idx] 
        dt_bias_local = self.dt_bias.view(self.num_heads, self.head_dim)[start_idx:end_idx]
        
        # Add singleton dummy dim for element wise matrix op
        A_11Hlocal1 = A_log_local.exp().view(1, 1, self.num_heads_local, 1) 
        dt_bias_11HlocalK = dt_bias_local.view(1, 1, self.num_heads_local, self.head_dim)

        g_linear_BTHlocalK = g_local_BTHlocalK * A_11Hlocal1 + dt_bias_11HlocalK

        g_BTHlocalK = F.logsigmoid(g_linear_BTHlocalK)
        
        # 4.
        beta_BTH = self.tp_project(self.b_proj, hidden_BTD)  # [B, T, H] (full)
        beta_local_BTHlocal = beta_BTH[:, :, start_idx:end_idx]  # [B, T, H_local]
        beta_local_BTHlocal = beta_local_BTHlocal.sigmoid()
        
        # 5-6.
        q_BTHlocalK = q_BTHlocalK.view(B, T, self.num_heads_local, self.head_dim)
        k_BTHlocalK = k_BTHlocalK.view(B, T, self.num_heads_local, self.head_dim)
        v_BTHlocalK = v_BTHlocalK.view(B, T, self.num_heads_local, self.head_dim)
        
        
        initial_state_BHlocalKK = cache.recurrent_state if cache is not None else None
        o_BTHlocalK, final_state_BHlocalKK = chunk_kda(
            q_BTHlocalK, k_BTHlocalK, v_BTHlocalK, g_BTHlocalK, beta_local_BTHlocal,
            scale=self.head_dim ** -0.5,
            initial_state=initial_state_BHlocalKK,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        
        # 7.
        g2_a_BTK = self.tp_project(self.g_a_proj, hidden_BTD)
        g2_raw_BTHK = self.tp_project(self.g_b_proj, g2_a_BTK)  # [B, T, H * K] (full)
        

        g2_full_BTHK = g2_raw_BTHK.view(B, T, self.num_heads, self.head_dim)
        g2_local_BTHlocalK = g2_full_BTHK[:, :, start_idx:end_idx, :]  # [B, T, H_local, K]
        
        o_BTHlocalK = self.o_norm(o_BTHlocalK, g2_local_BTHlocalK)
        
        o_BTHlocalK = o_BTHlocalK.reshape(B, T, -1)  # [B, T, H_local * K]
        output_BTD = self.tp_project(self.o_proj, o_BTHlocalK)
        
        if inference_cache is not None:
            inference_cache.recurrent_state = final_state_BHlocalKK
            inference_cache.conv_buffer_q = cache_q
            inference_cache.conv_buffer_k = cache_k
            inference_cache.conv_buffer_v = cache_v
        else:
            self.recurrent_state = final_state_BHlocalKK
            self.conv_cache_q = cache_q
            self.conv_cache_k = cache_k
            self.conv_cache_v = cache_v
        
        return output_BTD
