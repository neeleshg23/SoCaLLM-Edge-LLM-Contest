
"""Implementation for Jamba architecture."""
import math
from tvm.relax.frontend.nn import spec

import dataclasses
from typing import Any, Dict, Optional, List, Union, Tuple

from tvm import te, tir  
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op

from mlc_llm import op as op_ext
from mlc_llm.nn import PagedKVCache, RopeMode
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class JambaConfig(ConfigBase):
    """Configuration for the Jamba model."""

    model_type: str = "jamba"
    keys_to_ignore_at_inference: List[str] = dataclasses.field(default_factory=lambda: ["past_key_values"])

    n_layers: int = 12    
    # Model dimensions
    ssm_state_size = 128
    hidden_size: int = 4096
    intermediate_size: int = 14336
    hidden_act: str = "silu"
    num_hidden_layers: int = 32
    pad_token_id: int = 50257
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    
    # Attention params
    attention_dropout: float = 0.0
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = None  # Will be computed in post_init
    
    # Normalization 
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    
    # Vocab and embeddings
    vocab_size: int = 65536
    tie_word_embeddings: bool = False
    
    # Training 
    use_cache: bool = True
    num_logits_to_keep: int = 1
    sliding_window: Optional[int] = None
    max_position_embeddings: int = 262144
    
    # MoE params
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    num_experts_per_tok: int = 2 
    num_experts: int = 16
    expert_layer_period: int = 2
    expert_layer_offset: int = 1
    
    # Attention/Mamba scheduling
    attn_layer_period: int = 8
    attn_layer_offset: int = 4
    
    # Mamba specific params
    use_mamba_kernels: bool = True
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_dt_rank: Union[int, str] = "auto"
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False

    # Deployment params
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    prefill_chunk_size: int = 0
    context_window_size: int = 0
    position_embedding_base: int = 0
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        # Compute head dimension if not provided
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Validate attention configuration
        assert self.num_attention_heads % self.num_key_value_heads == 0
        
        # Handle mamba dt_rank auto setting
        if self.mamba_dt_rank == "auto":
            self.mamba_dt_rank = math.ceil(self.hidden_size / 16)
            
        # Configure position embeddings
        if self.position_embedding_base == 0:
            if "rope_theta" in self.kwargs:
                self.position_embedding_base = self.kwargs.pop("rope_theta")
            else:
                self.position_embedding_base = 10000
                
        # Configure context window
        if self.context_window_size == 0:
            if "max_position_embeddings" in self.kwargs:
                self.context_window_size = self.kwargs["max_position_embeddings"]
            elif "max_sequence_length" in self.kwargs:
                self.context_window_size = self.kwargs["max_sequence_length"]
            else:
                self.context_window_size = self.max_position_embeddings

        # Configure prefill chunk size
        if self.prefill_chunk_size == 0:
            logger.info(
                "%s defaults to %d",
                bold("prefill_chunk_size"),
                min(self.context_window_size, 8192)
            )
            self.prefill_chunk_size = min(self.context_window_size, 8192)
            
        # Validate scheduling offsets
        self._check_supported_offset("attention", self.attn_layer_period, self.attn_layer_offset)
        self._check_supported_offset("expert", self.expert_layer_period, self.expert_layer_offset)

    def _check_supported_offset(self, property_: str, period: int, offset: int):
        if offset >= period:
            raise ValueError(
                f"{property_} layer offset ({offset}) must be smaller than {property_} layer period ({period})"
            )
            
    @property
    def layers_block_type(self) -> List[str]:
        """Return the type of each layer - either 'attention' or 'mamba'."""
        return [
            "attention" if i % self.attn_layer_period == self.attn_layer_offset else "mamba"
            for i in range(self.num_hidden_layers)
        ]

    @property 
    def layers_num_experts(self) -> List[int]:
        """Return number of experts for each layer."""
        return [
            self.num_experts if i % self.expert_layer_period == self.expert_layer_offset else 1
            for i in range(self.num_hidden_layers)
        ]
        
# class JambaMLP:
#     """Standard feed-forward MLP with SiLU activation."""
    
#     def __init__(self, config):
#         
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
        
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
#         self.act_fn = nn.SiLU()

#     def forward(self, hidden_state: Tensor) -> Tensor:
#         """MLP forward pass with gated SiLU activation."""
#         return self.down_proj(
#             self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state)
#         )

from tvm.relax import Var
from tvm.relax.op import matmul, add

from tvm.relax import op

def relu(x):
    """
    Implements ReLU using TVM Relax.
    """
    return op.maximum(x, op.const(0, dtype=x.dtype))


class JambaMLP:  # Ensure this is a subclass of nn.Module
    def __init__(self, config):
        


        # Define weights and biases using relay.var
        fc1_weight = relay.var("fc1_weight", shape=(config.hidden_size, config.hidden_size * 4), dtype="float32")
        fc1_bias = relay.var("fc1_bias", shape=(config.hidden_size * 4,), dtype="float32")
        fc2_weight = relay.var("fc2_weight", shape=(config.hidden_size * 4, config.hidden_size), dtype="float32")


    def forward(x):
        # First fully connected layer
        x = matmul(x, fc1_weight)  # Matrix multiplication
        x = add(x, fc1_bias)       # Add bias
        x = relu(x)                # ReLU activation

        # Second fully connected layer
        x = matmul(x, fc2_weight)  # Matrix multiplication
        x = add(x, fc2_bias)       # Add bias

        return x

    # def forward(self, x):
    #     x = op.matmul(x, fc1_weight) + fc1_bias
    #     x = op.relu(x)
    #     x = op.matmul(x, fc2_weight) + fc2_bias
    #     return x

import numpy as np
class JambaSparseMoeBlock:
    """Mixture of Experts layer with sparse routing."""
    
    def __init__(self, config):
        
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok


        # Initialize hidden_states using tvm.nd.array
        self.hidden_states = tvm.nd.array(np.zeros((1, self.hidden_dim), dtype="float32"))

        # Router: Replace nn.Linear with a TVM Relay dense layer

        self.router = relay.var("router", shape=(self.num_experts, self.hidden_dim))

        # Experts: Create a list of TVM-compatible expert networks
        self.experts = [
            self.create_expert(config, idx)
            for idx in range(self.num_experts)
        ]

    def create_expert(self, config, idx):
        # Replace JambaMLP with a Relay computation graph for an expert
        def expert_network(input_tensor):
            weight = relay.var(f"expert_{idx}_weight", shape=(config.hidden_dim, config.hidden_dim))
            bias = relay.var(f"expert_{idx}_bias", shape=(config.hidden_dim,))
            dense = relay.nn.dense(input_tensor, weight)
            output = relay.nn.bias_add(dense, bias)
            return relay.nn.relu(output)

        return expert_network

    # def forward(self, input_tensor):
    #     # Router logic: Compute scores for each expert
    #     router_output = relay.nn.dense(input_tensor, self.router_weight)

    #     # Apply softmax to obtain routing probabilities
    #     routing_probs = relay.nn.softmax(router_output, axis=-1)

    #     # Pass the input through the experts
    #     expert_outputs = [expert(input_tensor) for expert in self.experts]

    #     # Combine the expert outputs based on routing probabilities
    #     combined_output = relay.zeros_like(input_tensor)
    #     for idx, expert_output in enumerate(expert_outputs):
    #         weight = relay.strided_slice(routing_probs, begin=[0, idx], end=[-1, idx + 1])
    #         expert_weighted_output = relay.multiply(expert_output, weight)
    #         combined_output = relay.add(combined_output, expert_weighted_output)

    #     return combined_output

        # # Router and expert networks
        # self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        # self.experts = nn.ModuleList([JambaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        """MoE forward pass with top-k expert routing."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        
        # Router forward
        router_logits = self.router(hidden_states.reshape(-1, hidden_dim))
        routing_weights = nn.Softmax(router_logits, dtype='float32')
        routing_weights, selected_experts = routing_weights.topk(self.top_k, dim=-1)
        routing_weights = routing_weights.astype(hidden_states.dtype)
        
        # Expert computation
        final_states = nn.zeros_like(hidden_states.reshape(-1, hidden_dim))
        expert_mask = nn.one_hot(selected_experts, self.num_experts).transpose()
        
        for expert_idx, expert in enumerate(self.experts):
            idx, tokens = nn.where(expert_mask[expert_idx])
            if tokens.shape[0] > 0:
                # Process tokens through expert and scale by router weights
                tokens_states = expert(hidden_states[None, tokens].reshape(-1, hidden_dim))
                scaled_states = tokens_states * routing_weights[tokens, idx, None]
                final_states = final_states.index_add(0, tokens, scaled_states)
                
        return final_states.reshape(batch_size, sequence_length, hidden_dim), router_logits

class JambaAttention:
    """Multi-head attention with grouped query/key/value projections."""
    
    def __init__(
        self,
        config: JambaConfig,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None
    ):
        
        self.layer_idx = layer_idx
        self.hidden_size = num_heads * head_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / math.sqrt(head_dim)
        
        # Handle grouped QKV setup
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        
        if (head_dim * num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by"
                f"num_heads {num_heads} * head_dim {head_dim}"
            )
            
        # QKV projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False
        )
        
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
        layer_id: Optional[int] = None,
    ) -> Tensor:
        """
        Compute multi-head attention with cached key/values.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            paged_kv_cache: Optional KV cache for attention
            layer_id: Layer index for cache lookup
            
        Returns:
            Output tensor [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)  
        v = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2)
        
        # Update KV cache if provided
        if paged_kv_cache is not None:
            k, v = paged_kv_cache.update(k, v, layer_id)
            
        # Repeat KV heads if using groups
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            
        # Compute scaled attention
        attn_weights = op.matmul(q, k.transpose(2, 3)) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        # Softmax attention weights
        attn_weights = nn.Softmax(attn_weights, dim=-1, dtype='float32')
        attn_weights = attn_weights.astype(q.dtype)
        
        # Compute attention outputs
        attn_output = op.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

# import torch
# import torch.nn as nn

from tvm.relax import op, Var

class RMSNorm:
    def __init__(self, dim, eps=1e-6):
        
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def rms_norm(x, weight, eps=1e-6):
        """
        Custom RMSNorm implementation using TVM relax operations.

        Args:
            x: Input tensor of shape [..., dim]
            weight: Learnable scale parameter of shape [dim]
            eps: Small constant for numerical stability

        Returns:
            Normalized tensor
        """
        # Step 1: Compute mean of squared elements across the last dimension
        mean_squared = op.mean(op.power(x, 2), axis=-1, keepdims=True)

        # Step 2: Compute the RMS normalization denominator
        denom = op.sqrt(mean_squared + eps)

        # Step 3: Normalize input by dividing by the denominator
        normalized_x = x / denom

        # Step 4: Scale by the weight parameter
        output = normalized_x * weight

        return output


from tvm.relax import Var, op
from tvm import te
from tvm import relay, relax
from tvm.relay import analysis, transform   

class JambaMambaMixer:
    def __init__(self, config, layer_idx):
        
        self.config = config
        self.layer_idx = layer_idx  # Store the layer index for use if needed


        # Define the variables using relay.var
        self.A_log = relay.var("A_log", shape=(config.intermediate_size, config.ssm_state_size), dtype="float32")
        self.D = relay.var("D", shape=(config.intermediate_size,), dtype="float32")

        
        # Core dimensions
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.time_step_rank = config.mamba_dt_rank
        # self.layer_idx = config.layer_idx
        
    def conv1d_with_conv2d(data, weight, bias=None, stride=1, padding=0, dilation=1):
        # Reshape input to 4D for Conv2d simulation
        data = tvm.topi.expand_dims(data, axis=-1)  # Add a trailing dimension
        weight = tvm.topi.expand_dims(weight, axis=-1)  # Add a trailing dimension to the kernel

        # Perform 2D convolution
        conv2d_output = tvm.topi.nn.conv2d(
            data,
            weight,
            strides=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )

        # Add bias if available
        if bias is not None:
            conv2d_output += tvm.topi.expand_dims(bias, axis=0)

        # Remove the trailing dimension to return 1D shape
        return tvm.topi.squeeze(conv2d_output, axis=-1)


        # Conv1D layer
        self.conv1d = op.nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=config.mamba_d_conv,
            groups=self.intermediate_size,
            padding=config.mamba_d_conv - 1,
            bias=config.mamba_conv_bias
        )
        
        # Projections and core parameters
        self.in_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size * 2, 
            bias=config.mamba_proj_bias
        )
        self.x_proj = nn.Linear(
            self.intermediate_size,
            self.time_step_rank + self.ssm_state_size * 2,
            bias=False
        )
        self.dt_proj = nn.Linear(
            self.time_step_rank,
            self.intermediate_size,
            bias=True
        )

        # Create A using TVM's range operation
        A = op.arange(1, self.ssm_state_size + 1, dtype="float32")[None, :]  # Shape: (1, ssm_state_size)

        # Expand A to match (intermediate_size, ssm_state_size)
        A_expanded = op.broadcast_to(A, (self.intermediate_size, self.ssm_state_size))

        # Take the logarithm of A
        A_log = op.log(A_expanded)

        # Declare A_log as a Relax variable for future computation
        self.A_log = Var("A_log", shape=(self.intermediate_size, self.ssm_state_size), dtype="float32")

        # Create a tensor of ones for D
        D_init = op.ones((self.intermediate_size,), dtype="float32")

        # Declare D as a Relax variable
        self.D = Var("D", shape=(self.intermediate_size,), dtype="float32")

        
        # Initialize S4D parameters
        A = op.arange(1, self.ssm_state_size + 1)[None, :]
        A = A.expand(self.intermediate_size, -1)
        self.A_log = nn.Parameter(op.log(A))
        self.D = nn.Parameter(op.ones(self.intermediate_size))
        
        # Output projection
        self.out_proj = nn.Linear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mamba_proj_bias
        )
        
        # Layer norms for stability
        # self.dt_layernorm = nn.RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        # self.b_layernorm = nn.RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        # self.c_layernorm = nn.RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        # Replace nn.RMSNorm with TVM variables for weights
        self.dt_weight = Var("dt_weight", shape=(self.time_step_rank,), dtype="float32")
        self.b_weight = Var("b_weight", shape=(self.ssm_state_size,), dtype="float32")
        self.c_weight = Var("c_weight", shape=(self.ssm_state_size,), dtype="float32")

        
        self.act = nn.SiLU()

    def forward(
        self,
        hidden_states: Tensor,
        paged_kv_cache: Optional[PagedKVCache] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Project and split into hidden states and gate
        projected = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected.chunk(2, dim=1)
        
        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 2. Apply causal convolution
        hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        # 3. State Space Model transformation
        # Get parameters from input-dependent projections
        ssm_params = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = op.split(
            ssm_params,
            [self.time_step_rank, self.ssm_state_size, self.ssm_state_size],
            dim=-1
        )
        
        # Apply layer norms
        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)
        
        # Get discrete parameters
        discrete_time = nn.functional.softplus(
            self.dt_proj(time_step)
        ).transpose(1, 2)
        
        # Core state space computation
        A = -op.exp(self.A_log.float())
        discrete_A = op.exp(A[None, :, None, :] * discrete_time[:, :, :, None])
        discrete_B = discrete_time[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # Run recurrence
        ssm_state = op.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            dtype=dtype, device=hidden_states.device
        )
        scan_outputs = []
        
        for i in range(seq_len):
            # Update state and compute output for each step
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = op.matmul(
                ssm_state.to(dtype),
                C[:, i, :].unsqueeze(-1)
            )
            scan_outputs.append(scan_output[:, :, 0])

        # Combine outputs
        scan_output = op.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = scan_output * self.act(gate)
        
        # 4. Final projection
        return self.out_proj(scan_output.transpose(1, 2))

class JambaDecoderLayer:
    """
    Base decoder layer combining either attention or mamba with MLP/MoE.
    Handles both attention and mamba variants through a unified interface.
    """
    def __init__(self, config: JambaConfig, layer_idx: int, is_mamba: bool = False):
        
        num_experts = config.layers_num_experts[layer_idx]
        
        # Main computation block (either attention or mamba)
        if is_mamba:
            self.block = JambaMambaMixer(config=config, layer_idx=layer_idx)
        else:
            self.block = JambaAttention(
                config=config,
                layer_idx=layer_idx,
                num_heads=config.num_attention_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads
            )
            
        # Feed forward network (MLP or MoE)
        self.feed_forward = (
            JambaSparseMoeBlock(config) if num_experts > 1 
            else JambaMLP(config)
        )
        
        # Layer norms
        self.input_layernorm = nn.RMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        self.pre_ff_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        # Layer properties
        self.is_mamba = is_mamba

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
        layer_id: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through either attention or mamba block followed by FFN.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Optional mask tensor 
            paged_kv_cache: Optional PagedKVCache for attention
            layer_id: Layer index for cache lookup
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_dim]
                - Optional router logits if using MoE
        """
        # First block (attention/mamba) with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        if self.is_mamba:
            hidden_states = self.block(
                hidden_states,
                attention_mask=attention_mask,
                paged_kv_cache=paged_kv_cache
            )
        else:
            hidden_states = self.block(
                hidden_states,
                attention_mask=attention_mask, 
                paged_kv_cache=paged_kv_cache,
                layer_id=layer_id
            )
            
        hidden_states = residual + hidden_states

        # Feed forward with residual
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_output = self.feed_forward(hidden_states)
        
        # Handle MoE vs MLP outputs
        if isinstance(ff_output, tuple):
            hidden_states, router_logits = ff_output
        else:
            hidden_states, router_logits = ff_output, None
            
        hidden_states = residual + hidden_states
        return hidden_states, router_logits

# Factory function for creating appropriate decoder layer
def create_decoder_layer(
    config: JambaConfig,
    layer_idx: int
) -> JambaDecoderLayer:
    """Creates attention or mamba decoder layer based on config."""
    is_mamba = (
        layer_idx % config.attn_layer_period != config.attn_layer_offset
    )
    return JambaDecoderLayer(config, layer_idx, is_mamba=is_mamba)

class JambaModel:
    """
    Jamba model combining attention and mamba layers for hybrid sequence modeling.
    """
    def __init__(self, config: JambaConfig):
        
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # Token embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            # padding_idx=config.pad_token_id
        )
        
        # Create decoder layers using TVM
        self.layers = [
            create_decoder_layer(config, idx)
            for idx in range(config.num_hidden_layers)
        ]
        
        # Final normalization
        self.final_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )

    def _create_attention_mask(
        self,
        attention_mask: Optional[Tensor],
        seq_length: int,
        dtype: str,
    ) -> Optional[Tensor]:
        """Create causal attention mask."""
        if attention_mask is None:
            return None
            
        # Create causal mask
        mask = op.triu(
            op.ones((seq_length, seq_length), dtype=dtype) * -float("inf"),
            k=1
        )
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            mask = mask.masked_fill(attention_mask[:, None, None, :] == 0, -float("inf"))
            
        return mask
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        paged_kv_cache: Optional[PagedKVCache] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through Jamba model.
        
        Args:
            input_ids: Input token IDs, optional if inputs_embeds provided
            inputs_embeds: Pre-computed embeddings, optional if input_ids provided
            attention_mask: Optional attention mask
            paged_kv_cache: Optional PagedKVCache for attention/mamba states
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Router logits if using MoE
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Must provide either input_ids or inputs_embeds")
            
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        hidden_states = inputs_embeds
        all_router_logits = []
        
        # Create attention mask if needed
        if attention_mask is not None:
            attention_mask = self._create_attention_mask(
                attention_mask,
                hidden_states.shape[1],
                hidden_states.dtype
            )
            
        # Process through decoder layers
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, router_logits = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                paged_kv_cache=paged_kv_cache,
                layer_id=idx
            )
            
            if router_logits is not None:
                all_router_logits.append(router_logits)
                
        # Final normalization
        hidden_states = self.final_layernorm(hidden_states)
        
        if all_router_logits:
            return hidden_states, tuple(all_router_logits)
        return hidden_states, None

from tvm import relay
import tvm
import torch
import torch.nn as nn

class JambaForCausalLM:
    def __init__(self, config):
        
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.embedding_dim=config.hidden_size
        # self.vocab_size=config.vocab_size
        self.embedding_weight = relay.var("embedding_weight", shape=(config.vocab_size, config.hidden_size), dtype="float32")

        # self.embedding_layer = relay.var("embedding_weight", shape=(config.vocab_size, config.hidden_dim), dtype="float32")
        # self.embedding_weight = self.embedding_layer
        
        # Initialize a simple transformer decoder
        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.n_layers,
            num_encoder_layers=0,  # For decoder-only architecture
            num_decoder_layers=config.n_layers,
            dim_feedforward=4 * config.hidden_size,
        )

        # Initialize the output layer for logits
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        self.config = config
        self.model = JambaModel(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.dtype = "float32"
        
        # LM head reuses embedding weights if specified
        if config.tie_word_embeddings:
            self.lm_head = lambda x: nn.Linear.reuse_weights(
                x, self.model.embed_tokens.weight.T, bias=None
            )
        else:
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False
            )

    def export_tvm(self, spec):
        import tvm
        from tvm import relay
        import numpy as np

        # Check if embedding_weight is symbolic or concrete
        if isinstance(self.embedding_weight, relay.Var):
            # If symbolic, use directly
            print("Using symbolic embedding_weight.")
            embedding_const = self.embedding_weight
        else:
            # Ensure embedding_weight is a concrete constant
            if not isinstance(self.embedding_weight, tvm.nd.NDArray):
                if isinstance(self.embedding_weight, np.ndarray):
                    self.embedding_weight = tvm.nd.array(self.embedding_weight.astype("float32"))
                else:
                    raise ValueError("self.embedding_weight must be a numpy.ndarray, tvm.nd.NDArray, or relay.Var.")
            embedding_const = relay.const(self.embedding_weight, dtype="float32")

        # Define input IDs and embeddings
        input_ids = relay.var("input_ids", shape=(1, self.config.vocab_size), dtype="int32")
        input_embeds = relay.take(embedding_const, input_ids, axis=0)

        # Define further Jamba logic...
        return


        # Jamba model logic
        # Replace placeholder logic with actual Jamba computation
        hidden_states = input_embeds
        for layer_idx in range(self.config.num_hidden_layers):
            layer_weight = relay.const(self.layer_weights[layer_idx], dtype="float32")
            layer_bias = relay.const(self.layer_biases[layer_idx], dtype="float32")
            hidden_states = relay.nn.dense(hidden_states, layer_weight)
            hidden_states = relay.nn.bias_add(hidden_states, layer_bias)
            hidden_states = relay.nn.relu(hidden_states)  # Example activation function

        # Project hidden states to vocabulary logits
        output_weight = relay.const(self.output_weight, dtype="float32")
        output_bias = relay.const(self.output_bias, dtype="float32")
        logits = relay.nn.dense(hidden_states, output_weight)
        logits = relay.nn.bias_add(logits, output_bias)

        # Define the computation as a Relay function
        output = logits
        func = relay.Function([input_ids], output)
        mod = tvm.IRModule({"main": func})

        # Compile the model
        target = "cuda"
        params = {
            "embedding_weight": self.embedding_weight,
            "output_weight": self.output_weight,
            "output_bias": self.output_bias,
        }
        for layer_idx in range(self.config.num_hidden_layers):
            params[f"layer_{layer_idx}_weight"] = self.layer_weights[layer_idx]
            params[f"layer_{layer_idx}_bias"] = self.layer_biases[layer_idx]

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        # Export the compiled model to a shared library
        lib.export_library("jamba_model.so")

        return mod, params




    def get_logits(self, hidden_states: Tensor) -> Tensor:
        """Get logits from hidden states, ensuring float32 output."""
        logits = self.lm_head(hidden_states)
        if logits.dtype != "float32":
            logits = logits.astype("float32")
        return logits
        
    def embed(self, input_ids: Tensor) -> Tensor:
        """Embed input tokens."""
        return self.model.embed_tokens(input_ids)

    def prefill(
        self,
        input_embed: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process full input sequence.
        
        Args:
            input_embed: Input embeddings [1, seq_len, hidden_size]
            attention_mask: Optional attention mask [1, seq_len]
            
        Returns:
            Output logits [1, seq_len, vocab_size]
        """
        hidden_states, _ = self.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
        )
        return self.get_logits(hidden_states)

    def decode(
        self,
        input_embed: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Process single token for generation.
        
        Args:
            input_embed: Input embedding [1, 1, hidden_size]
            attention_mask: Optional attention mask [1, 1]
            
        Returns:
            Output logits [1, 1, vocab_size]
        """
        hidden_states, _ = self.model(
            inputs_embeds=input_embed,
            attention_mask=attention_mask,
        )
        return self.get_logits(hidden_states)

    def batch_prefill(
        self,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
        logit_positions: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Batch process multiple sequences.
        
        Args:
            input_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask [batch_size, seq_len]
            logit_positions: Optional positions to compute logits for [batch_size]
        
        Returns:
            Output logits for specified positions
        """
        hidden_states, _ = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        
        if logit_positions is not None:
            hidden_states = op.take(hidden_states, logit_positions, axis=1)
            
        return self.get_logits(hidden_states)

    def batch_decode(
        self,
        input_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Batch process single tokens for multiple sequences.
        
        Args:
            input_embeds: Input embeddings [batch_size, 1, hidden_size]  
            attention_mask: Optional attention mask [batch_size, 1]
            
        Returns:
            Output logits [batch_size, 1, vocab_size]
        """
        hidden_states, _ = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        )
        return self.get_logits(hidden_states)


    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.long()
        # Convert input IDs to embeddings
        embedded = self.embedding(input_ids)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        # Transformer decoder
        transformer_output = self.transformer(
            src=embedded.permute(1, 0, 2),  # PyTorch Transformer expects (seq_len, batch, d_model)
            tgt=embedded.permute(1, 0, 2),
            src_key_padding_mask=(attention_mask == 0),
        )

        # Project to vocabulary size
        logits = self.output_layer(transformer_output.permute(1, 0, 2))  # Back to (batch, seq_len, vocab_size)

        return logits
        
    def get_default_spec(self):
        """Get TVM module specifications."""
        mod_spec = {
            "embed": {
                "input_ids": spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "attention_mask": spec.Tensor([1, "seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "attention_mask": spec.Tensor([1, 1], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": spec.Tensor(["batch_size", "seq_len", self.hidden_size], self.dtype),
                "attention_mask": spec.Tensor(["batch_size", "seq_len"], "int32"),
                "logit_positions": spec.Tensor(["batch_size"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": spec.Tensor(["batch_size", 1, self.hidden_size], self.dtype),
                "attention_mask": spec.Tensor(["batch_size", 1], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        return spec.ModuleSpec.from_raw(mod_spec, self)
