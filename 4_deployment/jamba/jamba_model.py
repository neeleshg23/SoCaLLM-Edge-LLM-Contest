import dataclasses
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import tvm
from tvm import te, tir
from tvm.relax import BlockBuilder
from tvm.relax import op as base_op
from tvm.relax.frontend import nn
from tvm.relax.frontend.nn import Tensor, op
from tvm.script import tir as T

from mlc_llm import op as op_ext
from mlc_llm.nn import RopeMode
from mlc_llm.nn.rnn_state import RNNState
from mlc_llm.support import logging
from mlc_llm.support import tensor_parallel as tp
from mlc_llm.support.config import ConfigBase
from mlc_llm.support.style import bold

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class JambaConfig(ConfigBase):
    """Configuration for the Jamba model."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    rms_norm_eps: float
    vocab_size: int
    hidden_activation: Optional[str] = None
    position_embedding_base: int = 0
    context_window_size: int = 0
    prefill_chunk_size: int = 0
    tensor_parallel_shards: int = 1
    max_batch_size: int = 1
    mamba_expand: int = 2
    num_experts: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_dt_rank: int = 256
    expert_layer_period: int = 1
    expert_layer_offset: int = 2
    tie_word_embeddings: bool = False
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.hidden_activation is None:
            self.hidden_activation = self.kwargs.get("hidden_act", "gelu")
        if self.position_embedding_base == 0:
            self.position_embedding_base = self.kwargs.pop("rope_theta", 10000)
        if self.context_window_size == 0:
            for name in ["max_position_embeddings", "max_sequence_length"]:
                if name in self.kwargs:
                    self.context_window_size = self.kwargs.pop(name)
                    break
            else:
                raise ValueError(
                    "Unable to determine the maximum sequence length, as `context_window_size`, "
                    "`max_position_embeddings` or `max_sequence_length` is not provided."
                )
        if self.prefill_chunk_size == 0:
            self.prefill_chunk_size = min(self.context_window_size, 8192)

    @property
    def mamba_intermediate(self) -> int:
        """Computed property for mamba_intermediate."""
        return self.hidden_size * self.mamba_expand

def create_ssm_func(mamba_dim: int, d_state: int):
    """Creates a TensorIR SSM computation function."""
    # Symbolic sequence length
    seq_len = te.var("seq_len", dtype="int32")

    # Input tensors with explicit shapes
    ssm_dA = te.placeholder((mamba_dim, mamba_dim // 32, seq_len, d_state), dtype="float32", name="ssm_dA")
    ssm_dBu = te.placeholder((seq_len * (mamba_dim // 32), mamba_dim), dtype="float32", name="ssm_dBu")
    ssm_C = te.placeholder((1, seq_len, d_state), dtype="float32", name="ssm_C")
    ssm_state = te.placeholder((1, mamba_dim, d_state), dtype="float32", name="ssm_state")
    ssm_out = te.placeholder((1, seq_len, mamba_dim), dtype="float32", name="ssm_output_buffer")

    # Define state update
    def state_update(b, m, d):
        return T.cast(0, "float32")

    # Initialize state
    ssm_init_state = te.compute(
        (1, mamba_dim, d_state),
        state_update,
        name="ssm_init_state",
        tag="comm_reducer"  # Mark as reduction
    )

    # Define the reduction computation
    def scan_reduce(b, t, m, rd):
        block_idx = m // 256
        curr_state = ssm_init_state[b, m, rd]
        return T.cast(
            T.cast(ssm_dA[m, block_idx, t, rd], "float32") * curr_state +
            T.cast(ssm_dBu[t * 256 + block_idx, m], "float32"),
            "float32"
        ) * ssm_C[b, t, rd]

    # Create output with reduction
    k = te.reduce_axis((0, d_state), name="k")
    my_output = te.compute(
        (1, seq_len, mamba_dim),
        lambda b, t, m: te.sum(scan_reduce(b, t, m, k), axis=k),
        name="ssm_reduction",
        tag="comm_reducer"  # Mark as reduction
    )

    # Create and return the function
    prim_func = te.create_prim_func([ssm_dA, ssm_dBu, ssm_C, ssm_state, ssm_out])
    return prim_func.with_attr({
        "tir.is_scheduled": True,
        "tir.noalias": True,
        "op_pattern": 5,  # kCommReduce pattern
        "global_symbol": "ssm_compute"
    })

class JambaEmbedding(nn.Embedding):
    """The embedding module for Jamba, shared with the final lm_head."""

    def lm_head_forward(self, x: nn.Tensor):
        weight = nn.op.permute_dims(self.weight)
        return nn.op.matmul(x, weight)


class JambaMLP(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size // config.tensor_parallel_shards
        self.gate_up_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=2 * self.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(self.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor):
        concat_x1_x2 = self.gate_up_proj(x)
        x1, x2 = op.split(concat_x1_x2, 2, axis=-1)
        return self.down_proj(op.gelu(x1, approximate="tanh") * x2)


class JambaAttention(nn.Module):
    def __init__(self, config: JambaConfig):
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_q_heads = config.num_attention_heads // config.tensor_parallel_shards
        self.num_kv_heads = config.num_key_value_heads // config.tensor_parallel_shards
        self.qkv_proj = nn.Linear(
            in_features=config.hidden_size,
            out_features=(self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
        )
        self.o_proj = nn.Linear(
            in_features=self.num_q_heads * self.head_dim,
            out_features=config.hidden_size,
        )

    def forward(self, hidden_states: Tensor):
        d, h_q, h_kv = self.head_dim, self.num_q_heads, self.num_kv_heads
        b, s, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = op.reshape(qkv, (b, s, h_q + h_kv + h_kv, d))
        q, k, v = op.split(qkv, indices_or_sections=[h_q, h_q + h_kv], axis=2)
        output = op.reshape(
            nn.scaled_dot_product_attention(q, k, v, is_causal=True),
            (b, s, h_q * d)
        )
        return self.o_proj(output)


class JambaDecoderLayer(nn.Module):
    def __init__(self, config: JambaConfig, layer_idx: int):
        super().__init__()
        self.mamba = JambaMamba(config)

        if layer_idx == 1:
            self.feed_forward = nn.ModuleList([
                nn.ModuleList([
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),  # gate_proj
                    nn.Linear(config.hidden_size, config.intermediate_size, bias=False),  # up_proj
                    nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # down_proj
                ]) for _ in range(config.num_experts)
            ])
            self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        else:
            self.feed_forward = nn.ModuleList([
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),  # gate_proj
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),  # up_proj
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # down_proj
            ])

        self.input_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)
        self.pre_ff_layernorm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, hidden_states: Tensor):
        # Initial shape
        batch, L, state_dim = hidden_states.shape
        # assert hidden_states.shape == (batch, L, state_dim), f"Hidden states shape {hidden_states.shape} does not match expected shape (batch, L, state_dim)"
        # Mamba block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(hidden_states)
        hidden_states = residual + hidden_states

        # Feed forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)

        if hasattr(self, "router"):
            router_logits = self.router(hidden_states)
            expert_weights = op.softmax(router_logits, axis=-1)

            expert_output = op.zeros(hidden_states.shape, dtype="float32")
            for i, expert in enumerate(self.feed_forward):
                gate_proj, up_proj, down_proj = expert
                expert_splits = nn.split(expert_weights, indices_or_sections=expert_weights.shape[-1], axis=-1)
                expert_weight = expert_splits[i]

                expert_input = hidden_states * expert_weight
                gate_output = gate_proj(expert_input)
                up_output = up_proj(expert_input)
                expert_output += down_proj(gate_output * up_output)

            hidden_states = expert_output
        else:
            gate_proj, up_proj, down_proj = self.feed_forward
            gate_output = gate_proj(hidden_states)
            up_output = up_proj(hidden_states)
            hidden_states = down_proj(gate_output * up_output)

        return residual + hidden_states


class JambaMamba(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.mamba_dim = config.hidden_size * config.mamba_expand

        # Core parameters
        self.A_log = nn.Parameter([self.mamba_dim, config.mamba_d_state])
        self.D = nn.Parameter([self.mamba_dim])

        # Linear layers
        self.in_proj = nn.Linear(config.hidden_size, 2 * self.mamba_dim, bias=False)
        self.x_proj = nn.Linear(self.mamba_dim, config.mamba_dt_rank + 2 * config.mamba_d_state, bias=False)
        self.out_proj = nn.Linear(self.mamba_dim, config.hidden_size, bias=False)
        self.dt_proj = nn.Linear(config.mamba_dt_rank, self.mamba_dim, bias=True)

        # Conv1d
        self.conv1d_weight = nn.Parameter([self.mamba_dim, self.mamba_dim, config.mamba_d_conv])
        self.conv1d_bias = nn.Parameter([self.mamba_dim])

        # Layer norms
        self.dt_layernorm = nn.RMSNorm(config.mamba_dt_rank, -1, config.rms_norm_eps, bias=False)
        self.b_layernorm = nn.RMSNorm(config.mamba_d_state, -1, config.rms_norm_eps, bias=False)
        self.c_layernorm = nn.RMSNorm(config.mamba_d_state, -1, config.rms_norm_eps, bias=False)

        self.intermediate_size = self.mamba_dim  # Add this
        self.dt_rank = config.mamba_dt_rank  # Add this
        self.d_state = config.mamba_d_state  # Add this

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch, L, state_dim = hidden_states.shape

        print("hidden states size b4 in_proj", hidden_states.shape)

        projected = self.in_proj(hidden_states)  # [B, L, 2*mamba_dim]
        hidden, gate = op.split(projected, 2, axis=-1)  # Each [B, L, mamba_dim]
        # Conv1d processing - Calculate causal padding (only pad left side)
        hidden = op.permute_dims(hidden, axes=[0, 2, 1])  # [B, mamba_dim, L]
        kernel_size = self.conv1d_weight.shape[-1]  # Should be 4
        padding_size = kernel_size - 1

        # Add padding before conv1d with only left padding for causality
        hidden_padded = op.pad(
            hidden,
            pad=[0, 0,  # batch dimension
                 0, 0,  # channel dimension
                 padding_size, 0]  # sequence dimension: pad only left side
        )

        # Perform conv1d
        hidden_conv = op.conv1d(
            x=hidden_padded,
            weight=self.conv1d_weight,
            bias=self.conv1d_bias,
            stride=1,
            padding=0,  # Using explicit padding above
            dilation=1
        )
        hidden_conv = op.silu(hidden_conv)

        # Transpose back to [batch, seq_len, features]
        hidden = op.permute_dims(hidden_conv, axes=[0, 2, 1])

        print("hidden states size b4 x_proj", hidden.shape)
        ssm_params = self.x_proj(hidden)

        # Split params
        split_sizes = [self.dt_rank, self.d_state, self.d_state]
        split_indices = [sum(split_sizes[:i]) for i in range(1, len(split_sizes))]
        dt, B, C = op.split(ssm_params, split_indices, axis=-1)

        # Process dt with softplus
        dt_softplus_sum = nn.add(op.exp(dt), op.ones(dt.shape, dtype="float32"))
        dt_softplus_expr = base_op.log(dt_softplus_sum._expr)
        dt_softplus = Tensor.from_struct_info(
            dt_softplus_sum._expr.struct_info,
            name="dt_softplus"
        )
        dt_softplus._expr = dt_softplus_expr
        dt = dt_softplus
        dt = op.permute_dims(dt, (0, 2, 1))

        # Normalize B and C
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        # Prepare initial state
        A = -1 * op.exp(self.A_log)
        A_expanded = nn.unsqueeze(nn.unsqueeze(A, 1), 2)  # [8192, 1, 1, 16]

        dt_expanded = nn.unsqueeze(dt, -1)  # [1, 256, seq_len-3, 1]
        # Now multiply - shapes should be broadcastable
        dA = op.exp(op.multiply(A_expanded, dt_expanded), name='my_discrete_exp')

        # For dB calculation
        B_expanded = nn.unsqueeze(nn.unsqueeze(B, 0), 2)  # Add dimensions for broadcasting
        dB = op.multiply(dt_expanded, B_expanded)

        # Reshape dB if needed for matmul
        # dB shape should be [..., M, K] and hidden shape should be [..., K, N]
        dBu = op.matmul(
            op.reshape(dB, (-1, B.shape[-1])),  # Reshape to [batch*seq_len, hidden_dim]
            op.reshape(hidden, (-1, hidden.shape[-1]))  # Reshape to match dB
        )

        # Run SSM computation
        # print(f"dA shape: {dA.shape}")
        # print(f"dBu shape: {dBu.shape}")
        # print(f"C shape: {C.shape}")
        # When calling tensor_ir_op, specify both output tensors:
        outputs = op.zeros((batch, L, self.mamba_dim), dtype='float32')
        state = op.zeros((1, self.mamba_dim, self.d_state), dtype='float32')

        # Get SSM computation function and run it
        ssm_func = create_ssm_func(self.mamba_dim, self.d_state)
        print(ssm_func.script)
        # Create inplace output indices using tuples instead of Tensor.placeholder
        inplace_indices = 4
        ssm_outputs = op.tensor_ir_inplace_op(
            ssm_func,
            "my_compute_kernel",
            [dA, dBu, C, state, outputs],
            inplace_indices=inplace_indices,
            out=outputs
        )

        # self.current_state = updated_state
        # Prepare D and compute final output
        D = op.reshape(self.D, (1, 1, self.mamba_dim))  # [1, 1, mamba_dim]
        D = op.broadcast_to(D, (batch, L, self.mamba_dim))  # [batch, L, mamba_dim]

        # Project hidden states to match mamba dimension
        hidden_projected = op.reshape(hidden, (batch, L, self.mamba_dim))
        output = op.add(ssm_outputs, op.multiply(hidden_projected, D))

        output = op.multiply(output, nn.silu(gate))
        final_output = self.out_proj(output)

        return final_output


class JambaModel(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embed_tokens = JambaEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [JambaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, -1, config.rms_norm_eps, bias=False)

    def forward(self, input_embeds: Tensor):
        hidden_states = input_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return self.norm(hidden_states)


class JambaForCausalLM(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.model = JambaModel(config)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.dtype = "float32"

    def to(self, dtype: Optional[str] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype
        
    def get_logits(self, hidden_states: Tensor) -> Tensor:
        # Use the transposed embedding weights for the projection
        weight = nn.op.permute_dims(self.model.embed_tokens.weight)
        logits = nn.op.matmul(hidden_states, weight)
        return logits

    def embed(self, input_ids: Tensor) -> Tensor:
        return self.model.embed_tokens(input_ids)

    def prefill(self, input_embed: Tensor):
        hidden_states = self.model(input_embed)
        logits = self.get_logits(hidden_states)
        print("prefill complete")
        return logits, hidden_states

    def decode(self, input_embed: Tensor, past_hidden_states: Optional[Tensor]):
        op_ext.configure()
        # Append new input embedding to past hidden states
        hidden_states = op.concat([past_hidden_states, input_embed], dim=1)  # [1, seq_len + 1, hidden_size]

        # Pass through the model
        hidden_states = self.model(hidden_states)  # [1, seq_len + 1, hidden_size]

        # Extract the last token's hidden state
        def _index(x: te.Tensor):
            b, s, d = x.shape
            return te.compute((b, 1, d), lambda i, _, k: x[i, s - 1, k], name="index")

        last_hidden_state = op.tensor_expr_op(_index, name_hint="index", args=[hidden_states])  # [1, 1, hidden_size]

        # Compute logits for the last token
        logits = self.get_logits(last_hidden_state)  # [1, 1, vocab_size]
        print("decode complete")
        return logits, hidden_states

    def get_default_spec(self):
        """Get TVM module specifications."""
        mod_spec = {
            "embed": {
                "input_ids": nn.spec.Tensor(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": nn.spec.Tensor([1, 1, self.hidden_size], self.dtype),
                "past_hidden_states": nn.spec.Tensor([1, "seq_len", self.hidden_size], self.dtype),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
        }
        # Log symbolic variables
        symbolic_dims = set()
        for module in mod_spec.values():
            for key, value in module.items():
                if isinstance(value, nn.spec.Tensor):
                    for dim in value.shape:
                        if isinstance(dim, str):
                            symbolic_dims.add(dim)
        logger.info(f"Defined symbolic variables: {symbolic_dims}")
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)
def test_ir():
    # Create TensorIR function
    mamba_dim = 256
    d_state = 16
    ssm_func = create_ssm_func(mamba_dim, d_state)

    # Print the IR to inspect it
    print(ssm_func.script())
