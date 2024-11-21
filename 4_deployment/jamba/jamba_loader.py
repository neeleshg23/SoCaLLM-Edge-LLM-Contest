import functools
import numpy as np
from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization
from .jamba_model import JambaConfig, JambaForCausalLM


def huggingface(model_config: JambaConfig, quantization: Quantization = None) -> ExternMapping:
    my_own_mapping_from_HF_to_TVM = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.final_layernorm.weight": "model.norm.weight",
        "model.layers.0.feed_forward.down_proj.weight": "model.layers.0.feed_forward.2.weight",
        "model.layers.0.feed_forward.gate_proj.weight": "model.layers.0.feed_forward.0.weight",
        "model.layers.0.feed_forward.up_proj.weight": "model.layers.0.feed_forward.1.weight",
        "model.layers.0.input_layernorm.weight": "model.layers.0.input_layernorm.weight",
        "model.layers.0.mamba.A_log": "model.layers.0.mamba.A_log",
        "model.layers.0.mamba.b_layernorm.weight": "model.layers.0.mamba.b_layernorm.weight",
        "model.layers.0.mamba.c_layernorm.weight": "model.layers.0.mamba.c_layernorm.weight",
        "model.layers.0.mamba.conv1d.bias": "model.layers.0.mamba.conv1d_bias",
        "model.layers.0.mamba.conv1d.weight": "model.layers.0.mamba.conv1d_weight",
        "model.layers.0.mamba.D": "model.layers.0.mamba.D",
        "model.layers.0.mamba.dt_layernorm.weight": "model.layers.0.mamba.dt_layernorm.weight",
        "model.layers.0.mamba.dt_proj.bias": "model.layers.0.mamba.dt_proj.bias",
        "model.layers.0.mamba.dt_proj.weight": "model.layers.0.mamba.dt_proj.weight",
        "model.layers.0.mamba.in_proj.weight": "model.layers.0.mamba.in_proj.weight",
        "model.layers.0.mamba.out_proj.weight": "model.layers.0.mamba.out_proj.weight",
        "model.layers.0.mamba.x_proj.weight": "model.layers.0.mamba.x_proj.weight",
        "model.layers.0.pre_ff_layernorm.weight": "model.layers.0.pre_ff_layernorm.weight",

        "model.layers.1.feed_forward.experts.0.down_proj.weight": "model.layers.1.feed_forward.0.2.weight",
        "model.layers.1.feed_forward.experts.0.gate_proj.weight": "model.layers.1.feed_forward.0.0.weight",
        "model.layers.1.feed_forward.experts.0.up_proj.weight": "model.layers.1.feed_forward.0.1.weight",
        "model.layers.1.feed_forward.experts.1.down_proj.weight": "model.layers.1.feed_forward.1.2.weight",
        "model.layers.1.feed_forward.experts.1.gate_proj.weight": "model.layers.1.feed_forward.1.0.weight",
        "model.layers.1.feed_forward.experts.1.up_proj.weight": "model.layers.1.feed_forward.1.1.weight",
        "model.layers.1.feed_forward.experts.2.down_proj.weight": "model.layers.1.feed_forward.2.2.weight",
        "model.layers.1.feed_forward.experts.2.gate_proj.weight": "model.layers.1.feed_forward.2.0.weight",
        "model.layers.1.feed_forward.experts.2.up_proj.weight": "model.layers.1.feed_forward.2.1.weight",
        "model.layers.1.feed_forward.experts.3.down_proj.weight": "model.layers.1.feed_forward.3.2.weight",
        "model.layers.1.feed_forward.experts.3.gate_proj.weight": "model.layers.1.feed_forward.3.0.weight",
        "model.layers.1.feed_forward.experts.3.up_proj.weight": "model.layers.1.feed_forward.3.1.weight",
        "model.layers.1.feed_forward.router.weight": "model.layers.1.router.weight",
        "model.layers.1.input_layernorm.weight": "model.layers.1.input_layernorm.weight",
        "model.layers.1.mamba.A_log": "model.layers.1.mamba.A_log",
        "model.layers.1.mamba.b_layernorm.weight": "model.layers.1.mamba.b_layernorm.weight",
        "model.layers.1.mamba.c_layernorm.weight": "model.layers.1.mamba.c_layernorm.weight",
        "model.layers.1.mamba.conv1d.bias": "model.layers.1.mamba.conv1d_bias",
        "model.layers.1.mamba.conv1d.weight": "model.layers.1.mamba.conv1d_weight",
        "model.layers.1.mamba.D": "model.layers.1.mamba.D",
        "model.layers.1.mamba.dt_layernorm.weight": "model.layers.1.mamba.dt_layernorm.weight",
        "model.layers.1.mamba.dt_proj.bias": "model.layers.1.mamba.dt_proj.bias",
        "model.layers.1.mamba.dt_proj.weight": "model.layers.1.mamba.dt_proj.weight",
        "model.layers.1.mamba.in_proj.weight": "model.layers.1.mamba.in_proj.weight",
        "model.layers.1.mamba.out_proj.weight": "model.layers.1.mamba.out_proj.weight",
        "model.layers.1.mamba.x_proj.weight": "model.layers.1.mamba.x_proj.weight",
        "model.layers.1.pre_ff_layernorm.weight": "model.layers.1.pre_ff_layernorm.weight",

        "model.layers.2.feed_forward.down_proj.weight": "model.layers.2.feed_forward.2.weight",
        "model.layers.2.feed_forward.gate_proj.weight": "model.layers.2.feed_forward.0.weight",
        "model.layers.2.feed_forward.up_proj.weight": "model.layers.2.feed_forward.1.weight",
        "model.layers.2.input_layernorm.weight": "model.layers.2.input_layernorm.weight",
        "model.layers.2.mamba.A_log": "model.layers.2.mamba.A_log",
        "model.layers.2.mamba.b_layernorm.weight": "model.layers.2.mamba.b_layernorm.weight",
        "model.layers.2.mamba.c_layernorm.weight": "model.layers.2.mamba.c_layernorm.weight",
        "model.layers.2.mamba.conv1d.bias": "model.layers.2.mamba.conv1d_bias",
        "model.layers.2.mamba.conv1d.weight": "model.layers.2.mamba.conv1d_weight",
        "model.layers.2.mamba.D": "model.layers.2.mamba.D",
        "model.layers.2.mamba.dt_layernorm.weight": "model.layers.2.mamba.dt_layernorm.weight",
        "model.layers.2.mamba.dt_proj.bias": "model.layers.2.mamba.dt_proj.bias",
        "model.layers.2.mamba.dt_proj.weight": "model.layers.2.mamba.dt_proj.weight",
        "model.layers.2.mamba.in_proj.weight": "model.layers.2.mamba.in_proj.weight",
        "model.layers.2.mamba.out_proj.weight": "model.layers.2.mamba.out_proj.weight",
        "model.layers.2.mamba.x_proj.weight": "model.layers.2.mamba.x_proj.weight",
        "model.layers.2.pre_ff_layernorm.weight": "model.layers.2.pre_ff_layernorm.weight",
    }

    # Initialize the mapping container
    mapping = ExternMapping()

    def reshape_conv1d_weights(x, expected_shape):
        current_shape = x.shape
        if len(current_shape) == 3 and current_shape[1] == 1:
            groups, _, kernel_size = current_shape
            _, c_in, _ = expected_shape
            if groups == c_in:  # Expand to match the expected shape
                return np.broadcast_to(x, expected_shape)
            else:
                raise ValueError(f"Cannot reshape weights with current shape {current_shape} to {expected_shape}")
        return x  # Return unchanged if reshape is unnecessary

    # Add mappings and convert parameters
    for hf_param, tvm_param in my_own_mapping_from_HF_to_TVM.items():
        if 'conv1d' in hf_param:
            expected_shape = (
            model_config.hidden_size * model_config.mamba_expand, model_config.hidden_size * model_config.mamba_expand,
            model_config.mamba_d_conv)
            mapping.add_mapping(
                tvm_param,
                [hf_param],
                functools.partial(
                    lambda x: reshape_conv1d_weights(x, expected_shape).astype(np.float32)
                ),
            )
            # print(f"Mapped and reshaped {hf_param} to {tvm_param}")
        else:
            mapping.add_mapping(
                tvm_param,
                [hf_param],
                functools.partial(lambda x: x.astype(np.float32)),  # Apply float32 casting
            )
            # print(f"Mapped {hf_param} to {tvm_param}")

    print("Completed parameter mapping.")
    return mapping
