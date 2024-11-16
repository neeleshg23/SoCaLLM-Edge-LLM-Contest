"""
This file specifies how MLC's Jamba parameter maps from HuggingFace PyTorch format.
Parameter mapping driven by model configuration.
"""
import functools
import numpy as np
from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization
from .jamba_model import JambaConfig, JambaForCausalLM

def huggingface(model_config: JambaConfig, quantization: Quantization) -> ExternMapping:
    """Maps HuggingFace Jamba parameters to MLC-LLM format.
    
    Parameters
    ----------
    model_config : JambaConfig
        The configuration of the Jamba model with specific architecture settings:
        - hidden_size: 4096
        - intermediate_size: 14336
        - num_hidden_layers: 3
        - num_attention_heads: 16
        - num_key_value_heads: 8
        - mamba_expand: 2 (gives 8192 intermediate size)
        - expert periods: 2/1, attention periods: 8/4
        
    quantization : Quantization
        The quantization configuration
        
    Returns
    -------
    param_map : ExternMapping
        Parameter mapping from MLC to HuggingFace PyTorch
    """
    model = JambaForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)

    _, named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(named_params)
    mapping = ExternMapping()

    # Configuration-driven sizes
    hidden_size = model_config.hidden_size  # 4096
    intermediate_size = model_config.intermediate_size  # 14336
    mamba_intermediate = hidden_size * model_config.mamba_expand  # 8192
    num_experts = model_config.num_experts  # 4
    d_state = model_config.mamba_d_state  # 16 
    d_conv = model_config.mamba_d_conv  # 4
    dt_rank = model_config.mamba_dt_rank  # 256
    vocab_size = model_config.vocab_size  # 128256

    # Embeddings and final norm [vocab_size, hidden_size] and [hidden_size]
    base_params = [
        "model.embed_tokens.weight",
        "model.final_layernorm.weight"
    ]
    for param in base_params:
        mapping.add_mapping(
            param,
            [param],
            functools.partial(
                lambda x, dtype: x.astype(dtype),
                dtype=named_parameters[param].dtype
            )
        )

    # Process each layer with correct scheduling
    for i in range(model_config.num_hidden_layers):
        base = f"model.layers.{i}"
        
        # Layer norms [hidden_size]
        norm_params = [
            f"{base}.input_layernorm.weight",
            f"{base}.pre_ff_layernorm.weight"
        ]
        for param in norm_params:
            mapping.add_mapping(
                param,
                [param],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=named_parameters[param].dtype
                )
            )

        # Mamba parameters using config sizes
        mamba_base = f"{base}.mamba"
        mamba_params = {
            # Core SSM parameters
            "A_log": [mamba_intermediate, d_state],
            "D": [mamba_intermediate],
            
            # Layer norms
            "b_layernorm.weight": [d_state],
            "c_layernorm.weight": [d_state],
            "dt_layernorm.weight": [dt_rank],
            
            # Convolution
            "conv1d.bias": [mamba_intermediate],
            "conv1d.weight": [mamba_intermediate, 1, d_conv],
            
            # Projections
            "dt_proj.bias": [mamba_intermediate],
            "dt_proj.weight": [mamba_intermediate, dt_rank],
            "in_proj.weight": [2 * mamba_intermediate, hidden_size],
            "out_proj.weight": [hidden_size, mamba_intermediate],
            "x_proj.weight": [dt_rank + 2 * d_state, mamba_intermediate]
        }
        
        for name, shape in mamba_params.items():
            param = f"{mamba_base}.{name}"
            mapping.add_mapping(
                param,
                [param],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=named_parameters[param].dtype
                )
            )

        # Feed forward network - check if expert layer
        ff_base = f"{base}.feed_forward"
        is_expert_layer = (i % model_config.expert_layer_period == model_config.expert_layer_offset)
        
        if is_expert_layer:
            # MoE layer
            for expert_idx in range(num_experts):
                expert_base = f"{ff_base}.experts.{expert_idx}"
                expert_params = {
                    "down_proj.weight": [hidden_size, intermediate_size],
                    "gate_proj.weight": [intermediate_size, hidden_size],
                    "up_proj.weight": [intermediate_size, hidden_size]
                }
                for name, shape in expert_params.items():
                    param = f"{expert_base}.{name}"
                    mapping.add_mapping(
                        param,
                        [param],
                        functools.partial(
                            lambda x, dtype: x.astype(dtype),
                            dtype=named_parameters[param].dtype
                        )
                    )
                    
            # Router weights [num_experts, hidden_size]
            router_param = f"{ff_base}.router.weight"
            mapping.add_mapping(
                router_param,
                [router_param],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=named_parameters[router_param].dtype
                )
            )
        else:
            # Regular MLP layer
            mlp_params = {
                "down_proj.weight": [hidden_size, intermediate_size],
                "gate_proj.weight": [intermediate_size, hidden_size],
                "up_proj.weight": [intermediate_size, hidden_size]
            }
            for name, shape in mlp_params.items():
                param = f"{ff_base}.{name}"
                mapping.add_mapping(
                    param,
                    [param],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=named_parameters[param].dtype
                    )
                )

    # Handle tied weights if specified
    if model_config.tie_word_embeddings:
        mapping.add_mapping(
            "lm_head.weight",
            ["model.embed_tokens.weight"],
            functools.partial(
                lambda x, dtype: x.T.astype(dtype),
                dtype=named_parameters["lm_head.weight"].dtype if "lm_head.weight" in named_parameters else "bfloat16"
            )
        )

    return mapping