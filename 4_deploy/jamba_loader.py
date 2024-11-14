"""
This file specifies how MLC's Jamba parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from mlc_llm.loader import ExternMapping
from mlc_llm.quantization import Quantization

from .jamba_model import JambaLMConfig, JambaLM


def huggingface(model_config: JambaLMConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : JambaLMConfig
        The configuration of the Jamba model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = JambaLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    mapping.add_mapping(
        "lm_head.weight",
        ["embedding.weight"],
        functools.partial(
            lambda x, dtype: x.astype(dtype),
            dtype=named_parameters["embedding.weight"].dtype,
        ),
    )

    for i in range(model_config.n_layers):
        mapping.add_unused(f"jamba.layers.{i}.attn.bias")

        # Transpose attention and MoE weights as needed for Jamba
        for weight_name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", 
                            "moe.gate_proj", "moe.up_proj", "moe.down_proj"]:
            src_name = f"jamba.layers.{i}.{weight_name}.weight"
            mlc_name = f"{src_name}"
            mapping.add_mapping(
                mlc_name,
                [src_name],
                functools.partial(
                    lambda x, dtype: x.transpose().astype(dtype),
                    dtype=named_parameters[mlc_name].dtype,
                ),
            )

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            # jamba.layers.0.self_attn.q_proj.weight --> layers.0.self_attn.q_proj.weight
            source_name = mlc_name.split(".", 1)[1]
            mapping.add_mapping(
                mlc_name,
                [source_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=mlc_param.dtype,
                ),
            )

    return mapping
