from typing import Tuple
from tvm.relax.frontend import nn
from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import NoQuantize

from .jamba_model import JambaConfig, JambaForCausalLM

def no_quant(
    model_config: JambaConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Load a Jamba model without applying quantization."""
    model: nn.Module = JambaForCausalLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map