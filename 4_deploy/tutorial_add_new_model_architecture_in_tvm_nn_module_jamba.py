"""
Implementation for GPT-2 architecture.
TODO: add docstring
"""
import tvm
import tvm.relay as relay

import dataclasses
from typing import Any, Dict, Optional, Union

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
from mambapy.mamba import MambaConfig, MambaBlock, RMSNorm


@dataclasses.dataclass
class JambaLMConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPT-2 model."""
    d_model: int
    n_layers: int
    mlp_size: int

    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba related
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True
    use_cuda: bool = False
    pscan: bool = True # use parallel scan mode or sequential mode when training

    # attention related
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    # MoE related
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    # language modeling
    vocab_size: int = 65536
    pad_token_id: int = 0
    tie_lm_weights: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = tir.ceil(self.d_model / 16)

        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank, d_state=self.d_state,
                                        expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale, rms_norm_eps=self.rms_norm_eps,
                                        bias=self.bias, conv_bias=self.conv_bias, inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)


# In[8]:


"""
Implementation for GPT-2 architecture.
TODO: add docstring
"""
import torch
import dataclasses
from typing import Any, Dict, Optional, Union

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
from mambapy.mamba import MambaConfig, MambaBlock, RMSNorm


@dataclasses.dataclass
class JambaLMConfig(ConfigBase):  # pylint: disable=too-many-instance-attributes
    """Configuration of the GPT-2 model."""
    d_model: int
    n_layers: int

    mlp_size: int

    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba related
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True
    use_cuda: bool = False
    pscan: bool = True # use parallel scan mode or sequential mode when training

    # attention related
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    # MoE related
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    # language modeling
    vocab_size: int = 65536
    pad_token_id: int = 0
    tie_lm_weights: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = tir.ceil(self.d_model / 16)

        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank, d_state=self.d_state,
                                        expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale, rms_norm_eps=self.rms_norm_eps,
                                        bias=self.bias, conv_bias=self.conv_bias, inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)


class JambaLM(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.jamba = Jamba(config)
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if self.config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)
        # router_logits : (B*L, n_experts) if n_experts>1

        x = self.embedding(tokens)

        x, router_logits = self.jamba(x)
        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        if self.config.num_experts == 1:
            return logits
        else:
            return logits, router_logits

    def step(self, tokens, caches):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x, caches = self.jamba.step(x, caches)
        x = self.final_layernorm(x)

        logits = self.lm_head(x)

        return logits, caches

    # TODO process prompt in parallel, and pass in sequential mode when prompt is finished ?
    def generate(self, tokenizer, prompt: str, max_tokens: int = 50, batch_size: int = 1, sample: bool = True, top_k: int = 40, temperature: float = 1.0):
        self.eval()

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(next(self.parameters()).device) # (1, num_tokens)
        input_ids = input_ids.repeat(batch_size, 1)

        # caches is a list of cache, one per layer
        # cache is composed of : - if Mamba layer : the hidden state, and the last d_conv-1 inputs (see more in mamba_lm.py)
        #                        - if Attention layer : the KV cache, ie 2 tensors of shape (B, num_kv_heads, L, head_dim)
        caches = [self.jamba.layers[i].get_empty_cache(batch_size, input_ids.device) for i in range(self.config.n_layers)]

        for i in range(input_ids.size(1) + max_tokens - 1):
            with relax.op.grad.no_grad():
                # forward the new output, get new cache
                next_token_logits, caches = self.step(input_ids[:, [i]], caches) # (batch_size, 1, vocab_size), caches
                next_token_logits = next_token_logits.squeeze(1)

            # sample (no sampling when the prompt is being processed)
            if i+1 >= input_ids.size(1):
                probs = relax.op.nn.softmax(next_token_logits / temperature, dim=-1) # (batch_size, vocab_size)

                if top_k is not None:
                    values, _ = relax.frontend.nn.topk(probs, k=top_k) # (batch_size, k) ordered from lowest to biggest
                    probs[probs < values[:, -1, None]] = 0
                    probs = probs / probs.sum(axis=1, keepdims=True)

                if sample:
                    uniform_sample = uniform_sample = tvm.contrib.random.uniform(low=0.0,
                                              high=1.0,
                                              size=(batch_size, 1))
                    next_token = relax.op.multinomial_from_uniform(probs, uniform_sample, sample_indices=None).squeeze(1)
                else:
                    next_token =relax.op.argmax(probs, axis=-1) # (batch_size)

                input_ids = relax.op.concat([input_ids, next_token.unsqueeze(1)], dim=1)

                if next_token.item() == tokenizer.eos_token_id:
                    break

        outputs = [tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in input_ids[:, 1:]]

        self.train()

        if batch_size==1:
            return outputs[0]
        else:
            return outputs

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class Jamba(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(config.n_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            num_experts = self.config.num_experts if is_expert else 1

            if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
            else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts))

        self.layers = nn.ModuleList(decoder_layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x):
        # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)

        router_logits = []

        for decoder_layer in self.layers:
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])

        return x, router_logits

    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches

class AttentionLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.self_attn = AttentionSDPA(config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # attention
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)
        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return (None, None)

class AttentionSDPA(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # attn_output: (B, L, D)

        B, L, _ = x.size()

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv cache implementation
        if cache is not None:
            past_keys, past_values = cache

            # not first in the sequence
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)

            cache = (keys, values) # prepare cache for next token

        # GQA related
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(queries, keys, values,dropout_p=self.attention_dropout if self.training else 0.0, is_causal=(cache is None))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache

class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        self.mamba = MambaBlock(config=config.mamba_config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x

        # FFN
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return (None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=device))

class SparseMoEBlock(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])

    def forward(self, x):
        # x: (B, L, D)

        # final_hidden_states: (B, L, D)
        # router_logits: (B*L, n_experts)

        #note : it is not clear why we work with shape (B*L, D) here.
        #I copied this code from the official jamba imple, and did not have time to think it through.

        batch_size, sequence_length, hidden_dim = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        x = x.view(-1, hidden_dim) # (B*L, D)

        router_logits = self.router(x) # (B*L, n_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        return final_hidden_states, router_logits

class MLP(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    # router_logits: list of router_logit, one per layer, each (B*D, n_experts)

    # moe_aux_loss : scalar

    router_logits = torch.cat([r for r in router_logits if r.shape[1] > 1], dim=0)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    moe_aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return moe_aux_loss * num_experts

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



# ### Define model architecture in jamba_model.py
# 
# With `tvm.relax.frontend.nn.Module`, we are able to define the model architecture in a modularized fashion. It looks pretty similar to the PyTorch style, except that the forward function does not actually perform the computation. It traces the operator graph using the placeholders that are passed as inputs.
# 
# Here we only present the GPT2Attention module. The entire model definition can be found [here](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/gpt2/gpt2_model.py).
# 
# You can optionally use `op._print(some_tensor)` to print out intermediate values of a tensor when running the compiled module. If you do this, you have to specify `debug=True` in `export_tvm()` and `jit()`. In addition to manual printing, we also provide an end-to-end debugging module `DebugChat` that will automatically dump intermediate values from all layers. Please refer to the
# [Debug Compiled MLC Model with DebugChat](#Debug-Compiled-MLC-Model-with-DebugChat) section below.

# In[9]:


import torch
import torch.nn as nn
from torch import Tensor

class JambaAttention(nn.Module):  # Adjusted for Jamba-specific configurations
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.num_heads = config.n_head // config.tensor_parallel_shards
        self.head_dim = config.head_dim
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx

        # Linear layers for QKV and output projections
        self.qkv_proj = nn.Linear(
            in_features=self.embed_dim,
            out_features=3 * self.num_heads * self.head_dim,
            bias=True,
        )
        self.output_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: Tensor, paged_kv_cache, layer_id: int):
        d, h = self.head_dim, self.num_heads
        b, s, _ = hidden_states.shape

        # Compute QKV projections
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(b, s, 3 * h, d)  # Reshape to separate heads and dimensions

        # Scaling attention scores if needed
        attn_score_scaling_factor = 1.0 / float(layer_id + 1) if self.scale_attn_by_inverse_layer_idx else 1.0

        # Perform fused attention with cache and Jamba-specific optimizations
        output = paged_kv_cache.attention_with_fused_qkv(
            layer_id, qkv, self.num_heads, attn_score_scaling_factor
        ).view(b, s, h * d)  # Reshape to merge heads

        # Project output back to embed dimension
        return self.output_proj(output)


# from jamba import jamba_model
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan



@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)

        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            self.selective_scan_cuda = selective_scan_fn

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        # x : (B, L, D)
        
        # y : (B, L, D)

        _, L, _ = x.shape

        xz = self.in_proj(x) # (B, L, 2*ED)
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        x = x.transpose(1, 2) # (B, ED, L)
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter
        x = x.transpose(1, 2) # (B, L, ED)

        x = F.silu(x)
        y = self.ssm(x, z)

        if self.config.use_cuda:
            output = self.out_proj(y) # (B, L, D)
            return output

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x, z):
        # x : (B, L, ED)

        # y : (B, L, ED)

        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        
        if self.config.use_cuda:
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        _, L, _ = x.shape

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        hs = []

        for t in range(0, L):
            h = deltaA[:, t] * h + BX[:, t]
            hs.append(h)
            
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output


from dataclasses import dataclass
import json
from typing import Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba import MambaConfig, MambaBlock, RMSNorm

@dataclass
class JambaConfig:

    n_embd: int
    tensor_parallel_shards: int
    head_dim: int
    vocab_size: int
    
    d_model: int
    n_layer: int
    n_layers: int
    n_head: int
    mlp_size: int
    
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5

    # mamba related
    d_state: int = 16 # N in paper
    expand_factor: int = 2 # N in paper
    d_conv: int = 4
    dt_rank: Union[int, str] = 'auto'

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4
    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = True
    use_cuda: bool = False # if True, use mamba_ssm by Albert Gu and Tri Dao. If False, fallsback to mamba.py
    pscan: bool = True # use parallel scan mode or sequential mode when training

    # attention related
    num_attention_heads: int = 32
    num_key_value_heads: int = 8 # GQA
    attention_dropout: float = 0.

    # MoE related
    num_experts: int = 16
    num_experts_per_tok: int = 2

    # structure
    attn_layer_offset: int = 4
    attn_layer_period: int = 8
    expert_layer_offset: int = 1
    expert_layer_period: int = 2

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        self.mamba_config = MambaConfig(d_model=self.d_model, n_layers=0, dt_rank=self.dt_rank, d_state=self.d_state,
                                        expand_factor=self.expand_factor, d_conv=self.d_conv, dt_min=self.dt_min, dt_max=self.dt_max,
                                        dt_init=self.dt_init, dt_scale=self.dt_scale, rms_norm_eps=self.rms_norm_eps,
                                        bias=self.bias, conv_bias=self.conv_bias, inner_layernorms=self.inner_layernorms,
                                        pscan=self.pscan, use_cuda=self.use_cuda)
        
    @classmethod
    def from_dict(cls, config_dict):
        """Creates an instance of JambaConfig from a dictionary."""
        return cls(**config_dict)

config_dict = {
    "n_embd": 768,  # or your specific embedding dimension
    "d_model": 768,
    "n_layer": 12,
    "n_layers": 12,
    "n_head": 12,
    "mlp_size": 3072,
    "head_dim": 64,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "num_attention_heads": 12,
    "num_key_value_heads": 8,
    "attention_dropout": 0.1,
    "vocab_size": 30522,
    "tensor_parallel_shards": 1,
}

# config_dict = {
#     "architectures": ["JambaLMHeadModel"],
#     "bos_token_id": 50256,
#     "eos_token_id": 50256,
#     "hidden_act": "gelu_new",
#     "n_ctx": 1024,
#     "n_embd": 768,
#     "n_head": 12,
#     "n_layer": 12,
#     "n_positions": 1024,
#     "layer_norm_epsilon": 1e-05,
#     "scale_attn_by_inverse_layer_idx": False,
#     "vocab_size": 50257,
# }

import torch
import torch.nn as nn
from typing import Optional
# from some_module import JambaConfig, JambaModel, PagedKVCache, RopeMode  # Replace with actual module imports as needed
from mlc_llm.nn import PagedKVCache, RopeMode

class Jamba(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()

        self.config = config

        # init each model layer, decide if it's mamba/attention and has experts or not
        decoder_layers = []
        for i in range(config.n_layers):
            is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
            is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

            num_experts = self.config.num_experts if is_expert else 1

            if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
            else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts))

        self.layers = nn.ModuleList(decoder_layers)

        # here you may want to init the weights in a particular manner if you don't use this jamba inside a JambaLM (see JambaLM)

    def forward(self, x, stop_at_layer: int = None):
        # x: (B, L, D)

        # logits: (B, L, D)

        router_logits = []

        for i, decoder_layer in enumerate(self.layers):
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])

            if stop_at_layer == i+1:
                return x
            
        if self.config.num_experts == 1:
            return x
        else:
            return x, router_logits
    
    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Define layers here

class MambaLayer(nn.Module):
    def __init__(self, config, num_experts):
        super().__init__()
        # Define layers here

class AttentionLayer(nn.Module):
    def __init__(self, config, num_experts):
        super().__init__()
        # Define layers here

import torch
import torch.nn as nn
import torch.nn.functional as F

class JambaLMHeadModel(nn.Module):
    def __init__(self, config: JambaConfig):
        super().__init__()
        self.transformer = Jamba(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.n_layer = config.n_layer
        self.n_embed = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.tensor_parallel_shards = config.tensor_parallel_shards
        self.dtype = torch.float32

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.attention = nn.MultiheadAttention(config.n_embd, config.num_attention_heads)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)


    def forward(self, input_ids, input_embed=None):
        # Print the shape of input_ids for debugging
        print(f"input_ids shape: {input_ids.shape}")

        # Ensure input_embed is in the correct shape
        if input_embed is None:
            input_embed = self.embedding(input_ids)  # Shape: [batch_size, seq_len, embed_dim]
        
        # Print shape after embedding layer
        print(f"input_embed shape after embedding: {input_embed.shape}")

        # Reshape to [seq_len, batch_size, embed_dim] for MultiheadAttention
        input_embed = input_embed.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        # Print shape before attention
        print(f"input_embed shape before attention: {input_embed.shape}")
        
        # Pass through multi-head attention
        attn_output, _ = self.attention(input_embed, input_embed, input_embed)
        
        # Transpose back to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(0, 1)
        
        # Print shape after attention
        print(f"attn_output shape after attention: {attn_output.shape}")
        
        # Project to vocabulary size
        output = self.linear(attn_output)  # Shape: [batch_size, seq_len, vocab_size]
        
        # Print final output shape
        print(f"output shape: {output.shape}")
        
        return output



 def export_tvm(self, spec, input_shape=(1, 128), dtype="float32"):       
    #def export_tvm(self, spec, input_shape=(1, 3, 224, 224), dtype="float32"):
        # Step 1: Trace the model to get a TorchScript model
        # Assuming input_shape is something like (batch_size, seq_len)
        # traced_model = torch.jit.trace(self, torch.randint(0, self.config.vocab_size, input_shape, dtype=torch.long))
        # Assuming input_shape is (batch_size, seq_len)
        # Ensure input_shape is in the form of (batch_size, seq_len)
        traced_model = torch.jit.trace(self, torch.randint(0, self.config.vocab_size, input_shape, dtype=torch.long))




        
        # Step 2: Convert to a TVM Relay module
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
        
        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        # Step 3: Compile with TVM (you can specify a target device here, like 'llvm' for CPU)
        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        
        # Step 4: Return the compiled library and parameter names
        return lib, params

    def to(self, dtype: Optional[torch.dtype] = None):
        super().to(dtype=dtype)
        if dtype is not None:
            self.dtype = dtype

    def batch_forward(self, input_embeds: torch.Tensor, paged_kv_cache: PagedKVCache, logit_positions: Optional[torch.Tensor] = None):
        hidden_states = self.transformer(input_embeds, paged_kv_cache)
        if logit_positions is not None:
            hidden_states = hidden_states.index_select(1, logit_positions)
        logits = self.lm_head(hidden_states)
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)
        return logits

    def embed(self, input_ids: torch.Tensor):
        if self.tensor_parallel_shards > 1:
            input_ids = self._ccl_broadcast_from_worker0(input_ids)
        return self.transformer.wte(input_ids)

    def prefill(self, input_embed: torch.Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = self.transformer(input_embed, paged_kv_cache)
        hidden_states = hidden_states[:, -1:, :]
        logits = self.lm_head(hidden_states)
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)
        return logits, paged_kv_cache

    def decode(self, input_embed: torch.Tensor, paged_kv_cache: PagedKVCache):
        hidden_states = self.transformer(input_embed, paged_kv_cache)
        logits = self.lm_head(hidden_states)
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)
        return logits, paged_kv_cache

    def batch_prefill(self, input_embeds: torch.Tensor, logit_positions: torch.Tensor, paged_kv_cache: PagedKVCache):
        if self.tensor_parallel_shards > 1:
            logit_positions = self._ccl_broadcast_from_worker0(logit_positions)
        logits = self.batch_forward(input_embeds, paged_kv_cache, logit_positions)
        return logits, paged_kv_cache

    def batch_decode(self, input_embeds: torch.Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def batch_verify(self, input_embeds: torch.Tensor, paged_kv_cache: PagedKVCache):
        logits = self.batch_forward(input_embeds, paged_kv_cache)
        return logits, paged_kv_cache

    def create_paged_kv_cache(self, max_batch_size: int, max_total_seq_len: int, prefill_chunk_size: int, page_size: int, support_sliding_window: int) -> PagedKVCache:
        return PagedKVCache.create_generic(
            max_batch_size=max_batch_size,
            max_total_seq_len=max_total_seq_len,
            prefill_chunk_size=prefill_chunk_size,
            page_size=page_size,
            support_sliding_window=support_sliding_window,
            num_hidden_layers=self.n_layer,
            num_attention_heads=self.n_head // self.tensor_parallel_shards,
            num_key_value_heads=self.n_head // self.tensor_parallel_shards,
            head_dim=self.head_dim,
            rope_mode=RopeMode.NONE,
            rope_scale=-1,
            rope_theta=-1,
            dtype=self.dtype,
        )

    def get_default_spec(self):
        mod_spec = {
            "embed": {
                "input_ids": tvm.relay.TensorType(["seq_len"], "int32"),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "prefill": {
                "input_embed": tvm.relay.TensorType([1, "seq_len", self.n_embed], "float32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "decode": {
                "input_embed": tvm.relay.TensorType([1, 1, self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_prefill": {
                "input_embeds": tvm.relay.TensorType([1, "seq_len", self.n_embed], self.dtype),
                "logit_positions": tvm.relay.TensorType(["batch_size"], "int32"),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_decode": {
                "input_embeds": tvm.relay.TensorType(["batch_size", 1, self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "batch_verify": {
                "input_embeds": tvm.relay.TensorType([1, "seq_len", self.n_embed], self.dtype),
                "paged_kv_cache": nn.spec.Object(object_type=PagedKVCache),
                "$": {
                    "param_mode": "packed",
                    "effect_mode": "none",
                },
            },
            "create_paged_kv_cache": {
                "max_batch_size": int,
                "max_total_seq_len": int,
                "prefill_chunk_size": int,
                "page_size": int,
                "support_sliding_window": int,
                "$": {
                    "param_mode": "none",
                    "effect_mode": "none",
                },
            },
        }
        return nn.spec.ModuleSpec.from_raw(mod_spec, self)


from tvm import relay

config = JambaConfig.from_dict(config_dict)
model = JambaLMHeadModel(config)

# Define spec with the corrected attribute name 'n_embd'
spec = {
    "input_ids": relay.TensorType(["seq_len"], "int32"),
    "input_embed": relay.TensorType([1, "seq_len", config.n_embd], "float32"),  # Updated to 'n_embd'
    # Add any additional required inputs here
}

# Export model to tvm with the specified configuration
mod, named_params = model.export_tvm(spec=spec)



# Uncomment the following line to show the model in Tensor IR
# mod.show(black_format=False)

for name, param in named_params:
    print(name, param.shape, param.dtype)


# ### Define a Loader in gpt2_loader.py
# 
# In `gpt2_loader.py`, we define how we convert the parameters from Huggingface to the format used by MLC model.
# 
# The loader class will return an [`ExternMapping`](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/loader/mapping.py) that contains two kinds of mappings:
# - Source -> MLC parameter mapping: for example, parameter renaming, parameter transformation, etc.
# - Unused mapping: parameters in the source that are not used in the MLC model definition.
# 
# In GPT2, we need to transpose c_attn, c_proj and c_fc weights since GPT-2 uses Conv1D. To do so, we will supply a mapping function as follows
# 
# ```
# for conv1d_weight_name in ["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_fc"]:
#     src_name = f"h.{i}.{conv1d_weight_name}.weight"
#     mlc_name = f"transformer.{src_name}"
#     mapping.add_mapping(
#         mlc_name,
#         [src_name],
#         functools.partial(
#             lambda x, dtype: x.transpose().astype(dtype),
#             dtype=named_parameters[mlc_name].dtype,
#         ),
#     )
# ```
# 
# Some renamings are also needed for GPT-2 parameters conversion to work. Please refer to [gpt2_loader.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/gpt2/gpt2_loader.py).

# ## Add the Model to the Supported Pre-built Model Workflow
# 
# Once the entire model is defined in TVM nn.module, including the model architecture, model loader and model quantitizer, we can then add it to the supported pre-built model workflow.
# 
# In [`mlc-llm/python/mlc_llm/model/model.py`](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/model.py), add the GPT-2 model to the `MODELS` list:
# 
# ```
# "gpt2": Model(
#     name="gpt2",
#     model=gpt2_model.GPT2LMHeadModel,
#     config=gpt2_model.GPT2Config,
#     source={
#         "huggingface-torch": gpt2_loader.huggingface,
#         "huggingface-safetensor": gpt2_loader.huggingface,
#     },
#     quantize={
#         "no-quant": gpt2_quantization.no_quant,
#         "group-quant": gpt2_quantization.group_quant,
#     },
# )
# ```

# ## Compile GPT-2 model libraries and weights
# 
# The following steps will be the same as the general model compilation workflow [here](https://llm.mlc.ai/docs/compilation/compile_models.html).

# In[ ]:


# conda install -c conda-forge git-lfs



# In[22]:


# Create directory
get_ipython().system('mkdir -p dist/models')
get_ipython().run_line_magic('cd', 'dist/models')

# Clone HF weights
get_ipython().system('git lfs install')
get_ipython().system('git clone https://huggingface.co/gpt2')
get_ipython().run_line_magic('cd', '../..')


# In[23]:


# Convert weight
get_ipython().system('mlc_llm convert_weight ./dist/models/gpt2/ --device cuda --quantization q0f16 -o dist/gpt2-q0f16-MLC')


# In[24]:


# 1. gen_config: generate mlc-chat-config.json and process tokenizers
get_ipython().system('mlc_llm gen_config ./dist/models/gpt2      --quantization q0f16 --conv-template gpt2      -o dist/gpt2-q0f16-MLC/')

# 2. compile: compile model library with specification in mlc-chat-config.json
get_ipython().system('mlc_llm compile ./dist/gpt2-q0f16-MLC/mlc-chat-config.json      --device cuda -o dist/gpt2-q0f16-MLC/gpt2-q0f16-cuda.so')


# ## Debug Compiled MLC Model with DebugChat

# After successfully compiling the model library and converting the model weights, it is important to check whether the model generates the correct output. One way to check this is to compare the output logits of the model with its Huggingface PyTorch counterpart under the same input tokens.
# 
# To help with debugging the MLC model, we provide a `mlc_llm.testing.DebugChat` module that
# 
# - Loads the MLC model we just compiled
# - Runs the entire `forward` flow of the model using a user-specified prompt
# - Dumps the intermediate values from all layers.
# 
# You can then compare the intermediate values with those from the Huggingface PyTorch model. (For PyTorch, you can extract intermediate values using [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook))

# In[26]:


get_ipython().system('python -m mlc_llm.testing.debug_chat --model ./dist/gpt2-q0f16-MLC/ --model-lib dist/gpt2-q0f16-MLC/gpt2-q0f16-cuda.so --device cuda --debug-dir ./debug-gpt2 --generate-len 5 "Hey how are you doing today?"')




import numpy as np

data = np.load('./debug-gpt2/decode_2/f0_take3.npz')
print(data)
print(data["arg_0"])
print(data["arg_1"])
print(data["arg_2"]) # This is the output of the take function

