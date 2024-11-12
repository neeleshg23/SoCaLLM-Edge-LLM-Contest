"""
This file defines the Jamba model structure for use within MLC, compatible with parameters loaded from HuggingFace or other external formats.
"""

import torch
import torch.nn as nn
from mlc_llm.model import RMSNorm  # Assuming RMSNorm is part of mlc_llm.model or defined similarly to Jamba's implementation

from .jamba import JambaLMConfig, AttentionLayer, MambaLayer, SparseMoEBlock

class JambaLM(nn.Module):
    def __init__(self, config: JambaLMConfig):
        """
        Initializes the JambaLM model based on the configuration.

        Parameters
        ----------
        config : JambaLMConfig
            Configuration object for the Jamba model specifying architecture details.
        """
        super().__init__()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.jamba = Jamba(config)
        self.final_layernorm = RMSNorm(config.d_model, config.rms_norm_eps)

        # Language model head tied to embedding if specified
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_lm_weights:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids):
        """
        Forward pass for the JambaLM model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tensor of input token IDs with shape (batch_size, sequence_length).

        Returns
        -------
        logits : torch.Tensor
            Logits for each token with shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(input_ids)
        x, _ = self.jamba(x)
        x = self.final_layernorm(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


class Jamba(nn.Module):
    def __init__(self, config: JambaLMConfig):
        """
        Core model structure that stacks Jamba layers with Mamba and Attention interleaved.

        Parameters
        ----------
        config : JambaLMConfig
            Configuration object for the Jamba model specifying architecture details.
        """
        super().__init__()

        self.config = config
        layers = []
        for i in range(config.n_layers):
            is_attn = (i - config.attn_layer_offset) % config.attn_layer_period == 0
            is_expert = (i - config.expert_layer_offset) % config.expert_layer_period == 0

            if is_attn:
                layers.append(AttentionLayer(config, num_experts=config.num_experts if is_expert else 1))
            else:
                layers.append(MambaLayer(config, num_experts=config.num_experts if is_expert else 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward pass for the Jamba core model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, sequence_length, d_model).

        Returns
        -------
        x : torch.Tensor
            Output tensor after processing through Jamba layers.
        router_logits : list
            List of router logits from MoE layers for load balancing.
        """
        router_logits = []
        for layer in self.layers:
            x, router_logit = layer(x)
            router_logits.append(router_logit)
        return x, router_logits
