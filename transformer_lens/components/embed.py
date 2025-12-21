"""Hooked Transformer Embed Component.

This module contains all the component :class:`Embed`.
"""
from typing import Dict, Union

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

import einops
from transformer_lens.components import LayerNorm
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        """Embed tokens using the embedding matrix W_E.
        
        Args:
            tokens: Token IDs of shape [batch, position]
            
        Returns:
            Embeddings of shape [batch, position, d_model]
        """
        return self.W_E[tokens]
