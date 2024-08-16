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
        self.W_E = nn.Parameter(t.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, vecs: Float[Tensor, "batch position d_vocab"]) -> Float[Tensor, "batch position d_model"]:
        # SOLUTION
        return einops.einsum(vecs, self.W_E, "batch position d_vocab, d_vocab d_model -> batch position d_model")
