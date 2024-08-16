"""Hooked Transformer POS Embed Component.

This module contains all the component :class:`PosEmbed`.
"""
from typing import Dict, Optional, Union

import einops
import torch
from torch import Tensor
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

class PosEmbed(nn.Module):
    def __init__(self, cfg: HookedTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, vecs: Float[Tensor, "batch seq d_vocab"]) -> Float[Tensor, "batch seq d_model"]:
        # SOLUTION
        batch, seq_len, _ = vecs.shape
        return einops.repeat(self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch)
