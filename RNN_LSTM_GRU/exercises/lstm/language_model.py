"""Rebuild the reference LSTM language model from memory."""

from __future__ import annotations

import torch
from torch import nn

from common.generation import sample_from_logits
from .torch_impl import ManualRecurrentLayer


class CharLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.recurrent = ManualRecurrentLayer(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(self, token_ids: torch.Tensor, state=None):
        """CORE-L08: embed, recur, and project."""

        raise NotImplementedError("CORE-L08: see lstm/prompts.md")

    @torch.no_grad()
    def generate(self, prefix: torch.Tensor, new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None):
        """CORE-L09: generate while carrying ``(h, c)``."""

        # Use sample_from_logits inside your autoregressive loop.
        raise NotImplementedError("CORE-L09: see lstm/prompts.md")
