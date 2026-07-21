"""Character language model powered by the reference manual GRU."""

from __future__ import annotations

import torch
from torch import nn

from common.generation import sample_from_logits
from .torch_impl import ManualRecurrentLayer


class CharLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        if min(vocabulary_size, embedding_size, hidden_size) <= 0:
            raise ValueError("all dimensions must be positive")
        self.vocabulary_size = vocabulary_size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.recurrent = ManualRecurrentLayer(embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed, recur, and project to vocabulary logits."""

        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have shape [batch, nonzero time]")
        embedded = self.embedding(token_ids)
        recurrent_output, final_state = self.recurrent(embedded, state)
        return self.output(recurrent_output), final_state

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend prefixes while carrying hidden state."""

        if prefix.ndim != 2 or prefix.shape[1] == 0:
            raise ValueError("prefix must have shape [batch, nonzero time]")
        if new_tokens < 0:
            raise ValueError("new_tokens must be non-negative")
        tokens = prefix.clone()
        if new_tokens == 0:
            return tokens
        logits, state = self(tokens)
        for index in range(new_tokens):
            next_token = sample_from_logits(logits[:, -1], temperature, top_k, generator)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            if index + 1 < new_tokens:
                logits, state = self(next_token[:, None], state)
        return tokens
