"""Sampling helpers that deliberately do not implement model recurrence."""

from __future__ import annotations


def sample_from_logits(logits, temperature: float = 1.0, top_k: int | None = None, generator=None):
    """Sample one token per batch row from logits shaped ``[batch, vocab]``."""

    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = logits / temperature
    if top_k is not None:
        if not 1 <= top_k <= scaled.shape[-1]:
            raise ValueError("top_k must be within the vocabulary size")
        threshold = torch.topk(scaled, top_k, dim=-1).values[:, -1:]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)

