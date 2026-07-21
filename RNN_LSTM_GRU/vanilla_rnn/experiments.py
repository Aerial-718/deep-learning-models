"""Analysis helpers for the Vanilla RNN notebook."""

from __future__ import annotations

import torch


def hidden_gradient_norms(hidden_states: torch.Tensor, scalar_loss: torch.Tensor) -> torch.Tensor:
    """Return ``||d loss / d h_t||_2`` for each time index, averaged over batch."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, time, hidden]")
    gradients = torch.autograd.grad(
        scalar_loss,
        hidden_states,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradients.norm(dim=-1).mean(dim=0)

