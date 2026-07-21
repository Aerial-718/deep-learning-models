"""LSTM-specific analysis helpers."""

from __future__ import annotations

import torch


def state_gradient_norms(
    hidden_states: torch.Tensor,
    cell_states: torch.Tensor,
    scalar_loss: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean per-time gradient norms for hidden and cell states."""

    if hidden_states.ndim != 3 or cell_states.shape != hidden_states.shape:
        raise ValueError("hidden_states and cell_states must share shape [B, T, H]")
    dh, dc = torch.autograd.grad(
        scalar_loss,
        (hidden_states, cell_states),
        retain_graph=True,
        allow_unused=False,
    )
    return dh.norm(dim=-1).mean(dim=0), dc.norm(dim=-1).mean(dim=0)


def set_forget_bias(module: torch.nn.Module, value: float) -> None:
    """Set the input-side forget-bias slice on a manual LSTM cell."""

    hidden_size = int(module.hidden_size)  # type: ignore[attr-defined]
    with torch.no_grad():
        module.bias_ih[hidden_size : 2 * hidden_size].fill_(value)  # type: ignore[attr-defined]

