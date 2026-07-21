"""GRU-specific analysis helpers."""

from __future__ import annotations

import torch


def hidden_gradient_norms(hidden_states: torch.Tensor, scalar_loss: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, time, hidden]")
    gradients = torch.autograd.grad(
        scalar_loss,
        hidden_states,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return gradients.norm(dim=-1).mean(dim=0)


def gate_saturation_fraction(gates: torch.Tensor, threshold: float = 0.05) -> float:
    """Fraction of sigmoid gate values near zero or one."""

    if not 0.0 < threshold < 0.5:
        raise ValueError("threshold must be within (0, 0.5)")
    saturated = (gates < threshold) | (gates > 1.0 - threshold)
    return float(saturated.float().mean().item())

