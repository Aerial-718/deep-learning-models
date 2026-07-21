"""Reference PyTorch GRU without official recurrent modules."""

from __future__ import annotations

import math

import torch
from torch import nn


class ManualGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Apply the reset-after ``r, z, n`` GRU equations."""

        if x_t.ndim != 2 or h_prev.ndim != 2:
            raise ValueError("x_t and h_prev must be two-dimensional")
        if x_t.shape[0] != h_prev.shape[0]:
            raise ValueError("x_t and h_prev must have the same batch size")
        if x_t.shape[1] != self.input_size or h_prev.shape[1] != self.hidden_size:
            raise ValueError("input or hidden feature size does not match the cell")
        input_projection = x_t @ self.weight_ih.T + self.bias_ih
        hidden_projection = h_prev @ self.weight_hh.T + self.bias_hh
        x_r, x_z, x_n = input_projection.chunk(3, dim=1)
        h_r, h_z, h_n = hidden_projection.chunk(3, dim=1)
        r = torch.sigmoid(x_r + h_r)
        z = torch.sigmoid(x_z + h_z)
        n = torch.tanh(x_n + r * h_n)
        return (1.0 - z) * n + z * h_prev


class ManualRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ManualGRUCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll the cell over a batch-first sequence."""

        if x.ndim != 3 or x.shape[2] != self.input_size:
            raise ValueError("x must have shape [batch, time, input_size]")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one time step")
        if initial_state is None:
            h_t = x.new_zeros((x.shape[0], self.hidden_size))
        else:
            if initial_state.shape != (x.shape[0], self.hidden_size):
                raise ValueError("initial_state must have shape [batch, hidden_size]")
            h_t = initial_state
        steps: list[torch.Tensor] = []
        for time in range(x.shape[1]):
            h_t = self.cell(x[:, time], h_t)
            steps.append(h_t)
        return torch.stack(steps, dim=1), h_t
