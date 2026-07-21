"""Reference PyTorch LSTM without official recurrent modules."""

from __future__ import annotations

import math

import torch
from torch import nn


class ManualLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 4 * hidden_size
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one ``i, f, g, o`` LSTM step."""

        h_prev, c_prev = state
        if x_t.ndim != 2 or h_prev.ndim != 2 or c_prev.ndim != 2:
            raise ValueError("x_t, h_prev, and c_prev must be two-dimensional")
        if h_prev.shape != c_prev.shape or x_t.shape[0] != h_prev.shape[0]:
            raise ValueError("state shapes or batch size do not match")
        if x_t.shape[1] != self.input_size or h_prev.shape[1] != self.hidden_size:
            raise ValueError("input or hidden feature size does not match the cell")
        gates = (
            x_t @ self.weight_ih.T
            + self.bias_ih
            + h_prev @ self.weight_hh.T
            + self.bias_hh
        )
        i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=1)
        i = torch.sigmoid(i_pre)
        f = torch.sigmoid(f_pre)
        g = torch.tanh(g_pre)
        o = torch.sigmoid(o_pre)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class ManualRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Unroll the cell over a batch-first sequence."""

        if x.ndim != 3 or x.shape[2] != self.input_size:
            raise ValueError("x must have shape [batch, time, input_size]")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one time step")
        expected_shape = (x.shape[0], self.hidden_size)
        if initial_state is None:
            h_t = x.new_zeros(expected_shape)
            c_t = x.new_zeros(expected_shape)
        else:
            h_t, c_t = initial_state
            if h_t.shape != expected_shape or c_t.shape != expected_shape:
                raise ValueError("initial_state tensors must have shape [batch, hidden_size]")
        steps: list[torch.Tensor] = []
        for time in range(x.shape[1]):
            h_t, c_t = self.cell(x[:, time], (h_t, c_t))
            steps.append(h_t)
        return torch.stack(steps, dim=1), (h_t, c_t)
