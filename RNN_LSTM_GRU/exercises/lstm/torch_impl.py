"""Rebuild the reference PyTorch LSTM from memory."""

from __future__ import annotations

import math

import torch
from torch import nn


class ManualLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 4 * hidden_size
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))
        bound = 1.0 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        """CORE-L06: implement one ``i, f, g, o`` step."""

        raise NotImplementedError("CORE-L06: see lstm/prompts.md")


class ManualRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ManualLSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor,
                initial_state: tuple[torch.Tensor, torch.Tensor] | None = None):
        """CORE-L07: unroll the cell over a batch-first sequence."""

        raise NotImplementedError("CORE-L07: see lstm/prompts.md")

