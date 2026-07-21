"""Rebuild the reference PyTorch reset-after GRU from memory."""

from __future__ import annotations

import math

import torch
from torch import nn


class ManualGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        gate_size = 3 * hidden_size
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(gate_size))
        self.bias_hh = nn.Parameter(torch.empty(gate_size))
        bound = 1.0 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """CORE-G06: implement one reset-after step."""

        raise NotImplementedError("CORE-G06: see gru/prompts.md")


class ManualRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ManualGRUCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None):
        """CORE-G07: unroll the cell over a batch-first sequence."""

        raise NotImplementedError("CORE-G07: see gru/prompts.md")

