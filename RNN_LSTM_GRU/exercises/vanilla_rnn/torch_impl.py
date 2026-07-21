"""Rebuild the reference PyTorch Vanilla RNN from memory."""

from __future__ import annotations

import math

import torch
from torch import nn


class ManualRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        bound = 1.0 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """CORE-V07: implement one recurrent step."""

        raise NotImplementedError("CORE-V07: see vanilla_rnn/prompts.md")


class ManualRecurrentLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = ManualRNNCell(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CORE-V08: unroll the cell over a batch-first sequence."""

        raise NotImplementedError("CORE-V08: see vanilla_rnn/prompts.md")

