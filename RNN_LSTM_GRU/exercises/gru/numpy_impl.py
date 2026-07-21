"""Rebuild the reference reset-after NumPy GRU from memory."""

from __future__ import annotations

from typing import Any

import numpy as np

Parameters = dict[str, np.ndarray]
StepCache = dict[str, Any]


def sigmoid(x: np.ndarray) -> np.ndarray:
    positive = x >= 0
    output = np.empty_like(x, dtype=np.result_type(x, np.float32))
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x / (1.0 + exp_x)
    return output


def init_parameters(input_size: int, hidden_size: int, rng: np.random.Generator) -> Parameters:
    """CORE-G01: initialize combined ``r, z, n`` parameters."""

    raise NotImplementedError("CORE-G01: see gru/prompts.md")


def step_forward(x_t: np.ndarray, h_prev: np.ndarray, params: Parameters):
    """CORE-G02: compute one reset-after GRU step."""

    raise NotImplementedError("CORE-G02: see gru/prompts.md")


def sequence_forward(x: np.ndarray, h0: np.ndarray, params: Parameters):
    """CORE-G03: unroll over a batch-first sequence."""

    raise NotImplementedError("CORE-G03: see gru/prompts.md")


def sequence_backward(doutputs: np.ndarray, cache: list[StepCache],
                      dh_last: np.ndarray | None = None):
    """CORE-G05: perform explicit BPTT through all GRU branches."""

    raise NotImplementedError("CORE-G05: see gru/prompts.md")

