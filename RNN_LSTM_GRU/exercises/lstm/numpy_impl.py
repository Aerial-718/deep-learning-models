"""Rebuild the reference NumPy LSTM from memory."""

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
    """CORE-L01: initialize combined ``i, f, g, o`` parameters."""

    raise NotImplementedError("CORE-L01: see lstm/prompts.md")


def step_forward(x_t: np.ndarray, state: tuple[np.ndarray, np.ndarray], params: Parameters):
    """CORE-L02: compute one LSTM step."""

    raise NotImplementedError("CORE-L02: see lstm/prompts.md")


def sequence_forward(x: np.ndarray, initial_state: tuple[np.ndarray, np.ndarray], params: Parameters):
    """CORE-L03: unroll over a batch-first sequence."""

    raise NotImplementedError("CORE-L03: see lstm/prompts.md")


def sequence_backward(doutputs: np.ndarray, cache: list[StepCache],
                      dh_last: np.ndarray | None = None, dc_last: np.ndarray | None = None):
    """CORE-L05: perform explicit BPTT through hidden and cell paths."""

    raise NotImplementedError("CORE-L05: see lstm/prompts.md")

