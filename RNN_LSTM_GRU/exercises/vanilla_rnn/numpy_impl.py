"""Rebuild the reference Vanilla RNN from memory."""

from __future__ import annotations

from typing import Any

import numpy as np

Parameters = dict[str, np.ndarray]
StepCache = dict[str, Any]


def init_parameters(input_size: int, hidden_size: int, rng: np.random.Generator) -> Parameters:
    """CORE-V01: initialize PyTorch-layout parameters as float64 arrays."""

    raise NotImplementedError("CORE-V01: see vanilla_rnn/prompts.md")


def step_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, StepCache]:
    """CORE-V02: compute one tanh recurrent step and its backward cache."""

    raise NotImplementedError("CORE-V02: see vanilla_rnn/prompts.md")


def sequence_forward(
    x: np.ndarray,
    h0: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray, list[StepCache]]:
    """CORE-V03: unroll over batch-first input and return all hidden states."""

    raise NotImplementedError("CORE-V03: see vanilla_rnn/prompts.md")


def sequence_backward(
    doutputs: np.ndarray,
    cache: list[StepCache],
    dh_last: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Parameters]:
    """CORE-V05: perform explicit BPTT."""

    raise NotImplementedError("CORE-V05: see vanilla_rnn/prompts.md")


def clip_gradients(
    gradients: Parameters,
    max_norm: float,
    epsilon: float = 1e-12,
) -> tuple[Parameters, float]:
    """CORE-V06: clip a gradient dictionary by one global L2 norm."""

    raise NotImplementedError("CORE-V06: see vanilla_rnn/prompts.md")

