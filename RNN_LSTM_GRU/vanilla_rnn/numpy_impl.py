"""Reference NumPy implementation of a single-layer Vanilla RNN."""

from __future__ import annotations

from typing import Any

import numpy as np

Parameters = dict[str, np.ndarray]
StepCache = dict[str, Any]


def init_parameters(
    input_size: int,
    hidden_size: int,
    rng: np.random.Generator,
) -> Parameters:
    """Initialize ``weight_ih``, ``weight_hh``, and two biases.

    Weight shapes follow PyTorch: ``[hidden, input]`` and ``[hidden, hidden]``.
    Return float64 arrays so finite-difference checks remain reliable.
    """

    if input_size <= 0 or hidden_size <= 0:
        raise ValueError("input_size and hidden_size must be positive")
    bound = 1.0 / np.sqrt(hidden_size)
    return {
        "weight_ih": rng.uniform(-bound, bound, (hidden_size, input_size)).astype(np.float64),
        "weight_hh": rng.uniform(-bound, bound, (hidden_size, hidden_size)).astype(np.float64),
        "bias_ih": np.zeros(hidden_size, dtype=np.float64),
        "bias_hh": np.zeros(hidden_size, dtype=np.float64),
    }


def _validate_parameters(params: Parameters, input_size: int, hidden_size: int) -> None:
    expected = {
        "weight_ih": (hidden_size, input_size),
        "weight_hh": (hidden_size, hidden_size),
        "bias_ih": (hidden_size,),
        "bias_hh": (hidden_size,),
    }
    if set(params) != set(expected):
        raise ValueError(f"parameter keys must be {sorted(expected)}")
    for name, shape in expected.items():
        if np.asarray(params[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def step_forward(
    x_t: np.ndarray,
    h_prev: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, StepCache]:
    """Compute one tanh RNN step and save a minimal backward cache."""

    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    if x_t.ndim != 2 or h_prev.ndim != 2:
        raise ValueError("x_t and h_prev must be two-dimensional")
    if x_t.shape[0] != h_prev.shape[0]:
        raise ValueError("x_t and h_prev must have the same batch size")
    hidden_size = h_prev.shape[1]
    _validate_parameters(params, x_t.shape[1], hidden_size)
    pre_activation = (
        x_t @ params["weight_ih"].T
        + params["bias_ih"]
        + h_prev @ params["weight_hh"].T
        + params["bias_hh"]
    )
    h_t = np.tanh(pre_activation)
    cache: StepCache = {"x_t": x_t, "h_prev": h_prev, "h_t": h_t, "params": params}
    return h_t, cache


def sequence_forward(
    x: np.ndarray,
    h0: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray, list[StepCache]]:
    """Unroll :func:`step_forward` over batch-first input.

    Returns:
        outputs: every hidden state, shape ``[B, T, H]``.
        h_last: final hidden state, shape ``[B, H]``.
        cache: one step cache per time index.
    """

    x = np.asarray(x)
    h0 = np.asarray(h0)
    if x.ndim != 3:
        raise ValueError("x must have shape [batch, time, input]")
    if x.shape[1] == 0:
        raise ValueError("x must contain at least one time step")
    if h0.ndim != 2 or h0.shape[0] != x.shape[0]:
        raise ValueError("h0 must have shape [batch, hidden]")
    _validate_parameters(params, x.shape[2], h0.shape[1])

    outputs = np.empty(
        (x.shape[0], x.shape[1], h0.shape[1]),
        dtype=np.result_type(x, h0, params["weight_ih"]),
    )
    caches: list[StepCache] = []
    h_t = h0
    for time in range(x.shape[1]):
        h_t, step_cache = step_forward(x[:, time], h_t, params)
        outputs[:, time] = h_t
        caches.append(step_cache)
    return outputs, h_t, caches


def sequence_backward(
    doutputs: np.ndarray,
    cache: list[StepCache],
    dh_last: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Parameters]:
    """Backpropagate through every cached time step.

    ``doutputs`` contains direct gradients for every emitted hidden state.
    ``dh_last`` is an optional additional gradient arriving at the final state.
    Return ``dx``, ``dh0``, and a gradient dict matching the parameter keys.
    """

    doutputs = np.asarray(doutputs)
    if not cache:
        raise ValueError("cache must contain at least one time step")
    first = cache[0]
    x_first = np.asarray(first["x_t"])
    h_first = np.asarray(first["h_prev"])
    params: Parameters = first["params"]
    batch_size, input_size = x_first.shape
    hidden_size = h_first.shape[1]
    time_steps = len(cache)
    if doutputs.shape != (batch_size, time_steps, hidden_size):
        raise ValueError("doutputs shape does not match the cached sequence")
    if dh_last is None:
        dh_next = np.zeros_like(h_first)
    else:
        dh_next = np.asarray(dh_last)
        if dh_next.shape != h_first.shape:
            raise ValueError("dh_last must have shape [batch, hidden]")

    dx = np.zeros(
        (batch_size, time_steps, input_size),
        dtype=np.result_type(doutputs, x_first, params["weight_ih"]),
    )
    gradients = {name: np.zeros_like(value) for name, value in params.items()}
    for time in range(time_steps - 1, -1, -1):
        step = cache[time]
        x_t = step["x_t"]
        h_prev = step["h_prev"]
        h_t = step["h_t"]
        dh_total = doutputs[:, time] + dh_next
        da = dh_total * (1.0 - h_t**2)
        gradients["weight_ih"] += da.T @ x_t
        gradients["weight_hh"] += da.T @ h_prev
        bias_gradient = da.sum(axis=0)
        gradients["bias_ih"] += bias_gradient
        gradients["bias_hh"] += bias_gradient
        dx[:, time] = da @ params["weight_ih"]
        dh_next = da @ params["weight_hh"]
    return dx, dh_next, gradients


def clip_gradients(
    gradients: Parameters,
    max_norm: float,
    epsilon: float = 1e-12,
) -> tuple[Parameters, float]:
    """Clip a gradient dictionary by its single global L2 norm.

    Return a new dictionary and the norm *before* clipping. Do not mutate input.
    """

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    squared_norm = sum(float(np.sum(np.asarray(value, dtype=np.float64) ** 2)) for value in gradients.values())
    norm = float(np.sqrt(squared_norm))
    scale = min(1.0, max_norm / (norm + epsilon))
    return {name: np.asarray(value).copy() * scale for name, value in gradients.items()}, norm
