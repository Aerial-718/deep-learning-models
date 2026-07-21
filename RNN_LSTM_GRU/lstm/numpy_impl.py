"""Reference NumPy LSTM implementation with explicit BPTT."""

from __future__ import annotations

from typing import Any

import numpy as np

Parameters = dict[str, np.ndarray]
StepCache = dict[str, Any]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid supplied as non-core infrastructure."""

    positive = x >= 0
    output = np.empty_like(x, dtype=np.result_type(x, np.float32))
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    output[~positive] = exp_x / (1.0 + exp_x)
    return output


def init_parameters(
    input_size: int,
    hidden_size: int,
    rng: np.random.Generator,
) -> Parameters:
    """Create four-gate parameters ordered ``i, f, g, o``."""

    if input_size <= 0 or hidden_size <= 0:
        raise ValueError("input_size and hidden_size must be positive")
    bound = 1.0 / np.sqrt(hidden_size)
    gate_size = 4 * hidden_size
    return {
        "weight_ih": rng.uniform(-bound, bound, (gate_size, input_size)).astype(np.float64),
        "weight_hh": rng.uniform(-bound, bound, (gate_size, hidden_size)).astype(np.float64),
        "bias_ih": np.zeros(gate_size, dtype=np.float64),
        "bias_hh": np.zeros(gate_size, dtype=np.float64),
    }


def _validate_parameters(params: Parameters, input_size: int, hidden_size: int) -> None:
    gate_size = 4 * hidden_size
    expected = {
        "weight_ih": (gate_size, input_size),
        "weight_hh": (gate_size, hidden_size),
        "bias_ih": (gate_size,),
        "bias_hh": (gate_size,),
    }
    if set(params) != set(expected):
        raise ValueError(f"parameter keys must be {sorted(expected)}")
    for name, shape in expected.items():
        if np.asarray(params[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")


def step_forward(
    x_t: np.ndarray,
    state: tuple[np.ndarray, np.ndarray],
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray, StepCache]:
    """Compute one LSTM step.

    Args:
        x_t: ``[B, D]``.
        state: ``(h_prev, c_prev)``, each ``[B, H]``.
    Returns:
        ``h_t, c_t, cache``.
    """

    x_t = np.asarray(x_t)
    h_prev, c_prev = (np.asarray(value) for value in state)
    if x_t.ndim != 2 or h_prev.ndim != 2 or c_prev.ndim != 2:
        raise ValueError("x_t, h_prev, and c_prev must be two-dimensional")
    if h_prev.shape != c_prev.shape or x_t.shape[0] != h_prev.shape[0]:
        raise ValueError("state shapes or batch size do not match")
    hidden_size = h_prev.shape[1]
    _validate_parameters(params, x_t.shape[1], hidden_size)
    gates = (
        x_t @ params["weight_ih"].T
        + params["bias_ih"]
        + h_prev @ params["weight_hh"].T
        + params["bias_hh"]
    )
    i_pre, f_pre, g_pre, o_pre = np.split(gates, 4, axis=1)
    i = sigmoid(i_pre)
    f = sigmoid(f_pre)
    g = np.tanh(g_pre)
    o = sigmoid(o_pre)
    c_t = f * c_prev + i * g
    h_t = o * np.tanh(c_t)
    cache: StepCache = {
        "x_t": x_t,
        "h_prev": h_prev,
        "c_prev": c_prev,
        "i": i,
        "f": f,
        "g": g,
        "o": o,
        "c_t": c_t,
        "h_t": h_t,
        "params": params,
    }
    return h_t, c_t, cache


def sequence_forward(
    x: np.ndarray,
    initial_state: tuple[np.ndarray, np.ndarray],
    params: Parameters,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], list[StepCache]]:
    """Unroll the LSTM over batch-first input."""

    x = np.asarray(x)
    h0, c0 = (np.asarray(value) for value in initial_state)
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError("x must have shape [batch, nonzero time, input]")
    if h0.ndim != 2 or h0.shape != c0.shape or h0.shape[0] != x.shape[0]:
        raise ValueError("initial_state must contain matching [batch, hidden] arrays")
    _validate_parameters(params, x.shape[2], h0.shape[1])
    outputs = np.empty(
        (x.shape[0], x.shape[1], h0.shape[1]),
        dtype=np.result_type(x, h0, c0, params["weight_ih"]),
    )
    caches: list[StepCache] = []
    h_t, c_t = h0, c0
    for time in range(x.shape[1]):
        h_t, c_t, step_cache = step_forward(x[:, time], (h_t, c_t), params)
        outputs[:, time] = h_t
        caches.append(step_cache)
    return outputs, (h_t, c_t), caches


def sequence_backward(
    doutputs: np.ndarray,
    cache: list[StepCache],
    dh_last: np.ndarray | None = None,
    dc_last: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], Parameters]:
    """Run BPTT through both hidden and cell-state paths."""

    doutputs = np.asarray(doutputs)
    if not cache:
        raise ValueError("cache must contain at least one time step")
    first = cache[0]
    x_first = first["x_t"]
    h_first = first["h_prev"]
    params: Parameters = first["params"]
    batch_size, input_size = x_first.shape
    hidden_size = h_first.shape[1]
    time_steps = len(cache)
    if doutputs.shape != (batch_size, time_steps, hidden_size):
        raise ValueError("doutputs shape does not match the cached sequence")
    dh_next = np.zeros_like(h_first) if dh_last is None else np.asarray(dh_last)
    dc_next = np.zeros_like(h_first) if dc_last is None else np.asarray(dc_last)
    if dh_next.shape != h_first.shape or dc_next.shape != h_first.shape:
        raise ValueError("final state gradients must have shape [batch, hidden]")

    dx = np.zeros(
        (batch_size, time_steps, input_size),
        dtype=np.result_type(doutputs, x_first, params["weight_ih"]),
    )
    gradients = {name: np.zeros_like(value) for name, value in params.items()}
    for time in range(time_steps - 1, -1, -1):
        step = cache[time]
        x_t, h_prev, c_prev = step["x_t"], step["h_prev"], step["c_prev"]
        i, f, g, o, c_t = step["i"], step["f"], step["g"], step["o"], step["c_t"]
        dh_total = doutputs[:, time] + dh_next
        tanh_c = np.tanh(c_t)
        do = dh_total * tanh_c
        dc_total = dc_next + dh_total * o * (1.0 - tanh_c**2)
        df = dc_total * c_prev
        dc_next = dc_total * f
        di = dc_total * g
        dg = dc_total * i
        di_pre = di * i * (1.0 - i)
        df_pre = df * f * (1.0 - f)
        dg_pre = dg * (1.0 - g**2)
        do_pre = do * o * (1.0 - o)
        dgate = np.concatenate((di_pre, df_pre, dg_pre, do_pre), axis=1)
        gradients["weight_ih"] += dgate.T @ x_t
        gradients["weight_hh"] += dgate.T @ h_prev
        bias_gradient = dgate.sum(axis=0)
        gradients["bias_ih"] += bias_gradient
        gradients["bias_hh"] += bias_gradient
        dx[:, time] = dgate @ params["weight_ih"]
        dh_next = dgate @ params["weight_hh"]
    return dx, (dh_next, dc_next), gradients
