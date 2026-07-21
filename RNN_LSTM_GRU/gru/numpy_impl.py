"""Reference NumPy GRU using PyTorch-compatible reset-after equations."""

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
    """Create three-gate parameters ordered ``r, z, n``."""

    if input_size <= 0 or hidden_size <= 0:
        raise ValueError("input_size and hidden_size must be positive")
    bound = 1.0 / np.sqrt(hidden_size)
    gate_size = 3 * hidden_size
    return {
        "weight_ih": rng.uniform(-bound, bound, (gate_size, input_size)).astype(np.float64),
        "weight_hh": rng.uniform(-bound, bound, (gate_size, hidden_size)).astype(np.float64),
        "bias_ih": np.zeros(gate_size, dtype=np.float64),
        "bias_hh": np.zeros(gate_size, dtype=np.float64),
    }


def _validate_parameters(params: Parameters, input_size: int, hidden_size: int) -> None:
    gate_size = 3 * hidden_size
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
    h_prev: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, StepCache]:
    """Compute one PyTorch-compatible reset-after GRU step."""

    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    if x_t.ndim != 2 or h_prev.ndim != 2:
        raise ValueError("x_t and h_prev must be two-dimensional")
    if x_t.shape[0] != h_prev.shape[0]:
        raise ValueError("x_t and h_prev must have the same batch size")
    hidden_size = h_prev.shape[1]
    _validate_parameters(params, x_t.shape[1], hidden_size)
    input_projection = x_t @ params["weight_ih"].T + params["bias_ih"]
    hidden_projection = h_prev @ params["weight_hh"].T + params["bias_hh"]
    x_r, x_z, x_n = np.split(input_projection, 3, axis=1)
    h_r, h_z, h_n = np.split(hidden_projection, 3, axis=1)
    r = sigmoid(x_r + h_r)
    z = sigmoid(x_z + h_z)
    n = np.tanh(x_n + r * h_n)
    h_t = (1.0 - z) * n + z * h_prev
    cache: StepCache = {
        "x_t": x_t,
        "h_prev": h_prev,
        "r": r,
        "z": z,
        "n": n,
        "h_n": h_n,
        "h_t": h_t,
        "params": params,
    }
    return h_t, cache


def sequence_forward(
    x: np.ndarray,
    h0: np.ndarray,
    params: Parameters,
) -> tuple[np.ndarray, np.ndarray, list[StepCache]]:
    """Unroll the GRU over batch-first input."""

    x = np.asarray(x)
    h0 = np.asarray(h0)
    if x.ndim != 3 or x.shape[1] == 0:
        raise ValueError("x must have shape [batch, nonzero time, input]")
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
    """Backpropagate through all GRU branches and time steps."""

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
    if dh_next.shape != h_first.shape:
        raise ValueError("dh_last must have shape [batch, hidden]")

    dx = np.zeros(
        (batch_size, time_steps, input_size),
        dtype=np.result_type(doutputs, x_first, params["weight_ih"]),
    )
    gradients = {name: np.zeros_like(value) for name, value in params.items()}
    for time in range(time_steps - 1, -1, -1):
        step = cache[time]
        x_t, h_prev = step["x_t"], step["h_prev"]
        r, z, n, h_n = step["r"], step["z"], step["n"], step["h_n"]
        dh_total = doutputs[:, time] + dh_next
        dn = dh_total * (1.0 - z)
        dz = dh_total * (h_prev - n)
        dh_direct = dh_total * z
        dn_pre = dn * (1.0 - n**2)
        dr = dn_pre * h_n
        dh_n = dn_pre * r
        dr_pre = dr * r * (1.0 - r)
        dz_pre = dz * z * (1.0 - z)
        dinput_projection = np.concatenate((dr_pre, dz_pre, dn_pre), axis=1)
        dhidden_projection = np.concatenate((dr_pre, dz_pre, dh_n), axis=1)
        gradients["weight_ih"] += dinput_projection.T @ x_t
        gradients["weight_hh"] += dhidden_projection.T @ h_prev
        gradients["bias_ih"] += dinput_projection.sum(axis=0)
        gradients["bias_hh"] += dhidden_projection.sum(axis=0)
        dx[:, time] = dinput_projection @ params["weight_ih"]
        dh_next = dh_direct + dhidden_projection @ params["weight_hh"]
    return dx, dh_next, gradients
