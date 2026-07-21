"""Finite-difference helpers for checking hand-written backward passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class GradientCheckResult:
    name: str
    relative_error: float
    max_absolute_error: float
    passed: bool


def numerical_gradient(
    scalar_function: Callable[[], float],
    parameter: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute a central finite-difference gradient in place, then restore input."""

    if not np.issubdtype(parameter.dtype, np.floating):
        raise TypeError("parameter must use a floating dtype")
    gradient = np.zeros_like(parameter, dtype=np.float64)
    iterator = np.nditer(parameter, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = float(parameter[index])
        parameter[index] = original + epsilon
        positive = float(scalar_function())
        parameter[index] = original - epsilon
        negative = float(scalar_function())
        parameter[index] = original
        gradient[index] = (positive - negative) / (2.0 * epsilon)
        iterator.iternext()
    return gradient


def relative_error(actual: np.ndarray, expected: np.ndarray, floor: float = 1e-12) -> float:
    """Return max(|a-b| / max(floor, |a|+|b|))."""

    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    denominator = np.maximum(floor, np.abs(actual) + np.abs(expected))
    return float(np.max(np.abs(actual - expected) / denominator, initial=0.0))


def check_gradient(
    name: str,
    analytic: np.ndarray,
    numerical: np.ndarray,
    tolerance: float = 1e-5,
) -> GradientCheckResult:
    error = relative_error(analytic, numerical)
    absolute = float(np.max(np.abs(np.asarray(analytic) - np.asarray(numerical)), initial=0.0))
    return GradientCheckResult(
        name=name,
        relative_error=error,
        max_absolute_error=absolute,
        passed=error < tolerance,
    )

