"""Shared, non-core utilities for the RNN learning projects."""

from .data import CharVocabulary, contiguous_split, delayed_recall_batch, random_char_batch
from .gradcheck import GradientCheckResult, check_gradient, numerical_gradient, relative_error

__all__ = [
    "CharVocabulary",
    "GradientCheckResult",
    "check_gradient",
    "contiguous_split",
    "delayed_recall_batch",
    "numerical_gradient",
    "random_char_batch",
    "relative_error",
]

