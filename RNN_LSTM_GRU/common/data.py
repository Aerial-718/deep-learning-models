"""Deterministic datasets used by all three model projects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CharVocabulary:
    """A deterministic character-level vocabulary."""

    chars: tuple[str, ...]

    @classmethod
    def build(cls, text: str) -> "CharVocabulary":
        if not text:
            raise ValueError("text must not be empty")
        return cls(tuple(sorted(set(text))))

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.chars)}

    @property
    def size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        mapping = self.stoi
        try:
            return np.asarray([mapping[char] for char in text], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} is not in the vocabulary") from exc

    def decode(self, token_ids: Iterable[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            if not 0 <= index < self.size:
                raise ValueError(f"token id {index} is outside [0, {self.size})")
            decoded.append(self.chars[index])
        return "".join(decoded)


def contiguous_split(
    tokens: np.ndarray,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a one-dimensional token stream without shuffling."""

    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if len(tokens) < 3:
        raise ValueError("at least three tokens are required")
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1")

    train_end = int(len(tokens) * ratios[0])
    validation_end = train_end + int(len(tokens) * ratios[1])
    if train_end == 0 or validation_end == train_end or validation_end == len(tokens):
        raise ValueError("the token stream is too short for the requested ratios")
    return tokens[:train_end], tokens[train_end:validation_end], tokens[validation_end:]


def random_char_batch(
    tokens: np.ndarray,
    batch_size: int,
    sequence_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw next-character windows from a one-dimensional token stream."""

    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 1:
        raise ValueError("tokens must be one-dimensional")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    max_start = len(tokens) - sequence_length - 1
    if max_start < 0:
        raise ValueError("token stream is shorter than sequence_length + 1")

    starts = rng.integers(0, max_start + 1, size=batch_size)
    x = np.stack([tokens[start : start + sequence_length] for start in starts])
    y = np.stack([tokens[start + 1 : start + sequence_length + 1] for start in starts])
    return x, y


def delayed_recall_batch(
    batch_size: int,
    delay: int,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a delayed-recall classification batch.

    Input IDs use ``0..num_classes-1`` for values, ``num_classes`` for a
    distractor marker, and ``num_classes+1`` for the final query marker.
    The first value token is the target; intermediate positions mix random
    values with distractor markers to prevent a constant-input shortcut.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if delay < 1:
        raise ValueError("delay must be at least 1")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")

    targets = rng.integers(0, num_classes, size=batch_size, dtype=np.int64)
    inputs = rng.integers(
        0,
        num_classes + 1,
        size=(batch_size, delay + 2),
        dtype=np.int64,
    )
    inputs[:, 0] = targets
    inputs[:, -1] = num_classes + 1
    return inputs, targets


def one_hot(token_ids: np.ndarray, vocabulary_size: int) -> np.ndarray:
    """Convert integer token IDs to float32 one-hot vectors."""

    token_ids = np.asarray(token_ids)
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    if token_ids.size and (token_ids.min() < 0 or token_ids.max() >= vocabulary_size):
        raise ValueError("token id outside vocabulary")
    return np.eye(vocabulary_size, dtype=np.float32)[token_ids]

