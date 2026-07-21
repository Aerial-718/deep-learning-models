import numpy as np
import pytest

from common.data import (
    CharVocabulary,
    contiguous_split,
    delayed_recall_batch,
    one_hot,
    random_char_batch,
)


def test_character_vocabulary_is_sorted_and_round_trips() -> None:
    vocabulary = CharVocabulary.build("cabca\n")
    assert vocabulary.chars == ("\n", "a", "b", "c")
    encoded = vocabulary.encode("cab\n")
    assert encoded.dtype == np.int64
    assert vocabulary.decode(encoded) == "cab\n"


def test_contiguous_split_preserves_order() -> None:
    tokens = np.arange(100)
    train, validation, test = contiguous_split(tokens)
    assert (len(train), len(validation), len(test)) == (90, 5, 5)
    np.testing.assert_array_equal(np.concatenate([train, validation, test]), tokens)


def test_random_char_batch_returns_shifted_targets() -> None:
    tokens = np.arange(30)
    x, y = random_char_batch(tokens, batch_size=4, sequence_length=6, rng=np.random.default_rng(7))
    assert x.shape == y.shape == (4, 6)
    np.testing.assert_array_equal(y[:, :-1], x[:, 1:])


def test_delayed_recall_places_target_and_query() -> None:
    x, y = delayed_recall_batch(16, delay=20, num_classes=8, rng=np.random.default_rng(3))
    assert x.shape == (16, 22)
    np.testing.assert_array_equal(x[:, 0], y)
    np.testing.assert_array_equal(x[:, -1], np.full(16, 9))
    assert np.all((0 <= y) & (y < 8))


def test_one_hot_validates_ids() -> None:
    encoded = one_hot(np.asarray([[0, 2], [1, 0]]), vocabulary_size=3)
    assert encoded.shape == (2, 2, 3)
    np.testing.assert_allclose(encoded.sum(axis=-1), 1.0)
    with pytest.raises(ValueError):
        one_hot(np.asarray([3]), vocabulary_size=3)

