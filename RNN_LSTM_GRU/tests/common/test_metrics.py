import math

import numpy as np

from common.metrics import accuracy, bits_per_character, mean_and_sample_std
from common.training import AverageMeter, EarlyStopping


def test_bits_per_character_uses_log_base_two() -> None:
    assert math.isclose(bits_per_character(math.log(2.0)), 1.0)


def test_accuracy() -> None:
    assert accuracy(np.asarray([1, 0, 2]), np.asarray([1, 2, 2])) == 2 / 3


def test_mean_and_sample_std() -> None:
    mean, std = mean_and_sample_std([1.0, 2.0, 3.0])
    assert mean == 2.0
    assert std == 1.0


def test_average_meter_and_early_stopping() -> None:
    meter = AverageMeter()
    meter.update(2.0, count=2)
    meter.update(5.0)
    assert meter.average == 3.0

    stopping = EarlyStopping(patience=2)
    assert not stopping.update(3.0)
    assert not stopping.update(3.1)
    assert stopping.update(3.2)

