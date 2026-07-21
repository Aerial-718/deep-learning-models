import numpy as np
import pytest

torch = pytest.importorskip("torch")

from common.data import delayed_recall_batch
from common.modeling import DelayedRecallClassifier

pytestmark = [pytest.mark.core, pytest.mark.integration]


@pytest.mark.parametrize("model_name", ["vanilla", "lstm", "gru"])
def test_delayed_recall_forward_contract(model_name: str) -> None:
    rng = np.random.default_rng(42)
    x, targets = delayed_recall_batch(8, delay=5, num_classes=4, rng=rng)
    model = DelayedRecallClassifier(
        model_name,
        vocabulary_size=6,
        embedding_size=7,
        hidden_size=9,
        num_classes=4,
    )
    logits = model(torch.from_numpy(x))
    assert logits.shape == (len(targets), 4)

