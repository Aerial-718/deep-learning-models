import pytest

torch = pytest.importorskip("torch")

from common.generation import sample_from_logits


def test_top_one_sampling_is_deterministic() -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0], [5.0, 0.0, -1.0]])
    sampled = sample_from_logits(logits, top_k=1)
    torch.testing.assert_close(sampled, torch.tensor([1, 0]))


def test_sampling_validates_temperature() -> None:
    with pytest.raises(ValueError):
        sample_from_logits(torch.zeros(1, 3), temperature=0.0)

