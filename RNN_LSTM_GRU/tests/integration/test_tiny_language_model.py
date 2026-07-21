import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from common.modeling import build_language_model

pytestmark = [pytest.mark.core, pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("model_name", ["vanilla", "lstm", "gru"])
def test_tiny_language_model_overfits_one_batch(model_name: str) -> None:
    torch.manual_seed(123)
    model = build_language_model(model_name, vocabulary_size=5, embedding_size=8, hidden_size=16)
    tokens = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 1, 2],
            [2, 3, 4, 0, 1, 2, 3, 4],
        ]
    )
    targets = torch.roll(tokens, shifts=-1, dims=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < 0.35 * losses[0]

    generated = model.generate(tokens[:1, :3], new_tokens=4, top_k=1)
    assert generated.shape == (1, 7)
    torch.testing.assert_close(generated[:, :3], tokens[:1, :3])

