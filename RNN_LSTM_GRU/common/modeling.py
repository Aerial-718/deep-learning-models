"""Model factories used by the task-agnostic training scripts."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ModelName = Literal["vanilla", "lstm", "gru"]


def build_language_model(
    model_name: ModelName,
    vocabulary_size: int,
    embedding_size: int,
    hidden_size: int,
) -> nn.Module:
    if model_name == "vanilla":
        from vanilla_rnn.language_model import CharLanguageModel
    elif model_name == "lstm":
        from lstm.language_model import CharLanguageModel
    elif model_name == "gru":
        from gru.language_model import CharLanguageModel
    else:
        raise ValueError(f"unknown model: {model_name}")
    return CharLanguageModel(vocabulary_size, embedding_size, hidden_size)


def build_recurrent_layer(
    model_name: ModelName,
    input_size: int,
    hidden_size: int,
) -> nn.Module:
    if model_name == "vanilla":
        from vanilla_rnn.torch_impl import ManualRecurrentLayer
    elif model_name == "lstm":
        from lstm.torch_impl import ManualRecurrentLayer
    elif model_name == "gru":
        from gru.torch_impl import ManualRecurrentLayer
    else:
        raise ValueError(f"unknown model: {model_name}")
    return ManualRecurrentLayer(input_size, hidden_size)


class DelayedRecallClassifier(nn.Module):
    """Shared task head; the recurrent layer remains learner-implemented."""

    def __init__(
        self,
        model_name: ModelName,
        vocabulary_size: int,
        embedding_size: int,
        hidden_size: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.recurrent = build_recurrent_layer(model_name, embedding_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        outputs, _ = self.recurrent(embedded)
        return self.classifier(outputs[:, -1])

