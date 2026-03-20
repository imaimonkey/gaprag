from __future__ import annotations

import numpy as np
import torch

from gaprag.gap_injector import GapInjector


class _DummyEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class _DummyModel:
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 8) -> None:
        self.embedding = _DummyEmbedding(vocab_size=vocab_size, hidden_dim=hidden_dim)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding


def test_prefix_bias_injector_expands_sequence_length() -> None:
    model = _DummyModel()
    injector = GapInjector(mode="prefix_bias", alpha=0.2, prefix_length=3)
    encoded = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
    }
    gap = np.ones(8, dtype=np.float32)

    payload = injector.prepare(model, encoded, gap, mode="prefix_bias", alpha=0.2, prefix_length=3)

    assert payload.meta["applied"] is True
    assert payload.meta["prefix_length"] == 3
    assert payload.model_kwargs["inputs_embeds"].shape == (1, 7, 8)
    assert payload.model_kwargs["attention_mask"].shape == (1, 7)
