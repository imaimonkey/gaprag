from __future__ import annotations

from typing import Literal

import torch

Pooling = Literal["last_token", "mean_pool", "final_token"]


def _last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden: [B, T, H], attention_mask: [B, T]
    indices = attention_mask.long().sum(dim=1) - 1
    indices = torch.clamp(indices, min=0)
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, indices, :]


def extract_hidden_vector(
    hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    layer_index: int = -1,
    pooling: Pooling = "last_token",
) -> torch.Tensor:
    if not hidden_states:
        raise ValueError("hidden_states is empty")
    layer_hidden = hidden_states[layer_index]

    if pooling in {"last_token", "final_token"}:
        return _last_valid_token(layer_hidden, attention_mask)
    if pooling == "mean_pool":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (layer_hidden * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / denom
    raise ValueError(f"Unsupported pooling mode: {pooling}")
