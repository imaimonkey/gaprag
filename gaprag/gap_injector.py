from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class InjectionPayload:
    model_kwargs: dict[str, Any]
    meta: dict[str, Any]


class GapInjector:
    def __init__(self, mode: str = "residual_hidden", alpha: float = 0.3, prefix_length: int = 1) -> None:
        self.mode = mode
        self.alpha = float(alpha)
        self.prefix_length = int(max(prefix_length, 1))

    def _match_hidden_dim(self, vector: np.ndarray, hidden_dim: int) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32).reshape(-1)
        if v.shape[0] == hidden_dim:
            return v
        if v.shape[0] > hidden_dim:
            return v[:hidden_dim]
        out = np.zeros(hidden_dim, dtype=np.float32)
        out[: v.shape[0]] = v
        return out

    def prepare(
        self,
        model,
        encoded_inputs: dict[str, torch.Tensor],
        gap_vector: np.ndarray,
        mode: str | None = None,
        alpha: float | None = None,
        target: str = "last_token",
        prefix_length: int | None = None,
    ) -> InjectionPayload:
        inj_mode = mode or self.mode
        inj_alpha = self.alpha if alpha is None else float(alpha)
        prefix_len = self.prefix_length if prefix_length is None else int(prefix_length)

        input_ids = encoded_inputs["input_ids"]
        attention_mask = encoded_inputs["attention_mask"]

        if inj_mode in {"none", "no_injection"}:
            return InjectionPayload(
                model_kwargs={"input_ids": input_ids, "attention_mask": attention_mask},
                meta={"mode": inj_mode, "applied": False},
            )

        embedding_layer = model.get_input_embeddings()
        embeds = embedding_layer(input_ids)
        hidden_dim = embeds.shape[-1]
        gap = self._match_hidden_dim(gap_vector, hidden_dim)
        gap_tensor = torch.tensor(gap, dtype=embeds.dtype, device=embeds.device).view(1, 1, -1)

        if inj_mode == "residual_hidden":
            if target == "all_tokens":
                mask = attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)
                embeds = embeds + inj_alpha * gap_tensor * mask
            else:
                last_idx = torch.clamp(attention_mask.long().sum(dim=1) - 1, min=0)
                for b in range(embeds.shape[0]):
                    embeds[b, last_idx[b], :] = embeds[b, last_idx[b], :] + inj_alpha * gap_tensor[0, 0, :]

            return InjectionPayload(
                model_kwargs={"inputs_embeds": embeds, "attention_mask": attention_mask},
                meta={"mode": inj_mode, "applied": True, "target": target, "alpha": inj_alpha},
            )

        if inj_mode == "prefix_bias":
            prefix = inj_alpha * gap_tensor.repeat(embeds.shape[0], prefix_len, 1)
            cat_embeds = torch.cat([prefix, embeds], dim=1)
            prefix_mask = torch.ones(
                (attention_mask.shape[0], prefix_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            cat_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            return InjectionPayload(
                model_kwargs={"inputs_embeds": cat_embeds, "attention_mask": cat_mask},
                meta={"mode": inj_mode, "applied": True, "prefix_length": prefix_len, "alpha": inj_alpha},
            )

        if inj_mode == "attention_bias":
            raise NotImplementedError(
                "attention_bias injection is reserved for future extension. "
                "Use residual_hidden or prefix_bias for MVP."
            )

        raise ValueError(f"Unsupported injection mode: {inj_mode}")
