from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BaseGapMemory:
    def get(self, query_key: np.ndarray | None = None) -> np.ndarray | None:
        raise NotImplementedError

    def update(self, gap_vec: np.ndarray, query_key: np.ndarray | None = None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class NoMemory(BaseGapMemory):
    def get(self, query_key: np.ndarray | None = None) -> np.ndarray | None:
        return None

    def update(self, gap_vec: np.ndarray, query_key: np.ndarray | None = None) -> None:
        _ = (gap_vec, query_key)

    def reset(self) -> None:
        return


class EMAMemory(BaseGapMemory):
    def __init__(self, decay: float = 0.9) -> None:
        self.decay = float(decay)
        self.state: np.ndarray | None = None

    def get(self, query_key: np.ndarray | None = None) -> np.ndarray | None:
        _ = query_key
        return None if self.state is None else self.state.copy()

    def update(self, gap_vec: np.ndarray, query_key: np.ndarray | None = None) -> None:
        _ = query_key
        g = np.asarray(gap_vec, dtype=np.float32)
        if self.state is None:
            self.state = g.copy()
        else:
            self.state = self.decay * self.state + (1.0 - self.decay) * g

    def reset(self) -> None:
        self.state = None


@dataclass
class KeyedEntry:
    key: np.ndarray
    gap: np.ndarray


class QueryKeyedMemory(BaseGapMemory):
    def __init__(self, max_items: int = 256, temperature: float = 0.1) -> None:
        self.max_items = int(max_items)
        self.temperature = float(max(temperature, 1e-4))
        self.entries: list[KeyedEntry] = []

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / den)

    def get(self, query_key: np.ndarray | None = None) -> np.ndarray | None:
        if not self.entries:
            return None
        if query_key is None:
            return np.mean(np.stack([e.gap for e in self.entries], axis=0), axis=0)

        q = np.asarray(query_key, dtype=np.float32).reshape(-1)
        sims = np.asarray([self._cosine(q, e.key) for e in self.entries], dtype=np.float32)
        weights = np.exp(sims / self.temperature)
        weights = weights / (np.sum(weights) + 1e-8)
        gaps = np.stack([e.gap for e in self.entries], axis=0)
        return np.sum(gaps * weights[:, None], axis=0)

    def update(self, gap_vec: np.ndarray, query_key: np.ndarray | None = None) -> None:
        g = np.asarray(gap_vec, dtype=np.float32).reshape(-1)
        if query_key is None:
            key = g.copy()
        else:
            key = np.asarray(query_key, dtype=np.float32).reshape(-1)

        self.entries.append(KeyedEntry(key=key, gap=g))
        if len(self.entries) > self.max_items:
            self.entries = self.entries[-self.max_items :]

    def reset(self) -> None:
        self.entries = []


class BoundedGapBank(BaseGapMemory):
    def __init__(self, max_items: int = 128) -> None:
        self.max_items = int(max_items)
        self.bank: list[np.ndarray] = []

    def get(self, query_key: np.ndarray | None = None) -> np.ndarray | None:
        _ = query_key
        if not self.bank:
            return None
        return np.mean(np.stack(self.bank, axis=0), axis=0)

    def update(self, gap_vec: np.ndarray, query_key: np.ndarray | None = None) -> None:
        _ = query_key
        self.bank.append(np.asarray(gap_vec, dtype=np.float32).reshape(-1))
        if len(self.bank) > self.max_items:
            self.bank = self.bank[-self.max_items :]

    def reset(self) -> None:
        self.bank = []


def build_memory(memory_type: str, memory_cfg: dict | None = None) -> BaseGapMemory:
    cfg = memory_cfg or {}
    if memory_type == "none":
        return NoMemory()
    if memory_type == "ema":
        return EMAMemory(decay=float(cfg.get("ema_decay", 0.9)))
    if memory_type == "keyed":
        return QueryKeyedMemory(
            max_items=int(cfg.get("max_items", 256)),
            temperature=float(cfg.get("temperature", 0.1)),
        )
    if memory_type == "bounded":
        return BoundedGapBank(max_items=int(cfg.get("max_items", 128)))
    raise ValueError(f"Unsupported memory type: {memory_type}")
