from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GapEstimate:
    vector: np.ndarray
    query_proj: np.ndarray
    evidence_proj: np.ndarray
    confidence_weight: float
    raw_gap_norm: float


class FixedLinearAligner:
    """Deterministic frozen linear alignment by truncate/pad.

    This is equivalent to a fixed matrix projection without learned parameters.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def transform(self, vec: np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(-1)
        if x.shape[0] == self.out_dim:
            return x
        if x.shape[0] > self.out_dim:
            return x[: self.out_dim]
        out = np.zeros(self.out_dim, dtype=np.float32)
        out[: x.shape[0]] = x
        return out


class GapEstimator:
    def __init__(
        self,
        gap_type: str = "diff",
        target_dim: int | None = None,
        confidence_weight_source: str = "retrieval",
        normalize_inputs: bool = True,
        normalize_gap: bool = True,
        max_gap_norm: float | None = 1.0,
    ) -> None:
        self.gap_type = gap_type
        self.target_dim = target_dim
        self.confidence_weight_source = confidence_weight_source
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_gap = bool(normalize_gap)
        self.max_gap_norm = None if max_gap_norm is None else float(max_gap_norm)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        x = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(x))
        if norm <= 1e-8:
            return x
        return x / norm

    def _align_pair(self, query_vec: np.ndarray, evidence_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        d = np.asarray(evidence_vec, dtype=np.float32).reshape(-1)

        if self.target_dim is not None:
            q_aligner = FixedLinearAligner(q.shape[0], self.target_dim)
            d_aligner = FixedLinearAligner(d.shape[0], self.target_dim)
            return q_aligner.transform(q), d_aligner.transform(d)

        common_dim = max(q.shape[0], d.shape[0])
        q_aligner = FixedLinearAligner(q.shape[0], common_dim)
        d_aligner = FixedLinearAligner(d.shape[0], common_dim)
        return q_aligner.transform(q), d_aligner.transform(d)

    def compute(
        self,
        query_vec: np.ndarray,
        evidence_vec: np.ndarray,
        retrieval_confidence: float | None = None,
        generation_uncertainty: float | None = None,
    ) -> GapEstimate:
        q_proj, d_proj = self._align_pair(query_vec, evidence_vec)
        if self.normalize_inputs:
            q_proj = self._l2_normalize(q_proj)
            d_proj = self._l2_normalize(d_proj)

        if self.gap_type in {"diff", "proj_diff", "token_level_aligned"}:
            gap = d_proj - q_proj
        elif self.gap_type == "confidence_weighted":
            gap = d_proj - q_proj
        else:
            raise ValueError(f"Unsupported gap type: {self.gap_type}")
        raw_gap_norm = float(np.linalg.norm(gap))

        confidence = 1.0
        if self.gap_type == "confidence_weighted":
            if self.confidence_weight_source == "retrieval" and retrieval_confidence is not None:
                confidence = float(max(0.0, retrieval_confidence))
            elif self.confidence_weight_source == "uncertainty" and generation_uncertainty is not None:
                confidence = float(max(0.0, generation_uncertainty))
            elif retrieval_confidence is not None:
                confidence = float(max(0.0, retrieval_confidence))
            gap = gap * confidence

        if self.normalize_gap:
            gap = self._l2_normalize(gap)
        if self.max_gap_norm is not None:
            gap_norm = float(np.linalg.norm(gap))
            if gap_norm > self.max_gap_norm and gap_norm > 1e-8:
                gap = gap * (self.max_gap_norm / gap_norm)

        return GapEstimate(
            vector=gap.astype(np.float32),
            query_proj=q_proj.astype(np.float32),
            evidence_proj=d_proj.astype(np.float32),
            confidence_weight=float(confidence),
            raw_gap_norm=raw_gap_norm,
        )
