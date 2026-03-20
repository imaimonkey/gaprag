from __future__ import annotations

import numpy as np

from gapverify.gap_estimator import GapEstimator


def test_gap_estimator_preserves_raw_gap_norm_when_not_normalizing() -> None:
    estimator = GapEstimator(
        gap_type="diff",
        target_dim=None,
        normalize_inputs=False,
        normalize_gap=False,
        max_gap_norm=None,
    )
    query = np.asarray([0.0, 0.0], dtype=np.float32)
    evidence = np.asarray([3.0, 4.0], dtype=np.float32)

    estimate = estimator.compute(query_vec=query, evidence_vec=evidence)

    assert np.allclose(estimate.vector, np.asarray([3.0, 4.0], dtype=np.float32))
    assert estimate.raw_gap_norm == 5.0


def test_gap_estimator_clips_gap_norm_without_erasing_raw_norm() -> None:
    estimator = GapEstimator(
        gap_type="diff",
        target_dim=None,
        normalize_inputs=False,
        normalize_gap=False,
        max_gap_norm=1.25,
    )
    query = np.asarray([0.0, 0.0], dtype=np.float32)
    evidence = np.asarray([3.0, 4.0], dtype=np.float32)

    estimate = estimator.compute(query_vec=query, evidence_vec=evidence)

    assert estimate.raw_gap_norm == 5.0
    assert np.isclose(np.linalg.norm(estimate.vector), 1.25)
