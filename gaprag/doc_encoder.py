from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-8)


class EvidenceEncoder:
    def __init__(self, method: str = "score_weighted_mean") -> None:
        self.method = method

    def aggregate(
        self,
        doc_embeddings: np.ndarray,
        retrieval_scores: np.ndarray | None = None,
        query_embedding: np.ndarray | None = None,
    ) -> np.ndarray:
        if doc_embeddings.size == 0:
            raise ValueError("doc_embeddings is empty")

        if self.method == "mean":
            return np.mean(doc_embeddings, axis=0)

        if self.method == "score_weighted_mean":
            if retrieval_scores is None:
                return np.mean(doc_embeddings, axis=0)
            weights = _softmax(np.asarray(retrieval_scores, dtype=np.float32))
            return np.sum(doc_embeddings * weights[:, None], axis=0)

        if self.method == "query_aware_weighted_mean":
            if query_embedding is None:
                if retrieval_scores is None:
                    return np.mean(doc_embeddings, axis=0)
                weights = _softmax(np.asarray(retrieval_scores, dtype=np.float32))
                return np.sum(doc_embeddings * weights[:, None], axis=0)

            query_norm = np.linalg.norm(query_embedding) + 1e-8
            doc_norm = np.linalg.norm(doc_embeddings, axis=1) + 1e-8
            cosine = (doc_embeddings @ query_embedding) / (doc_norm * query_norm)

            if retrieval_scores is not None:
                combined = cosine + np.asarray(retrieval_scores, dtype=np.float32)
            else:
                combined = cosine
            weights = _softmax(combined)
            return np.sum(doc_embeddings * weights[:, None], axis=0)

        raise ValueError(f"Unsupported evidence encoder method: {self.method}")
