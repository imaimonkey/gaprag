from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

from .utils import safe_div


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    norm_pred = normalize_answer(prediction)
    for answer in answers:
        if norm_pred == normalize_answer(str(answer)):
            return 1.0
    return 0.0


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for answer in answers:
        ans_tokens = normalize_answer(str(answer)).split()
        if not ans_tokens:
            continue
        common = Counter(pred_tokens) & Counter(ans_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = safe_div(num_same, len(pred_tokens))
        recall = safe_div(num_same, len(ans_tokens))
        score = safe_div(2 * precision * recall, precision + recall)
        best = max(best, score)
    return float(best)


def retrieval_hit_at_k(retrieved_doc_ids: list[str], gold_doc_ids: list[str]) -> float:
    if not gold_doc_ids:
        return 0.0
    gold = set(str(x) for x in gold_doc_ids)
    hit = any(str(doc_id) in gold for doc_id in retrieved_doc_ids)
    return 1.0 if hit else 0.0


def hallucination_proxy(prediction: str, retrieved_texts: list[str], min_overlap_tokens: int = 1) -> float:
    pred = set(normalize_answer(prediction).split())
    evidence = set()
    for text in retrieved_texts:
        evidence.update(normalize_answer(text).split())
    overlap = len(pred & evidence)
    return 0.0 if overlap >= min_overlap_tokens else 1.0


def summarize_scores(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "exact_match": 0.0,
            "f1": 0.0,
            "retrieval_hit_at_k": 0.0,
            "hallucination_proxy": 0.0,
        }
    n = len(rows)
    return {
        "count": n,
        "exact_match": sum(float(r.get("exact_match", 0.0)) for r in rows) / n,
        "f1": sum(float(r.get("f1", 0.0)) for r in rows) / n,
        "retrieval_hit_at_k": sum(float(r.get("retrieval_hit_at_k", 0.0)) for r in rows) / n,
        "hallucination_proxy": sum(float(r.get("hallucination_proxy", 0.0)) for r in rows) / n,
        "avg_gap_norm": sum(float(r.get("gap_norm", 0.0)) for r in rows) / n,
        "avg_memory_norm": sum(float(r.get("memory_norm", 0.0)) for r in rows) / n,
        "avg_retrieval_score": sum(float(r.get("avg_retrieval_score", 0.0)) for r in rows) / n,
        "avg_prediction_confidence": sum(float(r.get("prediction_confidence", 0.0)) for r in rows) / n,
    }
