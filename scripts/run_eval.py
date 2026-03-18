#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaprag.datasets import load_qa_dataset
from gaprag.logging_utils import create_run_dir, setup_logger, snapshot_config
from gaprag.metrics import exact_match, hallucination_proxy, retrieval_hit_at_k, summarize_scores, token_f1
from gaprag.pipeline import GapRAGPipeline
from gaprag.utils import load_yaml, save_json, save_jsonl, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GapRAG single-pass evaluation")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--stateless", action="store_true", help="Reset memory for each query")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    mode = args.mode or cfg.get("pipeline", {}).get("mode", "gap_memory_ema")
    run_name = args.run_name or f"eval_{mode}"
    run_dir = create_run_dir(cfg.get("experiment", {}).get("output_dir", "outputs/runs"), run_name)
    logger = setup_logger(run_dir / "run.log")

    snapshot_config(cfg, run_dir)
    logger.info("run_dir=%s", run_dir)
    logger.info("mode=%s seed=%d", mode, seed)

    dataset = load_qa_dataset(cfg)
    logger.info("loaded dataset size=%d", len(dataset))

    pipeline = GapRAGPipeline.from_config(cfg)

    rows = []
    for i, item in enumerate(tqdm(dataset, desc=f"eval/{mode}")):
        question = item["question"]
        answers = item.get("answers", [])
        session_id = str(item.get("session_id", "default"))
        if args.stateless:
            session_id = str(item.get("id", i))

        out = pipeline.run_query(question=question, mode=mode, session_id=session_id)
        out_dict = asdict(out)

        retrieved_ids = [d["doc_id"] for d in out.retrieved_docs]
        retrieved_texts = [d["text"] for d in out.retrieved_docs]

        row = {
            "id": item.get("id", i),
            "session_id": session_id,
            "question": question,
            "answers": answers,
            "prediction": out.prediction,
            "mode": mode,
            "exact_match": exact_match(out.prediction, answers),
            "f1": token_f1(out.prediction, answers),
            "retrieval_hit_at_k": retrieval_hit_at_k(retrieved_ids, item.get("context", [])),
            "hallucination_proxy": hallucination_proxy(out.prediction, retrieved_texts),
            "gap_norm": out.gap_norm,
            "memory_norm": out.memory_norm,
            "prediction_confidence": out.prediction_confidence,
            "avg_retrieval_score": out.gap_stats.get("avg_retrieval_score", 0.0),
            "elapsed_sec": out.elapsed_sec,
            "pipeline_output": out_dict,
        }
        rows.append(row)

    summary = summarize_scores(rows)
    summary["mode"] = mode
    summary["seed"] = seed
    summary["stateless"] = bool(args.stateless)

    pred_jsonl = run_dir / "predictions.jsonl"
    metrics_json = run_dir / "metrics_summary.json"
    metrics_csv = run_dir / "metrics_table.csv"

    save_jsonl(rows, pred_jsonl)
    save_json(summary, metrics_json)
    pd.DataFrame(rows).drop(columns=["pipeline_output"]).to_csv(metrics_csv, index=False)

    logger.info("saved predictions=%s", pred_jsonl)
    logger.info("saved metrics summary=%s", metrics_json)
    logger.info("EM=%.4f F1=%.4f Hit@k=%.4f", summary["exact_match"], summary["f1"], summary["retrieval_hit_at_k"])


if __name__ == "__main__":
    main()
