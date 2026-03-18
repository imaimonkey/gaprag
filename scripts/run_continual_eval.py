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


def run_setting(
    pipeline: GapRAGPipeline,
    dataset: list[dict],
    mode: str,
    setting: str,
) -> tuple[list[dict], dict]:
    pipeline.reset_session(None)

    rows: list[dict] = []
    correct_so_far = 0.0
    for step_idx, item in enumerate(tqdm(dataset, desc=f"{setting}/{mode}"), start=1):
        if setting == "stateless":
            session_id = str(item.get("id", step_idx))
            pipeline.reset_session(session_id)
        else:
            session_id = str(item.get("session_id", "default"))

        out = pipeline.run_query(question=item["question"], mode=mode, session_id=session_id)

        retrieved_ids = [d["doc_id"] for d in out.retrieved_docs]
        retrieved_texts = [d["text"] for d in out.retrieved_docs]

        em = exact_match(out.prediction, item.get("answers", []))
        f1 = token_f1(out.prediction, item.get("answers", []))
        hit = retrieval_hit_at_k(retrieved_ids, item.get("context", []))
        hall = hallucination_proxy(out.prediction, retrieved_texts)

        correct_so_far += em
        cumulative_em = correct_so_far / step_idx

        row = {
            "setting": setting,
            "mode": mode,
            "step": step_idx,
            "id": item.get("id", step_idx),
            "session_id": session_id,
            "question": item.get("question", ""),
            "answers": item.get("answers", []),
            "prediction": out.prediction,
            "exact_match": em,
            "f1": f1,
            "retrieval_hit_at_k": hit,
            "hallucination_proxy": hall,
            "gap_norm": out.gap_norm,
            "memory_norm": out.memory_norm,
            "avg_retrieval_score": out.gap_stats.get("avg_retrieval_score", 0.0),
            "prediction_confidence": out.prediction_confidence,
            "elapsed_sec": out.elapsed_sec,
            "cumulative_em": cumulative_em,
            "pipeline_output": asdict(out),
        }
        rows.append(row)

    summary = summarize_scores(rows)
    summary["mode"] = mode
    summary["setting"] = setting
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stateless vs continual evaluation for GapRAG")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed)

    mode = args.mode or cfg.get("pipeline", {}).get("mode", "gap_memory_ema")
    run_name = args.run_name or f"continual_{mode}"
    run_dir = create_run_dir(cfg.get("experiment", {}).get("output_dir", "outputs/runs"), run_name)
    logger = setup_logger(run_dir / "run.log")

    snapshot_config(cfg, run_dir)

    dataset = load_qa_dataset(cfg)
    logger.info("loaded dataset size=%d", len(dataset))
    pipeline = GapRAGPipeline.from_config(cfg)

    rows_stateless, summary_stateless = run_setting(pipeline, dataset, mode, setting="stateless")
    rows_continual, summary_continual = run_setting(pipeline, dataset, mode, setting="continual")

    save_jsonl(rows_stateless, run_dir / "predictions_stateless.jsonl")
    save_jsonl(rows_continual, run_dir / "predictions_continual.jsonl")

    df_stateless = pd.DataFrame(rows_stateless)
    df_continual = pd.DataFrame(rows_continual)
    df_stateless.drop(columns=["pipeline_output"]).to_csv(run_dir / "step_metrics_stateless.csv", index=False)
    df_continual.drop(columns=["pipeline_output"]).to_csv(run_dir / "step_metrics_continual.csv", index=False)

    save_json(summary_stateless, run_dir / "summary_stateless.json")
    save_json(summary_continual, run_dir / "summary_continual.json")

    compare = {
        "mode": mode,
        "seed": seed,
        "stateless": summary_stateless,
        "continual": summary_continual,
        "delta_exact_match": summary_continual["exact_match"] - summary_stateless["exact_match"],
        "delta_f1": summary_continual["f1"] - summary_stateless["f1"],
        "delta_retrieval_hit_at_k": summary_continual["retrieval_hit_at_k"] - summary_stateless["retrieval_hit_at_k"],
    }
    save_json(compare, run_dir / "compare_summary.json")

    logger.info("run_dir=%s", run_dir)
    logger.info("mode=%s seed=%d", mode, seed)
    logger.info("stateless EM=%.4f | continual EM=%.4f | delta=%.4f", summary_stateless["exact_match"], summary_continual["exact_match"], compare["delta_exact_match"])


if __name__ == "__main__":
    main()
