#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaprag.utils import ensure_dir


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GapRAG run outputs")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = ensure_dir(args.out_dir)

    compare_rows = []
    continual_rows = []

    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue

        compare_path = run_dir / "compare_summary.json"
        if compare_path.exists():
            d = _load_json(compare_path)
            compare_rows.append(
                {
                    "run": run_dir.name,
                    "mode": d.get("mode", "unknown"),
                    "stateless_em": d.get("stateless", {}).get("exact_match", 0.0),
                    "continual_em": d.get("continual", {}).get("exact_match", 0.0),
                    "delta_em": d.get("delta_exact_match", 0.0),
                    "stateless_f1": d.get("stateless", {}).get("f1", 0.0),
                    "continual_f1": d.get("continual", {}).get("f1", 0.0),
                    "delta_f1": d.get("delta_f1", 0.0),
                }
            )

        step_cont_path = run_dir / "step_metrics_continual.csv"
        if step_cont_path.exists():
            step_df = pd.read_csv(step_cont_path)
            step_df["run"] = run_dir.name
            continual_rows.append(step_df)

    if not compare_rows and not continual_rows:
        print("No run artifacts found. Run scripts/run_continual_eval.py first.")
        return

    if compare_rows:
        comp_df = pd.DataFrame(compare_rows)
        comp_df.to_csv(Path(args.runs_dir) / "analysis_compare_table.csv", index=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=comp_df, x="run", y="continual_em", color="steelblue", label="continual")
        sns.barplot(data=comp_df, x="run", y="stateless_em", color="lightgray", label="stateless")
        plt.xticks(rotation=45, ha="right")
        plt.title("Overall EM: Stateless vs Continual")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "overall_em_bar.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.barplot(data=comp_df, x="run", y="delta_em", color="darkgreen")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.title("Continual Gain (Delta EM)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "continual_gain_bar.png", dpi=200)
        plt.close()

    if continual_rows:
        all_steps = pd.concat(continual_rows, ignore_index=True)

        plt.figure(figsize=(10, 5))
        sns.lineplot(data=all_steps, x="step", y="cumulative_em", hue="run")
        plt.title("Continual Accuracy Curve (Cumulative EM)")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "continual_accuracy_curve.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=all_steps, x="gap_norm", y="exact_match", hue="run", alpha=0.6)
        plt.title("Gap Norm vs Correctness")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "gap_norm_vs_correctness.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=all_steps, x="memory_norm", y="exact_match", hue="run", alpha=0.6)
        plt.title("Memory Norm vs Correctness")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "memory_norm_vs_correctness.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=all_steps, x="retrieval_hit_at_k", y="exact_match", hue="run", alpha=0.6)
        plt.title("Retrieval Hit@k vs Correctness")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / "retrieval_vs_correctness.png", dpi=200)
        plt.close()

        all_steps.to_csv(Path(args.runs_dir) / "analysis_step_table.csv", index=False)

    print(f"Analysis artifacts saved in: {out_dir}")


if __name__ == "__main__":
    main()
