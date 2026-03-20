# GapRAG

GapRAG (Persistent Latent Discrepancy Modeling for Training-Free Continual RAG) is a modular research codebase for continual retrieval-augmented generation without updating model weights.

## Core Idea

Standard RAG uses retrieved evidence independently per query. GapRAG estimates a latent discrepancy (gap) between:
- the model's query-conditioned hidden state, and
- the retrieved evidence representation,

then accumulates that discrepancy into persistent memory and injects it back into generation at inference time.

## What Makes It Different from Standard RAG

- Standard RAG: use retrieved docs for the current query only.
- GapRAG: maintain cross-query persistent latent discrepancy memory.
- Training-free: no optimizer, no parameter updates.
- Memory stores only latent gap vectors, not raw documents.

## Implemented Modes

- `vanilla_lm`
- `standard_rag`
- `gap_current` (current gap only, no memory accumulation)
- `gap_memory_ema` (persistent EMA memory)
- `gap_memory_keyed` (query-keyed gap memory)

## Project Structure

```text
gaprag/
  README.md
  TODO.md
  pyproject.toml
  requirements.txt

  configs/
    base.yaml
    nq.yaml
    hotpotqa.yaml
    fever.yaml
    continual_qa.yaml
    rag.yaml
    gaprag_no_memory.yaml
    gaprag_memory.yaml
    ablation_gap_defs.yaml
    ablation_memory.yaml
    ablation_injection.yaml

  data/
    raw/
    processed/
    indices/

  scripts/
    prepare_benchmark_data.py
    build_index.py
    run_eval.py
    run_continual_eval.py
    run_ablation.py
    analyze_results.py
    run_experiments.sh

  gaprag/
    retriever.py
    generator.py
    hidden_extractor.py
    doc_encoder.py
    gap_estimator.py
    gap_memory.py
    gap_injector.py
    pipeline.py
    datasets.py
    metrics.py
    utils.py
    logging_utils.py

  outputs/
    runs/
    tables/
    figures/
```

## Installation

### Option A: pip

```bash
cd /home/kimhj/GapRAG
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option B: uv

```bash
cd /home/kimhj/GapRAG
uv venv
source .venv/bin/activate
uv sync
```

## Data Preparation

This repo includes small demo files:
- `data/processed/demo_corpus.jsonl`
- `data/processed/demo_qa.jsonl`

Research benchmark prep (NQ / HotpotQA / FEVER / Continual QA):

```bash
python scripts/prepare_benchmark_data.py --benchmark nq --nq-limit 500
python scripts/prepare_benchmark_data.py --benchmark hotpotqa --hotpotqa-limit 500
python scripts/prepare_benchmark_data.py --benchmark fever --fever-limit 500
python scripts/prepare_benchmark_data.py --benchmark continual_qa --nq-limit 500 --hotpotqa-limit 500 --fever-limit 500
```

Expected QA schema:

```python
{
  "id": "...",
  "question": "...",
  "answers": ["..."],
  "context": ["optional_gold_doc_ids"],
  "metadata": {...},
  "session_id": "..."
}
```

## Build Retrieval Index

```bash
python scripts/build_index.py --config configs/base.yaml
```

## Run Baselines / GapRAG

### Single-pass eval (stateless)

```bash
python scripts/run_eval.py --config configs/rag.yaml --mode standard_rag --stateless
python scripts/run_eval.py --config configs/gaprag_no_memory.yaml --mode gap_current --stateless
python scripts/run_eval.py --config configs/gaprag_memory.yaml --mode gap_memory_ema --stateless
```

### Continual eval (stateless vs continual)

```bash
python scripts/run_continual_eval.py --config configs/gaprag_memory.yaml --mode gap_memory_ema
```

Outputs include:
- `predictions_stateless.jsonl`
- `predictions_continual.jsonl`
- `step_metrics_stateless.csv`
- `step_metrics_continual.csv`
- `compare_summary.json`

## Run Ablations

```bash
python scripts/run_ablation.py --base-config configs/base.yaml --ablation-config configs/ablation_gap_defs.yaml
python scripts/run_ablation.py --base-config configs/base.yaml --ablation-config configs/ablation_memory.yaml
python scripts/run_ablation.py --base-config configs/base.yaml --ablation-config configs/ablation_injection.yaml
```

## Analyze Results

```bash
python scripts/analyze_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

Generates:
- overall EM bar charts
- continual gain charts
- continual accuracy curves
- gap norm vs correctness
- memory norm vs correctness
- retrieval hit vs correctness

## Slurm Batch Execution

`run_experiments.sh` supports index build / eval / continual / ablation toggles, and now also supports `MODE_SUITE` for one-allocation method sweeps.

```bash
sbatch scripts/run_experiments.sh
```

Main env options:
- `BENCHMARK_PROFILE` (`demo|nq|hotpotqa|fever|continual_qa`, default: `demo`)
- `CONFIG_PATH` (default: `configs/base.yaml`)
- `MODE` (default: `gap_memory_ema`)
- `MODE_SUITE` (optional comma-separated sweep, example: `standard_rag,gap_current,gap_memory_keyed,gap_memory_ema`)
- `RUN_NAME` (optional)
- `PREP_BENCHMARK_DATA` (`auto/true/false`, default: `demo=auto`, `non-demo=true`)
- `RUN_BUILD_INDEX` (`auto/true/false`, default: `demo=auto`, `non-demo=true`)
- `RUN_EVAL_STATELESS` (`true/false`)
- `RUN_EVAL_CONTINUAL` (`auto/true/false`, default: `auto`)
- `RUN_ABLATION` (`true/false`)
- `ABLATION_CONFIG` (default: `configs/ablation_gap_defs.yaml`)

Recommended benchmark roles:
- `nq`, `hotpotqa`, `fever`: stateless/general QA or verification evaluation
- `continual_qa`: persistent-memory evaluation
- with `RUN_EVAL_CONTINUAL=auto`, continual comparison runs only for `demo` and `continual_qa`

Example:

```bash
BENCHMARK_PROFILE=continual_qa \
MODE=gap_memory_ema \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_STATELESS=true \
RUN_EVAL_CONTINUAL=true \
RUN_ABLATION=false \
sbatch scripts/run_experiments.sh
```

Recommended method sweep on the continual benchmark:

```bash
BENCHMARK_PROFILE=continual_qa \
MODE_SUITE=standard_rag,gap_current,gap_memory_keyed,gap_memory_ema \
RUN_NAME=run_continual_suite \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
sbatch scripts/run_experiments.sh
```

## Implemented Gap / Memory / Injection Variants

### Gap definitions
- `diff`: `g_t = h_D - h_q`
- `proj_diff`: fixed alignment + difference
- `confidence_weighted`: confidence-scaled gap

### Memory rules
- `none` (current gap only)
- `ema`
- `keyed`
- `bounded` (optional bank)

### Injection rules
- `residual_hidden` (MVP core)
- `prefix_bias`
- `attention_bias` (declared, not implemented)

## Notes

- This repository is designed for reproducibility first.
- It intentionally avoids gradient-based test-time training.
- Hidden extraction, gap estimation, memory update, and injection are separated modules for clean ablations.
- Full step-by-step runbook: `EXPERIMENT_MANUAL.md`.
