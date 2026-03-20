# GapVerify

GapVerify is a modular research codebase for studying **latent discrepancy control in retrieval-grounded verification**.

The codebase originally explored persistent latent discrepancy memory for continual RAG. Current empirical results support a narrower and more defensible position:
- `gap_current` is the main method of interest.
- evidence-grounded **verification** is the main task family.
- persistent memory (`gap_memory_ema`, `gap_memory_keyed`) is currently a diagnostic/negative-result branch, not the core claim.

## Current Research Position

Current supported empirical takeaways from this repo are:
- `FEVER`: latent discrepancy injection can help label-style verification.
- `NQ`, `HotpotQA`: the same injection does **not** transfer cleanly to free-form QA.
- `continual_qa`: persistent memory does not currently yield positive continual gain.

So this repository should be read as:
- **main question**: can model-evidence latent discrepancy act as a training-free control signal?
- **main task**: retrieval-grounded verification / fact checking.
- **secondary question**: where does this signal fail to transfer?

See also:
- [Research Redefinition](/home/kimhj/GapVerify/RESEARCH_REDEFINITION.md)
- [Benchmark Priority](/home/kimhj/GapVerify/BENCHMARK_PRIORITY.md)
- [Restructure Plan](/home/kimhj/GapVerify/RESTRUCTURE_PLAN.md)
- [Experiment Manual](/home/kimhj/GapVerify/EXPERIMENT_MANUAL.md)

## Implemented Modes

- `vanilla_lm`
- `standard_rag`
- `gap_current`
- `gap_memory_ema`
- `gap_memory_keyed`

Interpretation:
- `standard_rag`: baseline
- `gap_current`: primary experimental method
- `gap_memory_*`: secondary diagnostic methods

## Supported Benchmarks in This Repo

Implemented now:
- `fever`
- `nq`
- `hotpotqa`
- `continual_qa`

Recommended roles:
- `fever`: main verification benchmark in the current codebase
- `nq`, `hotpotqa`: transfer-boundary / negative-transfer analysis
- `continual_qa`: memory diagnostic benchmark

Not yet integrated, but highest-priority next verification benchmarks:
- `AVeriTeC`
- `HoVer`
- `FEVEROUS`
- optional: `SciFact`, `Climate-FEVER`

## Core Idea

For a given query/claim and retrieved evidence, GapVerify estimates a latent discrepancy between:
- the model's query-conditioned hidden state, and
- the evidence-aligned hidden representation.

That discrepancy is then injected back into inference as a training-free control signal.

The currently supported, evidence-backed claim is:
- this signal may help **verification-style label decisions**,
- but does not currently improve free-form QA generation in a stable way.

## Project Structure

```text
gapverify/
  README.md
  EXPERIMENT_MANUAL.md
  TODO.md
  RESEARCH_REDEFINITION.md
  BENCHMARK_PRIORITY.md
  RESTRUCTURE_PLAN.md
  pyproject.toml
  requirements.txt

  configs/
    base.yaml
    nq.yaml
    hotpotqa.yaml
    fever.yaml
    continual_qa.yaml
    rag.yaml
    gapverify_current.yaml
    gapverify_memory.yaml
    ablation_gap_defs.yaml
    ablation_memory.yaml
    ablation_injection.yaml
    smoke_tiny.yaml

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

  gapverify/
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
cd /home/kimhj/GapVerify
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e .
```

### Option B: uv

```bash
cd /home/kimhj/GapVerify
uv venv
uv sync
```

## Data Preparation

Current local benchmark adapters prepare:
- `nq`
- `hotpotqa`
- `fever`
- `continual_qa`

```bash
uv run python scripts/prepare_benchmark_data.py --benchmark nq --nq-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark hotpotqa --hotpotqa-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark fever --fever-limit 500
uv run python scripts/prepare_benchmark_data.py --benchmark continual_qa --nq-limit 500 --hotpotqa-limit 500 --fever-limit 500
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
uv run python scripts/build_index.py --config configs/base.yaml
```

## Recommended Experiment Menu

### 1. Main verification run

Current best-supported main run in this repo:

```bash
BENCHMARK_PROFILE=fever \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME=run_fever_verification \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_STATELESS=true \
RUN_EVAL_CONTINUAL=auto \
sbatch scripts/run_experiments.sh
```

### 2. Transfer-boundary run

```bash
BENCHMARK_SUITE=nq,hotpotqa \
MODE_SUITE=standard_rag,gap_current \
RUN_NAME_PREFIX=run_transfer_boundary \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
RUN_EVAL_CONTINUAL=auto \
sbatch scripts/run_experiments.sh
```

### 3. Memory diagnostic run

```bash
BENCHMARK_PROFILE=continual_qa \
MODE_SUITE=standard_rag,gap_current,gap_memory_keyed,gap_memory_ema \
RUN_NAME=run_continual_suite \
PREP_BENCHMARK_DATA=auto \
RUN_BUILD_INDEX=auto \
sbatch scripts/run_experiments.sh
```

Interpretation:
- use `fever` as the main positive benchmark
- use `nq` / `hotpotqa` to show task-boundary failure cases
- use `continual_qa` to keep persistent memory claims honest

## Run Baselines / Methods Directly

### Stateless

```bash
python scripts/run_eval.py --config configs/fever.yaml --mode standard_rag --stateless --run-name fever_rag
python scripts/run_eval.py --config configs/fever.yaml --mode gap_current --stateless --run-name fever_gap_current
```

### Continual diagnostic

```bash
python scripts/run_continual_eval.py --config configs/continual_qa.yaml --mode gap_memory_ema --run-name continual_gap_memory_ema
```

## Analyze Results

```bash
python scripts/analyze_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

Current interpretation guidance:
- `compare_summary.json`: use for continual diagnostic comparisons
- `metrics_summary.json`: use for stateless benchmark comparisons
- `changed_raw_count`, `changed_prediction_count`: use to separate "output changed" from "accuracy improved"

## Slurm Batch Execution

Primary script:

```bash
sbatch scripts/run_experiments.sh
```

Important env options:
- `BENCHMARK_PROFILE` (`demo|nq|hotpotqa|fever|continual_qa`)
- `BENCHMARK_SUITE` (comma-separated profile sweep)
- `MODE` (single mode)
- `MODE_SUITE` (comma-separated mode sweep)
- `RUN_NAME`, `RUN_NAME_PREFIX`
- `PREP_BENCHMARK_DATA`
- `RUN_BUILD_INDEX`
- `RUN_EVAL_STATELESS`
- `RUN_EVAL_CONTINUAL`
- `RUN_ABLATION`

Meaning of `RUN_EVAL_CONTINUAL=auto`:
- runs continual comparison only for `demo` and `continual_qa`
- prevents accidental over-claiming on benchmarks that are not meaningful continual tests

## What This Repo Currently Does Not Claim

This codebase does **not** currently support the following strong claims:
- persistent memory improves future queries in a robust continual setting
- discrepancy injection is a general-purpose QA improvement method
- current benchmark support already covers the full modern fact-checking suite

Those are open research questions or future work items, not established conclusions.
