# GapRAG Experiment Manual

## 1. Environment Setup (uv)

```bash
cd /home/kimhj/GapRAG
uv venv
source .venv/bin/activate
uv add numpy pandas pyyaml torch transformers accelerate datasets sentence-transformers faiss-cpu scikit-learn matplotlib seaborn tqdm
uv add --dev pytest ruff
uv lock
uv sync --extra dev
```

Notes:
- `uv init` is already satisfied by existing `pyproject.toml`.
- `uv.lock` is the source of reproducible dependency resolution.

## 2. Quick Validation (Smoke)

### 2.1 Build index

```bash
uv run python scripts/build_index.py --config configs/smoke_tiny.yaml
```

### 2.2 Single-pass eval

```bash
uv run python scripts/run_eval.py --config configs/smoke_tiny.yaml --mode vanilla_lm --stateless --run-name smoke_vanilla
uv run python scripts/run_eval.py --config configs/smoke_tiny.yaml --mode standard_rag --stateless --run-name smoke_rag
uv run python scripts/run_eval.py --config configs/smoke_tiny.yaml --mode gap_current --stateless --run-name smoke_gap_current
uv run python scripts/run_eval.py --config configs/smoke_tiny.yaml --mode gap_memory_ema --stateless --run-name smoke_gap_memory
uv run python scripts/run_eval.py --config configs/smoke_tiny.yaml --mode gap_memory_keyed --stateless --run-name smoke_gap_keyed
```

### 2.3 Continual eval (stateless vs continual)

```bash
uv run python scripts/run_continual_eval.py --config configs/smoke_tiny.yaml --mode gap_memory_ema --run-name smoke_continual
```

### 2.4 Ablation

```bash
uv run python scripts/run_ablation.py --base-config configs/smoke_tiny.yaml --ablation-config configs/ablation_gap_defs.yaml --run-name smoke_ablation_gap
uv run python scripts/run_ablation.py --base-config configs/smoke_tiny.yaml --ablation-config configs/ablation_memory.yaml --run-name smoke_ablation_memory
uv run python scripts/run_ablation.py --base-config configs/smoke_tiny.yaml --ablation-config configs/ablation_injection.yaml --run-name smoke_ablation_injection
```

### 2.5 Analysis plot generation

```bash
uv run python scripts/analyze_results.py --runs-dir outputs/runs --out-dir outputs/figures
```

## 3. SLURM Batch Run

Primary script: `scripts/run_experiments.sh`

```bash
sbatch scripts/run_experiments.sh
```

Current header is aligned to your existing environment:
- `#SBATCH --nodelist=server2`
- spool-safe repo root resolution via `SLURM_SUBMIT_DIR`
- preflight package checks (`torch/transformers/datasets/sentence-transformers/faiss`)

If your cluster changes, edit only these header lines in `scripts/run_experiments.sh`.

### 3.1 Runtime toggles

- `CONFIG_PATH` (default: `configs/base.yaml`)
- `MODE` (default: `gap_memory_ema`)
- `RUN_NAME` (optional)
- `RUN_BUILD_INDEX` (`auto/true/false`, default: `auto`)
- `RUN_EVAL_STATELESS` (`true/false`)
- `RUN_EVAL_CONTINUAL` (`true/false`)
- `RUN_ABLATION` (`true/false`)
- `ABLATION_CONFIG` (default: `configs/ablation_gap_defs.yaml`)

### 3.2 Example

```bash
CONFIG_PATH=configs/gaprag_memory.yaml \
MODE=gap_memory_ema \
RUN_NAME=exp_gap_memory \
RUN_BUILD_INDEX=auto \
RUN_EVAL_STATELESS=true \
RUN_EVAL_CONTINUAL=true \
RUN_ABLATION=false \
sbatch scripts/run_experiments.sh
```

## 4. Recommended Experiment Menu

### Menu A (Baseline sanity)
- `vanilla_lm`
- `standard_rag`

### Menu B (Gap effect)
- `gap_current`
- `gap_memory_ema`
- `gap_memory_keyed`

### Menu C (Ablations)
- Gap definitions: `configs/ablation_gap_defs.yaml`
- Memory rules: `configs/ablation_memory.yaml`
- Injection rules: `configs/ablation_injection.yaml`

### Menu D (Full report)
1. Run Menu A/B
2. Run Menu C
3. Execute `analyze_results.py`
4. Collect from:
   - `outputs/runs/*/metrics_summary.json`
   - `outputs/runs/*/compare_summary.json`
   - `outputs/figures/*.png`

## 5. Artifact Paths

- Per-run outputs: `outputs/runs/<RUN_NAME>/`
- Analysis tables: `outputs/runs/analysis_compare_table.csv`, `outputs/runs/analysis_step_table.csv`
- Figures: `outputs/figures/*.png`

## 6. Troubleshooting

- HF warning about unauthenticated requests is non-fatal. For faster downloads set `HF_TOKEN`.
- If model download is slow, run smoke config first (`configs/smoke_tiny.yaml`).
- For reproducibility, keep `uv.lock` committed and run with `uv sync` before experiments.
