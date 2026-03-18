#!/bin/bash
#SBATCH --job-name=gap_eval
#SBATCH --output=/home/kimhj/GapRAG/logs/gaprag_%j.out
#SBATCH --error=/home/kimhj/GapRAG/logs/gaprag_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=999:00:00
#SBATCH --nodelist=server2

set -euo pipefail
trap 'echo "[ERROR] line=${LINENO} cmd=${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

find_repo_root() {
  local start_dir="$1"
  local dir="$start_dir"
  local remaining=8
  while ((remaining > 0)); do
    if [[ -f "$dir/pyproject.toml" && -d "$dir/gaprag" ]]; then
      echo "$dir"
      return 0
    fi
    local parent
    parent="$(cd -- "$dir/.." && pwd)"
    [[ "$parent" == "$dir" ]] && break
    dir="$parent"
    remaining=$((remaining - 1))
  done
  return 1
}

# NOTE: Slurm can execute from spool paths; prefer submit dir first.
ROOT_CANDIDATE="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
ROOT_DIR="$(find_repo_root "$ROOT_CANDIDATE")" || {
  echo "Could not locate repo root from '$ROOT_CANDIDATE' (need pyproject.toml + gaprag/)."
  exit 1
}
cd "$ROOT_DIR"

mkdir -p logs outputs outputs/runs outputs/figures outputs/tables

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
else
  echo "Warning: .venv/bin/activate not found, using system Python environment."
fi

# Hugging Face token auto-load (if not already exported)
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.secrets/hf_token}"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(tr -d '\r\n' < "$HF_TOKEN_FILE")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  export HF_HUB_ENABLE_HF_TRANSFER=1
fi

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python not found on PATH (after optional .venv activation)."
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch, transformers, datasets, sentence_transformers, faiss" >/dev/null 2>&1; then
  echo "Python env is missing one or more required packages (torch/transformers/datasets/sentence-transformers/faiss)."
  echo "Current python: $PYTHON_BIN"
  echo "Fix:"
  echo "  cd $ROOT_DIR"
  echo "  uv sync --extra dev"
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/base.yaml}"
MODE="${MODE:-gap_memory_ema}"
RUN_NAME="${RUN_NAME:-}"

RUN_BUILD_INDEX="${RUN_BUILD_INDEX:-auto}"
RUN_EVAL_STATELESS="${RUN_EVAL_STATELESS:-true}"
RUN_EVAL_CONTINUAL="${RUN_EVAL_CONTINUAL:-true}"
RUN_ABLATION="${RUN_ABLATION:-false}"
ABLATION_CONFIG="${ABLATION_CONFIG:-configs/ablation_gap_defs.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

is_true() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

resolve_index_path() {
  "$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
cfg_path = Path("$CONFIG_PATH")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
print(cfg.get("retriever", {}).get("index_path", "data/indices/demo.faiss"))
PY
}

should_build_index() {
  local mode
  mode="$(echo "$RUN_BUILD_INDEX" | tr '[:upper:]' '[:lower:]')"
  if [[ "$mode" == "auto" ]]; then
    local index_path
    index_path="$(resolve_index_path)"
    [[ ! -f "$index_path" ]]
    return
  fi
  is_true "$RUN_BUILD_INDEX"
}

echo "=========================================="
echo "GapRAG batch run"
echo "ROOT_DIR=$ROOT_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "MODE=$MODE"
echo "RUN_BUILD_INDEX=$RUN_BUILD_INDEX"
echo "RUN_EVAL_STATELESS=$RUN_EVAL_STATELESS"
echo "RUN_EVAL_CONTINUAL=$RUN_EVAL_CONTINUAL"
echo "RUN_ABLATION=$RUN_ABLATION"
echo "HF_TOKEN_LOADED=$([[ -n \"${HF_TOKEN:-}\" ]] && echo true || echo false)"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "=========================================="

if should_build_index; then
  echo "[RUN] build_index"
  "$PYTHON_BIN" scripts/build_index.py --config "$CONFIG_PATH"
fi

COMMON_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$RUN_NAME" ]]; then
  COMMON_ARGS+=(--run-name "$RUN_NAME")
fi

if is_true "$RUN_EVAL_STATELESS"; then
  echo "[RUN] eval_stateless mode=$MODE"
  "$PYTHON_BIN" scripts/run_eval.py "${COMMON_ARGS[@]}" --mode "$MODE" --stateless
fi

if is_true "$RUN_EVAL_CONTINUAL"; then
  echo "[RUN] eval_continual mode=$MODE"
  "$PYTHON_BIN" scripts/run_continual_eval.py "${COMMON_ARGS[@]}" --mode "$MODE"
fi

if is_true "$RUN_ABLATION"; then
  if [[ ! -f "$ABLATION_CONFIG" ]]; then
    echo "Ablation config file not found: $ABLATION_CONFIG"
    exit 1
  fi
  ABLATION_ARGS=(--base-config "$CONFIG_PATH" --ablation-config "$ABLATION_CONFIG")
  if [[ -n "$RUN_NAME" ]]; then
    ABLATION_ARGS+=(--run-name "$RUN_NAME")
  fi
  echo "[RUN] ablation config=$ABLATION_CONFIG"
  "$PYTHON_BIN" scripts/run_ablation.py "${ABLATION_ARGS[@]}"
fi

echo "Done."
