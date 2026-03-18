#!/bin/bash
#SBATCH --job-name=gaprag_eval
#SBATCH --output=/home/kimhj/GapRAG/logs/gaprag_%j.out
#SBATCH --error=/home/kimhj/GapRAG/logs/gaprag_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs/runs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "python not found"
  exit 1
fi

CONFIG_PATH="${CONFIG_PATH:-configs/base.yaml}"
MODE="${MODE:-gap_memory_ema}"
RUN_NAME="${RUN_NAME:-}"

RUN_BUILD_INDEX="${RUN_BUILD_INDEX:-false}"
RUN_EVAL_STATELESS="${RUN_EVAL_STATELESS:-true}"
RUN_EVAL_CONTINUAL="${RUN_EVAL_CONTINUAL:-true}"
RUN_ABLATION="${RUN_ABLATION:-false}"
ABLATION_CONFIG="${ABLATION_CONFIG:-configs/ablation_gap_defs.yaml}"

is_true() {
  local value
  value="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
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
echo "=========================================="

if is_true "$RUN_BUILD_INDEX"; then
  "$PYTHON_BIN" scripts/build_index.py --config "$CONFIG_PATH"
fi

COMMON_ARGS=(--config "$CONFIG_PATH")
if [[ -n "$RUN_NAME" ]]; then
  COMMON_ARGS+=(--run-name "$RUN_NAME")
fi

if is_true "$RUN_EVAL_STATELESS"; then
  "$PYTHON_BIN" scripts/run_eval.py "${COMMON_ARGS[@]}" --mode "$MODE" --stateless
fi

if is_true "$RUN_EVAL_CONTINUAL"; then
  "$PYTHON_BIN" scripts/run_continual_eval.py "${COMMON_ARGS[@]}" --mode "$MODE"
fi

if is_true "$RUN_ABLATION"; then
  ABLATION_ARGS=(--base-config "$CONFIG_PATH" --ablation-config "$ABLATION_CONFIG")
  if [[ -n "$RUN_NAME" ]]; then
    ABLATION_ARGS+=(--run-name "$RUN_NAME")
  fi
  "$PYTHON_BIN" scripts/run_ablation.py "${ABLATION_ARGS[@]}"
fi

echo "Done."
