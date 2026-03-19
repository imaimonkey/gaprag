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

# Unified HF cache paths across projects to avoid duplicate downloads.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}"

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

BENCHMARK_SUITE="${BENCHMARK_SUITE:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-run}"
if [[ -n "$BENCHMARK_SUITE" && -z "${_GAPRAG_SUITE_CHILD:-}" ]]; then
  IFS=',' read -r -a SUITE_PROFILES <<< "$BENCHMARK_SUITE"
  if [[ ${#SUITE_PROFILES[@]} -eq 0 ]]; then
    echo "BENCHMARK_SUITE is set but empty after parsing: '$BENCHMARK_SUITE'"
    exit 1
  fi
  echo "=========================================="
  echo "GapRAG suite mode (single Slurm allocation)"
  echo "BENCHMARK_SUITE=$BENCHMARK_SUITE"
  echo "RUN_NAME_PREFIX=$RUN_NAME_PREFIX"
  echo "=========================================="
  for raw_profile in "${SUITE_PROFILES[@]}"; do
    profile="$(echo "$raw_profile" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [[ -z "$profile" ]] && continue
    case "$profile" in
      hotpotqa) run_suffix="hotpot" ;;
      continual_qa) run_suffix="continual" ;;
      *) run_suffix="$profile" ;;
    esac
    run_name="${RUN_NAME_PREFIX}_${run_suffix}"
    echo "[SUITE] >>> profile=$profile run_name=$run_name"
    if ! env _GAPRAG_SUITE_CHILD=1 BENCHMARK_SUITE="" BENCHMARK_PROFILE="$profile" RUN_NAME="$run_name" bash "$0"; then
      echo "[SUITE] failed at profile=$profile"
      exit 1
    fi
    echo "[SUITE] <<< done profile=$profile"
  done
  echo "[SUITE] all profiles completed."
  exit 0
fi

BENCHMARK_PROFILE="${BENCHMARK_PROFILE:-demo}"
CONFIG_PATH="${CONFIG_PATH:-}"
if [[ -z "$CONFIG_PATH" ]]; then
  case "$BENCHMARK_PROFILE" in
    demo) CONFIG_PATH="configs/base.yaml" ;;
    nq) CONFIG_PATH="configs/nq.yaml" ;;
    hotpotqa) CONFIG_PATH="configs/hotpotqa.yaml" ;;
    fever) CONFIG_PATH="configs/fever.yaml" ;;
    continual_qa) CONFIG_PATH="configs/continual_qa.yaml" ;;
    *)
      echo "Unknown BENCHMARK_PROFILE='$BENCHMARK_PROFILE'"
      echo "Valid: demo, nq, hotpotqa, fever, continual_qa"
      exit 1
      ;;
  esac
fi
MODE="${MODE:-gap_memory_ema}"
RUN_NAME="${RUN_NAME:-}"

PREP_BENCHMARK_DATA="${PREP_BENCHMARK_DATA:-}"
if [[ -z "$PREP_BENCHMARK_DATA" ]]; then
  if [[ "$BENCHMARK_PROFILE" == "demo" ]]; then
    PREP_BENCHMARK_DATA="auto"
  else
    PREP_BENCHMARK_DATA="true"
  fi
fi
PREP_BENCHMARK="${PREP_BENCHMARK:-$BENCHMARK_PROFILE}"
NQ_PREP_LIMIT="${NQ_PREP_LIMIT:-500}"
HOTPOTQA_PREP_LIMIT="${HOTPOTQA_PREP_LIMIT:-500}"
FEVER_PREP_LIMIT="${FEVER_PREP_LIMIT:-500}"
NQ_PREP_SPLIT="${NQ_PREP_SPLIT:-validation}"
HOTPOTQA_PREP_SPLIT="${HOTPOTQA_PREP_SPLIT:-validation}"
FEVER_PREP_SPLIT="${FEVER_PREP_SPLIT:-validation}"

RUN_BUILD_INDEX="${RUN_BUILD_INDEX:-}"
if [[ -z "$RUN_BUILD_INDEX" ]]; then
  if [[ "$BENCHMARK_PROFILE" == "demo" ]]; then
    RUN_BUILD_INDEX="auto"
  else
    RUN_BUILD_INDEX="true"
  fi
fi
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

resolve_data_paths() {
  "$PYTHON_BIN" - <<PY
import yaml
from pathlib import Path
cfg_path = Path("$CONFIG_PATH")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
print(cfg.get("dataset", {}).get("path", ""))
print(cfg.get("corpus", {}).get("path", ""))
PY
}

should_prepare_data() {
  local mode
  mode="$(echo "$PREP_BENCHMARK_DATA" | tr '[:upper:]' '[:lower:]')"
  if [[ "$mode" == "auto" ]]; then
    if [[ "$BENCHMARK_PROFILE" == "demo" ]]; then
      return 1
    fi
    local -a _paths
    local dataset_path
    local corpus_path
    mapfile -t _paths < <(resolve_data_paths)
    dataset_path="${_paths[0]:-}"
    corpus_path="${_paths[1]:-}"
    if [[ -z "$dataset_path" || -z "$corpus_path" ]]; then
      return 0
    fi
    [[ ! -f "$dataset_path" || ! -f "$corpus_path" ]]
    return
  fi
  is_true "$PREP_BENCHMARK_DATA"
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
echo "BENCHMARK_PROFILE=$BENCHMARK_PROFILE"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "MODE=$MODE"
echo "PREP_BENCHMARK_DATA=$PREP_BENCHMARK_DATA (benchmark=$PREP_BENCHMARK)"
echo "RUN_BUILD_INDEX=$RUN_BUILD_INDEX"
echo "RUN_EVAL_STATELESS=$RUN_EVAL_STATELESS"
echo "RUN_EVAL_CONTINUAL=$RUN_EVAL_CONTINUAL"
echo "RUN_ABLATION=$RUN_ABLATION"
echo "HF_TOKEN_LOADED=$([[ -n \"${HF_TOKEN:-}\" ]] && echo true || echo false)"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "=========================================="

if should_prepare_data; then
  if [[ "$PREP_BENCHMARK" == "demo" ]]; then
    echo "PREP_BENCHMARK=demo is not valid for benchmark data generation."
    echo "Set PREP_BENCHMARK to one of: nq, hotpotqa, fever, continual_qa, all"
    exit 1
  fi
  echo "[RUN] prepare_benchmark_data benchmark=$PREP_BENCHMARK"
  "$PYTHON_BIN" scripts/prepare_benchmark_data.py \
    --benchmark "$PREP_BENCHMARK" \
    --nq-limit "$NQ_PREP_LIMIT" \
    --hotpotqa-limit "$HOTPOTQA_PREP_LIMIT" \
    --fever-limit "$FEVER_PREP_LIMIT" \
    --nq-split "$NQ_PREP_SPLIT" \
    --hotpotqa-split "$HOTPOTQA_PREP_SPLIT" \
    --fever-split "$FEVER_PREP_SPLIT"
fi

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
