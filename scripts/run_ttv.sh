#!/bin/bash
#SBATCH --job-name=gapverify_tt_v
#SBATCH --output=/home/kimhj/GapVerify/logs/gapverify_top_%j.out
#SBATCH --error=/home/kimhj/GapVerify/logs/gapverify_top_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=999:00:00
#SBATCH --nodelist=server2

set -euo pipefail

find_repo_root() {
  local start_dir="$1"
  local dir="$start_dir"
  local remaining=8
  while ((remaining > 0)); do
    if [[ -f "$dir/pyproject.toml" && -d "$dir/gapverify" && -f "$dir/scripts/run_experiments.sh" ]]; then
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

ROOT_CANDIDATE="${SLURM_SUBMIT_DIR:-$(pwd)}"
ROOT_DIR="$(find_repo_root "$ROOT_CANDIDATE")" || {
  echo "Could not locate repo root from '$ROOT_CANDIDATE'" >&2
  exit 1
}

# Top-tier-oriented preset within current GapVerify capabilities.
# What it does:
# - full validation/dev splits by default (prep limits = 0)
# - stronger retriever config (BGE-large)
# - stronger generator config (Qwen2.5-7B-Instruct)
# - larger top-k and query-aware evidence aggregation
# What it still does NOT add:
# - BM25 / hybrid retrieval
# - reranking / cross-encoder re-ranking
# - benchmark-specific classifier heads
# - official benchmark evaluators
# If 7B does not fit on the allocated GPU, override CONFIG_VARIANT=standard.

export EXPERIMENT_PRESET="${EXPERIMENT_PRESET:-verification_core_toptier}"
exec bash "$ROOT_DIR/scripts/run_experiments.sh"
