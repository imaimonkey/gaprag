#!/bin/bash
#SBATCH --job-name=gapverify_std
#SBATCH --output=/home/kimhj/GapVerify/logs/gapverify_std_%j.out
#SBATCH --error=/home/kimhj/GapVerify/logs/gapverify_std_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Field-standard-inspired preset within current GapVerify capabilities.
# This uses stronger local HF models and larger retrieval depth, but it does
# not add unavailable components such as BM25, hybrid retrieval, rerankers, or
# benchmark-specific classifier heads.

export EXPERIMENT_PRESET="${EXPERIMENT_PRESET:-verification_core_standard}"
exec bash "$ROOT_DIR/scripts/run_experiments.sh"
