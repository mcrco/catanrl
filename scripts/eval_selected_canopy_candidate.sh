#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_EXPERIMENT SELECTED_CHECKPOINT_FILE OUTPUT_DIR [SEED]" >&2
  exit 2
fi

source_experiment=$1
selected_file=$2
output_dir=$3
seed=${4:-12043}

if [[ ! -s "$selected_file" ]]; then
  echo "selected checkpoint file is missing or empty: $selected_file" >&2
  exit 1
fi
IFS= read -r selector < "$selected_file"
if ! [[ "$selector" =~ ^[0-9]+$ ]]; then
  echo "selected checkpoint must be a non-negative integer, got: $selector" >&2
  exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
exec env -u VIRTUAL_ENV \
  UV_CACHE_DIR=/tmp/catanrl-uv-cache \
  uv run python scripts/eval_native_mcts_budget_sweep.py \
  --experiment "$source_experiment" \
  --which "$selector" \
  --budgets 1600 \
  --skip-probes \
  --games-per-seat 100 \
  --game-opponent value \
  --authoritative-engine catanatron \
  --num-workers 8 \
  --games-per-worker 1 \
  --inference-batch-size 64 \
  --inference-wait-ms 2 \
  --c-puct 2.5 \
  --search-selection completed-q \
  --c-visit 50 \
  --c-scale 1 \
  --root-dirichlet-fraction 0 \
  --value-scale 1 \
  --canonical-pruning \
  --tree-reuse \
  --seed "$seed" \
  --max-actions 2000 \
  --device cuda \
  --output-dir "$output_dir"
