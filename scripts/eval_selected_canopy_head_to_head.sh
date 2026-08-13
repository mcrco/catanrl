#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "usage: $0 SOURCE_EXPERIMENT SELECTED_CHECKPOINT_FILE CANOPY_BINARY CANOPY_CHECKPOINT OUTPUT [SEED]" >&2
  exit 2
fi

source_experiment=$1
selected_file=$2
canopy_binary=$3
canopy_checkpoint=$4
output=$5
games_per_seat=$6
seed=${7:-52043}

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
  uv run python scripts/eval_canopy_head_to_head.py \
  --experiment "$source_experiment" \
  --which "$selector" \
  --canopy-binary "$canopy_binary" \
  --canopy-checkpoint "$canopy_checkpoint" \
  --simulations 1600 \
  --games-per-seat "$games_per_seat" \
  --num-workers 8 \
  --inference-batch-size 64 \
  --inference-wait-ms 2 \
  --canopy-batch-size 8 \
  --canopy-wait-ms 5 \
  --seed "$seed" \
  --max-actions 2000 \
  --device cuda \
  --output "$output"
