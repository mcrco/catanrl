#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 EXPERIMENT OUTPUT SELECTED_OUTPUT SEED" >&2
  exit 2
fi

experiment=$1
output=$2
selected_output=$3
seed=$4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
f_selected_output="${selected_output}.f-screen"
direct_output="${output%.json}-direct-canopy.json"
canopy_binary=${CATANRL_CANOPY_BINARY:-/tmp/cullback-canopy-nexus-v3/target/release/examples/catan}
canopy_checkpoint=${CATANRL_CANOPY_CHECKPOINT:-/home/marco/dev/catanrl-c-engine/experiments/canopy-release-reference/checkpoint/model_iter_315.mpk}
direct_seed=$((seed + 10000))

env -u VIRTUAL_ENV \
  UV_CACHE_DIR=/tmp/catanrl-uv-cache \
  uv run python scripts/screen_search_checkpoints.py \
  --experiment "$experiment" \
  --selectors 0 1 2 3 4 5 6 7 8 9 10 \
  --budget 64 \
  --games-per-seat 24 \
  --top-k 3 \
  --num-workers 8 \
  --games-per-worker 1 \
  --authoritative-engine catanatron \
  --inference-batch-size 64 \
  --inference-wait-ms 2 \
  --seed "$seed" \
  --max-actions 2000 \
  --device cuda \
  --output "$output" \
  --selected-output "$f_selected_output"

exec env -u VIRTUAL_ENV \
  UV_CACHE_DIR=/tmp/catanrl-uv-cache \
  uv run python scripts/screen_canopy_shortlist.py \
  --experiment "$experiment" \
  --search-screen "$output" \
  --top-k 3 \
  --canopy-binary "$canopy_binary" \
  --canopy-checkpoint "$canopy_checkpoint" \
  --simulations 64 \
  --games-per-seat 24 \
  --num-workers 8 \
  --inference-batch-size 64 \
  --inference-wait-ms 2 \
  --canopy-batch-size 8 \
  --canopy-wait-ms 5 \
  --seed "$direct_seed" \
  --max-actions 2000 \
  --device cuda \
  --output "$direct_output" \
  --selected-output "$selected_output"
