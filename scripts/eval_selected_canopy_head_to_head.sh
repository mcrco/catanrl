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
smoke_simulations=${CATANRL_CANOPY_H2H_SMOKE_SIMULATIONS:-0}
selection_result=${CATANRL_CANOPY_SELECTION_RESULT:-}

if ! [[ "$smoke_simulations" =~ ^[0-9]+$ ]]; then
  echo "CATANRL_CANOPY_H2H_SMOKE_SIMULATIONS must be a non-negative integer" >&2
  exit 2
fi

if [[ ! -s "$selected_file" ]]; then
  echo "selected checkpoint file is missing or empty: $selected_file" >&2
  exit 1
fi
IFS= read -r selector < "$selected_file"
if ! [[ "$selector" =~ ^[0-9]+$ ]]; then
  echo "selected checkpoint must be a non-negative integer, got: $selector" >&2
  exit 1
fi
if [[ -n "$selection_result" ]]; then
  if [[ ! -s "$selection_result" ]]; then
    echo "direct Canopy selection result is missing or empty: $selection_result" >&2
    exit 1
  fi
  env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/catanrl-uv-cache \
    uv run python -c \
    'import json, pathlib, sys
p=json.load(open(sys.argv[1]))
s=pathlib.Path(sys.argv[2]).read_text().strip()
assert p.get("status") == "complete", "direct Canopy selection did not complete"
assert p.get("selection_opponent") == "released cullback/canopy nexus-v3"
assert str(p.get("selected")) == s, "selection JSON and checkpoint file disagree"
assert str(p["ranking"][0]["selector"]) == s, "selected checkpoint is not direct Canopy rank 1"' \
    "$selection_result" "$selected_file"
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
if (( smoke_simulations > 0 )); then
  smoke_output=${output%.json}.smoke.json
  env -u VIRTUAL_ENV \
    UV_CACHE_DIR=/tmp/catanrl-uv-cache \
    uv run python scripts/eval_canopy_head_to_head.py \
    --experiment "$source_experiment" \
    --which "$selector" \
    --canopy-binary "$canopy_binary" \
    --canopy-checkpoint "$canopy_checkpoint" \
    --simulations "$smoke_simulations" \
    --games-per-seat 1 \
    --num-workers 2 \
    --inference-batch-size 8 \
    --inference-wait-ms 2 \
    --canopy-batch-size 2 \
    --canopy-wait-ms 5 \
    --seed "$seed" \
    --max-actions 2000 \
    --device cuda \
    --output "$smoke_output"
fi

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
