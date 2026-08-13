#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 PARITY_RESULT SOURCE_EXPERIMENT SELECTED_CHECKPOINT_FILE EXPERIMENT_NAME [SEED]" >&2
  exit 2
fi

parity_result=$1
source_experiment=$2
selected_file=$3
experiment_name=$4
seed=${5:-53}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

if [[ ! -s "$parity_result" ]]; then
  echo "parity result is missing or empty: $parity_result" >&2
  exit 1
fi
passes=$(env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/catanrl-uv-cache \
  uv run python -c \
  'import json, sys; print(json.load(open(sys.argv[1]))["noninferiority"]["passes"])' \
  "$parity_result")
case "$passes" in
  True)
    echo "Canopy noninferiority gate passed; no continuation training needed."
    exit 0
    ;;
  False) ;;
  *)
    echo "invalid noninferiority gate value: $passes" >&2
    exit 1
    ;;
esac

if [[ ! -s "$selected_file" ]]; then
  echo "selected checkpoint file is missing or empty: $selected_file" >&2
  exit 1
fi
IFS= read -r selector < "$selected_file"
if ! [[ "$selector" =~ ^[0-9]+$ ]]; then
  echo "selected checkpoint must be a non-negative integer, got: $selector" >&2
  exit 1
fi

echo "Canopy gate failed; continuing checkpoint $selector for 10 AlphaZero iterations."
exec env \
  CATANRL_PARITY_ITERATIONS=10 \
  CATANRL_PARITY_WANDB=0 \
  "$repo_root/scripts/continue_canopy_alphazero.sh" \
  "$source_experiment" \
  "$selector" \
  "$experiment_name" \
  "$seed"
