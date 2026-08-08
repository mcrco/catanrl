#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 PREFIX [retain|reset] [SEED]" >&2
  exit 2
fi

prefix=$1
value_init=${2:-reset}
seed=${3:-43}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

case "$value_init" in
  retain|reset) ;;
  *)
    echo "value initialization must be 'retain' or 'reset'" >&2
    exit 2
    ;;
esac

dagger_experiment="${prefix}-dagger10"
alphazero_experiment="${prefix}-alphazero-${value_init}"

if [[ -e "experiments/${dagger_experiment}/metadata.json" ]]; then
  echo "experiment already exists: ${dagger_experiment}" >&2
  exit 1
fi
if [[ -e "experiments/${alphazero_experiment}/metadata.json" ]]; then
  echo "experiment already exists: ${alphazero_experiment}" >&2
  exit 1
fi

mkdir -p "experiments/${dagger_experiment}"
scripts/train_canopy_dagger10.sh "$dagger_experiment" "$seed" \
  2>&1 | tee "experiments/${dagger_experiment}/run.log"

mkdir -p "experiments/${alphazero_experiment}"
scripts/train_canopy_alphazero.sh \
  "$dagger_experiment" \
  "$alphazero_experiment" \
  "$value_init" \
  "$seed" \
  2>&1 | tee "experiments/${alphazero_experiment}/run.log"

if [[ "${CATANRL_PARITY_RUN_EVAL:-1}" == "1" ]]; then
  screen_root="experiments/eval-${alphazero_experiment}-checkpoint-screen"
  selected_checkpoint_path="${screen_root}/selected-checkpoint.txt"
  mkdir -p "$screen_root"
  env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 uv run python scripts/screen_search_checkpoints.py \
    --experiment "$alphazero_experiment" \
    --budget "${CATANRL_PARITY_SCREEN_BUDGET:-64}" \
    --games-per-seat "${CATANRL_PARITY_SCREEN_GAMES_PER_SEAT:-24}" \
    --top-k "${CATANRL_PARITY_SCREEN_TOP_K:-3}" \
    --num-workers "${CATANRL_PARITY_EVAL_WORKERS:-16}" \
    --max-actions 2000 \
    --output "${screen_root}/results.json" \
    --selected-output "$selected_checkpoint_path" \
    2>&1 | tee "${screen_root}/run.log"
  selected_checkpoint=$(<"$selected_checkpoint_path")
  if [[ -z "$selected_checkpoint" ]]; then
    echo "checkpoint screen did not select a checkpoint" >&2
    exit 1
  fi
  eval_root="experiments/eval-${alphazero_experiment}-${selected_checkpoint}-s12043"
  mkdir -p "$eval_root"
  scripts/eval_canopy_parity.sh "$alphazero_experiment" "$selected_checkpoint" 12043 \
    2>&1 | tee "$eval_root/run.log"
fi
