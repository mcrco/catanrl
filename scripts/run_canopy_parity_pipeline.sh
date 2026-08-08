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
    --games-per-worker "${CATANRL_PARITY_EVAL_GAMES_PER_WORKER:-2}" \
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

  canopy_release_repo=${CATANRL_CANOPY_RELEASE_REPO:-}
  canopy_release_checkpoint=${CATANRL_CANOPY_RELEASE_CHECKPOINT:-}
  if [[ -n "$canopy_release_repo" || -n "$canopy_release_checkpoint" ]]; then
    if [[ -z "$canopy_release_repo" || -z "$canopy_release_checkpoint" ]]; then
      echo "Set both CATANRL_CANOPY_RELEASE_REPO and CATANRL_CANOPY_RELEASE_CHECKPOINT" >&2
      exit 2
    fi
    parity_budget=${CATANRL_PARITY_FULL_SIMULATIONS:-1600}
    reference_games=${CATANRL_CANOPY_REFERENCE_GAMES:-800}
    reference_root=${CATANRL_CANOPY_REFERENCE_OUTPUT_DIR:-experiments/canopy-release-reference}
    reference_result="$reference_root/canopy-nexus-v3-s${parity_budget}-vs-random.json"
    candidate_result="$eval_root/native-s${parity_budget}-vs-random/results.json"
    comparison_result="$eval_root/official-canopy-nexus-v3-noninferiority.json"

    if [[ ! -f "$reference_result" ]]; then
      CATANRL_CANOPY_REFERENCE_SIMULATIONS="$parity_budget" \
        scripts/eval_canopy_release_reference.sh \
          "$canopy_release_repo" \
          "$canopy_release_checkpoint" \
          "$reference_games" \
          "$reference_root"
    fi
    env -u VIRTUAL_ENV uv run python scripts/compare_canopy_reference.py \
      --candidate "$candidate_result" \
      --reference "$reference_result" \
      --budget "$parity_budget" \
      --noninferiority-margin "${CATANRL_CANOPY_NONINFERIORITY_MARGIN:-0.05}" \
      --output "$comparison_result"
  fi
fi
