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
eval_root="experiments/eval-${alphazero_experiment}-best-s12043"

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
  mkdir -p "$eval_root"
  scripts/eval_canopy_parity.sh "$alphazero_experiment" best 12043 \
    2>&1 | tee "$eval_root/run.log"
fi

