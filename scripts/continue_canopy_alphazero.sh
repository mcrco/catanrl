#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 SOURCE_EXPERIMENT CHECKPOINT_STEP EXPERIMENT_NAME [SEED]" >&2
  exit 2
fi

source_experiment=$1
checkpoint_step=$2
experiment_name=$3
seed=${4:-53}
if ! [[ "$checkpoint_step" =~ ^[0-9]+$ ]]; then
  echo "CHECKPOINT_STEP must be a non-negative integer" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec env \
  CATANRL_PARITY_LOAD_WHICH="$checkpoint_step" \
  CATANRL_PARITY_SELF_PLAY_ITERATION_OFFSET="$checkpoint_step" \
  CATANRL_PARITY_REQUIRE_TERMINAL_DAGGER=0 \
  "$repo_root/scripts/train_canopy_alphazero.sh" \
  "$source_experiment" \
  "$experiment_name" \
  retain \
  "$seed"
