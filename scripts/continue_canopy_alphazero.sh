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
global_iteration=$(env -u VIRTUAL_ENV UV_CACHE_DIR=/tmp/catanrl-uv-cache \
  uv run python "$repo_root/scripts/canopy_checkpoint_global_iteration.py" \
  --experiment "$source_experiment" \
  --checkpoint-step "$checkpoint_step")
if ! [[ "$global_iteration" =~ ^[0-9]+$ ]]; then
  echo "Could not resolve global self-play iteration: $global_iteration" >&2
  exit 1
fi
echo "Continuing global self-play schedule from iteration $global_iteration" >&2
exec env \
  CATANRL_PARITY_LOAD_WHICH="$checkpoint_step" \
  CATANRL_PARITY_SELF_PLAY_ITERATION_OFFSET="$global_iteration" \
  CATANRL_PARITY_REQUIRE_TERMINAL_DAGGER=0 \
  "$repo_root/scripts/train_canopy_alphazero.sh" \
  "$source_experiment" \
  "$experiment_name" \
  retain \
  "$seed"
