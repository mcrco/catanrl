#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 CANOPY_REPO MODEL_ITER_CHECKPOINT [NUM_GAMES] [OUTPUT_DIR]" >&2
  exit 2
fi

canopy_repo_arg=$1
checkpoint_arg=$2
# Match the native final confirmation's power for a five-point independent
# noninferiority margin. Callers can still pass a smaller exploratory count.
num_games=${3:-800}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir=${4:-"$repo_root/experiments/canopy-release-reference"}
simulations=${CATANRL_CANOPY_REFERENCE_SIMULATIONS:-1600}
max_actions=${CATANRL_CANOPY_REFERENCE_MAX_ACTIONS:-2000}

if [[ ! -f "$canopy_repo_arg/Cargo.toml" ]]; then
  echo "Canopy repository not found at $canopy_repo_arg" >&2
  exit 2
fi
if [[ ! -f "$checkpoint_arg" ]]; then
  echo "Canopy checkpoint not found at $checkpoint_arg" >&2
  exit 2
fi
if [[ ! "$num_games" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GAMES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$simulations" =~ ^[1-9][0-9]*$ ]]; then
  echo "CATANRL_CANOPY_REFERENCE_SIMULATIONS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$max_actions" =~ ^[1-9][0-9]*$ ]]; then
  echo "CATANRL_CANOPY_REFERENCE_MAX_ACTIONS must be a positive integer" >&2
  exit 2
fi

canopy_repo=$(realpath "$canopy_repo_arg")
checkpoint=$(realpath "$checkpoint_arg")
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
log_path="$output_dir/canopy-nexus-v3-s${simulations}-vs-random.log"
result_path="$output_dir/canopy-nexus-v3-s${simulations}-vs-random.json"

(
  cd "$canopy_repo"
  RUST_LOG=info cargo run --release --example catan --features cuda -- \
    --random-dice \
    --vp-limit 15 \
    --discard-threshold 9 \
    --num-games "$num_games" \
    --p1-eval "nexus-v3:$checkpoint" \
    --p2-eval random \
    --p1-sims "$simulations" \
    --p2-sims 0 \
    --p1-c-puct 2.5 \
    --p1-c-visit 50 \
    --p1-c-scale 1 \
    --max-actions "$max_actions"
) 2>&1 | tee "$log_path"

cd "$repo_root"
env -u VIRTUAL_ENV uv run python scripts/parse_canopy_tournament.py \
  "$log_path" \
  --output "$result_path" \
  --checkpoint "$checkpoint" \
  --opponent random \
  --simulations "$simulations" \
  --max-actions "$max_actions"
