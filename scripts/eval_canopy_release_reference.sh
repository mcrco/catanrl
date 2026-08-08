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
expected_tag=catan-nexus-v3
expected_commit=6185983a88ba6802e7fa9893cef5a76a15de2595
expected_checkpoint_sha256=f8e4e6858930243a30243e38c1b2b96b1a8da23970f5cba69906c65b268c60cc
expected_harness_patch_sha256=506fb91bff6bbedb764031e8d252dd4cf6ef92ce91d1d22c9bbe353bcc4e3f67

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
release_tag=$(git -C "$canopy_repo" describe --tags --exact-match HEAD 2>/dev/null || true)
release_commit=$(git -C "$canopy_repo" rev-parse HEAD)
checkpoint_sha256=$(sha256sum "$checkpoint" | cut -d' ' -f1)
harness_changed_files=$(git -C "$canopy_repo" diff HEAD --name-only)
harness_patch_sha256=$(git -C "$canopy_repo" diff HEAD --binary -- src/mcts/mod.rs | sha256sum | cut -d' ' -f1)
if [[ "$release_tag" != "$expected_tag" ]]; then
  echo "Canopy checkout must be exact tag $expected_tag, got: ${release_tag:-untagged}" >&2
  exit 2
fi
if [[ "$release_commit" != "$expected_commit" ]]; then
  echo "Canopy release commit mismatch: $release_commit" >&2
  exit 2
fi
if [[ "$checkpoint_sha256" != "$expected_checkpoint_sha256" ]]; then
  echo "Canopy checkpoint checksum mismatch: $checkpoint_sha256" >&2
  exit 2
fi
if [[ "$harness_changed_files" != "src/mcts/mod.rs" ]]; then
  echo "Canopy harness must contain only the approved src/mcts/mod.rs patch" >&2
  echo "Apply it with: git -C $canopy_repo apply $repo_root/patches/canopy-nexus-v3-reset-search-budget.patch" >&2
  exit 2
fi
if [[ "$harness_patch_sha256" != "$expected_harness_patch_sha256" ]]; then
  echo "Canopy harness patch checksum mismatch: $harness_patch_sha256" >&2
  exit 2
fi

canopy_binary=${CATANRL_CANOPY_BINARY:-}
if [[ -n "$canopy_binary" ]]; then
  canopy_binary=$(realpath "$canopy_binary")
  if [[ ! -x "$canopy_binary" ]]; then
    echo "CATANRL_CANOPY_BINARY is not executable: $canopy_binary" >&2
    exit 2
  fi
fi
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
log_path="$output_dir/canopy-nexus-v3-s${simulations}-vs-random.log"
result_path="$output_dir/canopy-nexus-v3-s${simulations}-vs-random.json"

(
  cd "$canopy_repo"
  if [[ -n "$canopy_binary" ]]; then
    canopy_command=("$canopy_binary")
  else
    canopy_command=(cargo run --release --example catan --features cuda --)
  fi
  cargo_env=()
  if [[ -n "${CATANRL_CANOPY_CUDARC_VERSION:-}" ]]; then
    cargo_env=(env "CUDARC_CUDA_VERSION=$CATANRL_CANOPY_CUDARC_VERSION")
  fi
  RUST_LOG=info "${cargo_env[@]}" "${canopy_command[@]}" \
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
  --release-tag "$release_tag" \
  --release-commit "$release_commit" \
  --checkpoint-sha256 "$checkpoint_sha256" \
  --harness-patch-sha256 "$harness_patch_sha256" \
  --opponent random \
  --simulations "$simulations" \
  --max-actions "$max_actions"
