#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CANOPY_CHECKOUT [OUTPUT_BINARY]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
canopy_repo=$(realpath "$1")
output_binary=${2:-"$repo_root/experiments/canopy-release-reference/catanatron-bridge"}
expected_commit=6185983a88ba6802e7fa9893cef5a76a15de2595
actual_commit=$(git -C "$canopy_repo" rev-parse HEAD)

if [[ "$actual_commit" != "$expected_commit" ]]; then
  echo "Canopy checkout must be release commit $expected_commit, got $actual_commit" >&2
  exit 1
fi

patch_path="$repo_root/patches/canopy-catanatron-bridge.patch"
if git -C "$canopy_repo" apply --check "$patch_path"; then
  git -C "$canopy_repo" apply "$patch_path"
elif git -C "$canopy_repo" apply --reverse --check "$patch_path"; then
  echo "Canopy Catanatron bridge patch is already applied"
else
  echo "Canopy checkout has incompatible changes; bridge patch cannot be applied" >&2
  exit 1
fi

cuda_home=${CUDA_HOME:-/opt/cuda}
cudarc_cuda_version=${CUDARC_CUDA_VERSION:-13010}
target_dir=${CARGO_TARGET_DIR:-"$canopy_repo/target"}

env \
  CUDA_HOME="$cuda_home" \
  CUDARC_CUDA_VERSION="$cudarc_cuda_version" \
  CARGO_TARGET_DIR="$target_dir" \
  cargo build \
    --manifest-path "$canopy_repo/Cargo.toml" \
    --release \
    --example catan \
    --features cuda

mkdir -p "$(dirname "$output_binary")"
install -m 755 "$target_dir/release/examples/catan" "$output_binary"
echo "Built Canopy Catanatron bridge: $output_binary"
