#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 EXPERIMENT [CHECKPOINT] [SEED]" >&2
  exit 2
fi

experiment=$1
checkpoint=${2:-best}
seed=${3:-12043}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

experiment_label=$(basename "$experiment")
output_root="experiments/eval-${experiment_label}-${checkpoint}-s${seed}"
raw_games_per_seat=${CATANRL_PARITY_RAW_EVAL_GAMES_PER_SEAT:-500}
search_games_per_seat=${CATANRL_PARITY_SEARCH_EVAL_GAMES_PER_SEAT:-100}
search_workers=${CATANRL_PARITY_EVAL_WORKERS:-16}
search_simulations=${CATANRL_PARITY_FULL_SIMULATIONS:-1600}
mkdir -p "$output_root"

env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 uv run python scripts/eval_vs_catanatron.py \
  --experiment "$experiment" \
  --which "$checkpoint" \
  --opponents F \
  --num-games "$raw_games_per_seat" \
  --nn-seat both \
  --seed "$seed" \
  --vps-to-win 15 \
  --discard-limit 9 \
  --device cuda \
  --paired-results-out "$output_root/raw-vs-f.json" \
  --wandb \
  --wandb-project catan \
  --wandb-run-name "eval-raw-${experiment_label}-${checkpoint}-s${seed}" \
  --wandb-group canopy-parity-eval

env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 uv run python scripts/eval_native_mcts_budget_sweep.py \
  --experiment "$experiment" \
  --which "$checkpoint" \
  --budgets "$search_simulations" \
  --skip-probes \
  --games-per-seat "$search_games_per_seat" \
  --game-opponent value \
  --num-workers "$search_workers" \
  --inference-batch-size 64 \
  --inference-wait-ms 2.0 \
  --c-puct 2.5 \
  --search-selection completed-q \
  --c-visit 50 \
  --c-scale 1.0 \
  --value-scale 1.0 \
  --tree-reuse \
  --canonical-pruning \
  --seed "$seed" \
  --turns-limit 1000 \
  --device cuda \
  --output-dir "$output_root/native-s${search_simulations}-vs-f" \
  --wandb \
  --wandb-project catan \
  --wandb-run-name "eval-native-s${search_simulations}-${experiment_label}-${checkpoint}-s${seed}" \
  --wandb-group canopy-parity-eval
