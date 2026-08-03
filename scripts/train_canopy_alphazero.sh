#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 DAGGER_EXPERIMENT EXPERIMENT_NAME [retain|reset] [SEED]" >&2
  exit 2
fi

dagger_experiment=$1
experiment_name=$2
value_init=${3:-retain}
seed=${4:-43}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

case "$value_init" in
  retain)
    value_init_args=()
    value_init_tag=dagger-value-head
    ;;
  reset)
    value_init_args=(--reset-loaded-value-head)
    value_init_tag=fresh-value-head
    ;;
  *)
    echo "value initialization must be 'retain' or 'reset'" >&2
    exit 2
    ;;
esac

parity_workers=${CATANRL_PARITY_WORKERS:-32}
parity_games=${CATANRL_PARITY_GAMES_PER_ITERATION:-256}
parity_iterations=${CATANRL_PARITY_ITERATIONS:-60}
parity_full_simulations=${CATANRL_PARITY_FULL_SIMULATIONS:-1600}
parity_fast_simulations=${CATANRL_PARITY_FAST_SIMULATIONS:-64}

env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 uv run python -m catanrl.experiments.train_alphazero \
  --mode iterate \
  --teacher-update latest \
  --load-from-experiment "$dagger_experiment" \
  --load-from-which best \
  "${value_init_args[@]}" \
  --self-play-backend cppanatron \
  --ismcts-determinizations 1 \
  --simulations "$parity_full_simulations" \
  --fast-simulations "$parity_fast_simulations" \
  --full-search-probability 0.25 \
  --c-puct 2.5 \
  --value-scale 1.0 \
  --tree-reuse \
  --canonical-pruning \
  --search-selection completed-q \
  --policy-target completed-q \
  --c-visit 50 \
  --c-scale 1.0 \
  --search-value-weight-max 0.85 \
  --search-value-weight-ramp-iterations 60 \
  --iterations "$parity_iterations" \
  --games-per-iteration "$parity_games" \
  --optimizer-epochs 2 \
  --num-workers "$parity_workers" \
  --inference-batch-size 64 \
  --inference-wait-ms 2.0 \
  --temperature 1.0 \
  --final-temperature 0.1 \
  --target-temperature 1.0 \
  --temperature-drop-move 30 \
  --noise-turns 24 \
  --dirichlet-alpha 0.3 \
  --dirichlet-frac 0.25 \
  --buffer-size 500000 \
  --batch-size 1024 \
  --policy-lr 1e-4 \
  --critic-lr 1e-4 \
  --policy-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --soft-policy-temperature 4.0 \
  --soft-policy-weight 8.0 \
  --max-grad-norm 1.0 \
  --eval-every-iterations 5 \
  --eval-games 200 \
  --eval-seed 123 \
  --h2h-games 200 \
  --h2h-seed 67 \
  --save-every-updates 5 \
  --device cuda \
  --seed "$seed" \
  --experiment-name "$experiment_name" \
  --wandb \
  --wandb-project catan \
  --wandb-run-name "$experiment_name" \
  --wandb-group canopy-parity \
  --wandb-tags native-cppanatron compact-xdim corrected-board-layout \
    shared-backbone full-full win-reward fresh-dagger-pretrain \
    canopy-playout-cap canopy-completed-q canopy-soft-policy \
    continuous-teacher tree-reuse "$value_init_tag"
