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

# Keep OS process count at this host's 16 logical CPUs. Multiplex two independent
# native games per process to expose the same 32 inference streams without the
# reliability cost of 32 spawned Python processes.
parity_workers=${CATANRL_PARITY_WORKERS:-16}
parity_games_per_worker=${CATANRL_PARITY_GAMES_PER_WORKER:-2}
# Canopy's Nexus-v3 preset collects 150k fresh decision samples per iteration.
# Native Catan self-play currently yields about 250 trainable decisions per game,
# so 600 games matches that data budget without changing game or reward semantics.
parity_games=${CATANRL_PARITY_GAMES_PER_ITERATION:-600}
parity_iterations=${CATANRL_PARITY_ITERATIONS:-60}
parity_full_simulations=${CATANRL_PARITY_FULL_SIMULATIONS:-1600}
parity_fast_simulations=${CATANRL_PARITY_FAST_SIMULATIONS:-64}
parity_stall_timeout=${CATANRL_PARITY_STALL_TIMEOUT_SECONDS:-600}
parity_inference_timeout=${CATANRL_PARITY_INFERENCE_TIMEOUT_SECONDS:-120}
case "${CATANRL_PARITY_AUX_VALUES:-0}" in
  0)
    aux_value_args=()
    aux_value_tag=no-aux-values
    ;;
  1)
    aux_value_args=(--aux-value-horizons 10 50 150 --aux-value-weight 0.5)
    aux_value_tag=aux-values
    ;;
  *)
    echo "CATANRL_PARITY_AUX_VALUES must be 0 or 1" >&2
    exit 2
    ;;
esac
wandb_args=()
case "${CATANRL_PARITY_WANDB:-0}" in
  0) ;;
  1)
    wandb_args=(
      --wandb
      --wandb-project catan
      --wandb-run-name "$experiment_name"
      --wandb-group canopy-parity
      --wandb-tags native-cppanatron catan-graph nexus-v3 road-aware \
        corrected-board-layout shared-backbone full-full win-reward fresh-dagger-pretrain \
        canopy-playout-cap canopy-completed-q canopy-soft-policy \
        canopy-adamw continuous-teacher tree-reuse "$aux_value_tag" "$value_init_tag"
    )
    ;;
  *)
    echo "CATANRL_PARITY_WANDB must be 0 or 1" >&2
    exit 2
    ;;
esac

env -u VIRTUAL_ENV uv run python scripts/verify_canopy_contract.py \
  --experiment "$dagger_experiment" \
  --which best \
  --require-terminal-dagger

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
  --games-per-worker "$parity_games_per_worker" \
  --inference-batch-size 64 \
  --inference-wait-ms 2.0 \
  --max-actions 2000 \
  --self-play-stall-timeout-seconds "$parity_stall_timeout" \
  --inference-response-timeout-seconds "$parity_inference_timeout" \
  --self-play-result-chunk-size 64 \
  --self-play-max-attempts 3 \
  --temperature 1.0 \
  --final-temperature 0.1 \
  --target-temperature 1.0 \
  --temperature-drop-move 30 \
  --trajectory-action-selection canopy \
  --explore-actions 24 \
  --noise-turns 24 \
  --dirichlet-alpha 0.05 \
  --dirichlet-frac 0.25 \
  --buffer-size 500000 \
  --batch-size 1024 \
  --policy-lr 1e-4 \
  --critic-lr 1e-4 \
  --weight-decay 0.0004 \
  --optimizer-epsilon 1e-5 \
  --policy-loss-weight 1.0 \
  --value-loss-weight 1.0 \
  --soft-policy-temperature 4.0 \
  --soft-policy-weight 8.0 \
  "${aux_value_args[@]}" \
  --max-grad-norm 1.0 \
  --eval-every-iterations 5 \
  --eval-games 200 \
  --eval-seed 123 \
  --h2h-games 200 \
  --h2h-seed 67 \
  --save-every-updates 1 \
  --device cuda \
  --seed "$seed" \
  --experiment-name "$experiment_name" \
  "${wandb_args[@]}"
