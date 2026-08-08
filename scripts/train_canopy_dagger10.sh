#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 EXPERIMENT_NAME [SEED]" >&2
  exit 2
fi

experiment_name=$1
seed=${2:-43}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
model_config=${CATANRL_PARITY_MODEL_CONFIG:-configs/models/xdim-compact-medium-flat-2p-full-shared.yaml}
wandb_args=()
case "${CATANRL_PARITY_WANDB:-0}" in
  0) ;;
  1)
    wandb_args=(
      --wandb
      --wandb-project catan
      --wandb-run-name "$experiment_name"
      --wandb-group dagger
      --wandb-tags native-cppanatron compact-xdim corrected-board-layout \
        shared-backbone full-full win-reward fresh-init canopy-pretrain
    )
    ;;
  *)
    echo "CATANRL_PARITY_WANDB must be 0 or 1" >&2
    exit 2
    ;;
esac

env -u VIRTUAL_ENV uv run python scripts/verify_canopy_contract.py \
  --config "$model_config"

env -u VIRTUAL_ENV PYTHONUNBUFFERED=1 uv run python -m catanrl.experiments.train_dagger \
  --config "$model_config" \
  --iterations 10 \
  --steps-per-iter 8192 \
  --train-epochs 2 \
  --batch-size 1024 \
  --policy-lr 1e-4 \
  --critic-lr 1e-4 \
  --gamma 1.0 \
  --expert F \
  --opponents F \
  --num-envs 8 \
  --env-backend cppanatron \
  --reward-function win \
  --beta-init 1.0 \
  --beta-decay 0.97 \
  --beta-min 0.1 \
  --max-dataset-size 1500000 \
  --eviction-strategy fifo \
  --fresh-eval-games-per-opponent 500 \
  --eval-every-iterations 5 \
  --imitation-eval-games 80 \
  --imitation-eval-max-decision-points 4000 \
  --imitation-eval-seed 67 \
  --save-every-updates 5 \
  --device cuda \
  --seed "$seed" \
  --experiment-name "$experiment_name" \
  "${wandb_args[@]}"
