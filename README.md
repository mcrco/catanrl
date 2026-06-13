# Catan Bot

## NN Architecture 

- CNN features for board,
- MLP for numeric features (e.g. player hands)
- then concat them and do one more MLP on the fused features
- basically a super washed down version of [this paper](https://arxiv.org/abs/2008.07079).
- see `src/catanrl/models` 

## Training Algorithms

- first do imitation learning (DAgger) on the Catanatron `ValueFunctionPlayer` for a decent base model
- then do PPO against the `ValueFunctionPlayer`

## Winrate 

- 70.6% over 1000 games against ValueFunction when going first
- 67.7% over 1000 games against ValueFunction when going second
- 69.7% over 1000 games against AlphaBeta when going first
- 66.0% over 1000 games against AlphaBeta when going second

## Experiments

Every training run is a **self-describing experiment** under `experiments/<name>/`
(gitignored). The directory holds everything needed to reload a model later:

```
experiments/<name>/
  metadata.json      # exact architecture (backbone + head type) of each network
  checkpoints.json   # best/latest/step -> checkpoint file, per role (policy/critic)
  checkpoints/       # the .pt state_dicts
```

Because `metadata.json` records the precise architecture, you never have to
re-specify `--backbone-type` / `--policy-hidden-dims` / `--map-type` etc. at eval
time — the model rebuilds itself from the experiment.

### Training

All training scripts take `--experiment-name` (it also becomes the W&B run name,
so the run and the weights always cross-reference). If omitted, it falls back to
`--wandb-run-name`, else an auto-generated `<algo>-<timestamp>`.

```
uv run train-sarl-ppo \
  --experiment-name ppo-sarl-f-winreward-xdim-flat-pretrained-dagger \
  --backbone-type xdim --model-type flat --hidden-dims 2048,2048 \
  --opponents F --wandb
```

Checkpoints are written to `experiments/<name>/checkpoints/` and metadata is
emitted automatically when training finishes.

### Loading a model in code

```python
from catanrl.experiment_store import load_experiment

exp = load_experiment("ppo-sarl-f-winreward-xdim-flat-pretrained-dagger")
policy = exp.build_policy(which="best", device="cuda")   # arch rebuilt + weights loaded
critic = exp.build_critic(which="best", device="cuda")   # if the run has a critic
```

`which` accepts `"best"`, `"latest"`, or an explicit training step.

### Evaluating

Pass `--experiment` (architecture is read from metadata); `--which` selects the
checkpoint. The legacy `--policy-weights` + architecture flags still work.

```
uv run scripts/eval_vs_catanatron.py \
  --experiment ppo-sarl-f-winreward-xdim-flat-pretrained-dagger \
  --which best \
  --num-games 1000 \
  --seed 67 \
  --nn-seat {first/second} \
  --opponents {AB:2/F}
```

### Managing experiments

Experiments are written automatically by the training entrypoints under
`src/catanrl/experiments/`. Each run is self-describing: list them with
`ls experiments/`, and inspect any run by reading its plain-JSON
`experiments/<name>/metadata.json` and `experiments/<name>/checkpoints.json`.
Because each run records its exact network architecture in `metadata.json`
directly from the trained model, `load_experiment(...)` rebuilds loadable models
without re-deriving anything from the weights.

### Reference run

- best weights: [google drive link](https://drive.google.com/file/d/1C3c5Rk9Xlz2WTLCES4553aQwFK6XAOFp/view?usp=sharing)
- DAgger: [wandb run](https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/regwxcqf?nw=nwusermyang2)
- PPO: [wandb run](https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/rcew2svr?nw=nwusermyang2)

## TODO

- [ ] Merge Catanatron fork with upstream (main difference is discard actions, which upstream seems to be implementing soon in [this PR](https://github.com/bcollazo/catanatron/pull/368/changes).
- [ ] Track all public information (e.g. in 2 player game, only private info should be dev cards, and in 3+ player game, only private info should be dev cards and stolen cards between opponents) in features, and then retrain model
- [ ] Train sequential model architecture (e.g. RNN, LSTM) or represent all timestep-based information in features, like last time player has played knight and last time robber was moved
- [ ] Train using self-play with central critic PPO (code is there, just haven't had success with it but in theory it should be good)
- [ ] Train with MCTS instead of just having MCTS harness for PPO-trained model
