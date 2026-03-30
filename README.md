# Catan Bot

Current nn architecture (see `src/catanrl/backbones`) is 

- CNN features for board,
- MLP for numeric features (e.g. player hands)
- then concat them and do one more MLP on the fused features
- basically a super washed down version of [this paper](https://arxiv.org/abs/2008.07079).

Current training is

- first do imitation learning (DAgger) on the Catanatron `ValueFunctionPlayer` for a decent base model
- then do PPO against the `ValueFunctionPlayer`

Winrate against the Catanatron value function bot when going first is 70.6%, winrate against the Catanatron value function player going second is ~67.7% on 100 games.

- best weights: [google drive link](https://drive.google.com/file/d/1C3c5Rk9Xlz2WTLCES4553aQwFK6XAOFp/view?usp=sharing)
- DAgger: [wandb run](https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/regwxcqf?nw=nwusermyang2)
- PPO: [wandb run](https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/rcew2svr?nw=nwusermyang2)
- eval command:
```
uv run scripts/eval_vs_catanatron.py \
  --model-type flat \
  --backbone-type xdim \
  --policy-hidden-dims 2048 2048 \
  --policy-weights weights/ppo-sarl-f-winreward-xdim-flat-pretrained-dagger/policy_best.pt \
  --num-games 1000 \
  --seed 67 \
  --nn-seat {first/second}
```

## TODO

- [] Merge Catanatron fork with upstream (main difference is discard actions, which upstream seems to be implementing soon in [this PR](https://github.com/bcollazo/catanatron/pull/368/changes).
- [] Track all public information (e.g. in 2 player game, only private info should be dev cards, and in 3+ player game, only private info should be dev cards and stolen cards between opponents) in features, and then retrain model
- [] Train sequential model architecture (e.g. RNN, LSTM) or represent all timestep-based information in features, like last time player has played knight and last time robber was moved
- [] Train using self-play with central critic PPO (code is there, just haven't had success with it but in theory it should be good)
- [] Train with MCTS instead of just having MCTS harness for PPO-trained model
