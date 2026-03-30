# Catan Bot

Current architecture (see `src/catanrl/backbones`) is 

- CNN features for board,
- MLP for numeric features (e.g. player hands)
- then concat them and do one more MLP on the fused features
- pretty much what [https://arxiv.org/abs/2008.07079](this paper) does, but without the residual connections).

Winrate going first is 70.6%, winrate going second is ~67.7% on 100 games.

- best weights: [https://drive.google.com/file/d/1C3c5Rk9Xlz2WTLCES4553aQwFK6XAOFp/view?usp=sharing](google drive link)
- DAgger: [https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/regwxcqf?nw=nwusermyang2](wandb run)
- PPO: [https://wandb.ai/myang2-california-institute-of-technology-caltech/catan-rl/runs/rcew2svr?nw=nwusermyang2](wandb run)
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
