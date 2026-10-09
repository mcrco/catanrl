# catanrl

Reinforcement learning for two-player Catan. Agents train on the
[Catanatron](https://github.com/bcollazo/catanatron) rules (BASE map, 15 VP to
win, discard limit 9, official-spiral number placement) and are measured
against F, Catanatron's `ValueFunctionPlayer`. Training combines imitation of F
(DAgger), self-play multi-agent PPO with a central critic, and MCTS at
deployment. Rollouts, evaluations and search run on
[cppanatron](https://github.com/mcrco/cppanatron), a C++20 port of Catanatron.

**Experiments are recorded in [`report/report.pdf`](report/report.pdf)**
(source: [`report/report.typ`](report/report.typ)). The report is updated with
every new experiment; this README only covers setup and entry points.

## Current result

The best agent is the graph-s reference
(`m-graph-s-full-full-native-g1-from-warm-s42-20261001`, update 1500): a 0.5M
parameter graph network ported from Canopy's Nexus-v3 network (see Credits
below), trained by DAgger from F and then self-play PPO.

| Agent | Win rate vs F |
| --- | --- |
| Raw policy, belief-averaged (fair, 2,000 games) | 89.65% (95% CI 88.2–90.9) |
| MCTS, 256 simulations, determinized (fair, 1,000 games) | 90.5% (88.5–92.2) |
| MCTS, 1024 simulations on the true game (sees hidden cards, 1,000 games) | 92.7% (90.9–94.2) |

## Installation

Requires [uv](https://docs.astral.sh/uv/), Python 3.11+ and a C++20 compiler.
Both engines are git submodules, and `cppanatron` is an editable dependency:
`uv sync` builds its nanobind extension and rebuilds it whenever the engine
sources change.

```bash
git clone --recurse-submodules git@github.com:mcrco/catanrl.git
cd catanrl
# In an existing clone: git submodule update --init --recursive
uv sync --extra cu130                    # PyTorch from the CUDA 13.0 index
uv run pytest cppanatron/tests/python    # engine rules and parity with Catanatron
uv run pytest                            # catanrl tests
```

To build the report, install [Typst](https://typst.app) and run
`typst compile report/report.typ report/report.pdf`.

## Entry points

Training (console scripts from `pyproject.toml`):

- `uv run train-dagger`: imitation learning (DAgger) from F; `--env-backend cppanatron` collects on the native engine.
- `uv run train-marl-cc`: self-play MARL PPO with a central critic; `--env-backend cppanatron-native` for the vectorized native env, `--eval-env-backend cppanatron` for native training-time evals.
- `uv run train-sarl-ppo`: single-agent PPO against fixed bots (earlier experiments).
- `uv run train-alphazero`: search distillation and AlphaZero-style training (earlier experiments).

Every training run needs `--config configs/models/<preset>.yaml` or
`--load-from-experiment <name>`, plus `--experiment-name`. Current presets:
`catan-graph-s-flat-2p-full-sep.yaml`, `catan-graph-m-flat-2p-full-sep.yaml` and
`xdim-flat-2p-full-sep.yaml`. Boards default to `--number-placement
official_spiral`; `random` also shuffles the number tokens.

Evaluation:

- `scripts/eval_search_matchup.py`: paired native games between raw policies, MCTS agents (`mcts:sims=256,dets=1`) and F, with Wilson CIs and McNemar tests.
- `scripts/eval_belief_policy.py`: fair belief-averaged evaluation of a full-information policy vs F (`--also-oracle` adds the oracle on the same seeds).
- `scripts/eval_policy_h2h.py`: head-to-head between two experiments (`--deployable` for belief averaging).
- `scripts/eval_vs_catanatron.py`: evaluation vs Catanatron bots on the Python engine.
- `scripts/compare_paired_eval.py`: exact McNemar test between two paired result files.
- `scripts/native_eval_vs_f.py EXP WHICH GAMES_PER_SEAT OUT_JSON` and `scripts/native_h2h.py EXP WHICH OPP_EXP OPP_WHICH GAMES_PER_SEAT OUT_JSON`: the native vs-F and head-to-head evaluations behind the report's numbers (seed 67, both seats).

### Web UI

To play against saved experiments in Catanatron's web UI:

```bash
uv sync --extra cu130 --extra web               # adds catanatron[web]
uv run catanrl-web                              # API on :5001; optional --experiments a,b --device cuda
cd catanatron/ui && npm install && npm start    # UI on http://localhost:3000
```

Every saved experiment becomes a player in Catanatron's registry, keyed
`CATANRL_<EXPERIMENT>`. Full-information policies appear twice:
`<experiment> (belief-averaged)`, which does not see your hidden cards, and the
oracle (`CATANRL_ORACLE_<EXPERIMENT>`), which does. A model only plays games
with the player count and map it was trained on.

## Experiments

Each run is saved under `experiments/<name>/` (gitignored): `metadata.json`
holds the exact architecture, game settings and training config, and
`checkpoints.json` indexes the checkpoints. Models rebuild themselves from it:

```python
from catanrl.experiment_store import load_experiment

exp = load_experiment("m-graph-s-full-full-native-g1-from-warm-s42-20261001")
policy = exp.build_policy(which="1500", device="cuda")  # "best", "latest" or an update
critic = exp.build_critic(which="1500", device="cuda")
```

## Layout

```
src/catanrl/      algorithms (DAgger, PPO, AlphaZero), models, envs (Puffer, cppanatron),
                  features, beliefs, players, eval
configs/models/   architecture and game presets
scripts/          evaluation scripts
report/           research report (Typst source, PDF, plotted data)
catanatron/       Catanatron fork (submodule)
cppanatron/       C++20 engine (submodule)
```

## Credits / third-party sources

- [Catanatron](https://github.com/bcollazo/catanatron) by Bryan Collazo
  (GPL-3.0): the reference engine and the baseline opponents, including F.
  We use a fork as the `catanatron/` submodule.
- [cppanatron](https://github.com/mcrco/cppanatron) (GPL-3.0-or-later): our
  C++20 rewrite of Catanatron, with the native vectorized environment, native F
  and native MCTS.
- [Canopy](https://github.com/cullback/canopy) by cullback: the `catan_graph`
  backbone (`CatanGraphBackbone` in `src/catanrl/models/backbones.py`), its graph
  policy and value heads (`src/catanrl/models/heads.py`) and the graph-s/graph-m
  presets are a port of Canopy's Nexus-v3 network at release tag
  `catan-nexus-v3` (commit `6185983a88ba6802e7fa9893cef5a76a15de2595`;
  `examples/catan/model/nexus_v3.rs` and `examples/catan/encoder/nexus_v3.rs`),
  adapted to Catanatron's observation vector and action space. Canopy has no
  license file.
- [rl_games](https://github.com/Denys88/rl_games) by Denys Makoviichuk: the
  KL-adaptive policy learning rate is its `AdaptiveScheduler`, which applies the
  PPO paper's adaptive-KL rule to the step size.
- [Playing Catan with Cross-dimensional Neural Network](https://arxiv.org/abs/2008.07079)
  by Gendre and Kaneko: the basis of the xdim CNN backbone.
