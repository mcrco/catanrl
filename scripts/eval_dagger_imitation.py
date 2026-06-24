"""
Post-hoc imitation-quality eval for DAgger (Phase 1) model-size comparison.

Win-rate-vs-F is a poor discriminator here: we *imitate* ``F``
(``ValueFunctionPlayer``), so the ceiling is a ~50% mirror match and the metric
is noisy and flat across model sizes. This script instead measures how well a
policy reproduces the expert's *decision function* on a frozen, shared set of
decision points, using graded metrics that resolve small quality differences:

- value_regret_norm: F picks argmax of a heuristic value function, so for each
  state we know the value of *every* legal action. Regret is the normalized gap
  between the value of F's best action and the value of the action the model
  chose (0 = picked the best action, 1 = picked the worst). This is the graded
  version of top-1 agreement and is the headline metric.
- expert_ce / expert_action_prob: cross-entropy (NLL) of the expert's chosen
  action under the model's masked policy, i.e. the held-out DAgger loss. Smooth,
  low-variance, sensitive to size.
- agreement / top3_agreement: top-1 (== "picked F's value-best action") and
  top-3, restricted to non-forced states (>1 legal action).

The decision-point set is generated once from F-vs-F self-play with a fixed seed
(identical states for every checkpoint and every model size -> apples-to-apples),
the expert values are computed once, then every step checkpoint of every
experiment is scored cheaply via a single batched forward pass.

Results can be backfilled into each run's existing W&B run (summary for the
"best" checkpoint, plus a curve over training iterations on a custom x-axis so it
does not collide with the original ``global_step`` history).

Example::

    uv run scripts/eval_dagger_imitation.py \
        --experiments dagger-d-s dagger-d-m dagger-d-l dagger-d-shared \
        --num-games 80 --max-decision-points 4000 --seed 67 --wandb
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from catanatron.game import Game
from catanatron.players.value import ValueFunctionPlayer, get_value_fn

from catanrl.experiment_store import load_experiment
from catanrl.eval.vectorized_rollout import _get_policy_logits
from catanrl.features.catanatron_utils import (
    COLOR_ORDER,
    full_game_to_features,
    game_to_features,
    get_observation_indices_from_full,
)
from catanrl.utils.catanatron_action_space import to_action_space
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

_EPS = 1e-8


@dataclass
class DecisionPoint:
    """A single non-forced decision encountered during F-vs-F self-play."""

    private_features: np.ndarray  # game_to_features (private observation level)
    full_features: np.ndarray  # full_game_to_features (super-set of all levels)
    legal_indices: np.ndarray  # action-space index of each playable action
    expert_values: np.ndarray  # F's value of each playable action (aligned)
    expert_offset: int  # argmax(expert_values) == F's chosen action offset


class _RecordingValuePlayer(ValueFunctionPlayer):
    """``ValueFunctionPlayer`` that records every non-forced decision it makes.

    Behaves identically to ``F`` (returns the argmax-value action) so the
    trajectory distribution is exactly F-vs-F self-play, but it also stores the
    per-action values and observation features for later scoring.
    """

    def __init__(self, color, sink: List[DecisionPoint], map_type: str, max_points: int):
        super().__init__(color)
        self._sink = sink
        self._map_type = map_type
        self._max_points = max_points

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        value_fn = get_value_fn(self.value_fn_builder_name, self.params)
        values = np.empty(len(playable_actions), dtype=np.float64)
        for i, action in enumerate(playable_actions):
            game_copy = game.copy()
            game_copy.execute(action)
            values[i] = value_fn(game_copy, self.color)
        best_offset = int(np.argmax(values))

        if len(self._sink) < self._max_points:
            num_players = len(game.state.colors)
            game_colors = tuple(game.state.colors)
            legal_indices = np.array(
                [to_action_space(a, num_players, self._map_type, game_colors) for a in playable_actions],
                dtype=np.int64,
            )
            self._sink.append(
                DecisionPoint(
                    private_features=np.asarray(
                        game_to_features(game, self.color, num_players, self._map_type),
                        dtype=np.float32,
                    ),
                    full_features=np.asarray(
                        full_game_to_features(game, num_players, self._map_type, base_color=self.color),
                        dtype=np.float32,
                    ),
                    legal_indices=legal_indices,
                    expert_values=values.copy(),
                    expert_offset=best_offset,
                )
            )

        return playable_actions[best_offset]


def generate_decision_points(
    map_type: str,
    num_players: int,
    vps_to_win: int,
    discard_limit: int,
    num_games: int,
    max_decision_points: int,
    seed: int,
) -> List[DecisionPoint]:
    """Play F-vs-F games with a fixed seed and collect non-forced decisions."""
    if num_players != 2:
        raise ValueError("Phase 1 imitation eval assumes 2 players (F vs F).")

    sink: List[DecisionPoint] = []
    episode_seeds = [derive_seed(seed, "imitation_episode", g) for g in range(num_games)]

    from tqdm import tqdm

    for episode_seed in tqdm(episode_seeds, desc="self-play (F vs F)", unit="game"):
        if len(sink) >= max_decision_points:
            break
        players = [
            _RecordingValuePlayer(COLOR_ORDER[i], sink, map_type, max_decision_points)
            for i in range(num_players)
        ]
        for player in players:
            player.reset_state()
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        game = Game(
            players=players,
            catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        game.play()

    return sink[:max_decision_points]


def _build_observation_matrix(
    decision_points: Sequence[DecisionPoint],
    num_players: int,
    map_type: str,
    observation_level: str,
) -> np.ndarray:
    """Stack per-decision features for the model's observation level."""
    if observation_level == "private":
        return np.stack([dp.private_features for dp in decision_points], axis=0)
    indices = get_observation_indices_from_full(num_players, map_type, observation_level)
    full = np.stack([dp.full_features for dp in decision_points], axis=0)
    return full[:, indices]


def score_checkpoint(
    model: torch.nn.Module,
    model_type: str,
    decision_points: Sequence[DecisionPoint],
    observations: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, float]:
    """Compute imitation metrics for one checkpoint over the frozen set."""
    n = len(decision_points)
    if n == 0:
        raise ValueError("No decision points to score.")

    logits_chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = torch.from_numpy(observations[start : start + batch_size]).float().to(device)
            logits = _get_policy_logits(model, chunk, model_type)
            logits_chunks.append(logits.cpu().numpy())
    all_logits = np.concatenate(logits_chunks, axis=0)

    agree = np.zeros(n, dtype=np.float64)
    top3 = np.zeros(n, dtype=np.float64)
    regret = np.zeros(n, dtype=np.float64)
    ce = np.zeros(n, dtype=np.float64)
    p_expert = np.zeros(n, dtype=np.float64)

    for i, dp in enumerate(decision_points):
        legal_logits = all_logits[i, dp.legal_indices]
        # Masked policy = softmax over legal actions only.
        shifted = legal_logits - legal_logits.max()
        exp = np.exp(shifted)
        probs = exp / (exp.sum() + _EPS)

        model_offset = int(np.argmax(legal_logits))
        e = dp.expert_offset

        agree[i] = float(model_offset == e)
        k = min(3, legal_logits.shape[0])
        topk = np.argpartition(legal_logits, -k)[-k:]
        top3[i] = float(e in topk)

        v = dp.expert_values
        v_best = v[e]
        denom = v_best - v.min()
        regret[i] = 0.0 if denom <= _EPS else float((v_best - v[model_offset]) / denom)

        pe = float(probs[e])
        p_expert[i] = pe
        ce[i] = -float(np.log(pe + _EPS))

    return {
        "imitation/value_regret_norm": float(regret.mean()),
        "imitation/expert_ce": float(ce.mean()),
        "imitation/expert_action_prob": float(p_expert.mean()),
        "imitation/agreement": float(agree.mean()),
        "imitation/top3_agreement": float(top3.mean()),
        "imitation/num_decision_points": float(n),
    }


def _policy_step_checkpoints(experiment) -> List[int]:
    """Sorted list of integer training steps that have a policy checkpoint."""
    steps = set()
    for entry in experiment.registry.checkpoints:
        if entry.get("role") != "policy":
            files = entry.get("files")
            if not (isinstance(files, dict) and "policy" in files):
                continue
        step = entry.get("step")
        if step is not None:
            steps.add(int(step))
    return sorted(steps)


def _log_to_wandb(
    experiment,
    curve: Dict[int, Dict[str, float]],
    best_metrics: Optional[Dict[str, float]],
    entity_override: Optional[str],
    project_override: Optional[str],
) -> None:
    import wandb

    run_id = experiment.metadata.wandb.get("id")
    if not run_id:
        print(f"  [wandb] no run id in metadata for {experiment.metadata.name}; skipping.")
        return
    project = project_override or experiment.metadata.wandb.get("project") or "catan"
    entity = entity_override or experiment.metadata.wandb.get("entity")

    run = wandb.init(project=project, entity=entity, id=run_id, resume="must")
    try:
        if best_metrics is not None:
            for key, value in best_metrics.items():
                run.summary[f"{key}_best"] = value

        # Backfill the curve on an independent x-axis so we don't fight W&B's
        # monotonic global_step history on a finished run.
        metric_keys = sorted({k for m in curve.values() for k in m})
        wandb.define_metric("imitation/iter")
        for key in metric_keys:
            wandb.define_metric(key, step_metric="imitation/iter")
        for step in sorted(curve):
            wandb.log({"imitation/iter": step, **curve[step]})
    finally:
        run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", required=True, help="Experiment names under experiments/.")
    parser.add_argument("--num-games", type=int, default=80, help="F-vs-F self-play games used to build the state set.")
    parser.add_argument("--max-decision-points", type=int, default=4000, help="Cap on recorded non-forced decisions.")
    parser.add_argument("--seed", type=int, default=67, help="Seed for state-set generation (frozen across runs).")
    parser.add_argument("--which", nargs="+", default=None, help="Override selectors to score (default: all step checkpoints).")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--wandb", action="store_true", help="Backfill metrics into each run's existing W&B run.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Override W&B entity.")
    parser.add_argument("--wandb-project", type=str, default=None, help="Override W&B project.")
    parser.add_argument("--out-dir", type=str, default=None, help="Where to write per-experiment JSON (default: the experiment dir).")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # All Phase-1 runs share the same game config; read it from the first run.
    ref = load_experiment(args.experiments[0])
    map_type = ref.map_type
    num_players = ref.num_players
    vps_to_win = ref.metadata.game.vps_to_win
    discard_limit = ref.metadata.game.discard_limit

    print(f"Generating frozen decision-point set ({map_type}, {num_players}p, F vs F, seed={args.seed})...")
    decision_points = generate_decision_points(
        map_type=map_type,
        num_players=num_players,
        vps_to_win=vps_to_win,
        discard_limit=discard_limit,
        num_games=args.num_games,
        max_decision_points=args.max_decision_points,
        seed=args.seed,
    )
    print(f"Collected {len(decision_points)} non-forced decision points.\n")

    headline = ["imitation/value_regret_norm", "imitation/expert_ce", "imitation/agreement", "imitation/top3_agreement"]

    for exp_name in args.experiments:
        exp = load_experiment(exp_name)
        if exp.map_type != map_type or exp.num_players != num_players:
            raise ValueError(
                f"Experiment '{exp_name}' game config ({exp.map_type}, {exp.num_players}p) "
                f"differs from the reference ({map_type}, {num_players}p); generate a matching state set."
            )
        observation_level = exp.policy_spec.observation_level or "private"
        model_type = exp.model_type or "flat"
        observations = _build_observation_matrix(decision_points, num_players, map_type, observation_level)

        if args.which is not None:
            selectors: List = list(args.which)
            step_selectors = [s for s in selectors if str(s).isdigit()]
        else:
            step_selectors = _policy_step_checkpoints(exp)
            selectors = list(step_selectors) + ["best"]

        print(f"=== {exp_name} (obs={observation_level}, {model_type}) ===")
        results: Dict[str, Dict[str, float]] = {}
        curve: Dict[int, Dict[str, float]] = {}
        for selector in selectors:
            model = exp.build_policy(which=selector, device=device)
            metrics = score_checkpoint(model, model_type, decision_points, observations, device)
            results[str(selector)] = metrics
            if str(selector).isdigit():
                curve[int(selector)] = metrics
            summary = "  ".join(f"{k.split('/')[-1]}={metrics[k]:.4f}" for k in headline)
            print(f"  {str(selector):>6}: {summary}")

        best_metrics = results.get("best")

        out_dir = args.out_dir or exp.path
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "imitation_eval.json")
        with open(out_path, "w") as f:
            json.dump(
                {
                    "experiment": exp_name,
                    "state_set": {
                        "map_type": map_type,
                        "num_players": num_players,
                        "num_games": args.num_games,
                        "num_decision_points": len(decision_points),
                        "seed": args.seed,
                    },
                    "observation_level": observation_level,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"  wrote {out_path}")

        if args.wandb:
            _log_to_wandb(exp, curve, best_metrics, args.wandb_entity, args.wandb_project)
        print()


if __name__ == "__main__":
    main()
