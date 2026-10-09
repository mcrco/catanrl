"""Large deterministic vs-F eval of one checkpoint on the native engine.

Usage: scripts/native_eval_vs_f.py EXPERIMENT WHICH GAMES_PER_SEAT OUT_JSON
"""
import json
import sys

from catanrl.eval.vectorized_rollout import run_policy_value_eval_vectorized
from catanrl.experiment_store import load_experiment

name, which, games, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
exp = load_experiment(name)
policy = exp.build_policy(which=which, device="cuda")
critic = exp.build_critic(which=which, device="cuda")
actor_level = exp.policy_spec.observation_level or "full"
critic_spec = exp.metadata.networks.get("critic")
critic_level = (critic_spec.observation_level if critic_spec is not None else None) or actor_level
result = {"experiment": name, "which": which, "games_per_seat": games, "seed": 67}
for seat in ("first", "second"):
    wins, turns, *_ = run_policy_value_eval_vectorized(
        policy, critic, "flat", exp.map_type, num_envs=64, num_games=games, gamma=0.99,
        opponent_configs=["F"], nn_seat=seat, actor_observation_level=actor_level,
        critic_observation_level=critic_level, seed=67, env_backend="cppanatron", num_workers=8,
        number_placement=exp.number_placement,
    )
    result[f"wins_{seat}"] = wins
    result[f"mean_turns_{seat}"] = sum(turns) / len(turns)
result["win_rate"] = (result["wins_first"] + result["wins_second"]) / (2 * games)
print(json.dumps(result))
json.dump(result, open(out, "w"), indent=2)
