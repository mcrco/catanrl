"""Native head-to-head between two experiment checkpoints (balanced seats; any architectures).

Usage: scripts/native_h2h.py EXPERIMENT WHICH OPPONENT_EXPERIMENT OPPONENT_WHICH GAMES_PER_SEAT OUT_JSON
"""
import json
import sys

from catanrl.eval.training_eval import eval_policy_against_champion
from catanrl.experiment_store import load_experiment

name, which, opp_name, opp_which, games, out = sys.argv[1:7]
games = int(games)
exp, opp = load_experiment(name), load_experiment(opp_name)
policy = exp.build_policy(which=which, device="cuda")
opponent = opp.build_policy(which=opp_which, device="cuda")
result = {"experiment": name, "which": which, "opponent": opp_name, "opponent_which": opp_which,
          "games_per_seat": games, "seed": 67}
for seat in ("first", "second"):
    m = eval_policy_against_champion(
        policy, "flat", exp.map_type, exp.num_players, "", num_games=games, seed=67,
        log_to_wandb=False, num_envs=128, device="cuda", nn_seat=seat,
        actor_observation_level=exp.policy_spec.observation_level,
        number_placement=exp.number_placement, env_backend="cppanatron",
        champion_model=opponent, champion_actor_observation_level=opp.policy_spec.observation_level,
    )
    result[f"win_rate_{seat}"] = m["eval_h2h/win_rate_vs_champion"]
result["win_rate"] = (result["win_rate_first"] + result["win_rate_second"]) / 2
print(json.dumps(result))
json.dump(result, open(out, "w"), indent=2)
