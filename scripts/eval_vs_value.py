import torch
from catanatron.players.value import ValueFunctionPlayer
import argparse

from catanrl.eval.eval_nn_vs_catanatron import eval
from catanrl.features.catanatron_utils import COLOR_ORDER, compute_feature_vector_dim
from catanrl.models import (
    BackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import build_flat_policy_network, build_hierarchical_policy_network
from catanrl.players import NNPolicyPlayer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    backbone_config = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(
            input_dim=compute_feature_vector_dim(2, "BASE"),
            hidden_dims=[4096, 4096],
        ),
    )
    model = build_flat_policy_network(backbone_config)
    model_path = "weights/ppo-central-critic/policy_best.pt"
    model.load_state_dict(
        torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    )

    nn_player = NNPolicyPlayer(color=COLOR_ORDER[0], model_type="flat", model=model)
    opponents = [ValueFunctionPlayer(COLOR_ORDER[1])]
    wins, vps, total_vps, turns = eval(
        nn_player, opponents, map_type="BASE", num_games=args.num_games, seed=args.seed, show_tqdm=True
    )
    print(f"Wins: {wins}")
    print(f"Average VPS: {sum(vps) / len(vps)}")
    print(f"Average Total VPS: {sum(total_vps) / len(total_vps)}")
    print(f"Average Turns: {sum(turns) / len(turns)}")
