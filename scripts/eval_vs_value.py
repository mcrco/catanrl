from catanrl.models import BackboneConfig, MLPBackboneConfig
from catanrl.players import create_nn_policy_player
from catanatron.players.value import ValueFunctionPlayer
from catanrl.features.catanatron_utils import compute_feature_vector_dim, COLOR_ORDER
from catanrl.eval.eval_nn_vs_catanatron import eval

if __name__ == "__main__":
    backbone_config = BackboneConfig(
        architecture="mlp",
        args=MLPBackboneConfig(
            input_dim=compute_feature_vector_dim(2, "BASE"),
            hidden_dims=[512, 512],
        ),
    )
    nn_player = create_nn_policy_player(
        COLOR_ORDER[0],
        model_type="hierarchical",
        backbone_config=backbone_config,
        model_path="weights/policy_value_rl_best.pt",
    )
    opponents = [ValueFunctionPlayer(COLOR_ORDER[1])]
    wins, vps, total_vps, turns = eval(
        nn_player, opponents, map_type="BASE", num_games=100, seed=42
    )
    print(f"Wins: {wins}")
    print(f"Average VPS: {sum(vps) / len(vps)}")
    print(f"Average Total VPS: {sum(total_vps) / len(total_vps)}")
    print(f"Average Turns: {sum(turns) / len(turns)}")
