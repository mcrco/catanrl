"""
Evaluate a trained policy network against ValueFunctionPlayer opponents.

This script uses the same model configuration parameters as dagger.py,
allowing plug-and-play evaluation of DAgger-trained models.
"""

import argparse
from typing import Literal, Sequence

import torch
import wandb
from catanatron.players.value import ValueFunctionPlayer

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.eval.eval_nn_vs_catanatron import eval
from catanrl.experiment_store import (
    backbone_display_type,
    backbone_hidden_dims,
    load_experiment,
)
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT
from catanrl.features.catanatron_utils import ActorObservationLevel, COLOR_ORDER
from catanrl.models.backbones import (
    BackboneConfig,
    MLPBackboneConfig,
    CrossDimensionalBackboneConfig,
)
from catanrl.models.models import build_flat_policy_network, build_hierarchical_policy_network
from catanrl.models.wrappers import PolicyNetworkWrapper
from catanrl.players import NNPolicyPlayer
from catanrl.utils.catanatron_action_space import get_action_space_size


BOARD_WIDTH = 21
BOARD_HEIGHT = 11


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    actor_observation_level: ActorObservationLevel,
    device: torch.device,
) -> PolicyNetworkWrapper:
    """Build a policy network with the same configuration as dagger.py."""
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        actor_observation_level=actor_observation_level,
    )
    actor_dim = dims["actor_dim"]
    numeric_dim = dims["numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=actor_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    if model_type == "flat":
        model = build_flat_policy_network(
            backbone_config=backbone_config,
            num_actions=get_action_space_size(num_players, map_type),
        )
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(
            backbone_config=backbone_config,
            num_players=num_players,
            map_type=map_type,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy network against ValueFunctionPlayer opponents."
    )
    # Model architecture (matches dagger.py)
    parser.add_argument(
        "--model-type",
        type=str,
        default="flat",
        choices=["flat", "hierarchical"],
        help="Policy head type: 'flat' or 'hierarchical'",
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="mlp",
        choices=["mlp", "xdim"],
        help="Backbone architecture: 'mlp' or 'xdim' (cross-dimensional with CNN)",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for policy backbone",
    )

    # Map and opponent configuration (matches dagger.py)
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Catan map type",
    )
    parser.add_argument(
        "--opponent-configs",
        type=str,
        nargs="+",
        default=["F"],
        help="Opponent player config strings (e.g., 'F' for ValueFunctionPlayer, 'random')",
    )
    parser.add_argument(
        "--actor-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="private",
        help=(
            "Information level used by the policy network: private, public "
            "(1v1 opponent resources), or full/privileged."
        ),
    )

    # Experiment-based loading (preferred): rebuilds the model from metadata.
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Experiment name (under experiments/) or path. When set, the model "
            "architecture and observation level are read from metadata.json and "
            "the architecture flags above are ignored."
        ),
    )
    parser.add_argument(
        "--which",
        type=str,
        default="best",
        help="Checkpoint selector for --experiment: 'best', 'latest', or a step.",
    )
    # Model weights (legacy path: requires matching architecture flags).
    parser.add_argument(
        "--policy-weights",
        type=str,
        default=None,
        help="Path to policy model weights (.pt file). Alternative to --experiment.",
    )

    # Evaluation settings
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--nn-seat",
        type=str,
        default="random",
        choices=["random", "first", "second"],
        help="Seat the NN player randomly, first, or second in turn order",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--vps-to-win",
        type=int,
        default=15,
        help="Victory points required to win each game (default: 15)",
    )
    parser.add_argument(
        "--discard-limit",
        type=int,
        default=9,
        help="Discard threshold when a 7 is rolled (default: 9)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log evaluation config and summary metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional Weights & Biases group name",
    )

    args = parser.parse_args()

    if (args.experiment is None) == (args.policy_weights is None):
        parser.error("Provide exactly one of --experiment or --policy-weights.")

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Compute number of players from opponent configs
    num_players = len(args.opponent_configs) + 1

    # Experiment path: rebuild model from metadata and adopt its architecture.
    experiment_model = None
    if args.experiment is not None:
        exp = load_experiment(args.experiment)
        args.model_type = exp.model_type or args.model_type
        args.map_type = exp.map_type
        args.backbone_type = backbone_display_type(exp.policy_spec.backbone)
        args.policy_hidden_dims = backbone_hidden_dims(exp.policy_spec.backbone)
        if exp.policy_spec.observation_level is not None:
            args.actor_observation_level = exp.policy_spec.observation_level
        if exp.num_players != num_players:
            parser.error(
                f"--opponent-configs implies {num_players} players but experiment "
                f"'{args.experiment}' was trained for {exp.num_players}."
            )
        experiment_model = exp.build_policy(which=args.which, device=device)

    print(f"\n{'=' * 60}")
    print("Policy Evaluation vs ValueFunctionPlayer")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Actor observation: {args.actor_observation_level}")
    print(f"Hidden dims: {args.policy_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Opponents: {args.opponent_configs}")
    print(f"NN seat: {args.nn_seat}")
    print(f"Games: {args.num_games}")
    print(f"VPs to win: {args.vps_to_win} | Discard limit: {args.discard_limit}")

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="eval",
            config=vars(args) | {"num_players": num_players},
        )

    if experiment_model is not None:
        model = experiment_model
        print(f"Loaded policy from experiment '{args.experiment}' ({args.which})")
    else:
        model = build_policy_model(
            backbone_type=args.backbone_type,
            model_type=args.model_type,
            hidden_dims=args.policy_hidden_dims,
            num_players=num_players,
            map_type=args.map_type,
            actor_observation_level=args.actor_observation_level,
            device=device,
        )
        state_dict = torch.load(args.policy_weights, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded policy weights from {args.policy_weights}")

    # Create opponents (ValueFunctionPlayer for each)
    opponents = [ValueFunctionPlayer(COLOR_ORDER[i + 1]) for i in range(len(args.opponent_configs))]

    # Create NN player
    nn_player = NNPolicyPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        model=model,
        map_type=args.map_type,
        actor_observation_level=args.actor_observation_level,
    )

    # Run evaluation
    print(f"\nRunning {args.num_games} games...")
    wins, vps, total_vps, turns = eval(
        nn_player,
        opponents,
        map_type=args.map_type,
        num_games=args.num_games,
        seed=args.seed,
        vps_to_win=args.vps_to_win,
        discard_limit=args.discard_limit,
        show_tqdm=True,
        nn_seat=args.nn_seat,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Wins: {wins} / {args.num_games} ({100 * wins / args.num_games:.1f}%)")
    print(f"Average VPs: {sum(vps) / len(vps):.2f}")
    print(f"Average Total VPs: {sum(total_vps) / len(total_vps):.2f}")
    print(f"Average Turns: {sum(turns) / len(turns):.1f}")

    if args.wandb:
        wandb.log(
            {
                "wins": wins,
                "win_rate": wins / args.num_games if args.num_games else 0.0,
                "avg_vps": sum(vps) / len(vps) if vps else 0.0,
                "avg_total_vps": sum(total_vps) / len(total_vps) if total_vps else 0.0,
                "avg_turns": sum(turns) / len(turns) if turns else 0.0,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
