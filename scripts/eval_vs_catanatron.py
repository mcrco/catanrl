"""
Evaluate a trained policy network against configurable Catanatron opponents.

This script uses the same model configuration parameters as dagger.py,
allowing plug-and-play evaluation of DAgger-trained models.
"""

import argparse
from typing import Literal, Sequence

import torch
from catanatron.cli.cli_players import CLI_PLAYERS
from catanatron.models.player import Player

from catanrl.eval.eval_nn_vs_catanatron import eval
from catanrl.envs.gym.single_env import compute_single_agent_dims
from catanrl.features.catanatron_utils import COLOR_ORDER
from catanrl.models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import build_flat_policy_network, build_hierarchical_policy_network
from catanrl.models.wrappers import PolicyNetworkWrapper
from catanrl.players import NNPolicyPlayer


BOARD_WIDTH = 21
BOARD_HEIGHT = 11


def build_policy_model(
    backbone_type: str,
    model_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    device: torch.device,
) -> PolicyNetworkWrapper:
    """Build a policy network with the same configuration as dagger.py."""
    dims = compute_single_agent_dims(num_players, map_type)
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
        model = build_flat_policy_network(backbone_config=backbone_config)
    elif model_type == "hierarchical":
        model = build_hierarchical_policy_network(backbone_config=backbone_config)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return model.to(device)


def parse_opponents(opponents_str: str) -> list[Player]:
    """Parse catanatron-play style player config string, e.g. 'F,AB:2,R'."""
    if not opponents_str.strip():
        raise ValueError("--opponents cannot be empty")

    player_keys = [key.strip() for key in opponents_str.split(",") if key.strip()]
    if not player_keys:
        raise ValueError("--opponents cannot be empty")

    if len(player_keys) > len(COLOR_ORDER) - 1:
        raise ValueError(
            f"Too many opponents ({len(player_keys)}). "
            f"Maximum supported is {len(COLOR_ORDER) - 1}."
        )

    players: list[Player] = []
    cli_players_by_code = {cli_player.code: cli_player for cli_player in CLI_PLAYERS}
    supported_codes = ", ".join(sorted(cli_players_by_code.keys()))

    for i, key in enumerate(player_keys):
        parts = key.split(":")
        code = parts[0]
        cli_player = cli_players_by_code.get(code)
        if cli_player is None:
            raise ValueError(f"Unknown opponent code '{code}'. Available codes: {supported_codes}")
        params = [COLOR_ORDER[i + 1]] + parts[1:]
        players.append(cli_player.import_fn(*params))

    return players


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained policy network against Catanatron opponents."
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

    # Map and opponent configuration
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Catan map type",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="F",
        help=(
            "Comma-separated catanatron-play style opponent configs. "
            "Examples: 'F', 'AB:2', 'F,AB:2,R'"
        ),
    )

    # Model weights
    parser.add_argument(
        "--policy-weights",
        type=str,
        required=True,
        help="Path to policy model weights (.pt file)",
    )

    # Evaluation settings
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of games to play",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )

    args = parser.parse_args()

    opponents = parse_opponents(args.opponents)
    num_players = len(opponents) + 1

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'=' * 60}")
    print("Policy Evaluation vs Catanatron Bot")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Hidden dims: {args.policy_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Opponents: {args.opponents}")
    print(f"Games: {args.num_games}")

    # Build model
    model = build_policy_model(
        backbone_type=args.backbone_type,
        model_type=args.model_type,
        hidden_dims=args.policy_hidden_dims,
        num_players=num_players,
        map_type=args.map_type,
        device=device,
    )

    # Load weights
    state_dict = torch.load(args.policy_weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded policy weights from {args.policy_weights}")

    # Create NN player
    nn_player = NNPolicyPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        model=model,
        map_type=args.map_type,
    )

    # Run evaluation
    print(f"\nRunning {args.num_games} games...")
    wins, vps, total_vps, turns = eval(
        nn_player,
        opponents,
        map_type=args.map_type,
        num_games=args.num_games,
        seed=args.seed,
        show_tqdm=True,
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Wins: {wins} / {args.num_games} ({100 * wins / args.num_games:.1f}%)")
    print(f"Average VPs: {sum(vps) / len(vps):.2f}")
    print(f"Average Total VPs: {sum(total_vps) / len(total_vps):.2f}")
    print(f"Average Turns: {sum(turns) / len(turns):.1f}")


if __name__ == "__main__":
    main()
