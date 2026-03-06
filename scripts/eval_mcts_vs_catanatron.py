"""
Evaluate a neural-guided MCTS player against configurable Catanatron opponents.

This script mirrors eval_vs_catanatron.py while loading both policy and critic
networks, then using them inside NNMCTSPlayer.
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
from catanrl.models.models import (
    build_flat_policy_network,
    build_hierarchical_policy_network,
    build_value_network,
)
from catanrl.models.wrappers import PolicyNetworkWrapper, ValueNetworkWrapper
from catanrl.players import NNMCTSPlayer


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


def build_critic_model(
    backbone_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    device: torch.device,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(num_players, map_type)
    critic_dim = dims["critic_dim"]
    full_numeric_dim = dims["full_numeric_dim"]
    board_channels = dims["board_channels"]

    if backbone_type == "mlp":
        backbone_config = BackboneConfig(
            architecture="mlp",
            args=MLPBackboneConfig(input_dim=critic_dim, hidden_dims=list(hidden_dims)),
        )
    elif backbone_type == "xdim":
        backbone_config = BackboneConfig(
            architecture="cross_dimensional",
            args=CrossDimensionalBackboneConfig(
                board_height=BOARD_HEIGHT,
                board_width=BOARD_WIDTH,
                board_channels=board_channels,
                numeric_dim=full_numeric_dim,
                cnn_channels=[64, 128, 128],
                cnn_kernel_size=(3, 5),
                numeric_hidden_dims=list(hidden_dims),
                fusion_hidden_dim=hidden_dims[-1] if hidden_dims else 256,
                output_dim=hidden_dims[-1] if hidden_dims else 256,
            ),
        )
    else:
        raise ValueError(f"Unknown backbone_type '{backbone_type}'")

    model = build_value_network(backbone_config=backbone_config)
    return model.to(device)


def parse_opponents(opponents_str: str) -> list[Player]:
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
        description="Evaluate a neural-guided MCTS player against Catanatron opponents."
    )

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
        help="Backbone architecture for both policy and critic",
    )
    parser.add_argument(
        "--policy-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for policy backbone",
    )
    parser.add_argument(
        "--critic-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for critic backbone",
    )
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
    parser.add_argument(
        "--policy-weights",
        type=str,
        required=True,
        help="Path to policy model weights (.pt file)",
    )
    parser.add_argument(
        "--critic-weights",
        type=str,
        required=True,
        help="Path to critic model weights (.pt file)",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=128,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant",
    )
    parser.add_argument(
        "--prunning",
        action="store_true",
        help="Enable catanatron action prunning heuristics",
    )
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

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'=' * 60}")
    print("Neural MCTS Evaluation vs Catanatron Bot")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Policy hidden dims: {args.policy_hidden_dims}")
    print(f"Critic hidden dims: {args.critic_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Critic weights: {args.critic_weights}")
    print(f"MCTS simulations: {args.num_simulations} | c_puct: {args.c_puct}")
    print(f"Prunning: {args.prunning}")
    print(f"Opponents: {args.opponents}")
    print(f"Games: {args.num_games}")

    policy_model = build_policy_model(
        backbone_type=args.backbone_type,
        model_type=args.model_type,
        hidden_dims=args.policy_hidden_dims,
        num_players=num_players,
        map_type=args.map_type,
        device=device,
    )
    critic_model = build_critic_model(
        backbone_type=args.backbone_type,
        hidden_dims=args.critic_hidden_dims,
        num_players=num_players,
        map_type=args.map_type,
        device=device,
    )

    policy_state = torch.load(args.policy_weights, map_location=device)
    critic_state = torch.load(args.critic_weights, map_location=device)
    policy_model.load_state_dict(policy_state)
    critic_model.load_state_dict(critic_state)
    policy_model.eval()
    critic_model.eval()
    print(f"Loaded policy weights from {args.policy_weights}")
    print(f"Loaded critic weights from {args.critic_weights}")

    nn_mcts_player = NNMCTSPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        policy_model=policy_model,
        critic_model=critic_model,
        map_type=args.map_type,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        prunning=args.prunning,
    )

    print(f"\nRunning {args.num_games} games...")
    wins, vps, total_vps, turns = eval(
        nn_mcts_player,
        opponents,
        map_type=args.map_type,
        num_games=args.num_games,
        seed=args.seed,
        show_tqdm=True,
    )

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"Wins: {wins} / {args.num_games} ({100 * wins / args.num_games:.1f}%)")
    print(f"Average VPs: {sum(vps) / len(vps):.2f}")
    print(f"Average Total VPs: {sum(total_vps) / len(total_vps):.2f}")
    print(f"Average Turns: {sum(turns) / len(turns):.1f}")


if __name__ == "__main__":
    main()
