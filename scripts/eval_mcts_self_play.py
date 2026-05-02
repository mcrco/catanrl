"""
Evaluate a neural-guided MCTS player in self-play.

Loads a policy network and a critic network, uses the critic as the MCTS value
network, and seats a separate NNMCTSPlayer instance at every player color.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal, Sequence

import torch
from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color
from catanatron.state_functions import get_actual_victory_points
from tqdm import tqdm

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
from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.catanatron_map import build_catan_map
from catanrl.utils.seeding import derive_map_and_game_seeds, derive_seed

BOARD_WIDTH = 21
BOARD_HEIGHT = 11


@dataclass
class PlayerStats:
    wins: int = 0
    vps: list[int] = field(default_factory=list)

    @property
    def avg_vps(self) -> float:
        return sum(self.vps) / len(self.vps) if self.vps else 0.0


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


def build_self_play_players(
    policy_model: PolicyNetworkWrapper,
    critic_model: ValueNetworkWrapper,
    model_type: str,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_players: int,
    num_simulations: int,
    c_puct: float,
    prunning: bool,
    critic_mode: str,
) -> list[NNMCTSPlayer]:
    return [
        NNMCTSPlayer(
            color=color,
            model_type=model_type,
            policy_model=policy_model,
            critic_model=critic_model,
            map_type=map_type,
            num_simulations=num_simulations,
            c_puct=c_puct,
            prunning=prunning,
            opponent_policy="self",
            critic_mode=critic_mode,
        )
        for color in COLOR_ORDER[:num_players]
    ]


def run_self_play_eval(
    players: list[NNMCTSPlayer],
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    num_games: int,
    seed: int,
    vps_to_win: int,
    discard_limit: int,
    show_tqdm: bool = True,
) -> tuple[dict[Color, PlayerStats], list[int]]:
    stats = {player.color: PlayerStats() for player in players}
    turns: list[int] = []
    episode_seeds = [derive_seed(seed, "self_play_episode", game_idx) for game_idx in range(num_games)]

    for episode_seed in tqdm(episode_seeds, disable=not show_tqdm):
        map_seed, game_seed = derive_map_and_game_seeds(episode_seed)
        for player in players:
            player.reset_state()

        game = Game(
            players=players,
            catan_map=build_catan_map(map_type, seed=map_seed, number_placement="random"),
            seed=game_seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
        )
        force_player_order(game, players)
        game.play()

        winning_color = game.winning_color()
        if winning_color in stats:
            stats[winning_color].wins += 1

        for color in game.state.colors:
            stats[color].vps.append(get_actual_victory_points(game.state, color))
        turns.append(game.state.num_turns)

    return stats, turns


def color_label(color: Color) -> str:
    return getattr(color, "value", str(color))


def force_player_order(game: Game, players: list[NNMCTSPlayer]) -> None:
    colors = tuple(player.color for player in players)
    game.state.players = list(players)
    game.state.colors = colors
    game.state.color_to_index = {color: idx for idx, color in enumerate(colors)}
    game.state.current_player_index = 0
    game.state.current_turn_index = 0
    game.playable_actions = generate_playable_actions(game.state)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a neural-guided MCTS policy/critic in self-play."
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
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of self-play players",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Catan map type",
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
        "--critic-mode",
        type=str,
        default="full",
        choices=["full", "guess"],
        help=(
            "Critic observation mode. 'full' uses privileged critic features; "
            "'guess' samples hidden opponent dev cards before scoring."
        ),
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=100,
        help="Number of self-play games to play",
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
        help="Victory points required to win each game",
    )
    parser.add_argument(
        "--discard-limit",
        type=int,
        default=9,
        help="Discard threshold when a 7 is rolled",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu). Defaults to cuda if available.",
    )

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n{'=' * 60}")
    print("Neural MCTS Self-Play Evaluation")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {args.num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Policy hidden dims: {args.policy_hidden_dims}")
    print(f"Critic hidden dims: {args.critic_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Critic weights: {args.critic_weights}")
    print(f"Critic mode: {args.critic_mode}")
    print(f"MCTS simulations: {args.num_simulations} | c_puct: {args.c_puct}")
    print(f"Prunning: {args.prunning}")
    print(f"Games: {args.num_games}")
    print(f"VPs to win: {args.vps_to_win} | Discard limit: {args.discard_limit}")

    policy_model = build_policy_model(
        backbone_type=args.backbone_type,
        model_type=args.model_type,
        hidden_dims=args.policy_hidden_dims,
        num_players=args.num_players,
        map_type=args.map_type,
        device=device,
    )
    critic_model = build_critic_model(
        backbone_type=args.backbone_type,
        hidden_dims=args.critic_hidden_dims,
        num_players=args.num_players,
        map_type=args.map_type,
        device=device,
    )

    policy_model.load_state_dict(torch.load(args.policy_weights, map_location=device))
    critic_model.load_state_dict(torch.load(args.critic_weights, map_location=device))
    policy_model.eval()
    critic_model.eval()
    print(f"Loaded policy weights from {args.policy_weights}")
    print(f"Loaded critic weights from {args.critic_weights}")

    players = build_self_play_players(
        policy_model=policy_model,
        critic_model=critic_model,
        model_type=args.model_type,
        map_type=args.map_type,
        num_players=args.num_players,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        prunning=args.prunning,
        critic_mode=args.critic_mode,
    )

    print(f"\nRunning {args.num_games} self-play games...")
    stats, turns = run_self_play_eval(
        players=players,
        map_type=args.map_type,
        num_games=args.num_games,
        seed=args.seed,
        vps_to_win=args.vps_to_win,
        discard_limit=args.discard_limit,
        show_tqdm=True,
    )

    print(f"\n{'=' * 60}")
    print("Results")
    print(f"{'=' * 60}")
    print(f"{'Player':<12} {'Wins':>8} {'Win Rate':>10} {'Avg VPs':>10}")
    for player in players:
        player_stats = stats[player.color]
        win_rate = player_stats.wins / args.num_games if args.num_games else 0.0
        print(
            f"{color_label(player.color):<12} "
            f"{player_stats.wins:>8} "
            f"{100 * win_rate:>9.1f}% "
            f"{player_stats.avg_vps:>10.2f}"
        )
    avg_turns = sum(turns) / len(turns) if turns else 0.0
    print(f"\nAverage Turns: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
