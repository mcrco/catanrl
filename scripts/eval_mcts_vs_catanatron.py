"""
Evaluate a neural-guided MCTS player against configurable Catanatron opponents.

This script mirrors eval_vs_catanatron.py while loading both policy and critic
networks, then using them inside NNMCTSPlayer.
"""

import argparse
from typing import Literal, Sequence

import torch
import wandb
from catanatron.cli.cli_players import CLI_PLAYERS
from catanatron.models.player import Player

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.eval.eval_nn_vs_catanatron import eval
from catanrl.experiment_store import (
    KIND_POLICY_VALUE,
    backbone_display_type,
    backbone_hidden_dims,
    load_experiment,
)
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT
from catanrl.experiments.network_config import resolve_observation_network_args
from catanrl.features.catanatron_utils import (
    ActorObservationLevel,
    COLOR_ORDER,
    CriticObservationLevel,
)
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


def build_critic_model(
    backbone_type: str,
    hidden_dims: Sequence[int],
    num_players: int,
    map_type: Literal["BASE", "MINI", "TOURNAMENT"],
    critic_observation_level: CriticObservationLevel,
    device: torch.device,
) -> ValueNetworkWrapper:
    dims = compute_single_agent_dims(
        num_players,
        map_type,
        critic_observation_level=critic_observation_level,
    )
    critic_dim = dims["critic_dim"]
    critic_numeric_dim = dims["critic_numeric_dim"]
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
                numeric_dim=critic_numeric_dim,
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
            f"Too many opponents ({len(player_keys)}). Maximum supported is {len(COLOR_ORDER) - 1}."
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
    opponent_policy_choices = [
        "self",
        "random",
        "weighted",
        "value",
        "victorypoint",
        "alphabeta",
        "sameturnalphabeta",
        "mcts",
        "playouts",
    ]

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
        "--actor-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="private",
        help=(
            "Information level used by the policy network: private, public "
            "(1v1 opponent resources), or full/privileged."
        ),
    )
    parser.add_argument(
        "--critic-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="full",
        help=(
            "Information level used by the critic network: private, public "
            "(1v1 opponent resources), or full/privileged."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help=(
            "Experiment name (under experiments/) or path. When set, both policy "
            "and critic are rebuilt from metadata.json and the architecture flags "
            "above are ignored."
        ),
    )
    parser.add_argument(
        "--which",
        type=str,
        default="best",
        help="Checkpoint selector for --experiment: 'best', 'latest', or a step.",
    )
    parser.add_argument(
        "--policy-weights",
        type=str,
        default=None,
        help="Path to policy model weights (.pt file). Alternative to --experiment.",
    )
    parser.add_argument(
        "--critic-weights",
        type=str,
        default=None,
        help="Path to critic model weights (.pt file). Alternative to --experiment.",
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=128,
        help="MCTS simulations per move",
    )
    parser.add_argument(
        "--ismcts-determinizations",
        "--is-mcts-determinizations",
        dest="ismcts_determinizations",
        type=int,
        default=1,
        help=(
            "Information-Set MCTS: number of belief determinizations of opponents' "
            "hidden dev cards searched per move. 1 disables IS-MCTS (plain search); "
            ">1 requires actor/critic observation public (1v1) or full."
        ),
    )
    parser.add_argument(
        "--num-search-workers",
        type=int,
        default=1,
        help="Threaded MCTS search workers for the NN player. 1 preserves sequential search.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=32,
        help="Maximum leaf evaluations per batched NN inference step.",
    )
    parser.add_argument(
        "--inference-wait-ms",
        type=float,
        default=1.0,
        help="Maximum milliseconds the inference server waits to fill a batch.",
    )
    parser.add_argument(
        "--virtual-loss",
        type=float,
        default=1.0,
        help="Virtual loss applied during threaded tree traversal.",
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
        "--adversarial-policy",
        type=str,
        default="self",
        choices=opponent_policy_choices,
        help=(
            "Opponent policy used inside NNMCTS tree expansion for non-agent turns. "
            "'self' uses the NN policy/value for all players; other options delegate "
            "opponent turns to the selected Catanatron bot policy."
        ),
    )
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
        help="Seat the NN-guided player randomly, first, or second in turn order",
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

    use_experiment = args.experiment is not None
    if use_experiment:
        if args.policy_weights is not None or args.critic_weights is not None:
            parser.error("Do not combine --experiment with --policy-weights/--critic-weights.")
    elif args.policy_weights is None or args.critic_weights is None:
        parser.error("Provide --experiment, or both --policy-weights and --critic-weights.")

    opponents = parse_opponents(args.opponents)
    num_players = len(opponents) + 1

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Experiment path: rebuild policy + critic from metadata.
    experiment_policy = None
    experiment_critic = None
    if use_experiment:
        exp = load_experiment(args.experiment)
        args.model_type = exp.model_type or args.model_type
        args.map_type = exp.map_type
        args.backbone_type = backbone_display_type(exp.policy_spec.backbone)
        args.policy_hidden_dims = backbone_hidden_dims(exp.policy_spec.backbone)
        if exp.policy_spec.observation_level is not None:
            args.actor_observation_level = exp.policy_spec.observation_level
        critic_spec = exp.metadata.networks.get("critic")
        if critic_spec is not None:
            args.critic_hidden_dims = backbone_hidden_dims(critic_spec.backbone)
            if critic_spec.observation_level is not None:
                args.critic_observation_level = critic_spec.observation_level
        if exp.num_players != num_players:
            parser.error(
                f"--opponents implies {num_players} players but experiment "
                f"'{args.experiment}' was trained for {exp.num_players}."
            )
        if exp.policy_spec.kind == KIND_POLICY_VALUE:
            experiment_policy = exp.build_policy(
                which=args.which, device=device, as_policy_only=False
            )
            experiment_critic = None
        else:
            experiment_policy = exp.build_policy(which=args.which, device=device)
            experiment_critic = exp.build_critic(which=args.which, device=device)

    print(f"\n{'=' * 60}")
    print("Neural MCTS Evaluation vs Catanatron Bot")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(
        f"Actor observation: {args.actor_observation_level} | "
        f"Critic observation: {args.critic_observation_level}"
    )
    print(f"Policy hidden dims: {args.policy_hidden_dims}")
    print(f"Critic hidden dims: {args.critic_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Critic weights: {args.critic_weights}")
    print(f"MCTS simulations: {args.num_simulations} | c_puct: {args.c_puct}")
    print(f"IS-MCTS determinizations: {args.ismcts_determinizations}")
    print(
        "MCTS workers: "
        f"{args.num_search_workers} | inference batch: {args.inference_batch_size} "
        f"| wait: {args.inference_wait_ms} ms | virtual loss: {args.virtual_loss}"
    )
    print(f"Prunning: {args.prunning}")
    print(f"Adversarial policy (inside MCTS): {args.adversarial_policy}")
    print(f"Opponents: {args.opponents}")
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

    if use_experiment:
        policy_model = experiment_policy
        critic_model = experiment_critic
        print(f"Loaded policy + critic from experiment '{args.experiment}' ({args.which})")
    else:
        policy_model = build_policy_model(
            backbone_type=args.backbone_type,
            model_type=args.model_type,
            hidden_dims=args.policy_hidden_dims,
            num_players=num_players,
            map_type=args.map_type,
            actor_observation_level=args.actor_observation_level,
            device=device,
        )
        critic_model = build_critic_model(
            backbone_type=args.backbone_type,
            hidden_dims=args.critic_hidden_dims,
            num_players=num_players,
            map_type=args.map_type,
            critic_observation_level=args.critic_observation_level,
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

    resolve_observation_network_args(
        args,
        policy_mode_default=args.actor_observation_level,
        critic_mode_default=args.critic_observation_level,
    )

    nn_mcts_player = NNMCTSPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        policy_model=policy_model,
        critic_model=critic_model,
        map_type=args.map_type,
        num_simulations=args.num_simulations,
        ismcts_determinizations=args.ismcts_determinizations,
        c_puct=args.c_puct,
        prunning=args.prunning,
        opponent_policy=args.adversarial_policy,
        actor_observation_level=args.actor_observation_level,
        critic_observation_level=args.critic_observation_level,
        num_search_workers=args.num_search_workers,
        inference_batch_size=args.inference_batch_size,
        inference_wait_ms=args.inference_wait_ms,
        virtual_loss=args.virtual_loss,
    )

    print(f"\nRunning {args.num_games} games...")
    wins, vps, total_vps, turns = eval(
        nn_mcts_player,
        opponents,
        map_type=args.map_type,
        num_games=args.num_games,
        seed=args.seed,
        vps_to_win=args.vps_to_win,
        discard_limit=args.discard_limit,
        show_tqdm=True,
        nn_seat=args.nn_seat,
    )

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
