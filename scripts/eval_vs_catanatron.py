"""
Evaluate a trained policy network against configurable Catanatron opponents.

This script uses the same model configuration parameters as dagger.py,
allowing plug-and-play evaluation of DAgger-trained models.
"""

import argparse
from typing import Literal, Sequence

import torch
import wandb
from catanatron.cli.cli_players import CLI_PLAYERS
from catanatron.models.player import Player

from catanrl.envs.puffer.common import compute_single_agent_dims
from catanrl.eval.eval_nn_vs_catanatron import eval
from catanrl.eval.reporting import (
    EvalResult,
    log_wandb_eval_results,
    print_eval_rows,
    summarize_eval_results,
)
from catanrl.experiment_store import (
    backbone_display_type,
    backbone_hidden_dims,
    load_experiment,
)
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT
from catanrl.features.catanatron_utils import ActorObservationLevel, COLOR_ORDER
from catanrl.models.backbones import (
    BackboneConfig,
    CrossDimensionalBackboneConfig,
    MLPBackboneConfig,
)
from catanrl.models.models import build_flat_policy_network, build_hierarchical_policy_network
from catanrl.models.wrappers import PolicyNetworkWrapper
from catanrl.players import NNPolicyPlayer
from catanrl.utils.catanatron_action_space import get_action_space_size
from catanrl.utils.seeding import derive_seed


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


def parse_opponents(opponents_str: str) -> list[Player]:
    """Parse catanatron-play style player config string, e.g. 'F,AB:2,R'."""
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
    parser.add_argument(
        "--actor-observation-level",
        type=str,
        choices=["private", "public", "full"],
        default="private",
        help=(
            "Information level used by the policy network: private, public "
            "(1v1 opponent resources), or full."
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
        choices=["random", "first", "second", "both"],
        help=(
            "Seat the NN player randomly, first, second, or run a balanced "
            "first+second evaluation. --num-games is per seat for 'both'."
        ),
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

    if args.num_games < 1:
        parser.error("--num-games must be at least 1")
    if (args.experiment is None) == (args.policy_weights is None):
        parser.error("Provide exactly one of --experiment or --policy-weights.")

    opponents = parse_opponents(args.opponents)
    num_players = len(opponents) + 1

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

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
                f"--opponents implies {num_players} players but experiment "
                f"'{args.experiment}' was trained for {exp.num_players}."
            )
        experiment_model = exp.build_policy(which=args.which, device=device)

    print(f"\n{'=' * 60}")
    print("Policy Evaluation vs Catanatron Bot")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Actor observation: {args.actor_observation_level}")
    print(f"Hidden dims: {args.policy_hidden_dims}")
    print(f"Policy weights: {args.policy_weights}")
    print(f"Opponents: {args.opponents}")
    print(f"NN seat: {args.nn_seat}")
    print(f"Games: {args.num_games}")
    print(f"VPs to win: {args.vps_to_win} | Discard limit: {args.discard_limit}")

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

    # Create NN player
    nn_player = NNPolicyPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        model=model,
        map_type=args.map_type,
        actor_observation_level=args.actor_observation_level,
    )

    # Run evaluation
    seat_modes = ("first", "second") if args.nn_seat == "both" else (args.nn_seat,)
    seat_results: dict[str, EvalResult] = {}
    for seat_mode in seat_modes:
        print(f"\nRunning {args.num_games} games ({seat_mode} seat)...")
        seat_results[seat_mode] = EvalResult.from_tuple(
            eval(
                nn_player,
                opponents,
                map_type=args.map_type,
                num_games=args.num_games,
                seed=(
                    derive_seed(args.seed, "seat", seat_mode)
                    if args.nn_seat == "both"
                    else args.seed
                ),
                vps_to_win=args.vps_to_win,
                discard_limit=args.discard_limit,
                show_tqdm=True,
                nn_seat=seat_mode,
            )
        )

    label = args.experiment or "policy"
    checkpoint = args.which if args.experiment else str(args.policy_weights)
    rows = [summarize_eval_results(label, checkpoint, seat_results)]
    print_eval_rows(rows)

    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="eval",
            config=vars(args) | {"num_players": num_players},
        )
        log_wandb_eval_results(run, rows, wandb, chart_title="Policy win rate vs Catanatron")
        run.finish()


if __name__ == "__main__":
    main()
