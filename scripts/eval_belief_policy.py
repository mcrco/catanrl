"""Evaluate a full-information policy under belief marginalization.

The idea ("test-time compute"): train a policy on full/perfect information, then
at decision time marginalize it over the belief about the opponent's hidden
development cards instead of feeding it partial information. In 1v1 the belief is
enumerated exactly, so the resulting policy is a fair, no-cheating player built
from a full-info network.

This script loads a full-info policy (from an experiment or weights), wraps it in
a BeliefAveragedPolicyPlayer, and evaluates it against ValueFunctionPlayer
opponents. Optionally it also evaluates the same network as a non-marginalized
oracle (which illegally sees the true opponent dev cards) as an upper bound.
"""

import argparse
import torch
import wandb
from catanatron.players.value import ValueFunctionPlayer

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
from catanrl.features.catanatron_utils import COLOR_ORDER
from catanrl.experiments.common_args import DEFAULT_WANDB_PROJECT
from catanrl.players import BeliefAveragedPolicyPlayer, NNPolicyPlayer
from catanrl.utils.seeding import derive_seed


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a full-info policy under belief marginalization."
    )
    parser.add_argument("--model-type", type=str, default="flat", choices=["flat", "hierarchical"])
    parser.add_argument("--backbone-type", type=str, default="xdim", choices=["mlp", "xdim"])
    parser.add_argument("--policy-hidden-dims", type=int, nargs="+", default=[512, 512])
    parser.add_argument(
        "--map-type", type=str, default="BASE", choices=["BASE", "MINI", "TOURNAMENT"]
    )
    parser.add_argument(
        "--opponent-configs",
        type=str,
        nargs="+",
        default=["F"],
        help="Opponent configs; only ValueFunctionPlayer ('F') is supported here.",
    )

    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--which", type=str, default="best")
    parser.add_argument("--policy-weights", type=str, default=None)

    parser.add_argument("--num-games", type=int, default=100)
    parser.add_argument(
        "--nn-seat",
        type=str,
        default="random",
        choices=["random", "first", "second", "both"],
        help=(
            "Seat the evaluated player randomly, first, second, or run a balanced "
            "first+second evaluation. --num-games is per seat for 'both'."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vps-to-win", type=int, default=15)
    parser.add_argument("--discard-limit", type=int, default=9)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample actions from the marginal instead of taking the argmax.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Belief samples for >2 players (ignored in 1v1, which is exact).",
    )
    parser.add_argument(
        "--also-oracle",
        action="store_true",
        help="Additionally evaluate the non-marginalized full-info policy (cheats; upper bound).",
    )
    parser.add_argument(
        "--allow-non-full",
        action="store_true",
        help="Permit a model whose training observation level is not 'full' (not recommended).",
    )
    parser.add_argument("--wandb", action="store_true", help="Log config and results to W&B.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)

    args = parser.parse_args()

    if args.num_games < 1:
        parser.error("--num-games must be at least 1")
    if (args.experiment is None) == (args.policy_weights is None):
        parser.error("Provide exactly one of --experiment or --policy-weights.")

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    num_players = len(args.opponent_configs) + 1
    observation_level: str = "full"

    experiment_model = None
    if args.experiment is not None:
        exp = load_experiment(args.experiment)
        args.model_type = exp.model_type or args.model_type
        args.map_type = exp.map_type
        args.backbone_type = backbone_display_type(exp.policy_spec.backbone)
        args.policy_hidden_dims = backbone_hidden_dims(exp.policy_spec.backbone)
        observation_level = exp.policy_spec.observation_level or "full"
        if exp.num_players != num_players:
            parser.error(
                f"--opponent-configs implies {num_players} players but experiment "
                f"'{args.experiment}' was trained for {exp.num_players}."
            )
        experiment_model = exp.build_policy(which=args.which, device=device)

    if observation_level != "full" and not args.allow_non_full:
        parser.error(
            f"Belief marginalization requires a full-info policy, but the model's "
            f"observation level is '{observation_level}'. Re-run with --allow-non-full "
            f"to override (the network input layout must still match the full vector)."
        )

    print(f"\n{'=' * 60}")
    print("Belief-Averaged Policy Evaluation vs ValueFunctionPlayer")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Map type: {args.map_type} | Players: {num_players}")
    print(f"Backbone: {args.backbone_type} | Model type: {args.model_type}")
    print(f"Training observation level: {observation_level}")
    print(f"Action selection: {'sample' if args.sample else 'argmax'}")
    print(f"NN seat: {args.nn_seat} | Games: {args.num_games}")
    print(f"VPs to win: {args.vps_to_win} | Discard limit: {args.discard_limit}")

    if experiment_model is not None:
        model = experiment_model
        print(f"Loaded policy from experiment '{args.experiment}' ({args.which})")
    else:
        # Reuse the builder from the sibling value-eval script (same scripts/
        # directory is on sys.path when this file is run directly).
        from eval_vs_value import build_policy_model

        model = build_policy_model(
            backbone_type=args.backbone_type,
            model_type=args.model_type,
            hidden_dims=args.policy_hidden_dims,
            num_players=num_players,
            map_type=args.map_type,
            actor_observation_level="full",
            device=device,
        )
        state_dict = torch.load(args.policy_weights, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded policy weights from {args.policy_weights}")

    def make_opponents():
        return [ValueFunctionPlayer(COLOR_ORDER[i + 1]) for i in range(len(args.opponent_configs))]

    belief_player = BeliefAveragedPolicyPlayer(
        color=COLOR_ORDER[0],
        model_type=args.model_type,
        model=model,
        map_type=args.map_type,
        num_samples=args.num_samples,
        sample=args.sample,
        seed=args.seed,
    )

    seat_modes = ("first", "second") if args.nn_seat == "both" else (args.nn_seat,)
    checkpoint = args.which if args.experiment else str(args.policy_weights)

    def evaluate_player(player, variant: str) -> dict[str, EvalResult]:
        seat_results: dict[str, EvalResult] = {}
        for seat_mode in seat_modes:
            print(f"\nRunning {args.num_games} games ({variant}, {seat_mode} seat)...")
            seat_results[seat_mode] = EvalResult.from_tuple(
                eval(
                    player,
                    make_opponents(),
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
        return seat_results

    rows = [
        summarize_eval_results(
            "belief-averaged", checkpoint, evaluate_player(belief_player, "belief")
        )
    ]

    if args.also_oracle:
        oracle_player = NNPolicyPlayer(
            color=COLOR_ORDER[0],
            model_type=args.model_type,
            model=model,
            map_type=args.map_type,
            actor_observation_level="full",
        )
        rows.append(
            summarize_eval_results(
                "full-info-oracle", checkpoint, evaluate_player(oracle_player, "oracle")
            )
        )

    print_eval_rows(rows)
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            job_type="eval",
            config=vars(args) | {"num_players": num_players},
        )
        log_wandb_eval_results(run, rows, wandb, chart_title="Belief-policy evaluation win rate")
        run.finish()


if __name__ == "__main__":
    main()
