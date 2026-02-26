import argparse
import os
import wandb
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE, COLOR_ORDER
from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.models.player import RandomPlayer, Color

from ..features.catanatron_utils import game_to_features
from ..envs.gym.single_env import create_opponents
from ..algorithms.ppo.sarl_ppo import train


def main():
    parser = argparse.ArgumentParser(
        description="Train Catan Policy-Value Network with Self-Play RL (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["flat", "hierarchical"],
        default="hierarchical",
        help="Model architecture type: flat or hierarchical (default: flat)",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to pre-trained weights from supervised learning",
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to train (default: 1000)"
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=4096,
        help="Update model every N rollout steps (default: 32)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)"
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)"
    )
    parser.add_argument(
        "--value-coef", type=float, default=0.5, help="Value loss coefficient (default: 0.5)"
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.01, help="Entropy coefficient (default: 0.01)"
    )
    parser.add_argument(
        "--activity-coef",
        type=float,
        default=0.0,
        help="Activity regularization coefficient on raw policy logits (default: 0.0)",
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Number of PPO epochs per update (default: 10)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for PPO update (default: 64)"
    )
    parser.add_argument(
        "--hidden-dims", type=str, default="512,512", help="Hidden dimensions (default: 512,512)"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save model (default: weights/{wandb-run-name}/best.pt if using wandb, else weights/policy_value_rl_best.pt)",
    )
    parser.add_argument(
        "--save-every-updates",
        type=int,
        default=1,
        help="Save policy snapshot every N PPO updates (default: 1)",
    )
    parser.add_argument(
        "--map-type",
        type=str,
        default="BASE",
        choices=["BASE", "MINI", "TOURNAMENT"],
        help="Map type to use (default: BASE)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random"],
        help="""Opponent bot types (1-3 opponents for 2-4 player game). Options:
                - random, R: RandomPlayer
                - weighted, W: WeightedRandomPlayer  
                - value, F: ValueFunctionPlayer
                - victorypoint, VP: VictoryPointPlayer
                - alphabeta, AB: AlphaBetaPlayer (use AB:depth:prunning for custom params)
                - sameturnalphabeta, SAB: SameTurnAlphaBetaPlayer
                - mcts, M: MCTSPlayer (use M:num_sims for custom simulations)
                - playouts, G: GreedyPlayoutsPlayer (use G:num_playouts for custom playouts)
                Examples: --opponents random random random
                            --opponents F W AB
                            --opponents AB:3:False M:500
                (default: random)""",
    )
    parser.add_argument(
        "--reward",
        type=str,
        default="shaped",
        choices=["shaped", "win"],
        help="Reward function type (default: shaped)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="catan-rl",
        help="Wandb project name (default: catan-rl)",
    )
    parser.add_argument(
        "--wandb-run-name", type=str, default=None, help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments for vectorized training (default: 4)",
    )
    parser.add_argument(
        "--use-lr-scheduler",
        action="store_true",
        help="Enable linear learning rate scheduling (default: False)",
    )
    parser.add_argument(
        "--lr-scheduler-start-factor",
        type=float,
        default=1.0,
        help="Start factor for LinearLR scheduler (default: 1.0)",
    )
    parser.add_argument(
        "--lr-scheduler-end-factor",
        type=float,
        default=0.0,
        help="End factor for LinearLR scheduler (default: 0.0)",
    )
    parser.add_argument(
        "--lr-scheduler-total-iters",
        type=int,
        default=None,
        help="Total iterations for LinearLR scheduler (default: n_episodes)",
    )
    parser.add_argument(
        "--metric-window",
        type=int,
        default=200,
        help="Window size for metrics (avg reward and length), also used for best model saving (default: 200)",
    )
    parser.add_argument(
        "--eval-games-per-opponent",
        type=int,
        default=0,
        help="Fresh evaluation games per opponent at each eval point; 0 disables eval (default: 0)",
    )
    parser.add_argument(
        "--trend-eval-games-per-opponent",
        type=int,
        default=None,
        help="Fixed-seed trend evaluation games per opponent (default: use eval-games-per-opponent)",
    )
    parser.add_argument(
        "--trend-eval-seed",
        type=int,
        default=42,
        help="Seed for trend evaluation runs (default: 42)",
    )
    parser.add_argument(
        "--eval-every-updates",
        type=int,
        default=0,
        help="Run evaluation every N PPO updates; 0 disables eval (default: 0)",
    )
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use deterministic (argmax) policy during rollout collection",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping norm for PPO updates (default: 0.5)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Optional KL threshold for PPO early stopping (default: disabled)",
    )

    args = parser.parse_args()

    # Check if weights file exists
    if args.load_weights and not os.path.exists(args.load_weights):
        print(f"Error: Weights file '{args.load_weights}' not found!")
        print("Please train a model first")
        return

    # Set default save path
    if args.save_path is None:
        if args.wandb and args.wandb_run_name:
            # Use wandb run name as directory
            args.save_path = f"weights/{args.wandb_run_name}"
        else:
            # Default fallback
            args.save_path = "weights/sarl_ppo"

    # Create save directory
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory '{save_dir}'")
        os.makedirs(save_dir, exist_ok=True)

    # Create opponents first to determine number of players
    if not args.opponents:
        args.opponents = ["random"]
    temp_opponents = create_opponents(args.opponents)
    num_players = len(temp_opponents) + 1  # +1 for the RL agent (BLUE)

    # Calculate input dimension based on actual number of players

    # Create dummy players matching the actual game setup
    dummy_players = [RandomPlayer(color) for color in COLOR_ORDER[:num_players]]
    dummy_game = Game(dummy_players, catan_map=build_map(args.map_type))
    dummy_tensor = game_to_features(dummy_game, Color.RED, num_players, args.map_type)
    input_dim = dummy_tensor.shape[0]
    print(f"Number of players: {num_players}")
    print(f"Input dimension: {input_dim}")

    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]

    # Prepare LR scheduler kwargs
    lr_scheduler_kwargs = None
    if args.use_lr_scheduler:
        lr_scheduler_kwargs = {
            "start_factor": args.lr_scheduler_start_factor,
            "end_factor": args.lr_scheduler_end_factor,
        }
        if args.lr_scheduler_total_iters is not None:
            lr_scheduler_kwargs["total_iters"] = args.lr_scheduler_total_iters

    # Prepare wandb config
    wandb_config = None
    if args.wandb:
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": {
                "algorithm": "PPO",
                "episodes": args.episodes,
                "rollout_steps": args.rollout_steps,
                "reward_function": args.reward,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_epsilon": args.clip_epsilon,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "activity_coef": args.activity_coef,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "hidden_dims": hidden_dims,
                "map_type": args.map_type,
                "load_weights": args.load_weights,
                "opponents": args.opponents,
                "num_envs": args.num_envs,
                "use_lr_scheduler": args.use_lr_scheduler,
                "lr_scheduler_kwargs": lr_scheduler_kwargs,
                "eval_games_per_opponent": args.eval_games_per_opponent,
                "trend_eval_games_per_opponent": args.trend_eval_games_per_opponent,
                "trend_eval_seed": args.trend_eval_seed,
                "eval_every_updates": args.eval_every_updates,
                "save_every_updates": args.save_every_updates,
                "deterministic_policy": args.deterministic_policy,
                "max_grad_norm": args.max_grad_norm,
                "target_kl": args.target_kl,
            },
        }

    # Train model
    train(
        input_dim=input_dim,
        num_actions=ACTION_SPACE_SIZE,
        model_type=args.model_type,
        hidden_dims=hidden_dims,
        load_weights=args.load_weights,
        n_episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        activity_coef=args.activity_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        save_every_updates=args.save_every_updates,
        wandb_config=wandb_config,
        map_type=args.map_type,
        opponent_configs=args.opponents,
        reward_function=args.reward,
        num_envs=args.num_envs,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        metric_window=args.metric_window,
        eval_games_per_opponent=args.eval_games_per_opponent,
        trend_eval_games_per_opponent=args.trend_eval_games_per_opponent,
        trend_eval_seed=args.trend_eval_seed,
        eval_every_updates=args.eval_every_updates,
        deterministic_policy=args.deterministic_policy,
        max_grad_norm=args.max_grad_norm,
        target_kl=args.target_kl,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved: {args.save_path}")

    # Finish wandb
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
