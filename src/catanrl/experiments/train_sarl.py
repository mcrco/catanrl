import argparse
import os
import wandb
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from catanatron.game import Game
from catanatron.models.map import build_map
from catanatron.features import get_feature_ordering
from catanatron.gym.board_tensor_features import create_board_tensor, is_graph_feature
from catanatron.models.player import RandomPlayer, Color

from ..envs.single_env import create_opponents
from ..training.sarl import train

def main():
    parser = argparse.ArgumentParser(
        description="Train Catan Policy-Value Network with Self-Play RL (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--update-freq", type=int, default=32, help="Update model every N episodes (default: 32)"
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
        "--n-epochs", type=int, default=4, help="Number of PPO epochs per update (default: 4)"
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
        "--save-freq", type=int, default=100, help="Save checkpoint every N episodes (default: 100)"
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
            args.save_path = f"weights/{args.wandb_run_name}/best.pt"
        else:
            # Default fallback
            args.save_path = "weights/policy_value_rl_best.pt"

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
    colors_list = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    dummy_players = [RandomPlayer(colors_list[i]) for i in range(num_players)]
    dummy_game = Game(dummy_players, catan_map=build_map(args.map_type))
    # For vectorized training we use 'mixed' observation: numeric (non-graph) + board tensor
    all_features = get_feature_ordering(num_players, args.map_type)
    numeric_len = len([f for f in all_features if not is_graph_feature(f)])
    board_tensor = create_board_tensor(dummy_game, dummy_game.state.colors[0])
    input_dim = numeric_len + board_tensor.size
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
                "update_freq": args.update_freq,
                "lr": args.lr,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "clip_epsilon": args.clip_epsilon,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "n_epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "hidden_dims": hidden_dims,
                "map_type": args.map_type,
                "load_weights": args.load_weights,
                "opponents": args.opponents,
                "num_envs": args.num_envs,
                "use_lr_scheduler": args.use_lr_scheduler,
                "lr_scheduler_kwargs": lr_scheduler_kwargs,
            },
        }

    # Train model
    train(
        input_dim=input_dim,
        num_actions=ACTION_SPACE_SIZE,
        hidden_dims=hidden_dims,
        load_weights=args.load_weights,
        n_episodes=args.episodes,
        update_freq=args.update_freq,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        save_freq=args.save_freq,
        wandb_config=wandb_config,
        map_type=args.map_type,
        opponent_configs=args.opponents,
        num_envs=args.num_envs,
        use_lr_scheduler=args.use_lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
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