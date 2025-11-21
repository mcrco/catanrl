import argparse
import os
import wandb
from catanrl.algorithms.ppo.marl_ippo import train


def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-agent self-play PPO policy using the PettingZoo Catanatron env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--map-type", type=str, default="BASE", choices=["BASE", "MINI"])
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--representation", type=str, default="vector", choices=["vector", "mixed"])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dims", type=str, default="512,512")
    parser.add_argument("--save-path", type=str, default="weights/policy_value_marl.pt")
    parser.add_argument("--save-freq", type=int, default=50)
    parser.add_argument("--load-weights", type=str, default=None)
    parser.add_argument("--invalid-action-reward", type=float, default=-1.0)
    parser.add_argument("--max-invalid-actions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="catan-marl")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--deterministic-policy",
        action="store_true",
        help="Use greedy action selection during data collection (default: sample)",
    )

    args = parser.parse_args()

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir and not os.path.exists(save_dir):
            print(f"Creating directory {save_dir}")
            os.makedirs(save_dir, exist_ok=True)

    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",") if dim.strip()]

    wandb_config = None
    if args.wandb:
        config_dict = {
            "episodes": args.episodes,
            "rollout_steps": args.rollout_steps,
            "lr": args.lr,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_epsilon": args.clip_epsilon,
            "value_coef": args.value_coef,
            "entropy_coef": args.entropy_coef,
            "ppo_epochs": args.ppo_epochs,
            "batch_size": args.batch_size,
            "hidden_dims": hidden_dims,
            "num_players": args.num_players,
            "map_type": args.map_type,
            "representation": args.representation,
            "invalid_action_reward": args.invalid_action_reward,
            "max_invalid_actions": args.max_invalid_actions,
        }
        wandb_config = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": config_dict,
        }

    train(
        num_players=args.num_players,
        map_type=args.map_type,
        vps_to_win=args.vps_to_win,
        representation=args.representation,
        episodes=args.episodes,
        rollout_steps=args.rollout_steps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        hidden_dims=hidden_dims,
        save_path=args.save_path,
        save_freq=args.save_freq,
        load_weights=args.load_weights,
        wandb_config=wandb_config,
        invalid_action_reward=args.invalid_action_reward,
        max_invalid_actions=args.max_invalid_actions,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        deterministic_policy=args.deterministic_policy,
    )

    print("\n" + "=" * 60)
    print("Multi-agent training complete!")
    print("=" * 60)
    if args.save_path:
        print(f"Model saved to: {args.save_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()