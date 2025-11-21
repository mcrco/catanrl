import argparse
import os
import wandb
from catanrl.algorithms.supervised.sl import train
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE

def main():
    parser = argparse.ArgumentParser(
        description='Train Catan Joint Policy-Value Network (Flat or Hierarchical)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['flat', 'hierarchical'],
        default='flat',
        help='Model architecture type: flat or hierarchical (default: flat)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel data loading workers (default: 4)'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=32768,
        help='Shuffle buffer size - larger = more random but more memory (default: 32768)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=str,
        default='512,512',
        help='Hidden dimensions for the shared backbone (default: 512,512)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='weights/policy_value_network_best.pt',
        help='Path to save model (default: weights/policy_value_network_best.pt)'
    )
    parser.add_argument(
        '--load-weights',
        type=str,
        default=None,
        help='Path to pre-trained weights to continue training from (default: None)'
    )
    parser.add_argument(
        '--policy-loss-weight',
        type=float,
        default=1.0,
        help='Weight for policy loss in combined loss (default: 1.0)'
    )
    parser.add_argument(
        '--value-loss-weight',
        type=float,
        default=1.0,
        help='Weight for value loss in combined loss (default: 1.0)'
    )
    parser.add_argument(
        '--action-type-weight',
        type=float,
        default=1.0,
        help='Weight for action type loss (hierarchical only, default: 1.0)'
    )
    parser.add_argument(
        '--param-weight',
        type=float,
        default=1.0,
        help='Weight for parameter loss (hierarchical only, default: 1.0)'
    )
    parser.add_argument(
        '--use-class-weights',
        action='store_true',
        help='Use class weights to handle imbalanced actions (downweight ROLL/END_TURN)'
    )
    parser.add_argument(
        '--weight-power',
        type=float,
        default=0.5,
        help='Power for inverse frequency weighting (0.5=sqrt, 1.0=inverse, default: 0.5)'
    )
    parser.add_argument(
        '--weight-sample-fraction',
        type=float,
        default=1.0,
        help='Fraction of data to sample for class weight computation (1.0=all, 0.1=10%% for 10x speedup, default: 1.0)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='catan',
        help='Wandb project name (default: catan)'
    )
    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Wandb run name (default: auto-generated)'
    )
    parser.add_argument(
        '--log-batch-freq',
        type=int,
        default=1,
        help='Log batch metrics every N batches (default: 1)'
    )
    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found!")
        print("Please ensure the directory exists and contains .parquet files")
        return

    # Create save directory if needed
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory '{save_dir}'")
        os.makedirs(save_dir, exist_ok=True)

    # Auto-set save path based on run name
    if args.wandb and args.wandb_run_name and args.save_path == 'weights/policy_value_network_best.pt':
        args.save_path = f"weights/{args.wandb_run_name}_policy_value.pt"

    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]

    # Prepare wandb config (will be initialized inside train function)
    wandb_config = None
    if args.wandb:
        config_dict = {
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'hidden_dims': hidden_dims,
            'learning_rate': args.lr,
            'num_workers': args.num_workers,
            'buffer_size': args.buffer_size,
            'action_space_size': ACTION_SPACE_SIZE,
            'log_batch_freq': args.log_batch_freq,
            'policy_loss_weight': args.policy_loss_weight,
            'value_loss_weight': args.value_loss_weight,
            'use_class_weights': args.use_class_weights,
        }
        if args.model_type == 'hierarchical':
            config_dict['action_type_weight'] = args.action_type_weight
            config_dict['param_weight'] = args.param_weight
        if args.use_class_weights:
            config_dict['weight_power'] = args.weight_power
            config_dict['weight_sample_fraction'] = args.weight_sample_fraction
        
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_run_name,
            'config': config_dict
        }

    # Train model
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=hidden_dims,
        save_path=args.save_path,
        log_batch_freq=args.log_batch_freq,
        num_workers=args.num_workers,
        buffer_size=args.buffer_size,
        wandb_config=wandb_config,
        load_weights=args.load_weights,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        model_type=args.model_type,
        action_type_weight=args.action_type_weight,
        param_weight=args.param_weight,
        use_class_weights=args.use_class_weights,
        weight_power=args.weight_power,
        weight_sample_fraction=args.weight_sample_fraction,
    )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved: {args.save_path}")

    # Finish wandb
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()