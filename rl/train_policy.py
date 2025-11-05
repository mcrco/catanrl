#!/usr/bin/env python3
"""
Training script for Catan Policy Network.
Run: python train_policy.py --data-dir <path>
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from data import CatanDataset, InterleavedFileLoader
from models import PolicyNetwork


def train_policy_network(
    dataset,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'policy_network_best.pt',
    log_batch_freq: int = 10,
    num_files_loaded: int = 4,
):
    """Train the policy network to predict actions (behavior cloning)."""
    
    print(f"\n{'='*60}")
    print("Training Policy Network")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create model
    input_dim = len(dataset.feature_cols)
    num_actions = ACTION_SPACE_SIZE
    model = PolicyNetwork(input_dim, num_actions).to(device)
    
    # Split data into train/val by file count (90/10 split)
    num_train_files = int(0.9 * len(dataset.parquet_files))
    train_indices = list(range(num_train_files))
    val_indices = list(range(num_train_files, len(dataset.parquet_files)))
    
    print(f"Train: {len(train_indices)} files, Val: {len(val_indices)} files")
    
    # Create loaders with file subsets
    train_loader = InterleavedFileLoader(
        dataset,
        batch_size=batch_size,
        num_files_loaded=num_files_loaded,
        drop_last=True,
        file_indices=train_indices
    )
    
    val_loader = InterleavedFileLoader(
        dataset,
        batch_size=batch_size,
        num_files_loaded=min(num_files_loaded, len(val_indices)),
        drop_last=False,
        file_indices=val_indices
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Start epoch: shuffle files and load initial pool
        train_loader.start_epoch()
        
        num_train_batches = 0
        for batch_idx, (features, actions, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            features = features.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            num_train_batches += 1
            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == actions).sum().item()
            batch_total = actions.size(0)
            train_correct += batch_correct
            train_total += batch_total
            
            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                batch_acc = 100 * batch_correct / batch_total
                wandb.log({
                    'policy/batch_loss': batch_loss,
                    'policy/batch_acc': batch_acc,
                    'policy/batch_step': global_step,
                }, step=global_step)
            
            global_step += 1
        
        train_loss /= num_train_batches
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Start validation epoch
        val_loader.start_epoch()
        
        num_val_batches = 0
        with torch.no_grad():
            for features, actions, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                features = features.to(device)
                actions = actions.to(device)
                
                logits = model(features)
                loss = criterion(logits, actions)
                
                val_loss += loss.item()
                num_val_batches += 1
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == actions).sum().item()
                val_total += actions.size(0)
        
        val_loss /= num_val_batches
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log epoch metrics to wandb
        wandb.log({
            'policy/train_loss': train_loss,
            'policy/train_acc': train_acc,
            'policy/val_loss': val_loss,
            'policy/val_acc': val_acc,
            'policy/learning_rate': optimizer.param_groups[0]['lr'],
            'policy/epoch': epoch + 1
        }, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc: {val_acc:.2f}%)")
            # Log best model to wandb
            wandb.run.summary['policy/best_val_acc'] = best_val_acc
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Catan Policy Network (Behavior Cloning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Loading Modes:
  lazy      - Load files on-demand with LRU caching (RECOMMENDED)
              ✓ Memory efficient (only caches N files at a time)
              ✓ Supports train/val split, shuffling, and multi-worker loading
              
  memory    - Load all data into RAM (fastest, but needs large RAM)
              ✓ Fastest training speed
              ⚠ Requires enough RAM to hold entire dataset

Examples:
  # Basic training with default settings
  python train_policy.py --data-dir data/games
  
  # With wandb logging
  python train_policy.py --data-dir data/games --wandb --wandb-run-name policy-v1
  
  # Custom hyperparameters
  python train_policy.py --data-dir data/games --epochs 10 --batch-size 2048 --lr 5e-4
        """
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument(
        '--num-files-loaded',
        type=int,
        default=4,
        help='Number of parquet files to keep loaded in memory at once (default: 4)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='rl/weights/policy_network_best.pt',
        help='Path to save model (default: rl/weights/policy_network_best.pt)'
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
        default=10,
        help='Log batch metrics every N batches (default: 10)'
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/policy_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_policy.pt"
    
    # Load dataset
    print("Loading dataset...")
    dataset = CatanDataset(args.data_dir, max_files=None)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Feature dimension: {len(dataset.feature_cols)}")
    print(f"Action space size: {ACTION_SPACE_SIZE}")
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': 'policy_network',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'dataset_size': len(dataset),
                'feature_dim': len(dataset.feature_cols),
                'action_space_size': ACTION_SPACE_SIZE,
                'log_batch_freq': args.log_batch_freq,
            }
        )
        print(f"Initialized wandb: {args.wandb_project}/{args.wandb_run_name if args.wandb_run_name else wandb.run.name}")
    else:
        wandb.init(mode='disabled')
    
    # Train model
    train_policy_network(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        log_batch_freq=args.log_batch_freq,
        num_files_loaded=args.num_files_loaded,
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

