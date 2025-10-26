#!/usr/bin/env python3
"""
Training script for Catan Value Network.
Run: python train_value.py --data-dir <path>
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import List
from data import CatanDataset
from models import ValueNetwork


def train_value_network(
    dataset,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'value_network_best.pt',
    log_batch_freq: int = 10,
    num_workers: int = 4,
    hidden_dims: List[int] = [512, 512],
    prefetch_factor: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """Train the value network to predict returns."""
    
    print(f"\n{'='*60}")
    print("Training Value Network")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create model
    input_dim = len(dataset.feature_cols)
    model = ValueNetwork(input_dim, hidden_dims).to(device)
    
    # Split data into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.startswith('cuda')),
        prefetch_factor=prefetch_factor,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(pin_memory and device.startswith('cuda')),
        prefetch_factor=prefetch_factor,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (features, _, returns) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            features = features.to(device)
            returns = returns.to(device)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, returns)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                wandb.log({
                    'value/batch_loss': batch_loss,
                    'value/batch_step': global_step,
                }, step=global_step)
            
            global_step += 1
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, _, returns in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                features = features.to(device)
                returns = returns.to(device)
                
                predictions = model(features)
                loss = criterion(predictions, returns)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log epoch metrics to wandb
        wandb.log({
            'value/train_loss': train_loss,
            'value/val_loss': val_loss,
            'value/learning_rate': optimizer.param_groups[0]['lr'],
            'value/epoch': epoch + 1
        }, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_loss: {val_loss:.6f})")
            # Log best model to wandb
            wandb.run.summary['value/best_val_loss'] = best_val_loss
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Catan Value Network',
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
  python train_value.py --data-dir data/games
  
  # With wandb logging
  python train_value.py --data-dir data/games --wandb --wandb-run-name value-v1
  
  # Custom hyperparameters
  python train_value.py --data-dir data/games --epochs 10 --batch-size 2048 --lr 5e-4
  
  # In-memory mode (if you have enough RAM)
  python train_value.py --data-dir data/games --mode memory
        """
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument(
        '--mode', 
        type=str, 
        default='lazy', 
        choices=['memory', 'lazy'],
        help='Data loading mode (default: lazy)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=str,
        default='512,512',
        help='Hidden dimensions for the value network (default: 1024,1024)'
    )
    parser.add_argument(
        '--cache-size',
        type=int,
        default=10,
        help='Number of files to keep in cache for lazy mode (default: 10)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='rl/weights/value_network_best.pt',
        help='Path to save model (default: rl/weights/value_network_best.pt)'
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
    parser.add_argument(
        '--prefetch-factor',
        type=int,
        default=1,
        help='DataLoader prefetch_factor (default: 1, ignored if num_workers=0)'
    )
    parser.add_argument(
        '--no-pin-memory',
        action='store_true',
        help='Disable pin_memory for DataLoader (enabled by default when using CUDA)'
    )
    parser.add_argument(
        '--no-persistent-workers',
        action='store_true',
        help='Disable persistent_workers in DataLoader'
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/value_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_value.pt"
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load dataset
    print("Loading dataset...")
    # Scale cache per worker to avoid multiplying memory across workers
    workers = max(1, args.num_workers)
    effective_cache_size = max(1, args.cache_size // workers)
    if effective_cache_size < args.cache_size:
        print(f"Adjusting cache size from {args.cache_size} -> {effective_cache_size} per worker (workers={args.num_workers})")
    dataset = CatanDataset(args.data_dir, max_files=None, cache_size=effective_cache_size)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Feature dimension: {len(dataset.feature_cols)}")
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'model': 'value_network',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'hidden_dims': hidden_dims,
                'learning_rate': args.lr,
                'mode': args.mode,
                'cache_size': args.cache_size if args.mode == 'lazy' else None,
                'dataset_size': len(dataset),
                'feature_dim': len(dataset.feature_cols),
                'log_batch_freq': args.log_batch_freq,
            }
        )
        print(f"Initialized wandb: {args.wandb_project}/{args.wandb_run_name if args.wandb_run_name else wandb.run.name}")
    else:
        wandb.init(mode='disabled')
    
    # Train model
    train_value_network(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=hidden_dims,
        num_workers=args.num_workers,
        save_path=args.save_path,
        log_batch_freq=args.log_batch_freq,
        prefetch_factor=args.prefetch_factor,
        pin_memory=(not args.no_pin_memory),
        persistent_workers=(not args.no_persistent_workers),
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

