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
from tqdm import tqdm
import wandb
from typing import List
from data import create_dataloader, estimate_steps_per_epoch
from models import ValueNetwork


def train_value_network(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'value_network_best.pt',
    log_batch_freq: int = 1,
    hidden_dims: List[int] = [512, 512],
    num_workers: int = 4,
    buffer_size: int = 10000,
    wandb_config: dict = None,
):
    """Train the value network to predict returns."""
    
    print(f"\n{'='*60}")
    print("Training Value Network")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create data loaders
    train_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        split='train',
        shuffle=True,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )
    
    val_loader = create_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        split='validation',
        shuffle=False,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )
    
    # Get input dimension from first batch
    first_batch = next(iter(train_loader))
    input_dim = first_batch['features'].shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Initialize wandb after dataloaders are ready
    if wandb_config:
        wandb.init(**wandb_config)
        print(f"Initialized wandb: {wandb_config['project']}/{wandb_config.get('name', wandb.run.name)}")
    else:
        wandb.init(mode='disabled')
    
    # Create model
    model = ValueNetwork(input_dim, hidden_dims).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Calculate steps per epoch for progress bars
    train_steps = estimate_steps_per_epoch(data_dir, 'train', batch_size)
    val_steps = estimate_steps_per_epoch(data_dir, 'validation', batch_size)
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        num_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")):
            features = batch['features'].to(device, non_blocking=True)
            returns = batch['returns'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, returns)
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            num_train_batches += 1
            
            # Calculate accuracy (correct winner prediction based on sign)
            pred_sign = torch.sign(predictions)
            true_sign = torch.sign(returns)
            correct = (pred_sign == true_sign).sum().item()
            train_correct += correct
            train_total += returns.size(0)
            
            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                batch_accuracy = correct / returns.size(0)
                wandb.log({
                    'value/batch_loss': batch_loss,
                    'value/batch_accuracy': batch_accuracy,
                    'value/batch_step': global_step,
                }, step=global_step)
            
            global_step += 1
        
        train_loss /= num_train_batches
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        num_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=val_steps, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch"):
                features = batch['features'].to(device, non_blocking=True)
                returns = batch['returns'].to(device, non_blocking=True)
                
                predictions = model(features)
                loss = criterion(predictions, returns)
                val_loss += loss.item()
                num_val_batches += 1
                
                # Calculate accuracy (correct winner prediction based on sign)
                pred_sign = torch.sign(predictions)
                true_sign = torch.sign(returns)
                val_correct += (pred_sign == true_sign).sum().item()
                val_total += returns.size(0)
        
        val_loss /= num_val_batches
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log epoch metrics to wandb
        wandb.log({
            'value/train_loss': train_loss,
            'value/train_accuracy': train_accuracy,
            'value/val_loss': val_loss,
            'value/val_accuracy': val_accuracy,
            'value/learning_rate': optimizer.param_groups[0]['lr'],
            'value/epoch': epoch + 1
        }, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}")
        
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
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size (default: 1024)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel data loading workers (default: 4)'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=10000,
        help='Shuffle buffer size - larger = more random but more memory (default: 10000)'
    )
    parser.add_argument(
        '--hidden-dims',
        type=str,
        default='512,512',
        help='Hidden dimensions for the value network (default: 512,512)'
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/value_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_value.pt"
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Prepare wandb config (will be initialized inside train function)
    wandb_config = None
    if args.wandb:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_run_name,
            'config': {
                'model': 'value_network',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'hidden_dims': hidden_dims,
                'learning_rate': args.lr,
                'num_workers': args.num_workers,
                'buffer_size': args.buffer_size,
                'log_batch_freq': args.log_batch_freq,
            }
        }
    
    # Train model
    train_value_network(
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

