#!/usr/bin/env python3
"""
Joint training script for Catan Policy-Value Network with shared backbone.
Run: python train_policy_value.py --data-dir <path>
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
from typing import List
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from data import create_dataloader, estimate_steps_per_epoch
from models import PolicyValueNetwork


def train_policy_value_network(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'policy_value_network_best.pt',
    log_batch_freq: int = 1,
    hidden_dims: List[int] = [512, 512],
    num_workers: int = 4,
    buffer_size: int = 10000,
    wandb_config: dict = None,
    load_weights: str = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
):
    """Train the joint policy-value network with shared backbone."""
    
    print(f"\n{'='*60}")
    print("Training Joint Policy-Value Network")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Policy loss weight: {policy_loss_weight}")
    print(f"Value loss weight: {value_loss_weight}")
    
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
    num_actions = ACTION_SPACE_SIZE
    model = PolicyValueNetwork(input_dim, num_actions, hidden_dims).to(device)
    
    # Load pre-trained weights if provided
    if load_weights:
        if os.path.exists(load_weights):
            print(f"Loading weights from: {load_weights}")
            state_dict = torch.load(load_weights, map_location=device)
            model.load_state_dict(state_dict)
            print("  ✓ Weights loaded successfully")
        else:
            print(f"Warning: Weights file '{load_weights}' not found. Starting from scratch.")
    
    # Loss functions and optimizer
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Calculate steps per epoch for progress bars
    train_steps = estimate_steps_per_epoch(data_dir, 'train', batch_size)
    val_steps = estimate_steps_per_epoch(data_dir, 'validation', batch_size)
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_policy_loss = 0.0
        train_value_loss = 0.0
        train_total_loss = 0.0
        train_policy_correct = 0
        train_policy_total = 0
        train_value_correct = 0
        train_value_total = 0
        
        num_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")):
            features = batch['features'].to(device, non_blocking=True)
            actions = batch['actions'].to(device, non_blocking=True)
            returns = batch['returns'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_pred = model(features)
            
            # Calculate losses
            policy_loss = policy_criterion(policy_logits, actions)
            value_loss = value_criterion(value_pred, returns)
            total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            batch_policy_loss = policy_loss.item()
            batch_value_loss = value_loss.item()
            batch_total_loss = total_loss.item()
            train_policy_loss += batch_policy_loss
            train_value_loss += batch_value_loss
            train_total_loss += batch_total_loss
            num_train_batches += 1
            
            # Policy accuracy
            _, predicted = torch.max(policy_logits, 1)
            batch_policy_correct = (predicted == actions).sum().item()
            batch_policy_total = actions.size(0)
            train_policy_correct += batch_policy_correct
            train_policy_total += batch_policy_total
            
            # Value accuracy (correct winner prediction based on sign)
            pred_sign = torch.sign(value_pred)
            true_sign = torch.sign(returns)
            batch_value_correct = (pred_sign == true_sign).sum().item()
            train_value_correct += batch_value_correct
            train_value_total += returns.size(0)
            
            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                batch_policy_acc = 100 * batch_policy_correct / batch_policy_total
                batch_value_acc = batch_value_correct / batch_policy_total
                wandb.log({
                    'train/batch_policy_loss': batch_policy_loss,
                    'train/batch_value_loss': batch_value_loss,
                    'train/batch_total_loss': batch_total_loss,
                    'train/batch_policy_acc': batch_policy_acc,
                    'train/batch_value_acc': batch_value_acc,
                    'train/batch_step': global_step,
                }, step=global_step)
            
            global_step += 1
        
        train_policy_loss /= num_train_batches
        train_value_loss /= num_train_batches
        train_total_loss /= num_train_batches
        train_policy_acc = 100 * train_policy_correct / train_policy_total
        train_value_acc = train_value_correct / train_value_total if train_value_total > 0 else 0.0
        
        # Validation
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_total_loss = 0.0
        val_policy_correct = 0
        val_policy_total = 0
        val_value_correct = 0
        val_value_total = 0
        
        num_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=val_steps, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch"):
                features = batch['features'].to(device, non_blocking=True)
                actions = batch['actions'].to(device, non_blocking=True)
                returns = batch['returns'].to(device, non_blocking=True)
                
                # Forward pass
                policy_logits, value_pred = model(features)
                
                # Calculate losses
                policy_loss = policy_criterion(policy_logits, actions)
                value_loss = value_criterion(value_pred, returns)
                total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss
                
                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_total_loss += total_loss.item()
                num_val_batches += 1
                
                # Policy accuracy
                _, predicted = torch.max(policy_logits, 1)
                val_policy_correct += (predicted == actions).sum().item()
                val_policy_total += actions.size(0)
                
                # Value accuracy (correct winner prediction based on sign)
                pred_sign = torch.sign(value_pred)
                true_sign = torch.sign(returns)
                val_value_correct += (pred_sign == true_sign).sum().item()
                val_value_total += returns.size(0)
        
        val_policy_loss /= num_val_batches
        val_value_loss /= num_val_batches
        val_total_loss /= num_val_batches
        val_policy_acc = 100 * val_policy_correct / val_policy_total
        val_value_acc = val_value_correct / val_value_total if val_value_total > 0 else 0.0
        
        # Learning rate scheduling (based on total validation loss)
        scheduler.step(val_total_loss)
        
        # Log epoch metrics to wandb
        wandb.log({
            'train/policy_loss': train_policy_loss,
            'train/value_loss': train_value_loss,
            'train/total_loss': train_total_loss,
            'train/policy_acc': train_policy_acc,
            'train/value_acc': train_value_acc,
            'val/policy_loss': val_policy_loss,
            'val/value_loss': val_value_loss,
            'val/total_loss': val_total_loss,
            'val/policy_acc': val_policy_acc,
            'val/value_acc': val_value_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        }, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Policy Loss: {train_policy_loss:.4f}, Policy Acc: {train_policy_acc:.2f}% | "
              f"Value Loss: {train_value_loss:.6f}, Value Acc: {train_value_acc:.4f} | Total Loss: {train_total_loss:.4f}")
        print(f"  Val   - Policy Loss: {val_policy_loss:.4f}, Policy Acc: {val_policy_acc:.2f}% | "
              f"Value Loss: {val_value_loss:.6f}, Value Acc: {val_value_acc:.4f} | Total Loss: {val_total_loss:.4f}")
        
        # Save best model based on combined metric (policy accuracy and total loss)
        is_best = False
        if val_policy_acc > best_val_acc:
            best_val_acc = val_policy_acc
            best_val_loss = val_total_loss
            is_best = True
        elif val_policy_acc == best_val_acc and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            is_best = True
        
        if is_best:
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_policy_acc: {val_policy_acc:.2f}%, val_total_loss: {val_total_loss:.4f})")
            # Log best model metrics to wandb
            wandb.run.summary['best_val_policy_acc'] = best_val_acc
            wandb.run.summary['best_val_total_loss'] = best_val_loss
    
    print(f"\nBest validation policy accuracy: {best_val_acc:.2f}%")
    print(f"Best validation total loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train Catan Joint Policy-Value Network with Shared Backbone',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help='Hidden dimensions for the shared backbone (default: 512,512)'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='rl/weights/policy_value_network_best.pt',
        help='Path to save model (default: rl/weights/policy_value_network_best.pt)'
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/policy_value_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_policy_value.pt"
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Prepare wandb config (will be initialized inside train function)
    wandb_config = None
    if args.wandb:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_run_name,
            'config': {
                'model': 'policy_value_network',
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
            }
        }
    
    # Train model
    train_policy_value_network(
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

