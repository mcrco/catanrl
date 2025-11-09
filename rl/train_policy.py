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
from sklearn.metrics import f1_score
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from data import create_dataloader, estimate_steps_per_epoch
from models import PolicyNetwork


def train_policy_network(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'policy_network_best.pt',
    log_batch_freq: int = 10,
    num_workers: int = 4,
    buffer_size: int = 10000,
    wandb_config: dict = None,
    load_weights: str = None,
):
    """Train the policy network to predict actions (behavior cloning)."""
    
    print(f"\n{'='*60}")
    print("Training Policy Network")
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
    num_actions = ACTION_SPACE_SIZE
    model = PolicyNetwork(input_dim, num_actions).to(device)
    
    # Load pre-trained weights if provided
    if load_weights:
        if os.path.exists(load_weights):
            print(f"Loading weights from: {load_weights}")
            state_dict = torch.load(load_weights, map_location=device)
            model.load_state_dict(state_dict)
            print("  ✓ Weights loaded successfully")
        else:
            print(f"Warning: Weights file '{load_weights}' not found. Starting from scratch.")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # Calculate steps per epoch for progress bars
    train_steps = estimate_steps_per_epoch(data_dir, 'train', batch_size)
    val_steps = estimate_steps_per_epoch(data_dir, 'validation', batch_size)
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")
    
    best_val_acc = 0.0
    global_step = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_preds = []
        train_all_labels = []
        
        num_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")):
            features = batch['features'].to(device)
            actions = batch['actions'].to(device)
            
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
            
            # Store predictions and labels for F1 calculation
            train_all_preds.extend(predicted.cpu().numpy())
            train_all_labels.extend(actions.cpu().numpy())
            
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
        train_f1 = f1_score(train_all_labels, train_all_preds, average='macro', zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_all_preds = []
        val_all_labels = []
        
        num_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=val_steps, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch"):
                features = batch['features'].to(device)
                actions = batch['actions'].to(device)
                
                logits = model(features)
                loss = criterion(logits, actions)
                
                val_loss += loss.item()
                num_val_batches += 1
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == actions).sum().item()
                val_total += actions.size(0)
                
                # Store predictions and labels for F1 calculation
                val_all_preds.extend(predicted.cpu().numpy())
                val_all_labels.extend(actions.cpu().numpy())
        
        val_loss /= num_val_batches
        val_acc = 100 * val_correct / val_total
        val_f1 = f1_score(val_all_labels, val_all_preds, average='macro', zero_division=0)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log epoch metrics to wandb
        wandb.log({
            'policy/train_loss': train_loss,
            'policy/train_acc': train_acc,
            'policy/train_f1': train_f1,
            'policy/val_loss': val_loss,
            'policy/val_acc': val_acc,
            'policy/val_f1': val_f1,
            'policy/learning_rate': optimizer.param_groups[0]['lr'],
            'policy/epoch': epoch + 1
        }, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}")
        
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
        '--save-path',
        type=str,
        default='rl/weights/policy_network_best.pt',
        help='Path to save model (default: rl/weights/policy_network_best.pt)'
    )
    parser.add_argument(
        '--load-weights',
        type=str,
        default=None,
        help='Path to pre-trained weights to continue training from (default: None)'
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/policy_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_policy.pt"
    
    # Prepare wandb config (will be initialized inside train function)
    wandb_config = None
    if args.wandb:
        wandb_config = {
            'project': args.wandb_project,
            'name': args.wandb_run_name,
            'config': {
                'model': 'policy_network',
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'num_workers': args.num_workers,
                'buffer_size': args.buffer_size,
                'action_space_size': ACTION_SPACE_SIZE,
                'log_batch_freq': args.log_batch_freq,
            }
        }
    
    # Train model
    train_policy_network(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        log_batch_freq=args.log_batch_freq,
        num_workers=args.num_workers,
        buffer_size=args.buffer_size,
        wandb_config=wandb_config,
        load_weights=args.load_weights,
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

