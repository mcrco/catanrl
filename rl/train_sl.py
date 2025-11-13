#!/usr/bin/env python3
"""
Joint training script for Catan Policy-Value Networks.
Supports both flat and hierarchical policy architectures.
Run: python train_joint.py --data-dir <path> --model-type [flat|hierarchical]
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb
import numpy as np
from typing import List
from sklearn.metrics import f1_score
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pyarrow.parquet as pq
from catanatron.gym.envs.catanatron_env import ACTION_SPACE_SIZE
from data import create_dataloader, create_dataloader_from_shards, estimate_steps_per_epoch
from models import PolicyValueNetwork, HierarchicalPolicyValueNetwork
from loss_utils import create_loss_computer


def read_actions_from_parquet(parquet_file: str) -> np.ndarray:
    """Read only the ACTION column from a parquet file."""
    table = pq.read_table(parquet_file, columns=['ACTION'])
    return table['ACTION'].to_numpy()


def collect_train_actions_fast(
    data_dir: str,
    seed: int = 42,
    test_size: float = 0.2,
    max_workers: int = 4,
    sample_fraction: float = 1.0
) -> np.ndarray:
    """
    Fast collection of training actions by reading only ACTION column from parquet files.
    
    Args:
        data_dir: Directory containing parquet files
        seed: Random seed for train/val split
        test_size: Fraction for validation split
        max_workers: Number of parallel workers
        sample_fraction: Fraction of data to sample (1.0 = all data, 0.1 = 10%)
        
    Returns:
        Array of action indices
    """
    data_path = Path(data_dir)
    
    # Get all parquet files
    if data_path.is_file():
        all_files = [data_path]
    else:
        all_files = sorted(data_path.glob('*.parquet'))
    
    print(f"Found {len(all_files)} parquet files")
    
    # Split files into train/val (same logic as in data.py)
    if len(all_files) > 1:
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        
        n_val = max(1, int(len(all_files) * test_size))
        n_train = len(all_files) - n_val
        
        train_indices = indices[:n_train]
        train_files = [all_files[i] for i in sorted(train_indices)]
    else:
        train_files = all_files
    
    # Sample files if requested
    if sample_fraction < 1.0:
        n_sample = max(1, int(len(train_files) * sample_fraction))
        train_files = train_files[:n_sample]
        print(f"Sampling {n_sample} files ({sample_fraction*100:.1f}% of training data)")
    
    print(f"Reading actions from {len(train_files)} training files...")
    
    # Read actions in parallel
    actions_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(read_actions_from_parquet, str(f)) for f in train_files]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Reading parquet files", unit="file"):
            actions_list.append(future.result())
    
    # Concatenate all actions
    all_actions = np.concatenate(actions_list)
    
    return all_actions


def train(
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
    model_type: str = 'flat',
    action_type_weight: float = 1.0,
    param_weight: float = 1.0,
    use_class_weights: bool = False,
    weight_power: float = 0.5,
    weight_sample_fraction: float = 1.0,
):
    """Train the joint policy-value network (flat or hierarchical) with shared backbone."""
    
    print(f"\n{'='*60}")
    print(f"Training Joint Policy-Value Network ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Policy loss weight: {policy_loss_weight}")
    print(f"Value loss weight: {value_loss_weight}")
    if model_type == 'hierarchical':
        print(f"Action type weight: {action_type_weight}")
        print(f"Parameter weight: {param_weight}")
    print(f"Use class weights: {use_class_weights}")
    if use_class_weights:
        print(f"Weight power: {weight_power}")
    
    # Detect if using sharded data
    use_sharded_loader = '_sharded' in data_dir
    if use_sharded_loader:
        print(f"Detected sharded data directory, using HF Datasets loader")
    
    # Create data loaders
    loader_fn = create_dataloader_from_shards if use_sharded_loader else create_dataloader
    
    train_loader = loader_fn(
        data_dir=data_dir,
        batch_size=batch_size,
        split='train',
        shuffle=True,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )
    
    val_loader = loader_fn(
        data_dir=data_dir,
        batch_size=batch_size,
        split='validation',
        shuffle=False,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )
    
    # Get input dimension from first batch
    first_batch = next(iter(train_loader))
    # Handle both [batch, features] and [features] shapes safely
    input_dim = first_batch['features'].shape[-1]
    print(f"Input dimension: {input_dim}")
    
    # Collect training actions for class weight computation
    train_actions = None
    if use_class_weights:
        print("\nCollecting training actions for class weight computation...")
        print("Using fast parallel parquet reading (ACTION column only)...")
        if weight_sample_fraction < 1.0:
            print(f"  Sampling {weight_sample_fraction*100:.1f}% of data for faster collection")
        
        # Fast collection: read only ACTION column in parallel from parquet files
        train_actions = collect_train_actions_fast(
            data_dir=data_dir,
            seed=42,
            test_size=0.2,
            max_workers=num_workers,  # Use same worker count
            sample_fraction=weight_sample_fraction
        )
        
        print(f"✓ Collected {len(train_actions)} training actions")
        print(f"  Action distribution will be computed for class weighting\n")
    
    # Initialize wandb after dataloaders are ready
    if wandb_config:
        wandb.init(**wandb_config)
        print(f"Initialized wandb: {wandb_config['project']}/{wandb_config.get('name', wandb.run.name)}")
    else:
        wandb.init(mode='disabled')
    
    # Create model based on type
    if model_type == 'flat':
        num_actions = ACTION_SPACE_SIZE
        model = PolicyValueNetwork(input_dim, num_actions, hidden_dims).to(device)
        print(f"Created flat PolicyValueNetwork")
        print(f"  - Action space size: {num_actions}")
    elif model_type == 'hierarchical':
        model = HierarchicalPolicyValueNetwork(input_dim, hidden_dims).to(device)
        print(f"Created HierarchicalPolicyValueNetwork")
        print(f"  - Action types: {model.NUM_ACTION_TYPES}")
        print(f"  - Total action space size: {model.action_space_size}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Load pre-trained weights if provided
    if load_weights:
        if os.path.exists(load_weights):
            print(f"Loading weights from: {load_weights}")
            state_dict = torch.load(load_weights, map_location=device)
            model.load_state_dict(state_dict)
            print("  ✓ Weights loaded successfully")
        else:
            print(f"Warning: Weights file '{load_weights}' not found. Starting from scratch.")
    
    # Create loss computer
    loss_computer = create_loss_computer(
        model=model,
        train_actions=train_actions,
        policy_weight=policy_loss_weight,
        value_weight=value_loss_weight,
        action_type_weight=action_type_weight,
        param_weight=param_weight,
        use_class_weights=use_class_weights,
        weight_power=weight_power,
        device=device
    )
    print("Loss computer created")
    
    # Optimizer and scheduler
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
        train_policy_all_preds = []
        train_policy_all_labels = []
        train_value_all_pred_signs = []
        train_value_all_true_signs = []
        train_action_type_loss = 0.0
        train_param_loss = 0.0
        
        num_train_batches = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch+1}/{epochs} [Train]", unit="batch")):
            features = batch['features'].to(device, non_blocking=True)
            actions = batch['actions'].to(device, non_blocking=True)
            returns = batch['returns'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass (different for flat vs hierarchical)
            if model_type == 'flat':
                policy_logits, value_pred = model(features)
                
                # Calculate losses
                total_loss, policy_loss, value_loss = loss_computer.compute_flat_loss(
                    policy_logits, value_pred, actions, returns
                )
                action_type_loss = torch.tensor(0.0)
                param_loss = torch.tensor(0.0)
                
            elif model_type == 'hierarchical':
                action_type_logits, param_logits, value_pred = model(features)
                
                # Calculate losses
                total_loss, action_type_loss, param_loss, value_loss = loss_computer.compute_hierarchical_loss(
                    action_type_logits, param_logits, value_pred, actions, returns,
                    model.flat_to_hierarchical,
                    action_type_weight=action_type_weight,
                    param_weight=param_weight
                )
                
                # For metrics, convert hierarchical to flat policy logits
                policy_logits = model.get_flat_action_logits(action_type_logits, param_logits)
                policy_loss = action_type_loss  # Use action type loss as proxy for policy loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            batch_policy_loss = policy_loss.item()
            batch_value_loss = value_loss.item()
            batch_total_loss = total_loss.item()
            batch_action_type_loss = action_type_loss.item() if model_type == 'hierarchical' else 0.0
            batch_param_loss = param_loss.item() if model_type == 'hierarchical' else 0.0
            train_policy_loss += batch_policy_loss
            train_value_loss += batch_value_loss
            train_total_loss += batch_total_loss
            train_action_type_loss += batch_action_type_loss
            train_param_loss += batch_param_loss
            num_train_batches += 1
            
            # Policy accuracy
            _, predicted = torch.max(policy_logits, 1)
            batch_policy_correct = (predicted == actions).sum().item()
            batch_policy_total = actions.size(0)
            train_policy_correct += batch_policy_correct
            train_policy_total += batch_policy_total
            
            # Store policy predictions and labels for F1 calculation
            train_policy_all_preds.extend(predicted.cpu().numpy())
            train_policy_all_labels.extend(actions.cpu().numpy())
            
            # Value accuracy (correct winner prediction based on sign)
            pred_sign = torch.sign(value_pred)
            true_sign = torch.sign(returns)
            batch_value_correct = (pred_sign == true_sign).sum().item()
            train_value_correct += batch_value_correct
            train_value_total += returns.size(0)
            
            # Store value predictions and labels for F1 calculation (convert to binary: 1 for positive, 0 for negative/zero)
            pred_binary = (pred_sign > 0).cpu().numpy().astype(int)
            true_binary = (true_sign > 0).cpu().numpy().astype(int)
            train_value_all_pred_signs.extend(pred_binary)
            train_value_all_true_signs.extend(true_binary)
            
            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                batch_policy_acc = 100 * batch_policy_correct / batch_policy_total
                batch_value_acc = batch_value_correct / batch_policy_total
                
                # Compute F1 scores for this batch
                batch_policy_f1 = f1_score(
                    actions.cpu().numpy(), 
                    predicted.cpu().numpy(), 
                    average='macro', 
                    zero_division=0
                )
                
                log_dict = {
                    'train/batch_policy_loss': batch_policy_loss,
                    'train/batch_value_loss': batch_value_loss,
                    'train/batch_total_loss': batch_total_loss,
                    'train/batch_policy_acc': batch_policy_acc,
                    'train/batch_policy_f1': batch_policy_f1,
                    'train/batch_value_acc': batch_value_acc,
                    'train/batch_step': global_step,
                }
                if model_type == 'hierarchical':
                    log_dict['train/batch_action_type_loss'] = batch_action_type_loss
                    log_dict['train/batch_param_loss'] = batch_param_loss
                wandb.log(log_dict, step=global_step)
            
            global_step += 1
        
        train_policy_loss /= num_train_batches
        train_value_loss /= num_train_batches
        train_total_loss /= num_train_batches
        train_action_type_loss /= num_train_batches
        train_param_loss /= num_train_batches
        train_policy_acc = 100 * train_policy_correct / train_policy_total
        train_value_acc = train_value_correct / train_value_total if train_value_total > 0 else 0.0
        train_policy_f1 = f1_score(train_policy_all_labels, train_policy_all_preds, average='macro', zero_division=0)
        train_value_f1 = f1_score(train_value_all_true_signs, train_value_all_pred_signs, average='binary', zero_division=0)
        
        # Validation
        model.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_total_loss = 0.0
        val_policy_correct = 0
        val_policy_total = 0
        val_value_correct = 0
        val_value_total = 0
        val_policy_all_preds = []
        val_policy_all_labels = []
        val_value_all_pred_signs = []
        val_value_all_true_signs = []
        val_action_type_loss = 0.0
        val_param_loss = 0.0
        
        num_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, total=val_steps, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch"):
                features = batch['features'].to(device, non_blocking=True)
                actions = batch['actions'].to(device, non_blocking=True)
                returns = batch['returns'].to(device, non_blocking=True)
                
                # Forward pass (different for flat vs hierarchical)
                if model_type == 'flat':
                    policy_logits, value_pred = model(features)
                    
                    # Calculate losses
                    total_loss, policy_loss, value_loss = loss_computer.compute_flat_loss(
                        policy_logits, value_pred, actions, returns
                    )
                    action_type_loss = torch.tensor(0.0)
                    param_loss = torch.tensor(0.0)
                    
                elif model_type == 'hierarchical':
                    action_type_logits, param_logits, value_pred = model(features)
                    
                    # Calculate losses
                    total_loss, action_type_loss, param_loss, value_loss = loss_computer.compute_hierarchical_loss(
                        action_type_logits, param_logits, value_pred, actions, returns,
                        model.flat_to_hierarchical,
                        action_type_weight=action_type_weight,
                        param_weight=param_weight
                    )
                    
                    # For metrics, convert hierarchical to flat policy logits
                    policy_logits = model.get_flat_action_logits(action_type_logits, param_logits)
                    policy_loss = action_type_loss  # Use action type loss as proxy for policy loss
                
                val_policy_loss += policy_loss.item()
                val_value_loss += value_loss.item()
                val_total_loss += total_loss.item()
                val_action_type_loss += action_type_loss.item() if model_type == 'hierarchical' else 0.0
                val_param_loss += param_loss.item() if model_type == 'hierarchical' else 0.0
                num_val_batches += 1
                
                # Policy accuracy
                _, predicted = torch.max(policy_logits, 1)
                val_policy_correct += (predicted == actions).sum().item()
                val_policy_total += actions.size(0)
                
                # Store policy predictions and labels for F1 calculation
                val_policy_all_preds.extend(predicted.cpu().numpy())
                val_policy_all_labels.extend(actions.cpu().numpy())
                
                # Value accuracy (correct winner prediction based on sign)
                pred_sign = torch.sign(value_pred)
                true_sign = torch.sign(returns)
                val_value_correct += (pred_sign == true_sign).sum().item()
                val_value_total += returns.size(0)
                
                # Store value predictions and labels for F1 calculation (convert to binary: 1 for positive, 0 for negative/zero)
                pred_binary = (pred_sign > 0).cpu().numpy().astype(int)
                true_binary = (true_sign > 0).cpu().numpy().astype(int)
                val_value_all_pred_signs.extend(pred_binary)
                val_value_all_true_signs.extend(true_binary)
        
        val_policy_loss /= num_val_batches
        val_value_loss /= num_val_batches
        val_total_loss /= num_val_batches
        val_action_type_loss /= num_val_batches
        val_param_loss /= num_val_batches
        val_policy_acc = 100 * val_policy_correct / val_policy_total
        val_value_acc = val_value_correct / val_value_total if val_value_total > 0 else 0.0
        val_policy_f1 = f1_score(val_policy_all_labels, val_policy_all_preds, average='macro', zero_division=0)
        val_value_f1 = f1_score(val_value_all_true_signs, val_value_all_pred_signs, average='binary', zero_division=0)
        
        # Learning rate scheduling (based on total validation loss)
        scheduler.step(val_total_loss)
        
        # Log epoch metrics to wandb
        log_dict = {
            'train/policy_loss': train_policy_loss,
            'train/value_loss': train_value_loss,
            'train/total_loss': train_total_loss,
            'train/policy_acc': train_policy_acc,
            'train/policy_f1': train_policy_f1,
            'train/value_acc': train_value_acc,
            'train/value_f1': train_value_f1,
            'val/policy_loss': val_policy_loss,
            'val/value_loss': val_value_loss,
            'val/total_loss': val_total_loss,
            'val/policy_acc': val_policy_acc,
            'val/policy_f1': val_policy_f1,
            'val/value_acc': val_value_acc,
            'val/value_f1': val_value_f1,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1
        }
        if model_type == 'hierarchical':
            log_dict.update({
                'train/action_type_loss': train_action_type_loss,
                'train/param_loss': train_param_loss,
                'val/action_type_loss': val_action_type_loss,
                'val/param_loss': val_param_loss,
            })
        wandb.log(log_dict, step=global_step)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train - Policy Loss: {train_policy_loss:.4f}, Policy Acc: {train_policy_acc:.2f}%, Policy F1: {train_policy_f1:.4f} | "
              f"Value Loss: {train_value_loss:.6f}, Value Acc: {train_value_acc:.4f}, Value F1: {train_value_f1:.4f} | Total Loss: {train_total_loss:.4f}")
        print(f"  Val   - Policy Loss: {val_policy_loss:.4f}, Policy Acc: {val_policy_acc:.2f}%, Policy F1: {val_policy_f1:.4f} | "
              f"Value Loss: {val_value_loss:.6f}, Value Acc: {val_value_acc:.4f}, Value F1: {val_value_f1:.4f} | Total Loss: {val_total_loss:.4f}")
        
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
    if args.wandb and args.wandb_run_name and args.save_path == 'rl/weights/policy_value_network_best.pt':
        args.save_path = f"rl/weights/{args.wandb_run_name}_policy_value.pt"
    
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

