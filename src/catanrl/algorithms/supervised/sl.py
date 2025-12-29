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
from ...data.sharded_loader import create_dataloader_from_shards
from ...data.custom_parquet_iterable import create_dataloader, estimate_steps_per_epoch
from ...models.models import PolicyValueNetwork, HierarchicalPolicyValueNetwork
from .loss_utils import create_loss_computer
from .metrics import PolicyValueMetrics


def read_actions_from_parquet(parquet_file: str) -> np.ndarray:
    """Read only the ACTION column from a parquet file."""
    table = pq.read_table(parquet_file, columns=["ACTION"])
    return table["ACTION"].to_numpy()


def collect_train_actions_fast(
    data_dir: str,
    seed: int = 42,
    test_size: float = 0.2,
    max_workers: int = 4,
    sample_fraction: float = 1.0,
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
        all_files = sorted(data_path.glob("*.parquet"))

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
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "policy_value_network_best.pt",
    log_batch_freq: int = 1,
    hidden_dims: List[int] = [512, 512],
    num_workers: int = 4,
    buffer_size: int = 10000,
    wandb_config: dict = None,
    load_weights: str = None,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    model_type: str = "flat",
    action_type_weight: float = 1.0,
    param_weight: float = 1.0,
    use_class_weights: bool = False,
    weight_power: float = 0.5,
    weight_sample_fraction: float = 1.0,
    test_size: float = 0.2,
):
    """Train the joint policy-value network (flat or hierarchical) with shared backbone."""

    print(f"\n{'='*60}")
    print(f"Training Joint Policy-Value Network ({model_type.upper()})")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Model type: {model_type}")
    print(f"Policy loss weight: {policy_loss_weight}")
    print(f"Value loss weight: {value_loss_weight}")
    if model_type == "hierarchical":
        print(f"Action type weight: {action_type_weight}")
        print(f"Parameter weight: {param_weight}")
    print(f"Use class weights: {use_class_weights}")
    if use_class_weights:
        print(f"Weight power: {weight_power}")

    # Detect if using sharded data
    use_sharded_loader = "_sharded" in data_dir
    if use_sharded_loader:
        print(f"Detected sharded data directory, using HF Datasets loader")

    # Create data loaders
    loader_fn = create_dataloader_from_shards if use_sharded_loader else create_dataloader

    train_loader = loader_fn(
        data_dir=data_dir,
        batch_size=batch_size,
        split="train",
        shuffle=True,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )

    val_loader = loader_fn(
        data_dir=data_dir,
        batch_size=batch_size,
        split="validation",
        shuffle=False,
        buffer_size=buffer_size,
        num_workers=num_workers,
    )

    # Get input dimension from first batch
    first_batch = next(iter(train_loader))
    # Handle both [batch, features] and [features] shapes safely
    input_dim = first_batch["features"].shape[-1]
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
            test_size=test_size,
            max_workers=num_workers,  # Use same worker count
            sample_fraction=weight_sample_fraction,
        )

        print(f"✓ Collected {len(train_actions)} training actions")
        print(f"  Action distribution will be computed for class weighting\n")

    # Initialize wandb after dataloaders are ready
    if wandb_config:
        wandb.init(**wandb_config)
        print(
            f"Initialized wandb: {wandb_config['project']}/{wandb_config.get('name', wandb.run.name)}"
        )
    else:
        wandb.init(mode="disabled")

    # Create model based on type
    if model_type == "flat":
        num_actions = ACTION_SPACE_SIZE
        model = PolicyValueNetwork(input_dim, num_actions, hidden_dims).to(device)
        print(f"Created flat PolicyValueNetwork")
        print(f"  - Action space size: {num_actions}")
    elif model_type == "hierarchical":
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
        device=device,
    )
    print("Loss computer created")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Calculate steps per epoch for progress bars
    train_steps = estimate_steps_per_epoch(
        data_dir, "train", batch_size, max_workers=num_workers, use_processes=True
    )
    val_steps = estimate_steps_per_epoch(
        data_dir, "validation", batch_size, max_workers=num_workers, use_processes=True
    )
    print(f"Train steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")

    best_val_f1 = 0.0
    best_val_loss = float("inf")
    global_step = 0

    # Initialize metrics tracker
    train_metrics = PolicyValueMetrics(num_classes=ACTION_SPACE_SIZE, device=device)
    val_metrics = PolicyValueMetrics(num_classes=ACTION_SPACE_SIZE, device=device)

    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics.reset()  # Clear metrics from previous epoch
        for batch_idx, batch in enumerate(
            tqdm(
                train_loader,
                total=train_steps,
                desc=f"Epoch {epoch+1}/{epochs} [Train]",
                unit="batch",
            )
        ):
            features = batch["features"].to(device, non_blocking=True)
            actions = batch["actions"].to(device, non_blocking=True)
            returns = batch["returns"].to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass (different for flat vs hierarchical)
            if model_type == "flat":
                policy_logits, value_pred = model(features)

                # Calculate losses
                total_loss, policy_loss, value_loss = loss_computer.compute_flat_loss(
                    policy_logits, value_pred, actions, returns
                )
                action_type_loss = torch.tensor(0.0)
                param_loss = torch.tensor(0.0)

            elif model_type == "hierarchical":
                action_type_logits, param_logits, value_pred = model(features)

                # Calculate losses
                total_loss, action_type_loss, param_loss, value_loss = (
                    loss_computer.compute_hierarchical_loss(
                        action_type_logits,
                        param_logits,
                        value_pred,
                        actions,
                        returns,
                        model.flat_to_hierarchical,
                        action_type_weight=action_type_weight,
                        param_weight=param_weight,
                    )
                )

                # For metrics, convert hierarchical to flat policy logits
                policy_logits = model.get_flat_action_logits(action_type_logits, param_logits)
                policy_loss = action_type_loss  # Use action type loss as proxy for policy loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Extract scalar losses
            batch_policy_loss = policy_loss.item()
            batch_value_loss = value_loss.item()
            batch_total_loss = total_loss.item()
            batch_action_type_loss = (
                action_type_loss.item() if model_type == "hierarchical" else None
            )
            batch_param_loss = param_loss.item() if model_type == "hierarchical" else None

            # Update metrics tracker
            batch_metrics = train_metrics.update(
                policy_logits=policy_logits,
                value_pred=value_pred,
                actions=actions,
                returns=returns,
                policy_loss=batch_policy_loss,
                value_loss=batch_value_loss,
                total_loss=batch_total_loss,
                action_type_loss=batch_action_type_loss,
                param_loss=batch_param_loss,
            )

            # Log batch metrics to wandb
            if batch_idx % log_batch_freq == 0:
                # Compute F1 for logging (just this batch, not accumulated)
                _, predicted = torch.max(policy_logits, 1)
                batch_policy_f1 = f1_score(
                    actions.cpu().numpy(), predicted.cpu().numpy(), average="macro", zero_division=0
                )

                log_dict = {
                    "train/batch_policy_loss": batch_metrics["policy_loss"],
                    "train/batch_value_loss": batch_metrics["value_loss"],
                    "train/batch_total_loss": batch_metrics["total_loss"],
                    "train/batch_policy_acc": batch_metrics["policy_acc"],
                    "train/batch_policy_f1": batch_policy_f1,
                    "train/batch_value_acc": batch_metrics["value_acc"] * 100,
                    "train/batch_step": global_step,
                }
                if model_type == "hierarchical" and batch_action_type_loss is not None:
                    log_dict["train/batch_action_type_loss"] = batch_action_type_loss
                    log_dict["train/batch_param_loss"] = batch_param_loss
                wandb.log(log_dict, step=global_step)

            global_step += 1

        # Compute epoch-level metrics from accumulated statistics
        train_epoch_metrics = train_metrics.compute_epoch_metrics()

        # Validation
        model.eval()
        val_metrics.reset()  # Clear metrics from previous epoch/training

        with torch.no_grad():
            for batch in tqdm(
                val_loader, total=val_steps, desc=f"Epoch {epoch+1}/{epochs} [Val]", unit="batch"
            ):
                features = batch["features"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                returns = batch["returns"].to(device, non_blocking=True)

                # Forward pass (different for flat vs hierarchical)
                if model_type == "flat":
                    policy_logits, value_pred = model(features)

                    # Calculate losses
                    total_loss, policy_loss, value_loss = loss_computer.compute_flat_loss(
                        policy_logits, value_pred, actions, returns
                    )
                    action_type_loss = torch.tensor(0.0)
                    param_loss = torch.tensor(0.0)

                elif model_type == "hierarchical":
                    action_type_logits, param_logits, value_pred = model(features)

                    # Calculate losses
                    total_loss, action_type_loss, param_loss, value_loss = (
                        loss_computer.compute_hierarchical_loss(
                            action_type_logits,
                            param_logits,
                            value_pred,
                            actions,
                            returns,
                            model.flat_to_hierarchical,
                            action_type_weight=action_type_weight,
                            param_weight=param_weight,
                        )
                    )

                    # For metrics, convert hierarchical to flat policy logits
                    policy_logits = model.get_flat_action_logits(action_type_logits, param_logits)
                    policy_loss = action_type_loss  # Use action type loss as proxy for policy loss

                # Extract scalar losses
                batch_policy_loss = policy_loss.item()
                batch_value_loss = value_loss.item()
                batch_total_loss = total_loss.item()
                batch_action_type_loss = (
                    action_type_loss.item() if model_type == "hierarchical" else None
                )
                batch_param_loss = param_loss.item() if model_type == "hierarchical" else None

                # Update metrics tracker
                val_metrics.update(
                    policy_logits=policy_logits,
                    value_pred=value_pred,
                    actions=actions,
                    returns=returns,
                    policy_loss=batch_policy_loss,
                    value_loss=batch_value_loss,
                    total_loss=batch_total_loss,
                    action_type_loss=batch_action_type_loss,
                    param_loss=batch_param_loss,
                )

        # Compute epoch-level validation metrics from accumulated statistics
        val_epoch_metrics = val_metrics.compute_epoch_metrics()

        # Learning rate scheduling (based on total validation loss)
        scheduler.step(val_epoch_metrics["total_loss"])

        # Log epoch metrics to wandb
        log_dict = {
            "train/policy_loss": train_epoch_metrics["policy_loss"],
            "train/value_loss": train_epoch_metrics["value_loss"],
            "train/total_loss": train_epoch_metrics["total_loss"],
            "train/policy_acc": train_epoch_metrics["policy_acc"],
            "train/policy_f1": train_epoch_metrics["policy_f1"],
            "train/value_acc": train_epoch_metrics["value_acc"],
            "train/value_f1": train_epoch_metrics["value_f1"],
            "val/policy_loss": val_epoch_metrics["policy_loss"],
            "val/value_loss": val_epoch_metrics["value_loss"],
            "val/total_loss": val_epoch_metrics["total_loss"],
            "val/policy_acc": val_epoch_metrics["policy_acc"],
            "val/policy_f1": val_epoch_metrics["policy_f1"],
            "val/value_acc": val_epoch_metrics["value_acc"],
            "val/value_f1": val_epoch_metrics["value_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch + 1,
        }
        if model_type == "hierarchical":
            log_dict.update(
                {
                    "train/action_type_loss": train_epoch_metrics.get("action_type_loss", 0.0),
                    "train/param_loss": train_epoch_metrics.get("param_loss", 0.0),
                    "val/action_type_loss": val_epoch_metrics.get("action_type_loss", 0.0),
                    "val/param_loss": val_epoch_metrics.get("param_loss", 0.0),
                }
            )
        wandb.log(log_dict, step=global_step)

        print(f"Epoch {epoch+1}/{epochs}")
        print(
            f"  Train - Policy Loss: {train_epoch_metrics['policy_loss']:.4f}, Policy Acc: {train_epoch_metrics['policy_acc']:.2f}%, Policy F1: {train_epoch_metrics['policy_f1']:.4f} | "
            f"Value Loss: {train_epoch_metrics['value_loss']:.6f}, Value Acc: {train_epoch_metrics['value_acc']:.4f}, Value F1: {train_epoch_metrics['value_f1']:.4f} | Total Loss: {train_epoch_metrics['total_loss']:.4f}"
        )
        print(
            f"  Val   - Policy Loss: {val_epoch_metrics['policy_loss']:.4f}, Policy Acc: {val_epoch_metrics['policy_acc']:.2f}%, Policy F1: {val_epoch_metrics['policy_f1']:.4f} | "
            f"Value Loss: {val_epoch_metrics['value_loss']:.6f}, Value Acc: {val_epoch_metrics['value_acc']:.4f}, Value F1: {val_epoch_metrics['value_f1']:.4f} | Total Loss: {val_epoch_metrics['total_loss']:.4f}"
        )

        # Save best model based on combined metric (policy F1 and total loss)
        is_best = False
        val_policy_f1 = val_epoch_metrics["policy_f1"]
        val_total_loss = val_epoch_metrics["total_loss"]

        if val_policy_f1 > best_val_f1:
            best_val_f1 = val_policy_f1
            best_val_loss = val_total_loss
            is_best = True
        elif val_policy_f1 == best_val_f1 and val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            is_best = True

        if is_best:
            torch.save(model.state_dict(), save_path)
            print(
                f"  → Saved best model (val_policy_f1: {val_policy_f1:.4f}, val_total_loss: {val_total_loss:.4f})"
            )
            # Log best model metrics to wandb
            wandb.run.summary["best_val_policy_f1"] = best_val_f1
            wandb.run.summary["best_val_total_loss"] = best_val_loss

    print(f"\nBest validation policy F1: {best_val_f1:.4f}")
    print(f"Best validation total loss: {best_val_loss:.4f}")
    return model
