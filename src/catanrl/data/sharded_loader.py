from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from datasets import load_dataset

from .data_utils import is_non_graph_feature_parquet

def create_dataloader_from_shards(
    data_dir: str,
    batch_size: int = 1024,
    split: str = 'train',
    shuffle: bool = True,
    buffer_size: int = 10000,
    num_workers: int = 0,
    seed: int = 42,
    value_type: str = 'RETURN',
    max_files: Optional[int] = None,
    split_by_files: bool = True,
    test_size: float = 0.2,
):
    """
    Create a fast PyTorch DataLoader for sharded parquet files using HuggingFace Datasets.
    
    This is optimized for larger sharded files (e.g., 5k-20k rows per file) where file open
    overhead is minimal and HF Datasets' streaming + shuffle works well.
    
    Args:
        data_dir: Directory containing sharded .parquet files
        batch_size: Training batch size
        split: 'train' or 'validation'
        shuffle: If True, shuffle with buffer_size reservoir
        buffer_size: Size of shuffle buffer (larger = more random but more memory)
        num_workers: Ignored (HF Datasets handles parallelism internally)
        seed: Random seed for deterministic shuffling
        value_type: Which return column to use (e.g., 'RETURN' or 'TOURNAMENT_RETURN')
        max_files: Limit number of files (for debugging or sampling)
        split_by_files: If True, split entire files into train/val
        test_size: Fraction for validation split when splitting by files
    
    Returns:
        A DataLoader yielding dicts with keys: 'features', 'actions', 'returns'
    """
    data_path = Path(data_dir)
    print(f"\n{'='*60}")
    print(f"Loading dataset from shards: {data_dir}")
    print(f"{'='*60}")
    
    # Collect files
    if data_path.is_file():
        all_files = [data_path]
        print(f"Single file mode: {data_path.name}")
    else:
        all_files = sorted(data_path.glob('*.parquet'))
        if max_files:
            all_files = all_files[:max_files]
        print(f"Found {len(all_files)} shard files")
    
    # File-based split
    if split_by_files and len(all_files) > 1 and split in ['train', 'validation']:
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        n_val = max(1, int(len(all_files) * test_size))
        n_train = len(all_files) - n_val
        if split == 'train':
            selected_indices = indices[:n_train]
            print(f"File-based split: using {n_train} shards for training")
        else:
            selected_indices = indices[n_train:]
            print(f"File-based split: using {n_val} shards for validation")
        data_files = [str(all_files[i]) for i in sorted(selected_indices)]
    else:
        data_files = [str(f) for f in all_files]
    
    if not data_files:
        raise ValueError(f"No parquet files found in: {data_dir}")
    
    # Load dataset with streaming
    dataset = load_dataset(
        'parquet',
        data_files=data_files,
        streaming=True,
        split='train'  # HF naming convention
    )
    
    # Get feature columns from first sample
    print("Identifying feature columns...")
    first_sample = next(iter(dataset))
    # Keep BT_* and non-graph F_* only (exclude F_NODE*, F_EDGE*, F_TILE*, F_PORT*)
    feature_cols = [c for c in first_sample.keys() if c.startswith('BT_') or is_non_graph_feature_parquet(c)]
    return_col = value_type
    print(f"Features: {len(feature_cols)} columns")
    
    # Prune unused columns EARLY to reduce memory
    keep_cols = set(feature_cols + ['ACTION', return_col])
    remove_cols = [c for c in first_sample.keys() if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
        print(f"Removed {len(remove_cols)} unused columns")
    
    # Shuffle if requested (after pruning to reduce memory)
    if shuffle:
        # Cap buffer size to prevent excessive memory use with large shards
        effective_buffer = min(buffer_size, batch_size * 8)
        dataset = dataset.shuffle(buffer_size=effective_buffer, seed=seed)
        print(f"Shuffling with buffer_size={effective_buffer}")
    
    # Map to PyTorch format
    def format_batch(batch):
        # Extract feature columns efficiently (vectorized)
        # Stack columns into 2D array: [batch_size, num_features]
        feature_arrays = [np.array(batch[col], dtype=np.float32) for col in feature_cols]
        features = np.stack(feature_arrays, axis=1)
        
        actions = np.array(batch['ACTION'], dtype=np.int64)
        returns = np.array(batch[return_col], dtype=np.float32)
        
        return {
            'features': torch.from_numpy(features),
            'actions': torch.from_numpy(actions),
            'returns': torch.from_numpy(returns)
        }
    
    # Apply formatting with batching
    dataset = dataset.map(format_batch, batched=True, batch_size=batch_size)
    
    # Use batch_size=1 and unwrap to avoid double-batching
    # (HF map already produces batched tensors, so we don't want DataLoader to batch again)
    def _unwrap_collate(items):
        return items[0]
    
    # Create PyTorch DataLoader
    # Note: num_workers=0 because HF Datasets handles parallelism internally
    loader = DataLoader(
        dataset,
        batch_size=1,                # Important: avoid double-batching
        num_workers=0,
        collate_fn=_unwrap_collate,  # Unwrap the pre-batched item
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"DataLoader ready: batch_size={batch_size}")
    print(f"Note: HF Datasets handles parallelism internally")
    print(f"{'='*60}\n")
    
    return loader
