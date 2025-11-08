import torch
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from pyarrow.parquet import ParquetFile


def estimate_steps_per_epoch(
    data_dir: str,
    split: str,
    batch_size: int,
    seed: int = 42,
    test_size: float = 0.2,
    max_files: Optional[int] = None,
    split_by_files: bool = True,
) -> int:
    """
    Estimate the number of steps (batches) per epoch by reading Parquet metadata.
    
    Args:
        data_dir: Directory containing .parquet files or path to single file
        split: 'train' or 'validation'
        batch_size: Batch size for training
        seed: Random seed (must match dataloader seed)
        test_size: Fraction of data for validation
        max_files: Maximum number of files to use
        split_by_files: If True, split entire files into train/val
    
    Returns:
        Number of batches (steps) per epoch
    """
    data_path = Path(data_dir)
    
    # Get all files
    if data_path.is_file():
        all_files = [data_path]
    else:
        all_files = sorted(data_path.glob('*.parquet'))
        if max_files:
            all_files = all_files[:max_files]
    
    # Split files into train/val if needed (must match create_dataloader logic)
    if split_by_files and len(all_files) > 1 and split in ['train', 'validation']:
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        
        n_val = max(1, int(len(all_files) * test_size))
        n_train = len(all_files) - n_val
        
        if split == 'train':
            selected_indices = indices[:n_train]
        else:
            selected_indices = indices[n_train:]
        
        selected_files = [all_files[i] for i in sorted(selected_indices)]
    else:
        selected_files = all_files
    
    # Read metadata to get total rows
    total_rows = sum(ParquetFile(str(p)).metadata.num_rows for p in selected_files)
    
    # Calculate steps per epoch
    steps = int(np.ceil(total_rows / batch_size))
    
    return steps


def create_dataloader(
    data_dir: str,
    batch_size: int = 1024,
    split: str = 'train',
    shuffle: bool = True,
    buffer_size: int = 10000,
    num_workers: int = 4,
    seed: int = 42,
    value_type: str = 'RETURN',
    max_files: Optional[int] = None,
    split_by_files: bool = True,
    test_size: float = 0.2,
):
    """
    Create a PyTorch DataLoader using HuggingFace Datasets for efficient parquet streaming.
    
    Args:
        data_dir: Directory containing .parquet files or path to single file
        batch_size: Batch size for training
        split: 'train' or 'validation' (after train/test split)
        shuffle: Whether to shuffle the data
        buffer_size: Size of shuffle buffer (larger = more random but more memory)
        num_workers: (Ignored - kept for API compatibility. Parallelism handled by HF Datasets)
        seed: Random seed for shuffling and splitting
        value_type: Which return column to use (default 'RETURN')
        max_files: Maximum number of files to use (None = all)
        split_by_files: If True, split entire files into train/val (recommended).
                       If False, split by rows across all files.
        test_size: Fraction of data to use for validation (default 0.1 = 10%)
    
    Returns:
        PyTorch DataLoader that yields (features, actions, returns) batches
    """
    data_path = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"Loading dataset from {data_dir}")
    print(f"{'='*60}")
    
    # Determine data files pattern
    if data_path.is_file():
        data_files = str(data_path)
        print(f"Single file mode: {data_path.name}")
        all_files = [data_path]
    else:
        all_files = sorted(data_path.glob('*.parquet'))
        if max_files:
            all_files = all_files[:max_files]
        print(f"Found {len(all_files)} parquet files")
    
    # Split files into train/val if multiple files
    if split_by_files and len(all_files) > 1 and split in ['train', 'validation']:
        # File-based split: entire files go to either train or val
        rng = np.random.RandomState(seed)
        indices = np.arange(len(all_files))
        rng.shuffle(indices)
        
        n_val = max(1, int(len(all_files) * test_size))
        n_train = len(all_files) - n_val
        
        if split == 'train':
            selected_indices = indices[:n_train]
            print(f"File-based split: using {n_train} files for training")
        else:
            selected_indices = indices[n_train:]
            print(f"File-based split: using {n_val} files for validation")
        
        selected_files = [all_files[i] for i in sorted(selected_indices)]
        data_files = [str(f) for f in selected_files]
    else:
        # Use all files (row-based split may be applied later)
        data_files = [str(f) for f in all_files]
    
    # Load dataset with streaming
    dataset = load_dataset(
        'parquet',
        data_files=data_files,
        streaming=True,
        split='train'  # HF naming convention
    )
    
    # Get feature columns BEFORE shuffling to avoid deadlock
    print("Identifying feature columns...")
    first_sample = next(iter(dataset))
    feature_cols = [c for c in first_sample.keys() if c.startswith('F_') or c.startswith('BT_')]
    return_col = value_type if value_type != 'RETURN' else 'RETURN'
    print(f"Features: {len(feature_cols)} columns")
    
    # Prune unused columns EARLY to reduce shuffle memory
    keep_cols = set(feature_cols + ['ACTION', return_col])
    remove_cols = [c for c in first_sample.keys() if c not in keep_cols]
    if remove_cols:
        dataset = dataset.remove_columns(remove_cols)
    
    # Row-based split if not splitting by files
    if not split_by_files and split in ['train', 'validation']:
        print("Row-based split: splitting samples across all files")
        dataset_splits = dataset.train_test_split(test_size=test_size, seed=seed)
        dataset = dataset_splits['train' if split == 'train' else 'test']
        print(f"Split: {split}")
    
    # Shuffle if requested (after pruning)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        print(f"Shuffling with buffer_size={buffer_size}")
    
    # Map to PyTorch format
    def format_batch(batch):
        # Extract feature columns efficiently (vectorized)
        # Stack columns into 2D array: [batch_size, num_features]
        feature_arrays = [np.array(batch[col], dtype=np.float32) for col in feature_cols]
        features = np.stack(feature_arrays, axis=1)
        
        actions = np.array(batch['ACTION'], dtype=np.int64)
        
        # Get return values and normalize
        returns = np.array(batch[return_col], dtype=np.float32)
        returns_normalized = (returns + 1000) / 2010
        
        return {
            'features': torch.from_numpy(features),
            'actions': torch.from_numpy(actions),
            'returns': torch.from_numpy(returns_normalized)
        }
    
    # Apply formatting
    dataset = dataset.map(format_batch, batched=True, batch_size=batch_size)
    
    # Create PyTorch DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"DataLoader ready: batch_size={batch_size}")
    print(f"{'='*60}\n")
    
    return loader