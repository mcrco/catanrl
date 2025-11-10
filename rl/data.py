import torch
from pathlib import Path
from typing import Optional, Iterator, List
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import numpy as np
from pyarrow.parquet import ParquetFile
import pyarrow.parquet as pq
import math
import random
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datasets import load_dataset


class ParquetBatchIterable(IterableDataset):
    """
    IterableDataset that yields fully-formed batches from individual parquet files 
    by taking a buffer_size number of rows from multiple files, shuffling them, and yielding 
    batches.
    """
    def __init__(
        self,
        files: List[Path],
        required_cols: List[str],
        feature_cols: List[str],
        return_col: str,
        batch_size: int,
        buffer_size: int,
        shuffle: bool,
        seed: int,
    ):
        super().__init__()
        self.files = files
        self.required_cols = required_cols
        self.feature_cols = feature_cols
        self.return_col = return_col
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
    
    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        # Partition files across workers
        if worker is None:
            worker_id = 0
            num_workers_local = 1
        else:
            worker_id = worker.id
            num_workers_local = worker.num_workers
        
        per_worker = int(math.ceil(len(self.files) / num_workers_local))
        start = worker_id * per_worker
        end = min(start + per_worker, len(self.files))
        worker_files = self.files[start:end]
        
        # Seed per worker for deterministic yet distinct shuffles
        rng = random.Random(self.seed + worker_id)
        if self.shuffle:
            rng.shuffle(worker_files)
        
        # Shuffle buffer: accumulate buffer_size rows from multiple files,
        # shuffle them, then yield batches
        shuffle_buf_features = []
        shuffle_buf_actions = []
        shuffle_buf_returns = []
        
        for fp in worker_files:
            # Read only required columns from file
            pf = pq.ParquetFile(str(fp))
            table = pf.read(columns=self.required_cols, use_threads=True)
            # Convert to pandas for efficient 2D numpy block extraction
            df = table.to_pandas()  # zero-copy for numeric types where possible
            
            feats = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            acts = df['ACTION'].to_numpy(dtype=np.int64, copy=False)
            rets = df[self.return_col].to_numpy(dtype=np.float32, copy=False)
            
            # Add rows to shuffle buffer
            shuffle_buf_features.append(feats)
            shuffle_buf_actions.append(acts)
            shuffle_buf_returns.append(rets)
            
            # Calculate current buffer size
            current_buffer_size = sum(len(a) for a in shuffle_buf_actions)
            
            # Once buffer is full, shuffle and yield batches
            while current_buffer_size >= self.buffer_size:
                # Concatenate buffered data
                all_features = np.concatenate(shuffle_buf_features, axis=0)
                all_actions = np.concatenate(shuffle_buf_actions, axis=0)
                all_returns = np.concatenate(shuffle_buf_returns, axis=0)
                
                # Take buffer_size samples
                n_samples = len(all_actions)
                take_size = min(self.buffer_size, n_samples)
                
                # Shuffle indices if requested
                if self.shuffle:
                    indices = np.arange(take_size)
                    rng.shuffle(indices)
                    buffer_features = all_features[indices]
                    buffer_actions = all_actions[indices]
                    buffer_returns = all_returns[indices]
                else:
                    buffer_features = all_features[:take_size]
                    buffer_actions = all_actions[:take_size]
                    buffer_returns = all_returns[:take_size]
                
                # Yield batches from shuffled buffer
                for i in range(0, take_size, self.batch_size):
                    end_idx = min(i + self.batch_size, take_size)
                    yield {
                        'features': torch.from_numpy(buffer_features[i:end_idx]),
                        'actions': torch.from_numpy(buffer_actions[i:end_idx]),
                        'returns': torch.from_numpy(buffer_returns[i:end_idx]),
                    }
                
                # Keep remaining data in buffer
                if n_samples > take_size:
                    shuffle_buf_features = [all_features[take_size:]]
                    shuffle_buf_actions = [all_actions[take_size:]]
                    shuffle_buf_returns = [all_returns[take_size:]]
                else:
                    shuffle_buf_features = []
                    shuffle_buf_actions = []
                    shuffle_buf_returns = []
                
                current_buffer_size = sum(len(a) for a in shuffle_buf_actions)
        
        # Process remaining buffer at end
        if shuffle_buf_features:
            all_features = np.concatenate(shuffle_buf_features, axis=0)
            all_actions = np.concatenate(shuffle_buf_actions, axis=0)
            all_returns = np.concatenate(shuffle_buf_returns, axis=0)
            
            n_samples = len(all_actions)
            
            # Shuffle if requested
            if self.shuffle and n_samples > 1:
                indices = np.arange(n_samples)
                rng.shuffle(indices)
                all_features = all_features[indices]
                all_actions = all_actions[indices]
                all_returns = all_returns[indices]
            
            # Yield remaining batches
            for i in range(0, n_samples, self.batch_size):
                end_idx = min(i + self.batch_size, n_samples)
                yield {
                    'features': torch.from_numpy(all_features[i:end_idx]),
                    'actions': torch.from_numpy(all_actions[i:end_idx]),
                    'returns': torch.from_numpy(all_returns[i:end_idx]),
                }
    

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
    Create a fast PyTorch DataLoader over Parquet files using a PyArrow-backed
    IterableDataset that yields fully-formed batches. This avoids Python-level
    per-row overhead and the small-files penalty from streaming shuffles.
    
    Args:
        data_dir: Directory containing .parquet files or path to single file
        batch_size: Training batch size
        split: 'train' or 'validation'
        shuffle: If True, shuffle files and rows within each file
        buffer_size: Unused in this implementation (kept for API compatibility)
        num_workers: DataLoader workers for parallel prefetching
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
    print(f"Loading dataset from {data_dir}")
    print(f"{'='*60}")
    
    # Collect files
    if data_path.is_file():
        all_files = [data_path]
        print(f"Single file mode: {data_path.name}")
    else:
        all_files = sorted(data_path.glob('*.parquet'))
        if max_files:
            all_files = all_files[:max_files]
        print(f"Found {len(all_files)} parquet files")
    
    # File-based split
    if split_by_files and len(all_files) > 1 and split in ['train', 'validation']:
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
        files = [all_files[i] for i in sorted(selected_indices)]
    else:
        files = all_files
    
    if not files:
        raise ValueError(f"No parquet files found in: {data_dir}")
    
    # Identify feature columns from schema of the first file
    sample_pf = ParquetFile(str(files[0]))
    schema_names = list(sample_pf.schema_arrow.names)
    feature_cols: List[str] = [c for c in schema_names if c.startswith('F_') or c.startswith('BT_')]
    return_col = value_type if value_type != 'RETURN' else 'RETURN'
    required_cols = feature_cols + ['ACTION', return_col]
    print(f"Features: {len(feature_cols)} columns")
    
    dataset = ParquetBatchIterable(
        files=files,
        required_cols=required_cols,
        feature_cols=feature_cols,
        return_col=return_col,
        batch_size=batch_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        seed=seed,
    )
    
    # Wrap in DataLoader for multi-worker prefetching; we yield complete batches
    # so we use batch_size=1 with an unwrapping collate_fn.
    def _unwrap_collate(items):
        return items[0]
    
    if num_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=num_workers,
            collate_fn=_unwrap_collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=_unwrap_collate,
            pin_memory=torch.cuda.is_available(),
        )
    
    print(f"DataLoader ready: batch_size={batch_size}, workers={num_workers}")
    print(f"{'='*60}\n")
    return loader


def estimate_steps_per_epoch(
    data_dir: str,
    split: str,
    batch_size: int,
    seed: int = 42,
    test_size: float = 0.2,
    max_files: Optional[int] = None,
    split_by_files: bool = True,
    max_workers: Optional[int] = None,
    use_processes: bool = False,
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
        max_workers: Parallel workers for metadata reads (default: min(32, 2*CPU) for threads, CPU count for processes)
        use_processes: If True, use ProcessPoolExecutor for true multi-core parallelism.
                       If False (default), use ThreadPoolExecutor (better for I/O-bound metadata reading)
    
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
    
    # Read metadata to get total rows (in parallel)
    paths = [str(p) for p in selected_files]
    
    def _read_num_rows(path: str) -> int:
        # Using read_metadata avoids constructing full ParquetFile objects
        return pq.read_metadata(path).num_rows
    
    # Use parallel processing for multiple files
    if len(paths) == 1:
        total_rows = _read_num_rows(paths[0])
    else:
        if use_processes:
            # ProcessPoolExecutor: True multi-core parallelism (each process has own GIL)
            # Better for CPU-bound work, but has process spawning overhead
            workers = max_workers or os.cpu_count() or 8
            with ProcessPoolExecutor(max_workers=workers) as pool:
                row_counts = list(pool.map(_read_num_rows, paths))
                total_rows = sum(row_counts)
            print(f"Read metadata from {len(paths)} files using {workers} processes")
        else:
            # ThreadPoolExecutor: Concurrent I/O (releases GIL during I/O operations)
            # Better for I/O-bound metadata reading from disk
            workers = max_workers or min(32, (os.cpu_count() or 8) * 2)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                row_counts = list(pool.map(_read_num_rows, paths))
                total_rows = sum(row_counts)
            print(f"Read metadata from {len(paths)} files using {workers} threads")
    
    # Calculate steps per epoch
    steps = int(np.ceil(total_rows / batch_size))
    
    return steps


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
    feature_cols = [c for c in first_sample.keys() if c.startswith('F_') or c.startswith('BT_')]
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
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
        print(f"Shuffling with buffer_size={buffer_size}")
    
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
    
    # Create PyTorch DataLoader
    # Note: num_workers=0 because HF Datasets handles parallelism internally
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    
    print(f"DataLoader ready: batch_size={batch_size}")
    print(f"Note: HF Datasets handles parallelism internally")
    print(f"{'='*60}\n")
    
    return loader

