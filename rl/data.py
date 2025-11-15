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
from tqdm import tqdm


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
        
        # Use a fixed-size reservoir buffer to avoid repeated large concatenations
        num_features = len(self.feature_cols)
        buf_features = np.empty((self.buffer_size, num_features), dtype=np.float32)
        buf_actions = np.empty((self.buffer_size,), dtype=np.int64)
        buf_returns = np.empty((self.buffer_size,), dtype=np.float32)
        filled = 0
        rng_np = np.random.RandomState(self.seed + worker_id)
        
        for fp in worker_files:
            # Read only required columns from file
            pf = pq.ParquetFile(str(fp))
            table = pf.read(columns=self.required_cols, use_threads=True)
            # Convert to pandas for efficient 2D numpy block extraction
            df = table.to_pandas()  # zero-copy for numeric types where possible
            
            feats = df[self.feature_cols].to_numpy(dtype=np.float32, copy=False)
            acts = df['ACTION'].to_numpy(dtype=np.int64, copy=False)
            rets = df[self.return_col].to_numpy(dtype=np.float32, copy=False)
            
            # Stream rows into the fixed-size reservoir buffer
            offset = 0
            n_rows = len(acts)
            while offset < n_rows:
                take = min(self.buffer_size - filled, n_rows - offset)
                buf_features[filled:filled + take] = feats[offset:offset + take]
                buf_actions[filled:filled + take] = acts[offset:offset + take]
                buf_returns[filled:filled + take] = rets[offset:offset + take]
                filled += take
                offset += take
                
                # Once buffer is full, (optionally) shuffle and yield batches
                if filled == self.buffer_size:
                    if self.shuffle and filled > 1:
                        perm = rng_np.permutation(filled)
                        buffer_features = buf_features[perm]
                        buffer_actions = buf_actions[perm]
                        buffer_returns = buf_returns[perm]
                    else:
                        buffer_features = buf_features[:filled]
                        buffer_actions = buf_actions[:filled]
                        buffer_returns = buf_returns[:filled]
                    
                    for i in range(0, filled, self.batch_size):
                        end_idx = min(i + self.batch_size, filled)
                        yield {
                            'features': torch.from_numpy(buffer_features[i:end_idx]),
                            'actions': torch.from_numpy(buffer_actions[i:end_idx]),
                            'returns': torch.from_numpy(buffer_returns[i:end_idx]),
                        }
                    filled = 0
        
        # Process remaining buffer at end
        if filled > 0:
            if self.shuffle and filled > 1:
                perm = rng_np.permutation(filled)
                buffer_features = buf_features[:filled][perm]
                buffer_actions = buf_actions[:filled][perm]
                buffer_returns = buf_returns[:filled][perm]
            else:
                buffer_features = buf_features[:filled]
                buffer_actions = buf_actions[:filled]
                buffer_returns = buf_returns[:filled]
            
            # Yield remaining batches
            for i in range(0, filled, self.batch_size):
                end_idx = min(i + self.batch_size, filled)
                yield {
                    'features': torch.from_numpy(buffer_features[i:end_idx]),
                    'actions': torch.from_numpy(buffer_actions[i:end_idx]),
                    'returns': torch.from_numpy(buffer_returns[i:end_idx]),
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
    prefetch_factor: int = 2,
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
    # Keep BT_* and non-graph F_* only (exclude F_NODE*, F_EDGE*, F_TILE*, F_PORT*)
    def _is_non_graph_feature(col: str) -> bool:
        if not col.startswith('F_'):
            return False
        return not (
            col.startswith('F_NODE')
            or col.startswith('F_EDGE')
            or col.startswith('F_TILE')
            or col.startswith('F_PORT')
        )
    feature_cols: List[str] = [
        c for c in schema_names if c.startswith('BT_') or _is_non_graph_feature(c)
    ]
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
            persistent_workers=False,
            prefetch_factor=prefetch_factor,
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


def _read_parquet_num_rows(path: str) -> int:
    """Helper function to read number of rows from a parquet file.
    
    Must be at module level for ProcessPoolExecutor pickling.
    """
    return pq.read_metadata(path).num_rows


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
    
    # Use parallel processing for multiple files
    if len(paths) == 1:
        total_rows = _read_parquet_num_rows(paths[0])
    else:
        if use_processes:
            workers = max_workers or os.cpu_count() or 8
            with ProcessPoolExecutor(max_workers=workers) as pool:
                row_counts = list(tqdm(
                    pool.map(_read_parquet_num_rows, paths),
                    total=len(paths),
                    desc=f"Reading metadata to estimate steps per {split} epoch",
                    unit="file"
                ))
                total_rows = sum(row_counts)
        else:
            workers = max_workers or min(32, (os.cpu_count() or 8) * 2)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                row_counts = list(tqdm(
                    pool.map(_read_parquet_num_rows, paths),
                    total=len(paths),
                    desc=f"Reading metadata to estimate steps per {split} epoch",
                    unit="file"
                ))
                total_rows = sum(row_counts)
    
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
    # Keep BT_* and non-graph F_* only (exclude F_NODE*, F_EDGE*, F_TILE*, F_PORT*)
    def _is_non_graph_feature(col: str) -> bool:
        if not col.startswith('F_'):
            return False
        return not (
            col.startswith('F_NODE')
            or col.startswith('F_EDGE')
            or col.startswith('F_TILE')
            or col.startswith('F_PORT')
        )
    feature_cols = [c for c in first_sample.keys() if c.startswith('BT_') or _is_non_graph_feature(c)]
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

