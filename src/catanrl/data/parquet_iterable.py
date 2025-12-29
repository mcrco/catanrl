import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from pyarrow.parquet import ParquetFile
from torch.utils.data import DataLoader
from torchdata.datapipes import iter as dp_iter
from tqdm import tqdm

from ..features.catanatron_utils import is_non_graph_feature_parquet


def _gather_files(data_dir: Path, max_files: Optional[int]) -> List[Path]:
    if data_dir.is_file():
        return [data_dir]
    files = sorted(data_dir.glob('*.parquet'))
    if max_files:
        files = files[:max_files]
    return files


def _split_files(
    files: Sequence[Path],
    split: str,
    seed: int,
    test_size: float,
    split_by_files: bool,
) -> List[Path]:
    if not split_by_files or len(files) <= 1 or split not in {'train', 'validation'}:
        return list(files)

    rng = np.random.RandomState(seed)
    indices = np.arange(len(files))
    rng.shuffle(indices)

    n_val = max(1, int(len(files) * test_size))
    n_train = len(files) - n_val

    if split == 'train':
        selected = indices[:n_train]
    else:
        selected = indices[n_train:]

    return [files[i] for i in sorted(selected)]


def _infer_feature_columns(sample_file: Path) -> List[str]:
    sample_pf = ParquetFile(str(sample_file))
    schema_names = list(sample_pf.schema_arrow.names)
    return [c for c in schema_names if c.startswith('BT_') or is_non_graph_feature_parquet(c)]


def _iter_rows_from_file(
    file_path: Path,
    feature_cols: Sequence[str],
    return_col: str,
) -> Iterator[Tuple[np.ndarray, np.int64, np.float32]]:
    required_cols = list(feature_cols) + ['ACTION', return_col]
    pf = pq.ParquetFile(str(file_path))
    table = pf.read(columns=required_cols, use_threads=True)
    df = table.to_pandas()

    features = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    actions = df['ACTION'].to_numpy(dtype=np.int64, copy=False)
    returns = df[return_col].to_numpy(dtype=np.float32, copy=False)

    for idx in range(len(actions)):
        yield features[idx], actions[idx], returns[idx]


def _collate_batch(batch: List[Tuple[np.ndarray, np.int64, np.float32]]) -> dict:
    feature_block = np.asarray([row[0] for row in batch], dtype=np.float32)
    actions = np.asarray([row[1] for row in batch], dtype=np.int64)
    returns = np.asarray([row[2] for row in batch], dtype=np.float32)

    return {
        'features': torch.from_numpy(feature_block),
        'actions': torch.from_numpy(actions),
        'returns': torch.from_numpy(returns),
    }


def _build_parquet_datapipe(
    files: Sequence[Path],
    feature_cols: Sequence[str],
    return_col: str,
    batch_size: int,
    buffer_size: int,
    shuffle: bool,
    seed: int,
):
    files_dp = dp_iter.IterableWrapper([Path(f) for f in files])

    if shuffle and len(files) > 1:
        files_dp = files_dp.shuffle(buffer_size=len(files), seed=seed)

    files_dp = files_dp.sharding_filter()

    rows_dp = files_dp.flatmap(
        lambda file_path: _iter_rows_from_file(file_path, feature_cols, return_col)
    )

    if shuffle:
        effective_buffer = max(buffer_size, batch_size * 4)
        rows_dp = rows_dp.shuffle(buffer_size=effective_buffer, seed=seed)

    batches_dp = rows_dp.batch(batch_size=batch_size, drop_last=False)
    batches_dp = batches_dp.map(_collate_batch)

    return batches_dp


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
    Build a DataLoader backed by a torchdata DataPipe that streams Parquet batches.
    """
    data_path = Path(data_dir)
    print(f"\n{'=' * 60}")
    print(f"Loading dataset (DataPipe) from {data_dir}")
    print(f"{'=' * 60}")

    files = _gather_files(data_path, max_files)
    if not files:
        raise ValueError(f"No parquet files found in: {data_dir}")

    files = _split_files(files, split, seed, test_size, split_by_files)
    print(f"Using {len(files)} parquet files for split='{split}'")

    feature_cols = _infer_feature_columns(files[0])
    return_col = value_type if value_type != 'RETURN' else 'RETURN'
    print(f"Identified {len(feature_cols)} feature columns (torchdata)")

    datapipe = _build_parquet_datapipe(
        files=files,
        feature_cols=feature_cols,
        return_col=return_col,
        batch_size=batch_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        seed=seed,
    )

    loader_kwargs = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(datapipe, **loader_kwargs)
    print(f"DataLoader ready (torchdata pipeline): batch_size={batch_size}, workers={num_workers}")
    print(f"{'=' * 60}\n")
    return loader


def _read_parquet_num_rows(path: str) -> int:
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
    data_path = Path(data_dir)
    files = _gather_files(data_path, max_files)
    files = _split_files(files, split, seed, test_size, split_by_files)
    paths = [str(p) for p in files]

    if not paths:
        raise ValueError(f"No parquet files found in: {data_dir}")

    if len(paths) == 1:
        total_rows = _read_parquet_num_rows(paths[0])
    else:
        if use_processes:
            workers = max_workers or os.cpu_count() or 8
            executor = ProcessPoolExecutor(max_workers=workers)
        else:
            workers = max_workers or min(32, (os.cpu_count() or 8) * 2)
            executor = ThreadPoolExecutor(max_workers=workers)

        with executor as pool:
            row_counts = list(
                tqdm(
                    pool.map(_read_parquet_num_rows, paths),
                    total=len(paths),
                    desc=f"Reading metadata to estimate steps per {split} epoch (torchdata)",
                    unit="file",
                )
            )
            total_rows = sum(row_counts)

    return int(np.ceil(total_rows / batch_size))

