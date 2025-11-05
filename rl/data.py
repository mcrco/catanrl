import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from typing import List as TList


class CatanDataset(Dataset):
    """
    Dataset that provides file metadata and loading utilities for interleaved batch loading.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        value_type: str = 'RETURN',
        use_pyarrow: bool = True,
        validate_files: bool = False
    ):
        """
        Args:
            data_dir: Directory containing .parquet files
            max_files: Maximum number of files to use (None = all)
            use_pyarrow: Use PyArrow for faster loading
            validate_files: Check parquet files for corruption by reading metadata
                before use (default False). Set to True to check upfront.
            value_type: Which return column to use (default 'RETURN')
        """
        self.data_dir = Path(data_dir)
        self.use_pyarrow = use_pyarrow
        self.value_type = value_type
        self.validate_files = validate_files
        
        print(f"Scanning parquet files in {data_dir}...")
        
        # Get all parquet files
        if self.data_dir.is_file():
            # Single file mode
            all_files = [self.data_dir]
        else:
            # Directory mode
            all_files = sorted(self.data_dir.glob('*.parquet'))
            
            if max_files:
                all_files = all_files[:max_files]
        
        print(f"Found {len(all_files)} parquet files")
        
        # Resolve parquet files (optionally skip integrity checks)
        if self.data_dir.is_file():
            if self.validate_files:
                try:
                    pq.read_metadata(str(self.data_dir))
                except Exception as e:
                    raise ValueError(f"Parquet file failed integrity check: {self.data_dir} ({type(e).__name__}: {e})")
            self.parquet_files = [self.data_dir]
        else:
            self.parquet_files, _ = get_valid_parquet_files(self.data_dir, check_integrity=self.validate_files)
            if max_files:
                self.parquet_files = self.parquet_files[:max_files]

        print(f"Valid files: {len(self.parquet_files)} / {len(all_files)}")
        
        if len(self.parquet_files) == 0:
            raise ValueError("No valid parquet files found!")
        
        # Get metadata from first file to determine schema
        print("Reading schema from first file...")
        first_df = self.load_file(self.parquet_files[0])
        self.feature_cols = [c for c in first_df.columns if c.startswith('F_')] + [c for c in first_df.columns if c.startswith('BT_')]
        return_col = self.value_type if self.value_type != 'RETURN' else 'RETURN'
        self.required_columns = self.feature_cols + ['ACTION', return_col]
        
        # Count total rows for __len__
        print("Counting total rows...")
        self.total_samples = sum(
            pq.read_metadata(str(f)).num_rows 
            for f in tqdm(self.parquet_files, desc="Scanning files")
        )
        
        print(f"\n{'='*60}")
        print("Dataset Ready")
        print(f"{'='*60}")
        print(f"Total samples: {self.total_samples:,}")
        print(f"Total files: {len(self.parquet_files)}")
        print(f"Features: {len(self.feature_cols)} columns")
        print("Use InterleavedFileLoader for efficient batch loading")
        print(f"{'='*60}\n")
    
    def load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a parquet file into memory.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            DataFrame containing the file data
        """
        try:
            columns = getattr(self, 'required_columns', None)
            if self.use_pyarrow:
                table = pq.read_table(file_path, columns=columns)
                df = table.to_pandas()
            else:
                if columns is not None:
                    df = pd.read_parquet(file_path, columns=columns)
                else:
                    df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            print(f"⚠️  ERROR LOADING FILE: {file_path}")
            print(f"    Error: {type(e).__name__}: {e}")
            raise
    
    def __len__(self):
        """Return total number of samples across all files."""
        return self.total_samples

class InterleavedFileLoader:
    """
    Loads data by keeping N files in memory, interleaving rows uniformly from all files.
    When a file is exhausted, it loads the next file from the shuffled file list.
    File order is re-shuffled at the start of each epoch.
    """
    
    def __init__(
        self,
        dataset: CatanDataset,
        batch_size: int,
        num_files_loaded: int = 4,
        drop_last: bool = False,
        file_indices: Optional[TList[int]] = None
    ):
        """
        Args:
            dataset: CatanDataset instance
            batch_size: Number of samples per batch
            num_files_loaded: Number of files to keep loaded simultaneously
            drop_last: Whether to drop the last incomplete batch
            file_indices: Optional list of file indices to use (for train/val split).
                         If None, uses all files from dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Select which files to use
        if file_indices is not None:
            self.file_list = [dataset.parquet_files[i] for i in file_indices]
        else:
            self.file_list = dataset.parquet_files
        
        self.num_files_loaded = min(num_files_loaded, len(self.file_list))
        self.drop_last = drop_last
        
        # Will be set at epoch start
        self.file_order: TList[Path] = []
        self.file_order_idx = 0  # Index into file_order for next file to load
        
        # Pool of currently loaded files
        # Each entry: {'data': dict with arrays, 'indices': shuffled row indices, 'cursor': current position}
        self.loaded_files: TList[Optional[Dict]] = [None] * self.num_files_loaded
        
        print(f"InterleavedFileLoader: batch_size={batch_size}, num_files_loaded={num_files_loaded}, total_files={len(self.file_list)}")
    
    def start_epoch(self):
        """
        Initialize a new epoch: shuffle file order and load initial file pool.
        """
        # Shuffle file order
        self.file_order = self.file_list.copy()
        rng = np.random.default_rng()
        rng.shuffle(self.file_order)
        self.file_order_idx = 0
        
        # Load initial pool of files
        for i in range(self.num_files_loaded):
            if self.file_order_idx < len(self.file_order):
                self._load_file_into_slot(i)
    
    def _load_file_into_slot(self, slot_idx: int):
        """
        Load the next file from file_order into the specified slot.
        """
        if self.file_order_idx >= len(self.file_order):
            # No more files to load
            self.loaded_files[slot_idx] = None
            return
        
        file_path = self.file_order[self.file_order_idx]
        self.file_order_idx += 1
        
        # Load file as dataframe
        df = self.dataset.load_file(file_path)
        
        # Convert to numpy arrays for faster access
        return_col = self.dataset.value_type if self.dataset.value_type != 'RETURN' else 'RETURN'
        features_arr = df[self.dataset.feature_cols].values.astype(np.float32, copy=False)
        action_arr = df['ACTION'].values
        return_arr = df[return_col].values.astype(np.float32, copy=False)
        
        # Create shuffled indices for this file
        num_rows = len(df)
        indices = np.arange(num_rows)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        
        self.loaded_files[slot_idx] = {
            'features': features_arr,
            'action': action_arr,
            'return': return_arr,
            'indices': indices,
            'cursor': 0
        }
    
    def __iter__(self):
        """
        Iterate through batches by interleaving rows from loaded files.
        """
        while True:
            # Check how many files are still available
            active_files = [f for f in self.loaded_files if f is not None and f['cursor'] < len(f['indices'])]
            
            if len(active_files) == 0:
                # All files exhausted
                break
            
            # Build a batch by taking samples uniformly from active files
            batch_features = []
            batch_actions = []
            batch_returns = []
            
            samples_collected = 0
            
            # Round-robin through active files to collect samples
            file_idx = 0
            while samples_collected < self.batch_size:
                # Find next active file
                slot_idx = 0
                attempts = 0
                while slot_idx < len(self.loaded_files) and attempts < len(self.loaded_files):
                    file_data = self.loaded_files[(file_idx + attempts) % len(self.loaded_files)]
                    if file_data is not None and file_data['cursor'] < len(file_data['indices']):
                        slot_idx = (file_idx + attempts) % len(self.loaded_files)
                        break
                    attempts += 1
                
                if attempts >= len(self.loaded_files):
                    # No more active files
                    break
                
                file_data = self.loaded_files[slot_idx]
                
                # Take one sample from this file
                idx = file_data['indices'][file_data['cursor']]
                file_data['cursor'] += 1
                
                batch_features.append(file_data['features'][idx])
                batch_actions.append(file_data['action'][idx])
                batch_returns.append(file_data['return'][idx])
                
                samples_collected += 1
                file_idx = (slot_idx + 1) % len(self.loaded_files)
                
                # If this file is exhausted, load next file into this slot
                if file_data['cursor'] >= len(file_data['indices']):
                    self._load_file_into_slot(slot_idx)
            
            # Check if we have a complete batch or should drop it
            if len(batch_features) < self.batch_size and self.drop_last:
                break
            
            if len(batch_features) == 0:
                break
            
            # Convert to tensors and normalize returns
            features_tensor = torch.from_numpy(np.array(batch_features, dtype=np.float32))
            actions_tensor = torch.tensor(batch_actions, dtype=torch.long)
            returns_normalized = (np.array(batch_returns, dtype=np.float32) + 1000) / 2010
            returns_tensor = torch.from_numpy(returns_normalized)
            
            yield features_tensor, actions_tensor, returns_tensor


def get_valid_parquet_files(input_dir: str, check_integrity: bool = True) -> Tuple[TList[Path], int]:
    """
    Check if all parquet files in a directory are valid.

    Args:
        input_dir: Directory containing .parquet files
        check_integrity: If False, skip reading metadata and return all files
    
    Returns:
        Tuple of (valid_files, num_total_files)
    """
    input_path = Path(input_dir)
    parquet_files: TList[Path] = sorted(input_path.glob('*.parquet'))
    num_total_files = len(parquet_files)

    if not check_integrity:
        return parquet_files, num_total_files

    valid_files: TList[Path] = []
    for f in tqdm(parquet_files, desc="Checking files", unit="file"):
        try:
            pq.read_metadata(str(f))
            valid_files.append(f)
        except Exception as e:
            print(f"⚠️  SKIPPING CORRUPTED FILE: {f}")
            print(f"    Error: {type(e).__name__}: {e}")
    return valid_files, num_total_files