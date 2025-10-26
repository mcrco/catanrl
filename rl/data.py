import torch
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict
from tqdm import tqdm
from collections import OrderedDict


class CatanDataset(Dataset):
    """
    Memory-efficient dataset that loads parquet files on-demand with LRU caching.
    """
    
    def __init__(
        self,
        data_dir: str,
        max_files: Optional[int] = None,
        cache_size: int = 10,
        value_type: str = 'RETURN',
        use_pyarrow: bool = True
    ):
        """
        Args:
            data_dir: Directory containing .parquet files
            max_files: Maximum number of files to use (None = all)
            cache_size: Number of files to keep in memory (LRU cache)
            use_pyarrow: Use PyArrow for faster loading
        """
        self.data_dir = Path(data_dir)
        self.cache_size = cache_size
        self.use_pyarrow = use_pyarrow
        self.value_type = value_type
        # Columns to load from parquet files once discovered
        self.required_columns = None
        
        # File cache (LRU)
        self._file_cache: OrderedDict[str, pd.DataFrame] = OrderedDict()
        
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
        
        # Filter out corrupted files
        self.parquet_files, _ = get_valid_parquet_files(self.data_dir)
        
        print(f"Valid files: {len(self.parquet_files)} / {len(all_files)}")
        
        if len(self.parquet_files) == 0:
            raise ValueError("No valid parquet files found!")
        
        # Build index mapping
        print("Building index mapping...")
        self._build_index_mapping()
        
        # Get metadata from first file
        first_df = self._load_file(self.parquet_files[0])
        self.feature_cols = [c for c in first_df.columns if c.startswith('F_')] + [c for c in first_df.columns if c.startswith('BT_')]
        # From now on, only load the columns we actually use
        return_col = self.value_type if self.value_type != 'RETURN' else 'RETURN'
        self.required_columns = self.feature_cols + ['ACTION', return_col]
        
        print(f"\n{'='*60}")
        print("Lazy-Loading Dataset Ready")
        print(f"{'='*60}")
        print(f"Total samples: {self.total_samples:,}")
        print(f"Total files: {len(self.parquet_files)}")
        print(f"Features: {len(self.feature_cols)} columns")
        print(f"Cache size: {cache_size} files")
        print(f"{'='*60}\n")
    
    def _build_index_mapping(self):
        """
        Build mapping from global index to (file_index, local_index).
        Uses cumulative row counts for O(1) lookups with binary search.
        """
        file_row_counts = []
        for file in tqdm(self.parquet_files, desc="Counting rows"):
            metadata = pq.read_metadata(str(file))
            file_row_counts.append(metadata.num_rows)
        
        # Build cumulative offsets for binary search
        # E.g., [0, 1000, 2500, 4000] means:
        #   - File 0: indices [0, 1000)
        #   - File 1: indices [1000, 2500)
        #   - File 2: indices [2500, 4000)
        self.file_offsets = np.cumsum([0] + file_row_counts)
        self.total_samples = int(self.file_offsets[-1])
        
        print(f"Total samples: {self.total_samples:,}")
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a parquet file with LRU caching.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            DataFrame containing the file data
        """
        file_str = str(file_path)
        
        # Check if file is in cache
        if file_str in self._file_cache:
            # Move to end (most recently used)
            self._file_cache.move_to_end(file_str)
            return self._file_cache[file_str]
        
        # Load file with error handling
        try:
            # After initialization, restrict to only needed columns to reduce RAM
            columns = getattr(self, 'required_columns', None)
            if self.use_pyarrow:
                table = pq.read_table(file_path, columns=columns)
                df = table.to_pandas()
            else:
                if columns is not None:
                    df = pd.read_parquet(file_path, columns=columns)
                else:
                    df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"⚠️  ERROR LOADING FILE: {file_path}")
            print(f"    Error: {type(e).__name__}: {e}")
            raise
        
        # Add to cache
        self._file_cache[file_str] = df
        
        # Evict oldest if cache is full
        if len(self._file_cache) > self.cache_size:
            self._file_cache.popitem(last=False)  # Remove oldest (first item)
        
        return df
    
    def _get_file_and_local_index(self, idx: int) -> Tuple[Path, int]:
        """
        Map global index to (file_path, local_index) using binary search.
        
        Args:
            idx: Global index in range [0, total_samples)
            
        Returns:
            (file_path, local_index_in_file)
        """
        # Binary search to find which file contains this index
        file_idx = np.searchsorted(self.file_offsets[1:], idx, side='right')
        
        # Compute local index within that file
        local_idx = idx - self.file_offsets[file_idx]
        
        return self.parquet_files[file_idx], local_idx
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get a single sample by global index.
        
        Args:
            idx: Global index in range [0, total_samples)
            
        Returns:
            (features, action, normalized_return)
        """
        # Map to file and local index
        file_path, local_idx = self._get_file_and_local_index(idx)
        
        # Load file (from cache if available)
        df = self._load_file(file_path)
        
        # Extract sample
        features = df[self.feature_cols].iloc[local_idx].values.astype(np.float32)
        action = df['ACTION'].iloc[local_idx]
        return_val = df[self.value_type].iloc[local_idx] if self.value_type != 'RETURN' else df['RETURN'].iloc[local_idx]
        
        # Normalize return
        return_normalized = (return_val + 1000) / 2010
        
        return (
            torch.from_numpy(features),
            int(action),
            np.float32(return_normalized)
        )
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for debugging."""
        return {
            'cache_size': len(self._file_cache),
            'max_cache_size': self.cache_size,
            'total_files': len(self.parquet_files),
        }

def get_valid_parquet_files(input_dir: str) -> Tuple[int, int]:
    """
    Check if all parquet files in a directory are valid.

    Args:
        input_dir: Directory containing .parquet files
    
    Returns:
        Tuple of (valid_files, num_total_files)
    """
    valid_files = []
    num_total_files = 0
    input_path = Path(input_dir)
    parquet_files = sorted(input_path.glob('*.parquet'))
    for f in tqdm(parquet_files, desc="Checking files", unit="file"):
        num_total_files += 1
        try:
            pq.read_metadata(str(f))
            valid_files.append(f)
        except Exception as e:
            print(f"⚠️  SKIPPING CORRUPTED FILE: {f}")
            print(f"    Error: {type(e).__name__}: {e}")
    return valid_files, num_total_files


def merge_parquet_files(input_dir: str, output_file: str = None) -> str:
    """
    Merge all parquet files in a directory into a single file.
    
    Args:
        input_dir: Directory containing .parquet files
        output_file: Output merged file path (default: {input_dir}_merged.parquet)
    
    Returns:
        Path to the created merged file
    """
    try:
        input_path = Path(input_dir)
        
        if output_file is None:
            output_file = f"{input_dir.rstrip('/')}_merged.parquet"
        
        print(f"Input directory: {input_dir}", flush=True)
        print(f"Output file: {output_file}", flush=True)
        
        # Get all parquet files
        valid_files, num_total_files = get_valid_parquet_files(input_dir)
        print(f"Found {num_total_files} parquet files", flush=True)
        
        if num_total_files == 0:
            print("No parquet files found!", flush=True)
            return None
        
        # Use PyArrow dataset for efficient reading
        print("Creating dataset from valid files...", flush=True)
        dataset = ds.dataset(
            [str(f) for f in valid_files],
            format='parquet'
        )
        
        # Get schema from first batch
        print("Reading schema...", flush=True)
        schema = dataset.schema
        print(f"Schema: {len(schema)} columns", flush=True)
        
        # Write output file
        print("Merging files (writing batches)...", flush=True)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = None
        total_rows = 0
        batch_count = 0
        
        try:
            # Stream batches directly to output file
            for batch in tqdm(dataset.to_batches(), desc="Writing batches", unit="batch"):
                if writer is None:
                    # Create writer on first batch
                    writer = pq.ParquetWriter(
                        str(output_path),
                        schema=batch.schema,
                        compression='snappy',
                        use_dictionary=True
                    )
                
                writer.write_batch(batch)
                total_rows += batch.num_rows
                batch_count += 1
        finally:
            # Always close the writer, even if there's an exception
            if writer is not None:
                writer.close()
        
        # Verify the merged file
        if output_path.exists():
            merged_size = output_path.stat().st_size / 1e9
            print(f"\n✓ Success! Created {output_file}", flush=True)
            print(f"✓ Total rows written: {total_rows:,}", flush=True)
            print(f"✓ Total batches: {batch_count:,}", flush=True)
            print(f"✓ File size: {merged_size:.2f} GB", flush=True)
            print(f"\nTo use this in training, update DATA_DIR in train.py to: '{output_file}'", flush=True)
            print("And the dataset will load instantly!", flush=True)
            return str(output_path)
        else:
            print(f"✗ ERROR: File was not created at {output_file}", flush=True)
            return None
            
    except Exception as e:
        print(f"✗ ERROR during merge: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


def create_sharded_files(input_dir: str, num_shards: int = 10, output_dir: str = None) -> str:
    """
    Split data into N sharded files for balanced parallel loading.
    Useful if single file is too large for memory.
    
    Args:
        input_dir: Directory containing .parquet files
        num_shards: Number of output shards to create
        output_dir: Output directory (default: {input_dir}_sharded)
    
    Returns:
        Path to the created output directory
    """
    try:
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_dir = f"{input_dir.rstrip('/')}_sharded"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"Input directory: {input_dir}", flush=True)
        print(f"Output directory: {output_dir}", flush=True)
        print(f"Number of shards: {num_shards}", flush=True)
        
        # Get all parquet files
        valid_files, num_total_files = get_valid_parquet_files(input_dir)
        print(f"Found {num_total_files} parquet files", flush=True)
        
        print(f"Valid files: {len(valid_files)} / {num_total_files}", flush=True)
        
        if num_total_files == 0:
            print("No valid parquet files found!", flush=True)
            return None
        
        # Count total rows from metadata (fast)
        print("Counting total rows from metadata...", flush=True)
        total_rows = 0
        for f in tqdm(valid_files, desc="Counting rows"):
            total_rows += pq.read_metadata(str(f)).num_rows

        rows_per_shard = -(-total_rows // num_shards)  # Ceiling division
        print(f"Total rows to be sharded: {total_rows:,}", flush=True)
        print(f"Target rows per shard: ~{rows_per_shard:,}", flush=True)
        
        # Read all data with progress bar
        print("Creating dataset from valid files...", flush=True)
        dataset = ds.dataset([str(f) for f in valid_files], format='parquet')
        
        # Get schema from the dataset
        schema = dataset.schema
        
        # Write shards sequentially to avoid too many open files
        print(f"Writing {num_shards} shards sequentially...", flush=True)
        
        total_rows_written = 0
        batch_count = 0
        shard_idx = 0
        rows_in_current_shard = 0
        writer = None

        try:
            for batch in tqdm(dataset.to_batches(batch_size=65_536), desc="Writing batches", unit="batch"):
                if writer is None:
                    # Start a new shard
                    output_file = output_path / f"shard_{shard_idx:04d}.parquet"
                    writer = pq.ParquetWriter(
                        str(output_file),
                        schema=schema,
                        compression='snappy'
                    )
                    rows_in_current_shard = 0

                writer.write_batch(batch)
                rows_in_current_shard += batch.num_rows
                total_rows_written += batch.num_rows
                batch_count += 1

                # Close current shard if it's full and not the last shard
                if rows_in_current_shard >= rows_per_shard and shard_idx < num_shards - 1:
                    writer.close()
                    writer = None
                    shard_idx += 1
        finally:
            if writer is not None:
                writer.close()
                
        print(f"\n✓ Success! Created {shard_idx + 1} shards in {output_dir}", flush=True)
        print(f"✓ Total rows written: {total_rows_written:,}", flush=True)
        print(f"✓ Total batches: {batch_count:,}", flush=True)
        print(f"To use in training, update DATA_DIR in train.py to: '{output_dir}'", flush=True)
        return str(output_path)
    
    except Exception as e:
        print(f"✗ ERROR during sharding: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None