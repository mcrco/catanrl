#!/usr/bin/env python3
"""
Shard many small parquet files into fewer large parquet files for faster training.
Parallelizable across multiple workers.

Usage:
    python shard_parquet.py --data-dir data/games --rows-per-shard 10000 --num-workers 8
"""

import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np


def count_rows(path):
    """Count rows in a parquet file by reading metadata."""
    try:
        return pq.read_metadata(str(path)).num_rows
    except:
        return 0


def shard_file_chunk(
    files: List[Path],
    output_dir: Path,
    rows_per_shard: int,
    shard_offset: int,
    worker_id: int,
) -> int:
    """
    Process a chunk of files assigned to a worker using memory-efficient PyArrow Tables.
    
    The algorithm keeps at most one partial buffer (< rows_per_shard rows) in memory
    and streams rows from each file to fill and flush shards. This avoids repeated
    concatenations and large intermediate pandas DataFrames.
    
    Args:
        files: List of parquet files to process
        output_dir: Directory to write sharded files
        rows_per_shard: Target rows per output shard
        shard_offset: Starting shard number for this worker
        worker_id: Worker ID for progress tracking
        
    Returns:
        Number of shards created by this worker
    """
    shard_num = shard_offset
    shards_created = 0
    
    # Partial buffer across files (PyArrow Table or None)
    buffer_table = None
    base_schema = None  # Ensure consistent column order
    
    for file_path in tqdm(files, desc=f"Worker {worker_id}", position=worker_id, leave=False):
        try:
            # Read parquet file as Arrow Table (fast, zero-copy when possible)
            table = pq.read_table(str(file_path), memory_map=True)
            if table.num_rows == 0:
                continue
            
            # Initialize/align schema and column order
            if base_schema is None:
                base_schema = table.schema
            else:
                # Reorder columns to match base schema (assumes compatible schemas)
                table = table.select(base_schema.names)
            
            # Fast path: if we already have buffered rows, try to fill a shard
            if buffer_table is not None and buffer_table.num_rows > 0:
                need = rows_per_shard - buffer_table.num_rows
                if table.num_rows >= need:
                    # Complete shard from buffer + slice of current table
                    shard_table = pa.concat_tables([buffer_table, table.slice(0, need)], promote_options="default")
                    output_path = output_dir / f"shard_{shard_num:06d}.parquet"
                    pq.write_table(shard_table, output_path, compression="snappy", data_page_version="2.0", write_statistics=True, row_group_size=rows_per_shard)
                    shard_num += 1
                    shards_created += 1
                    
                    # Advance current table and clear buffer
                    table = table.slice(need)
                    buffer_table = None
                else:
                    # Not enough to complete a shard; just extend the buffer and continue
                    buffer_table = pa.concat_tables([buffer_table, table], promote_options="default")
                    continue
            
            # Now buffer is empty. We may be able to write multiple full shards from 'table'
            while table.num_rows >= rows_per_shard:
                shard_table = table.slice(0, rows_per_shard)
                output_path = output_dir / f"shard_{shard_num:06d}.parquet"
                pq.write_table(shard_table, output_path, compression="snappy", data_page_version="2.0", write_statistics=True, row_group_size=rows_per_shard)
                shard_num += 1
                shards_created += 1
                table = table.slice(rows_per_shard)
            
            # Any remaining rows go into the buffer (less than rows_per_shard)
            if table.num_rows > 0:
                buffer_table = table if buffer_table is None else pa.concat_tables([buffer_table, table], promote_options="default")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Write final partial shard if any data remains in buffer
    if buffer_table is not None and buffer_table.num_rows > 0:
        output_path = output_dir / f"shard_{shard_num:06d}.parquet"
        pq.write_table(buffer_table, output_path, compression="snappy", data_page_version="2.0", write_statistics=True, row_group_size=min(rows_per_shard, buffer_table.num_rows))
        shards_created += 1
    
    return shards_created


def main():
    parser = argparse.ArgumentParser(
        description='Shard many small parquet files into fewer large shards for faster training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shard with 10k rows per shard using 8 workers
  python shard_parquet.py --data-dir data/games --rows-per-shard 10000 --num-workers 8
  
  # Shard to custom output directory
  python shard_parquet.py --data-dir data/games --rows-per-shard 5000 --output-dir data/custom_shards
  
  # Quick test with 100 files
  python shard_parquet.py --data-dir data/games --rows-per-shard 10000 --max-files 100
"""
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing input parquet files'
    )
    parser.add_argument(
        '--rows-per-shard',
        type=int,
        default=10000,
        help='Target number of rows per output shard (default: 10000)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of parallel workers (default: 8)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for sharded files (default: {data_dir}_sharded)'
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of input files to process (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print stats without creating shards'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: same parent dir with _sharded suffix
        parent = data_dir.parent
        dir_name = data_dir.name
        output_dir = parent / f"{dir_name}_sharded"
    
    print(f"\n{'='*60}")
    print(f"Parquet Sharding")
    print(f"{'='*60}")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rows per shard: {args.rows_per_shard:,}")
    print(f"Workers: {args.num_workers}")
    
    # Collect input files
    input_files = sorted(data_dir.glob('*.parquet'))
    if args.max_files:
        input_files = input_files[:args.max_files]
    
    if not input_files:
        print(f"Error: No parquet files found in {data_dir}")
        return
    
    print(f"Found {len(input_files):,} input files")
    
    # Estimate output stats
    print("\nAnalyzing input files...")
    
    # Sample a few files to estimate total rows
    sample_size = min(100, len(input_files))
    sample_files = np.random.choice(input_files, size=sample_size, replace=False)
    
    with ProcessPoolExecutor(max_workers=min(8, args.num_workers)) as pool:
        sample_rows = list(pool.map(count_rows, sample_files))
    
    avg_rows_per_file = np.mean([r for r in sample_rows if r > 0])
    estimated_total_rows = int(avg_rows_per_file * len(input_files))
    estimated_shards = int(np.ceil(estimated_total_rows / args.rows_per_shard))
    
    print(f"  Avg rows per file: {avg_rows_per_file:.1f}")
    print(f"  Estimated total rows: {estimated_total_rows:,}")
    print(f"  Estimated output shards: {estimated_shards:,}")
    print(f"  Estimated compression: {len(input_files) / max(1, estimated_shards):.1f}x fewer files")
    
    if args.dry_run:
        print("\nDry run complete. No files created.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Partition files across workers
    files_per_worker = int(np.ceil(len(input_files) / args.num_workers))
    worker_tasks = []
    
    for worker_id in range(args.num_workers):
        start_idx = worker_id * files_per_worker
        end_idx = min(start_idx + files_per_worker, len(input_files))
        
        if start_idx >= len(input_files):
            break
        
        worker_files = input_files[start_idx:end_idx]
        # Shard offset ensures unique shard numbers across workers
        shard_offset = worker_id * 10000  # Large enough gap to avoid collisions
        
        worker_tasks.append((worker_files, output_dir, args.rows_per_shard, shard_offset, worker_id))
    
    print(f"\nSharding with {len(worker_tasks)} workers...")
    print(f"  Files per worker: ~{files_per_worker:,}")
    
    # Execute sharding in parallel
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(shard_file_chunk, files, out_dir, rows, offset, wid)
                for files, out_dir, rows, offset, wid in worker_tasks
            ]
            
            # Wait for completion
            results = [f.result() for f in futures]
    else:
        # Single worker mode
        results = [shard_file_chunk(*worker_tasks[0])]
    
    total_shards = sum(results)
    
    print(f"\n{'='*60}")
    print(f"Sharding Complete!")
    print(f"{'='*60}")
    print(f"Created {total_shards:,} shards in {output_dir}")
    print(f"Compression: {len(input_files):,} → {total_shards:,} files ({len(input_files)/max(1, total_shards):.1f}x)")
    
    # Verify output
    output_files = sorted(output_dir.glob('*.parquet'))
    if len(output_files) != total_shards:
        print(f"Warning: Expected {total_shards} files but found {len(output_files)}")
    
    # Sample a few shards to verify
    if output_files:
        sample_shard = output_files[len(output_files) // 2]
        sample_df = pd.read_parquet(sample_shard)
        print(f"\nSample shard: {sample_shard.name}")
        print(f"  Shape: {sample_df.shape}")
        print(f"  Columns: {list(sample_df.columns[:5])}...")
    
    print(f"\nYou can now train with: --data-dir {output_dir}")


if __name__ == "__main__":
    main()

