#!/usr/bin/env python3
"""
Shuffle and shard a directory of parquet files using Dask.

This script differs from `scripts/shard_parquet.py` by performing a full-data
shuffle across every input file before writing new shards, ensuring that
examples are uniformly mixed instead of being grouped by input file.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

try:
    import dask
    import dask.dataframe as dd
except ModuleNotFoundError as exc:  # pragma: no cover - explicit import guard
    print(
        "Error: Dask is required for this script. "
        "Install it with `pip install \"dask[dataframe]\"`.",
        file=sys.stderr,
    )
    raise exc


def _add_random_shuffle_key(partition, seed: int, partition_info=None):
    """Attach a deterministic random key per row for global shuffling."""
    if len(partition) == 0:
        partition["__shuffle_key"] = []
        return partition

    part_id = partition_info["number"] if partition_info else 0
    rng = np.random.default_rng(seed + part_id)
    keys = rng.random(len(partition), dtype=np.float64)
    partition["__shuffle_key"] = keys
    return partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shuffle + shard parquet directory with Dask",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing *.parquet files to read",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write shuffled shards (default: data_dir + '_shuffled')",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=100_000,
        help="Approximate number of rows per output shard",
    )
    parser.add_argument(
        "--num-output-files",
        type=int,
        default=None,
        help="If provided, overrides rows-per-shard with an exact partition count",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="snappy",
        choices=("snappy", "gzip", "brotli", "zstd", "lz4", "none"),
        help="Compression codec for output parquet files",
    )
    parser.add_argument(
        "--shuffle-algorithm",
        type=str,
        default="tasks",
        choices=("tasks", "p2p"),
        help="Dask shuffle algorithm",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Base RNG seed for deterministic shuffling",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="threads",
        choices=("threads", "processes", "single-threaded"),
        help="Dask compute scheduler to use during write",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without writing output",
    )
    return parser.parse_args()


def _validate_paths(input_dir: Path, output_dir: Path, overwrite: bool):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    if not any(input_dir.glob("*.parquet")):
        raise FileNotFoundError(f"No parquet files found in '{input_dir}'.")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{output_dir}' already exists. "
                "Use --overwrite to replace it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def _summarize_partitions(partition_sizes: Iterable[int]) -> tuple[int, float]:
    sizes = list(partition_sizes)
    total = int(sum(sizes))
    avg = float(total / max(1, len(sizes)))
    return total, avg


def main():
    args = parse_args()

    data_dir: Path = args.data_dir
    output_dir: Path = (
        args.output_dir
        if args.output_dir is not None
        else data_dir.parent / f"{data_dir.name}_shuffled"
    )
    _validate_paths(data_dir, output_dir, args.overwrite)

    file_pattern = str(data_dir / "*.parquet")
    print("=" * 70)
    print("Dask Shuffle + Shard")
    print("=" * 70)
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Rows per shard:   {args.rows_per_shard:,}")
    if args.num_output_files:
        print(f"Target files:     {args.num_output_files:,}")
    print(f"Compression:      {args.compression}")
    print(f"Shuffle algo:     {args.shuffle_algorithm}")
    print(f"Scheduler:        {args.scheduler}")

    df = dd.read_parquet(
        file_pattern,
        engine="pyarrow",
        gather_statistics=True,
        aggregate_files=False,
    )
    print(f"\nLoaded dataset with {df.npartitions} partitions.")

    # Attach random shuffle key if no column provided
    meta = df._meta.assign(__shuffle_key=np.float64())
    df = df.map_partitions(
        _add_random_shuffle_key,
        seed=args.seed,
        partition_info=True,
        meta=meta,
    )
    df = df.shuffle("__shuffle_key", shuffle=args.shuffle_algorithm)
    df = df.drop(columns="__shuffle_key")

    partition_len_series = df.map_partitions(len)
    partition_sizes = partition_len_series.compute()
    total_rows, avg_rows = _summarize_partitions(partition_sizes)

    print("\nDataset stats after shuffle:")
    print(f"  Total rows:     {total_rows:,}")
    print(f"  Avg rows/part:  {avg_rows:,.0f}")

    target_partitions: Optional[int] = args.num_output_files
    if target_partitions is None and args.rows_per_shard > 0:
        target_partitions = max(1, math.ceil(total_rows / args.rows_per_shard))

    if target_partitions is not None:
        print(f"  Target shards:  {target_partitions:,}")
        if target_partitions != df.npartitions:
            try:
                df = df.repartition(npartitions=target_partitions)
                print(f"  Repartitioned to {df.npartitions} partitions.")
            except ValueError as err:
                print(
                    f"  Warning: unable to repartition to {target_partitions} partitions "
                    f"({err}). Continuing with {df.npartitions} partitions."
                )

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    write_kwargs = {
        "engine": "pyarrow",
        "write_index": False,
        "compression": None if args.compression == "none" else args.compression,
        "overwrite": True,
    }

    print("\nWriting shuffled shards...")
    df.to_parquet(output_dir, **write_kwargs, compute_kwargs={"scheduler": args.scheduler})

    output_files = sorted(output_dir.glob("*.parquet"))
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"Wrote {len(output_files)} shards to {output_dir}")
    if len(output_files) > 0:
        print(
            f"Approx rows/shard: {total_rows / max(1, len(output_files)):,.0f} "
            "(actual counts may vary slightly)."
        )
    print("Use this directory as --data-dir for downstream training.")


if __name__ == "__main__":
    main()

