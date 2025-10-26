#!/usr/bin/env python3
"""
Script to prepare the Catan dataset by merging or sharding parquet files.
This should be run once to preprocess the data for faster training.
"""
import argparse
from data import merge_parquet_files, create_sharded_files

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Catan dataset by merging or sharding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all parquet files in a directory into a single file
  python rl/prepare_data.py merge --input-dir data/games --output-file data/games_merged.parquet

  # Shard all parquet files into 100 shards
  python rl/prepare_data.py shard --input-dir data/games --output-dir data/games_sharded --num-shards 100
        """
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='sub-command help')

    # Merge command
    parser_merge = subparsers.add_parser('merge', help='Merge parquet files into a single file')
    parser_merge.add_argument('--input-dir', type=str, required=True, help='Directory containing .parquet files')
    parser_merge.add_argument('--output-file', type=str, required=True, help='Output merged file path')

    # Shard command
    parser_shard = subparsers.add_parser('shard', help='Shard parquet files into multiple smaller files')
    parser_shard.add_argument('--input-dir', type=str, required=True, help='Directory containing .parquet files')
    parser_shard.add_argument('--output-dir', type=str, required=True, help='Output directory for shards')
    parser_shard.add_argument('--num-shards', type=int, default=100, help='Number of shards to create (default: 100)')

    args = parser.parse_args()

    if args.command == 'merge':
        print("Starting merge process...")
        merge_parquet_files(args.input_dir, args.output_file)
        print(f"\nMerge complete. Merged file created at: {args.output_file}")
        print(f"You can now use this file for training by setting --data-dir {args.output_file}")
    elif args.command == 'shard':
        print("Starting sharding process...")
        create_sharded_files(args.input_dir, args.num_shards, args.output_dir)
        print(f"\nSharding complete. Shards created in: {args.output_dir}")
        print(f"You can now use this directory for training by setting --data-dir {args.output_dir}")

if __name__ == "__main__":
    main()

