#!/usr/bin/env python3
"""
Script to identify and remove corrupted Parquet files.
"""
import argparse
from pathlib import Path
from pyarrow.parquet import ParquetFile
import sys
from tqdm import tqdm


def check_parquet_file(filepath: Path) -> bool:
    """
    Check if a parquet file is valid.
    
    Args:
        filepath: Path to the parquet file
        
    Returns:
        True if valid, False if corrupted
    """
    try:
        pf = ParquetFile(str(filepath))
        # Try to access metadata to ensure file is readable
        _ = pf.metadata.num_rows
        return True
    except Exception as e:
        return False


def scan_and_remove_corrupted(
    data_dir: str,
    dry_run: bool = True,
    backup: bool = False
) -> tuple[list[Path], list[Path]]:
    """
    Scan directory for corrupted parquet files and optionally remove them.
    
    Args:
        data_dir: Directory containing parquet files
        dry_run: If True, only report corrupted files without removing
        backup: If True, move corrupted files to a backup directory instead of deleting
        
    Returns:
        Tuple of (valid_files, corrupted_files)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        sys.exit(1)
    
    # Get all parquet files
    if data_path.is_file():
        all_files = [data_path]
    else:
        all_files = sorted(data_path.glob('*.parquet'))
    
    if not all_files:
        print(f"No parquet files found in {data_dir}")
        return [], []
    
    print(f"Scanning {len(all_files)} parquet files...")
    
    valid_files = []
    corrupted_files = []
    
    for filepath in tqdm(all_files, desc="Checking files", unit="file"):
        if check_parquet_file(filepath):
            valid_files.append(filepath)
        else:
            tqdm.write(f"✗ CORRUPTED: {filepath.name}")
            corrupted_files.append(filepath)
    
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Corrupted files: {len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files:")
        for f in corrupted_files:
            print(f"  - {f}")
        
        if not dry_run:
            if backup:
                # Create backup directory
                backup_dir = data_path / "corrupted_backup"
                backup_dir.mkdir(exist_ok=True)
                print(f"\nMoving corrupted files to {backup_dir}...")
                
                for f in corrupted_files:
                    dest = backup_dir / f.name
                    f.rename(dest)
                    print(f"  Moved: {f.name}")
            else:
                print(f"\nRemoving corrupted files...")
                for f in corrupted_files:
                    f.unlink()
                    print(f"  Removed: {f.name}")
            
            print(f"✓ Done! Removed {len(corrupted_files)} corrupted file(s)")
        else:
            print(f"\n⚠ DRY RUN: No files were removed.")
            print(f"Run with --execute to actually remove corrupted files.")
            if not backup:
                print(f"Add --backup to move files to backup directory instead of deleting.")
    else:
        print(f"\n✓ All files are valid!")
    
    return valid_files, corrupted_files


def main():
    parser = argparse.ArgumentParser(
        description="Scan for and remove corrupted Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (default) - just report corrupted files
  python remove_corrupted_parquet.py /path/to/data

  # Actually remove corrupted files
  python remove_corrupted_parquet.py /path/to/data --execute

  # Move corrupted files to backup directory instead of deleting
  python remove_corrupted_parquet.py /path/to/data --execute --backup
        """
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing parquet files'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually remove corrupted files (default is dry run)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Move corrupted files to backup directory instead of deleting'
    )
    
    args = parser.parse_args()
    
    scan_and_remove_corrupted(
        args.data_dir,
        dry_run=not args.execute,
        backup=args.backup
    )


if __name__ == '__main__':
    main()

