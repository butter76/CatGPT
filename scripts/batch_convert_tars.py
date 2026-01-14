#!/usr/bin/env python3
"""Batch convert Leela Chess .tar files to .bagz format in parallel."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catgpt.core.data.grain.tar_to_bag import convert_tar_to_bag


def convert_single_tar(tar_path: Path, output_dir: Path) -> tuple[str, int, str | None]:
    """Convert a single tar file to bagz format.

    Returns:
        Tuple of (tar_name, games_written, error_message or None)
    """
    output_path = output_dir / (tar_path.stem + ".bagz")

    try:
        games = convert_tar_to_bag(
            tar_path,
            output_path=output_path,
            compress=True,
            verbose=False,
        )
        return (tar_path.name, games, None)
    except Exception as e:
        return (tar_path.name, 0, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert Leela Chess .tar files to .bagz format in parallel."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path.home() / "lczero_training_data",
        help="Input directory containing .tar files (default: ~/lczero_training_data)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.home() / "processed_bag",
        help="Output directory for .bagz files (default: ~/processed_bag)",
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.jobs

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .tar files
    tar_files = sorted(input_dir.glob("*.tar"))
    if not tar_files:
        print(f"No .tar files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(tar_files)} .tar files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing with {max_workers} parallel workers...")
    print()

    completed = 0
    failed = 0
    total_games = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_single_tar, tar_path, output_dir): tar_path
            for tar_path in tar_files
        }

        for future in as_completed(futures):
            tar_name, games, error = future.result()

            if error:
                print(f"✗ {tar_name}: FAILED - {error}")
                failed += 1
            else:
                print(f"✓ {tar_name}: {games} games")
                completed += 1
                total_games += games

    print()
    print("=" * 50)
    print(f"Completed: {completed}/{len(tar_files)} files")
    print(f"Total games: {total_games}")
    if failed:
        print(f"Failed: {failed} files")


if __name__ == "__main__":
    main()
