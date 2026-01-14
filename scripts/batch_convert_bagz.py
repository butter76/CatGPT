#!/usr/bin/env python3
"""Batch convert .bagz game files to .bag training files in parallel."""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from catgpt.core.data.grain.bagz_to_bag import convert_bagz_to_bag


def convert_single_bagz(bagz_path: Path, output_dir: Path) -> tuple[str, int, int, int, str | None]:
    """Convert a single bagz file to bag format.

    Returns:
        Tuple of (bagz_name, games_processed, positions_before_dedup, positions_written, error_message or None)
    """
    output_path = output_dir / (bagz_path.stem + ".bag")

    try:
        games, before, after = convert_bagz_to_bag(
            bagz_path,
            output_path=output_path,
            verbose=False,
        )
        return (bagz_path.name, games, before, after, None)
    except Exception as e:
        return (bagz_path.name, 0, 0, 0, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert .bagz game files to .bag training files in parallel."
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path.home() / "processed_bag",
        help="Input directory containing .bagz files (default: ~/processed_bag)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path.home() / "training_bag",
        help="Output directory for .bag files (default: ~/training_bag)",
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

    # Find all .bagz files
    bagz_files = sorted(input_dir.glob("*.bagz"))
    if not bagz_files:
        print(f"No .bagz files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(bagz_files)} .bagz files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Processing with {max_workers} parallel workers...")
    print()

    completed = 0
    failed = 0
    total_games = 0
    total_positions_before = 0
    total_positions_after = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_single_bagz, bagz_path, output_dir): bagz_path
            for bagz_path in bagz_files
        }

        for future in as_completed(futures):
            bagz_name, games, before, after, error = future.result()

            if error:
                print(f"✗ {bagz_name}: FAILED - {error}")
                failed += 1
            else:
                dedup_pct = ((before - after) / before * 100) if before > 0 else 0
                print(f"✓ {bagz_name}: {games} games -> {after} positions ({dedup_pct:.1f}% dedup)")
                completed += 1
                total_games += games
                total_positions_before += before
                total_positions_after += after

    print()
    print("=" * 60)
    print(f"Completed: {completed}/{len(bagz_files)} files")
    print(f"Total games processed: {total_games}")
    print(f"Positions after mod-3 selection: {total_positions_before}")
    print(f"Unique positions written: {total_positions_after}")
    if total_positions_before > 0:
        overall_dedup = (total_positions_before - total_positions_after) / total_positions_before * 100
        print(f"Overall deduplication: {overall_dedup:.1f}%")
    if failed:
        print(f"Failed: {failed} files")


if __name__ == "__main__":
    main()
