#!/usr/bin/env python3
"""Extract FEN strings from a PuzzlePosition .bag file, one per line.

Usage:
    uv run python scripts/bag_to_fens.py \
        ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        -o ~/lichess_puzzles/shard-0000-fens.txt

    # Limit to first N positions
    uv run python scripts/bag_to_fens.py \
        ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        -o ~/lichess_puzzles/shard-0000-fens.txt \
        --max 10000
"""

import argparse
import sys
from pathlib import Path

from catgpt.core.data.grain.bagz import BagFileReader

import msgpack


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FENs from a PuzzlePosition .bag file."
    )
    parser.add_argument("input", help="Path to .bag file")
    parser.add_argument("-o", "--output", required=True, help="Output FEN file")
    parser.add_argument(
        "--max", type=int, default=None, help="Max positions to extract"
    )
    args = parser.parse_args()

    reader = BagFileReader(args.input)
    total = len(reader)
    limit = min(total, args.max) if args.max else total

    print(f"Reading {limit} / {total} positions from {args.input}", flush=True)

    with open(args.output, "w") as f:
        for i in range(limit):
            data = msgpack.unpackb(reader[i], raw=False)
            f.write(data["fen"] + "\n")

            if (i + 1) % 500_000 == 0:
                print(f"  {i + 1:,} / {limit:,}", flush=True)

    print(f"Wrote {limit:,} FENs to {args.output}")


if __name__ == "__main__":
    main()
