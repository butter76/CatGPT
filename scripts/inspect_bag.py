#!/usr/bin/env python3
"""Print sample entries from a .bag file for inspection.

Usage:
    uv run python scripts/inspect_bag.py ~/parent_bag/training-run1-test80-20250707-2317.bag
    uv run python scripts/inspect_bag.py ~/parent_bag/training-run1-test80-20250707-2317.bag --count 5
    uv run python scripts/inspect_bag.py ~/parent_bag/training-run1-test80-20250707-2317.bag --index 100
"""

import argparse
import json
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect entries in a .bag file.")
    parser.add_argument("path", help="Path to .bag file.")
    parser.add_argument("--count", "-n", type=int, default=3, help="Number of entries to print (default: 3).")
    parser.add_argument("--index", "-i", type=int, default=None, help="Print a specific index (overrides --count).")
    parser.add_argument("--keys", "-k", action="store_true", help="Only print top-level keys.")
    args = parser.parse_args()

    path = str(Path(args.path).expanduser())
    reader = BagReader(path)
    print(f"File: {path}")
    print(f"Total records: {len(reader):,}\n")

    if args.index is not None:
        indices = [args.index]
    else:
        step = max(1, len(reader) // args.count)
        indices = list(range(0, len(reader), step))[: args.count]

    for idx in indices:
        raw = reader[idx]
        data = msgpack.unpackb(raw, raw=False)

        print(f"{'=' * 72}")
        print(f"Record #{idx}")
        print(f"{'=' * 72}")

        if args.keys:
            print(f"Keys: {sorted(data.keys())}")
            print()
            continue

        for key, value in sorted(data.items()):
            # # Truncate long values for readability
            # if isinstance(value, dict) and len(value) > 10:
            #     preview = dict(list(value.items())[:5])
            #     print(f"  {key}: {preview} ... ({len(value)} items)")
            # elif isinstance(value, list) and len(value) > 10:
            #     print(f"  {key}: {value[:5]} ... ({len(value)} items)")
            # else:
            #     print(f"  {key}: {value}")
            print(f"  {key}: {value}")
        print()


if __name__ == "__main__":
    main()
