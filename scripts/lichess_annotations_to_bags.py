#!/usr/bin/env python3
"""Convert corrected lichess puzzle annotation JSONL files into training .bag shards.

Reads the 8 corrected annotation JSONL files and the corresponding puzzle .bag
files (for ground-truth game_result), maps each position to a TrainingPositionData
record, and round-robins them into N output .bag shards.

Field mapping:
    fen          <- annotation fen
    legal_moves  <- annotation moves: [[uci, visits_fraction], ...]
    root_q       <- annotation root_q
    root_d       <- annotation root_d
    best_q       <- annotation best_q
    best_move_uci <- annotation best_move
    game_result  <- from puzzle bag: +1, -1, or 0
    st_q         <- +1 for game_result=+1, -1 for game_result=-1, best_q for 0

Usage:
    uv run python scripts/lichess_annotations_to_bags.py \
        --dir ~/lichess_puzzles/ \
        --output-dir ~/lichess_training/ \
        --num-shards 40

    # Limit input for testing
    uv run python scripts/lichess_annotations_to_bags.py \
        --dir ~/lichess_puzzles/ \
        --output-dir ~/lichess_training/ \
        --num-shards 40 --max-input-shards 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagFileReader, BagWriter


def _discover_shards(directory: Path) -> list[str]:
    pattern = re.compile(r"^shard-(\d+)-corrected\.jsonl$")
    shards: list[str] = []
    for p in sorted(directory.iterdir()):
        m = pattern.match(p.name)
        if m:
            shards.append(m.group(1))
    return shards


def _encode_training_position(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def convert(
    directory: Path,
    output_dir: Path,
    num_shards: int,
    max_input_shards: int | None = None,
) -> None:
    input_shards = _discover_shards(directory)
    if not input_shards:
        sys.exit(f"No shard-*-corrected.jsonl files found in {directory}")

    if max_input_shards is not None:
        input_shards = input_shards[:max_input_shards]

    print(f"Input shards: {len(input_shards)}")
    print(f"Output shards: {num_shards}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    writers = [
        BagWriter(str(output_dir / f"shard-{i}-lichess-puzzles.bag"), compress=False)
        for i in range(num_shards)
    ]

    position_counter = 0
    skipped = 0
    t0 = time.time()

    try:
        for shard_num, shard_id in enumerate(input_shards):
            bag_path = directory / f"lichess-puzzles-shard-{shard_id}.bag"
            corrected_path = directory / f"shard-{shard_id}-corrected.jsonl"

            if not bag_path.exists() or not corrected_path.exists():
                print(f"  SKIP shard {shard_id}: missing files", file=sys.stderr)
                continue

            reader = BagFileReader(str(bag_path))
            bag_len = len(reader)

            with open(corrected_path) as f:
                for line in f:
                    ann = json.loads(line)
                    idx = ann["idx"]

                    if idx >= bag_len:
                        skipped += 1
                        continue

                    pos_data = msgpack.unpackb(reader[idx], raw=False)
                    game_result = pos_data["game_result"]

                    if game_result == 0:
                        skipped += 1
                        continue

                    if game_result == 1:
                        st_q = 1.0
                    elif game_result == -1:
                        st_q = -1.0
                    else:
                        st_q = ann["best_q"]

                    legal_moves = [
                        [m["uci"], m["visits"]] for m in ann["moves"]
                    ]

                    record = {
                        "fen": ann["fen"],
                        "legal_moves": legal_moves,
                        "root_q": ann["root_q"],
                        "root_d": ann.get("root_d", 0.0),
                        "best_q": ann["best_q"],
                        "best_move_uci": ann["best_move"],
                        "game_result": game_result,
                        "st_q": st_q,
                    }

                    shard_idx = position_counter % num_shards
                    writers[shard_idx].write(_encode_training_position(record))
                    position_counter += 1

            elapsed = time.time() - t0
            rate = position_counter / elapsed if elapsed > 0 else 0
            print(
                f"  shard {shard_id} done  "
                f"({shard_num + 1}/{len(input_shards)})  "
                f"total={position_counter:,}  "
                f"{rate:,.0f} pos/s",
                flush=True,
            )
    finally:
        for w in writers:
            w.close()

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"  Positions written:  {position_counter:,}")
    print(f"  Skipped (out of bag range): {skipped:,}")
    print(f"  Output shards:      {num_shards}")
    print(f"  ~positions/shard:   {position_counter // num_shards:,}")
    print(f"  Time:               {elapsed:.1f}s")
    print(f"  Output:             {output_dir}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert corrected lichess puzzle annotations to training .bag shards."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory with shard-{NNNN}-corrected.jsonl and lichess-puzzles-shard-{NNNN}.bag",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for .bag shards",
    )
    parser.add_argument(
        "--num-shards", type=int, default=40,
        help="Number of output .bag shards (default: 40)",
    )
    parser.add_argument(
        "--max-input-shards", type=int, default=None,
        help="Limit number of input shards (for testing)",
    )
    args = parser.parse_args()

    convert(
        directory=Path(args.dir),
        output_dir=Path(args.output_dir),
        num_shards=args.num_shards,
        max_input_shards=args.max_input_shards,
    )


if __name__ == "__main__":
    main()
