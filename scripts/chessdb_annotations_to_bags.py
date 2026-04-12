#!/usr/bin/env python3
"""Convert corrected chessDB annotation JSONL files into training .bag shards.

Reads the 128 corrected annotation JSONL files and their manifests, maps each
position to a TrainingPositionData record, and round-robins them into N output
.bag shards suitable for the training pipeline.

Field mapping:
    fen          <- annotation fen
    legal_moves  <- annotation moves: [[uci, visits_fraction], ...]
    root_q       <- annotation root_q
    root_d       <- annotation root_d
    best_q       <- annotation best_q
    best_move_uci <- annotation best_move
    game_result  <- W: +1, L: -1, R: sampled from WDL(best_q, best_d)
    st_q         <- W: 1.0, L: -1.0, R: best_q

Usage:
    uv run python scripts/chessdb_annotations_to_bags.py \
        --dir ~/chessdb_annotation/ \
        --output-dir ~/chessdb_training/ \
        --num-shards 40

    # Limit input for testing
    uv run python scripts/chessdb_annotations_to_bags.py \
        --dir ~/chessdb_annotation/ \
        --output-dir ~/chessdb_training/ \
        --num-shards 40 --max-input-shards 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagWriter


def _discover_shards(directory: Path) -> list[str]:
    pattern = re.compile(r"^chessdb-(\d+)-corrected\.jsonl$")
    shards: list[str] = []
    for p in sorted(directory.iterdir()):
        m = pattern.match(p.name)
        if m:
            shards.append(m.group(1))
    return shards


def _encode_training_position(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


def _sample_game_result(best_q: float, best_d: float, rng: random.Random) -> int:
    """Sample a game result from the search WDL distribution."""
    best_d = max(0.0, min(1.0, best_d))
    w_prob = max(0.0, (1.0 + best_q - best_d) / 2.0)
    l_prob = max(0.0, (1.0 - best_q - best_d) / 2.0)
    d_prob = best_d
    total = w_prob + d_prob + l_prob
    if total < 1e-9:
        return 0
    r = rng.random() * total
    if r < w_prob:
        return 1
    elif r < w_prob + d_prob:
        return 0
    else:
        return -1


def convert(
    directory: Path,
    output_dir: Path,
    num_shards: int,
    max_input_shards: int | None = None,
    seed: int = 42,
) -> None:
    input_shards = _discover_shards(directory)
    if not input_shards:
        sys.exit(f"No chessdb-*-corrected.jsonl files found in {directory}")

    if max_input_shards is not None:
        input_shards = input_shards[:max_input_shards]

    print(f"Input shards: {len(input_shards)}")
    print(f"Output shards: {num_shards}")
    print(f"Output dir: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    writers = [
        BagWriter(str(output_dir / f"shard-{i}-chessdb-annotation.bag"), compress=False)
        for i in range(num_shards)
    ]

    rng = random.Random(seed)
    position_counter = 0
    skipped = 0
    t0 = time.time()

    try:
        for shard_num, shard_id in enumerate(input_shards):
            manifest_path = directory / f"chessdb-{shard_id}-manifest.jsonl"
            corrected_path = directory / f"chessdb-{shard_id}-corrected.jsonl"

            if not manifest_path.exists() or not corrected_path.exists():
                print(f"  SKIP shard {shard_id}: missing files", file=sys.stderr)
                continue

            manifests: dict[int, dict] = {}
            with open(manifest_path) as f:
                for line in f:
                    entry = json.loads(line)
                    manifests[entry["fen_idx"]] = entry

            with open(corrected_path) as f:
                for line in f:
                    ann = json.loads(line)
                    idx = ann["idx"]
                    manifest = manifests.get(idx)
                    if manifest is None:
                        skipped += 1
                        continue

                    pos_type = manifest["pos_type"]

                    # game_result
                    if pos_type == "W":
                        game_result = 1
                    elif pos_type == "L":
                        game_result = -1
                    else:
                        game_result = _sample_game_result(
                            ann["best_q"], ann.get("best_d", 0.0), rng
                        )

                    # st_q
                    if pos_type == "W":
                        st_q = 1.0
                    elif pos_type == "L":
                        st_q = -1.0
                    else:
                        st_q = ann["best_q"]

                    # legal_moves: [[uci, visits_fraction], ...]
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
    print(f"  Skipped (no manifest): {skipped:,}")
    print(f"  Output shards:      {num_shards}")
    print(f"  ~positions/shard:   {position_counter // num_shards:,}")
    print(f"  Time:               {elapsed:.1f}s")
    print(f"  Output:             {output_dir}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert corrected chessDB annotations to training .bag shards."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory with chessdb-{NNN}-{corrected,manifest}.jsonl files",
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
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for R position game_result sampling (default: 42)",
    )
    args = parser.parse_args()

    convert(
        directory=Path(args.dir),
        output_dir=Path(args.output_dir),
        num_shards=args.num_shards,
        max_input_shards=args.max_input_shards,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
