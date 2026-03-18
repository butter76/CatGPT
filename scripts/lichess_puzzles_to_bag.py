#!/usr/bin/env python3
"""Convert Lichess puzzle CSV to .bag training data.

Step 1: Parse the puzzle CSV and generate intermediate position records.
        For each puzzle, replays the move sequence with python-chess and emits
        one record per non-starting, non-terminal position.

Step 2 (future): Annotate positions with engine evaluations (policy, Q, D).

Usage:
    uv run python scripts/lichess_puzzles_to_bag.py \
        --input ~/lichess_puzzles/lichess_db_puzzle.csv \
        --output-dir ~/lichess_puzzles/parsed \
        --num-shards 8

    # Dry run (stats only, no output)
    uv run python scripts/lichess_puzzles_to_bag.py \
        --input ~/lichess_puzzles/lichess_db_puzzle.csv \
        --dry-run

    # Small test
    uv run python scripts/lichess_puzzles_to_bag.py \
        --input ~/lichess_puzzles/lichess_db_puzzle.csv \
        --output-dir ~/lichess_puzzles/parsed \
        --max-puzzles 1000
"""

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import chess
import msgpack

from catgpt.core.data.grain.bagz import BagWriter


@dataclass
class PuzzlePosition:
    """Intermediate position data parsed from a Lichess puzzle.

    Contains everything computable without an engine. Step 2 will read these
    and add policy distributions, Q/D values, and st_q.
    """

    fen: str
    legal_moves: list[str]  # UCI strings for all legal moves
    correct_move_uci: str  # the next move in the puzzle sequence
    game_result: int  # +1=win, -1=loss, 0=draw from side-to-move
    puzzle_id: str
    puzzle_rating: int


def encode_puzzle_position(pos: PuzzlePosition) -> bytes:
    data = {
        "fen": pos.fen,
        "legal_moves": pos.legal_moves,
        "correct_move_uci": pos.correct_move_uci,
        "game_result": pos.game_result,
        "puzzle_id": pos.puzzle_id,
        "puzzle_rating": pos.puzzle_rating,
    }
    return msgpack.packb(data, use_bin_type=True)


def decode_puzzle_position(encoded: bytes) -> PuzzlePosition:
    data = msgpack.unpackb(encoded, raw=False)
    return PuzzlePosition(
        fen=data["fen"],
        legal_moves=data["legal_moves"],
        correct_move_uci=data["correct_move_uci"],
        game_result=data["game_result"],
        puzzle_id=data["puzzle_id"],
        puzzle_rating=data["puzzle_rating"],
    )


def parse_puzzle(
    puzzle_id: str,
    fen: str,
    moves_str: str,
    themes: str,
    rating: int,
) -> list[PuzzlePosition]:
    """Parse a single puzzle into a list of intermediate positions.

    The Lichess puzzle format:
    - FEN: position before the puzzle starts
    - Moves: space-separated UCI moves. Move 1 is the opponent's blunder
      (setup). Moves 2, 4, 6, ... are the puzzle player's correct responses.
      Moves 3, 5, 7, ... are the opponent's forced replies.

    We emit positions 1 through N-1 (after each move except the last),
    where each position has a known correct next move.

    Returns:
        List of PuzzlePosition records. Empty list if the puzzle is invalid.
    """
    is_equality = "equality" in themes.split()
    move_ucis = moves_str.split()
    n_moves = len(move_ucis)

    if n_moves < 2:
        return []

    board = chess.Board(fen)
    positions: list[PuzzlePosition] = []

    for i, uci in enumerate(move_ucis):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return []

        board.push(move)

        # Skip the last move (terminal position — no next move to predict)
        if i >= n_moves - 1:
            break

        next_move_uci = move_ucis[i + 1]

        # The puzzle player makes moves at indices 1, 3, 5, ... in the
        # move sequence. At this point we're looking at the board AFTER
        # move i, so the side-to-move is the one who plays move i+1.
        # Move i+1 is a puzzle-player move when (i+1) is odd, i.e. i is even.
        # Wait — move index 0 is the blunder (opponent), index 1 is the
        # player's first response, index 2 is opponent's forced reply, etc.
        # After move i, it's the turn of whoever plays move i+1.
        # Move i+1 is a puzzle-player move if (i+1) % 2 == 1, i.e. i % 2 == 0.
        is_puzzle_player_turn = (i % 2 == 0)

        if is_equality:
            game_result = 0
        elif is_puzzle_player_turn:
            game_result = 1
        else:
            game_result = -1

        legal_moves = [m.uci() for m in board.legal_moves]

        positions.append(
            PuzzlePosition(
                fen=board.fen(),
                legal_moves=legal_moves,
                correct_move_uci=next_move_uci,
                game_result=game_result,
                puzzle_id=puzzle_id,
                puzzle_rating=rating,
            )
        )

    return positions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse Lichess puzzle CSV into intermediate .bag files."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to lichess_db_puzzle.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for output .bag files. Required unless --dry-run.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of output shards (default: 8)",
    )
    parser.add_argument(
        "--max-puzzles",
        type=int,
        default=None,
        help="Process at most this many puzzles (for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute stats, don't write output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shard assignment (default: 42)",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=5,
        help="Number of sample positions to print (default: 5)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.output_dir:
        parser.error("--output-dir is required unless --dry-run is set")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    # Stats
    puzzles_total = 0
    puzzles_skipped = 0
    positions_total = 0
    positions_by_result = {-1: 0, 0: 0, 1: 0}
    positions_by_theme = {"mate": 0, "crushing": 0, "advantage": 0, "equality": 0}
    samples: list[PuzzlePosition] = []

    # Set up writers
    writers: list[BagWriter] | None = None
    positions_per_shard: list[int] = []
    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = [
            output_dir / f"lichess-puzzles-shard-{i:04d}.bag"
            for i in range(args.num_shards)
        ]
        writers = [BagWriter(str(p), compress=False) for p in shard_paths]
        for w in writers:
            w.__enter__()
        positions_per_shard = [0] * args.num_shards

    t_start = time.perf_counter()

    try:
        with open(input_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                if args.max_puzzles and puzzles_total >= args.max_puzzles:
                    break

                puzzles_total += 1

                puzzle_id = row[0]
                fen = row[1]
                moves_str = row[2]
                rating = int(row[3])
                themes = row[7]

                positions = parse_puzzle(puzzle_id, fen, moves_str, themes, rating)

                if not positions:
                    puzzles_skipped += 1
                    continue

                for pos in positions:
                    positions_total += 1
                    positions_by_result[pos.game_result] += 1

                    # Track theme category
                    theme_set = set(themes.split())
                    if "mate" in theme_set:
                        positions_by_theme["mate"] += 1
                    elif "crushing" in theme_set:
                        positions_by_theme["crushing"] += 1
                    elif "advantage" in theme_set:
                        positions_by_theme["advantage"] += 1
                    elif "equality" in theme_set:
                        positions_by_theme["equality"] += 1

                    if len(samples) < args.show_samples:
                        samples.append(pos)

                    if writers is not None:
                        encoded = encode_puzzle_position(pos)
                        shard_idx = rng.randrange(args.num_shards)
                        writers[shard_idx].write(encoded)
                        positions_per_shard[shard_idx] += 1

                if puzzles_total % 200_000 == 0:
                    elapsed = time.perf_counter() - t_start
                    rate = puzzles_total / elapsed
                    print(
                        f"  {puzzles_total:>8,} puzzles | "
                        f"{positions_total:>10,} positions | "
                        f"{rate:,.0f} puzzles/sec",
                        flush=True,
                    )

    finally:
        if writers is not None:
            for w in writers:
                w.__exit__(None, None, None)

    elapsed = time.perf_counter() - t_start

    print()
    print("=" * 60)
    print(f"  Puzzles processed:  {puzzles_total:>12,}")
    print(f"  Puzzles skipped:    {puzzles_skipped:>12,} (invalid moves)")
    print(f"  Positions emitted:  {positions_total:>12,}")
    print(f"  Time:               {elapsed:>11.1f}s")
    print(f"  Rate:               {puzzles_total / elapsed:>11,.0f} puzzles/sec")
    print()
    print("  Positions by game_result:")
    print(f"    +1 (player wins): {positions_by_result[1]:>12,}")
    print(f"    -1 (player loses):{positions_by_result[-1]:>12,}")
    print(f"     0 (equality):    {positions_by_result[0]:>12,}")
    print()
    print("  Positions by puzzle type:")
    for theme, count in sorted(positions_by_theme.items(), key=lambda x: -x[1]):
        print(f"    {theme:>12}: {count:>12,}")
    print()

    if not args.dry_run and positions_per_shard:
        print(f"  Shards: {args.num_shards}")
        mn = min(positions_per_shard)
        mx = max(positions_per_shard)
        print(f"    min/max positions per shard: {mn:,} / {mx:,}")
        print()

    if samples:
        print("  Sample positions:")
        for i, pos in enumerate(samples):
            print(f"    [{i}] puzzle={pos.puzzle_id} rating={pos.puzzle_rating}")
            print(f"        FEN:    {pos.fen}")
            print(f"        Move:   {pos.correct_move_uci}")
            print(f"        Result: {pos.game_result}")
            print(f"        Legal:  {len(pos.legal_moves)} moves")
            print()


if __name__ == "__main__":
    main()
