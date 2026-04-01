#!/usr/bin/env python3
"""Find puzzles the NN doesn't instantly solve from raw policy/value outputs.

A puzzle is "instantly solved" by the NN if:
  - nn_policy on the correct move >= --policy-threshold (default 20%), OR
  - The value head already trivially solves it:
      game_result=+1: orig_q > --q-trivial (default 0.8)
      game_result= 0: |orig_q| < --q-equality (default 0.3)

Displays all qualifying puzzles (below a rating cap, on the puzzle's turn)
that the NN does NOT instantly solve.

Usage:
    uv run python scripts/find_unsolved_puzzles.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max-rating 1500

    # Limit to first N positions
    uv run python scripts/find_unsolved_puzzles.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max-rating 1500 --max 5000

    # Adjust thresholds
    uv run python scripts/find_unsolved_puzzles.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max-rating 1200 --policy-threshold 0.1 --q-trivial 0.9
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagFileReader


@dataclass
class UnsolvedPuzzle:
    idx: int
    fen: str
    puzzle_id: str
    puzzle_rating: int
    game_result: int
    correct_move: str
    policy_on_correct: float
    orig_q: float
    best_move: str
    best_q: float
    top_policy_moves: list[tuple[str, float]] = field(default_factory=list)


def find_unsolved(
    bag_path: str,
    annotations_path: str,
    max_rating: int,
    max_positions: int | None = None,
    policy_threshold: float = 0.20,
    q_trivial: float = 0.80,
    q_equality: float = 0.30,
) -> tuple[list[UnsolvedPuzzle], int, int]:
    """Returns (unsolved_puzzles, total_checked, total_qualifying)."""
    reader = BagFileReader(bag_path)
    total_in_bag = len(reader)

    unsolved: list[UnsolvedPuzzle] = []
    total_checked = 0
    total_qualifying = 0

    with open(annotations_path) as f:
        for line_no, line in enumerate(f):
            if max_positions and line_no >= max_positions:
                break

            ann = json.loads(line)
            idx = ann["idx"]

            if idx >= total_in_bag:
                break

            total_checked += 1
            pos_data = msgpack.unpackb(reader[idx], raw=False)

            game_result = pos_data["game_result"]
            puzzle_rating = pos_data["puzzle_rating"]

            # Only puzzle solver's turn (+1) and equality (0)
            if game_result not in (1, 0):
                continue

            if puzzle_rating > max_rating:
                continue

            total_qualifying += 1

            correct_move = pos_data["correct_move_uci"]
            orig_q = ann.get("orig_q", ann["root_q"])

            # Find nn_policy on the correct move
            policy_on_correct = 0.0
            for m in ann["moves"]:
                if m["uci"] == correct_move:
                    policy_on_correct = m["nn_policy"]
                    break

            # Check if trivially solved by value head
            if game_result == 1 and orig_q > q_trivial:
                continue
            if game_result == 0 and abs(orig_q) < q_equality:
                continue

            # Check if solved by policy head
            if policy_on_correct >= policy_threshold:
                continue

            # Top policy moves for context
            sorted_moves = sorted(ann["moves"], key=lambda m: m["nn_policy"], reverse=True)
            top_moves = [(m["uci"], m["nn_policy"]) for m in sorted_moves[:5]]

            unsolved.append(
                UnsolvedPuzzle(
                    idx=idx,
                    fen=pos_data["fen"],
                    puzzle_id=pos_data["puzzle_id"],
                    puzzle_rating=puzzle_rating,
                    game_result=game_result,
                    correct_move=correct_move,
                    policy_on_correct=policy_on_correct,
                    orig_q=orig_q,
                    best_move=ann["best_move"],
                    best_q=ann["best_q"],
                    top_policy_moves=top_moves,
                )
            )

    return unsolved, total_checked, total_qualifying


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find puzzles the NN doesn't instantly solve."
    )
    parser.add_argument("--bag", required=True, help="Path to PuzzlePosition .bag")
    parser.add_argument(
        "--annotations", required=True, help="Path to annotated .jsonl"
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        required=True,
        help="Only consider puzzles with rating <= this value",
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max annotation lines to process"
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=0.20,
        help="Min nn_policy on correct move to count as 'solved' (default: 0.20)",
    )
    parser.add_argument(
        "--q-trivial",
        type=float,
        default=0.80,
        help="orig_q above this excludes winning positions as trivially solved (default: 0.80)",
    )
    parser.add_argument(
        "--q-equality",
        type=float,
        default=0.30,
        help="|orig_q| below this excludes equality positions as trivially solved (default: 0.30)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show every unsolved puzzle (default: first 100)",
    )
    parser.add_argument(
        "--max-shown",
        type=int,
        default=100,
        help="Max unsolved puzzles to print (default: 100, ignored with --show-all)",
    )
    args = parser.parse_args()

    unsolved, total_checked, total_qualifying = find_unsolved(
        bag_path=args.bag,
        annotations_path=args.annotations,
        max_rating=args.max_rating,
        max_positions=args.max,
        policy_threshold=args.policy_threshold,
        q_trivial=args.q_trivial,
        q_equality=args.q_equality,
    )

    n_unsolved = len(unsolved)
    n_solved = total_qualifying - n_unsolved
    pct_solved = 100 * n_solved / total_qualifying if total_qualifying else 0

    print("=" * 74)
    print(f"  Annotations scanned:   {total_checked:>8,}")
    print(f"  Qualifying (rating <= {args.max_rating}, puzzle turn):  {total_qualifying:>8,}")
    print(f"  Instantly solved:      {n_solved:>8,} ({pct_solved:.1f}%)")
    print(f"  Unsolved:              {n_unsolved:>8,} ({100 - pct_solved:.1f}%)")
    print()
    print(f"  Thresholds: policy >= {args.policy_threshold:.0%}  |  "
          f"q_trivial > {args.q_trivial}  |  q_equality < {args.q_equality}")
    print("=" * 74)

    # Rating distribution of unsolved puzzles
    if unsolved:
        ratings = [p.puzzle_rating for p in unsolved]
        print(f"\n  Unsolved rating range: {min(ratings)} – {max(ratings)}")
        buckets = {}
        for r in ratings:
            bucket = (r // 200) * 200
            buckets[bucket] = buckets.get(bucket, 0) + 1
        print("  Rating distribution:")
        for b in sorted(buckets):
            bar = "#" * min(buckets[b], 60)
            print(f"    {b:>5}–{b + 199:<5}: {buckets[b]:>5,}  {bar}")
        print()

    # Print unsolved puzzles
    max_show = len(unsolved) if args.show_all else args.max_shown
    shown = unsolved[:max_show]

    if shown:
        print(f"  Unsolved puzzles (showing {len(shown)} / {n_unsolved}):")
        print("-" * 74)
        for p in shown:
            turn_label = {1: "player", 0: "equality"}[p.game_result]
            print(
                f"  [{p.idx}] puzzle={p.puzzle_id}  rating={p.puzzle_rating}  "
                f"turn={turn_label}"
            )
            print(f"    FEN: {p.fen}")
            print(
                f"    correct={p.correct_move}  policy={p.policy_on_correct:.4f}  "
                f"orig_q={p.orig_q:.4f}  best_move={p.best_move}  best_q={p.best_q:.4f}"
            )
            if p.top_policy_moves:
                top_str = "  ".join(
                    f"{uci}:{pol:.4f}" for uci, pol in p.top_policy_moves
                )
                print(f"    top_policy: {top_str}")
            print()

    if n_unsolved > max_show:
        print(
            f"  ... and {n_unsolved - max_show} more unsolved puzzles "
            f"(use --show-all)"
        )


if __name__ == "__main__":
    main()
