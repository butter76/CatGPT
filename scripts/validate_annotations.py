#!/usr/bin/env python3
"""Validate annotated puzzle positions against puzzle ground truth.

Checks that:
1. The engine's best move matches the puzzle's correct move,
   OR best_q > 0.9 (position is so winning that the exact move doesn't matter).
2. The value evaluation is directionally correct:
   - Puzzle player's turn (game_result=+1): best_q > 0.7
   - Opponent's turn (game_result=-1): best_q < -0.7
   - Equality (game_result=0): |best_q| < 0.3

Usage:
    uv run python scripts/validate_annotations.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max 5000

    # Show all failures
    uv run python scripts/validate_annotations.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max 5000 --show-all-failures

    # Adjust thresholds
    uv run python scripts/validate_annotations.py \
        --bag ~/lichess_puzzles/parsed/lichess-puzzles-shard-0000.bag \
        --annotations ~/lichess_puzzles/shard-0000-annotated.jsonl \
        --max 5000 --q-threshold 0.8 --q-lazy 0.95 --q-equality 0.2
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagFileReader


@dataclass
class ValidationResult:
    idx: int
    fen: str
    puzzle_id: str
    puzzle_rating: int
    correct_move: str
    best_move: str
    best_q: float
    game_result: int  # +1, -1, 0
    move_ok: bool
    value_ok: bool
    failure_reasons: list[str]


def validate(
    bag_path: str,
    annotations_path: str,
    max_positions: int | None = None,
    q_threshold: float = 0.7,
    q_lazy: float = 0.9,
    q_equality: float = 0.3,
) -> list[ValidationResult]:
    reader = BagFileReader(bag_path)
    total_in_bag = len(reader)

    results: list[ValidationResult] = []

    with open(annotations_path) as f:
        for line_no, line in enumerate(f):
            if max_positions and line_no >= max_positions:
                break

            ann = json.loads(line)
            idx = ann["idx"]

            if idx >= total_in_bag:
                break

            pos_data = msgpack.unpackb(reader[idx], raw=False)

            correct_move = pos_data["correct_move_uci"]
            game_result = pos_data["game_result"]
            puzzle_id = pos_data["puzzle_id"]
            puzzle_rating = pos_data["puzzle_rating"]

            best_move = ann["best_move"]
            best_q = ann["best_q"]

            failures: list[str] = []

            # Check 1: Move correctness (only on puzzle player's turn)
            move_ok = True
            if game_result == 1:
                move_match = best_move == correct_move
                lazy_ok = best_q > q_lazy
                move_ok = move_match or lazy_ok

                if not move_ok:
                    failures.append(
                        f"move: engine={best_move} correct={correct_move} "
                        f"(best_q={best_q:.4f}, need >{q_lazy:.2f} to excuse)"
                    )

            # Check 2: Value correctness
            value_ok = True
            if game_result == 1:
                if best_q <= q_threshold:
                    value_ok = False
                    failures.append(
                        f"value: best_q={best_q:.4f} but need >{q_threshold:.2f} "
                        f"(puzzle player's turn)"
                    )
            elif game_result == -1:
                if best_q >= -q_threshold:
                    value_ok = False
                    failures.append(
                        f"value: best_q={best_q:.4f} but need <{-q_threshold:.2f} "
                        f"(opponent's turn)"
                    )
            elif game_result == 0:
                if abs(best_q) >= q_equality:
                    value_ok = False
                    failures.append(
                        f"value: best_q={best_q:.4f} but need |q|<{q_equality:.2f} "
                        f"(equality puzzle)"
                    )

            results.append(
                ValidationResult(
                    idx=idx,
                    fen=pos_data["fen"],
                    puzzle_id=puzzle_id,
                    puzzle_rating=puzzle_rating,
                    correct_move=correct_move,
                    best_move=best_move,
                    best_q=best_q,
                    game_result=game_result,
                    move_ok=move_ok,
                    value_ok=value_ok,
                    failure_reasons=failures,
                )
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate annotated puzzle positions."
    )
    parser.add_argument("--bag", required=True, help="Path to PuzzlePosition .bag")
    parser.add_argument(
        "--annotations", required=True, help="Path to annotated .jsonl"
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max positions to validate"
    )
    parser.add_argument(
        "--q-threshold",
        type=float,
        default=0.63,
        help="Min |best_q| for player/opponent turns (default: 0.7)",
    )
    parser.add_argument(
        "--q-lazy",
        type=float,
        default=0.9,
        help="best_q threshold to excuse wrong move (default: 0.9)",
    )
    parser.add_argument(
        "--q-equality",
        type=float,
        default=0.3,
        help="Max |best_q| for equality puzzles (default: 0.3)",
    )
    parser.add_argument(
        "--show-all-failures",
        action="store_true",
        help="Show every failed position (default: first 50)",
    )
    parser.add_argument(
        "--max-failures-shown",
        type=int,
        default=50,
        help="Max failures to print (default: 50, ignored with --show-all-failures)",
    )
    args = parser.parse_args()

    results = validate(
        bag_path=args.bag,
        annotations_path=args.annotations,
        max_positions=args.max,
        q_threshold=args.q_threshold,
        q_lazy=args.q_lazy,
        q_equality=args.q_equality,
    )

    total = len(results)
    passed = sum(1 for r in results if not r.failure_reasons)
    failed = total - passed

    move_fails = sum(1 for r in results if not r.move_ok)
    value_fails = sum(1 for r in results if not r.value_ok)
    both_fail = sum(1 for r in results if not r.move_ok and not r.value_ok)

    # Break down by game_result
    by_result: dict[int, dict[str, int]] = {}
    for r in results:
        bucket = by_result.setdefault(r.game_result, {"total": 0, "pass": 0})
        bucket["total"] += 1
        if not r.failure_reasons:
            bucket["pass"] += 1

    print("=" * 70)
    print(f"  Positions validated:  {total:>8,}")
    print(f"  Passed:               {passed:>8,} ({100 * passed / total:.1f}%)")
    print(f"  Failed:               {failed:>8,} ({100 * failed / total:.1f}%)")
    print()
    print(f"  Move failures:        {move_fails:>8,} ({100 * move_fails / total:.1f}%)")
    print(f"  Value failures:       {value_fails:>8,} ({100 * value_fails / total:.1f}%)")
    print(f"  Both failed:          {both_fail:>8,} ({100 * both_fail / total:.1f}%)")
    print()
    print("  By game_result:")
    for gr in sorted(by_result.keys()):
        b = by_result[gr]
        label = {1: "+1 (player)", -1: "-1 (opponent)", 0: " 0 (equality)"}[gr]
        pct = 100 * b["pass"] / b["total"] if b["total"] else 0
        print(f"    {label}: {b['pass']:,} / {b['total']:,} pass ({pct:.1f}%)")
    print()

    # Print failures
    failures = [r for r in results if r.failure_reasons]
    max_show = len(failures) if args.show_all_failures else args.max_failures_shown
    shown = failures[:max_show]

    if shown:
        print(f"  Failed positions (showing {len(shown)} / {len(failures)}):")
        print("-" * 70)
        for r in shown:
            result_label = {1: "player", -1: "opponent", 0: "equality"}[
                r.game_result
            ]
            print(
                f"  [{r.idx}] puzzle={r.puzzle_id} rating={r.puzzle_rating} "
                f"turn={result_label}"
            )
            print(f"    FEN: {r.fen}")
            for reason in r.failure_reasons:
                print(f"    FAIL: {reason}")
            print()

    if len(failures) > max_show:
        print(f"  ... and {len(failures) - max_show} more failures (use --show-all-failures)")


if __name__ == "__main__":
    main()
