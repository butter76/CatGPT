#!/usr/bin/env python3
"""Validate chessDB annotated positions against ground-truth labels.

W (won) positions should have high best_q  (engine agrees it's winning).
L (lost) positions should have low  best_q (engine agrees it's losing).
R (either) positions are skipped entirely.

For W positions we also check whether the engine's best move is among the
known winning moves from the TSV.

Usage:
    uv run python scripts/validate_chessdb_annotations.py \
        --shard 000 --dir ~/chessdb_annotation/ --max 5000

    # Multiple shards
    uv run python scripts/validate_chessdb_annotations.py \
        --shard 000 001 002 --dir ~/chessdb_annotation/

    # All shards in the directory
    uv run python scripts/validate_chessdb_annotations.py \
        --all --dir ~/chessdb_annotation/

    # Adjust thresholds
    uv run python scripts/validate_chessdb_annotations.py \
        --shard 000 --dir ~/chessdb_annotation/ \
        --q-won 0.7 --q-lost -0.7

    # Show failures
    uv run python scripts/validate_chessdb_annotations.py \
        --shard 000 --dir ~/chessdb_annotation/ \
        --max 5000 --show-all-failures
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    fen_idx: int
    tsv_row: int
    pos_type: str  # "W" or "L"
    fen: str
    best_move: str
    best_q: float
    move_ok: bool | None  # None for L positions (no move check)
    value_ok: bool
    failure_reasons: list[str] = field(default_factory=list)


def _parse_winning_moves(moves_str: str) -> set[str]:
    """Extract UCI move strings from 'move:count,move:count,...'."""
    result: set[str] = set()
    for entry in moves_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        uci = entry.split(":")[0]
        if uci:
            result.add(uci)
    return result


def validate_shard(
    directory: Path,
    shard: str,
    max_positions: int | None = None,
    q_won: float = 0.7,
    q_lost: float = -0.7,
    q_lazy: float = 0.9,
) -> list[ValidationResult]:
    manifest_path = directory / f"chessdb-{shard}-manifest.jsonl"
    annotations_path = directory / f"chessdb-{shard}-annotated.jsonl"

    if not manifest_path.exists():
        print(f"  SKIP shard {shard}: {manifest_path} not found", file=sys.stderr)
        return []
    if not annotations_path.exists():
        print(f"  SKIP shard {shard}: {annotations_path} not found", file=sys.stderr)
        return []

    manifests: dict[int, dict] = {}
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            manifests[entry["fen_idx"]] = entry

    results: list[ValidationResult] = []
    validated = 0

    with open(annotations_path) as f:
        for line in f:
            ann = json.loads(line)
            idx = ann["idx"]
            manifest = manifests.get(idx)
            if manifest is None:
                continue

            pos_type = manifest["pos_type"]
            if pos_type == "R":
                continue

            if max_positions is not None and validated >= max_positions:
                break

            best_q = ann["best_q"]
            best_move = ann["best_move"]
            failures: list[str] = []

            # Value check
            value_ok = True
            if pos_type == "W":
                if best_q < q_won:
                    value_ok = False
                    failures.append(
                        f"value: best_q={best_q:.4f} but need >={q_won:.2f} "
                        f"(W position)"
                    )
            elif pos_type == "L":
                if best_q > q_lost:
                    value_ok = False
                    failures.append(
                        f"value: best_q={best_q:.4f} but need <={q_lost:.2f} "
                        f"(L position)"
                    )

            # Move check (W positions only)
            move_ok: bool | None = None
            if pos_type == "W":
                winning_moves = _parse_winning_moves(manifest.get("moves", ""))
                if winning_moves:
                    move_match = best_move in winning_moves
                    lazy_ok = best_q > q_lazy
                    move_ok = move_match or lazy_ok
                    if not move_ok:
                        failures.append(
                            f"move: engine={best_move} winning={winning_moves} "
                            f"(best_q={best_q:.4f}, need >{q_lazy:.2f} to excuse)"
                        )

            results.append(
                ValidationResult(
                    fen_idx=idx,
                    tsv_row=manifest["tsv_row"],
                    pos_type=pos_type,
                    fen=manifest["fen"],
                    best_move=best_move,
                    best_q=best_q,
                    move_ok=move_ok,
                    value_ok=value_ok,
                    failure_reasons=failures,
                )
            )
            validated += 1

    return results


def _discover_shards(directory: Path) -> list[str]:
    pattern = re.compile(r"^chessdb-(\d+)-manifest\.jsonl$")
    shards: list[str] = []
    for p in sorted(directory.iterdir()):
        m = pattern.match(p.name)
        if m:
            shards.append(m.group(1))
    return shards


def print_report(
    results: list[ValidationResult],
    label: str,
    show_all_failures: bool,
    max_failures_shown: int,
) -> None:
    total = len(results)
    if total == 0:
        print(f"  {label}: no W/L positions found")
        return

    passed = sum(1 for r in results if not r.failure_reasons)
    failed = total - passed

    w_results = [r for r in results if r.pos_type == "W"]
    l_results = [r for r in results if r.pos_type == "L"]

    w_pass = sum(1 for r in w_results if not r.failure_reasons)
    l_pass = sum(1 for r in l_results if not r.failure_reasons)

    move_checked = [r for r in results if r.move_ok is not None]
    move_fails = sum(1 for r in move_checked if not r.move_ok)
    value_fails = sum(1 for r in results if not r.value_ok)

    pct = lambda n, d: f"{100 * n / d:.1f}%" if d else "n/a"

    print("=" * 70)
    print(f"  {label}")
    print(f"  Positions validated:  {total:>10,}")
    print(f"  Passed:               {passed:>10,} ({pct(passed, total)})")
    print(f"  Failed:               {failed:>10,} ({pct(failed, total)})")
    print()
    print(f"  W positions:          {len(w_results):>10,}  pass {w_pass:>10,} ({pct(w_pass, len(w_results))})")
    print(f"  L positions:          {len(l_results):>10,}  pass {l_pass:>10,} ({pct(l_pass, len(l_results))})")
    print()
    print(f"  Move failures (W):    {move_fails:>10,} / {len(move_checked):,}")
    print(f"  Value failures:       {value_fails:>10,} / {total:,}")
    print()

    failures = [r for r in results if r.failure_reasons]
    max_show = len(failures) if show_all_failures else max_failures_shown
    shown = failures[:max_show]

    if shown:
        print(f"  Failed positions (showing {len(shown)} / {len(failures)}):")
        print("-" * 70)
        for r in shown:
            print(
                f"  [fen_idx={r.fen_idx}] tsv_row={r.tsv_row} type={r.pos_type} "
                f"best_q={r.best_q:.4f}"
            )
            print(f"    FEN: {r.fen}")
            for reason in r.failure_reasons:
                print(f"    FAIL: {reason}")
            print()

    if len(failures) > max_show:
        print(f"  ... and {len(failures) - max_show} more (use --show-all-failures)")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate chessDB annotated positions against W/L labels."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing chessdb-{NNN}-{manifest,annotated}.jsonl files",
    )
    shard_group = parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument(
        "--shard", nargs="+", help="Shard id(s), e.g. 000 001 042"
    )
    shard_group.add_argument(
        "--all", action="store_true", help="Discover and validate all shards"
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Max W/L positions per shard to validate"
    )
    parser.add_argument(
        "--q-won",
        type=float,
        default=0.7,
        help="Min best_q for W positions (default: 0.7)",
    )
    parser.add_argument(
        "--q-lost",
        type=float,
        default=-0.7,
        help="Max best_q for L positions (default: -0.7)",
    )
    parser.add_argument(
        "--q-lazy",
        type=float,
        default=0.9,
        help="best_q above which a wrong move on W is excused (default: 0.9)",
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
        help="Max failures to print per shard (default: 50)",
    )
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        sys.exit(f"Not a directory: {directory}")

    shards: list[str]
    if args.all:
        shards = _discover_shards(directory)
        if not shards:
            sys.exit(f"No chessdb-*-manifest.jsonl files found in {directory}")
        print(f"Discovered {len(shards)} shards")
    else:
        shards = args.shard

    all_results: list[ValidationResult] = []

    for shard in shards:
        results = validate_shard(
            directory=directory,
            shard=shard,
            max_positions=args.max,
            q_won=args.q_won,
            q_lost=args.q_lost,
            q_lazy=args.q_lazy,
        )
        if results:
            print_report(
                results,
                label=f"shard {shard}",
                show_all_failures=args.show_all_failures,
                max_failures_shown=args.max_failures_shown,
            )
            all_results.extend(results)

    if len(shards) > 1 and all_results:
        print()
        print_report(
            all_results,
            label=f"TOTAL ({len(shards)} shards)",
            show_all_failures=False,
            max_failures_shown=0,
        )


if __name__ == "__main__":
    main()
