#!/usr/bin/env python3
"""Correct lichess puzzle annotations that fail validation.

For game_result=+1 (player's turn) positions whose best_q is too low or whose
best move is wrong, game_result=-1 (opponent's turn) positions whose best_q is
too high, and game_result=0 (equality) positions whose |best_q| is too large,
we adaptively inject virtual visits and downscale other moves so the total
visit count stays roughly constant.

Outputs per shard:
    shard-{NNNN}-corrected.jsonl       -- full annotation (passthrough + fixed)
    shard-{NNNN}-correction-log.jsonl  -- one entry per corrected position

Usage:
    uv run python scripts/correct_lichess_annotations.py \
        --shard 0000 --dir ~/lichess_puzzles/

    # All shards, 8 parallel workers
    uv run python scripts/correct_lichess_annotations.py \
        --all --dir ~/lichess_puzzles/ --workers 8

    # Custom thresholds
    uv run python scripts/correct_lichess_annotations.py \
        --shard 0000 --dir ~/lichess_puzzles/ \
        --q-threshold 0.63 --q-equality 0.3 \
        --target-q-player 0.5 --target-q-opponent -0.5 --target-q-equality 0.5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import msgpack

from catgpt.core.data.grain.bagz import BagFileReader


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def _discover_shards(directory: Path) -> list[str]:
    pattern = re.compile(r"^shard-(\d+)-annotated\.jsonl$")
    shards: list[str] = []
    for p in sorted(directory.iterdir()):
        m = pattern.match(p.name)
        if m:
            shards.append(m.group(1))
    return shards


# ---------------------------------------------------------------------------
# Validation (mirrors validate_annotations.py logic)
# ---------------------------------------------------------------------------

def _fails_validation(
    game_result: int,
    best_q: float,
    best_move: str,
    correct_move: str,
    q_threshold: float,
    q_lazy: float,
    q_equality: float,
) -> list[str]:
    reasons: list[str] = []
    if game_result == 1:
        if best_q <= q_threshold:
            reasons.append("value")
        move_match = best_move == correct_move
        lazy_ok = best_q > q_lazy
        if not (move_match or lazy_ok):
            reasons.append("move")
    elif game_result == -1:
        if best_q >= -q_threshold:
            reasons.append("value")
    elif game_result == 0:
        if abs(best_q) >= q_equality:
            reasons.append("value")
    return reasons


# ---------------------------------------------------------------------------
# Core correction (same algorithm as correct_chessdb_annotations.py,
# extended with equality support via inject_q=0.0)
# ---------------------------------------------------------------------------

def _correct_annotation(
    ann: dict,
    game_result: int,
    correct_move: str | None,
    target_q: float,
    inject_q: float,
) -> tuple[dict, dict]:
    moves = ann["moves"]
    total_visits = ann["visits"]

    # --- pick the target move index ---
    if game_result == 1 and correct_move:
        candidates = [
            (i, m) for i, m in enumerate(moves) if m["uci"] == correct_move
        ]
        if candidates:
            target_idx = max(candidates, key=lambda t: t[1]["visits"])[0]
        else:
            target_idx = 0
    else:
        target_idx = max(range(len(moves)), key=lambda i: moves[i]["visits"])

    target_move = moves[target_idx]
    raw_visits = [max(1, round(m["visits"] * total_visits)) for m in moves]
    raw_target = raw_visits[target_idx]
    old_q = target_move["q"]

    # --- compute n: min visits to add so new_q crosses target ---
    # new_q = (old_q * raw_target + inject_q * n) / (raw_target + n)
    # For equality (inject_q=0): new_q = old_q * raw_target / (raw_target + n)
    #   |new_q| < target_q => n >= raw_target * (|old_q|/target_q - 1)
    if inject_q == 0.0:
        if abs(old_q) > target_q:
            n_q = max(1, math.ceil(raw_target * (abs(old_q) / target_q - 1)))
        else:
            n_q = 1
    else:
        denom = inject_q - target_q
        if abs(denom) < 1e-9:
            n_q = 1
        else:
            n_q_exact = raw_target * (target_q - old_q) / denom
            n_q = max(1, math.ceil(n_q_exact))

    # --- compute n so that target move dominates visits ---
    n = n_q
    for _ in range(200):
        new_raw_target = raw_target + n
        sum_others_old = sum(raw_visits) - raw_target
        num_others = len(moves) - 1
        if num_others == 0:
            break
        min_others = num_others
        reducible = max(0, sum_others_old - min_others)
        actual_reduction = min(n, reducible)
        scale = (sum_others_old - actual_reduction) / sum_others_old if sum_others_old > 0 else 0

        max_other_after = max(
            (max(1, round(raw_visits[i] * scale)) for i in range(len(moves)) if i != target_idx),
            default=0,
        )
        if new_raw_target > max_other_after:
            break
        n += max(1, max_other_after - new_raw_target + 1)
    else:
        n += raw_target

    # --- apply correction ---
    new_raw_target = raw_target + n
    sum_others_old = sum(raw_visits) - raw_target
    num_others = len(moves) - 1
    if num_others > 0 and sum_others_old > 0:
        min_others = num_others
        reducible = max(0, sum_others_old - min_others)
        actual_reduction = min(n, reducible)
        scale = (sum_others_old - actual_reduction) / sum_others_old
    else:
        scale = 1.0

    new_raw = []
    for i in range(len(moves)):
        if i == target_idx:
            new_raw.append(new_raw_target)
        else:
            new_raw.append(max(1, round(raw_visits[i] * scale)))

    new_total = sum(new_raw)
    new_target_q = (old_q * raw_target + inject_q * n) / (raw_target + n)

    corrected_moves = []
    for i, m in enumerate(moves):
        cm = dict(m)
        cm["visits"] = new_raw[i] / new_total
        if i == target_idx:
            cm["q"] = new_target_q
        corrected_moves.append(cm)

    new_root_q = sum(cm["visits"] * cm["q"] for cm in corrected_moves)

    corrected = dict(ann)
    corrected["moves"] = corrected_moves
    corrected["visits"] = new_total
    corrected["best_move"] = target_move["uci"]
    corrected["best_q"] = new_target_q
    corrected["root_q"] = new_root_q

    label = {1: "player", -1: "opponent", 0: "equality"}[game_result]
    log_entry = {
        "idx": ann["idx"],
        "game_result": game_result,
        "label": label,
        "fen": ann["fen"],
        "before": {
            "best_move": ann["best_move"],
            "best_q": round(ann["best_q"], 6),
            "root_q": round(ann["root_q"], 6),
        },
        "after": {
            "best_move": corrected["best_move"],
            "best_q": round(new_target_q, 6),
            "root_q": round(new_root_q, 6),
        },
        "visits_added": n,
    }

    return corrected, log_entry


# ---------------------------------------------------------------------------
# Per-shard processing
# ---------------------------------------------------------------------------

def process_shard(
    directory: Path,
    shard: str,
    q_threshold: float,
    q_lazy: float,
    q_equality: float,
    target_q_player: float,
    target_q_opponent: float,
    target_q_equality: float,
) -> dict:
    bag_path = directory / f"lichess-puzzles-shard-{shard}.bag"
    annotations_path = directory / f"shard-{shard}-annotated.jsonl"
    corrected_path = directory / f"shard-{shard}-corrected.jsonl"
    log_path = directory / f"shard-{shard}-correction-log.jsonl"

    if not bag_path.exists() or not annotations_path.exists():
        return {"shard": shard, "status": "skipped", "reason": "missing files"}

    reader = BagFileReader(str(bag_path))
    bag_len = len(reader)

    total = 0
    corrected_count = 0
    player_corrected = 0
    opponent_corrected = 0
    equality_corrected = 0

    with (
        open(annotations_path) as fin,
        open(corrected_path, "w") as f_out,
        open(log_path, "w") as f_log,
    ):
        for line in fin:
            ann = json.loads(line)
            idx = ann["idx"]
            total += 1

            if idx >= bag_len:
                f_out.write(json.dumps(ann, separators=(",", ":")) + "\n")
                continue

            pos_data = msgpack.unpackb(reader[idx], raw=False)
            game_result = pos_data["game_result"]
            correct_move = pos_data["correct_move_uci"]

            reasons = _fails_validation(
                game_result=game_result,
                best_q=ann["best_q"],
                best_move=ann["best_move"],
                correct_move=correct_move,
                q_threshold=q_threshold,
                q_lazy=q_lazy,
                q_equality=q_equality,
            )

            if not reasons:
                f_out.write(json.dumps(ann, separators=(",", ":")) + "\n")
                continue

            if game_result == 1:
                inject_q = 1.0
                target_q = target_q_player
            elif game_result == -1:
                inject_q = -1.0
                target_q = target_q_opponent
            else:
                inject_q = 0.0
                target_q = target_q_equality

            corrected_ann, log_entry = _correct_annotation(
                ann=ann,
                game_result=game_result,
                correct_move=correct_move if game_result == 1 else None,
                target_q=target_q,
                inject_q=inject_q,
            )
            log_entry["reason"] = reasons

            f_out.write(json.dumps(corrected_ann, separators=(",", ":")) + "\n")
            f_log.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
            corrected_count += 1
            if game_result == 1:
                player_corrected += 1
            elif game_result == -1:
                opponent_corrected += 1
            else:
                equality_corrected += 1

    return {
        "shard": shard,
        "status": "ok",
        "total": total,
        "corrected": corrected_count,
        "player_corrected": player_corrected,
        "opponent_corrected": opponent_corrected,
        "equality_corrected": equality_corrected,
    }


def _process_shard_wrapper(args: tuple) -> dict:
    return process_shard(*args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct lichess puzzle annotations that fail validation."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory with lichess-puzzles-shard-{NNNN}.bag and shard-{NNNN}-annotated.jsonl",
    )
    shard_group = parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--shard", nargs="+", help="Shard id(s), e.g. 0000 0001")
    shard_group.add_argument("--all", action="store_true", help="All shards")
    parser.add_argument("--q-threshold", type=float, default=0.63,
                        help="Min |best_q| for player/opponent (default: 0.63)")
    parser.add_argument("--q-lazy", type=float, default=0.9,
                        help="best_q to excuse wrong move (default: 0.9)")
    parser.add_argument("--q-equality", type=float, default=0.3,
                        help="Max |best_q| for equality (default: 0.3)")
    parser.add_argument("--target-q-player", type=float, default=0.5,
                        help="Corrected player move target Q (default: 0.5)")
    parser.add_argument("--target-q-opponent", type=float, default=-0.5,
                        help="Corrected opponent move target Q (default: -0.5)")
    parser.add_argument("--target-q-equality", type=float, default=0.5,
                        help="Corrected equality move target |Q| (default: 0.5)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    args = parser.parse_args()

    directory = Path(args.dir)
    if not directory.is_dir():
        sys.exit(f"Not a directory: {directory}")

    if args.all:
        shards = _discover_shards(directory)
        if not shards:
            sys.exit(f"No shards found in {directory}")
        print(f"Discovered {len(shards)} shards")
    else:
        shards = args.shard

    task_args = [
        (directory, s, args.q_threshold, args.q_lazy, args.q_equality,
         args.target_q_player, args.target_q_opponent, args.target_q_equality)
        for s in shards
    ]

    summaries: list[dict] = []

    if args.workers <= 1:
        for ta in task_args:
            summaries.append(_process_shard_wrapper(ta))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_shard_wrapper, ta): ta[1] for ta in task_args}
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    print(f"  shard {shard_id} FAILED: {exc}", file=sys.stderr)
                    summaries.append({"shard": shard_id, "status": "error", "error": str(exc)})

    summaries.sort(key=lambda s: s["shard"])

    total_positions = 0
    total_corrected = 0
    total_player = 0
    total_opponent = 0
    total_equality = 0

    print()
    print("=" * 70)
    for s in summaries:
        if s["status"] == "skipped":
            print(f"  shard {s['shard']}: SKIPPED ({s.get('reason', '')})")
        elif s["status"] == "error":
            print(f"  shard {s['shard']}: ERROR ({s.get('error', '')})")
        else:
            pct = 100 * s["corrected"] / s["total"] if s["total"] else 0
            print(
                f"  shard {s['shard']}: {s['corrected']:,} / {s['total']:,} "
                f"corrected ({pct:.2f}%)  "
                f"[player={s['player_corrected']:,} opponent={s['opponent_corrected']:,} "
                f"equality={s['equality_corrected']:,}]"
            )
            total_positions += s["total"]
            total_corrected += s["corrected"]
            total_player += s["player_corrected"]
            total_opponent += s["opponent_corrected"]
            total_equality += s["equality_corrected"]

    if len(shards) > 1 and total_positions:
        pct = 100 * total_corrected / total_positions
        print("-" * 70)
        print(
            f"  TOTAL: {total_corrected:,} / {total_positions:,} corrected ({pct:.2f}%)  "
            f"[player={total_player:,} opponent={total_opponent:,} "
            f"equality={total_equality:,}]"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
