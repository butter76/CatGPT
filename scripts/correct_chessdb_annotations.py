#!/usr/bin/env python3
"""Correct chessDB annotations that fail validation.

For W positions whose best_q is too low or whose best move is wrong, and
for L positions whose best_q is too high, we adaptively inject virtual
visits (Q=1.0 for W, Q=-1.0 for L) into the correct/best move and
downscale other moves so the total visit count stays roughly constant.

Outputs per shard:
    chessdb-{NNN}-corrected.jsonl   -- full annotation (passthrough + fixed)
    chessdb-{NNN}-correction-log.jsonl -- one entry per corrected position

Usage:
    uv run python scripts/correct_chessdb_annotations.py \
        --shard 000 --dir ~/chessdb_annotation/

    # All shards, 8 parallel workers
    uv run python scripts/correct_chessdb_annotations.py \
        --all --dir ~/chessdb_annotation/ --workers 8

    # Custom thresholds
    uv run python scripts/correct_chessdb_annotations.py \
        --shard 000 --dir ~/chessdb_annotation/ \
        --q-won 0.7 --q-lost -0.7 --target-q-won 0.5 --target-q-lost -0.5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers (same as validate_chessdb_annotations.py)
# ---------------------------------------------------------------------------

def _parse_winning_moves(moves_str: str) -> set[str]:
    result: set[str] = set()
    for entry in moves_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        uci = entry.split(":")[0]
        if uci:
            result.add(uci)
    return result


def _discover_shards(directory: Path) -> list[str]:
    pattern = re.compile(r"^chessdb-(\d+)-manifest\.jsonl$")
    shards: list[str] = []
    for p in sorted(directory.iterdir()):
        m = pattern.match(p.name)
        if m:
            shards.append(m.group(1))
    return shards


# ---------------------------------------------------------------------------
# Validation (inlined so the script is self-contained)
# ---------------------------------------------------------------------------

def _fails_validation(
    pos_type: str,
    best_q: float,
    best_move: str,
    winning_moves: set[str],
    q_won: float,
    q_lost: float,
    q_lazy: float,
) -> list[str]:
    """Return a list of failure reason tags (empty == passes)."""
    reasons: list[str] = []
    if pos_type == "W":
        if best_q < q_won:
            reasons.append("value")
        if winning_moves:
            move_match = best_move in winning_moves
            lazy_ok = best_q > q_lazy
            if not (move_match or lazy_ok):
                reasons.append("move")
    elif pos_type == "L":
        if best_q > q_lost:
            reasons.append("value")
    return reasons


# ---------------------------------------------------------------------------
# Core correction
# ---------------------------------------------------------------------------

def _correct_annotation(
    ann: dict,
    pos_type: str,
    winning_moves: set[str],
    target_q: float,
    inject_q: float,
) -> tuple[dict, dict]:
    """Return (corrected_ann, log_entry).

    inject_q: 1.0 for W, -1.0 for L.
    target_q: the Q the corrected move must reach (e.g. 0.5 or -0.5).
    """
    moves = ann["moves"]
    total_visits = ann["visits"]

    # --- pick the target move index ---
    if pos_type == "W" and winning_moves:
        # Among the engine's moves that are winning, pick highest visits
        candidates = [
            (i, m) for i, m in enumerate(moves) if m["uci"] in winning_moves
        ]
        if candidates:
            target_idx = max(candidates, key=lambda t: t[1]["visits"])[0]
        else:
            # None of the winning moves appear -- fall back to the first move
            # that appears in the list, or just the engine's best.
            target_idx = 0
    else:
        # L positions (or W without winning-move info): boost the engine's
        # best move (highest visits).
        target_idx = max(range(len(moves)), key=lambda i: moves[i]["visits"])

    target_move = moves[target_idx]
    raw_visits = [max(1, round(m["visits"] * total_visits)) for m in moves]
    raw_target = raw_visits[target_idx]
    old_q = target_move["q"]

    # --- compute n: min visits to add so new_q crosses target_q ---
    # new_q = (old_q * raw_target + inject_q * n) / (raw_target + n)
    # For W (inject_q=1, target_q=0.5):  n >= raw_target*(target_q - old_q) / (inject_q - target_q)
    # For L (inject_q=-1, target_q=-0.5): same formula works
    denom = inject_q - target_q
    if abs(denom) < 1e-9:
        n_q = 1
    else:
        n_q_exact = raw_target * (target_q - old_q) / denom
        n_q = max(1, math.ceil(n_q_exact))

    # --- compute n so that target move dominates visits ---
    # After adding n to target and downscaling others, target must have more
    # raw visits than any other move.  We solve iteratively: start from n_q,
    # check dominance, bump if needed.
    n = n_q
    for _ in range(200):
        new_raw_target = raw_target + n
        sum_others_old = sum(raw_visits) - raw_target
        num_others = len(moves) - 1
        if num_others == 0:
            break
        min_others = num_others  # 1 visit each
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
        n += raw_target  # safety fallback

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

    # Build corrected moves list
    corrected_moves = []
    for i, m in enumerate(moves):
        cm = dict(m)
        cm["visits"] = new_raw[i] / new_total
        if i == target_idx:
            cm["q"] = new_target_q
        corrected_moves.append(cm)

    # Recompute root_q as visit-weighted average Q
    new_root_q = sum(cm["visits"] * cm["q"] for cm in corrected_moves)

    corrected = dict(ann)
    corrected["moves"] = corrected_moves
    corrected["visits"] = new_total
    corrected["best_move"] = target_move["uci"]
    corrected["best_q"] = new_target_q
    corrected["root_q"] = new_root_q

    log_entry = {
        "fen_idx": ann["idx"],
        "pos_type": pos_type,
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
    q_won: float,
    q_lost: float,
    q_lazy: float,
    target_q_won: float,
    target_q_lost: float,
) -> dict:
    """Process a single shard. Returns a summary dict."""
    manifest_path = directory / f"chessdb-{shard}-manifest.jsonl"
    annotations_path = directory / f"chessdb-{shard}-annotated.jsonl"
    corrected_path = directory / f"chessdb-{shard}-corrected.jsonl"
    log_path = directory / f"chessdb-{shard}-correction-log.jsonl"

    if not manifest_path.exists() or not annotations_path.exists():
        return {"shard": shard, "status": "skipped", "reason": "missing files"}

    manifests: dict[int, dict] = {}
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            manifests[entry["fen_idx"]] = entry

    total = 0
    corrected_count = 0
    w_corrected = 0
    l_corrected = 0

    with (
        open(annotations_path) as fin,
        open(corrected_path, "w") as f_out,
        open(log_path, "w") as f_log,
    ):
        for line in fin:
            ann = json.loads(line)
            idx = ann["idx"]
            manifest = manifests.get(idx)
            total += 1

            if manifest is None or manifest["pos_type"] == "R":
                f_out.write(json.dumps(ann, separators=(",", ":")) + "\n")
                continue

            pos_type = manifest["pos_type"]
            winning_moves = (
                _parse_winning_moves(manifest.get("moves", ""))
                if pos_type == "W"
                else set()
            )

            reasons = _fails_validation(
                pos_type=pos_type,
                best_q=ann["best_q"],
                best_move=ann["best_move"],
                winning_moves=winning_moves,
                q_won=q_won,
                q_lost=q_lost,
                q_lazy=q_lazy,
            )

            if not reasons:
                f_out.write(json.dumps(ann, separators=(",", ":")) + "\n")
                continue

            inject_q = 1.0 if pos_type == "W" else -1.0
            target_q = target_q_won if pos_type == "W" else target_q_lost

            corrected_ann, log_entry = _correct_annotation(
                ann=ann,
                pos_type=pos_type,
                winning_moves=winning_moves,
                target_q=target_q,
                inject_q=inject_q,
            )
            log_entry["reason"] = reasons

            f_out.write(json.dumps(corrected_ann, separators=(",", ":")) + "\n")
            f_log.write(json.dumps(log_entry, separators=(",", ":")) + "\n")
            corrected_count += 1
            if pos_type == "W":
                w_corrected += 1
            else:
                l_corrected += 1

    return {
        "shard": shard,
        "status": "ok",
        "total": total,
        "corrected": corrected_count,
        "w_corrected": w_corrected,
        "l_corrected": l_corrected,
        "corrected_path": str(corrected_path),
        "log_path": str(log_path),
    }


def _process_shard_wrapper(args: tuple) -> dict:
    """Unpack tuple for ProcessPoolExecutor.map()."""
    return process_shard(*args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correct chessDB annotations that fail W/L validation."
    )
    parser.add_argument(
        "--dir", required=True,
        help="Directory with chessdb-{NNN}-{manifest,annotated}.jsonl files",
    )
    shard_group = parser.add_mutually_exclusive_group(required=True)
    shard_group.add_argument("--shard", nargs="+", help="Shard id(s)")
    shard_group.add_argument("--all", action="store_true", help="All shards")
    parser.add_argument("--q-won", type=float, default=0.7)
    parser.add_argument("--q-lost", type=float, default=-0.7)
    parser.add_argument("--q-lazy", type=float, default=0.9)
    parser.add_argument("--target-q-won", type=float, default=0.5,
                        help="Corrected W move must reach this Q (default: 0.5)")
    parser.add_argument("--target-q-lost", type=float, default=-0.5,
                        help="Corrected L move must reach this Q (default: -0.5)")
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
        (directory, s, args.q_won, args.q_lost, args.q_lazy,
         args.target_q_won, args.target_q_lost)
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
    total_w = 0
    total_l = 0

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
                f"[W={s['w_corrected']:,} L={s['l_corrected']:,}]"
            )
            total_positions += s["total"]
            total_corrected += s["corrected"]
            total_w += s["w_corrected"]
            total_l += s["l_corrected"]

    if len(shards) > 1 and total_positions:
        pct = 100 * total_corrected / total_positions
        print("-" * 70)
        print(
            f"  TOTAL: {total_corrected:,} / {total_positions:,} corrected ({pct:.2f}%)  "
            f"[W={total_w:,} L={total_l:,}]"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
