#!/usr/bin/env python3
"""Pipeline: Download .bagz files from Wasabi, convert to sharded .bag training data, upload back.

This script orchestrates a massively parallel pipeline that:
1. Lists .bagz files in the Wasabi source bucket (leela-processed-bagz)
2. Filters by date range and sorts chronologically
3. Skips files whose shards already exist in the destination bucket
4. For each .bagz file (bounded concurrency):
   a. Downloads from Wasabi source bucket
   b. Converts to sharded .bag files using convert_bagz_to_bag (as a subprocess)
   c. Uploads all shards to Wasabi destination bucket (catgpt-training-data)
   d. Deletes local files to free disk space

Conversion runs as independent subprocesses (not a process pool), so a crash in
one conversion cannot poison others.

Backpressure: An in-flight semaphore caps the number of .bagz files simultaneously
in the pipeline (downloaded + converting + uploading), bounding disk usage to roughly
max_in_flight * ~1.5 GB.

Usage:
    uv run python scripts/pipeline_bagz_to_training.py \\
        --start-date 2024-04-01 --end-date 2024-04-30

    uv run python scripts/pipeline_bagz_to_training.py \\
        --start-date 2024-04-01 --end-date 2024-06-30 \\
        --download-workers 8 --convert-workers 4 --upload-workers 8

    # Dry run
    uv run python scripts/pipeline_bagz_to_training.py \\
        --start-date 2024-04-01 --end-date 2024-04-07 --dry-run
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WASABI_ENDPOINT = "https://s3.us-west-1.wasabisys.com"
SOURCE_BUCKET = "leela-processed-bagz"
DEST_BUCKET = "catgpt-training-data"

BAGZ_DATE_RE = re.compile(r"training-run1-test80-(\d{8})-\d{4}\.bagz$")

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")


# ---------------------------------------------------------------------------
# Phase 0: Discover files
# ---------------------------------------------------------------------------


async def list_source_bucket() -> list[str]:
    """List all .bagz object keys in the source bucket."""
    print("Listing .bagz files in source bucket...")
    proc = await asyncio.create_subprocess_exec(
        "aws", "s3api", "list-objects-v2",
        "--bucket", SOURCE_BUCKET,
        "--endpoint-url", WASABI_ENDPOINT,
        "--query", "Contents[].Key",
        "--output", "text",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip()
        if "None" in stdout.decode():
            return []
        print(f"  Warning listing source bucket: {err}")
        return []

    keys: list[str] = []
    for token in stdout.decode().strip().split():
        token = token.strip()
        if token and token != "None" and token.endswith(".bagz"):
            keys.append(token)
    return keys


async def list_dest_bucket() -> set[str]:
    """List all object keys in the destination bucket."""
    print("Checking existing files in destination bucket...")
    proc = await asyncio.create_subprocess_exec(
        "aws", "s3api", "list-objects-v2",
        "--bucket", DEST_BUCKET,
        "--endpoint-url", WASABI_ENDPOINT,
        "--query", "Contents[].Key",
        "--output", "text",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        err = stderr.decode().strip()
        if "NoSuchKey" in err or "None" in stdout.decode():
            return set()
        print(f"  Warning listing dest bucket: {err}")
        return set()

    existing: set[str] = set()
    for token in stdout.decode().strip().split():
        token = token.strip()
        if token and token != "None":
            existing.add(token)
    return existing


def extract_date(filename: str) -> str | None:
    """Extract YYYYMMDD date string from a .bagz filename."""
    m = BAGZ_DATE_RE.search(filename)
    return m.group(1) if m else None


def filter_and_sort_by_date(
    keys: list[str],
    start_date: datetime,
    end_date: datetime,
) -> list[str]:
    """Filter .bagz keys by date range and sort chronologically."""
    filtered: list[tuple[str, str]] = []
    for key in keys:
        date_str = extract_date(key)
        if date_str is None:
            continue
        file_date = datetime.strptime(date_str, "%Y%m%d")
        if start_date <= file_date <= end_date:
            filtered.append((key, date_str))

    filtered.sort(key=lambda x: x[0])
    return [k for k, _ in filtered]


def already_converted(bagz_key: str, existing_keys: set[str], num_shards: int) -> bool:
    """Check if all shards for a given .bagz file exist in the dest bucket."""
    stem = bagz_key.removesuffix(".bagz")
    if num_shards == 1:
        return f"{stem}.bag" in existing_keys
    return all(f"shard-{i}-{stem}.bag" in existing_keys for i in range(num_shards))


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def download_from_wasabi(
    key: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download a .bagz file from the source Wasabi bucket."""
    async with semaphore:
        t0 = time.monotonic()
        print(f"  ⬇  Downloading {key}...")
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp",
            f"s3://{SOURCE_BUCKET}/{key}",
            str(dest),
            "--endpoint-url", WASABI_ENDPOINT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            print(f"  ✗  Download FAILED {key}: {stderr.decode().strip()}")
            dest.unlink(missing_ok=True)
            return False

        elapsed = time.monotonic() - t0
        size_mb = dest.stat().st_size / (1024 * 1024)
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"  ✓  Downloaded {key} ({size_mb:.0f} MB in {elapsed:.0f}s, {speed:.1f} MB/s)")
        return True


def _build_convert_script(
    bagz_path: str,
    output_dir: str,
    num_shards: int,
    skip_verify: bool,
    seed: int | None,
) -> str:
    """Build a self-contained Python script string for the conversion subprocess."""
    return textwrap.dedent(f"""\
        import json, sys
        sys.path.insert(0, {_SRC_DIR!r})
        from catgpt.core.data.grain.bagz_to_bag import convert_bagz_to_bag
        try:
            games, before, after = convert_bagz_to_bag(
                {bagz_path!r},
                output_path={output_dir!r},
                verbose=False,
                skip_verify={skip_verify!r},
                num_shards={num_shards!r},
                seed={seed!r},
            )
            json.dump({{"ok": True, "games": games, "before": before, "after": after}}, sys.stdout)
        except Exception as e:
            json.dump({{"ok": False, "error": str(e)}}, sys.stdout)
            sys.exit(1)
    """)


async def convert_file(
    bagz_path: Path,
    output_dir: Path,
    num_shards: int,
    skip_verify: bool,
    seed: int | None,
    convert_sem: asyncio.Semaphore,
) -> tuple[int, int, int, str | None]:
    """Convert .bagz → sharded .bag files as an independent subprocess.

    Returns (games, positions_before, positions_after, error_or_none).
    """
    async with convert_sem:
        t0 = time.monotonic()
        print(f"  🔄 Converting {bagz_path.name}...")

        script = _build_convert_script(
            str(bagz_path), str(output_dir), num_shards, skip_verify, seed,
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()

        elapsed = time.monotonic() - t0

        # Parse structured JSON output from the subprocess
        try:
            result = json.loads(stdout_bytes.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            stderr_tail = stderr_bytes.decode(errors="replace").strip()[-500:]
            error = f"Subprocess produced no valid output (exit {proc.returncode}): {stderr_tail}"
            print(f"  ✗  Convert FAILED {bagz_path.name}: {error}")
            return 0, 0, 0, error

        if not result.get("ok"):
            error = result.get("error", "unknown error")
            print(f"  ✗  Convert FAILED {bagz_path.name}: {error}")
            return 0, 0, 0, error

        games = result["games"]
        before = result["before"]
        after = result["after"]
        print(
            f"  ✓  Converted {bagz_path.name} → {games} games, "
            f"{after} positions ({elapsed:.0f}s)"
        )
        return games, before, after, None


async def upload_to_wasabi(
    local_path: Path,
    s3_key: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Upload a file to the destination Wasabi bucket."""
    async with semaphore:
        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp",
            str(local_path),
            f"s3://{DEST_BUCKET}/{s3_key}",
            "--endpoint-url", WASABI_ENDPOINT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        elapsed = time.monotonic() - t0
        if proc.returncode == 0:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            speed = size_mb / elapsed if elapsed > 0 else 0
            print(f"  ⬆  Uploaded {s3_key} ({size_mb:.0f} MB, {speed:.1f} MB/s)")
            return True
        else:
            print(f"  ✗  Upload FAILED {s3_key}: {stderr.decode().strip()}")
            return False


# ---------------------------------------------------------------------------
# Full pipeline for one file
# ---------------------------------------------------------------------------


async def process_one_file(
    bagz_key: str,
    work_dir: Path,
    download_sem: asyncio.Semaphore,
    convert_sem: asyncio.Semaphore,
    upload_sem: asyncio.Semaphore,
    num_shards: int,
    skip_verify: bool,
    seed: int | None,
    stats: dict,
) -> None:
    """Download → convert → upload → cleanup for a single .bagz file."""
    stem = bagz_key.removesuffix(".bagz")
    bagz_local = work_dir / bagz_key
    shard_dir = work_dir / f"shards_{stem}"

    try:
        # 1. Download .bagz from source bucket
        ok = await download_from_wasabi(bagz_key, bagz_local, download_sem)
        if not ok:
            stats["failed"] += 1
            return

        # 2. Convert .bagz → sharded .bag files
        shard_dir.mkdir(parents=True, exist_ok=True)
        games, _before, after, error = await convert_file(
            bagz_local, shard_dir, num_shards, skip_verify, seed, convert_sem,
        )

        # 3. Delete .bagz immediately to reclaim ~1 GB
        bagz_local.unlink(missing_ok=True)

        if error:
            stats["failed"] += 1
            return

        # 4. Upload all shard .bag files
        shard_files = sorted(shard_dir.glob("*.bag"))
        if not shard_files:
            print(f"  ⚠  No .bag shards produced for {bagz_key}")
            stats["failed"] += 1
            return

        upload_tasks = [
            upload_to_wasabi(shard, shard.name, upload_sem)
            for shard in shard_files
        ]
        results = await asyncio.gather(*upload_tasks)
        upload_ok = all(results)

        if not upload_ok:
            failed_count = sum(1 for r in results if not r)
            print(f"  ⚠  {failed_count}/{len(shard_files)} shard uploads failed for {bagz_key}")
            stats["failed"] += 1
            return

        # 5. Success
        stats["completed"] += 1
        stats["total_games"] += games
        stats["total_positions"] += after
        c, f, t = stats["completed"], stats["failed"], stats["total_files"]
        print(
            f"  ✅ Done {bagz_key} | "
            f"{c}/{t} completed, {f} failed, "
            f"{stats['total_games']} games, {stats['total_positions']} positions"
        )

    except Exception as e:
        print(f"  ✗  Pipeline error {bagz_key}: {e}")
        stats["failed"] += 1
    finally:
        bagz_local.unlink(missing_ok=True)
        if shard_dir.exists():
            shutil.rmtree(shard_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # 1. Discover .bagz files in source bucket
    all_keys = await list_source_bucket()
    print(f"Found {len(all_keys)} .bagz files in source bucket\n")

    # 2. Filter by date range and sort chronologically
    filtered = filter_and_sort_by_date(all_keys, start_date, end_date)
    skipped_range = len(all_keys) - len(filtered)
    print(
        f"Date range {args.start_date} → {args.end_date}: "
        f"{len(filtered)} files (skipped {skipped_range} outside range)"
    )

    if not filtered:
        print("No files to process!")
        return

    # 3. Skip files already converted in dest bucket
    existing = await list_dest_bucket()
    to_process = [
        k for k in filtered
        if not already_converted(k, existing, args.num_shards)
    ]
    skipped_existing = len(filtered) - len(to_process)
    if skipped_existing:
        print(f"Skipping {skipped_existing} files already in destination bucket")
    print(f"Will process {len(to_process)} files\n")

    if not to_process:
        print("Nothing to do — all files already converted!")
        return

    # 4. Dry run?
    if args.dry_run:
        print("DRY RUN — files that would be processed:")
        for f in to_process:
            print(f"  {f}")
        return

    # 5. Run pipeline
    download_sem = asyncio.Semaphore(args.download_workers)
    convert_sem = asyncio.Semaphore(args.convert_workers)
    upload_sem = asyncio.Semaphore(args.upload_workers)
    in_flight_sem = asyncio.Semaphore(args.max_in_flight)

    stats = {
        "completed": 0,
        "failed": 0,
        "total_games": 0,
        "total_positions": 0,
        "total_files": len(to_process),
    }

    print("=" * 65)
    print(f"  Source bucket    : {SOURCE_BUCKET}")
    print(f"  Dest bucket      : {DEST_BUCKET}")
    print(f"  Download workers : {args.download_workers}")
    print(f"  Convert workers  : {args.convert_workers}")
    print(f"  Upload workers   : {args.upload_workers}")
    print(f"  Max in-flight    : {args.max_in_flight}")
    print(f"  Num shards       : {args.num_shards}")
    print(f"  Skip verify      : {args.skip_verify}")
    print(f"  Work directory   : {work_dir}")
    print(f"  Files to process : {len(to_process)}")
    print("=" * 65)
    print()

    t_start = time.monotonic()

    async def bounded_process(bagz_key: str) -> None:
        async with in_flight_sem:
            await process_one_file(
                bagz_key, work_dir,
                download_sem, convert_sem, upload_sem,
                args.num_shards, args.skip_verify, args.seed,
                stats,
            )

    tasks = [bounded_process(k) for k in to_process]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    print()
    print("=" * 65)
    print(f"  Pipeline complete!  ({hours}h {mins}m {secs}s)")
    print(f"  Completed  : {stats['completed']}/{stats['total_files']}")
    print(f"  Failed     : {stats['failed']}")
    print(f"  Games      : {stats['total_games']}")
    print(f"  Positions  : {stats['total_positions']}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download .bagz from Wasabi, convert to sharded .bag training data, upload back.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Process one week of data
  uv run python scripts/pipeline_bagz_to_training.py \\
      --start-date 2024-04-01 --end-date 2024-04-07

  # Process a month with more parallelism
  uv run python scripts/pipeline_bagz_to_training.py \\
      --start-date 2024-04-01 --end-date 2024-04-30 \\
      --download-workers 8 --convert-workers 6 --upload-workers 8 \\
      --num-shards 40

  # Dry run to see what would be processed
  uv run python scripts/pipeline_bagz_to_training.py \\
      --start-date 2024-04-01 --end-date 2024-04-07 --dry-run
""",
    )
    parser.add_argument(
        "--start-date", required=True,
        help="Start date inclusive, YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-date", required=True,
        help="End date inclusive, YYYY-MM-DD",
    )
    parser.add_argument(
        "--download-workers", type=int, default=5,
        help="Max concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--convert-workers", type=int, default=4,
        help="Max concurrent conversion subprocesses (default: 4)",
    )
    parser.add_argument(
        "--upload-workers", type=int, default=5,
        help="Max concurrent uploads (default: 5)",
    )
    parser.add_argument(
        "--max-in-flight", type=int, default=10,
        help="Max files in pipeline at once — bounds disk usage (default: 10)",
    )
    parser.add_argument(
        "--num-shards", type=int, default=40,
        help="Number of output shards per .bagz file (default: 40)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip game integrity verification during conversion (faster)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for shard assignment (default: random)",
    )
    parser.add_argument(
        "--work-dir", type=str,
        default=str(Path.home() / "bagz_training_pipeline_work"),
        help="Local scratch directory for temp files (default: ~/bagz_training_pipeline_work)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List files that would be processed without doing anything",
    )

    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
