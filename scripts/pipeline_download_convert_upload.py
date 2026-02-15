#!/usr/bin/env python3
"""Pipeline: Download Leela Chess .tar files, convert to .bagz, upload to Wasabi.

This script orchestrates a massively parallel pipeline that:
1. Fetches the file listing from data.lczero.org
2. Filters by date range
3. Skips files already uploaded to Wasabi
4. For each file (bounded concurrency):
   a. Downloads .tar from lczero.org
   b. Converts to .bagz using existing conversion code
   c. Uploads .bagz to Wasabi S3
   d. Deletes local .tar and .bagz to free disk space

The pipeline never waits for all downloads to finish before converting —
as soon as a .tar finishes downloading, a conversion worker picks it up.

Usage:
    uv run python scripts/pipeline_download_convert_upload.py \\
        --start-date 2024-04-01 --end-date 2024-04-30

    uv run python scripts/pipeline_download_convert_upload.py \\
        --start-date 2024-04-01 --end-date 2024-06-30 \\
        --download-workers 8 --convert-workers 4 --upload-workers 8

    # Dry run to see what would be processed
    uv run python scripts/pipeline_download_convert_upload.py \\
        --start-date 2024-04-01 --end-date 2024-04-07 --dry-run
"""

import argparse
import asyncio
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Add src to path so process pool workers can import catgpt
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from catgpt.core.data.grain.tar_to_bag import convert_tar_to_bag  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL = "https://data.lczero.org/files/training_data/test80/"
WASABI_ENDPOINT = "https://s3.us-west-1.wasabisys.com"
WASABI_BUCKET = "leela-processed-bagz"

# Regex to extract filename + embedded date from Apache directory listing
# Example: <a href="training-run1-test80-20240401-0017.tar">
FILENAME_RE = re.compile(
    r'<a\s+href="(training-run1-test80-(\d{8})-\d{4}\.tar)">'
)

# Files smaller than this are empty tars (10240 bytes) — skip them
MIN_FILE_SIZE = 100_000


# ---------------------------------------------------------------------------
# Phase 0: Discover files
# ---------------------------------------------------------------------------


def fetch_file_listing() -> list[tuple[str, str, int]]:
    """Fetch and parse the directory listing from lczero.org.

    Returns:
        List of (filename, date_str_YYYYMMDD, file_size_bytes) tuples.
    """
    print("Fetching file listing from lczero.org...")
    resp = requests.get(BASE_URL, timeout=120)
    resp.raise_for_status()
    html = resp.text

    results: list[tuple[str, str, int]] = []
    for line in html.split("\n"):
        m = FILENAME_RE.search(line)
        if not m:
            continue
        filename = m.group(1)
        date_str = m.group(2)
        # Try to grab the file size from the end of the line
        size_m = re.search(r"(\d{4,})\s*$", line.strip())
        file_size = int(size_m.group(1)) if size_m else 0
        results.append((filename, date_str, file_size))

    return results


async def get_existing_in_wasabi() -> set[str]:
    """List all object keys currently in the Wasabi bucket."""
    print("Checking existing files in Wasabi...")
    proc = await asyncio.create_subprocess_exec(
        "aws", "s3api", "list-objects-v2",
        "--bucket", WASABI_BUCKET,
        "--endpoint-url", WASABI_ENDPOINT,
        "--query", "Contents[].Key",
        "--output", "text",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        # Empty bucket returns error or empty — that's fine
        err = stderr.decode().strip()
        if "NoSuchKey" in err or "None" in stdout.decode():
            return set()
        # Otherwise just warn and continue (maybe empty bucket)
        print(f"  Warning listing bucket: {err}")
        return set()

    existing: set[str] = set()
    for token in stdout.decode().strip().split():
        token = token.strip()
        if token and token != "None":
            existing.add(token)
    return existing


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def download_file(
    filename: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download a .tar file using wget (handles retries, resume, etc.)."""
    url = BASE_URL + filename
    async with semaphore:
        t0 = time.monotonic()
        print(f"  ⬇  Downloading {filename}...")
        proc = await asyncio.create_subprocess_exec(
            "wget", "-q",
            "--continue",
            "--timeout=300",
            "--tries=5",
            "--waitretry=10",
            "-O", str(dest),
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            print(f"  ✗  Download FAILED {filename}: {stderr.decode().strip()}")
            dest.unlink(missing_ok=True)
            return False

        elapsed = time.monotonic() - t0
        size_mb = dest.stat().st_size / (1024 * 1024)
        speed = size_mb / elapsed if elapsed > 0 else 0
        print(f"  ✓  Downloaded {filename} ({size_mb:.0f} MB in {elapsed:.0f}s, {speed:.1f} MB/s)")
        return True


def _convert_tar_worker(tar_path: str, bagz_path: str) -> tuple[str, int, str | None]:
    """Convert a single .tar → .bagz. Runs in a separate process."""
    try:
        games = convert_tar_to_bag(
            tar_path,
            output_path=bagz_path,
            compress=True,
            verbose=False,
        )
        return (Path(tar_path).name, games, None)
    except Exception as e:
        return (Path(tar_path).name, 0, str(e))


async def convert_file(
    tar_path: Path,
    bagz_path: Path,
    pool: ProcessPoolExecutor,
) -> tuple[int, str | None]:
    """Convert .tar → .bagz in the process pool."""
    loop = asyncio.get_running_loop()
    t0 = time.monotonic()
    print(f"  🔄 Converting {tar_path.name}...")
    _name, games, error = await loop.run_in_executor(
        pool,
        _convert_tar_worker,
        str(tar_path),
        str(bagz_path),
    )
    elapsed = time.monotonic() - t0
    if error:
        print(f"  ✗  Convert FAILED {tar_path.name}: {error}")
    else:
        print(f"  ✓  Converted {tar_path.name} → {games} games ({elapsed:.0f}s)")
    return games, error


async def upload_file(
    bagz_path: Path,
    s3_key: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Upload a .bagz file to Wasabi using aws CLI."""
    async with semaphore:
        t0 = time.monotonic()
        print(f"  ⬆  Uploading {s3_key}...")
        proc = await asyncio.create_subprocess_exec(
            "aws", "s3", "cp",
            str(bagz_path),
            f"s3://{WASABI_BUCKET}/{s3_key}",
            "--endpoint-url", WASABI_ENDPOINT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        elapsed = time.monotonic() - t0
        if proc.returncode == 0:
            size_mb = bagz_path.stat().st_size / (1024 * 1024)
            speed = size_mb / elapsed if elapsed > 0 else 0
            print(f"  ✓  Uploaded {s3_key} ({size_mb:.0f} MB in {elapsed:.0f}s, {speed:.1f} MB/s)")
            return True
        else:
            print(f"  ✗  Upload FAILED {s3_key}: {stderr.decode().strip()}")
            return False


# ---------------------------------------------------------------------------
# Full pipeline for one file
# ---------------------------------------------------------------------------


async def process_one_file(
    filename: str,
    work_dir: Path,
    download_sem: asyncio.Semaphore,
    upload_sem: asyncio.Semaphore,
    pool: ProcessPoolExecutor,
    stats: dict,
) -> None:
    """Download → convert → upload → cleanup for a single file."""
    tar_path = work_dir / filename
    bagz_name = filename.replace(".tar", ".bagz")
    bagz_path = work_dir / bagz_name

    try:
        # 1. Download
        ok = await download_file(filename, tar_path, download_sem)
        if not ok:
            stats["failed"] += 1
            return

        # 2. Convert
        games, error = await convert_file(tar_path, bagz_path, pool)

        # 3. Delete .tar immediately to reclaim ~1.7 GB
        tar_path.unlink(missing_ok=True)

        if error:
            stats["failed"] += 1
            return

        # 4. Upload .bagz
        ok = await upload_file(bagz_path, bagz_name, upload_sem)
        if not ok:
            stats["failed"] += 1
            return

        # 5. Delete .bagz
        bagz_path.unlink(missing_ok=True)

        stats["completed"] += 1
        stats["total_games"] += games
        c, f, t = stats["completed"], stats["failed"], stats["total_files"]
        print(
            f"  ✅ Done {filename} | "
            f"{c}/{t} completed, {f} failed, {stats['total_games']} games"
        )

    except Exception as e:
        print(f"  ✗  Pipeline error {filename}: {e}")
        stats["failed"] += 1
    finally:
        # Always clean up local files
        tar_path.unlink(missing_ok=True)
        bagz_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_pipeline(args: argparse.Namespace) -> None:
    """Main async pipeline."""
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # 1. Fetch listing
    all_files = fetch_file_listing()
    print(f"Found {len(all_files)} total files on server\n")

    # 2. Filter by date range + minimum size
    filtered: list[str] = []
    for filename, date_str, file_size in all_files:
        file_date = datetime.strptime(date_str, "%Y%m%d")
        if start_date <= file_date <= end_date and file_size >= MIN_FILE_SIZE:
            filtered.append(filename)

    print(
        f"Date range {args.start_date} → {args.end_date}: "
        f"{len(filtered)} files (skipped {len(all_files) - len(filtered)} "
        f"outside range or too small)"
    )

    if not filtered:
        print("No files to process!")
        return

    # 3. Skip files already in Wasabi
    existing = await get_existing_in_wasabi()
    to_process = [f for f in filtered if f.replace(".tar", ".bagz") not in existing]
    skipped = len(filtered) - len(to_process)
    if skipped:
        print(f"Skipping {skipped} files already in Wasabi")
    print(f"Will process {len(to_process)} files\n")

    if not to_process:
        print("Nothing to do — all files already uploaded!")
        return

    # 4. Dry run?
    if args.dry_run:
        print("DRY RUN — files that would be processed:")
        for f in to_process:
            print(f"  {f}")
        return

    # 5. Run pipeline
    download_sem = asyncio.Semaphore(args.download_workers)
    upload_sem = asyncio.Semaphore(args.upload_workers)
    in_flight_sem = asyncio.Semaphore(args.max_in_flight)

    stats = {
        "completed": 0,
        "failed": 0,
        "total_games": 0,
        "total_files": len(to_process),
    }

    print("=" * 65)
    print(f"  Download workers : {args.download_workers}")
    print(f"  Convert workers  : {args.convert_workers}")
    print(f"  Upload workers   : {args.upload_workers}")
    print(f"  Max in-flight    : {args.max_in_flight}")
    print(f"  Work directory   : {work_dir}")
    print(f"  Files to process : {len(to_process)}")
    print("=" * 65)
    print()

    t_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.convert_workers) as pool:

        async def bounded_process(filename: str) -> None:
            """Wraps process_one_file with an in-flight semaphore."""
            async with in_flight_sem:
                await process_one_file(
                    filename, work_dir,
                    download_sem, upload_sem, pool, stats,
                )

        tasks = [bounded_process(f) for f in to_process]
        await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    mins, secs = divmod(rem, 60)

    print()
    print("=" * 65)
    print(f"  Pipeline complete!  ({hours}h {mins}m {secs}s)")
    print(f"  Completed : {stats['completed']}/{stats['total_files']}")
    print(f"  Failed    : {stats['failed']}")
    print(f"  Games     : {stats['total_games']}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Leela .tar files, convert to .bagz, upload to Wasabi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Process one week of data
  uv run python scripts/pipeline_download_convert_upload.py \\
      --start-date 2024-04-01 --end-date 2024-04-07

  # Process a month with more parallelism
  uv run python scripts/pipeline_download_convert_upload.py \\
      --start-date 2024-04-01 --end-date 2024-04-30 \\
      --download-workers 8 --convert-workers 6 --upload-workers 8

  # Dry run to see what would be processed
  uv run python scripts/pipeline_download_convert_upload.py \\
      --start-date 2024-04-01 --end-date 2024-04-07 --dry-run
""",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date inclusive, YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date inclusive, YYYY-MM-DD format",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=5,
        help="Max concurrent downloads (default: 5)",
    )
    parser.add_argument(
        "--convert-workers",
        type=int,
        default=4,
        help="Max concurrent conversions / process pool size (default: 4)",
    )
    parser.add_argument(
        "--upload-workers",
        type=int,
        default=5,
        help="Max concurrent uploads (default: 5)",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=10,
        help="Max files in the pipeline at once — bounds disk usage (default: 10)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(Path.home() / "lczero_pipeline_work"),
        help="Local scratch directory for temp files (default: ~/lczero_pipeline_work)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without doing anything",
    )

    args = parser.parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
