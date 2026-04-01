#!/usr/bin/env python3
"""Multi-shard training orchestrator for S1.

Trains sequentially on 40 shards, uploading checkpoints to Wasabi between shards.
Each shard runs as a separate subprocess for clean memory management.

Usage:
    # Start from shard 0
    uv run python scripts/train_shards.py

    # Resume from shard 15
    uv run python scripts/train_shards.py --start-shard 15

    # Dry run (print commands without executing)
    uv run python scripts/train_shards.py --dry-run
"""

import argparse
import shutil
import subprocess
from pathlib import Path

WASABI_ENDPOINT = "https://s3.us-west-1.wasabisys.com"
WASABI_BUCKET = "chessbench-checkpoints"
WASABI_PREFIX = "S1"

RUN_NAME = "S1"
WANDB_PROJECT = "S1"

TOTAL_SHARDS = 40

# LR schedule spans the full run; each shard trains for steps_per_run steps
TOTAL_STEPS = 6_000_000
STEPS_PER_RUN = 150_000

# Local paths
CHECKPOINT_BASE = Path("checkpoints_jax")
RESUME_CHECKPOINT_DIR = Path("/mnt/nvme0/checkpoints/resume_from")


def get_shard_data_path(shard_index: int) -> str:
    """Get the local data path for a shard.

    Shards are distributed across NVMe drives round-robin.
    Expected structure: /mnt/nvmeN/shard_X/*.bag
    """
    nvme_index = shard_index % 6
    return f"/mnt/nvme{nvme_index}/shard_{shard_index}/*.bag"


def upload_checkpoint(shard_index: int, checkpoint_path: Path, dry_run: bool = False) -> None:
    """Upload checkpoint to Wasabi after completing a shard."""
    s3_path = f"s3://{WASABI_BUCKET}/{WASABI_PREFIX}/shard_{shard_index:02d}/"

    cmd = [
        "aws", "s3", "sync",
        str(checkpoint_path),
        s3_path,
        f"--endpoint-url={WASABI_ENDPOINT}",
    ]

    print(f"\n📤 Uploading checkpoint to {s3_path}")
    if dry_run:
        print(f"   [DRY RUN] {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)
        print(f"   ✓ Upload complete")


def download_checkpoint(shard_index: int, local_path: Path, dry_run: bool = False) -> None:
    """Download checkpoint from Wasabi before starting a new shard."""
    s3_path = f"s3://{WASABI_BUCKET}/{WASABI_PREFIX}/shard_{shard_index:02d}/"

    cmd = [
        "aws", "s3", "sync",
        s3_path,
        str(local_path),
        f"--endpoint-url={WASABI_ENDPOINT}",
    ]

    print(f"\n📥 Downloading checkpoint from {s3_path}")
    if dry_run:
        print(f"   [DRY RUN] {' '.join(cmd)}")
    else:
        local_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)
        print(f"   ✓ Download complete")


def train_shard(
    shard_index: int,
    resume_from: Path | None = None,
    dry_run: bool = False,
    test_mode: bool = False,
) -> None:
    """Run training for a single shard as a subprocess."""
    shard_data_path = get_shard_data_path(shard_index)

    total_steps = TOTAL_STEPS
    steps_per_run = STEPS_PER_RUN

    if test_mode:
        total_steps = 100   # 5 shards × 20 steps — enough for a full test run
        steps_per_run = 20

    cmd = [
        "uv", "run", "python", "scripts/train_jax.py",
        f"data.train_path={shard_data_path}",
        f"data.val_path={shard_data_path}",
        f"+run_name={RUN_NAME}",
        f"wandb.project={WANDB_PROJECT}",
        f"wandb.run_name=shard_{shard_index:02d}",
        f"wandb.tags=[S1,shard_{shard_index:02d}]",
        f"training.total_steps={total_steps}",
        f"training.steps_per_run={steps_per_run}",
    ]

    if test_mode:
        cmd.extend([
            "training.steps_per_epoch=10",
            "training.batch_size=64",
            "wandb.enabled=false",
        ])

    if resume_from is not None:
        cmd.append(f"+resume_from={resume_from}")

    print(f"\n🚀 Training shard {shard_index}")
    print(f"   Data: {shard_data_path}")
    print(f"   Steps: {steps_per_run} (total_steps={total_steps})")
    if resume_from:
        print(f"   Resuming from: {resume_from}")
    if test_mode:
        print(f"   [TEST MODE] batch_size=64, steps_per_run=20")

    if dry_run:
        print(f"   [DRY RUN] {' '.join(cmd)}")
    else:
        subprocess.run(cmd, check=True)


def cleanup_local_checkpoint(path: Path, dry_run: bool = False) -> None:
    """Remove local checkpoint directory to free disk space."""
    if path.exists():
        print(f"\n🗑️  Cleaning up {path}")
        if dry_run:
            print(f"   [DRY RUN] Would remove {path}")
        else:
            shutil.rmtree(path)
            print(f"   ✓ Removed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-shard training orchestrator")
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Shard index to start from (default: 0)",
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=TOTAL_SHARDS - 1,
        help=f"Shard index to end at, inclusive (default: {TOTAL_SHARDS - 1})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: small batch size (64), few steps (20), W&B disabled",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"S1 Multi-Shard Training")
    print(f"Shards: {args.start_shard} → {args.end_shard}")
    print(f"W&B Project: {WANDB_PROJECT}")
    print(f"Wasabi: s3://{WASABI_BUCKET}/{WASABI_PREFIX}/")
    print("=" * 60)

    for shard_index in range(args.start_shard, args.end_shard + 1):
        print(f"\n{'='*60}")
        print(f"SHARD {shard_index}/{args.end_shard}")
        print(f"{'='*60}")

        # Download previous checkpoint (if not first shard)
        resume_from = None
        if shard_index > 0:
            resume_from = RESUME_CHECKPOINT_DIR
            download_checkpoint(shard_index - 1, resume_from, dry_run=args.dry_run)

        # Train this shard
        train_shard(shard_index, resume_from, dry_run=args.dry_run, test_mode=args.test)

        # Upload checkpoint
        checkpoint_path = CHECKPOINT_BASE / RUN_NAME / "final"
        upload_checkpoint(shard_index, checkpoint_path, dry_run=args.dry_run)

        # Cleanup resume checkpoint to free disk space
        if resume_from is not None:
            cleanup_local_checkpoint(resume_from, dry_run=args.dry_run)

        print(f"\n✅ Shard {shard_index} complete!")

    print("\n" + "=" * 60)
    print("🎉 All shards complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
