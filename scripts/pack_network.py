#!/usr/bin/env python3
"""Pack N TensorRT engine files into a single .network container.

The .network file is the canonical packaged-network artifact for the C++
multi-engine BatchEvaluator. Each input .trt file holds an engine built with
exactly one optimization profile pinned at min == opt == max == bucket_size.
The packer concatenates them with a small TOC header so the C++ loader can
deserialize each per-bucket ICudaEngine in one call.

Format (all little-endian, no padding):

    offset  size       field
    ------  ----       -----
    0       16         magic    = b"CATGPT_NETWORK\\0\\0"
    16      4          version  = uint32 (1)
    20      4          num_engines = uint32 (N)
    24      N * 20     TOC entries:
                         uint32 bucket_size
                         uint64 offset      // byte offset from start of file
                         uint64 size        // engine blob size in bytes
    24+N*20 ...        concatenated engine blobs (in TOC order)

Bucket sizes are inferred from input filenames matching ``*.b{N}.trt``; pass
``--bucket-size`` if your filenames don't follow that convention (only valid
when packing a single file).

Usage:
    uv run scripts/pack_network.py -o main.network main.b1.trt main.b2.trt ...
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from pathlib import Path

MAGIC = b"CATGPT_NETWORK\0\0"
assert len(MAGIC) == 16
VERSION = 1
HEADER_SIZE = 24            # magic(16) + version(4) + num_engines(4)
TOC_ENTRY_SIZE = 4 + 8 + 8  # bucket(4) + offset(8) + size(8)

_BUCKET_RE = re.compile(r"\.b(\d+)\.trt$")


def infer_bucket(path: Path) -> int:
    m = _BUCKET_RE.search(path.name)
    if not m:
        raise ValueError(
            f"Cannot infer bucket size from filename {path.name!r}; "
            f"expected suffix '.b{{N}}.trt' (e.g. main.b8.trt)"
        )
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", required=True, type=Path, help="Output .network path")
    ap.add_argument(
        "--bucket-size",
        type=int,
        default=None,
        help="Override bucket size for the (single) input file. "
             "Only valid when packing exactly one .trt file.",
    )
    ap.add_argument("inputs", nargs="+", type=Path, help="Input .trt engine files")
    args = ap.parse_args()

    if args.bucket_size is not None and len(args.inputs) != 1:
        ap.error("--bucket-size is only valid with exactly one input file")

    # Resolve (bucket_size, path, blob) tuples.
    entries: list[tuple[int, Path, bytes]] = []
    for p in args.inputs:
        if not p.is_file():
            print(f"error: input not found: {p}", file=sys.stderr)
            return 1
        bucket = args.bucket_size if args.bucket_size is not None else infer_bucket(p)
        if bucket <= 0:
            print(f"error: bucket size must be positive (got {bucket} for {p})", file=sys.stderr)
            return 1
        blob = p.read_bytes()
        if not blob:
            print(f"error: empty engine file: {p}", file=sys.stderr)
            return 1
        entries.append((bucket, p, blob))

    # Sort ascending by bucket size for predictable layout, and reject duplicates.
    entries.sort(key=lambda e: e[0])
    seen: set[int] = set()
    for bucket, path, _ in entries:
        if bucket in seen:
            print(f"error: duplicate bucket size {bucket} (last seen in {path})", file=sys.stderr)
            return 1
        seen.add(bucket)

    n = len(entries)
    blob_offset = HEADER_SIZE + n * TOC_ENTRY_SIZE

    with args.output.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<II", VERSION, n))

        # TOC
        cursor = blob_offset
        for bucket, _, blob in entries:
            out.write(struct.pack("<IQQ", bucket, cursor, len(blob)))
            cursor += len(blob)

        # Blobs (in TOC order)
        for _, _, blob in entries:
            out.write(blob)

    total_bytes = args.output.stat().st_size
    print(
        f"Packed {n} engine(s) into {args.output} ({total_bytes / (1024 * 1024):.1f} MB)",
        file=sys.stderr,
    )
    for bucket, path, blob in entries:
        print(
            f"  bucket={bucket:>3}  size={len(blob) / (1024 * 1024):>6.1f} MB  src={path.name}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
