#!/usr/bin/env python3
"""Generate per-move teacher evaluations using a pre-trained model.

For each position in the input .bag files, this script:
1. Enumerates all legal moves
2. For terminal children (checkmate/stalemate/draw): assigns HL-Gauss distributions
   at fixed values (using the same sigma as training)
3. For non-terminal children: runs the teacher model to get predictions
4. Stores per-child bestQ probability distribution (HL-Gauss) and WDL value,
   flipped to the parent's perspective
5. Writes enriched .bag files with a new 'child_evals' field

Each child evaluation contains:
- move_uci:     The move in UCI format (e.g. "e2e4")
- bestq_probs:  Full HL-Gauss bestQ distribution (num_bins float16 bytes),
                from the parent's perspective (reversed from child's eval)
- wdl_value:    Scalar P(W)+0.5*P(D) from the parent's perspective

Architecture (3-level pipeline for maximum GPU utilization):
- Process pool:      expand positions + tokenize children (CPU-parallel, bypasses GIL)
- Background thread: collect pool results + assemble pre-tokenized GPU batches
- Main thread:       GPU inference + result mapping + ordered writes

The enriched .bag files are a superset of the original format — existing fields are
preserved, so they remain compatible with the current training pipeline (new fields
are simply ignored if the corresponding heads are disabled).

Usage:
    uv run python scripts/generate_move_values.py \\
        --checkpoint checkpoints_jax/WDL_main/final \\
        --input "~/training_bag/training-run1-test80-202507-00.bag" \\
        --output-dir ~/training_bag_enriched/ \\
        --batch-size 1024 \\
        --num-workers 8

    # Process multiple files with glob
    uv run python scripts/generate_move_values.py \\
        --checkpoint checkpoints_jax/WDL_main/final \\
        --input "~/training_bag/training-run1-test80-202507*.bag" \\
        --output-dir ~/training_bag_enriched/ \\
        --batch-size 1024
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
from loguru import logger
from scipy import special as scipy_special

from catgpt.core.data.grain.bagz import BagReader, BagWriter
from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize
from catgpt.jax.evaluation.checkpoint import load_checkpoint

# dtype mapping
DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}


# =============================================================================
# HL-Gauss helpers (module-level, used by both main thread and workers)
# =============================================================================


def hl_gauss_distribution(
    value: float,
    num_bins: int,
    sigma_ratio: float,
) -> np.ndarray:
    """Create an HL-Gauss probability distribution for a scalar value.

    Same algorithm as _hl_gauss_transform_numpy in the dataloader, but for
    a single scalar value (not batched).

    Args:
        value: Scalar target in [0, 1].
        num_bins: Number of bins.
        sigma_ratio: Ratio of sigma to bin_width.

    Returns:
        Probability distribution over bins, shape (num_bins,), float32.
    """
    edges = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float64)
    bin_width = 1.0 / num_bins
    sigma = sigma_ratio * bin_width
    sqrt2_sigma = np.sqrt(2.0) * sigma

    cdf_vals = scipy_special.erf((edges - value) / sqrt2_sigma)
    probs = cdf_vals[1:] - cdf_vals[:-1]
    z = cdf_vals[-1] - cdf_vals[0]
    if z > 0:
        probs /= z
    return probs.astype(np.float32)


# =============================================================================
# Worker process functions (module-level for pickling)
# =============================================================================

_worker_tok_config: TokenizerConfig | None = None
# Pre-computed terminal distributions (value → float16 bytes)
_worker_terminal_bestq: dict[float, bytes] = {}
_worker_terminal_wdl: dict[float, float] = {}


def _init_expand_worker(
    tok_config_dict: dict,
    terminal_bestq: dict[float, bytes],
    terminal_wdl: dict[float, float],
) -> None:
    """Initialize worker process with configs and precomputed terminal data.

    Args:
        tok_config_dict: Dict to construct TokenizerConfig.
        terminal_bestq: Precomputed HL-Gauss distributions for terminal values,
            stored as float16 bytes. Keys: 0.0, 0.5, 1.0.
        terminal_wdl: WDL values for terminal positions. Keys: 0.0, 0.5, 1.0.
    """
    global _worker_tok_config, _worker_terminal_bestq, _worker_terminal_wdl
    _worker_tok_config = TokenizerConfig(**tok_config_dict)
    _worker_terminal_bestq = terminal_bestq
    _worker_terminal_wdl = terminal_wdl


def _expand_position(
    args: tuple[int, bytes],
) -> tuple[
    int,
    dict,
    list[tuple[str, bytes, float]],
    list[tuple[str, np.ndarray]],
]:
    """Expand a position's legal moves and tokenize non-terminal children.

    Runs in a worker process. Returns pre-tokenized data ready for GPU batching.

    Args:
        args: Tuple of (parent_id, raw_bytes from .bag file).

    Returns:
        Tuple of:
            - parent_id: Position index in the .bag file
            - data: Original msgpack-decoded dict (passed through for enrichment)
            - terminals: list of (move_uci, bestq_probs_f16_bytes, wdl_value)
                for terminal positions (from parent's perspective)
            - children: list of (move_uci, tokens_array) for non-terminal positions
    """
    parent_id, raw_bytes = args
    data = msgpack.unpackb(raw_bytes, raw=False)
    board = chess.Board(data["fen"])

    terminals: list[tuple[str, bytes, float]] = []
    children: list[tuple[str, np.ndarray]] = []

    for move in board.legal_moves:
        move_uci = move.uci()
        board.push(move)

        if board.is_checkmate():
            # Side-to-move is checkmated → they lose
            # From parent's perspective: we win → 1.0
            terminals.append((
                move_uci,
                _worker_terminal_bestq[1.0],
                _worker_terminal_wdl[1.0],
            ))
        elif (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            # Draw → 0.5 from either perspective
            terminals.append((
                move_uci,
                _worker_terminal_bestq[0.5],
                _worker_terminal_wdl[0.5],
            ))
        else:
            tokens = tokenize(board.fen(), _worker_tok_config)
            children.append((move_uci, tokens))

        board.pop()

    return parent_id, data, terminals, children


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class PendingParent:
    """A parent position waiting for all children to be evaluated."""

    original_data: dict  # Original msgpack-decoded position
    total_children: int  # Number of legal moves
    completed: int = 0  # How many children have results
    # move_uci → (bestq_probs_f16_bytes, wdl_value) from parent's perspective
    child_evals: dict[str, tuple[bytes, float]] = field(default_factory=dict)


@dataclass
class ProducerBatch:
    """A pre-assembled batch from the producer thread, ready for GPU inference."""

    # Pre-assembled token array for GPU, or None for a parent-only update
    padded_tokens: np.ndarray | None  # (batch_size, seq_len)
    # (parent_id, move_uci) for each inference slot — used to map GPU results back
    batch_items: list[tuple[int, str]]
    # Number of real children in this batch (rest is padding)
    actual_size: int
    # Parents fully expanded since the last batch (need to be registered by main thread)
    new_parents: list[tuple[int, PendingParent]]


# =============================================================================
# Generator
# =============================================================================


class MoveValueGenerator:
    """Generates per-move teacher evaluations using a pre-trained model.

    Uses a 3-level pipeline for maximum GPU utilization:
    - Process pool: expand positions + tokenize children (CPU-parallel)
    - Background thread: collect pool results + assemble pre-tokenized GPU batches
    - Main thread: GPU inference + result mapping + ordered writes

    Per-child outputs (from parent's perspective):
    - bestq_probs: Full HL-Gauss bestQ distribution (num_bins float16 values)
    - wdl_value:   Scalar P(W)+0.5*P(D)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        batch_size: int = 1024,
        compute_dtype: str = "bfloat16",
        num_workers: int | None = None,
        prefetch_batches: int = 4,
    ) -> None:
        """Initialize the generator with a teacher model.

        Args:
            checkpoint_path: Path to teacher model checkpoint.
            batch_size: Fixed batch size for inference (padded for JIT stability).
            compute_dtype: Compute dtype string ("float32", "bfloat16", "float16").
            num_workers: Number of CPU worker processes for position expansion.
                Defaults to max(1, cpu_count - 2).
            prefetch_batches: Number of GPU batches to prefetch in the queue.
                Higher = more memory but better GPU utilization.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers or max(1, (os.cpu_count() or 4) - 2)
        self.prefetch_batches = prefetch_batches

        # Load teacher model
        logger.info(f"Loading teacher model from {checkpoint_path}")
        loaded = load_checkpoint(checkpoint_path)
        self.model = loaded.model
        self.params = loaded.params
        self.model_config = loaded.model_config
        self.tokenizer_config = loaded.tokenizer_config

        # Resolve compute dtype
        dtype = DTYPE_MAP.get(compute_dtype, jnp.bfloat16)
        self.compute_dtype = dtype

        # Create tokenizer config
        self._tok_config = TokenizerConfig(
            sequence_length=self.tokenizer_config.sequence_length,
            include_halfmove=self.tokenizer_config.include_halfmove,
        )
        self._seq_length = self.tokenizer_config.sequence_length

        # HL-Gauss parameters (from model config)
        num_bins = self.model_config.output_heads.value_num_bins
        sigma_ratio = self.model_config.output_heads.value_sigma_ratio
        self._num_bins = num_bins
        self._sigma_ratio = sigma_ratio

        # Precompute terminal distributions (HL-Gauss at known values)
        # These are from the parent's perspective and shared with workers.
        self._terminal_bestq: dict[float, bytes] = {}
        self._terminal_wdl: dict[float, float] = {}
        for val in [0.0, 0.5, 1.0]:
            probs = hl_gauss_distribution(val, num_bins, sigma_ratio)
            self._terminal_bestq[val] = probs.astype(np.float16).tobytes()
            self._terminal_wdl[val] = val  # WDL value equals the win probability for terminals

        # JIT compile teacher inference
        self._apply_fn = jax.jit(
            lambda params, x: self.model.apply(
                params, x, train=False, compute_dtype=dtype
            )
        )

        # Warmup JIT with fixed batch size
        logger.info(f"Warming up JIT with batch_size={batch_size}")
        dummy_input = jnp.zeros((batch_size, self._seq_length), dtype=jnp.int32)
        outputs = self._apply_fn(self.params, dummy_input)
        jax.block_until_ready(outputs)

        # Verify teacher has required heads
        if "bestq_logit" not in outputs:
            raise ValueError(
                "Teacher model must have value_head enabled. "
                "The model output does not contain 'bestq_logit'."
            )
        if "wdl_value" not in outputs:
            raise ValueError(
                "Teacher model must have value_head enabled (with WDL). "
                "The model output does not contain 'wdl_value'."
            )

        logger.info(
            f"Teacher model loaded: {sum(p.size for p in jax.tree_util.tree_leaves(self.params)):,} params, "
            f"HL-Gauss bins={num_bins}, sigma_ratio={sigma_ratio}, "
            f"compute_dtype={compute_dtype}, "
            f"workers={self.num_workers}, prefetch={self.prefetch_batches}"
        )

    # -------------------------------------------------------------------------
    # Background producer thread
    # -------------------------------------------------------------------------

    def _batch_producer(
        self,
        reader: BagReader,
        num_positions: int,
        batch_queue: queue.Queue[ProducerBatch | None],
        producer_stats: dict[str, int],
    ) -> None:
        """Background thread: parallel expansion via process pool → batch assembly → queue.

        Uses multiprocessing.Pool.imap to expand positions in parallel while
        assembling pre-tokenized GPU batches. The pool works ahead of consumption,
        providing natural pipelining.

        Args:
            reader: BagReader for the input file.
            num_positions: Total number of positions in the file.
            batch_queue: Thread-safe queue to put assembled batches into.
            producer_stats: Shared dict for the producer to report stats.
        """
        tok_dict = {
            "sequence_length": self._tok_config.sequence_length,
            "include_halfmove": self._tok_config.include_halfmove,
        }

        # Internal buffer for assembling batches
        inference_items: list[tuple[int, str, np.ndarray]] = []  # (pid, move_uci, tokens)
        parent_updates: list[tuple[int, PendingParent]] = []

        total_children = 0
        terminal_children = 0

        try:
            with multiprocessing.Pool(
                self.num_workers,
                initializer=_init_expand_worker,
                initargs=(tok_dict, self._terminal_bestq, self._terminal_wdl),
            ) as pool:
                # imap returns results in order and works ahead (pool expands future
                # positions while we assemble batches from earlier results)
                results_iter = pool.imap(
                    _expand_position,
                    ((pid, reader[pid]) for pid in range(num_positions)),
                    chunksize=64,
                )

                for parent_id, data, terminals, children_tokens in results_iter:
                    # Create PendingParent with terminals pre-filled
                    pending = PendingParent(
                        original_data=data,
                        total_children=len(terminals) + len(children_tokens),
                    )
                    for move_uci, bestq_bytes, wdl_val in terminals:
                        pending.child_evals[move_uci] = (bestq_bytes, wdl_val)
                        pending.completed += 1

                    parent_updates.append((parent_id, pending))
                    total_children += len(terminals) + len(children_tokens)
                    terminal_children += len(terminals)

                    # Add non-terminal children to inference buffer
                    for move_uci, tokens in children_tokens:
                        inference_items.append((parent_id, move_uci, tokens))

                    # Assemble and send full batches
                    while len(inference_items) >= self.batch_size:
                        batch_slice = inference_items[: self.batch_size]
                        inference_items = inference_items[self.batch_size :]

                        padded = np.zeros(
                            (self.batch_size, self._seq_length), dtype=np.uint8
                        )
                        batch_items = []
                        for i, (pid, muci, tokens) in enumerate(batch_slice):
                            padded[i] = tokens
                            batch_items.append((pid, muci))

                        batch_queue.put(
                            ProducerBatch(
                                padded_tokens=padded,
                                batch_items=batch_items,
                                actual_size=len(batch_slice),
                                new_parents=parent_updates,
                            )
                        )
                        parent_updates = []

                    # Flush accumulated parents if no inference work (all-terminal runs)
                    # to prevent the main thread from stalling on writes
                    if len(parent_updates) > 200 and len(inference_items) < self.batch_size:
                        batch_queue.put(
                            ProducerBatch(
                                padded_tokens=None,
                                batch_items=[],
                                actual_size=0,
                                new_parents=parent_updates,
                            )
                        )
                        parent_updates = []

            # --- Pool closed, send remaining items ---

            # Final partial batch (< batch_size children)
            if inference_items:
                actual = len(inference_items)
                padded = np.zeros(
                    (self.batch_size, self._seq_length), dtype=np.uint8
                )
                batch_items = []
                for i, (pid, muci, tokens) in enumerate(inference_items):
                    padded[i] = tokens
                    batch_items.append((pid, muci))

                batch_queue.put(
                    ProducerBatch(
                        padded_tokens=padded,
                        batch_items=batch_items,
                        actual_size=actual,
                        new_parents=parent_updates,
                    )
                )
                parent_updates = []

            # Remaining parent-only updates (parents with all terminal children)
            if parent_updates:
                batch_queue.put(
                    ProducerBatch(
                        padded_tokens=None,
                        batch_items=[],
                        actual_size=0,
                        new_parents=parent_updates,
                    )
                )

        except Exception:
            logger.exception("Producer thread failed")
        finally:
            producer_stats["total_children"] = total_children
            producer_stats["terminal_children"] = terminal_children
            batch_queue.put(None)  # Sentinel: done

    # -------------------------------------------------------------------------
    # Write helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_completed_parents(
        pending_parents: dict[int, PendingParent],
        write_queue: deque[int],
        writer: BagWriter,
    ) -> int:
        """Write completed parents to output .bag in order.

        Drains the write queue up to the first incomplete parent, preserving
        the original position ordering.

        Each position is enriched with a 'child_evals' field:
            list of [move_uci, bestq_probs_f16_bytes, wdl_value]

        The bestq_probs are stored as float16 numpy bytes for compactness.
        To decode: np.frombuffer(bestq_bytes, dtype=np.float16)

        Returns:
            Number of parents written.
        """
        written = 0
        while write_queue:
            pid = write_queue[0]
            parent = pending_parents.get(pid)
            if parent is None:
                write_queue.popleft()
                continue

            if parent.completed < parent.total_children:
                break  # Not done yet, stop (preserves order)

            # Enrich original data with child evaluations
            data = parent.original_data
            data["child_evals"] = [
                [move_uci, bestq_bytes, wdl_val]
                for move_uci, (bestq_bytes, wdl_val) in parent.child_evals.items()
            ]

            writer.write(msgpack.packb(data, use_bin_type=True))

            del pending_parents[pid]
            write_queue.popleft()
            written += 1

        return written

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def process_file(
        self, input_path: str, output_path: str
    ) -> dict[str, int]:
        """Process a single .bag file and write enriched output.

        Launches a background producer thread that expands positions via a
        process pool and assembles pre-tokenized GPU batches. The main thread
        consumes batches, runs GPU inference, maps results, and writes output.

        Args:
            input_path: Path to input .bag file.
            output_path: Path for output enriched .bag file.

        Returns:
            Dictionary with processing statistics.
        """
        reader = BagReader(input_path)
        num_positions = len(reader)
        logger.info(
            f"Processing {num_positions:,} positions from {input_path} "
            f"(workers={self.num_workers})"
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Shared state
        pending_parents: dict[int, PendingParent] = {}
        write_queue: deque[int] = deque()

        # Stats
        inference_batches = 0
        inferred_children = 0
        positions_written = 0
        producer_stats: dict[str, int] = {}

        # Prefetch queue: producer thread puts pre-assembled batches here
        batch_queue: queue.Queue[ProducerBatch | None] = queue.Queue(
            maxsize=self.prefetch_batches
        )

        # Launch producer thread (which internally uses a process pool)
        producer = threading.Thread(
            target=self._batch_producer,
            args=(reader, num_positions, batch_queue, producer_stats),
            daemon=True,
        )

        start_time = time.time()
        last_log_time = start_time
        producer.start()

        with BagWriter(output_path, compress=False) as writer:
            while True:
                item = batch_queue.get()
                if item is None:
                    break  # Producer done

                # Register new parents (expanded since last batch)
                for pid, pending in item.new_parents:
                    pending_parents[pid] = pending
                    write_queue.append(pid)

                # Run GPU inference if there's actual work
                if item.padded_tokens is not None and item.actual_size > 0:
                    tokens_jax = jnp.array(item.padded_tokens)
                    outputs = self._apply_fn(self.params, tokens_jax)

                    # Extract bestQ distribution (full softmax probs)
                    bestq_logits = outputs["bestq_logit"][: item.actual_size]
                    bestq_probs = jax.nn.softmax(
                        bestq_logits.astype(jnp.float32), axis=-1
                    )
                    # Flip to parent's perspective: reverse bin order
                    # If child sees value=0.8, parent sees value=0.2
                    # Reversing the prob array maps bin[i] → bin[N-1-i]
                    bestq_probs_parent = bestq_probs[:, ::-1]

                    # Extract WDL value and flip to parent's perspective
                    wdl_values = outputs["wdl_value"][: item.actual_size]
                    wdl_values_parent = 1.0 - wdl_values.astype(jnp.float32)

                    # Transfer to CPU
                    bestq_np = np.array(bestq_probs_parent)  # (actual_size, num_bins)
                    wdl_np = np.array(wdl_values_parent)  # (actual_size,)

                    # Map results back to parents
                    for i, (parent_id, move_uci) in enumerate(item.batch_items):
                        parent = pending_parents[parent_id]
                        # Store bestQ probs as float16 bytes for compactness
                        probs_bytes = bestq_np[i].astype(np.float16).tobytes()
                        wdl_val = round(float(wdl_np[i]), 6)
                        parent.child_evals[move_uci] = (probs_bytes, wdl_val)
                        parent.completed += 1

                    inference_batches += 1
                    inferred_children += item.actual_size

                # Write completed parents (preserving order)
                positions_written += self._write_completed_parents(
                    pending_parents, write_queue, writer
                )

                # Periodic logging
                now = time.time()
                if now - last_log_time > 30:
                    elapsed = now - start_time
                    pending_count = len(pending_parents)
                    qsize = batch_queue.qsize()
                    logger.info(
                        f"Progress: written={positions_written:,}/{num_positions:,} "
                        f"({positions_written / num_positions:.1%}) | "
                        f"{positions_written / elapsed:.0f} pos/s | "
                        f"pending={pending_count}, prefetch_queue={qsize}, "
                        f"batches={inference_batches}"
                    )
                    last_log_time = now

        producer.join()
        elapsed = time.time() - start_time

        # Merge stats from producer
        total_children = producer_stats.get("total_children", 0)
        terminal_children = producer_stats.get("terminal_children", 0)

        # Sanity checks
        if pending_parents:
            logger.error(
                f"BUG: {len(pending_parents)} parents still pending after processing! "
                f"IDs: {list(pending_parents.keys())[:10]}"
            )
        if positions_written != num_positions:
            logger.warning(
                f"Written {positions_written} != input {num_positions} positions"
            )

        logger.info(
            f"Completed {input_path} -> {output_path}\n"
            f"  Positions: {num_positions:,}\n"
            f"  Children: {total_children:,} "
            f"(terminal={terminal_children:,}, "
            f"inferred={inferred_children:,})\n"
            f"  Batches: {inference_batches:,}\n"
            f"  Time: {elapsed:.1f}s ({num_positions / elapsed:.0f} pos/s)"
        )

        return {
            "total_positions": num_positions,
            "total_children": total_children,
            "terminal_children": terminal_children,
            "inferred_children": inferred_children,
            "inference_batches": inference_batches,
            "positions_written": positions_written,
        }


def resolve_input_paths(input_pattern: str) -> list[str]:
    """Resolve input path pattern to a list of actual file paths.

    Supports tilde expansion, glob patterns, and directories (auto-globs *.bag).

    Args:
        input_pattern: File path, directory, or glob pattern (e.g., "~/data/*.bag").

    Returns:
        Sorted list of resolved file paths.
    """
    expanded = str(Path(input_pattern).expanduser())

    # If it's a directory, auto-glob for .bag files inside it
    if Path(expanded).is_dir():
        paths = sorted(glob.glob(os.path.join(expanded, "*.bag")))
        if not paths:
            raise FileNotFoundError(f"No .bag files found in directory: {expanded}")
        return paths

    if "*" in expanded:
        paths = sorted(str(p) for p in Path("/").glob(expanded.lstrip("/")))
    else:
        paths = [expanded]
    # Fallback to glob module for complex patterns with multiple wildcards
    if not paths or (len(paths) == 1 and not Path(paths[0]).exists()):
        paths = sorted(glob.glob(expanded))
    if not paths:
        raise FileNotFoundError(f"No files found matching: {input_pattern}")
    return paths


def main() -> None:
    """Main entry point for move value generation."""
    parser = argparse.ArgumentParser(
        description="Generate per-move teacher evaluations (bestQ probs + WDL value)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to teacher model checkpoint directory.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input .bag file path or glob pattern (e.g., '~/data/*.bag').",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for enriched .bag files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for teacher inference (default: 1024).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of CPU workers for position expansion (default: cpu_count - 2).",
    )
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=4,
        help="Number of GPU batches to prefetch (default: 4).",
    )
    parser.add_argument(
        "--compute-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Compute dtype for teacher inference (default: bfloat16).",
    )
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=["default", "high", "highest"],
        help="JAX matmul precision (default: high).",
    )

    args = parser.parse_args()

    # Set matmul precision
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    # Log device info
    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} - {devices}")
    logger.info(f"JAX backend: {jax.default_backend()}")

    # Create generator
    generator = MoveValueGenerator(
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        num_workers=args.num_workers,
        prefetch_batches=args.prefetch_batches,
    )

    # Resolve input files
    input_paths = resolve_input_paths(args.input)
    logger.info(f"Found {len(input_paths)} input file(s)")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    total_stats: dict[str, int] = {}
    for input_path in input_paths:
        # Output file has the same name, in the output directory
        output_path = output_dir / Path(input_path).name
        logger.info(f"Processing: {input_path} -> {output_path}")

        stats = generator.process_file(input_path, str(output_path))

        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value

    # Summary
    if len(input_paths) > 1:
        logger.info(
            f"\n=== TOTAL SUMMARY ({len(input_paths)} files) ===\n"
            f"  Positions: {total_stats.get('total_positions', 0):,}\n"
            f"  Children: {total_stats.get('total_children', 0):,} "
            f"(terminal={total_stats.get('terminal_children', 0):,}, "
            f"inferred={total_stats.get('inferred_children', 0):,})\n"
            f"  Batches: {total_stats.get('inference_batches', 0):,}"
        )


if __name__ == "__main__":
    main()
