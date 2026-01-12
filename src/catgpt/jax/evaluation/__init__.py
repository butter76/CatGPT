"""JAX evaluation utilities for CatGPT chess models.

This module provides:
- Checkpoint loading utilities
- Chess engines (ValueEngine for 1-move lookahead)
- Evaluation benchmarks (puzzle benchmarks)

Usage:
    from catgpt.jax.evaluation import ValueEngine, PuzzleBenchmark, load_checkpoint

    # Load a trained checkpoint
    checkpoint = load_checkpoint("checkpoints_jax/best")

    # Create an engine
    engine = ValueEngine.from_loaded_checkpoint(checkpoint)

    # Run a benchmark
    benchmark = PuzzleBenchmark("puzzles/puzzles.csv")
    result = benchmark.run(engine)
    print(result.summary())
"""

from catgpt.jax.evaluation.benchmarks import (
    Benchmark,
    BenchmarkResult,
    PuzzleBenchmark,
    PuzzleResult,
)
from catgpt.jax.evaluation.checkpoint import LoadedCheckpoint, load_checkpoint
from catgpt.jax.evaluation.engines import Engine, ValueEngine

__all__ = [
    # Checkpoint loading
    "LoadedCheckpoint",
    "load_checkpoint",
    # Engines
    "Engine",
    "ValueEngine",
    # Benchmarks
    "Benchmark",
    "BenchmarkResult",
    "PuzzleBenchmark",
    "PuzzleResult",
]
