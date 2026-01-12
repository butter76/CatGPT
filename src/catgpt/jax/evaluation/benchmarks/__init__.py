"""Evaluation benchmarks for chess engines."""

from catgpt.jax.evaluation.benchmarks.base import Benchmark, BenchmarkResult
from catgpt.jax.evaluation.benchmarks.puzzles import PuzzleBenchmark, PuzzleResult

__all__ = ["Benchmark", "BenchmarkResult", "PuzzleBenchmark", "PuzzleResult"]
