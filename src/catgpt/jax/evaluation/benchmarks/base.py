"""Base classes for evaluation benchmarks."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from catgpt.jax.evaluation.engines.base import Engine


@dataclass
class BenchmarkResult:
    """Results from running a benchmark.

    Attributes:
        name: Name of the benchmark.
        metrics: Dictionary of metric name -> value.
        details: Optional list of per-item results for detailed analysis.
        metadata: Optional additional metadata about the run.
    """

    name: str
    metrics: dict[str, float]
    details: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary of results."""
        lines = [f"Benchmark: {self.name}"]
        for key, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                if "accuracy" in key or "rate" in key:
                    lines.append(f"  {key}: {value:.2%}")
                else:
                    lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class Benchmark(Protocol):
    """Protocol for evaluation benchmarks.

    Benchmarks evaluate chess engines on specific tasks and return
    structured results with metrics.
    """

    @property
    def name(self) -> str:
        """Return the name of this benchmark."""
        ...

    def run(self, engine: Engine) -> BenchmarkResult:
        """Run the benchmark with the given engine.

        Args:
            engine: Chess engine to evaluate.

        Returns:
            BenchmarkResult containing metrics and optional details.
        """
        ...
