"""Common data types used across frameworks."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """A single data sample (framework-agnostic).

    Framework-specific datasets should convert this to tensors.
    """

    inputs: Any
    targets: Any
    metadata: dict[str, Any] | None = None


@dataclass
class Batch:
    """A batch of samples (framework-agnostic).

    Framework-specific data loaders should convert this to tensor batches.
    """

    inputs: Any
    targets: Any
    metadata: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        """Return batch size."""
        if hasattr(self.inputs, "__len__"):
            return len(self.inputs)
        return 1
