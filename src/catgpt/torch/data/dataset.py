"""PyTorch dataset implementations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for PyTorch datasets in CatGPT."""

    def __init__(self, root: str | Path, split: str = "train") -> None:
        """Initialize the dataset.

        Args:
            root: Root directory containing the data.
            split: Data split to use ('train', 'val', 'test').
        """
        self.root = Path(root)
        self.split = split
        self._validate_split()

    def _validate_split(self) -> None:
        """Validate that the split is valid."""
        valid_splits = {"train", "val", "test"}
        if self.split not in valid_splits:
            msg = f"Invalid split '{self.split}'. Must be one of {valid_splits}"
            raise ValueError(msg)

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            The sample at the given index.
        """
        ...
