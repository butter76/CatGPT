"""DataLoader utilities for PyTorch training.

This module provides a placeholder for PyGrain-based data loading.
The actual implementation will be added later.
"""

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from catgpt.core.configs.schema import DataConfig


class PlaceholderDataset(Dataset):
    """Placeholder dataset for testing the training pipeline.

    Generates random data with the expected format:
    - input: Token indices (batch, seq_len)
    - target: Win probability (batch,)
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 64,
        vocab_size: int = 28,
    ) -> None:
        """Initialize the placeholder dataset.

        Args:
            num_samples: Number of samples in the dataset.
            seq_length: Sequence length for inputs.
            vocab_size: Size of the token vocabulary.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Generate random tokens
        input_tokens = torch.randint(0, self.vocab_size, (self.seq_length,))

        # Generate random win probability (0 or 1 for simplicity)
        target = torch.tensor(idx % 2, dtype=torch.float32)

        return {
            "input": input_tokens,
            "target": target,
        }


def create_dataloader(
    config: "DataConfig",
    *,
    split: str = "train",
    seq_length: int = 64,
    vocab_size: int = 28,
    batch_size: int = 64,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Create a DataLoader for training or validation.

    NOTE: This is a placeholder implementation that returns random data.
    The actual PyGrain-based implementation will be added later.

    Args:
        config: Data configuration.
        split: Data split ("train" or "val").
        seq_length: Sequence length for inputs.
        vocab_size: Vocabulary size.
        batch_size: Batch size per GPU.
        world_size: Number of GPUs (for distributed sampler).
        rank: Current GPU rank.

    Returns:
        DataLoader instance.
    """
    # TODO: Replace with PyGrain-based data loading
    # from catgpt.core.data.grain.bagz import BagDataSource
    # data_path = config.train_path if split == "train" else config.val_path

    # For now, use placeholder dataset
    num_samples = 10000 if split == "train" else 1000
    dataset = PlaceholderDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        vocab_size=vocab_size,
    )

    # Create sampler for distributed training
    sampler = None
    shuffle = split == "train"

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        drop_last=split == "train",  # Drop incomplete batches during training
    )


# Placeholder for future PyGrain integration
class GrainDataLoader:
    """PyGrain-based DataLoader (to be implemented).

    This will wrap the BagDataSource from catgpt.core.data.grain.bagz
    to provide efficient, multi-worker data loading with:
    - Automatic sharding across workers
    - Prefetching and parallel decoding
    - Deterministic shuffling for reproducibility
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        num_workers: int = 4,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        """Initialize the PyGrain DataLoader.

        Args:
            data_path: Path to .bag file(s) (supports glob patterns).
            batch_size: Batch size per GPU.
            shuffle: Whether to shuffle data.
            num_workers: Number of worker processes.
            world_size: Number of GPUs.
            rank: Current GPU rank.
        """
        raise NotImplementedError(
            "GrainDataLoader is not yet implemented. "
            "Use create_dataloader() with PlaceholderDataset for now."
        )
