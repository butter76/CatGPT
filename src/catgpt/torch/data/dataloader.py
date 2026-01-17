"""DataLoader utilities for PyTorch training using PyGrain."""

from typing import TYPE_CHECKING

import grain.python as pygrain
import torch

from catgpt.core.data.grain.bagz import BagDataSource
from catgpt.core.data.grain.coders import ConvertTrainingBagDataToSequence

if TYPE_CHECKING:
    from catgpt.core.configs.schema import DataConfig
    from catgpt.core.utils import TokenizerConfig


class ConvertToTorch(pygrain.MapTransform):
    """Convert numpy arrays to PyTorch tensors.

    Transforms batched data from PyGrain (numpy) to PyTorch tensors.
    Expected input: tuple of (inputs, targets) where both are numpy arrays.
    """

    def map(self, element):
        """Convert a batched element to PyTorch tensors.

        Args:
            element: Tuple of (inputs, targets) as numpy arrays.

        Returns:
            Dictionary with 'input' and 'target' as PyTorch tensors.
        """
        inputs, targets = element

        # Convert to tensors
        input_tensor = torch.from_numpy(inputs).long()  # Token indices
        target_tensor = torch.from_numpy(targets).float()  # Win probabilities

        # Squeeze target if it has unnecessary dimensions
        if target_tensor.dim() > 1 and target_tensor.shape[-1] == 1:
            target_tensor = target_tensor.squeeze(-1)

        return {
            "input": input_tensor,
            "target": target_tensor,
        }


def create_grain_dataloader(
    config: "DataConfig",
    *,
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    world_size: int = 1,
    rank: int = 0,
    tokenizer_config: "TokenizerConfig | None" = None,
) -> pygrain.DataLoader:
    """Create a PyGrain DataLoader for chess position data.

    Args:
        config: Data configuration (includes seed).
        split: Data split ("train" or "val").
        batch_size: Batch size per worker.
        shuffle: Whether to shuffle the data.
        world_size: Total number of processes (for distributed training).
        rank: Current process rank (for distributed training).
        tokenizer_config: Tokenizer configuration (for proper sequence encoding).

    Returns:
        PyGrain DataLoader yielding batches.
    """
    # Determine data path
    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
        if data_path is None:
            raise ValueError("Validation path not configured")
    else:
        raise ValueError(f"Unknown split: {split}")

    # Create bag data source
    from loguru import logger

    bag_source = BagDataSource(data_path)
    logger.info(f"Loaded {split} data from {data_path}: {len(bag_source)} samples")

    # Create sampler with sharding for distributed training
    if world_size > 1:
        shard_options = pygrain.ShardByJaxProcess(shard_count=world_size, shard_index=rank)
    else:
        shard_options = pygrain.NoSharding()

    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=shard_options,
        shuffle=shuffle,
        seed=config.seed,  # Use seed from config
        num_epochs=None,  # Infinite epochs for step-based training
    )

    # Define transformations pipeline
    transformations = (
        ConvertTrainingBagDataToSequence(tokenizer_config),  # bytes -> (state, win_prob)
        pygrain.Batch(batch_size=batch_size, drop_remainder=split == "train"),
        ConvertToTorch(),  # numpy -> torch tensors dict
    )

    # Create and return dataloader
    dataloader = pygrain.DataLoader(
        data_source=bag_source,
        sampler=sampler,
        operations=transformations,
        worker_count=config.num_workers,
        read_options=pygrain.ReadOptions(
            prefetch_buffer_size=config.prefetch_factor,
        ),
    )

    logger.info(
        f"Created PyGrain DataLoader: batch_size={batch_size}, "
        f"shuffle={shuffle}, workers={config.num_workers}, "
        f"tokenizer_config={tokenizer_config}"
    )

    return dataloader


class PlaceholderDataset:
    """Placeholder dataset for testing the training pipeline.

    Generates random data with the expected format.
    This is kept for backward compatibility and testing.
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
    batch_size: int = 64,
    world_size: int = 1,
    rank: int = 0,
    tokenizer_config: "TokenizerConfig | None" = None,
) -> pygrain.DataLoader | torch.utils.data.DataLoader:
    """Create a DataLoader for training or validation.

    This is a compatibility wrapper that can use either PyGrain (real data)
    or the placeholder dataset (for testing).

    Args:
        config: Data configuration (includes seed).
        split: Data split ("train" or "val").
        seq_length: Sequence length for inputs (used for placeholder only).
        vocab_size: Vocabulary size (used for placeholder only).
        batch_size: Batch size per GPU.
        world_size: Number of GPUs (for distributed sampler).
        rank: Current GPU rank.
        tokenizer_config: Tokenizer configuration (for PyGrain).

    Returns:
        DataLoader instance.
    """
    shuffle = split == "train"
    return create_grain_dataloader(
        config,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        tokenizer_config=tokenizer_config,
    )
