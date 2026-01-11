"""DataLoader utilities for JAX training using PyGrain."""

from typing import TYPE_CHECKING

import grain.python as pygrain
import jax
import jax.numpy as jnp
import numpy as np

from catgpt.core.data.grain.bagz import BagDataSource
from catgpt.core.data.grain.coders import ConvertStateValueDataToSequence

if TYPE_CHECKING:
    from catgpt.jax.configs import JaxDataConfig, JaxTokenizerConfig


class ConvertToJax(pygrain.MapTransform):
    """Convert numpy arrays to JAX arrays.

    Transforms batched data from PyGrain (numpy) to JAX-compatible format.
    Expected input: tuple of (inputs, targets) where both are numpy arrays.
    """

    def map(self, element):
        """Convert a batched element to JAX-compatible arrays.

        Args:
            element: Tuple of (inputs, targets) as numpy arrays.

        Returns:
            Dictionary with 'input' and 'target' as numpy arrays
            (JAX will convert them to device arrays when needed).
        """
        inputs, targets = element

        # Keep as numpy - JAX will convert when needed
        # This avoids unnecessary device transfers during data loading
        input_array = np.asarray(inputs, dtype=np.int32)  # Token indices
        target_array = np.asarray(targets, dtype=np.float32)  # Win probabilities

        # Squeeze target if it has unnecessary dimensions
        if target_array.ndim > 1 and target_array.shape[-1] == 1:
            target_array = target_array.squeeze(-1)

        return {
            "input": input_array,
            "target": target_array,
        }


def create_grain_dataloader(
    config: "JaxDataConfig",
    *,
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    num_devices: int = 1,
    device_index: int = 0,
    tokenizer_config: "JaxTokenizerConfig | None" = None,
) -> pygrain.DataLoader:
    """Create a PyGrain DataLoader for chess position data (JAX).

    Args:
        config: Data configuration (includes seed).
        split: Data split ("train" or "val").
        batch_size: Batch size per device.
        shuffle: Whether to shuffle the data.
        num_devices: Total number of devices (for sharded loading).
        device_index: Current device index (for sharded loading).
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

    # Create sampler with sharding for multi-device training
    if num_devices > 1:
        shard_options = pygrain.ShardByJaxProcess(
            shard_count=num_devices, shard_index=device_index
        )
    else:
        shard_options = pygrain.NoSharding()

    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=shard_options,
        shuffle=shuffle,
        seed=config.seed,
        num_epochs=None,  # Infinite epochs for step-based training
    )

    # Convert tokenizer config if needed
    core_tokenizer_config = None
    if tokenizer_config is not None:
        from catgpt.core.utils.tokenizer import TokenizerConfig as CoreTokenizerConfig

        core_tokenizer_config = CoreTokenizerConfig(
            sequence_length=tokenizer_config.sequence_length,
            include_halfmove=tokenizer_config.include_halfmove,
        )

    # Define transformations pipeline
    transformations = (
        ConvertStateValueDataToSequence(core_tokenizer_config),  # bytes -> (state, win_prob)
        pygrain.Batch(batch_size=batch_size, drop_remainder=split == "train"),
        ConvertToJax(),  # numpy -> jax-compatible arrays
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
        f"Created JAX PyGrain DataLoader: batch_size={batch_size}, "
        f"shuffle={shuffle}, workers={config.num_workers}"
    )

    return dataloader


class PlaceholderDataset:
    """Placeholder dataset for testing the JAX training pipeline.

    Generates random data with the expected format.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 64,
        vocab_size: int = 28,
        seed: int = 42,
    ) -> None:
        """Initialize the placeholder dataset.

        Args:
            num_samples: Number of samples in the dataset.
            seq_length: Sequence length for inputs.
            vocab_size: Size of the token vocabulary.
            seed: Random seed for reproducibility.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        # Generate deterministic random tokens based on index
        rng = np.random.default_rng(self.rng.integers(0, 2**31) + idx)
        input_tokens = rng.integers(0, self.vocab_size, (self.seq_length,), dtype=np.int32)
        target = np.float32(idx % 2)

        return {
            "input": input_tokens,
            "target": target,
        }


class PlaceholderDataLoader:
    """Simple iterable dataloader for placeholder dataset."""

    def __init__(
        self,
        dataset: PlaceholderDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self._rng = np.random.default_rng(42)

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self._rng.shuffle(indices)

        batch_inputs = []
        batch_targets = []

        for idx in indices:
            sample = self.dataset[idx]
            batch_inputs.append(sample["input"])
            batch_targets.append(sample["target"])

            if len(batch_inputs) == self.batch_size:
                yield {
                    "input": np.stack(batch_inputs),
                    "target": np.array(batch_targets),
                }
                batch_inputs = []
                batch_targets = []

        # Handle last batch
        if batch_inputs and not self.drop_last:
            yield {
                "input": np.stack(batch_inputs),
                "target": np.array(batch_targets),
            }


def create_dataloader(
    config: "JaxDataConfig",
    *,
    split: str = "train",
    batch_size: int = 64,
    num_devices: int = 1,
    device_index: int = 0,
    tokenizer_config: "JaxTokenizerConfig | None" = None,
) -> pygrain.DataLoader | PlaceholderDataLoader:
    """Create a DataLoader for JAX training or validation.

    This is a compatibility wrapper that can use either PyGrain (real data)
    or the placeholder dataset (for testing).

    Args:
        config: Data configuration (includes seed).
        split: Data split ("train" or "val").
        batch_size: Batch size per device.
        num_devices: Number of devices (for sharded loading).
        device_index: Current device index.
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
        num_devices=num_devices,
        device_index=device_index,
        tokenizer_config=tokenizer_config,
    )
