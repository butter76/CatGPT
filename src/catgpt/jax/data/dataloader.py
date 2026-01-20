"""DataLoader utilities for JAX training using PyGrain."""

from typing import TYPE_CHECKING

import grain.python as pygrain
import numpy as np
from scipy import special as scipy_special

from catgpt.core.data.grain.bagz import BagDataSource
from catgpt.core.data.grain.coders import ConvertTrainingBagDataToSequence

if TYPE_CHECKING:
    from catgpt.jax.configs import (
        JaxDataConfig,
        JaxOutputHeadConfig,
        JaxTokenizerConfig,
    )


def _hl_gauss_transform_numpy(
    target: np.ndarray,
    num_bins: int = 81,
    sigma_ratio: float = 0.75,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """Convert scalar target(s) to HL-Gauss probability distribution (numpy version).

    This is a numpy implementation for use in data loading workers.

    Args:
        target: Scalar target value(s). Shape: (batch,) or (batch, 1)
        num_bins: Number of bins for the categorical distribution.
        sigma_ratio: Ratio of sigma to bin_width. sigma = sigma_ratio * bin_width.
        min_value: Minimum value of the target range.
        max_value: Maximum value of the target range.

    Returns:
        Probability distribution over bins. Shape: (batch, num_bins)
    """
    # Ensure target is 1D
    target = np.asarray(target, dtype=np.float32).squeeze()
    if target.ndim == 0:
        target = target[np.newaxis]

    # Bin edges (num_bins + 1 values)
    support = np.linspace(min_value, max_value, num_bins + 1, dtype=np.float32)

    # Compute sigma from bin width and ratio
    bin_width = (max_value - min_value) / num_bins
    sigma = sigma_ratio * bin_width

    # Expand target for broadcasting: (batch,) -> (batch, 1)
    target_expanded = target[:, np.newaxis]

    # Compute CDF at each bin edge using the error function
    cdf_evals = scipy_special.erf(
        (support - target_expanded) / (np.sqrt(2.0) * sigma)
    )

    # Probability in each bin = CDF(right) - CDF(left)
    bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1]

    # Normalize to ensure probabilities sum to 1
    z = cdf_evals[:, -1] - cdf_evals[:, 0]
    bin_probs = bin_probs / z[:, np.newaxis]

    return bin_probs.astype(np.float32)


class ConvertToJax(pygrain.MapTransform):
    """Convert numpy arrays to JAX-compatible arrays with HL-Gauss target transform.

    Transforms batched data from PyGrain (numpy) to JAX-compatible format.
    Expected input: tuple of (inputs, targets, [policy_targets], [next_capture], [next_pawn]).

    If HL-Gauss is enabled, converts scalar win probabilities to categorical
    distributions over bins for training with cross-entropy loss.
    """

    def __init__(
        self,
        output_heads_config: "JaxOutputHeadConfig | None" = None,
    ):
        """Initialize the transform.

        Args:
            output_heads_config: Output head configuration containing HL-Gauss
                parameters (value_num_bins, value_sigma_ratio) and head flags.
                If None, uses defaults.
        """
        super().__init__()
        self._output_heads_config = output_heads_config

        # Extract HL-Gauss params (use defaults if config not provided)
        if output_heads_config is not None:
            self._num_bins = output_heads_config.value_num_bins
            self._sigma_ratio = output_heads_config.value_sigma_ratio
            self._include_policy = output_heads_config.policy_head
            self._include_next_capture = output_heads_config.next_capture_head
            self._include_next_pawn_move = output_heads_config.next_pawn_move_head
        else:
            self._num_bins = 81
            self._sigma_ratio = 0.75
            self._include_policy = False
            self._include_next_capture = False
            self._include_next_pawn_move = False

    def map(self, element):
        """Convert a batched element to JAX-compatible arrays.

        Args:
            element: Tuple of (inputs, targets, [policy_targets], [next_capture], [next_pawn])
                as numpy arrays. Optional elements depend on head configuration.

        Returns:
            Dictionary with 'input', 'target', and optional targets as numpy arrays.
            - 'input': Token indices, shape (batch, seq_len)
            - 'target': HL-Gauss distribution, shape (batch, num_bins)
            - 'policy_target': Flattened policy distribution, shape (batch, 64*73)
            - 'next_capture_target': Square indices, shape (batch,), -1 = invalid
            - 'next_pawn_move_target': Square indices, shape (batch,), -1 = invalid
        """
        # Unpack tuple based on configuration
        element = list(element)
        inputs = element.pop(0)
        targets = element.pop(0)

        policy_targets = None
        next_capture_targets = None
        next_pawn_move_targets = None

        if self._include_policy:
            policy_targets = element.pop(0)
        if self._include_next_capture:
            next_capture_targets = element.pop(0)
        if self._include_next_pawn_move:
            next_pawn_move_targets = element.pop(0)

        # Keep as numpy - JAX will convert when needed
        # This avoids unnecessary device transfers during data loading
        input_array = np.asarray(inputs, dtype=np.int32)  # Token indices

        # Transform scalar win probabilities to HL-Gauss distribution
        target_array = _hl_gauss_transform_numpy(
            targets,
            num_bins=self._num_bins,
            sigma_ratio=self._sigma_ratio,
            min_value=0.0,
            max_value=1.0,
        )

        result = {
            "input": input_array,
            "target": target_array,  # Shape: (batch, num_bins)
        }

        # Add policy target if enabled
        if self._include_policy and policy_targets is not None:
            # policy_targets: (batch, 64, 73) -> flatten to (batch, 64*73)
            policy_array = np.asarray(policy_targets, dtype=np.float32)
            policy_array = policy_array.reshape(policy_array.shape[0], -1)
            result["policy_target"] = policy_array

        # Add next capture target if enabled
        if self._include_next_capture and next_capture_targets is not None:
            # next_capture_targets: (batch,) int in [-1, 63]
            result["next_capture_target"] = np.asarray(
                next_capture_targets, dtype=np.int32
            )

        # Add next pawn move target if enabled
        if self._include_next_pawn_move and next_pawn_move_targets is not None:
            # next_pawn_move_targets: (batch,) int in [-1, 63]
            result["next_pawn_move_target"] = np.asarray(
                next_pawn_move_targets, dtype=np.int32
            )

        return result


def create_grain_dataloader(
    config: "JaxDataConfig",
    *,
    split: str = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    num_devices: int = 1,
    device_index: int = 0,
    tokenizer_config: "JaxTokenizerConfig | None" = None,
    output_heads_config: "JaxOutputHeadConfig | None" = None,
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
        output_heads_config: Output head configuration (for HL-Gauss transform).

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

    # Determine which heads are enabled
    include_policy = (
        output_heads_config is not None and output_heads_config.policy_head
    )
    include_next_capture = (
        output_heads_config is not None and output_heads_config.next_capture_head
    )
    include_next_pawn_move = (
        output_heads_config is not None and output_heads_config.next_pawn_move_head
    )

    # Define transformations pipeline
    transformations = (
        ConvertTrainingBagDataToSequence(
            core_tokenizer_config,
            include_policy=include_policy,
            include_next_capture=include_next_capture,
            include_next_pawn_move=include_next_pawn_move,
        ),  # bytes -> (state, win_prob, [policy], [next_capture], [next_pawn])
        pygrain.Batch(batch_size=batch_size, drop_remainder=split == "train"),
        ConvertToJax(output_heads_config),  # numpy -> jax-compatible arrays with HL-Gauss
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

    # Log configuration
    num_bins = output_heads_config.value_num_bins if output_heads_config else 81
    sigma_ratio = output_heads_config.value_sigma_ratio if output_heads_config else 0.75
    logger.info(
        f"Created JAX PyGrain DataLoader: batch_size={batch_size}, "
        f"shuffle={shuffle}, workers={config.num_workers}, "
        f"hl_gauss(bins={num_bins}, sigma_ratio={sigma_ratio}), "
        f"policy={include_policy}, next_capture={include_next_capture}, "
        f"next_pawn_move={include_next_pawn_move}"
    )

    return dataloader


def create_dataloader(
    config: "JaxDataConfig",
    *,
    split: str = "train",
    batch_size: int = 64,
    num_devices: int = 1,
    device_index: int = 0,
    tokenizer_config: "JaxTokenizerConfig | None" = None,
    output_heads_config: "JaxOutputHeadConfig | None" = None,
) -> pygrain.DataLoader:
    """Create a DataLoader for JAX training or validation.

    Args:
        config: Data configuration (includes seed).
        split: Data split ("train" or "val").
        batch_size: Batch size per device.
        num_devices: Number of devices (for sharded loading).
        device_index: Current device index.
        tokenizer_config: Tokenizer configuration (for PyGrain).
        output_heads_config: Output head configuration (for HL-Gauss transform).

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
        output_heads_config=output_heads_config,
    )
