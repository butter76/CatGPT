"""Checkpoint loading utilities for JAX evaluation.

Provides functions to load trained model checkpoints, including model parameters,
model configuration, and tokenizer configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import OmegaConf

from catgpt.jax.configs import JaxModelConfig, JaxOutputHeadConfig, JaxTokenizerConfig
from catgpt.jax.models.transformer import BidirectionalTransformer, TransformerConfig


@dataclass
class LoadedCheckpoint:
    """Container for a loaded checkpoint.

    Attributes:
        model: The BidirectionalTransformer model instance.
        params: Model parameters (pytree).
        model_config: Model configuration.
        tokenizer_config: Tokenizer configuration.
        checkpoint_path: Path to the loaded checkpoint.
    """

    model: BidirectionalTransformer
    params: dict[str, Any]
    model_config: JaxModelConfig
    tokenizer_config: JaxTokenizerConfig
    checkpoint_path: Path


def load_checkpoint(path: Path | str) -> LoadedCheckpoint:
    """Load a model checkpoint from a directory.

    Supports both Orbax and simple msgpack checkpoint formats.

    Args:
        path: Path to the checkpoint directory containing params and config files.

    Returns:
        LoadedCheckpoint containing model, params, and configs.

    Raises:
        FileNotFoundError: If checkpoint directory or required files don't exist.
        ValueError: If checkpoint format is invalid.
    """
    path = Path(path)

    if not path.exists():
        msg = f"Checkpoint directory not found: {path}"
        raise FileNotFoundError(msg)

    logger.info(f"Loading checkpoint from {path}")

    # Load model config
    model_config = _load_model_config(path)

    # Load tokenizer config
    tokenizer_config = _load_tokenizer_config(path)

    # Create model from config
    transformer_config = TransformerConfig(
        hidden_size=model_config.hidden_size,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        ff_dim=model_config.ff_dim,
        vocab_size=model_config.vocab_size,
        seq_length=tokenizer_config.sequence_length,
        activation=model_config.activation,
        dropout_rate=model_config.dropout_rate,
        output_heads=model_config.output_heads,
    )
    model = BidirectionalTransformer(config=transformer_config)

    # Load parameters
    params = _load_params(path, model, tokenizer_config.sequence_length)

    logger.info(f"Loaded checkpoint with {_count_params(params):,} parameters")

    return LoadedCheckpoint(
        model=model,
        params=params,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
        checkpoint_path=path,
    )


def _load_model_config(path: Path) -> JaxModelConfig:
    """Load model configuration from checkpoint directory."""
    # Try model_config.yaml first, then config.yaml
    config_path = path / "model_config.yaml"
    if not config_path.exists():
        config_path = path / "config.yaml"

    if not config_path.exists():
        msg = f"Model config not found in {path}"
        raise FileNotFoundError(msg)

    config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    # Handle output_heads as dict or JaxOutputHeadConfig
    output_heads_data = config_dict.get("output_heads", {})
    if isinstance(output_heads_data, dict):
        output_heads = JaxOutputHeadConfig(**output_heads_data)
    else:
        output_heads = output_heads_data

    return JaxModelConfig(
        name=config_dict.get("name", "transformer"),
        hidden_size=config_dict["hidden_size"],
        num_layers=config_dict["num_layers"],
        num_heads=config_dict["num_heads"],
        ff_dim=config_dict.get("ff_dim"),
        vocab_size=config_dict.get("vocab_size", 28),
        seq_length=config_dict.get("seq_length", 64),
        activation=config_dict.get("activation", "gelu"),
        dropout_rate=config_dict.get("dropout_rate", 0.0),
        output_heads=output_heads,
    )


def _load_tokenizer_config(path: Path) -> JaxTokenizerConfig:
    """Load tokenizer configuration from checkpoint directory."""
    config_path = path / "tokenizer_config.yaml"

    if config_path.exists():
        config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        return JaxTokenizerConfig(
            sequence_length=config_dict.get("sequence_length", 64),
            include_halfmove=config_dict.get("include_halfmove", False),
        )

    # Fall back to model config's seq_length
    model_config_path = path / "model_config.yaml"
    if model_config_path.exists():
        config_dict = OmegaConf.to_container(
            OmegaConf.load(model_config_path), resolve=True
        )
        return JaxTokenizerConfig(
            sequence_length=config_dict.get("seq_length", 64),
            include_halfmove=False,
        )

    # Default
    logger.warning("No tokenizer config found, using defaults")
    return JaxTokenizerConfig()


def _load_params(path: Path, model: BidirectionalTransformer, seq_length: int) -> dict:
    """Load model parameters from checkpoint directory.

    For SPlus optimizer checkpoints, prefers EMA params (stored separately) over
    training params. EMA params are the bias-corrected exponential moving average
    which should be used for evaluation/inference.

    Tries Orbax format first, then falls back to simple msgpack format.
    """
    # First check for SPlus EMA params (preferred for evaluation)
    # Try Orbax format
    ema_params_dir = path / "ema_params"
    if ema_params_dir.exists():
        logger.info("Loading SPlus EMA params (Orbax format)")
        return _load_params_orbax(ema_params_dir)

    # Try simple msgpack format for EMA params
    ema_msgpack_path = path / "ema_params.msgpack"
    if ema_msgpack_path.exists():
        logger.info("Loading SPlus EMA params (msgpack format)")
        return _load_params_msgpack(ema_msgpack_path, model, seq_length)

    # Fall back to regular params
    # Try Orbax format (params/ directory)
    params_dir = path / "params"
    if params_dir.exists():
        return _load_params_orbax(params_dir)

    # Try simple msgpack format
    msgpack_path = path / "params.msgpack"
    if msgpack_path.exists():
        return _load_params_msgpack(msgpack_path, model, seq_length)

    # Try flax checkpoint format
    try:
        from flax.training import checkpoints

        params = checkpoints.restore_checkpoint(path, target=None, prefix="params_")
        if params is not None:
            return params
    except Exception:
        pass

    msg = f"No valid parameter checkpoint found in {path}"
    raise FileNotFoundError(msg)


def _load_params_orbax(params_dir: Path) -> dict:
    """Load parameters using Orbax checkpointer."""
    try:
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        params = checkpointer.restore(params_dir.resolve())
        return params
    except ImportError:
        msg = "Orbax checkpoint found but orbax-checkpoint is not installed"
        raise ImportError(msg)


def _load_params_msgpack(
    msgpack_path: Path, model: BidirectionalTransformer, seq_length: int
) -> dict:
    """Load parameters from msgpack file."""
    from flax.serialization import from_bytes

    # Need to initialize model to get parameter structure
    dummy_input = jnp.zeros((1, seq_length), dtype=jnp.int32)
    rng = jax.random.key(0)
    target_params = model.init(rng, dummy_input, train=False)

    with msgpack_path.open("rb") as f:
        params = from_bytes(target_params, f.read())

    return params


def _count_params(params: dict) -> int:
    """Count the number of parameters in a pytree."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))
