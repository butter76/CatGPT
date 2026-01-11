"""JAX implementations for CatGPT.

This module requires the 'jax' optional dependency:
    pip install catgpt[jax]
"""

try:
    import jax  # noqa: F401
except ImportError as e:
    msg = (
        "JAX is not installed. Install it with:\n"
        "  pip install catgpt[jax]\n"
        "or:\n"
        "  uv sync --extra jax"
    )
    raise ImportError(msg) from e

from catgpt.jax.configs import (
    JaxCheckpointConfig,
    JaxDataConfig,
    JaxDistributedConfig,
    JaxExperimentConfig,
    JaxModelConfig,
    JaxOptimizerConfig,
    JaxSchedulerConfig,
    JaxTokenizerConfig,
    JaxTrainingConfig,
    JaxWandbConfig,
    jax_config_from_dict,
    jax_config_to_dict,
)
from catgpt.jax.data import create_dataloader, create_grain_dataloader
from catgpt.jax.models import BaseModel, BidirectionalTransformer, TransformerConfig
from catgpt.jax.optimizers import create_optimizer, splus
from catgpt.jax.training import TrainState, Trainer

__all__ = [
    # Configs
    "JaxCheckpointConfig",
    "JaxDataConfig",
    "JaxDistributedConfig",
    "JaxExperimentConfig",
    "JaxModelConfig",
    "JaxOptimizerConfig",
    "JaxSchedulerConfig",
    "JaxTokenizerConfig",
    "JaxTrainingConfig",
    "JaxWandbConfig",
    "jax_config_from_dict",
    "jax_config_to_dict",
    # Models
    "BaseModel",
    "BidirectionalTransformer",
    "TransformerConfig",
    # Data
    "create_dataloader",
    "create_grain_dataloader",
    # Optimizers
    "create_optimizer",
    "splus",
    # Training
    "TrainState",
    "Trainer",
]
