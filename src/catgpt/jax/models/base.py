"""Base model class for JAX/Flax models in CatGPT."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Self

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
from omegaconf import OmegaConf


class BaseModel(nn.Module):
    """Abstract base class for all JAX/Flax models in CatGPT.

    All models should inherit from this class and implement the required
    abstract methods. Provides standardized save/load functionality.
    """

    # Subclasses should set this to their config class
    config_class: type = None  # type: ignore[assignment]

    @abstractmethod
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Forward pass of the model.

        Args:
            x: Input tensor of token indices, shape (batch, seq_len).
            train: Whether in training mode (affects dropout, etc.).

        Returns:
            Output tensor (model-specific shape).
        """
        ...

    @staticmethod
    def num_parameters(params: dict) -> int:
        """Count the number of parameters in the model.

        Args:
            params: Model parameters (pytree).

        Returns:
            Number of parameters.
        """
        return sum(p.size for p in jax.tree_util.tree_leaves(params))

    @classmethod
    def save_pretrained(
        cls,
        path: Path | str,
        params: dict,
        config: Any,
    ) -> None:
        """Save model weights and configuration to a directory.

        Creates a directory with:
        - params.msgpack: Model parameters
        - config.yaml: Model configuration

        Args:
            path: Directory to save to.
            params: Model parameters.
            config: Model configuration.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights using Flax checkpointing
        checkpoints.save_checkpoint(
            path,
            params,
            step=0,
            prefix="params_",
            overwrite=True,
        )

        # Save config as YAML
        config_dict = cls._config_to_dict(config)
        OmegaConf.save(OmegaConf.create(config_dict), path / "config.yaml")

    @classmethod
    def load_pretrained(
        cls,
        path: Path | str,
        **kwargs: Any,
    ) -> tuple[Self, dict]:
        """Load a model from a pretrained checkpoint directory.

        Args:
            path: Directory containing params and config.yaml.
            **kwargs: Additional arguments to override config values.

        Returns:
            Tuple of (model instance, parameters).
        """
        path = Path(path)

        # Load config
        config_path = path / "model_config.yaml"
        if not config_path.exists():
            config_path = path / "config.yaml"
        if not config_path.exists():
            msg = f"Config file not found in {path}"
            raise FileNotFoundError(msg)

        config_dict = OmegaConf.to_container(OmegaConf.load(config_path))
        config_dict.update(kwargs)  # type: ignore[union-attr]

        # Create model with config
        if cls.config_class is not None:
            import dataclasses

            if dataclasses.is_dataclass(cls.config_class):
                valid_fields = {f.name for f in dataclasses.fields(cls.config_class)}
                filtered_dict = {
                    k: v for k, v in config_dict.items() if k in valid_fields  # type: ignore[union-attr]
                }
                config = cls.config_class(**filtered_dict)
            else:
                config = cls.config_class(**config_dict)  # type: ignore[arg-type]
        else:
            config = config_dict

        model = cls(config)

        # Load parameters
        params = checkpoints.restore_checkpoint(path, target=None, prefix="params_")
        if params is None:
            # Try loading from model.msgpack (alternative format)
            import msgpack

            msgpack_path = path / "model.msgpack"
            if msgpack_path.exists():
                with msgpack_path.open("rb") as f:
                    params = msgpack.unpack(f, raw=False)
            else:
                raise FileNotFoundError(f"No checkpoint found in {path}")

        return model, params

    @staticmethod
    def _config_to_dict(config: Any) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict, is_dataclass

        if is_dataclass(config):
            return asdict(config)
        if hasattr(config, "__dict__"):
            return dict(config.__dict__)
        return dict(config)
