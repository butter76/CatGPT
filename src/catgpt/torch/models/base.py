"""Base model class for PyTorch models in CatGPT."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all PyTorch models in CatGPT.

    All models should inherit from this class and implement the required
    abstract methods. Provides standardized save/load functionality that
    preserves model configuration alongside weights.
    """

    # Subclasses should set this to their config class
    config_class: type = None  # type: ignore[assignment]

    def __init__(self, config: Any) -> None:
        """Initialize the base model.

        Args:
            config: Model configuration (dataclass or dict-like).
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of token indices, shape (batch, seq_len).

        Returns:
            Output tensor (model-specific shape).
        """
        ...

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: Path | str) -> None:
        """Save model weights and configuration to a directory.

        Creates a directory with:
        - model.pt: Model state dict
        - config.yaml: Model configuration

        Args:
            path: Directory to save to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), path / "model.pt")

        # Save config as YAML
        config_dict = self._config_to_dict()
        OmegaConf.save(OmegaConf.create(config_dict), path / "config.yaml")

    @classmethod
    def load_pretrained(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> Self:
        """Load a model from a pretrained checkpoint directory.

        Supports two checkpoint formats:
        - Standalone: config.yaml + model.pt (from save_pretrained)
        - Trainer: model_config.yaml + model.pt (from Trainer.save_checkpoint)

        Args:
            path: Directory containing model.pt and config.yaml/model_config.yaml.
            device: Device to load the model onto.
            **kwargs: Additional arguments to override config values.

        Returns:
            Loaded model instance.
        """
        path = Path(path)

        # Load config - prefer model_config.yaml (model-specific) over config.yaml (full experiment)
        config_path = path / "model_config.yaml"
        if not config_path.exists():
            config_path = path / "config.yaml"
        if not config_path.exists():
            msg = f"Config file not found in {path} (tried model_config.yaml and config.yaml)"
            raise FileNotFoundError(msg)

        config_dict = OmegaConf.to_container(OmegaConf.load(config_path))
        config_dict.update(kwargs)  # type: ignore[union-attr]

        # Create model with config, filtering unknown keys if needed
        if cls.config_class is not None:
            # Get valid field names from the config class
            import dataclasses
            if dataclasses.is_dataclass(cls.config_class):
                valid_fields = {f.name for f in dataclasses.fields(cls.config_class)}
                filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}  # type: ignore[union-attr]
                config = cls.config_class(**filtered_dict)
            else:
                config = cls.config_class(**config_dict)  # type: ignore[arg-type]
        else:
            config = config_dict

        model = cls(config)

        # Load weights
        weights_path = path / "model.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

        return model.to(device)

    def _config_to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict, is_dataclass

        if is_dataclass(self.config):
            return asdict(self.config)
        if hasattr(self.config, "__dict__"):
            return dict(self.config.__dict__)
        return dict(self.config)
