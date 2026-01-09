"""Base model class for all CatGPT models."""

from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in CatGPT.

    All models should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        ...

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the number of parameters in the model.

        Args:
            trainable_only: If True, count only trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def compile_model(self, **kwargs: object) -> Self:
        """Compile the model using torch.compile for optimization.

        Note: Named compile_model to avoid conflict with nn.Module.compile().

        Args:
            **kwargs: Arguments passed to torch.compile.

        Returns:
            Compiled model (same type as self).
        """
        return torch.compile(self, **kwargs)  # type: ignore[return-value]
