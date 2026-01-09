"""Base model class for all CatGPT models."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models in CatGPT.

    All models should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self._is_compiled = False

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

    def compile(self, **kwargs) -> "BaseModel":
        """Compile the model using torch.compile for optimization.

        Args:
            **kwargs: Arguments passed to torch.compile.

        Returns:
            Compiled model.
        """
        if self._is_compiled:
            return self
        compiled = torch.compile(self, **kwargs)
        compiled._is_compiled = True
        return compiled
