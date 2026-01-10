"""Base model class for JAX/Flax models."""

from abc import ABC, abstractmethod
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


class BaseModel(nn.Module, ABC):
    """Abstract base class for all JAX/Flax models in CatGPT.

    All models should inherit from this class and implement
    the required abstract methods.

    Note: Flax modules use setup() instead of __init__ for defining
    submodules, and __call__ instead of forward.
    """

    @abstractmethod
    def __call__(self, x: Array, train: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            x: Input array.
            train: Whether in training mode (affects dropout, batchnorm, etc.)

        Returns:
            Output array.
        """
        ...

    def num_parameters(self, params: dict[str, Any]) -> int:
        """Count the number of parameters in the model.

        Args:
            params: The parameter dictionary from model.init().

        Returns:
            Number of parameters.
        """
        return sum(p.size for p in jnp.tree_util.tree_leaves(params))
