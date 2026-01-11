"""JAX/Optax optimizer definitions."""

from catgpt.jax.optimizers.factory import (
    create_optimizer,
    create_optimizer_with_gradient_clipping,
)
from catgpt.jax.optimizers.splus import (
    SPlusState,
    splus,
    splus_get_eval_params,
)

__all__ = [
    "SPlusState",
    "create_optimizer",
    "create_optimizer_with_gradient_clipping",
    "splus",
    "splus_get_eval_params",
]
