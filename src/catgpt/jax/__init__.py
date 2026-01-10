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

from catgpt.jax.models import BaseModel

__all__ = ["BaseModel"]
