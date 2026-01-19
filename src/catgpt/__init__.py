"""CatGPT: ML research project for chess and beyond.

This package uses JAX/Flax as its ML framework:
- JAX: `from catgpt.jax import ...`

Shared utilities are in `catgpt.core`:
- `from catgpt.core import setup_logging, load_config`
- `from catgpt.core.chess import ChessEngine`
"""

__version__ = "0.1.0"

# Re-export common utilities for convenience
from catgpt.core import load_config, save_config, setup_logging
from catgpt.core.chess import ChessEngine

__all__ = [
    "ChessEngine",
    "__version__",
    "load_config",
    "save_config",
    "setup_logging",
]
