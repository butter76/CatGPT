"""Shared utilities for CatGPT."""

from catgpt.core.utils.logging import setup_logging
from catgpt.core.utils.squares import (
    FILES,
    RANKS,
    flip_square,
    index_to_square,
    parse_square,
)
from catgpt.core.utils.tokenizer import VOCAB_SIZE, TokenizerConfig, tokenize


def __getattr__(name: str):
    """Lazy import for torch-dependent utilities."""
    _distributed_names = {
        "all_reduce_mean",
        "barrier",
        "cleanup_distributed",
        "get_device",
        "get_local_rank",
        "get_rank",
        "get_world_size",
        "is_distributed",
        "is_main_process",
        "setup_distributed",
    }

    if name in _distributed_names:
        from catgpt.core.utils import distributed

        return getattr(distributed, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FILES",
    "RANKS",
    "VOCAB_SIZE",
    "TokenizerConfig",
    "all_reduce_mean",
    "barrier",
    "cleanup_distributed",
    "flip_square",
    "get_device",
    "get_local_rank",
    "get_rank",
    "get_world_size",
    "index_to_square",
    "is_distributed",
    "is_main_process",
    "parse_square",
    "setup_distributed",
    "setup_logging",
    "tokenize",
]
