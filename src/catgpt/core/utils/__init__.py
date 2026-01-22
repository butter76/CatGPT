"""Shared utilities for CatGPT."""

from catgpt.core.utils import tokenizer
from catgpt.core.utils.logging import setup_logging
from catgpt.core.utils.policy import (
    POLICY_SHAPE,
    POLICY_TO_DIM,
    encode_move_to_policy_index,
    encode_policy_target,
    parse_uci_move,
)
from catgpt.core.utils.squares import (
    FILES,
    RANKS,
    flip_square,
    index_to_square,
    parse_square,
)
from catgpt.core.utils.tokenizer import VOCAB_SIZE, TokenizerConfig, tokenize

__all__ = [
    "FILES",
    "POLICY_SHAPE",
    "POLICY_TO_DIM",
    "RANKS",
    "VOCAB_SIZE",
    "TokenizerConfig",
    "encode_move_to_policy_index",
    "encode_policy_target",
    "flip_square",
    "index_to_square",
    "parse_square",
    "parse_uci_move",
    "setup_logging",
    "tokenize",
    "tokenizer",
]
