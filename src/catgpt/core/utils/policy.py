"""Policy encoding utilities for chess move distributions.

This module provides utilities for encoding/decoding moves to/from the policy
tensor format used by the model's policy head.

Policy tensor shape: (64, 73)
- 64 = from_square (source square, row-major a8=0 to h1=63)
- 73 = to_square encoding:
  - 0-63: Normal destination squares (including queen promotions)
  - 64-66: Knight underpromotions (left capture, straight, right capture)
  - 67-69: Bishop underpromotions (left capture, straight, right capture)
  - 70-72: Rook underpromotions (left capture, straight, right capture)

Note: The tokenizer flips positions when black is to move, so policy indices
must also be flipped accordingly.
"""

import chess
import numpy as np

from catgpt.core.utils.squares import flip_square, parse_square

# Policy target dimensions
# 64 normal destination squares + 9 underpromotion targets
# Underpromotions: 3 pieces (knight, bishop, rook) x 3 directions (left, straight, right)
POLICY_TO_DIM = 73
POLICY_SHAPE = (64, POLICY_TO_DIM)  # (from_square, to_square)

# Underpromotion piece type to index offset
_UNDERPROMO_PIECE_OFFSET = {"n": 0, "b": 1, "r": 2}
_UNDERPROMO_INDEX_TO_PIECE = {0: "n", 1: "b", 2: "r"}


def parse_uci_move(uci: str) -> tuple[str, str, str | None]:
    """Parse a UCI move string into (from_square, to_square, promotion).

    Args:
        uci: UCI move string (e.g., "e2e4", "e7e8q", "a7a8n").

    Returns:
        Tuple of (from_square, to_square, promotion_piece or None).
        promotion_piece is lowercase: 'q', 'r', 'b', 'n'.
    """
    from_sq = uci[:2]
    to_sq = uci[2:4]
    promo = uci[4].lower() if len(uci) > 4 else None
    return from_sq, to_sq, promo


def encode_move_to_policy_index(
    move: chess.Move,
    flip: bool = False,
) -> tuple[int, int]:
    """Convert a chess.Move to policy tensor indices (from_idx, to_idx).

    Args:
        move: A chess.Move object.
        flip: Whether to flip squares (for black to move, to match tokenizer).

    Returns:
        Tuple of (from_idx, to_idx) for indexing into policy tensor.
        from_idx is in [0, 63], to_idx is in [0, 72].
    """
    uci = move.uci()
    from_sq, to_sq, promo = parse_uci_move(uci)

    if flip:
        from_sq = flip_square(from_sq)
        to_sq = flip_square(to_sq)

    from_idx = parse_square(from_sq)

    if promo and promo != "q":
        # Underpromotion: map to indices 64-72
        # file_diff: -1 (capture left), 0 (straight), +1 (capture right)
        file_diff = ord(to_sq[0]) - ord(from_sq[0])
        to_idx = 64 + _UNDERPROMO_PIECE_OFFSET[promo] * 3 + (file_diff + 1)
    else:
        # Normal move or queen promotion: use destination square
        to_idx = parse_square(to_sq)

    return from_idx, to_idx


def encode_policy_target(
    legal_moves: list[tuple[str, float]],
    flip: bool = False,
) -> np.ndarray:
    """Convert legal moves with policy to (64, 73) target tensor.

    The policy target encodes the move distribution over a (from_square, to_square)
    tensor. Normal moves use to_square indices 0-63. Underpromotions (non-queen
    promotions) use indices 64-72:
        64-66: knight promotions (left capture, straight, right capture)
        67-69: bishop promotions (left capture, straight, right capture)
        70-72: rook promotions (left capture, straight, right capture)

    Queen promotions use the normal destination square (0-63).

    Args:
        legal_moves: List of (uci_move, probability) tuples.
        flip: Whether to flip squares (for black to move, to match tokenizer).

    Returns:
        Shape (64, 73) array with policy probabilities.
    """
    target = np.zeros(POLICY_SHAPE, dtype=np.float32)

    for uci_move, prob in legal_moves:
        from_sq, to_sq, promo = parse_uci_move(uci_move)

        if flip:
            from_sq = flip_square(from_sq)
            to_sq = flip_square(to_sq)

        from_idx = parse_square(from_sq)

        if promo and promo != "q":
            # Underpromotion: map to indices 64-72
            # file_diff: -1 (capture left), 0 (straight), +1 (capture right)
            file_diff = ord(to_sq[0]) - ord(from_sq[0])
            to_idx = 64 + _UNDERPROMO_PIECE_OFFSET[promo] * 3 + (file_diff + 1)
        else:
            # Normal move or queen promotion: use destination square
            to_idx = parse_square(to_sq)

        target[from_idx, to_idx] = prob

    return target
