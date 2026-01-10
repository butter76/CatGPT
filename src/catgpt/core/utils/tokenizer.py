"""Tokenization of FEN (Forsyth-Edwards Notation) strings for chess positions.

This module provides utilities for converting chess board positions in FEN format
into a fixed-length sequence of tokens suitable for neural network input.

The tokenization scheme represents each square on the board as a single token,
with special handling for:
- Castling rights: Rooks that can castle are marked with 'C'/'c' instead of 'R'/'r'
- En passant: The en passant target square is marked with 'x'
- Board orientation: When it's black's turn, the board is flipped and pieces swapped
- Halfmove clock: Optionally appended as 2 digits (left-padded with '.')
"""

from dataclasses import dataclass

import numpy as np

# Character vocabulary for tokenization
# Digits 0-9 are used for the halfmove clock
# Lowercase letters are black pieces, uppercase are white pieces
# 'c'/'C' marks rooks with castling rights
# 'x' marks the en passant target square
# '.' is used for empty squares and padding
_CHARACTERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "p", "b", "n", "r", "c", "k", "q",  # Non-side-to-move pieces (c = castling rook)
    "P", "B", "N", "R", "C", "Q", "K",  # Side-to-move pieces (C = castling rook)
    "x",  # En passant target square
    ".",  # Empty square / padding
]
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACE_DIGITS = frozenset({"1", "2", "3", "4", "5", "6", "7", "8"})

# File letters and rank numbers for square parsing
_FILES = "abcdefgh"
_RANKS = "12345678"


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for FEN tokenization.

    Attributes:
        sequence_length: Total length of the output token sequence.
            Must be at least 66 if include_halfmove is True, otherwise at least 64.
        include_halfmove: Whether to include the halfmove clock (2 digits) in the output.
    """

    sequence_length: int = 67
    include_halfmove: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        min_length = 66 if self.include_halfmove else 64
        if self.sequence_length < min_length:
            msg = (
                f"sequence_length must be at least {min_length} "
                f"(got {self.sequence_length}, include_halfmove={self.include_halfmove})"
            )
            raise ValueError(msg)


def _parse_square(square: str, *, flip: bool = False) -> int:
    """Convert algebraic notation to board index (0-63, row-major from a8 to h1).

    Args:
        square: Algebraic notation for a square (e.g., 'a1', 'h8').
        flip: If True, flip the board vertically (swap ranks 1↔8, 2↔7, etc.).

    Returns:
        Index into a 64-element array representing the board in row-major order,
        starting from a8 (index 0) to h1 (index 63).

    Raises:
        ValueError: If the square notation is invalid.
    """
    if len(square) != 2:
        msg = f"Invalid square notation: {square!r}"
        raise ValueError(msg)

    file_char, rank_char = square[0].lower(), square[1]

    if file_char not in _FILES or rank_char not in _RANKS:
        msg = f"Invalid square notation: {square!r}"
        raise ValueError(msg)

    file_idx = _FILES.index(file_char)  # 0-7 (a-h)
    rank_idx = _RANKS.index(rank_char)  # 0-7 (1-8)

    if flip:
        rank_idx = 7 - rank_idx

    # Convert to row-major index (a8=0, h8=7, a7=8, ..., h1=63)
    # rank 8 is row 0, rank 1 is row 7
    row = 7 - rank_idx
    return row * 8 + file_idx


# Precomputed constants for castling square indices
_CASTLING_SQUARES = {
    "K": _parse_square("h1"),  # White kingside rook
    "Q": _parse_square("a1"),  # White queenside rook
    "k": _parse_square("h8"),  # Black kingside rook
    "q": _parse_square("a8"),  # Black queenside rook
}
_CASTLING_EXPECTED = {"K": "R", "Q": "R", "k": "r", "q": "r"}
_CASTLING_MARKER = {"K": "C", "Q": "C", "k": "c", "q": "c"}

# Lookup table for swapping piece colors (swapcase)
_SWAPCASE = {c: c.swapcase() if c.isalpha() else c for c in _CHARACTERS}

# Token index for padding/empty
_PAD_TOKEN = _CHARACTERS_INDEX["."]
_EN_PASSANT_TOKEN = _CHARACTERS_INDEX["x"]


def tokenize(fen: str, config: TokenizerConfig | None = None) -> np.ndarray:
    """Convert a FEN string to a token sequence. The side-to-move pawns move up the board.

    Args:
        fen: Board position in Forsyth-Edwards Notation (all 6 fields required).
        config: Tokenization configuration. Uses default TokenizerConfig if None.

    Returns:
        numpy array of uint8 tokens with length equal to config.sequence_length.

    Raises:
        ValueError: If the halfmove clock exceeds 2 digits.
        KeyError: If the FEN contains invalid characters.
        AssertionError: If castling rights reference a square without the expected rook.
    """
    if config is None:
        config = TokenizerConfig()

    # Parse FEN fields
    raw_board, side, castling, en_passant, halfmoves_str, _ = fen.split(" ")

    if len(halfmoves_str) > 2:
        raise ValueError(f"Halfmove clock must be 2 digits or less, got: {halfmoves_str!r}")

    # Build board as a list (mutable, efficient for modifications)
    board: list[str] = []
    for char in raw_board:
        if char == "/":
            continue
        elif char in _SPACE_DIGITS:
            board.extend(["."] * int(char))
        else:
            board.append(char)

    # Mark castling rooks
    for char in castling:
        if char in _CASTLING_SQUARES:
            sq = _CASTLING_SQUARES[char]
            expected = _CASTLING_EXPECTED[char]
            assert board[sq] == expected, f"Expected {expected} at castling square, got {board[sq]}"
            board[sq] = _CASTLING_MARKER[char]

    # Flip board if black to move: reverse row order and swap piece colors
    if side == "b":
        flipped: list[str] = []
        for row in range(7, -1, -1):
            for col in range(8):
                flipped.append(_SWAPCASE[board[row * 8 + col]])
        board = flipped

    # Mark en passant target square
    if en_passant != "-":
        en_sq = _parse_square(en_passant, flip=(side == "b"))
        assert board[en_sq] == ".", f"Expected empty square at {en_passant}, got {board[en_sq]}"
        board[en_sq] = "x"

    # Append halfmove clock if configured (left-padded with '.')
    if config.include_halfmove:
        board.append("." if len(halfmoves_str) == 1 else halfmoves_str[0])
        board.append(halfmoves_str[0] if len(halfmoves_str) == 1 else halfmoves_str[1])

    # Convert to numpy array with padding
    result = np.full(config.sequence_length, _PAD_TOKEN, dtype=np.uint8)
    for i, char in enumerate(board):
        result[i] = _CHARACTERS_INDEX[char]

    return result


# Vocabulary size for embedding layers
VOCAB_SIZE = len(_CHARACTERS)
