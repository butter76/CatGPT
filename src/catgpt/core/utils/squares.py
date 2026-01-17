"""Chess square utilities for board indexing.

This module provides utilities for converting between algebraic notation (e.g., "e4")
and board indices used throughout CatGPT.

Board indexing convention (row-major from a8):
    a8=0,  b8=1,  c8=2,  d8=3,  e8=4,  f8=5,  g8=6,  h8=7
    a7=8,  b7=9,  c7=10, d7=11, e7=12, f7=13, g7=14, h7=15
    ...
    a1=56, b1=57, c1=58, d1=59, e1=60, f1=61, g1=62, h1=63

This matches visual reading order (top-left to bottom-right when viewing
the board from white's perspective).
"""

# File letters and rank numbers for square parsing
FILES = "abcdefgh"
RANKS = "12345678"


def parse_square(square: str, *, flip: bool = False) -> int:
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

    if file_char not in FILES or rank_char not in RANKS:
        msg = f"Invalid square notation: {square!r}"
        raise ValueError(msg)

    file_idx = FILES.index(file_char)  # 0-7 (a-h)
    rank_idx = RANKS.index(rank_char)  # 0-7 (1-8)

    if flip:
        rank_idx = 7 - rank_idx

    # Convert to row-major index (a8=0, h8=7, a7=8, ..., h1=63)
    # rank 8 is row 0, rank 1 is row 7
    row = 7 - rank_idx
    return row * 8 + file_idx


def index_to_square(index: int, *, flip: bool = False) -> str:
    """Convert board index to algebraic notation.

    Args:
        index: Board index (0-63, row-major from a8).
        flip: If True, flip the board vertically (swap ranks 1↔8, 2↔7, etc.).

    Returns:
        Algebraic notation for the square (e.g., 'a1', 'h8').

    Raises:
        ValueError: If the index is out of range.
    """
    if not 0 <= index < 64:
        msg = f"Invalid board index: {index}"
        raise ValueError(msg)

    row = index // 8
    file_idx = index % 8

    # row 0 = rank 8, row 7 = rank 1
    rank_idx = 7 - row

    if flip:
        rank_idx = 7 - rank_idx

    return FILES[file_idx] + RANKS[rank_idx]


def flip_square(square: str) -> str:
    """Flip a square vertically (swap ranks 1↔8, 2↔7, etc.).

    Args:
        square: Algebraic notation for a square (e.g., 'e2').

    Returns:
        The flipped square (e.g., 'e7').
    """
    if len(square) != 2:
        msg = f"Invalid square notation: {square!r}"
        raise ValueError(msg)

    file_char, rank_char = square[0].lower(), square[1]

    if file_char not in FILES or rank_char not in RANKS:
        msg = f"Invalid square notation: {square!r}"
        raise ValueError(msg)

    rank_idx = RANKS.index(rank_char)
    flipped_rank_idx = 7 - rank_idx

    return file_char + RANKS[flipped_rank_idx]
