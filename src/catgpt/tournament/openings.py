"""Opening book loader for chess engine tournaments.

Supports EPD (Extended Position Description) and PGN formats.
EPD is preferred as it's simpler and directly provides FEN positions.

Common opening book sources:
- https://github.com/official-stockfish/books (Stockfish opening books)
- https://github.com/AndyGrant/openbench-books (OpenBench books)
"""

import random
from pathlib import Path

import chess
import chess.pgn
from loguru import logger


def load_openings(
    path: str | Path,
    *,
    shuffle: bool = True,
    seed: int | None = None,
    max_openings: int | None = None,
) -> list[str]:
    """Load opening positions from an EPD or PGN file.

    Args:
        path: Path to opening book file (.epd or .pgn).
        shuffle: Whether to randomize opening order.
        seed: Random seed for shuffling. None for random.
        max_openings: Maximum number of openings to load. None for all.

    Returns:
        List of FEN strings representing opening positions.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is not supported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Opening book not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".epd":
        openings = _load_epd(path)
    elif suffix == ".pgn":
        openings = _load_pgn(path)
    elif suffix == ".fen":
        openings = _load_fen(path)
    else:
        raise ValueError(
            f"Unsupported opening book format: {suffix}. "
            "Supported formats: .epd, .pgn, .fen"
        )

    logger.info(f"Loaded {len(openings)} openings from {path}")

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(openings)
        logger.debug(f"Shuffled openings (seed={seed})")

    if max_openings is not None and len(openings) > max_openings:
        openings = openings[:max_openings]
        logger.debug(f"Truncated to {max_openings} openings")

    return openings


def _load_epd(path: Path) -> list[str]:
    """Load openings from EPD file.

    EPD format is similar to FEN but may include operations (opcodes).
    We extract just the position part and convert to full FEN.

    Example EPD line:
        rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 hmvc 0; fmvn 1;

    Args:
        path: Path to EPD file.

    Returns:
        List of FEN strings.
    """
    openings = []

    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Parse EPD - chess library handles the format
                board = chess.Board()
                # EPD may have opcodes after the position, separated by semicolons
                # We only need the position part (first 4-6 space-separated fields)
                parts = line.split()

                # Construct FEN from EPD fields
                # EPD has: pieces active_color castling en_passant [opcodes...]
                # FEN adds: halfmove_clock fullmove_number
                if len(parts) >= 4:
                    fen_parts = parts[:4]

                    # Look for halfmove and fullmove in opcodes
                    halfmove = "0"
                    fullmove = "1"

                    for i, part in enumerate(parts[4:], 4):
                        if part.startswith("hmvc"):
                            # Halfmove clock: hmvc 5;
                            if i + 1 < len(parts):
                                halfmove = parts[i + 1].rstrip(";")
                        elif part.startswith("fmvn"):
                            # Fullmove number: fmvn 10;
                            if i + 1 < len(parts):
                                fullmove = parts[i + 1].rstrip(";")

                    fen = " ".join(fen_parts + [halfmove, fullmove])

                    # Validate by trying to create a board
                    board = chess.Board(fen)
                    openings.append(board.fen())
                else:
                    logger.warning(f"Line {line_num}: Invalid EPD format, skipping")

            except Exception as e:
                logger.warning(f"Line {line_num}: Failed to parse EPD: {e}")
                continue

    return openings


def _load_pgn(path: Path, moves_to_play: int = 8) -> list[str]:
    """Load openings from PGN file.

    Plays the first N moves of each game and extracts the resulting position.

    Args:
        path: Path to PGN file.
        moves_to_play: Number of half-moves (plies) to play from each game.

    Returns:
        List of FEN strings.
    """
    openings = []
    seen_fens = set()

    with path.open() as f:
        while True:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()

                # Play the opening moves
                for i, move in enumerate(game.mainline_moves()):
                    if i >= moves_to_play:
                        break
                    board.push(move)

                fen = board.fen()

                # Deduplicate
                if fen not in seen_fens:
                    seen_fens.add(fen)
                    openings.append(fen)

            except Exception as e:
                logger.warning(f"Failed to parse PGN game: {e}")
                continue

    return openings


def _load_fen(path: Path) -> list[str]:
    """Load openings from a simple FEN file (one FEN per line).

    Args:
        path: Path to FEN file.

    Returns:
        List of FEN strings.
    """
    openings = []

    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Validate FEN
                board = chess.Board(line)
                openings.append(board.fen())
            except Exception as e:
                logger.warning(f"Line {line_num}: Invalid FEN: {e}")
                continue

    return openings
