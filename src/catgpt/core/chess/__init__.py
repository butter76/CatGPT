"""Chess engine and utilities (framework-agnostic)."""

from catgpt.core.chess.engine import ChessEngine, MoveEvaluation
from catgpt.core.chess.validation import (
    FENValidationError,
    FischerRandomCastlingError,
    InvalidEnPassantError,
    TerminalPositionError,
    validate_fen_for_network,
)

__all__ = [
    "ChessEngine",
    "MoveEvaluation",
    "FENValidationError",
    "FischerRandomCastlingError",
    "InvalidEnPassantError",
    "TerminalPositionError",
    "validate_fen_for_network",
]
