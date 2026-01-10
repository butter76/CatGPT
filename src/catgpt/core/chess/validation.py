"""FEN validation for neural network training/inference.

This module provides validation utilities to ensure FEN strings are suitable
for use with neural networks. It enforces stricter rules than standard FEN
validation to ensure consistency and avoid edge cases that could confuse models.
"""

import chess


class FENValidationError(ValueError):
    """Base exception for FEN validation errors."""

    pass


class TerminalPositionError(FENValidationError):
    """Raised when position is terminal (checkmate, stalemate, or 50-move rule)."""

    pass


class InvalidEnPassantError(FENValidationError):
    """Raised when FEN contains an en passant square but no legal en passant capture."""

    pass


class FischerRandomCastlingError(FENValidationError):
    """Raised when castling rights exist but rooks are not in standard positions."""

    pass


def validate_fen_for_network(fen: str) -> None:
    """Validate a FEN string for use with neural networks.

    This function ensures the FEN represents a valid, non-terminal position
    that follows standard chess conventions (not Fischer Random / Chess960).

    Args:
        fen: The FEN string to validate.

    Raises:
        TerminalPositionError: If the position is checkmate, stalemate,
            or the halfmove clock is >= 100 (50-move rule).
        InvalidEnPassantError: If the FEN specifies an en passant square
            but no legal en passant capture exists.
        FischerRandomCastlingError: If castling rights are specified but
            rooks are not in their standard starting positions (h1/a1 for white,
            h8/a8 for black).
        ValueError: If the FEN is malformed and cannot be parsed.
    """
    # First check for Fischer Random-style castling by comparing input FEN
    # with what python-chess normalizes. We do this before creating the board
    # in standard mode because python-chess will strip invalid castling rights.
    _validate_standard_castling_from_fen(fen)

    # Parse the FEN - this will raise ValueError if malformed
    board = chess.Board(fen)

    # Check for terminal position
    _validate_not_terminal(board)

    # Check for invalid en passant
    _validate_en_passant(board, fen)


def _validate_not_terminal(board: chess.Board) -> None:
    """Validate that the position is not terminal.

    Args:
        board: The chess board to validate.

    Raises:
        TerminalPositionError: If the position is checkmate, stalemate,
            or halfmove clock >= 100.
    """
    if board.is_checkmate():
        raise TerminalPositionError(
            "Position is checkmate - terminal positions cannot be used for training"
        )

    if board.is_stalemate():
        raise TerminalPositionError(
            "Position is stalemate - terminal positions cannot be used for training"
        )

    if board.halfmove_clock >= 100:
        raise TerminalPositionError(
            f"Halfmove clock is {board.halfmove_clock} (>= 100) - "
            "position is a draw by 50-move rule"
        )


def _validate_en_passant(board: chess.Board, fen: str) -> None:
    """Validate that en passant square in FEN has a legal capture.

    Args:
        board: The chess board to validate.
        fen: The original FEN string (used for error messages).

    Raises:
        InvalidEnPassantError: If FEN specifies an en passant square
            but no legal en passant capture exists.
    """
    # Get the en passant field from the FEN
    fen_parts = fen.split()
    if len(fen_parts) < 4:
        return  # Malformed FEN will be caught elsewhere

    ep_field = fen_parts[3]

    if ep_field == "-":
        return  # No en passant claimed

    # An en passant square is specified - verify there's a legal capture
    # Note: python-chess normalizes FENs to remove invalid ep squares,
    # so if board.ep_square is None but FEN has an ep square, it's invalid
    if board.ep_square is None:
        raise InvalidEnPassantError(
            f"FEN specifies en passant square '{ep_field}' but no legal "
            "en passant capture exists (possibly blocked by pin or no capturing pawn)"
        )

    # Double-check: verify at least one legal move is an en passant capture
    has_legal_ep = any(
        board.is_en_passant(move)
        for move in board.legal_moves
    )

    if not has_legal_ep:
        raise InvalidEnPassantError(
            f"FEN specifies en passant square '{ep_field}' but no legal "
            "en passant capture is available"
        )


def _validate_standard_castling_from_fen(fen: str) -> None:
    """Validate that castling rights in FEN correspond to standard rook positions.

    This function parses the FEN in Chess960 mode to preserve the original
    castling rights, then checks if rooks are in standard positions.

    In standard chess, castling requires rooks on their original squares:
    - White kingside (K): rook on h1
    - White queenside (Q): rook on a1
    - Black kingside (k): rook on h8
    - Black queenside (q): rook on a8

    In Fischer Random (Chess960), rooks can start on different files.
    This function rejects such positions.

    Args:
        fen: The FEN string to validate.

    Raises:
        FischerRandomCastlingError: If castling rights exist but rooks
            are not in standard positions.
        ValueError: If the FEN is malformed.
    """
    # Parse in Chess960 mode to preserve original castling rights
    # (standard mode normalizes/removes invalid castling rights)
    board = chess.Board(fen, chess960=True)

    # Check white kingside castling (K in FEN)
    if board.has_kingside_castling_rights(chess.WHITE):
        rook = board.piece_at(chess.H1)
        if rook is None or rook.piece_type != chess.ROOK or rook.color != chess.WHITE:
            raise FischerRandomCastlingError(
                "White has kingside castling rights but no white rook on h1 - "
                "this appears to be a Fischer Random (Chess960) position"
            )

    # Check white queenside castling (Q in FEN)
    if board.has_queenside_castling_rights(chess.WHITE):
        rook = board.piece_at(chess.A1)
        if rook is None or rook.piece_type != chess.ROOK or rook.color != chess.WHITE:
            raise FischerRandomCastlingError(
                "White has queenside castling rights but no white rook on a1 - "
                "this appears to be a Fischer Random (Chess960) position"
            )

    # Check black kingside castling (k in FEN)
    if board.has_kingside_castling_rights(chess.BLACK):
        rook = board.piece_at(chess.H8)
        if rook is None or rook.piece_type != chess.ROOK or rook.color != chess.BLACK:
            raise FischerRandomCastlingError(
                "Black has kingside castling rights but no black rook on h8 - "
                "this appears to be a Fischer Random (Chess960) position"
            )

    # Check black queenside castling (q in FEN)
    if board.has_queenside_castling_rights(chess.BLACK):
        rook = board.piece_at(chess.A8)
        if rook is None or rook.piece_type != chess.ROOK or rook.color != chess.BLACK:
            raise FischerRandomCastlingError(
                "Black has queenside castling rights but no black rook on a8 - "
                "this appears to be a Fischer Random (Chess960) position"
            )
