"""Tests for chess engine implementation."""

import chess
import pytest

from catgpt.core.chess import (
    ChessEngine,
    FENValidationError,
    FischerRandomCastlingError,
    InvalidEnPassantError,
    MoveEvaluation,
    TerminalPositionError,
    validate_fen_for_network,
)


class TestChessEngine:
    """Tests for the ChessEngine class."""

    def test_engine_initializes_with_starting_position(self) -> None:
        """Test that engine starts with standard chess position."""
        engine = ChessEngine()
        assert engine.board.fen() == chess.STARTING_FEN

    def test_reset_returns_to_starting_position(self) -> None:
        """Test that reset() returns to starting position."""
        engine = ChessEngine()
        engine.make_move("e2e4")
        engine.reset()
        assert engine.board.fen() == chess.STARTING_FEN

    def test_set_position_from_fen(self) -> None:
        """Test setting position from FEN string."""
        engine = ChessEngine()
        # Use a FEN without en passant to avoid normalization differences
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        engine.set_position(fen)
        assert engine.board.fen() == fen

    def test_make_legal_move_succeeds(self) -> None:
        """Test that legal moves are accepted."""
        engine = ChessEngine()
        result = engine.make_move("e2e4")
        assert result is True
        assert engine.board.piece_at(chess.E4) == chess.Piece(chess.PAWN, chess.WHITE)

    def test_make_illegal_move_fails(self) -> None:
        """Test that illegal moves are rejected."""
        engine = ChessEngine()
        result = engine.make_move("e2e5")  # Can't move pawn 3 squares
        assert result is False

    def test_make_invalid_uci_fails(self) -> None:
        """Test that invalid UCI notation is rejected."""
        engine = ChessEngine()
        result = engine.make_move("invalid")
        assert result is False

    def test_get_best_move_returns_legal_move(self) -> None:
        """Test that get_best_move returns a legal move."""
        engine = ChessEngine()
        result = engine.get_best_move()

        assert result is not None
        assert isinstance(result, MoveEvaluation)
        assert result.move in engine.board.legal_moves

    def test_get_best_move_returns_none_when_no_moves(self) -> None:
        """Test that get_best_move returns None in checkmate."""
        engine = ChessEngine()
        # Fool's mate position - black is checkmated
        engine.set_position("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        result = engine.get_best_move()
        assert result is None

    def test_evaluate_position_starting(self) -> None:
        """Test evaluation of starting position is approximately equal."""
        engine = ChessEngine()
        score = engine.evaluate_position()
        assert score == 0.0  # Starting position is equal material

    def test_evaluate_position_material_advantage(self) -> None:
        """Test evaluation reflects material advantage."""
        engine = ChessEngine()
        # Position where white has an extra queen
        engine.set_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        baseline = engine.evaluate_position()

        # Remove black's queen
        engine.set_position("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        with_advantage = engine.evaluate_position()

        assert with_advantage > baseline  # White should be better without black's queen


class TestPythonChessEnPassantFEN:
    """Tests confirming python-chess behavior for en passant in FEN strings.

    Key rule: An en passant square should only appear in the FEN if there is
    at least one legal en passant capture available. If en passant would put
    the capturing player's king in check, the en passant square should not
    be displayed in the FEN.
    """

    def test_en_passant_square_shown_when_legal_capture_exists(self) -> None:
        """FEN should show en passant square when a legal capture is possible."""
        board = chess.Board()
        # Play 1. e4 e6 2. e5 d5 - white pawn on e5, black pawn just moved d7-d5
        board.push_san("e4")
        board.push_san("e6")
        board.push_san("e5")
        board.push_san("d5")

        fen = board.fen()
        # The FEN should contain "d6" as the en passant square
        assert "d6" in fen, f"Expected en passant square d6 in FEN: {fen}"

        # Verify that en passant is actually a legal move
        ep_move = chess.Move.from_uci("e5d6")
        assert ep_move in board.legal_moves, "En passant e5d6 should be legal"

    def test_en_passant_square_hidden_when_would_cause_check(self) -> None:
        """FEN should NOT show en passant square if capture would expose king to check.

        This tests the case where a pawn moved two squares and created an en passant
        opportunity, but capturing en passant would leave the capturing player's
        king in check (e.g., due to a pin along the rank).
        """
        # Position: White king on e1, white pawn on d5, black rook on a5,
        # black king on h8. If black plays c7-c5, the c6 en passant square
        # should NOT appear because capturing would expose white's king to
        # the rook on a5.
        #
        # But we need white's pawn to be able to capture, so let's construct
        # a different scenario where black to move, white just played pawn to 4th rank.

        # Better setup: White king on g5, black pawn on f5, white rook on a5,
        # Black plays e7-e5. En passant f5xe6 would be illegal due to discovered check.
        # Actually, let's use a well-known position for this.

        # Classic example: Pin along the rank preventing en passant
        # Position: 8/8/8/1k1pP1K1/8/8/8/8 w - d6 0 1
        # But this doesn't have a legal ep, so python-chess should normalize it.

        # Let's construct from scratch:
        # White: King on g5, Pawn on f5
        # Black: King on b5, Pawn on e5 (just moved from e7)
        # If white tries f5xe6 en passant, the e-file is cleared but that doesn't cause check.
        # We need the king to be on the same rank as an enemy rook/queen.

        # Setup: 8/8/8/RkPpK3/8/8/8/8 w - d6 0 1
        # White: King e5, Pawn c5, Rook a5
        # Black: King b5, Pawn d5 (just moved from d7)
        # If white plays cxd6 en passant, the b5 king is still protected... wait, we want
        # black's en passant to be blocked.

        # Let's try: White king on a5, white pawn on e5, black pawn on d5, black rook on h5
        # 8/8/8/K2pP2r/8/8/8/4k3 w - d6 0 1
        # Here white can play exd6 en passant. After dxe6 is removed, king on a5 is
        # not attacked by rook on h5 because pawns are in the way... actually after
        # en passant both the d5 and e5 pawns would be gone from the 5th rank.

        # Actually, the canonical example:
        # 8/8/8/2k1pP1K/8/8/8/8 w - e6 0 1
        # White king h5, white pawn f5, black king c5, black pawn e5
        # En passant fxe6 is illegal because it would leave white king in check? No...

        # The definitive example with a rook pin:
        # Position where en passant would expose the king to a rook attack on the same rank.
        # 8/8/8/k2pP2R/8/8/8/4K3 w - d6 0 1
        # Black king a5, black pawn d5, white pawn e5, white rook h5, white king e1
        # WAIT - this is white to move. White would capture exd6. Let's check if that's legal.
        # After exd6, the d5 pawn is gone, e5 pawn moves to d6. Black king on a5 not in check.

        # Let me think again. We need:
        # - It's white to move
        # - Black just played a pawn two squares (say d7-d5)
        # - White has a pawn that could capture en passant
        # - BUT capturing en passant would put WHITE's king in check

        # 8/8/8/r2pPK2/8/8/8/4k3 w - d6 0 1
        # Black rook a5, black pawn d5, white pawn e5, white king f5, black king e1
        # If white plays exd6, the d5 pawn is removed, e5 pawn goes to d6.
        # The 5th rank now has: rook a5, king f5 - CHECK!
        # So exd6 en passant is illegal.

        # Let's verify python-chess handles this correctly
        fen_with_ep_square = "8/8/8/r2pPK2/8/8/8/4k3 w - d6 0 1"
        board = chess.Board(fen_with_ep_square)

        # The en passant capture should be illegal
        ep_move = chess.Move.from_uci("e5d6")
        assert ep_move not in board.legal_moves, "En passant should be illegal due to discovered check"

        # The key behavior: python-chess should NOT include the en passant square
        # in the FEN it generates, since there's no legal en passant move
        generated_fen = board.fen()
        fen_parts = generated_fen.split()
        ep_field = fen_parts[3]  # En passant field is the 4th field

        assert ep_field == "-", (
            f"En passant square should not appear in FEN when capture is illegal. "
            f"Expected '-', got '{ep_field}'. Full FEN: {generated_fen}"
        )

    def test_en_passant_square_hidden_when_no_capturing_pawn_exists(self) -> None:
        """FEN should NOT show en passant square if no pawn can capture.

        Even if a pawn moved two squares, if there's no enemy pawn in position
        to capture en passant, the square should not be shown.
        """
        # Position: White pawn on e5, no black pawn on d5 or f5
        # Black plays h7-h5 - no white pawn can capture en passant
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e6")
        board.push_san("e5")
        board.push_san("h5")  # No white pawn adjacent to capture

        generated_fen = board.fen()
        fen_parts = generated_fen.split()
        ep_field = fen_parts[3]

        assert ep_field == "-", (
            f"En passant square should not appear when no capturing pawn exists. "
            f"Expected '-', got '{ep_field}'. Full FEN: {generated_fen}"
        )

    def test_en_passant_legal_capture_both_sides(self) -> None:
        """Test en passant when pawns on both sides can capture."""
        # White pawns on d5 and f5, black plays e7-e5
        board = chess.Board()
        # 1. d4 a6 2. d5 b6 3. f4 c6 4. f5 e5
        board.push_san("d4")
        board.push_san("a6")
        board.push_san("d5")
        board.push_san("b6")
        board.push_san("f4")
        board.push_san("c6")
        board.push_san("f5")
        board.push_san("e5")

        fen = board.fen()
        assert "e6" in fen, f"Expected en passant square e6 in FEN: {fen}"

        # Both captures should be legal
        assert chess.Move.from_uci("d5e6") in board.legal_moves
        assert chess.Move.from_uci("f5e6") in board.legal_moves


class TestPythonChessCastlingUCI:
    """Tests confirming python-chess UCI notation for castling.

    Key rule: Castling in UCI notation is represented by how the king moves,
    not as king-captures-rook. For example:
    - White kingside castling: e1g1 (king from e1 to g1)
    - White queenside castling: e1c1 (king from e1 to c1)
    - Black kingside castling: e8g8 (king from e8 to g8)
    - Black queenside castling: e8c8 (king from e8 to c8)
    """

    def test_white_kingside_castling_uci(self) -> None:
        """White kingside castling should be e1g1 in UCI notation."""
        # Position where white can castle kingside
        board = chess.Board()
        # Clear the path: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 - now white can castle
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("Bc4")
        board.push_san("Bc5")

        # Find the castling move
        castling_move = chess.Move.from_uci("e1g1")
        assert castling_move in board.legal_moves, "Kingside castling should be legal"

        # Verify UCI notation is e1g1 (king's destination), not e1h1 (rook's square)
        assert castling_move.uci() == "e1g1", f"Expected 'e1g1', got '{castling_move.uci()}'"

    def test_white_queenside_castling_uci(self) -> None:
        """White queenside castling should be e1c1 in UCI notation."""
        # Position where white can castle queenside
        board = chess.Board()
        # 1. d4 d5 2. Nc3 Nc6 3. Bf4 Bf5 4. Qd3 Qd6
        board.push_san("d4")
        board.push_san("d5")
        board.push_san("Nc3")
        board.push_san("Nc6")
        board.push_san("Bf4")
        board.push_san("Bf5")
        board.push_san("Qd3")
        board.push_san("Qd6")

        castling_move = chess.Move.from_uci("e1c1")
        assert castling_move in board.legal_moves, "Queenside castling should be legal"
        assert castling_move.uci() == "e1c1", f"Expected 'e1c1', got '{castling_move.uci()}'"

    def test_black_kingside_castling_uci(self) -> None:
        """Black kingside castling should be e8g8 in UCI notation."""
        board = chess.Board()
        # 1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. d3
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nf6")
        board.push_san("Bc4")
        board.push_san("Bc5")
        board.push_san("d3")

        castling_move = chess.Move.from_uci("e8g8")
        assert castling_move in board.legal_moves, "Black kingside castling should be legal"
        assert castling_move.uci() == "e8g8", f"Expected 'e8g8', got '{castling_move.uci()}'"

    def test_black_queenside_castling_uci(self) -> None:
        """Black queenside castling should be e8c8 in UCI notation."""
        board = chess.Board()
        # 1. d4 d5 2. Nf3 Nc6 3. e3 Bf5 4. Bd3 Qd7 5. O-O
        board.push_san("d4")
        board.push_san("d5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("e3")
        board.push_san("Bf5")
        board.push_san("Bd3")
        board.push_san("Qd7")
        board.push_san("O-O")

        castling_move = chess.Move.from_uci("e8c8")
        assert castling_move in board.legal_moves, "Black queenside castling should be legal"
        assert castling_move.uci() == "e8c8", f"Expected 'e8c8', got '{castling_move.uci()}'"

    def test_castling_move_from_board_legal_moves(self) -> None:
        """Verify castling moves from legal_moves use king-movement notation."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("Bc4")
        board.push_san("Bc5")

        # Find the castling move in legal moves by checking if it's a castling move
        castling_moves = [m for m in board.legal_moves if board.is_castling(m)]
        assert len(castling_moves) == 1, "Should have exactly one castling move available"

        castling_move = castling_moves[0]
        assert castling_move.uci() == "e1g1", (
            f"Castling move from legal_moves should be 'e1g1', got '{castling_move.uci()}'"
        )

    def test_castling_executed_correctly(self) -> None:
        """Verify that castling actually moves both king and rook."""
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("Bc4")
        board.push_san("Bc5")

        # Execute castling using UCI notation
        board.push_uci("e1g1")

        # King should be on g1
        assert board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE), (
            "King should be on g1 after castling"
        )
        # Rook should be on f1
        assert board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE), (
            "Rook should be on f1 after castling"
        )
        # Original squares should be empty
        assert board.piece_at(chess.E1) is None, "e1 should be empty after castling"
        assert board.piece_at(chess.H1) is None, "h1 should be empty after castling"

    def test_castling_uci_output_uses_king_destination(self) -> None:
        """Verify that UCI output for castling uses king's destination square.

        python-chess accepts both notations as input (e1g1 and e1h1 for kingside),
        but always outputs using the king's destination square (e1g1).
        """
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")
        board.push_san("Bc4")
        board.push_san("Bc5")

        # Get castling move from legal moves
        castling_moves = [m for m in board.legal_moves if board.is_castling(m)]
        assert len(castling_moves) == 1

        # The move's UCI representation should be e1g1 (king destination)
        assert castling_moves[0].uci() == "e1g1", (
            "Castling UCI output should use king's destination (g1), not rook's square (h1)"
        )

    def test_castling_accepts_both_input_notations(self) -> None:
        """python-chess accepts both king-destination and king-captures-rook notations.

        Note: While python-chess accepts both e1g1 and e1h1 as input for kingside
        castling, the standard UCI protocol specifies king-destination (e1g1).
        This test documents python-chess's permissive input handling.
        """
        # Test kingside castling with both notations
        board1 = chess.Board()
        board1.push_san("e4")
        board1.push_san("e5")
        board1.push_san("Nf3")
        board1.push_san("Nc6")
        board1.push_san("Bc4")
        board1.push_san("Bc5")

        board2 = board1.copy()

        # Both notations should work and produce the same result
        board1.push_uci("e1g1")  # Standard UCI: king destination
        board2.push_uci("e1h1")  # Alternative: king captures rook

        assert board1.fen() == board2.fen(), (
            "Both e1g1 and e1h1 should result in the same position after castling"
        )

        # Verify castling actually happened
        assert board1.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
        assert board1.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)


class TestValidateFenForNetwork:
    """Tests for the validate_fen_for_network() function.

    This function ensures FEN strings are suitable for neural network
    training/inference by rejecting terminal positions, invalid en passant,
    and Fischer Random (Chess960) castling configurations.
    """

    # ==================== Valid FEN Tests ====================

    def test_valid_starting_position(self) -> None:
        """Starting position should be valid."""
        validate_fen_for_network(chess.STARTING_FEN)

    def test_valid_mid_game_position(self) -> None:
        """A typical mid-game position should be valid."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        validate_fen_for_network(fen)

    def test_valid_position_with_legal_en_passant(self) -> None:
        """Position with legal en passant should be valid."""
        # White pawn on e5, black just played d7-d5
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e6")
        board.push_san("e5")
        board.push_san("d5")
        fen = board.fen()
        validate_fen_for_network(fen)

    def test_valid_position_no_castling_rights(self) -> None:
        """Position with no castling rights should be valid."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w - - 4 4"
        validate_fen_for_network(fen)

    def test_valid_endgame_position(self) -> None:
        """A typical endgame position should be valid."""
        fen = "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"
        validate_fen_for_network(fen)

    # ==================== Terminal Position Tests ====================

    def test_checkmate_raises_terminal_error(self) -> None:
        """Checkmate position should raise TerminalPositionError."""
        # Fool's mate - white is checkmated
        fen = "rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        with pytest.raises(TerminalPositionError, match="checkmate"):
            validate_fen_for_network(fen)

    def test_stalemate_raises_terminal_error(self) -> None:
        """Stalemate position should raise TerminalPositionError."""
        # Classic stalemate: black king in corner, white king and queen nearby
        fen = "k7/2Q5/1K6/8/8/8/8/8 b - - 0 1"
        with pytest.raises(TerminalPositionError, match="stalemate"):
            validate_fen_for_network(fen)

    def test_fifty_move_rule_raises_terminal_error(self) -> None:
        """Position with halfmove clock >= 100 should raise TerminalPositionError."""
        # Valid position but halfmove clock is 100
        fen = "8/8/4k3/8/8/4K3/4P3/8 w - - 100 60"
        with pytest.raises(TerminalPositionError, match="50-move rule"):
            validate_fen_for_network(fen)

    def test_halfmove_99_is_valid(self) -> None:
        """Position with halfmove clock of 99 should still be valid."""
        fen = "8/8/4k3/8/8/4K3/4P3/8 w - - 99 60"
        validate_fen_for_network(fen)

    def test_halfmove_101_raises_terminal_error(self) -> None:
        """Position with halfmove clock > 100 should raise TerminalPositionError."""
        fen = "8/8/4k3/8/8/4K3/4P3/8 w - - 150 80"
        with pytest.raises(TerminalPositionError, match="50-move rule"):
            validate_fen_for_network(fen)

    # ==================== Invalid En Passant Tests ====================

    def test_en_passant_no_capturing_pawn_raises_error(self) -> None:
        """En passant square with no capturing pawn should raise InvalidEnPassantError."""
        # Position claims en passant on e6, but no white pawn can capture
        # White pawn is on a5, not adjacent to e6
        fen = "8/8/8/P3p3/8/8/8/4K2k w - e6 0 1"
        with pytest.raises(InvalidEnPassantError, match="no legal en passant"):
            validate_fen_for_network(fen)

    def test_en_passant_blocked_by_pin_raises_error(self) -> None:
        """En passant that would expose king to check should raise InvalidEnPassantError."""
        # Black rook on a5, black pawn on d5 (just moved), white pawn on e5, white king on f5
        # En passant exd6 would expose white king to rook
        fen = "8/8/8/r2pPK2/8/8/8/4k3 w - d6 0 1"
        with pytest.raises(InvalidEnPassantError, match="no legal en passant"):
            validate_fen_for_network(fen)

    def test_valid_en_passant_not_blocked(self) -> None:
        """Valid en passant position should not raise error."""
        # White pawn on e5, black pawn on d5 (just moved d7-d5), kings far away
        fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
        validate_fen_for_network(fen)

    # ==================== Fischer Random Castling Tests ====================

    def test_fischer_random_white_kingside_raises_error(self) -> None:
        """Kingside castling rights without h1 rook should raise FischerRandomCastlingError."""
        # Position has K (white kingside castling) but no rook on h1
        # Rook is on g1 instead (Fischer Random style)
        fen = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBR1 w KQkq - 0 1"
        with pytest.raises(FischerRandomCastlingError, match="no white rook on h1"):
            validate_fen_for_network(fen)

    def test_fischer_random_white_queenside_raises_error(self) -> None:
        """Queenside castling rights without a1 rook should raise FischerRandomCastlingError."""
        # Position has Q (white queenside castling) but no rook on a1
        # Rook is on b1 instead
        fen = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/1RBQKBNR w KQkq - 0 1"
        with pytest.raises(FischerRandomCastlingError, match="no white rook on a1"):
            validate_fen_for_network(fen)

    def test_fischer_random_black_kingside_raises_error(self) -> None:
        """Kingside castling rights without h8 rook should raise FischerRandomCastlingError."""
        # Position has k (black kingside castling) but no rook on h8
        fen = "rnbqkbr1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with pytest.raises(FischerRandomCastlingError, match="no black rook on h8"):
            validate_fen_for_network(fen)

    def test_fischer_random_black_queenside_raises_error(self) -> None:
        """Queenside castling rights without a8 rook should raise FischerRandomCastlingError."""
        # Position has q (black queenside castling) but no rook on a8
        fen = "1rbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        with pytest.raises(FischerRandomCastlingError, match="no black rook on a8"):
            validate_fen_for_network(fen)

    def test_valid_castling_standard_position(self) -> None:
        """Standard position with castling rights should be valid."""
        validate_fen_for_network(chess.STARTING_FEN)

    def test_castling_rights_removed_after_rook_moves(self) -> None:
        """Position after rook moved should have updated castling rights and be valid."""
        # Position where white's h1 rook has moved, castling rights updated to Q only
        fen = "r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1"
        validate_fen_for_network(fen)

    def test_partial_castling_rights_valid(self) -> None:
        """Position with some castling rights should be valid if rooks are correct."""
        # White has only kingside, black has only queenside - rooks in correct positions
        fen = "r3kbn1/pppppppp/8/8/8/8/PPPPPPPP/1NBQKB1R w Kq - 0 1"
        validate_fen_for_network(fen)

    def test_no_castling_rights_missing_rooks_valid(self) -> None:
        """Position with no castling rights should be valid even without rooks."""
        fen = "4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"
        validate_fen_for_network(fen)

    # ==================== Malformed FEN Tests ====================

    def test_malformed_fen_raises_value_error(self) -> None:
        """Malformed FEN should raise ValueError."""
        with pytest.raises(ValueError):
            validate_fen_for_network("not a valid fen string")

    def test_invalid_piece_placement_raises_error(self) -> None:
        """FEN with invalid piece placement should raise ValueError."""
        with pytest.raises(ValueError):
            validate_fen_for_network("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")

    # ==================== Exception Hierarchy Tests ====================

    def test_exception_hierarchy(self) -> None:
        """All custom exceptions should inherit from FENValidationError."""
        assert issubclass(TerminalPositionError, FENValidationError)
        assert issubclass(InvalidEnPassantError, FENValidationError)
        assert issubclass(FischerRandomCastlingError, FENValidationError)
        assert issubclass(FENValidationError, ValueError)

    def test_can_catch_all_validation_errors(self) -> None:
        """All validation errors should be catchable with FENValidationError."""
        # Checkmate
        with pytest.raises(FENValidationError):
            validate_fen_for_network("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")

        # Invalid en passant
        with pytest.raises(FENValidationError):
            validate_fen_for_network("8/8/8/P3p3/8/8/8/4K2k w - e6 0 1")

        # Fischer Random castling (white kingside with rook on g1 instead of h1)
        with pytest.raises(FENValidationError):
            validate_fen_for_network("r3kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBR1 w Kkq - 0 1")
