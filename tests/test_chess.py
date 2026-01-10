"""Tests for chess engine implementation."""

import chess

from catgpt.core.chess import ChessEngine, MoveEvaluation


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
