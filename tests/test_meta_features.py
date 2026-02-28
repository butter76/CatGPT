"""Tests for game verification and utility functions in bagz_to_bag.py.

These tests ensure that the game verification logic correctly validates:
- find_move_between_positions: finding the legal move connecting two positions
- verify_game_integrity: full game verification (standard start, connectivity, legal moves)
- compute_game_result: game result computation from terminal evaluation
"""

import chess
import pytest

from catgpt.core.data.grain.bagz_to_bag import (
    VerificationError,
    compute_game_result,
    find_move_between_positions,
    verify_game_integrity,
)
from catgpt.core.data.grain.coders import LeelaPositionData


def make_position(fen: str, root_q: float = 0.0, root_d: float = 0.5) -> LeelaPositionData:
    """Create a minimal LeelaPositionData for testing.

    Uses default values for fields not relevant to verification.
    """
    board = chess.Board(fen)
    legal_moves = [(m.uci(), 1.0 / board.legal_moves.count()) for m in board.legal_moves]

    return LeelaPositionData(
        fen=fen,
        legal_moves=legal_moves,
        invariance_info=0,
        result=0,
        root_q=root_q,
        root_d=root_d,
        best_q=0.0,
        best_d=0.5,
        played_q=0.0,
        played_d=0.5,
        orig_q=0.0,
        orig_d=0.5,
        best_move_uci=None,
    )


def make_game_from_moves(moves: list[str], start_fen: str = chess.STARTING_FEN) -> list[LeelaPositionData]:
    """Create a list of positions from a sequence of UCI moves.

    Args:
        moves: List of moves in UCI notation (e.g., ["e2e4", "e7e5"]).
        start_fen: Starting position (default: standard starting position).

    Returns:
        List of LeelaPositionData, one for each position (including start).
    """
    board = chess.Board(start_fen)
    positions = [make_position(board.fen())]

    for move_uci in moves:
        board.push_uci(move_uci)
        positions.append(make_position(board.fen()))

    return positions


class TestFindMoveBetweenPositions:
    """Tests for the find_move_between_positions helper."""

    def test_simple_pawn_move(self) -> None:
        """Find a simple pawn move."""
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push_uci("e2e4")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "e2e4"

    def test_knight_move(self) -> None:
        """Find a knight move."""
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push_uci("g1f3")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "g1f3"

    def test_promotion_queen(self) -> None:
        """Find a queen promotion move."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        board1 = chess.Board(fen)
        board2 = chess.Board(fen)
        board2.push_uci("a7a8q")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "a7a8q"

    def test_promotion_knight(self) -> None:
        """Find a knight underpromotion move."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        board1 = chess.Board(fen)
        board2 = chess.Board(fen)
        board2.push_uci("a7a8n")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "a7a8n"

    def test_castling_kingside(self) -> None:
        """Find kingside castling."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        board1 = chess.Board(fen)
        board2 = chess.Board(fen)
        board2.push_uci("e1g1")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "e1g1"

    def test_en_passant(self) -> None:
        """Find en passant capture."""
        fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
        board1 = chess.Board(fen)
        board2 = chess.Board(fen)
        board2.push_uci("e5d6")

        move = find_move_between_positions(board1, board2)
        assert move.uci() == "e5d6"

    def test_no_legal_move_raises_error(self) -> None:
        """Raise error when no legal move connects positions."""
        board1 = chess.Board()
        board2 = chess.Board()
        board2.push_uci("e2e4")
        board2.push_uci("e7e5")  # Two moves - can't get there in one

        with pytest.raises(VerificationError, match="No legal move found"):
            find_move_between_positions(board1, board2)


class TestVerifyGameIntegrity:
    """Tests for the verify_game_integrity function."""

    def test_valid_game_passes(self) -> None:
        """A valid game from standard start should pass all checks."""
        positions = make_game_from_moves(["e2e4", "e7e5", "g1f3"])
        verify_game_integrity(positions, game_idx=0)  # Should not raise

    def test_empty_game_fails(self) -> None:
        """Empty game should fail verification."""
        with pytest.raises(VerificationError, match="Empty game"):
            verify_game_integrity([], game_idx=0)

    def test_chess960_detected(self) -> None:
        """Non-standard starting position should be detected."""
        # Chess960 starting position (different from standard)
        fen960 = "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w KQkq - 0 1"
        positions = [make_position(fen960)]
        with pytest.raises(VerificationError, match="Chess960"):
            verify_game_integrity(positions, game_idx=0)

    def test_disconnected_positions_detected(self) -> None:
        """Positions not connected by a legal move should be detected."""
        # Two positions that require two moves to connect
        board = chess.Board()
        pos0 = make_position(board.fen())
        board.push_uci("e2e4")
        board.push_uci("e7e5")  # Skip black's response
        pos1 = make_position(board.fen())

        with pytest.raises(VerificationError, match="No legal move connects"):
            verify_game_integrity([pos0, pos1], game_idx=0)

    def test_bad_legal_moves_detected(self) -> None:
        """Position with wrong legal moves should be detected."""
        pos = make_position(chess.STARTING_FEN)
        # Corrupt the legal moves
        pos = LeelaPositionData(
            fen=pos.fen,
            legal_moves=[("z1z2", 1.0)],  # Fake move
            invariance_info=pos.invariance_info,
            result=pos.result,
            root_q=pos.root_q,
            root_d=pos.root_d,
            best_q=pos.best_q,
            best_d=pos.best_d,
            played_q=pos.played_q,
            played_d=pos.played_d,
            orig_q=pos.orig_q,
            orig_d=pos.orig_d,
            best_move_uci=pos.best_move_uci,
        )
        with pytest.raises(VerificationError, match="Legal moves mismatch"):
            verify_game_integrity([pos], game_idx=0)

    def test_dangerous_invariance_detected(self) -> None:
        """Position with dangerous invariance bits should be detected."""
        pos = make_position(chess.STARTING_FEN)
        pos = LeelaPositionData(
            fen=pos.fen,
            legal_moves=pos.legal_moves,
            invariance_info=0x01,  # flip_transform
            result=pos.result,
            root_q=pos.root_q,
            root_d=pos.root_d,
            best_q=pos.best_q,
            best_d=pos.best_d,
            played_q=pos.played_q,
            played_d=pos.played_d,
            orig_q=pos.orig_q,
            orig_d=pos.orig_d,
            best_move_uci=pos.best_move_uci,
        )
        with pytest.raises(VerificationError, match="Dangerous invariance"):
            verify_game_integrity([pos], game_idx=0)


class TestComputeGameResult:
    """Tests for the compute_game_result function."""

    def test_win_from_root_q(self) -> None:
        """Positive root_q at terminal indicates win."""
        positions = make_game_from_moves(["e2e4"])
        # Override terminal position's root_q
        positions[-1] = make_position(positions[-1].fen, root_q=0.8, root_d=0.1)

        results = compute_game_result(positions)

        # Terminal position: win for side to move
        assert results[1] == 1
        # Position 0: one ply before terminal, opposite result
        assert results[0] == -1

    def test_loss_from_root_q(self) -> None:
        """Negative root_q at terminal indicates loss."""
        positions = make_game_from_moves(["e2e4"])
        positions[-1] = make_position(positions[-1].fen, root_q=-0.9, root_d=0.05)

        results = compute_game_result(positions)

        assert results[1] == -1
        assert results[0] == 1

    def test_draw_from_root_d(self) -> None:
        """High root_d at terminal indicates draw."""
        positions = make_game_from_moves(["e2e4"])
        positions[-1] = make_position(positions[-1].fen, root_q=0.0, root_d=0.9)

        results = compute_game_result(positions)

        assert results[0] == 0
        assert results[1] == 0

    def test_result_alternates(self) -> None:
        """Game result alternates sign every ply."""
        positions = make_game_from_moves(["e2e4", "e7e5", "g1f3", "b8c6"])
        # Set terminal to win
        positions[-1] = make_position(positions[-1].fen, root_q=0.7, root_d=0.1)

        results = compute_game_result(positions)

        assert results[4] == 1   # Terminal: win
        assert results[3] == -1  # One ply before
        assert results[2] == 1
        assert results[1] == -1
        assert results[0] == 1

    def test_empty_returns_empty(self) -> None:
        """Empty position list returns empty results."""
        assert compute_game_result([]) == []
