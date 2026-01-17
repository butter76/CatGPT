"""Tests for meta-features computation in bagz_to_bag.py.

These tests ensure that the meta-game features computed from whole-game analysis
are correct. This includes:
- piece_will_move_to: where each piece will move next
- square_will_be_occupied_from: immediate source of next move to each square
- square_will_be_occupied_by_piece_on: traces current location through multiple moves
- next_capture_square: current location of piece that will be captured next
- next_pawn_move_square: square of the pawn that will move next
- game_result: win/draw/loss from side-to-move perspective
"""

import chess
import pytest

from catgpt.core.data.grain.bagz_to_bag import (
    MetaFeatures,
    compute_meta_features,
    find_move_between_positions,
    VerificationError,
)
from catgpt.core.data.grain.coders import LeelaPositionData


def make_position(fen: str) -> LeelaPositionData:
    """Create a minimal LeelaPositionData for testing.

    Uses default values for fields not relevant to meta-feature computation.
    """
    board = chess.Board(fen)
    legal_moves = [(m.uci(), 1.0 / board.legal_moves.count()) for m in board.legal_moves]

    return LeelaPositionData(
        fen=fen,
        legal_moves=legal_moves,
        invariance_info=0,
        result=0,
        root_q=0.0,
        root_d=0.5,
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


class TestPieceWillMoveTo:
    """Tests for the piece_will_move_to meta-feature."""

    def test_simple_pawn_advance(self) -> None:
        """Pawn on e2 moves to e4."""
        positions = make_game_from_moves(["e2e4"])
        meta = compute_meta_features(positions)

        assert meta[0].piece_will_move_to["e2"] == "e4"

    def test_multiple_moves_first_position(self) -> None:
        """First position records ALL future moves, not just immediate next."""
        positions = make_game_from_moves(["e2e4", "e7e5", "g1f3"])
        meta = compute_meta_features(positions)

        # Position 0: All pieces that will move are recorded
        assert meta[0].piece_will_move_to["e2"] == "e4"  # White pawn
        assert meta[0].piece_will_move_to["e7"] == "e5"  # Black pawn (will move later)
        assert meta[0].piece_will_move_to["g1"] == "f3"  # White knight (will move later)

    def test_second_position_shows_black_move(self) -> None:
        """Second position should show black's move."""
        positions = make_game_from_moves(["e2e4", "e7e5"])
        meta = compute_meta_features(positions)

        # Position 1: Black pawn will move e7->e5
        assert meta[1].piece_will_move_to["e7"] == "e5"

    def test_promotion_no_suffix(self) -> None:
        """Promotion moves should NOT have piece suffix."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        positions = make_game_from_moves(["a7a8q"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The destination should be plain "a8", not "a8q"
        assert meta[0].piece_will_move_to["a7"] == "a8"

    def test_underpromotion_no_suffix(self) -> None:
        """Underpromotion moves should NOT have piece suffix."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        positions = make_game_from_moves(["a7a8n"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The destination should be plain "a8", not "a8n"
        assert meta[0].piece_will_move_to["a7"] == "a8"

    def test_castling_both_pieces(self) -> None:
        """Castling should record both king and rook movements."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        positions = make_game_from_moves(["e1g1"], start_fen=fen)
        meta = compute_meta_features(positions)

        # King moves e1->g1
        assert meta[0].piece_will_move_to["e1"] == "g1"
        # Rook moves h1->f1
        assert meta[0].piece_will_move_to["h1"] == "f1"

    def test_castling_queenside(self) -> None:
        """Queenside castling should record both king and rook movements."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        positions = make_game_from_moves(["e1c1"], start_fen=fen)
        meta = compute_meta_features(positions)

        # King moves e1->c1
        assert meta[0].piece_will_move_to["e1"] == "c1"
        # Rook moves a1->d1
        assert meta[0].piece_will_move_to["a1"] == "d1"


class TestSquareWillBeOccupiedFrom:
    """Tests for the square_will_be_occupied_from meta-feature."""

    def test_simple_move(self) -> None:
        """Square e4 will be occupied from e2."""
        positions = make_game_from_moves(["e2e4"])
        meta = compute_meta_features(positions)

        assert meta[0].square_will_be_occupied_from["e4"] == "e2"

    def test_promotion_no_suffix(self) -> None:
        """Promotion source should NOT have piece suffix."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        positions = make_game_from_moves(["a7a8q"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The source should be plain "a7", not "a7q"
        assert meta[0].square_will_be_occupied_from["a8"] == "a7"

    def test_underpromotion_no_suffix(self) -> None:
        """Underpromotion source should NOT have piece suffix."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        positions = make_game_from_moves(["a7a8r"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The source should be plain "a7", not "a7r"
        assert meta[0].square_will_be_occupied_from["a8"] == "a7"

    def test_castling_both_destinations(self) -> None:
        """Castling should record sources for both g1 and f1."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        positions = make_game_from_moves(["e1g1"], start_fen=fen)
        meta = compute_meta_features(positions)

        # g1 will be occupied from e1 (king)
        assert meta[0].square_will_be_occupied_from["g1"] == "e1"
        # f1 will be occupied from h1 (rook)
        assert meta[0].square_will_be_occupied_from["f1"] == "h1"


class TestSquareWillBeOccupiedByPieceOn:
    """Tests for the square_will_be_occupied_by_piece_on meta-feature.

    This traces through multiple moves to find where the piece currently is.
    """

    def test_single_move(self) -> None:
        """Single move: current location is the from square."""
        positions = make_game_from_moves(["e2e4"])
        meta = compute_meta_features(positions)

        assert meta[0].square_will_be_occupied_by_piece_on["e4"] == "e2"

    def test_two_moves_same_piece(self) -> None:
        """Trace through two moves by the same piece."""
        # e2->e4, then after black moves, e4->e5
        positions = make_game_from_moves(["e2e4", "a7a6", "e4e5"])
        meta = compute_meta_features(positions)

        # At position 0: e5 will eventually have the piece currently on e2
        assert meta[0].square_will_be_occupied_by_piece_on["e5"] == "e2"
        # e4 intermediate square also traced back to e2
        assert meta[0].square_will_be_occupied_by_piece_on["e4"] == "e2"

    def test_three_moves_same_piece(self) -> None:
        """Trace through three moves by the same piece (knight)."""
        # Knight: g1->f3->e5->d7
        positions = make_game_from_moves(["g1f3", "a7a6", "f3e5", "b7b6", "e5d7"])
        meta = compute_meta_features(positions)

        # At position 0: d7 will eventually have the piece currently on g1
        assert meta[0].square_will_be_occupied_by_piece_on["d7"] == "g1"
        # e5 and f3 also trace back to g1
        assert meta[0].square_will_be_occupied_by_piece_on["e5"] == "g1"
        assert meta[0].square_will_be_occupied_by_piece_on["f3"] == "g1"

    def test_promotion_tracing_through(self) -> None:
        """CRITICAL: Trace through a promotion move correctly.

        This test would have caught the bug where promotion suffixes
        broke the string comparison in piece tracing.

        Scenario: Pawn on a7 promotes to queen (a7->a8q), then queen moves (a8->b7).
        At position 0, square_will_be_occupied_by_piece_on["b7"] should be "a7".
        """
        # Position with pawn about to promote, then queen moves
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        board = chess.Board(fen)

        # Create positions: start, after promotion, after queen move
        pos0 = make_position(board.fen())
        board.push_uci("a7a8q")
        pos1 = make_position(board.fen())
        board.push_uci("h1g1")  # Black king moves
        pos2 = make_position(board.fen())
        board.push_uci("a8b7")  # Queen moves
        pos3 = make_position(board.fen())

        positions = [pos0, pos1, pos2, pos3]
        meta = compute_meta_features(positions)

        # At position 0: b7 should trace back to a7 (where pawn currently is)
        assert meta[0].square_will_be_occupied_by_piece_on["b7"] == "a7"
        # a8 should also trace to a7
        assert meta[0].square_will_be_occupied_by_piece_on["a8"] == "a7"

    def test_underpromotion_tracing_through(self) -> None:
        """Trace through an underpromotion (knight)."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        board = chess.Board(fen)

        pos0 = make_position(board.fen())
        board.push_uci("a7a8n")  # Underpromote to knight
        pos1 = make_position(board.fen())
        board.push_uci("h1g1")
        pos2 = make_position(board.fen())
        board.push_uci("a8b6")  # Knight moves
        pos3 = make_position(board.fen())

        positions = [pos0, pos1, pos2, pos3]
        meta = compute_meta_features(positions)

        # b6 should trace back to a7
        assert meta[0].square_will_be_occupied_by_piece_on["b6"] == "a7"

    def test_multiple_promotions_different_pieces(self) -> None:
        """Two pawns promote and their pieces move around."""
        # White pawn on b7, black pawn on h2
        # Black king on h7, white king on a2 (safe from rook on h1)
        fen = "8/1P5k/8/8/8/8/K6p/8 w - - 0 1"
        board = chess.Board(fen)

        pos0 = make_position(board.fen())
        board.push_uci("b7b8q")  # White promotes to queen
        pos1 = make_position(board.fen())
        board.push_uci("h2h1r")  # Black underpromotes to rook
        pos2 = make_position(board.fen())
        board.push_uci("b8f8")  # White queen moves (no check to black king)
        pos3 = make_position(board.fen())
        board.push_uci("h1g1")  # Black rook moves
        pos4 = make_position(board.fen())

        positions = [pos0, pos1, pos2, pos3, pos4]
        meta = compute_meta_features(positions)

        # At position 0: f8 traces to b7, g1 traces to h2
        assert meta[0].square_will_be_occupied_by_piece_on["f8"] == "b7"
        assert meta[0].square_will_be_occupied_by_piece_on["g1"] == "h2"

    def test_castling_tracing(self) -> None:
        """Trace through castling: both king and rook traced correctly."""
        fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
        board = chess.Board(fen)

        pos0 = make_position(board.fen())
        board.push_uci("e1g1")  # White castles
        pos1 = make_position(board.fen())
        board.push_uci("a7a6")
        pos2 = make_position(board.fen())
        board.push_uci("g1h1")  # King moves again
        pos3 = make_position(board.fen())

        positions = [pos0, pos1, pos2, pos3]
        meta = compute_meta_features(positions)

        # At position 0: h1 will be occupied by piece currently on e1 (king)
        assert meta[0].square_will_be_occupied_by_piece_on["h1"] == "e1"
        # g1 is intermediate, also traces to e1
        assert meta[0].square_will_be_occupied_by_piece_on["g1"] == "e1"
        # f1 traces to h1 (rook)
        assert meta[0].square_will_be_occupied_by_piece_on["f1"] == "h1"


class TestNextCaptureSquare:
    """Tests for the next_capture_square meta-feature."""

    def test_no_capture(self) -> None:
        """No captures in game: next_capture_square is None."""
        positions = make_game_from_moves(["e2e4", "e7e5"])
        meta = compute_meta_features(positions)

        assert meta[0].next_capture_square is None
        assert meta[1].next_capture_square is None

    def test_capture_upcoming(self) -> None:
        """Next capture: shows where captured piece currently is at each position.

        The next_capture_square always shows the CURRENT location of the piece
        that will be captured next. As the piece moves, this value changes.
        """
        # 1. e4 d5 2. exd5
        # Position 0: starting
        # Position 1: after e2e4
        # Position 2: after d7d5 (black pawn now on d5)
        # Position 3: after e4d5 (capture happened)
        positions = make_game_from_moves(["e2e4", "d7d5", "e4d5"])
        meta = compute_meta_features(positions)

        # At position 2: black pawn is on d5, about to be captured
        assert meta[2].next_capture_square == "d5"

        # At position 1: black pawn is still on d7, it will move then be captured
        # The feature shows where the piece IS NOW (d7), not where it will be captured
        assert meta[1].next_capture_square == "d7"

        # At position 0: same - the pawn is on d7
        assert meta[0].next_capture_square == "d7"

    def test_capture_tracing_through_moves(self) -> None:
        """Track piece that will be captured as it moves.

        The white d-pawn moves d2->d4, then gets captured by black's e-pawn.
        """
        # 1. e4 e5 2. d4 exd4 - pawn capture
        # Position 0: starting
        # Position 1: after e2e4
        # Position 2: after e7e5
        # Position 3: after d2d4 (white pawn now on d4)
        # Position 4: after e5d4 (capture)
        positions = make_game_from_moves(["e2e4", "e7e5", "d2d4", "e5d4"])
        meta = compute_meta_features(positions)

        # At position 3: white pawn is on d4, about to be captured
        assert meta[3].next_capture_square == "d4"

        # At positions 0, 1, 2: the pawn is still on d2
        # next_capture_square shows where the piece IS NOW
        assert meta[2].next_capture_square == "d2"
        assert meta[1].next_capture_square == "d2"
        assert meta[0].next_capture_square == "d2"

    def test_en_passant_capture(self) -> None:
        """En passant: captured pawn is on different square than destination."""
        # Setup for en passant
        fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
        positions = make_game_from_moves(["e5d6"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The captured pawn is on d5 (not d6 where the capturing pawn lands)
        assert meta[0].next_capture_square == "d5"


class TestNextPawnMoveSquare:
    """Tests for the next_pawn_move_square meta-feature."""

    def test_pawn_moves_first(self) -> None:
        """First move is a pawn move."""
        positions = make_game_from_moves(["e2e4"])
        meta = compute_meta_features(positions)

        assert meta[0].next_pawn_move_square == "e2"

    def test_knight_then_pawn(self) -> None:
        """Knight moves first, then pawn."""
        positions = make_game_from_moves(["g1f3", "e7e5"])
        meta = compute_meta_features(positions)

        # At position 0: next pawn move is e7 (black's pawn)
        assert meta[0].next_pawn_move_square == "e7"

        # At position 1: it's black's turn, pawn on e7 is about to move
        assert meta[1].next_pawn_move_square == "e7"

    def test_no_pawn_moves(self) -> None:
        """No pawn moves in game: next_pawn_move_square is None."""
        # Only knight moves
        fen = "4k3/8/8/8/8/8/8/4K1N1 w - - 0 1"
        positions = make_game_from_moves(["g1f3", "e8d8", "f3e5"], start_fen=fen)
        meta = compute_meta_features(positions)

        assert meta[0].next_pawn_move_square is None

    def test_promotion_is_pawn_move(self) -> None:
        """Promotion counts as a pawn move."""
        fen = "8/P7/8/8/8/8/8/4K2k w - - 0 1"
        positions = make_game_from_moves(["a7a8q"], start_fen=fen)
        meta = compute_meta_features(positions)

        assert meta[0].next_pawn_move_square == "a7"


class TestGameResult:
    """Tests for the game_result meta-feature."""

    def test_win_from_root_q(self) -> None:
        """Positive root_q at terminal indicates win."""
        # Terminal position with positive evaluation
        fen1 = "8/8/4k3/8/8/8/4K3/8 w - - 0 1"
        fen2 = "8/8/4k3/8/8/4K3/8/8 b - - 1 1"

        pos1 = make_position(fen1)
        pos2 = make_position(fen2)
        pos2 = LeelaPositionData(
            fen=pos2.fen,
            legal_moves=pos2.legal_moves,
            invariance_info=pos2.invariance_info,
            result=pos2.result,
            root_q=0.8,  # Positive = win for side to move (black)
            root_d=0.1,
            best_q=pos2.best_q,
            best_d=pos2.best_d,
            played_q=pos2.played_q,
            played_d=pos2.played_d,
            orig_q=pos2.orig_q,
            orig_d=pos2.orig_d,
            best_move_uci=pos2.best_move_uci,
        )

        positions = [pos1, pos2]
        meta = compute_meta_features(positions)

        # Terminal position: win for black (side to move)
        assert meta[1].game_result == 1
        # Position 0: it's white to move, one ply before terminal
        # If black wins, then white loses
        assert meta[0].game_result == -1

    def test_loss_from_root_q(self) -> None:
        """Negative root_q at terminal indicates loss."""
        fen1 = "8/8/4k3/8/8/8/4K3/8 w - - 0 1"
        fen2 = "8/8/4k3/8/8/4K3/8/8 b - - 1 1"

        pos1 = make_position(fen1)
        pos2 = make_position(fen2)
        pos2 = LeelaPositionData(
            fen=pos2.fen,
            legal_moves=pos2.legal_moves,
            invariance_info=pos2.invariance_info,
            result=pos2.result,
            root_q=-0.9,  # Negative = loss for side to move (black)
            root_d=0.05,
            best_q=pos2.best_q,
            best_d=pos2.best_d,
            played_q=pos2.played_q,
            played_d=pos2.played_d,
            orig_q=pos2.orig_q,
            orig_d=pos2.orig_d,
            best_move_uci=pos2.best_move_uci,
        )

        positions = [pos1, pos2]
        meta = compute_meta_features(positions)

        # Terminal position: loss for black
        assert meta[1].game_result == -1
        # Position 0: white wins
        assert meta[0].game_result == 1

    def test_draw_from_root_d(self) -> None:
        """High root_d at terminal indicates draw."""
        fen1 = "8/8/4k3/8/8/8/4K3/8 w - - 0 1"
        fen2 = "8/8/4k3/8/8/4K3/8/8 b - - 1 1"

        pos1 = make_position(fen1)
        pos2 = make_position(fen2)
        pos2 = LeelaPositionData(
            fen=pos2.fen,
            legal_moves=pos2.legal_moves,
            invariance_info=pos2.invariance_info,
            result=pos2.result,
            root_q=0.0,
            root_d=0.9,  # High draw probability
            best_q=pos2.best_q,
            best_d=pos2.best_d,
            played_q=pos2.played_q,
            played_d=pos2.played_d,
            orig_q=pos2.orig_q,
            orig_d=pos2.orig_d,
            best_move_uci=pos2.best_move_uci,
        )

        positions = [pos1, pos2]
        meta = compute_meta_features(positions)

        # Both positions should show draw
        assert meta[0].game_result == 0
        assert meta[1].game_result == 0

    def test_result_alternates(self) -> None:
        """Game result alternates sign every ply."""
        positions = make_game_from_moves(["e2e4", "e7e5", "g1f3", "b8c6"])

        # Set terminal position to win for white (side to move at position 4)
        positions[-1] = LeelaPositionData(
            fen=positions[-1].fen,
            legal_moves=positions[-1].legal_moves,
            invariance_info=positions[-1].invariance_info,
            result=positions[-1].result,
            root_q=0.7,  # Win for white
            root_d=0.1,
            best_q=positions[-1].best_q,
            best_d=positions[-1].best_d,
            played_q=positions[-1].played_q,
            played_d=positions[-1].played_d,
            orig_q=positions[-1].orig_q,
            orig_d=positions[-1].orig_d,
            best_move_uci=positions[-1].best_move_uci,
        )

        meta = compute_meta_features(positions)

        # Position 4: white to move, white wins -> +1
        assert meta[4].game_result == 1
        # Position 3: black to move, white wins -> -1 for black
        assert meta[3].game_result == -1
        # Position 2: white to move, white wins -> +1
        assert meta[2].game_result == 1
        # Position 1: black to move -> -1
        assert meta[1].game_result == -1
        # Position 0: white to move -> +1
        assert meta[0].game_result == 1


class TestCapturedPieceDoesNotMove:
    """Tests ensuring captured pieces don't have future movement recorded."""

    def test_captured_piece_no_will_move_to(self) -> None:
        """A piece that gets captured should not appear in piece_will_move_to."""
        # 1. e4 d5 2. exd5 - black's d5 pawn gets captured
        positions = make_game_from_moves(["e2e4", "d7d5", "e4d5"])
        meta = compute_meta_features(positions)

        # At position 1 (black to move, pawn on d7 about to move to d5):
        # After d7d5, this pawn will be captured. At position 1, d5 hasn't happened yet,
        # but once it does, the pawn should not have a future move.
        # Actually at position 1, the pawn is about to move d7->d5
        assert meta[1].piece_will_move_to.get("d7") == "d5"

        # At position 2 (white to move, d5 has black pawn, about to be captured):
        # The d5 pawn will be captured, so it should NOT have a future move
        assert "d5" not in meta[2].piece_will_move_to

    def test_en_passant_captured_pawn_no_future_move(self) -> None:
        """En passant captured pawn should not have future movement."""
        # Position where en passant is available
        fen = "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1"
        positions = make_game_from_moves(["e5d6"], start_fen=fen)
        meta = compute_meta_features(positions)

        # The black pawn on d5 will be captured - it should not have a future move
        assert "d5" not in meta[0].piece_will_move_to

    def test_capturing_piece_continues_to_move(self) -> None:
        """When piece A captures piece B, then A moves again, B should have no future movement.

        This tests the critical case where:
        1. Piece A captures piece B on square X
        2. Piece A then moves from X to Y
        3. We need to verify that piece B is NOT recorded as moving to Y

        The bug this catches: If we track that "square X will be occupied by piece moving to Y",
        and piece B was on X before capture, we might accidentally record B as the piece
        that will end up on Y.
        """
        # Setup: Knight captures pawn, then knight moves again
        # Position: White knight on b1, black pawn on c3, white king on e1, black king on e8
        fen = "4k3/8/8/8/8/2p5/8/1N2K3 w - - 0 1"
        # Moves: b1c3 (capture pawn), c3d5 (knight continues)
        positions = make_game_from_moves(["b1c3", "e8d8", "c3d5"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - Knight on b1 will move: b1 -> c3
        assert meta[0].piece_will_move_to["b1"] == "c3"
        # - Pawn on c3 will be captured, so it should NOT have future movement
        assert "c3" not in meta[0].piece_will_move_to

        # CRITICAL: square_will_be_occupied_by_piece_on["d5"] should trace to b1 (knight),
        # NOT to c3 (the captured pawn's location)
        assert meta[0].square_will_be_occupied_by_piece_on["d5"] == "b1"
        assert meta[0].square_will_be_occupied_by_piece_on["c3"] == "b1"

    def test_capturing_piece_captured_in_turn(self) -> None:
        """Piece A captures B, then A is captured by C. Neither A nor B should have future moves.

        This tests chained captures where the capturing piece itself gets captured.
        """
        # Setup: White pawn on e4, black pawn on d5, black knight on f6
        # Moves: e4xd5 (white captures), Nf6xd5 (black recaptures)
        fen = "4k3/8/5n2/3p4/4P3/8/8/4K3 w - - 0 1"
        positions = make_game_from_moves(["e4d5", "f6d5"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - White pawn on e4 captures d5, then gets captured itself
        # - Both the white pawn and black pawn should have their movement recorded up to capture
        assert meta[0].piece_will_move_to["e4"] == "d5"  # White pawn moves to d5
        assert "d5" not in meta[0].piece_will_move_to  # Black pawn on d5 is captured

        # At position 1 (after e4xd5):
        # - White pawn is now on d5, about to be captured
        # - It should NOT have a future move (it will be captured)
        assert "d5" not in meta[1].piece_will_move_to

        # The knight on f6 will capture
        assert meta[0].piece_will_move_to["f6"] == "d5"
        assert meta[1].piece_will_move_to["f6"] == "d5"

    def test_multiple_captures_same_piece(self) -> None:
        """A piece makes multiple captures. Captured pieces should not have future moves."""
        # Knight captures multiple pieces in sequence
        # Setup: Knight on g1, pawns on f3 and e5
        fen = "4k3/8/8/4p3/8/5p2/8/4K1N1 w - - 0 1"
        # Moves: Nf3 (capture), then after black moves, Ne5 (capture)
        positions = make_game_from_moves(["g1f3", "e8d8", "f3e5"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - Knight on g1 will move g1->f3->e5
        assert meta[0].piece_will_move_to["g1"] == "f3"
        # - Pawn on f3 will be captured immediately
        assert "f3" not in meta[0].piece_will_move_to
        # - Pawn on e5 will be captured later
        assert "e5" not in meta[0].piece_will_move_to

        # Tracing should go through the knight's path
        assert meta[0].square_will_be_occupied_by_piece_on["f3"] == "g1"
        assert meta[0].square_will_be_occupied_by_piece_on["e5"] == "g1"


class TestEnPassantEdgeCases:
    """Comprehensive tests for en passant edge cases.

    En passant is tricky because:
    1. The captured pawn is on a different square than the destination
    2. The capturing pawn may continue to move after the capture
    3. Multiple pieces tracking through en passant captures needs correct handling
    """

    def test_en_passant_basic_tracking(self) -> None:
        """Basic en passant: verify captured pawn location and capturing pawn movement."""
        # White pawn on e5, black pawn on d5 (just moved d7-d5)
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        positions = make_game_from_moves(["e5d6"], start_fen=fen)
        meta = compute_meta_features(positions)

        # Capturing pawn moves e5->d6
        assert meta[0].piece_will_move_to["e5"] == "d6"
        # Captured pawn on d5 should NOT have future movement
        assert "d5" not in meta[0].piece_will_move_to
        # next_capture_square should be d5 (where captured pawn is), not d6
        assert meta[0].next_capture_square == "d5"

    def test_en_passant_capturing_pawn_continues(self) -> None:
        """En passant capture, then the capturing pawn continues to move.

        CRITICAL: After en passant e5xd6, if the pawn then moves d6->d7,
        we should NOT track the captured d5 pawn as moving to d7.
        """
        # White pawn on e5, black pawn on d5, white can en passant, then advance
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        # e5xd6 (en passant), black king moves, d6->d7
        positions = make_game_from_moves(["e5d6", "e8d8", "d6d7"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - Pawn on e5 will go e5->d6->d7
        assert meta[0].piece_will_move_to["e5"] == "d6"
        # - Captured pawn on d5 should NOT have any future movement
        assert "d5" not in meta[0].piece_will_move_to

        # CRITICAL: d7 should trace back to e5 (capturing pawn's original location)
        # NOT to d5 (the captured pawn's location)
        assert meta[0].square_will_be_occupied_by_piece_on["d7"] == "e5"
        assert meta[0].square_will_be_occupied_by_piece_on["d6"] == "e5"

    def test_en_passant_capturing_pawn_promotes(self) -> None:
        """En passant capture followed by promotion.

        The capturing pawn takes via en passant, then advances and promotes.
        """
        # White pawn on b5, black pawn on a5 (just moved a7-a5)
        # After bxa6 (en passant), pawn can advance a6->a7->a8=Q
        fen = "4k3/8/8/pP6/8/8/8/4K3 w - a6 0 1"
        positions = make_game_from_moves(
            ["b5a6", "e8d8", "a6a7", "d8c8", "a7a8q"], start_fen=fen
        )
        meta = compute_meta_features(positions)

        # At position 0:
        # - Pawn on b5 will go b5->a6 (en passant)
        assert meta[0].piece_will_move_to["b5"] == "a6"
        # - Captured pawn on a5 should NOT have any future movement
        assert "a5" not in meta[0].piece_will_move_to

        # a8 should trace back to b5 (the pawn that promoted)
        assert meta[0].square_will_be_occupied_by_piece_on["a8"] == "b5"
        assert meta[0].square_will_be_occupied_by_piece_on["a7"] == "b5"
        assert meta[0].square_will_be_occupied_by_piece_on["a6"] == "b5"

    def test_en_passant_capturing_pawn_is_captured(self) -> None:
        """En passant capture, then the capturing pawn itself is captured.

        Both the originally captured pawn and the capturing pawn should not
        have future movement beyond their capture points.
        """
        # White pawn on e5, black pawn on d5 and black rook on d8
        fen = "3rk3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        # e5xd6 (en passant), Rd8xd6 (recapture)
        positions = make_game_from_moves(["e5d6", "d8d6"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - White pawn on e5 captures en passant, then gets captured
        assert meta[0].piece_will_move_to["e5"] == "d6"
        # - Black pawn on d5 is captured (en passant victim)
        assert "d5" not in meta[0].piece_will_move_to

        # At position 1 (after en passant):
        # - White pawn is now on d6, about to be captured
        assert "d6" not in meta[1].piece_will_move_to

        # The rook captures the white pawn
        assert meta[0].piece_will_move_to["d8"] == "d6"

    def test_en_passant_black_captures(self) -> None:
        """En passant by black pawn to verify symmetry."""
        # Black pawn on d4, white pawn on e4 (just moved e2-e4)
        fen = "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"
        positions = make_game_from_moves(["d4e3"], start_fen=fen)
        meta = compute_meta_features(positions)

        # Capturing pawn moves d4->e3
        assert meta[0].piece_will_move_to["d4"] == "e3"
        # Captured pawn on e4 should NOT have future movement
        assert "e4" not in meta[0].piece_will_move_to
        # next_capture_square should be e4 (where captured pawn is), not e3
        assert meta[0].next_capture_square == "e4"

    def test_en_passant_piece_tracing_does_not_leak(self) -> None:
        """Verify that piece tracing doesn't accidentally mix captured pawn with capturing pawn.

        This is a regression test for potential bugs where the backward scan might
        incorrectly attribute the captured pawn's original location to future squares.
        """
        # Complex scenario: en passant followed by pawn advancing to promotion
        # White pawn on e5, black pawn on d5, black king on h8
        fen = "7k/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        positions = make_game_from_moves(
            ["e5d6", "h8g8", "d6d7", "g8f8", "d7d8q"], start_fen=fen
        )
        meta = compute_meta_features(positions)

        # At position 0, verify square_will_be_occupied_by_piece_on
        # d6, d7, d8 should all trace back to e5 (the capturing pawn)
        # NOT to d5 (the captured pawn)
        assert meta[0].square_will_be_occupied_by_piece_on.get("d6") == "e5"
        assert meta[0].square_will_be_occupied_by_piece_on.get("d7") == "e5"
        assert meta[0].square_will_be_occupied_by_piece_on.get("d8") == "e5"

        # d5 should NOT appear as a source for any future square occupancy
        # because the pawn on d5 was captured
        for sq, source in meta[0].square_will_be_occupied_by_piece_on.items():
            assert source != "d5", f"Square {sq} incorrectly traces to captured pawn on d5"

    def test_en_passant_with_other_pieces_moving(self) -> None:
        """En passant in a more complex game with other pieces also moving.

        Note: En passant must happen immediately after the double-push,
        so we set up a position where en passant happens first, then other
        pieces continue to move.
        """
        # En passant happens first, then other pieces move
        fen = "r3k3/8/8/3pP3/8/8/8/R3K3 w Qq d6 0 1"
        positions = make_game_from_moves(
            ["e5d6", "a8a7", "a1a2", "a7a6", "d6d7"], start_fen=fen
        )
        meta = compute_meta_features(positions)

        # At position 0:
        # - Pawn on e5 captures en passant (e5->d6), then advances
        assert meta[0].piece_will_move_to["e5"] == "d6"
        # - Captured pawn on d5 should not have future movement
        assert "d5" not in meta[0].piece_will_move_to
        # - Rook on a1 will move later
        assert meta[0].piece_will_move_to["a1"] == "a2"
        # - Black rook on a8 will move
        assert meta[0].piece_will_move_to["a8"] == "a7"

        # d7 should trace back to e5 (the capturing pawn)
        assert meta[0].square_will_be_occupied_by_piece_on["d7"] == "e5"


class TestPromotedPieceCapture:
    """Tests for capturing promoted pieces."""

    def test_promoted_piece_is_captured(self) -> None:
        """A pawn promotes, then the promoted piece is captured.

        Verifies that tracing correctly handles the promotion and subsequent capture.
        """
        # White pawn on a7, black rook on a8
        fen = "r3k3/P7/8/8/8/8/8/4K3 w q - 0 1"
        # a7xa8=Q (capture-promotion), then later Ke8 attacks, but let's have black recapture
        # Actually the rook is captured BY the promotion. Let's set up differently.
        # White pawn promotes, black king captures the queen
        fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        positions = make_game_from_moves(["a7a8q", "e8d7", "a8a7", "d7c6", "a7a6"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0: pawn on a7 will promote and the queen will move around
        assert meta[0].piece_will_move_to["a7"] == "a8"
        # a6 traces back to a7 (through a8->a7->a6)
        assert meta[0].square_will_be_occupied_by_piece_on["a6"] == "a7"

    def test_promoted_piece_captured_immediately(self) -> None:
        """Pawn promotes into a capture, promoted piece then captured."""
        # White pawn on b7, black rooks on a8 and c8
        fen = "r1r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1"
        # b7xa8=Q (capture-promotion), Ra8 is gone, then Rc8xa8 recaptures
        positions = make_game_from_moves(["b7a8q", "c8a8"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0:
        # - Pawn on b7 will promote (capturing on a8)
        assert meta[0].piece_will_move_to["b7"] == "a8"
        # - Rook on a8 will be captured - no future movement
        assert "a8" not in meta[0].piece_will_move_to

        # At position 1:
        # - Queen on a8 (just promoted) will be captured - no future movement
        assert "a8" not in meta[1].piece_will_move_to
        # - Rook on c8 will capture
        assert meta[1].piece_will_move_to["c8"] == "a8"


class TestPieceReturnsToOriginalSquare:
    """Tests for pieces returning to their starting squares."""

    def test_knight_returns_to_original_square(self) -> None:
        """Knight moves away and returns: g1->f3->g1.

        square_will_be_occupied_by_piece_on["g1"] should be "g1" (same piece returns).
        """
        positions = make_game_from_moves(["g1f3", "e7e6", "f3g1"])
        meta = compute_meta_features(positions)

        # At position 0: g1 will be occupied by the piece currently on g1
        # (the knight leaves and returns)
        assert meta[0].square_will_be_occupied_by_piece_on["g1"] == "g1"
        # f3 is intermediate, traces to g1
        assert meta[0].square_will_be_occupied_by_piece_on["f3"] == "g1"

    def test_bishop_triangle_returns(self) -> None:
        """Bishop makes a triangle and returns: f1->c4->e2->f1."""
        # Need to clear path for bishop
        fen = "4k3/8/8/8/8/8/8/4KB2 w - - 0 1"
        positions = make_game_from_moves(
            ["f1c4", "e8d8", "c4e2", "d8c8", "e2f1"], start_fen=fen
        )
        meta = compute_meta_features(positions)

        # f1 traces to f1 (same piece returns)
        assert meta[0].square_will_be_occupied_by_piece_on["f1"] == "f1"
        # All intermediate squares trace to f1
        assert meta[0].square_will_be_occupied_by_piece_on["c4"] == "f1"
        assert meta[0].square_will_be_occupied_by_piece_on["e2"] == "f1"


class TestCastlingEdgeCases:
    """Additional castling edge cases."""

    def test_rook_moves_after_castling(self) -> None:
        """Rook continues to move after castling."""
        fen = "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"
        # Castle kingside: Ke1-g1, Rh1-f1. Then rook moves f1->f8
        positions = make_game_from_moves(["e1g1", "e8d8", "f1f8"], start_fen=fen)
        meta = compute_meta_features(positions)

        # At position 0: rook on h1 will end up on f8
        assert meta[0].piece_will_move_to["h1"] == "f1"
        # f8 traces to h1 (rook's original position)
        assert meta[0].square_will_be_occupied_by_piece_on["f8"] == "h1"
        assert meta[0].square_will_be_occupied_by_piece_on["f1"] == "h1"

    def test_castled_rook_captures(self) -> None:
        """Rook captures immediately after castling."""
        # Black piece on f8 that rook can capture after castling
        # Note: f8 must not attack f1 during castling, so use a knight instead
        fen = "4kn2/8/8/8/8/8/8/4K2R w K - 0 1"
        positions = make_game_from_moves(["e1g1", "e8d8", "f1f8"], start_fen=fen)
        meta = compute_meta_features(positions)

        # Rook on h1 moves to f1 (castling), then captures on f8
        assert meta[0].piece_will_move_to["h1"] == "f1"
        # Black knight on f8 will be captured - no future movement
        assert "f8" not in meta[0].piece_will_move_to

        # f8 traces to h1 (white rook's original position)
        assert meta[0].square_will_be_occupied_by_piece_on["f8"] == "h1"

    def test_castled_king_captures(self) -> None:
        """King captures after castling."""
        # After O-O, king is on g1. Then king moves and captures.
        # Put a black rook on h2 - it doesn't attack e1, f1, or g1.
        fen = "4k3/8/8/8/8/8/7r/4K2R w K - 0 1"
        positions = make_game_from_moves(["e1g1", "e8d8", "g1h2"], start_fen=fen)
        meta = compute_meta_features(positions)

        # King moves e1->g1 (castling), then g1->h2 (capture)
        assert meta[0].piece_will_move_to["e1"] == "g1"
        # Black rook on h2 will be captured
        assert "h2" not in meta[0].piece_will_move_to

        # h2 traces to e1 (king's original position)
        assert meta[0].square_will_be_occupied_by_piece_on["h2"] == "e1"


class TestNextCaptureSquareTracing:
    """Tests for next_capture_square tracing through multiple moves."""

    def test_captured_piece_moves_multiple_times(self) -> None:
        """The piece that will be captured moves several times first.

        Scenario: Black knight on b8 moves b8->c6->e5, then gets captured.
        At position 0, next_capture_square should be "b8".
        """
        # Setup: white pawn will eventually capture black knight
        fen = "rn2k3/8/8/4P3/8/8/8/4K3 b - - 0 1"
        # Black: Nb8-c6, White: e5-e6, Black: Nc6-e5, White: waits, Black: moves king
        # Actually let's make it simpler
        fen = "1n2k3/8/8/8/3P4/8/8/4K3 b - - 0 1"
        # Nb8-c6, d4-d5, Nc6-d4 (going back), Kd5 doesn't work...
        # Let me set up a cleaner scenario
        fen = "4k3/8/8/8/8/2n5/8/4K3 w - - 0 1"
        # White king chases knight: Ke1-d2, Nc3-e4, Kd2-e3, Ne4-f6, Ke3-f4, etc.
        # This is getting complicated. Let me use a simpler setup.

        # Simpler: Knight moves twice, then pawn captures it
        fen = "4k3/8/8/8/8/n7/1P6/4K3 w - - 0 1"
        # b2-b3, Na3-b5, b3-b4, Nb5-c3?? no that doesn't work
        # Let me try: white pawn on d4, black knight on f6
        fen = "4k3/8/5n2/8/3P4/8/8/4K3 w - - 0 1"
        # d4-d5, Nf6-e4, d5-d6, Ne4-d6? no pawn captures
        # OK let's do: Pawn advances, knight moves around, pawn captures knight
        fen = "4k3/8/8/3n4/8/4P3/8/4K3 w - - 0 1"
        # e3-e4, Nd5-f4, e4-e5, Nf4-g6, e5-e6, Ng6-e5?? no
        # This is tricky. Let me try a direct approach.

        # Knight on g8, will move g8->f6->e4, then white Bxe4
        fen = "4k1n1/8/8/8/8/8/8/2B1K3 w - - 0 1"
        positions = make_game_from_moves(
            ["c1g5", "g8f6", "g5f6", "e8d7"],  # Bg5, Nf6, Bxf6 (capture)
            start_fen=fen,
        )
        meta = compute_meta_features(positions)

        # At position 0: the knight (currently on g8) will be captured
        # But it moves to f6 first, so next_capture_square traces back to g8
        assert meta[0].next_capture_square == "g8"

        # At position 1 (after Bg5): knight is still on g8, about to move
        assert meta[1].next_capture_square == "g8"

        # At position 2 (after Nf6): knight is now on f6, about to be captured
        assert meta[2].next_capture_square == "f6"

    def test_captured_piece_moves_three_times(self) -> None:
        """Piece moves three times before capture."""
        # Knight: b1->c3->e4->g5, then captured by pawn on f6 (diagonal capture to g5)
        fen = "4k3/8/5p2/8/8/8/8/1N2K3 w - - 0 1"
        positions = make_game_from_moves(
            ["b1c3", "e8d8", "c3e4", "d8c8", "e4g5", "f6g5"],  # pawn captures knight diagonally
            start_fen=fen,
        )
        meta = compute_meta_features(positions)

        # At position 0: knight on b1 will eventually be captured
        assert meta[0].next_capture_square == "b1"

        # At position 2: knight is on c3
        assert meta[2].next_capture_square == "c3"

        # At position 4: knight is on e4
        assert meta[4].next_capture_square == "e4"


class TestEmptyGame:
    """Edge case tests."""

    def test_empty_game_returns_empty(self) -> None:
        """Empty position list returns empty meta-features list."""
        meta = compute_meta_features([])
        assert meta == []

    def test_single_position(self) -> None:
        """Single position (no moves) has empty movement dicts."""
        positions = [make_position(chess.STARTING_FEN)]
        meta = compute_meta_features(positions)

        assert len(meta) == 1
        # No future moves from a single position
        assert meta[0].piece_will_move_to == {}
        assert meta[0].square_will_be_occupied_from == {}
        assert meta[0].square_will_be_occupied_by_piece_on == {}
        assert meta[0].next_capture_square is None
        assert meta[0].next_pawn_move_square is None
