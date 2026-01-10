"""Tests for FEN tokenization.

These tests verify the tokenize() function with various chess positions that have
multiple interesting features: castling rights, en passant, different piece
configurations, and various halfmove clock values.
"""

import numpy as np
import pytest

from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize

# Character to index mapping (from tokenizer.py)
_CHAR_TO_IDX = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "p": 10, "b": 11, "n": 12, "r": 13, "c": 14, "k": 15, "q": 16,
    "P": 17, "B": 18, "N": 19, "R": 20, "C": 21, "Q": 22, "K": 23,
    "x": 24, ".": 25,
}


def _to_tokens(board_str: str, seq_len: int) -> np.ndarray:
    """Convert a board string representation to expected token array."""
    result = np.full(seq_len, _CHAR_TO_IDX["."], dtype=np.uint8)
    for i, char in enumerate(board_str):
        result[i] = _CHAR_TO_IDX[char]
    return result


class TestTokenizerBasic:
    """Basic tokenizer functionality tests."""

    def test_starting_position_white_to_move(self) -> None:
        """Test the starting position with white to move (no flip).

        Features: All castling rights, no en passant, halfmove 0.
        Config: sequence_length=67 (default), include_halfmove=True
        """
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        config = TokenizerConfig(sequence_length=67, include_halfmove=True)

        # Board: castling rooks marked with c/C
        # Rank 8: cnbqkbnc, Rank 7: pppppppp, Ranks 6-3: empty
        # Rank 2: PPPPPPPP, Rank 1: CNBQKBNC
        # Halfmove: ".0" (left-padded)
        expected_str = (
            "cnbqkbnc" +  # Rank 8 (a8-h8)
            "pppppppp" +  # Rank 7
            "........" +  # Rank 6
            "........" +  # Rank 5
            "........" +  # Rank 4
            "........" +  # Rank 3
            "PPPPPPPP" +  # Rank 2
            "CNBQKBNC" +  # Rank 1 (a1-h1)
            ".0"          # Halfmove clock (left-padded)
        )
        expected = _to_tokens(expected_str, 67)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_starting_position_black_to_move(self) -> None:
        """Test starting position with black to move (board flips).

        The position is symmetric, so the result should be identical to white-to-move.
        """
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
        config = TokenizerConfig(sequence_length=67, include_halfmove=True)

        # After flip+swapcase, symmetric position stays the same
        expected_str = (
            "cnbqkbnc" +
            "pppppppp" +
            "........" +
            "........" +
            "........" +
            "........" +
            "PPPPPPPP" +
            "CNBQKBNC" +
            ".0"
        )
        expected = _to_tokens(expected_str, 67)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)


class TestTokenizerEnPassant:
    """Tests for en passant handling."""

    def test_en_passant_white_to_move(self) -> None:
        """Test position after 1.e4 e5 2.d4 with white to move.

        Features: En passant on e6 available for white, all castling rights.
        Config: sequence_length=70, include_halfmove=True
        """
        fen = "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR w KQkq e6 0 3"
        config = TokenizerConfig(sequence_length=70, include_halfmove=True)

        # En passant at e6 (rank 6, file e) -> row 2, col 4 -> index 20
        expected_str = (
            "cnbqkbnc" +  # Rank 8
            "pppp.ppp" +  # Rank 7 (e7 pawn moved)
            "....x..." +  # Rank 6 (en passant on e6)
            "....p..." +  # Rank 5 (black pawn on e5)
            "...PP..." +  # Rank 4 (white pawns on d4, e4)
            "........" +  # Rank 3
            "PPP..PPP" +  # Rank 2 (d2, e2 pawns moved)
            "CNBQKBNC" +  # Rank 1
            ".0"
        )
        expected = _to_tokens(expected_str, 70)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_en_passant_black_to_move(self) -> None:
        """Test position after 1.e4 with black to move.

        Features: En passant on e3 available for black.
        When black moves, board flips and en passant square is transformed.
        Config: sequence_length=68, include_halfmove=True
        """
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        config = TokenizerConfig(sequence_length=68, include_halfmove=True)

        # After flip+swapcase: white pieces become lowercase (opponent), black uppercase (side-to-move)
        # e3 en passant flips to e6 position (row 2 in the flipped board)
        expected_str = (
            "cnbqkbnc" +  # Was rank 1 (white pieces), now top
            "pppp.ppp" +  # Was rank 2 (white pawns), e2 empty
            "....x..." +  # En passant marker (e3 flipped)
            "....p..." +  # Was rank 4 (white e4 pawn), now lowercase
            "........" +  # Was rank 5
            "........" +  # Was rank 6
            "PPPPPPPP" +  # Was rank 7 (black pawns), now uppercase
            "CNBQKBNC" +  # Was rank 8 (black pieces), now bottom
            ".0"
        )
        expected = _to_tokens(expected_str, 68)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)


class TestTokenizerCastling:
    """Tests for castling rights handling."""

    def test_partial_castling_white_kingside_only(self) -> None:
        """Test position where only white kingside castling is available.

        Features: White king and h1 rook unmoved, but queenside rook moved.
        Config: sequence_length=66, include_halfmove=True
        """
        fen = "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kk - 4 3"
        config = TokenizerConfig(sequence_length=66, include_halfmove=True)

        # Only h1 rook (white K) and h8 rook (black k) marked with castling
        expected_str = (
            ".nbqkbnc" +  # Rank 8: a8 rook gone, h8 rook can castle
            "pppppppp" +
            "........" +
            "........" +
            "........" +
            "........" +
            "PPPPPPPP" +
            ".NBQKBNC" +  # Rank 1: a1 rook gone, h1 rook can castle
            ".4"
        )
        expected = _to_tokens(expected_str, 66)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_no_castling_rights(self) -> None:
        """Test position with no castling rights remaining.

        Config: sequence_length=64, include_halfmove=False
        """
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w - - 6 5"
        config = TokenizerConfig(sequence_length=64, include_halfmove=False)

        # No castling markers - all rooks are regular 'r'/'R'
        expected_str = (
            "r.bqkb.r" +  # Rank 8: regular rooks
            "pppp.ppp" +
            "..n..n.." +  # Black knights on c6, f6
            "....p..." +
            "....P..." +
            "..N..N.." +  # White knights on c3, f3
            "PPPP.PPP" +
            "R.BQKB.R"   # Rank 1: regular rooks
        )
        expected = _to_tokens(expected_str, 64)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)


class TestTokenizerComplexPositions:
    """Tests for complex middlegame and endgame positions."""

    def test_sicilian_dragon_white_to_move(self) -> None:
        """Test a complex Sicilian Dragon position with white to move.

        Features: Both sides castled, active piece play, various pieces.
        Config: sequence_length=72, include_halfmove=True
        """
        fen = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9"
        config = TokenizerConfig(sequence_length=72, include_halfmove=True)

        # Castling: white K (h1) and Q (a1), black has castled (no rights)
        expected_str = (
            "r.bq.rk." +  # Rank 8: black castled kingside
            "pp..ppbp" +  # Rank 7
            "..np.np." +  # Rank 6: knights on c6, f6; pawn g6
            "........" +  # Rank 5
            "...NP..." +  # Rank 4: white knight d4, pawn e4
            "..N.BP.." +  # Rank 3: knight c3, bishop e3, pawn f3
            "PPPQ..PP" +  # Rank 2: queen d2
            "C...KB.C" +  # Rank 1: castling rooks a1, h1
            ".2"
        )
        expected = _to_tokens(expected_str, 72)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_sicilian_dragon_black_to_move(self) -> None:
        """Same position but with black to move (board flips).

        Config: sequence_length=72, include_halfmove=True
        """
        fen = "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQ - 2 9"
        config = TokenizerConfig(sequence_length=72, include_halfmove=True)

        # After flip+swapcase: black pieces become uppercase (side-to-move)
        # White castling rights -> affects original a1/h1, but those flip to a8/h8 position
        # and swapcase makes them lowercase 'c'
        expected_str = (
            "c...kb.c" +  # Was rank 1, castling rooks marked
            "pppq..pp" +  # Was rank 2
            "..n.bp.." +  # Was rank 3
            "...np..." +  # Was rank 4
            "........" +  # Was rank 5
            "..NP.NP." +  # Was rank 6
            "PP..PPBP" +  # Was rank 7
            "R.BQ.RK." +  # Was rank 8, black pieces now uppercase
            ".2"
        )
        expected = _to_tokens(expected_str, 72)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_rook_endgame_white_to_move(self) -> None:
        """Test a rook endgame position with white to move.

        Features: Few pieces, passed pawns, active kings.
        Config: sequence_length=67, include_halfmove=True
        """
        fen = "8/5pk1/4p1p1/3pP1P1/2pP4/2P2K2/8/4R3 w - - 15 42"
        config = TokenizerConfig(sequence_length=67, include_halfmove=True)

        expected_str = (
            "........" +  # Rank 8
            ".....pk." +  # Rank 7: black pawn f7, king g7
            "....p.p." +  # Rank 6: pawns e6, g6
            "...pP.P." +  # Rank 5: black d5, white e5, g5
            "..pP...." +  # Rank 4: black c4, white d4
            "..P..K.." +  # Rank 3: white pawn c3, king f3
            "........" +  # Rank 2
            "....R..." +  # Rank 1: white rook e1
            "15"
        )
        expected = _to_tokens(expected_str, 67)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_rook_endgame_black_to_move(self) -> None:
        """Same rook endgame but with black to move (board flips).

        Config: sequence_length=67, include_halfmove=True
        """
        fen = "8/5pk1/4p1p1/3pP1P1/2pP4/2P2K2/8/4R3 b - - 15 42"
        config = TokenizerConfig(sequence_length=67, include_halfmove=True)

        # After flip+swapcase
        expected_str = (
            "....r..." +  # Was rank 1
            "........" +  # Was rank 2
            "..p..k.." +  # Was rank 3 (white pieces -> lowercase)
            "..Pp...." +  # Was rank 4
            "...Pp.p." +  # Was rank 5
            "....P.P." +  # Was rank 6
            ".....PK." +  # Was rank 7 (black pieces -> uppercase)
            "........" +  # Was rank 8
            "15"
        )
        expected = _to_tokens(expected_str, 67)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_tactical_position_white_to_move(self) -> None:
        """Test a sharp tactical position with many pieces and tension.

        Features: Queens, multiple minor pieces, central tension.
        Config: sequence_length=80, include_halfmove=True
        """
        fen = "r2qr1k1/1b1nbppp/p1pp1n2/1p2p3/3PP3/1BN1BN2/PPP1QPPP/R4RK1 w - - 0 13"
        config = TokenizerConfig(sequence_length=80, include_halfmove=True)

        expected_str = (
            "r..qr.k." +  # Rank 8
            ".b.nbppp" +  # Rank 7
            "p.pp.n.." +  # Rank 6
            ".p..p..." +  # Rank 5
            "...PP..." +  # Rank 4
            ".BN.BN.." +  # Rank 3
            "PPP.QPPP" +  # Rank 2
            "R....RK." +  # Rank 1: white castled, no castling rights
            ".0"
        )
        expected = _to_tokens(expected_str, 80)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_tactical_position_black_to_move(self) -> None:
        """Same tactical position with black to move.

        Config: sequence_length=80, include_halfmove=True
        """
        fen = "r2qr1k1/1b1nbppp/p1pp1n2/1p2p3/3PP3/1BN1BN2/PPP1QPPP/R4RK1 b - - 0 13"
        config = TokenizerConfig(sequence_length=80, include_halfmove=True)

        # After flip+swapcase
        expected_str = (
            "r....rk." +  # Was rank 1
            "ppp.qppp" +  # Was rank 2
            ".bn.bn.." +  # Was rank 3
            "...pp..." +  # Was rank 4
            ".P..P..." +  # Was rank 5
            "P.PP.N.." +  # Was rank 6
            ".B.NBPPP" +  # Was rank 7
            "R..QR.K." +  # Was rank 8
            ".0"
        )
        expected = _to_tokens(expected_str, 80)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)


class TestTokenizerEdgeCases:
    """Tests for edge cases and unusual positions."""

    def test_promotion_tension_white_to_move(self) -> None:
        """Test position with pawns about to promote.

        Features: Advanced pawns on 7th/2nd rank, minimal pieces.
        Config: sequence_length=66, include_halfmove=True
        """
        fen = "4k3/2P2P2/8/8/8/8/2p2p2/4K3 w - - 0 50"
        config = TokenizerConfig(sequence_length=66, include_halfmove=True)

        expected_str = (
            "....k..." +  # Rank 8
            "..P..P.." +  # Rank 7: white pawns about to promote
            "........" +
            "........" +
            "........" +
            "........" +
            "..p..p.." +  # Rank 2: black pawns about to promote
            "....K..." +  # Rank 1
            ".0"
        )
        expected = _to_tokens(expected_str, 66)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_promotion_tension_black_to_move(self) -> None:
        """Same promotion position with black to move.

        Config: sequence_length=66, include_halfmove=True
        """
        fen = "4k3/2P2P2/8/8/8/8/2p2p2/4K3 b - - 0 50"
        config = TokenizerConfig(sequence_length=66, include_halfmove=True)

        # After flip+swapcase
        expected_str = (
            "....k..." +  # Was rank 1
            "..P..P.." +  # Was rank 2 (black pawns -> uppercase)
            "........" +
            "........" +
            "........" +
            "........" +
            "..p..p.." +  # Was rank 7 (white pawns -> lowercase)
            "....K..." +  # Was rank 8
            ".0"
        )
        expected = _to_tokens(expected_str, 66)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_high_halfmove_clock(self) -> None:
        """Test position with high halfmove clock (near 50-move rule).

        Config: sequence_length=67, include_halfmove=True
        """
        fen = "8/8/4k3/8/8/4K3/8/8 w - - 99 100"
        config = TokenizerConfig(sequence_length=67, include_halfmove=True)

        expected_str = (
            "........" +
            "........" +
            "....k..." +  # Rank 6: black king
            "........" +
            "........" +
            "....K..." +  # Rank 3: white king
            "........" +
            "........" +
            "99"          # High halfmove clock
        )
        expected = _to_tokens(expected_str, 67)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)

    def test_without_halfmove(self) -> None:
        """Test tokenization without halfmove clock.

        Config: sequence_length=64, include_halfmove=False
        """
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        config = TokenizerConfig(sequence_length=64, include_halfmove=False)

        # No halfmove appended
        expected_str = (
            "cnbqkbnc" +
            "pppppppp" +
            "........" +
            "........" +
            "........" +
            "........" +
            "PPPPPPPP" +
            "CNBQKBNC"
        )
        expected = _to_tokens(expected_str, 64)

        result = tokenize(fen, config)
        np.testing.assert_array_equal(result, expected)


class TestTokenizerConfig:
    """Tests for TokenizerConfig validation."""

    def test_config_default_values(self) -> None:
        """Test default configuration values."""
        config = TokenizerConfig()
        assert config.sequence_length == 67
        assert config.include_halfmove is True

    def test_config_minimum_length_with_halfmove(self) -> None:
        """Test minimum sequence length validation with halfmove."""
        with pytest.raises(ValueError, match="sequence_length must be at least 66"):
            TokenizerConfig(sequence_length=65, include_halfmove=True)

    def test_config_minimum_length_without_halfmove(self) -> None:
        """Test minimum sequence length validation without halfmove."""
        with pytest.raises(ValueError, match="sequence_length must be at least 64"):
            TokenizerConfig(sequence_length=63, include_halfmove=False)

    def test_config_exact_minimum_with_halfmove(self) -> None:
        """Test that exact minimum length works with halfmove."""
        config = TokenizerConfig(sequence_length=66, include_halfmove=True)
        assert config.sequence_length == 66

    def test_config_exact_minimum_without_halfmove(self) -> None:
        """Test that exact minimum length works without halfmove."""
        config = TokenizerConfig(sequence_length=64, include_halfmove=False)
        assert config.sequence_length == 64


class TestTokenizerErrors:
    """Tests for error handling."""

    def test_invalid_halfmove_clock(self) -> None:
        """Test that 3+ digit halfmove clock raises ValueError."""
        fen = "8/8/8/8/8/8/8/4K2k w - - 100 50"
        with pytest.raises(ValueError, match="Halfmove clock must be 2 digits or less"):
            tokenize(fen)
