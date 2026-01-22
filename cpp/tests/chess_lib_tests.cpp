/**
 * Chess Library Tests
 *
 * These tests verify that the chess-library behaves consistently with python-chess,
 * specifically for:
 * 1. En passant FEN handling - en passant square should only appear in FEN if
 *    there's a legal en passant capture available
 * 2. Castling UCI notation - castling should use king destination (e1g1, e1c1)
 *    not king-captures-rook notation
 * 3. Tokenizer compatibility with Python catgpt.core.utils.tokenizer
 *
 * IMPORTANT: For python-chess compatible en passant behavior, use makeMove<true>()
 * (the EXACT mode). The default makeMove<false>() will record en passant squares
 * even when the capture would be illegal due to pins/discovered check.
 *
 * Example:
 *   board.makeMove<true>(move);  // Use this for python-chess compatibility
 *   board.makeMove(move);        // Default: may show ep square even when illegal
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "../external/chess-library/tests/doctest/doctest.hpp"
#include "../external/chess-library/include/chess.hpp"
#include "../src/tokenizer.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace chess;

// ============================================================================
// En Passant FEN Tests
// ============================================================================
// These tests confirm that the chess-library handles en passant in FEN strings
// the same way as python-chess:
// - En passant square should only appear if there's a legal capture
// - If en passant would put the capturing player's king in check, it should NOT appear

TEST_SUITE("En Passant FEN Behavior (python-chess compatibility)") {

    TEST_CASE("En passant square shown when legal capture exists") {
        // Play 1. e4 e6 2. e5 d5 - white pawn on e5, black pawn just moved d7-d5
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));  // 1. e4
        board.makeMove<true>(uci::uciToMove(board, "e7e6"));  // 1... e6
        board.makeMove<true>(uci::uciToMove(board, "e4e5"));  // 2. e5
        board.makeMove<true>(uci::uciToMove(board, "d7d5"));  // 2... d5

        std::string fen = board.getFen();

        // The FEN should contain "d6" as the en passant square
        REQUIRE(fen.find("d6") != std::string::npos);

        // Verify that en passant is actually a legal move
        Move ep_move = uci::uciToMove(board, "e5d6");
        REQUIRE(ep_move != Move::NO_MOVE);

        // Verify it's in legal moves
        Movelist moves;
        movegen::legalmoves(moves, board);
        bool found = std::find(moves.begin(), moves.end(), ep_move) != moves.end();
        REQUIRE(found);
    }

    TEST_CASE("En passant square hidden when capture would cause discovered check") {
        // Classic example: Pin along the rank preventing en passant
        // Position: 8/8/8/r2pPK2/8/8/8/4k3 w - d6 0 1
        // Black rook a5, black pawn d5 (just moved), white pawn e5, white king f5
        // If white plays exd6 en passant, the d5 pawn is removed, e5 pawn goes to d6.
        // The 5th rank now has: rook a5, king f5 - CHECK!
        // So exd6 en passant is illegal.

        // First, set the position WITH the en passant square specified
        Board board("8/8/8/r2pPK2/8/8/8/4k3 w - d6 0 1");

        // The en passant capture should be illegal
        Move ep_move = uci::uciToMove(board, "e5d6");

        // Check if it's in legal moves (it should NOT be)
        Movelist moves;
        movegen::legalmoves(moves, board);
        bool ep_is_legal = std::find(moves.begin(), moves.end(), ep_move) != moves.end();
        REQUIRE_FALSE(ep_is_legal);

        // The key behavior: chess-library should NOT include the en passant square
        // in the FEN it generates, since there's no legal en passant move
        std::string generated_fen = board.getFen();

        // Parse the en passant field (4th field in FEN)
        std::istringstream iss(generated_fen);
        std::string piece_placement, side_to_move, castling, ep_field;
        iss >> piece_placement >> side_to_move >> castling >> ep_field;

        REQUIRE(ep_field == "-");
    }

    TEST_CASE("En passant square hidden when no capturing pawn exists") {
        // Position: White pawn on e5, no black pawn on d5 or f5
        // Black plays h7-h5 - no white pawn can capture en passant
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));  // 1. e4
        board.makeMove<true>(uci::uciToMove(board, "e7e6"));  // 1... e6
        board.makeMove<true>(uci::uciToMove(board, "e4e5"));  // 2. e5
        board.makeMove<true>(uci::uciToMove(board, "h7h5"));  // 2... h5 (no white pawn adjacent)

        std::string generated_fen = board.getFen();

        // Parse the en passant field
        std::istringstream iss(generated_fen);
        std::string piece_placement, side_to_move, castling, ep_field;
        iss >> piece_placement >> side_to_move >> castling >> ep_field;

        REQUIRE(ep_field == "-");
    }

    TEST_CASE("En passant legal capture from both sides") {
        // White pawns on d5 and f5, black plays e7-e5
        Board board;
        // 1. d4 a6 2. d5 b6 3. f4 c6 4. f5 e5
        board.makeMove<true>(uci::uciToMove(board, "d2d4"));
        board.makeMove<true>(uci::uciToMove(board, "a7a6"));
        board.makeMove<true>(uci::uciToMove(board, "d4d5"));
        board.makeMove<true>(uci::uciToMove(board, "b7b6"));
        board.makeMove<true>(uci::uciToMove(board, "f2f4"));
        board.makeMove<true>(uci::uciToMove(board, "c7c6"));
        board.makeMove<true>(uci::uciToMove(board, "f4f5"));
        board.makeMove<true>(uci::uciToMove(board, "e7e5"));

        std::string fen = board.getFen();
        REQUIRE(fen.find("e6") != std::string::npos);

        // Both captures should be legal
        Movelist moves;
        movegen::legalmoves(moves, board);

        Move d5e6 = uci::uciToMove(board, "d5e6");
        Move f5e6 = uci::uciToMove(board, "f5e6");

        bool d5e6_legal = std::find(moves.begin(), moves.end(), d5e6) != moves.end();
        bool f5e6_legal = std::find(moves.begin(), moves.end(), f5e6) != moves.end();

        REQUIRE(d5e6_legal);
        REQUIRE(f5e6_legal);
    }

    TEST_CASE("Invalid en passant square in input FEN is normalized") {
        // When given a FEN with an invalid en passant square (no legal capture),
        // the library should normalize it to '-'
        Board board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

        // e3 is specified but no black pawn can capture there
        std::string generated_fen = board.getFen();

        std::istringstream iss(generated_fen);
        std::string piece_placement, side_to_move, castling, ep_field;
        iss >> piece_placement >> side_to_move >> castling >> ep_field;

        REQUIRE(ep_field == "-");
    }

    TEST_CASE("makeMove<true> vs makeMove<false> for pinned en passant") {
        // This test demonstrates why makeMove<true> is needed for python-chess compatibility
        //
        // Position: Black to move, d7-d5
        // 8/3p4/8/r3PK2/8/8/8/4k3 b - - 0 1
        //
        // After d7-d5, white's pawn on e5 could theoretically capture en passant,
        // but doing so would expose the white king to the rook on a5.
        // Therefore, en passant is illegal and the FEN should show '-'

        // makeMove<false> (default) - does NOT check legality, shows d6
        Board board_default("8/3p4/8/r3PK2/8/8/8/4k3 b - - 0 1");
        board_default.makeMove(uci::uciToMove(board_default, "d7d5"));
        std::string fen_default = board_default.getFen();

        std::istringstream iss1(fen_default);
        std::string pp1, stm1, castling1, ep1;
        iss1 >> pp1 >> stm1 >> castling1 >> ep1;
        CHECK(ep1 == "d6");  // Default shows d6 (not python-chess compatible)

        // makeMove<true> (EXACT) - checks legality, correctly shows '-'
        Board board_exact("8/3p4/8/r3PK2/8/8/8/4k3 b - - 0 1");
        board_exact.makeMove<true>(uci::uciToMove(board_exact, "d7d5"));
        std::string fen_exact = board_exact.getFen();

        std::istringstream iss2(fen_exact);
        std::string pp2, stm2, castling2, ep2;
        iss2 >> pp2 >> stm2 >> castling2 >> ep2;
        CHECK(ep2 == "-");  // EXACT shows '-' (python-chess compatible!)
    }
}

// ============================================================================
// Castling UCI Notation Tests
// ============================================================================
// These tests confirm that castling in UCI notation uses king destination squares:
// - White kingside: e1g1 (not e1h1)
// - White queenside: e1c1 (not e1a1)
// - Black kingside: e8g8 (not e8h8)
// - Black queenside: e8c8 (not e8a8)

TEST_SUITE("Castling UCI Notation (python-chess compatibility)") {

    TEST_CASE("White kingside castling is e1g1 in UCI notation") {
        // Position where white can castle kingside
        Board board;
        // 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 - now white can castle
        board.makeMove(uci::uciToMove(board, "e2e4"));
        board.makeMove(uci::uciToMove(board, "e7e5"));
        board.makeMove(uci::uciToMove(board, "g1f3"));
        board.makeMove(uci::uciToMove(board, "b8c6"));
        board.makeMove(uci::uciToMove(board, "f1c4"));
        board.makeMove(uci::uciToMove(board, "f8c5"));

        // Parse the castling move using UCI notation e1g1
        Move castling_move = uci::uciToMove(board, "e1g1");
        REQUIRE(castling_move != Move::NO_MOVE);

        // Verify it's a legal move
        Movelist moves;
        movegen::legalmoves(moves, board);
        bool found = std::find(moves.begin(), moves.end(), castling_move) != moves.end();
        REQUIRE(found);

        // Verify UCI output is e1g1 (king's destination), not e1h1
        std::string uci_str = uci::moveToUci(castling_move);
        REQUIRE(uci_str == "e1g1");
    }

    TEST_CASE("White queenside castling is e1c1 in UCI notation") {
        // Position where white can castle queenside
        Board board;
        // 1. d4 d5 2. Nc3 Nc6 3. Bf4 Bf5 4. Qd3 Qd6
        board.makeMove(uci::uciToMove(board, "d2d4"));
        board.makeMove(uci::uciToMove(board, "d7d5"));
        board.makeMove(uci::uciToMove(board, "b1c3"));
        board.makeMove(uci::uciToMove(board, "b8c6"));
        board.makeMove(uci::uciToMove(board, "c1f4"));
        board.makeMove(uci::uciToMove(board, "c8f5"));
        board.makeMove(uci::uciToMove(board, "d1d3"));
        board.makeMove(uci::uciToMove(board, "d8d6"));

        Move castling_move = uci::uciToMove(board, "e1c1");
        REQUIRE(castling_move != Move::NO_MOVE);

        Movelist moves;
        movegen::legalmoves(moves, board);
        bool found = std::find(moves.begin(), moves.end(), castling_move) != moves.end();
        REQUIRE(found);

        std::string uci_str = uci::moveToUci(castling_move);
        REQUIRE(uci_str == "e1c1");
    }

    TEST_CASE("Black kingside castling is e8g8 in UCI notation") {
        Board board;
        // 1. e4 e5 2. Nf3 Nf6 3. Bc4 Bc5 4. d3
        board.makeMove(uci::uciToMove(board, "e2e4"));
        board.makeMove(uci::uciToMove(board, "e7e5"));
        board.makeMove(uci::uciToMove(board, "g1f3"));
        board.makeMove(uci::uciToMove(board, "g8f6"));
        board.makeMove(uci::uciToMove(board, "f1c4"));
        board.makeMove(uci::uciToMove(board, "f8c5"));
        board.makeMove(uci::uciToMove(board, "d2d3"));

        Move castling_move = uci::uciToMove(board, "e8g8");
        REQUIRE(castling_move != Move::NO_MOVE);

        Movelist moves;
        movegen::legalmoves(moves, board);
        bool found = std::find(moves.begin(), moves.end(), castling_move) != moves.end();
        REQUIRE(found);

        std::string uci_str = uci::moveToUci(castling_move);
        REQUIRE(uci_str == "e8g8");
    }

    TEST_CASE("Black queenside castling is e8c8 in UCI notation") {
        Board board;
        // 1. d4 d5 2. Nf3 Nc6 3. e3 Bf5 4. Bd3 Qd7 5. O-O
        board.makeMove(uci::uciToMove(board, "d2d4"));
        board.makeMove(uci::uciToMove(board, "d7d5"));
        board.makeMove(uci::uciToMove(board, "g1f3"));
        board.makeMove(uci::uciToMove(board, "b8c6"));
        board.makeMove(uci::uciToMove(board, "e2e3"));
        board.makeMove(uci::uciToMove(board, "c8f5"));
        board.makeMove(uci::uciToMove(board, "f1d3"));
        board.makeMove(uci::uciToMove(board, "d8d7"));
        board.makeMove(uci::uciToMove(board, "e1g1"));  // White castles kingside

        Move castling_move = uci::uciToMove(board, "e8c8");
        REQUIRE(castling_move != Move::NO_MOVE);

        Movelist moves;
        movegen::legalmoves(moves, board);
        bool found = std::find(moves.begin(), moves.end(), castling_move) != moves.end();
        REQUIRE(found);

        std::string uci_str = uci::moveToUci(castling_move);
        REQUIRE(uci_str == "e8c8");
    }

    TEST_CASE("Castling move from legal_moves uses king-movement notation") {
        Board board;
        board.makeMove(uci::uciToMove(board, "e2e4"));
        board.makeMove(uci::uciToMove(board, "e7e5"));
        board.makeMove(uci::uciToMove(board, "g1f3"));
        board.makeMove(uci::uciToMove(board, "b8c6"));
        board.makeMove(uci::uciToMove(board, "f1c4"));
        board.makeMove(uci::uciToMove(board, "f8c5"));

        // Find the castling move in legal moves
        Movelist moves;
        movegen::legalmoves(moves, board);

        std::vector<Move> castling_moves;
        for (const auto& m : moves) {
            if (m.typeOf() == Move::CASTLING) {
                castling_moves.push_back(m);
            }
        }

        REQUIRE(castling_moves.size() == 1);

        // The move's UCI representation should be e1g1 (king destination)
        std::string uci_str = uci::moveToUci(castling_moves[0]);
        REQUIRE(uci_str == "e1g1");
    }

    TEST_CASE("Castling executed correctly moves both king and rook") {
        Board board;
        board.makeMove(uci::uciToMove(board, "e2e4"));
        board.makeMove(uci::uciToMove(board, "e7e5"));
        board.makeMove(uci::uciToMove(board, "g1f3"));
        board.makeMove(uci::uciToMove(board, "b8c6"));
        board.makeMove(uci::uciToMove(board, "f1c4"));
        board.makeMove(uci::uciToMove(board, "f8c5"));

        // Execute castling using UCI notation
        Move castling = uci::uciToMove(board, "e1g1");
        board.makeMove(castling);

        // King should be on g1
        REQUIRE(board.at(Square::SQ_G1) == Piece::WHITEKING);
        // Rook should be on f1
        REQUIRE(board.at(Square::SQ_F1) == Piece::WHITEROOK);
        // Original squares should be empty
        REQUIRE(board.at(Square::SQ_E1) == Piece::NONE);
        REQUIRE(board.at(Square::SQ_H1) == Piece::NONE);
    }

    TEST_CASE("Standard chess accepts king-destination notation for castling input") {
        // In standard chess, e1g1 should be accepted for kingside castling
        Board board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");

        Move kingside = uci::uciToMove(board, "e1g1");
        REQUIRE(kingside != Move::NO_MOVE);
        REQUIRE(kingside.typeOf() == Move::CASTLING);

        Move queenside = uci::uciToMove(board, "e1c1");
        REQUIRE(queenside != Move::NO_MOVE);
        REQUIRE(queenside.typeOf() == Move::CASTLING);
    }
}

// ============================================================================
// Additional Compatibility Tests
// ============================================================================

TEST_SUITE("Additional python-chess Compatibility") {

    TEST_CASE("Starting position FEN matches") {
        Board board;
        REQUIRE(board.getFen() == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }

    TEST_CASE("FEN round-trip preserves position") {
        std::string fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        Board board(fen);
        REQUIRE(board.getFen() == fen);
    }

    TEST_CASE("Legal move count matches for starting position") {
        Board board;
        Movelist moves;
        movegen::legalmoves(moves, board);
        // Starting position has 20 legal moves (16 pawn moves + 4 knight moves)
        REQUIRE(moves.size() == 20);
    }

    TEST_CASE("Checkmate detection") {
        // Fool's mate position - white is checkmated
        Board board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
        Movelist moves;
        movegen::legalmoves(moves, board);
        REQUIRE(moves.size() == 0);

        // isGameOver() returns std::pair<GameResultReason, GameResult>
        auto game_over = board.isGameOver();
        GameResultReason reason = game_over.first;
        GameResult result = game_over.second;
        CHECK(result != GameResult::NONE);
        CHECK(reason == GameResultReason::CHECKMATE);
    }

    TEST_CASE("Stalemate detection") {
        // Classic stalemate: black king in corner
        Board board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1");
        Movelist moves;
        movegen::legalmoves(moves, board);
        REQUIRE(moves.size() == 0);

        // isGameOver() returns std::pair<GameResultReason, GameResult>
        auto game_over = board.isGameOver();
        GameResultReason reason = game_over.first;
        GameResult result = game_over.second;
        CHECK(result != GameResult::NONE);
        CHECK(reason == GameResultReason::STALEMATE);
    }
}

// ============================================================================
// Tokenizer Tests (Python Compatibility)
// ============================================================================
// These tests verify that the C++ tokenizer produces identical output to the
// Python tokenizer in src/catgpt/core/utils/tokenizer.py

TEST_SUITE("Tokenizer (python-chess compatibility)") {

    TEST_CASE("Vocabulary size matches Python") {
        REQUIRE(catgpt::VOCAB_SIZE == 26);
    }

    TEST_CASE("Character to token mapping") {
        // Digits 0-9 -> tokens 0-9
        for (int i = 0; i <= 9; ++i) {
            REQUIRE(catgpt::char_to_token('0' + i) == i);
        }

        // Lowercase pieces (non-side-to-move) -> tokens 10-16
        REQUIRE(catgpt::char_to_token('p') == 10);
        REQUIRE(catgpt::char_to_token('b') == 11);
        REQUIRE(catgpt::char_to_token('n') == 12);
        REQUIRE(catgpt::char_to_token('r') == 13);
        REQUIRE(catgpt::char_to_token('c') == 14);  // castling rook
        REQUIRE(catgpt::char_to_token('k') == 15);
        REQUIRE(catgpt::char_to_token('q') == 16);

        // Uppercase pieces (side-to-move) -> tokens 17-23
        REQUIRE(catgpt::char_to_token('P') == 17);
        REQUIRE(catgpt::char_to_token('B') == 18);
        REQUIRE(catgpt::char_to_token('N') == 19);
        REQUIRE(catgpt::char_to_token('R') == 20);
        REQUIRE(catgpt::char_to_token('C') == 21);  // castling rook
        REQUIRE(catgpt::char_to_token('Q') == 22);
        REQUIRE(catgpt::char_to_token('K') == 23);

        // Special tokens
        REQUIRE(catgpt::char_to_token('x') == 24);  // en passant
        REQUIRE(catgpt::char_to_token('.') == 25);  // empty/padding
        REQUIRE(catgpt::PAD_TOKEN == 25);           // constant should match
    }

    TEST_CASE("Starting position tokenization - white to move") {
        Board board;  // Starting position: white to move

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // In starting position with white to move:
        // - White pieces (side-to-move) are uppercase
        // - Black pieces (opponent) are lowercase
        // - Board is NOT flipped (white pawns on rank 2, black pawns on rank 7)
        // - Tokenizer order is a8=0, b8=1, ..., h1=63

        // Row 0 (rank 8): r n b q k b n r (black pieces, lowercase)
        REQUIRE(tokens[0] == catgpt::char_to_token('c'));   // a8: black rook with castling (queenside)
        REQUIRE(tokens[1] == catgpt::char_to_token('n'));   // b8: black knight
        REQUIRE(tokens[2] == catgpt::char_to_token('b'));   // c8: black bishop
        REQUIRE(tokens[3] == catgpt::char_to_token('q'));   // d8: black queen
        REQUIRE(tokens[4] == catgpt::char_to_token('k'));   // e8: black king
        REQUIRE(tokens[5] == catgpt::char_to_token('b'));   // f8: black bishop
        REQUIRE(tokens[6] == catgpt::char_to_token('n'));   // g8: black knight
        REQUIRE(tokens[7] == catgpt::char_to_token('c'));   // h8: black rook with castling (kingside)

        // Row 1 (rank 7): p p p p p p p p (black pawns)
        for (int i = 8; i < 16; ++i) {
            REQUIRE(tokens[i] == catgpt::char_to_token('p'));
        }

        // Rows 2-5 (ranks 6-3): empty squares
        for (int i = 16; i < 48; ++i) {
            REQUIRE(tokens[i] == catgpt::char_to_token('.'));
        }

        // Row 6 (rank 2): P P P P P P P P (white pawns, uppercase = side-to-move)
        for (int i = 48; i < 56; ++i) {
            REQUIRE(tokens[i] == catgpt::char_to_token('P'));
        }

        // Row 7 (rank 1): R N B Q K B N R (white pieces, uppercase = side-to-move)
        REQUIRE(tokens[56] == catgpt::char_to_token('C'));  // a1: white rook with castling (queenside)
        REQUIRE(tokens[57] == catgpt::char_to_token('N'));  // b1: white knight
        REQUIRE(tokens[58] == catgpt::char_to_token('B'));  // c1: white bishop
        REQUIRE(tokens[59] == catgpt::char_to_token('Q'));  // d1: white queen
        REQUIRE(tokens[60] == catgpt::char_to_token('K'));  // e1: white king
        REQUIRE(tokens[61] == catgpt::char_to_token('B'));  // f1: white bishop
        REQUIRE(tokens[62] == catgpt::char_to_token('N'));  // g1: white knight
        REQUIRE(tokens[63] == catgpt::char_to_token('C'));  // h1: white rook with castling (kingside)
    }

    TEST_CASE("Starting position tokenization - black to move") {
        // Position after 1. e4 (black to move)
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // With black to move:
        // - Board is FLIPPED (black's perspective)
        // - Black pieces (side-to-move) are uppercase
        // - White pieces (opponent) are lowercase

        // After flip: what was rank 1 is now row 0, what was rank 8 is now row 7
        // Row 0 (was rank 1): r n b q k b n r (white pieces, now lowercase = opponent)
        REQUIRE(tokens[0] == catgpt::char_to_token('c'));   // was a1, now a8 position: white QS rook
        REQUIRE(tokens[1] == catgpt::char_to_token('n'));   // was b1
        REQUIRE(tokens[2] == catgpt::char_to_token('b'));   // was c1
        REQUIRE(tokens[3] == catgpt::char_to_token('q'));   // was d1
        REQUIRE(tokens[4] == catgpt::char_to_token('k'));   // was e1
        REQUIRE(tokens[5] == catgpt::char_to_token('b'));   // was f1
        REQUIRE(tokens[6] == catgpt::char_to_token('n'));   // was g1
        REQUIRE(tokens[7] == catgpt::char_to_token('c'));   // was h1, white KS rook

        // Row 1 (was rank 2): white pawns, with e4 pawn missing from this row
        // After flip, the e-pawn is now on the opponent's 4th rank
        REQUIRE(tokens[8] == catgpt::char_to_token('p'));   // a2
        REQUIRE(tokens[9] == catgpt::char_to_token('p'));   // b2
        REQUIRE(tokens[10] == catgpt::char_to_token('p'));  // c2
        REQUIRE(tokens[11] == catgpt::char_to_token('p'));  // d2
        REQUIRE(tokens[12] == catgpt::char_to_token('.'));  // e2 - empty (pawn moved to e4)
        REQUIRE(tokens[13] == catgpt::char_to_token('p'));  // f2
        REQUIRE(tokens[14] == catgpt::char_to_token('p'));  // g2
        REQUIRE(tokens[15] == catgpt::char_to_token('p'));  // h2

        // Row 6 (was rank 7): black pawns (uppercase = side-to-move)
        for (int i = 48; i < 56; ++i) {
            REQUIRE(tokens[i] == catgpt::char_to_token('P'));
        }

        // Row 7 (was rank 8): black pieces (uppercase = side-to-move)
        REQUIRE(tokens[56] == catgpt::char_to_token('C'));  // was a8, black QS rook
        REQUIRE(tokens[57] == catgpt::char_to_token('N'));  // was b8
        REQUIRE(tokens[58] == catgpt::char_to_token('B'));  // was c8
        REQUIRE(tokens[59] == catgpt::char_to_token('Q'));  // was d8
        REQUIRE(tokens[60] == catgpt::char_to_token('K'));  // was e8
        REQUIRE(tokens[61] == catgpt::char_to_token('B'));  // was f8
        REQUIRE(tokens[62] == catgpt::char_to_token('N'));  // was g8
        REQUIRE(tokens[63] == catgpt::char_to_token('C'));  // was h8, black KS rook
    }

    TEST_CASE("En passant square marking") {
        // Position with legal en passant
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));
        board.makeMove<true>(uci::uciToMove(board, "a7a6"));
        board.makeMove<true>(uci::uciToMove(board, "e4e5"));
        board.makeMove<true>(uci::uciToMove(board, "d7d5"));  // d5 creates EP on d6

        // White to move, EP square is d6
        REQUIRE(board.sideToMove() == Color::WHITE);
        REQUIRE(board.enpassantSq() == Square::SQ_D6);

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // d6 in tokenizer order (a8=0): row 2 (rank 6), column 3 (d-file)
        // index = 2 * 8 + 3 = 19
        REQUIRE(tokens[19] == catgpt::char_to_token('x'));
    }

    TEST_CASE("Castling rights affect rook tokens") {
        // Position with only white kingside castling
        Board board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w K - 0 1");

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // White rooks: a1 = index 56, h1 = index 63
        // Only h1 (kingside) has castling rights
        REQUIRE(tokens[56] == catgpt::char_to_token('R'));  // a1: normal rook (no castling)
        REQUIRE(tokens[63] == catgpt::char_to_token('C'));  // h1: castling rook

        // Black rooks: a8 = index 0, h8 = index 7
        // No black castling rights
        REQUIRE(tokens[0] == catgpt::char_to_token('r'));   // a8: normal rook
        REQUIRE(tokens[7] == catgpt::char_to_token('r'));   // h8: normal rook
    }

    TEST_CASE("No castling rights after king move") {
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));
        board.makeMove<true>(uci::uciToMove(board, "e7e5"));
        board.makeMove<true>(uci::uciToMove(board, "e1e2"));  // King moves, loses both castling rights

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // Black to move after white's king move
        // Board is flipped, white pieces are lowercase

        // After flip: rank 1 -> row 0
        // h1 rook (was at index 63) is now at index 7
        // a1 rook (was at index 56) is now at index 0

        // White lost all castling rights
        REQUIRE(tokens[0] == catgpt::char_to_token('r'));  // white a1 rook - no castling
        REQUIRE(tokens[7] == catgpt::char_to_token('r'));  // white h1 rook - no castling

        // Black still has castling rights
        // After flip: rank 8 -> row 7
        REQUIRE(tokens[56] == catgpt::char_to_token('C'));  // black a8 rook - has castling
        REQUIRE(tokens[63] == catgpt::char_to_token('C'));  // black h8 rook - has castling
    }

    TEST_CASE("Halfmove clock encoding") {
        Board board;
        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // Starting position has 0 halfmoves
        // Should be encoded as "." + "0" (left-padded)
        REQUIRE(tokens[64] == catgpt::char_to_token('.'));
        REQUIRE(tokens[65] == catgpt::char_to_token('0'));
        REQUIRE(tokens[66] == catgpt::char_to_token('.'));  // padding
    }

    TEST_CASE("Halfmove clock double digit") {
        // Position with 42 halfmoves
        Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 42 1");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // 42 should be encoded as "4" + "2"
        REQUIRE(tokens[64] == catgpt::char_to_token('4'));
        REQUIRE(tokens[65] == catgpt::char_to_token('2'));
    }

    TEST_CASE("TRT tokenization returns correct size") {
        Board board;
        auto tokens = catgpt::tokenize_for_trt(board);

        REQUIRE(tokens.size() == 64);

        // Verify it's int32_t compatible
        for (const auto& t : tokens) {
            REQUIRE(t >= 0);
            REQUIRE(t < static_cast<std::int32_t>(catgpt::VOCAB_SIZE));
        }
    }

    TEST_CASE("Complex position tokenization") {
        // Italian Game position
        Board board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // White to move, pieces at various locations
        // Check a few key squares

        // c4: white bishop (uppercase B)
        // Tokenizer index: row 4 (rank 4), col 2 (c-file) = 4*8 + 2 = 34
        REQUIRE(tokens[34] == catgpt::char_to_token('B'));

        // e4: white pawn
        // Index: row 4, col 4 = 36
        REQUIRE(tokens[36] == catgpt::char_to_token('P'));

        // f3: white knight
        // Index: row 5, col 5 = 45
        REQUIRE(tokens[45] == catgpt::char_to_token('N'));

        // c6: black knight (lowercase n)
        // Index: row 2, col 2 = 18
        REQUIRE(tokens[18] == catgpt::char_to_token('n'));

        // f6: black knight
        // Index: row 2, col 5 = 21
        REQUIRE(tokens[21] == catgpt::char_to_token('n'));

        // e5: black pawn
        // Index: row 3, col 4 = 28
        REQUIRE(tokens[28] == catgpt::char_to_token('p'));
    }
}

// ============================================================================
// Comprehensive Tokenizer Tests (matching Python test_tokenizer.py)
// ============================================================================
// These tests mirror the Python tests in tests/test_tokenizer.py to ensure
// the C++ tokenizer produces identical output.

namespace {

/**
 * Helper to convert a board string representation to expected token array.
 * Mirrors Python's _to_tokens function.
 */
template <std::size_t N>
std::array<std::uint8_t, N> to_tokens(std::string_view board_str) {
    std::array<std::uint8_t, N> result;
    result.fill(catgpt::char_to_token('.'));
    for (std::size_t i = 0; i < board_str.size() && i < N; ++i) {
        result[i] = catgpt::char_to_token(board_str[i]);
    }
    return result;
}

/**
 * Helper to compare token arrays and print differences on failure.
 */
template <std::size_t N>
void require_tokens_equal(const std::array<std::uint8_t, N>& actual,
                          const std::array<std::uint8_t, N>& expected,
                          const std::string& context = "") {
    for (std::size_t i = 0; i < N; ++i) {
        if (actual[i] != expected[i]) {
            FAIL_CHECK("Token mismatch at index " << i << " in " << context
                       << ": expected " << static_cast<int>(expected[i])
                       << " ('" << catgpt::CHARACTERS[expected[i]] << "')"
                       << ", got " << static_cast<int>(actual[i])
                       << " ('" << catgpt::CHARACTERS[actual[i]] << "')");
        }
    }
    REQUIRE(actual == expected);
}

}  // namespace

TEST_SUITE("Tokenizer Python Compatibility (test_tokenizer.py)") {

    // ========================================================================
    // TestTokenizerBasic
    // ========================================================================

    TEST_CASE("Starting position white to move (matches Python)") {
        // Test the starting position with white to move (no flip).
        // Features: All castling rights, no en passant, halfmove 0.
        Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // Board: castling rooks marked with c/C
        // Rank 8: cnbqkbnc, Rank 7: pppppppp, Ranks 6-3: empty
        // Rank 2: PPPPPPPP, Rank 1: CNBQKBNC
        // Halfmove: ".0" (left-padded)
        std::string expected_str =
            "cnbqkbnc"   // Rank 8 (a8-h8)
            "pppppppp"   // Rank 7
            "........"   // Rank 6
            "........"   // Rank 5
            "........"   // Rank 4
            "........"   // Rank 3
            "PPPPPPPP"   // Rank 2
            "CNBQKBNC"   // Rank 1 (a1-h1)
            ".0";        // Halfmove clock (left-padded)

        auto expected = to_tokens<67>(expected_str);
        require_tokens_equal(tokens, expected, "starting position white to move");
    }

    TEST_CASE("Starting position black to move (matches Python)") {
        // Test starting position with black to move (board flips).
        // The position is symmetric, so the result should be identical to white-to-move.
        Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // After flip+swapcase, symmetric position stays the same
        std::string expected_str =
            "cnbqkbnc"
            "pppppppp"
            "........"
            "........"
            "........"
            "........"
            "PPPPPPPP"
            "CNBQKBNC"
            ".0";

        auto expected = to_tokens<67>(expected_str);
        require_tokens_equal(tokens, expected, "starting position black to move");
    }

    // ========================================================================
    // TestTokenizerEnPassant
    // ========================================================================

    TEST_CASE("En passant white to move (matches Python)") {
        // Create position where white can capture en passant on e6
        // Play: 1. e4 e6 2. e5 d5
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));  // 1. e4
        board.makeMove<true>(uci::uciToMove(board, "e7e6"));  // 1... e6
        board.makeMove<true>(uci::uciToMove(board, "e4e5"));  // 2. e5
        board.makeMove<true>(uci::uciToMove(board, "d7d5"));  // 2... d5 (creates ep on d6)

        // Verify en passant is available
        REQUIRE(board.enpassantSq() == Square::SQ_D6);

        auto tokens = catgpt::tokenize<70>(board, catgpt::TokenizerConfig{70, true});

        // En passant at d6 (rank 6, file d) -> row 2, col 3 -> index 19
        std::string expected_str =
            "cnbqkbnc"   // Rank 8
            "ppp..ppp"   // Rank 7 (d7 and e7 pawns moved)
            "...xp..."   // Rank 6 (en passant on d6, pawn on e6)
            "...pP..."   // Rank 5 (black pawn on d5, white pawn on e5)
            "........"   // Rank 4
            "........"   // Rank 3
            "PPPP.PPP"   // Rank 2 (e2 pawn moved)
            "CNBQKBNC"   // Rank 1
            ".0";

        auto expected = to_tokens<70>(expected_str);
        require_tokens_equal(tokens, expected, "en passant white to move");
    }

    TEST_CASE("En passant black to move (matches Python)") {
        // Create position where black can actually capture en passant
        // Play: 1. Nf3 d5 2. Nc3 d4 3. e4 - now black pawn on d4 can capture e3
        Board board;
        board.makeMove<true>(uci::uciToMove(board, "g1f3"));  // 1. Nf3
        board.makeMove<true>(uci::uciToMove(board, "d7d5"));  // 1... d5
        board.makeMove<true>(uci::uciToMove(board, "b1c3"));  // 2. Nc3
        board.makeMove<true>(uci::uciToMove(board, "d5d4"));  // 2... d4
        board.makeMove<true>(uci::uciToMove(board, "e2e4"));  // 3. e4

        // Verify en passant is available for black
        REQUIRE(board.enpassantSq() == Square::SQ_E3);

        auto tokens = catgpt::tokenize<68>(board, catgpt::TokenizerConfig{68, true});

        // After flip+swapcase: white pieces become lowercase (opponent), black uppercase (side-to-move)
        // FEN: rnbqkbnr/ppp1pppp/8/8/3pP3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq e3 0 3
        // Flipped board (black's view): rank 1 -> row 0, rank 8 -> row 7
        std::string expected_str =
            "c.bqkb.c"   // Row 0: was rank 1 (R1BQKB1R), b1/g1 empty, rooks have castling
            "pppp.ppp"   // Row 1: was rank 2 (PPPP1PPP), e2 empty
            "..n.xn.."   // Row 2: was rank 3, knights c3/f3 lowercase, en passant on e3
            "...Pp..."   // Row 3: was rank 4, black pawn d4 (P), white pawn e4 (p)
            "........"   // Row 4: was rank 5
            "........"   // Row 5: was rank 6
            "PPP.PPPP"   // Row 6: was rank 7, black pawns (d7 empty)
            "CNBQKBNC"   // Row 7: was rank 8, black back rank with castling rooks
            ".0";

        auto expected = to_tokens<68>(expected_str);
        require_tokens_equal(tokens, expected, "en passant black to move");
    }

    // ========================================================================
    // TestTokenizerCastling
    // ========================================================================

    TEST_CASE("Partial castling white kingside only (matches Python)") {
        // Test position where only white kingside and black kingside castling is available.
        Board board("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/1NBQKBNR w Kk - 4 3");

        auto tokens = catgpt::tokenize<66>(board, catgpt::TokenizerConfig{66, true});

        // Only h1 rook (white K) and h8 rook (black k) marked with castling
        std::string expected_str =
            ".nbqkbnc"   // Rank 8: a8 rook gone, h8 rook can castle
            "pppppppp"
            "........"
            "........"
            "........"
            "........"
            "PPPPPPPP"
            ".NBQKBNC"   // Rank 1: a1 rook gone, h1 rook can castle
            ".4";

        auto expected = to_tokens<66>(expected_str);
        require_tokens_equal(tokens, expected, "partial castling white kingside only");
    }

    TEST_CASE("No castling rights (matches Python)") {
        // Test position with no castling rights remaining.
        Board board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w - - 6 5");

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // No castling markers - all rooks are regular 'r'/'R'
        std::string expected_str =
            "r.bqkb.r"   // Rank 8: regular rooks
            "pppp.ppp"
            "..n..n.."   // Black knights on c6, f6
            "....p..."
            "....P..."
            "..N..N.."   // White knights on c3, f3
            "PPPP.PPP"
            "R.BQKB.R";  // Rank 1: regular rooks

        auto expected = to_tokens<64>(expected_str);
        require_tokens_equal(tokens, expected, "no castling rights");
    }

    // ========================================================================
    // TestTokenizerComplexPositions
    // ========================================================================

    TEST_CASE("Sicilian Dragon white to move (matches Python)") {
        // Test a complex Sicilian Dragon position with white to move.
        // Features: White has KQ castling rights, black has castled.
        Board board("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 2 9");

        auto tokens = catgpt::tokenize<72>(board, catgpt::TokenizerConfig{72, true});

        // Castling: white K (h1) and Q (a1), black has castled (no rights)
        std::string expected_str =
            "r.bq.rk."   // Rank 8: black castled kingside
            "pp..ppbp"   // Rank 7
            "..np.np."   // Rank 6: knights on c6, f6; pawn g6
            "........"   // Rank 5
            "...NP..."   // Rank 4: white knight d4, pawn e4
            "..N.BP.."   // Rank 3: knight c3, bishop e3, pawn f3
            "PPPQ..PP"   // Rank 2: queen d2
            "C...KB.C"   // Rank 1: castling rooks a1, h1
            ".2";

        auto expected = to_tokens<72>(expected_str);
        require_tokens_equal(tokens, expected, "Sicilian Dragon white to move");
    }

    TEST_CASE("Sicilian Dragon black to move (matches Python)") {
        // Same position but with black to move (board flips).
        Board board("r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R b KQ - 2 9");

        auto tokens = catgpt::tokenize<72>(board, catgpt::TokenizerConfig{72, true});

        // After flip+swapcase: black pieces become uppercase (side-to-move)
        // White castling rights -> affects original a1/h1, but those flip to a8/h8 position
        // and swapcase makes them lowercase 'c'
        std::string expected_str =
            "c...kb.c"   // Was rank 1, castling rooks marked
            "pppq..pp"   // Was rank 2
            "..n.bp.."   // Was rank 3
            "...np..."   // Was rank 4
            "........"   // Was rank 5
            "..NP.NP."   // Was rank 6
            "PP..PPBP"   // Was rank 7
            "R.BQ.RK."   // Was rank 8, black pieces now uppercase
            ".2";

        auto expected = to_tokens<72>(expected_str);
        require_tokens_equal(tokens, expected, "Sicilian Dragon black to move");
    }

    TEST_CASE("Rook endgame white to move (matches Python)") {
        // Test a rook endgame position with white to move.
        // Features: Few pieces, passed pawns, active kings.
        Board board("8/5pk1/4p1p1/3pP1P1/2pP4/2P2K2/8/4R3 w - - 15 42");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        std::string expected_str =
            "........"   // Rank 8
            ".....pk."   // Rank 7: black pawn f7, king g7
            "....p.p."   // Rank 6: pawns e6, g6
            "...pP.P."   // Rank 5: black d5, white e5, g5
            "..pP...."   // Rank 4: black c4, white d4
            "..P..K.."   // Rank 3: white pawn c3, king f3
            "........"   // Rank 2
            "....R..."   // Rank 1: white rook e1
            "15";

        auto expected = to_tokens<67>(expected_str);
        require_tokens_equal(tokens, expected, "rook endgame white to move");
    }

    TEST_CASE("Rook endgame black to move (matches Python)") {
        // Same rook endgame but with black to move (board flips).
        Board board("8/5pk1/4p1p1/3pP1P1/2pP4/2P2K2/8/4R3 b - - 15 42");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // After flip+swapcase
        std::string expected_str =
            "....r..."   // Was rank 1
            "........"   // Was rank 2
            "..p..k.."   // Was rank 3 (white pieces -> lowercase)
            "..Pp...."   // Was rank 4
            "...Pp.p."   // Was rank 5
            "....P.P."   // Was rank 6
            ".....PK."   // Was rank 7 (black pieces -> uppercase)
            "........"   // Was rank 8
            "15";

        auto expected = to_tokens<67>(expected_str);
        require_tokens_equal(tokens, expected, "rook endgame black to move");
    }

    TEST_CASE("Tactical position white to move (matches Python)") {
        // Test a sharp tactical position with many pieces and tension.
        // Features: Queens, multiple minor pieces, central tension.
        Board board("r2qr1k1/1b1nbppp/p1pp1n2/1p2p3/3PP3/1BN1BN2/PPP1QPPP/R4RK1 w - - 0 13");

        auto tokens = catgpt::tokenize<80>(board, catgpt::TokenizerConfig{80, true});

        std::string expected_str =
            "r..qr.k."   // Rank 8
            ".b.nbppp"   // Rank 7
            "p.pp.n.."   // Rank 6
            ".p..p..."   // Rank 5
            "...PP..."   // Rank 4
            ".BN.BN.."   // Rank 3
            "PPP.QPPP"   // Rank 2
            "R....RK."   // Rank 1: white castled, no castling rights
            ".0";

        auto expected = to_tokens<80>(expected_str);
        require_tokens_equal(tokens, expected, "tactical position white to move");
    }

    TEST_CASE("Tactical position black to move (matches Python)") {
        // Same tactical position with black to move.
        Board board("r2qr1k1/1b1nbppp/p1pp1n2/1p2p3/3PP3/1BN1BN2/PPP1QPPP/R4RK1 b - - 0 13");

        auto tokens = catgpt::tokenize<80>(board, catgpt::TokenizerConfig{80, true});

        // After flip+swapcase
        std::string expected_str =
            "r....rk."   // Was rank 1
            "ppp.qppp"   // Was rank 2
            ".bn.bn.."   // Was rank 3
            "...pp..."   // Was rank 4
            ".P..P..."   // Was rank 5
            "P.PP.N.."   // Was rank 6
            ".B.NBPPP"   // Was rank 7
            "R..QR.K."   // Was rank 8
            ".0";

        auto expected = to_tokens<80>(expected_str);
        require_tokens_equal(tokens, expected, "tactical position black to move");
    }

    // ========================================================================
    // TestTokenizerEdgeCases
    // ========================================================================

    TEST_CASE("Promotion tension white to move (matches Python)") {
        // Test position with pawns about to promote.
        // Features: Advanced pawns on 7th/2nd rank, minimal pieces.
        Board board("4k3/2P2P2/8/8/8/8/2p2p2/4K3 w - - 0 50");

        auto tokens = catgpt::tokenize<66>(board, catgpt::TokenizerConfig{66, true});

        std::string expected_str =
            "....k..."   // Rank 8
            "..P..P.."   // Rank 7: white pawns about to promote
            "........"
            "........"
            "........"
            "........"
            "..p..p.."   // Rank 2: black pawns about to promote
            "....K..."   // Rank 1
            ".0";

        auto expected = to_tokens<66>(expected_str);
        require_tokens_equal(tokens, expected, "promotion tension white to move");
    }

    TEST_CASE("Promotion tension black to move (matches Python)") {
        // Same promotion position with black to move.
        Board board("4k3/2P2P2/8/8/8/8/2p2p2/4K3 b - - 0 50");

        auto tokens = catgpt::tokenize<66>(board, catgpt::TokenizerConfig{66, true});

        // After flip+swapcase
        std::string expected_str =
            "....k..."   // Was rank 1
            "..P..P.."   // Was rank 2 (black pawns -> uppercase)
            "........"
            "........"
            "........"
            "........"
            "..p..p.."   // Was rank 7 (white pawns -> lowercase)
            "....K..."   // Was rank 8
            ".0";

        auto expected = to_tokens<66>(expected_str);
        require_tokens_equal(tokens, expected, "promotion tension black to move");
    }

    TEST_CASE("High halfmove clock (matches Python)") {
        // Test position with high halfmove clock (near 50-move rule).
        Board board("8/8/4k3/8/8/4K3/8/8 w - - 99 100");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        std::string expected_str =
            "........"
            "........"
            "....k..."   // Rank 6: black king
            "........"
            "........"
            "....K..."   // Rank 3: white king
            "........"
            "........"
            "99";        // High halfmove clock

        auto expected = to_tokens<67>(expected_str);
        require_tokens_equal(tokens, expected, "high halfmove clock");
    }

    TEST_CASE("Without halfmove (matches Python)") {
        // Test tokenization without halfmove clock.
        Board board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        auto tokens = catgpt::tokenize<64>(board, catgpt::NO_HALFMOVE_CONFIG);

        // No halfmove appended
        std::string expected_str =
            "cnbqkbnc"
            "pppppppp"
            "........"
            "........"
            "........"
            "........"
            "PPPPPPPP"
            "CNBQKBNC";

        auto expected = to_tokens<64>(expected_str);
        require_tokens_equal(tokens, expected, "without halfmove");
    }

    // ========================================================================
    // TestTokenizerConfig
    // ========================================================================

    TEST_CASE("Config default values") {
        catgpt::TokenizerConfig config;
        REQUIRE(config.sequence_length == 67);
        REQUIRE(config.include_halfmove == true);
    }

    TEST_CASE("Config exact minimum with halfmove") {
        catgpt::TokenizerConfig config{66, true};
        REQUIRE(config.sequence_length == 66);
        REQUIRE(config.include_halfmove == true);
    }

    TEST_CASE("Config exact minimum without halfmove") {
        catgpt::TokenizerConfig config{64, false};
        REQUIRE(config.sequence_length == 64);
        REQUIRE(config.include_halfmove == false);
    }

    // ========================================================================
    // Additional edge cases
    // ========================================================================

    TEST_CASE("Halfmove clock clamping at 99") {
        // C++ clamps halfmove to 99 (Python would throw ValueError)
        // Test with halfmove > 99
        Board board("8/8/8/8/8/8/8/4K2k w - - 100 50");

        auto tokens = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);

        // Should be clamped to 99
        REQUIRE(tokens[64] == catgpt::char_to_token('9'));
        REQUIRE(tokens[65] == catgpt::char_to_token('9'));
    }

    TEST_CASE("FEN round-trip tokenization consistency") {
        // Verify that tokenizing from FEN gives same result as from Board
        std::string fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
        Board board(fen);

        auto tokens_from_board = catgpt::tokenize<67>(board, catgpt::DEFAULT_CONFIG);
        auto tokens_from_fen = catgpt::tokenize_fen<67>(fen, catgpt::DEFAULT_CONFIG);

        REQUIRE(tokens_from_board == tokens_from_fen);
    }
}
