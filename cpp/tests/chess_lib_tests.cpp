/**
 * Chess Library Tests
 *
 * These tests verify that the chess-library behaves consistently with python-chess,
 * specifically for:
 * 1. En passant FEN handling - en passant square should only appear in FEN if
 *    there's a legal en passant capture available
 * 2. Castling UCI notation - castling should use king destination (e1g1, e1c1)
 *    not king-captures-rook notation
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
