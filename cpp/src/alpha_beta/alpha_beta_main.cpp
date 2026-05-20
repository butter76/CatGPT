/**
 * Standalone alpha-beta search demo (chess-library + dummy eval).
 *
 * Usage:
 *   alpha_beta_search [depth] [fen]
 *
 * Examples:
 *   alpha_beta_search
 *   alpha_beta_search 6
 *   alpha_beta_search 5 "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
 */

#include <cstdlib>
#include <iostream>
#include <string>

#include "../../external/chess-library/include/chess.hpp"
#include "alpha_beta_search.hpp"
#include "dummy_evaluator.hpp"

namespace {

constexpr const char* kStartpos =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

}  // namespace

int main(int argc, char* argv[]) {
    int depth = 5;
    std::string fen = kStartpos;

    if (argc > 1) {
        depth = std::atoi(argv[1]);
        if (depth < 1) {
            depth = 1;
        }
    }
    if (argc > 2) {
        fen = argv[2];
    }

    chess::Board board(fen);
    catgpt::alpha_beta::DummyEvalNetwork network;
    catgpt::alpha_beta::AlphaBetaSearch searcher(network);

    const auto result = searcher.search(board, depth);

    if (result.best_move == chess::Move::NO_MOVE) {
        std::cout << "No legal moves (game over). score_cp=" << result.score_cp << '\n';
        return 0;
    }

    std::cout << "fen: " << board.getFen() << '\n';
    std::cout << "depth: " << result.depth << '\n';
    std::cout << "nodes: " << result.nodes << '\n';
    std::cout << "score_cp: " << result.score_cp << '\n';
    std::cout << "bestmove: " << chess::uci::moveToUci(result.best_move) << '\n';

    return 0;
}
