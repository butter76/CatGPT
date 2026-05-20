/**
 * Basic negamax alpha-beta search with a DummyEvalNetwork at leaf nodes.
 */

#ifndef CATGPT_ALPHA_BETA_ALPHA_BETA_SEARCH_HPP
#define CATGPT_ALPHA_BETA_ALPHA_BETA_SEARCH_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "../../external/chess-library/include/chess.hpp"
#include "dummy_evaluator.hpp"

namespace catgpt::alpha_beta {

struct SearchStats {
    std::int64_t nodes = 0;
};

struct AlphaBetaResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    int score_cp = 0;  // From root side-to-move perspective
    int depth = 0;
    std::int64_t nodes = 0;
    std::vector<chess::Move> pv;
};

class AlphaBetaSearch {
public:
    explicit AlphaBetaSearch(DummyEvalNetwork eval = {}) : eval_(std::move(eval)) {}

    [[nodiscard]] AlphaBetaResult search(chess::Board board, int max_depth) {
        SearchStats stats;
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        AlphaBetaResult result;
        result.depth = max_depth;

        if (moves.empty()) {
            result.score_cp = terminal_score(board, 0);
            result.nodes = 0;
            return result;
        }

        if (moves.size() == 1) {
            result.best_move = moves[0];
            board.makeMove<true>(moves[0]);
            result.score_cp = -terminal_score(board, 1);
            board.unmakeMove(moves[0]);
            result.nodes = 1;
            return result;
        }

        int alpha = -kInfinity;
        const int beta = kInfinity;
        int best_score = -kInfinity;
        chess::Move best_move = chess::Move::NO_MOVE;

        for (const chess::Move move : moves) {
            board.makeMove<true>(move);
            const int score = -negamax(board, max_depth - 1, -beta, -alpha, 1, stats);
            board.unmakeMove(move);

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }
            alpha = std::max(alpha, score);
        }

        result.best_move = best_move;
        result.score_cp = best_score;
        result.nodes = stats.nodes;
        result.pv = {best_move};
        return result;
    }

private:
    static constexpr int kInfinity = 100'000'000;

    DummyEvalNetwork eval_;

    [[nodiscard]] int terminal_score(const chess::Board& board, int ply) const noexcept {
        const auto [reason, game_result] = board.isGameOver();
        (void)reason;
        if (game_result == chess::GameResult::LOSE) {
            return -kMateScore + ply;
        }
        if (game_result == chess::GameResult::DRAW) {
            return 0;
        }
        return eval_.evaluate(board).centipawns;
    }

    [[nodiscard]] int negamax(chess::Board& board, int depth, int alpha, int beta, int ply,
                              SearchStats& stats) {
        const auto [reason, game_result] = board.isGameOver();
        (void)reason;

        if (game_result == chess::GameResult::LOSE) {
            ++stats.nodes;
            return -kMateScore + ply;
        }
        if (game_result == chess::GameResult::DRAW) {
            ++stats.nodes;
            return 0;
        }

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) {
            ++stats.nodes;
            return board.inCheck() ? -kMateScore + ply : 0;
        }

        if (depth <= 0) {
            ++stats.nodes;
            return eval_.evaluate(board).centipawns;
        }

        int best = -kInfinity;

        for (const chess::Move move : moves) {
            board.makeMove<true>(move);
            const int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, stats);
            board.unmakeMove(move);

            best = std::max(best, score);
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                break;
            }
        }

        return best;
    }
};

}  // namespace catgpt::alpha_beta

#endif  // CATGPT_ALPHA_BETA_ALPHA_BETA_SEARCH_HPP
