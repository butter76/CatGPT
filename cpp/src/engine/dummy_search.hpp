/**
 * DummySearch - Minimal SearchAlgo implementation for testing.
 *
 * This is a placeholder implementation that simply returns the first
 * legal move. It's useful for testing the UCI infrastructure before
 * implementing a real search algorithm.
 */

#ifndef CATGPT_ENGINE_DUMMY_SEARCH_HPP
#define CATGPT_ENGINE_DUMMY_SEARCH_HPP

#include <atomic>
#include <chrono>

#include "../../external/chess-library/include/chess.hpp"
#include "search_algo.hpp"

namespace catgpt {

class DummySearch : public SearchAlgo {
public:
    DummySearch() : board_(STARTPOS_FEN), stop_flag_(false) {}

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
    }

    void makemove(const chess::Move& move) override {
        // Use makeMove<true> to ensure en passant squares are only set when legal
        // (as per the chess-library guidelines in the repo)
        board_.makeMove<true>(move);
    }

    SearchResult search(const SearchLimits& /* limits */) override {
        stop_flag_.store(false, std::memory_order_relaxed);

        auto start_time = std::chrono::steady_clock::now();

        // Generate all legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);

        SearchResult result;

        if (moves.empty()) {
            // No legal moves - checkmate or stalemate
            result.best_move = chess::Move::NO_MOVE;
            if (board_.inCheck()) {
                // Checkmate - we lose
                result.score = Score::mate(-0);
            } else {
                // Stalemate
                result.score = Score::cp(0);
            }
        } else {
            // Return the first legal move
            result.best_move = moves[0];
            result.score = Score::cp(0);  // Dummy evaluation

            // Build a trivial PV with just the best move
            result.pv = std::vector<chess::Move>{result.best_move};
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        result.depth = 1;
        result.seldepth = 1;
        result.nodes = static_cast<std::int64_t>(moves.size());
        result.time_ms = elapsed.count();
        result.nps = elapsed.count() > 0
            ? (result.nodes.value() * 1000) / elapsed.count()
            : result.nodes.value();

        return result;
    }

    void stop() override {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

private:
    chess::Board board_;
    std::atomic<bool> stop_flag_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_DUMMY_SEARCH_HPP
