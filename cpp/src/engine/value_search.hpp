/**
 * Value-based Search Algorithm.
 *
 * Chess engine using a value network with 1-move lookahead.
 * This mirrors the Python implementation in:
 *   src/catgpt/jax/evaluation/engines/value_engine.py
 *
 * The engine evaluates positions by looking one move ahead:
 *   1. For each legal move, compute the resulting position
 *   2. If checkmate: return immediately (winning move)
 *   3. If stalemate/draw: assign score 0.5
 *   4. Otherwise: evaluate position with value network
 *
 * The engine picks the move that minimizes the opponent's expected score
 * (i.e., maximizes our own winning probability).
 *
 * The model evaluates positions from the perspective of the side to move,
 * returning the probability that the side to move wins. Since we're evaluating
 * positions after our move (opponent to move), a lower score for the opponent
 * means a better position for us.
 */

#ifndef CATGPT_ENGINE_VALUE_SEARCH_HPP
#define CATGPT_ENGINE_VALUE_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <vector>

#include "../../external/chess-library/include/chess.hpp"
#include "../tokenizer.hpp"
#include "search_algo.hpp"
#include "trt_evaluator.hpp"

namespace catgpt {

/**
 * Value-based search using 1-move lookahead.
 *
 * For each legal move:
 *   - If it leads to checkmate: best possible move
 *   - If it leads to draw: score 0.5
 *   - Otherwise: evaluate with value network
 *
 * Select the move that minimizes opponent's win probability.
 */
class ValueSearch : public SearchAlgo {
public:
    /**
     * Construct ValueSearch with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     */
    explicit ValueSearch(std::shared_ptr<TrtEvaluator> evaluator)
        : evaluator_(std::move(evaluator))
        , board_(STARTPOS_FEN)
        , stop_flag_(false)
    {}

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
    }

    SearchResult search(const SearchLimits& /* limits */) override {
        stop_flag_.store(false, std::memory_order_relaxed);

        auto start_time = std::chrono::steady_clock::now();

        // Generate legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);

        SearchResult result;
        std::int64_t nodes_evaluated = 0;

        if (moves.empty()) {
            // No legal moves - checkmate or stalemate
            result.best_move = chess::Move::NO_MOVE;
            if (board_.inCheck()) {
                result.score = Score::mate(0);  // We are checkmated
            } else {
                result.score = Score::cp(0);  // Stalemate
            }
            return result;
        }

        // Single legal move - return immediately
        if (moves.size() == 1) {
            result.best_move = moves[0];
            result.depth = 1;
            result.nodes = 1;
            return result;
        }

        // Evaluate all moves with 1-move lookahead
        chess::Move best_move = chess::Move::NO_MOVE;
        float best_opponent_score = std::numeric_limits<float>::infinity();

        for (const auto& move : moves) {
            if (stop_flag_.load(std::memory_order_relaxed)) {
                break;
            }

            board_.makeMove<true>(move);

            float opponent_score;
            auto [reason, game_result] = board_.isGameOver();

            if (game_result == chess::GameResult::LOSE) {
                // Opponent is checkmated - instant win!
                board_.unmakeMove(move);
                result.best_move = move;
                result.score = Score::mate(1);
                result.depth = 1;
                result.nodes = nodes_evaluated + 1;

                auto end_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                result.time_ms = elapsed.count();

                return result;
            } else if (game_result == chess::GameResult::DRAW) {
                // Draw - score 0.5 for opponent (neutral for us)
                opponent_score = 0.5f;
            } else {
                // Evaluate position with neural network
                opponent_score = evaluate_position(board_);
                ++nodes_evaluated;
            }

            board_.unmakeMove(move);

            // Lower opponent score = better for us
            if (opponent_score < best_opponent_score) {
                best_opponent_score = opponent_score;
                best_move = move;
            }
        }

        result.best_move = best_move;

        // Convert opponent's win probability to centipawns from our perspective
        // Our win probability = 1 - opponent's win probability
        // Map [0, 1] to centipawns (rough approximation)
        float our_win_prob = 1.0f - best_opponent_score;
        int cp = static_cast<int>((our_win_prob - 0.5f) * 200.0f);  // [-100, 100]
        result.score = Score::cp(cp);

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = 1;
        result.nodes = nodes_evaluated;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (nodes_evaluated * 1000) / elapsed.count();
        }

        result.pv = std::vector<chess::Move>{best_move};

        return result;
    }

    void stop() override {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

    /**
     * Evaluate a single position with the value network.
     *
     * @param pos Position to evaluate.
     * @return Win probability for the side to move (0.0 to 1.0).
     */
    [[nodiscard]] float evaluate_position(const chess::Board& pos) const {
        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);
        auto nn_output = evaluator_->evaluate(tokens);
        return nn_output.value;  // Already [0, 1]
    }

private:
    std::shared_ptr<TrtEvaluator> evaluator_;
    chess::Board board_;
    std::atomic<bool> stop_flag_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_VALUE_SEARCH_HPP
