/**
 * Policy-based Search Algorithm.
 *
 * Chess engine that selects the move with highest policy probability.
 * This mirrors the Python implementation in:
 *   src/catgpt/jax/evaluation/engines/policy_engine.py
 *
 * This engine uses the policy head output directly without any search:
 *   1. Run the model to get policy logits (64*73 = 4672 values)
 *   2. For each legal move, look up its policy logit
 *   3. Select the move with the highest logit
 *
 * This is a "pure policy" engine useful for evaluating raw policy quality
 * without the influence of value-based move ordering or search.
 *
 * Note: Requires the model to have policy_head enabled.
 */

#ifndef CATGPT_ENGINE_POLICY_SEARCH_HPP
#define CATGPT_ENGINE_POLICY_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "../../external/chess-library/include/chess.hpp"
#include "../tokenizer.hpp"
#include "policy.hpp"
#include "search_algo.hpp"
#include "trt_evaluator.hpp"

namespace catgpt {

/**
 * Policy-based search that selects the move with highest policy probability.
 *
 * Simply queries the policy head and picks the legal move with highest logit.
 * No lookahead, no value evaluation - pure policy selection.
 */
class PolicySearch : public SearchAlgo {
public:
    /**
     * Construct PolicySearch with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     */
    explicit PolicySearch(std::shared_ptr<TrtEvaluator> evaluator)
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

        // Get policy logits from neural network
        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(board_, NO_HALFMOVE_CONFIG);
        auto nn_output = evaluator_->evaluate(tokens);

        // Determine if we need to flip (tokenizer flips for black to move)
        bool flip = board_.sideToMove() == chess::Color::BLACK;

        // Find move with highest policy logit
        chess::Move best_move = chess::Move::NO_MOVE;
        float best_logit = -std::numeric_limits<float>::infinity();
        float best_prob = 0.0f;

        // First pass: collect logits and find max for softmax
        std::vector<std::pair<chess::Move, float>> move_logits;
        move_logits.reserve(moves.size());
        float max_logit = -std::numeric_limits<float>::infinity();

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = nn_output.policy[flat_idx];

            move_logits.emplace_back(move, logit);
            max_logit = std::max(max_logit, logit);

            if (logit > best_logit) {
                best_logit = logit;
                best_move = move;
            }
        }

        // Compute softmax probability for the best move (for reporting)
        float sum_exp = 0.0f;
        float best_exp = 0.0f;
        for (const auto& [move, logit] : move_logits) {
            float exp_logit = std::exp(logit - max_logit);  // Numerical stability
            sum_exp += exp_logit;
            if (move == best_move) {
                best_exp = exp_logit;
            }
        }
        best_prob = best_exp / sum_exp;

        result.best_move = best_move;

        // Convert probability to centipawns (rough mapping)
        // Higher confidence = more positive score
        // prob=1.0 -> +100cp, prob=0.5 -> 0cp, prob=0.0 -> -100cp (roughly)
        int cp = static_cast<int>((best_prob - 0.5f) * 200.0f);
        result.score = Score::cp(cp);

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = 1;
        result.nodes = 1;  // Single NN evaluation
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = 1000 / elapsed.count();
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
     * Get scores for all legal moves (for analysis/debugging).
     *
     * @return Vector of (move, logit) pairs sorted by logit (best first).
     */
    [[nodiscard]] std::vector<std::pair<chess::Move, float>> get_move_scores() const {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);

        if (moves.empty()) {
            return {};
        }

        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(board_, NO_HALFMOVE_CONFIG);
        auto nn_output = evaluator_->evaluate(tokens);
        bool flip = board_.sideToMove() == chess::Color::BLACK;

        std::vector<std::pair<chess::Move, float>> move_scores;
        move_scores.reserve(moves.size());

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = nn_output.policy[flat_idx];
            move_scores.emplace_back(move, logit);
        }

        // Sort by logit descending (best first)
        std::sort(move_scores.begin(), move_scores.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        return move_scores;
    }

    /**
     * Get softmax probabilities for all legal moves.
     *
     * @return Vector of (move, probability) pairs sorted by probability (best first).
     */
    [[nodiscard]] std::vector<std::pair<chess::Move, float>> get_move_probabilities() const {
        auto move_scores = get_move_scores();
        if (move_scores.empty()) {
            return {};
        }

        // Find max logit for numerical stability
        float max_logit = move_scores[0].second;  // Already sorted descending

        // Compute softmax
        float sum_exp = 0.0f;
        for (auto& [move, logit] : move_scores) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        for (auto& [move, exp_val] : move_scores) {
            exp_val /= sum_exp;
        }

        return move_scores;
    }

private:
    std::shared_ptr<TrtEvaluator> evaluator_;
    chess::Board board_;
    std::atomic<bool> stop_flag_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_POLICY_SEARCH_HPP
