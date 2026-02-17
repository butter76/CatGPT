/**
 * Coroutine-based Fractional MCTS Search.
 *
 * This is the async version of FractionalMCTSSearch.  The algorithm
 * is identical — PUCT budget allocation, iterative deepening, binary
 * search for K — but every GPU evaluation point is a coroutine
 * suspension.  When evaluate_node() needs the neural network it
 * `co_await`s an EvalAwaitable, suspending the coroutine so the
 * worker thread can run a different game's search.  The GPU thread
 * batches these requests and resumes the coroutines when results
 * are ready.
 *
 * Each call to search_move() produces one best-move for one position
 * in one game.  The SelfPlayRunner manages multiple games.
 */

#ifndef CATGPT_SELFPLAY_COROUTINE_SEARCH_HPP
#define CATGPT_SELFPLAY_COROUTINE_SEARCH_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <coro/task.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "../engine/fractional_mcts/config.hpp"
#include "../engine/fractional_mcts/node.hpp"
#include "../engine/policy.hpp"
#include "../engine/search_result.hpp"
#include "../tokenizer.hpp"
#include "batch_evaluator.hpp"
#include "eval_request.hpp"

namespace catgpt {

/**
 * Result of a single move search (best move + metadata).
 */
struct MoveResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    int cp_score = 0;       // Centipawn evaluation from side-to-move perspective
    int gpu_evals = 0;      // GPU evaluations used
    int iterations = 0;     // Iterative deepening iterations
};

/**
 * Coroutine-based Fractional MCTS.
 *
 * Not a long-lived object — create one per search_move() call.
 * This keeps the design simple (no tree reuse between moves).
 */
class CoroutineSearch {
public:
    CoroutineSearch(BatchEvaluator& evaluator, const FractionalMCTSConfig& config)
        : evaluator_(evaluator)
        , config_(config)
        , total_gpu_evals_(0)
    {}

    /**
     * Search for the best move from the given position.
     * This is a coroutine — it will suspend whenever it needs a GPU eval.
     *
     * @param board The current position (will NOT be modified).
     * @return MoveResult with the best move and evaluation.
     */
    coro::task<MoveResult> search_move(chess::Board board) {
        MoveResult result;

        // Generate legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) {
            result.best_move = chess::Move::NO_MOVE;
            result.cp_score = board.inCheck() ? -32000 : 0;
            co_return result;
        }

        // Single legal move — return immediately
        if (moves.size() == 1) {
            result.best_move = moves[0];
            result.gpu_evals = 0;
            co_return result;
        }

        // Initialize root node
        auto root = std::make_unique<FractionalNode>();
        co_await evaluate_node(root.get(), board);

        // Determine target evals
        int target_evals = config_.min_total_evals;

        // Iterative deepening
        float N = config_.initial_budget;
        float last_used_N = N;
        int iteration = 0;

        while (total_gpu_evals_ < target_evals && iteration < 10000) {
            last_used_N = N;
            int alpha = compute_percentile_bin(root->distQ, 0.16f);
            int beta  = compute_percentile_bin(root->distQ, 0.84f);
            co_await recursive_search(root.get(), board, N, alpha, beta);
            N += 1.0f;
            ++iteration;
        }

        // Select best move by allocation
        auto final_allocations = compute_allocations(root.get(), last_used_N);

        chess::Move best_move = chess::Move::NO_MOVE;
        float best_allocation = -1.0f;
        for (const auto& [move, alloc] : final_allocations) {
            if (alloc > best_allocation) {
                best_allocation = alloc;
                best_move = move;
            }
        }

        if (best_move != chess::Move::NO_MOVE) {
            result.best_move = best_move;

            auto& chosen_child = root->children.at(best_move);
            float q = -chosen_child.Q;
            result.cp_score = static_cast<int>(90.0f * std::tan(q * 1.5637541897f));
        } else {
            result.best_move = moves[0];
        }

        result.gpu_evals = total_gpu_evals_;
        result.iterations = iteration;
        co_return result;
    }

private:
    // ─── Neural network evaluation (the suspension point) ───────────────

    /**
     * Evaluate a position via the GPU.
     * Suspends the coroutine until batched inference completes.
     */
    coro::task<void> evaluate_node(FractionalNode* node, const chess::Board& pos) {
        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);

        // co_await suspends here → GPU thread batches & evaluates
        RawNNOutput raw = co_await EvalAwaitable(evaluator_, tokens);

        // Post-process: convert value from [0,1] to [-1,1]
        float value = 2.0f * raw.value - 1.0f;

        // Build policy priors via softmax over legal moves
        bool flip = pos.sideToMove() == chess::Color::BLACK;

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, pos);

        std::vector<std::pair<chess::Move, float>> move_logits;
        move_logits.reserve(moves.size());

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = raw.policy[flat_idx];
            move_logits.emplace_back(move, logit);
        }

        // Softmax
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto& [move, logit] : move_logits) {
            max_logit = std::max(max_logit, logit);
        }
        float sum_exp = 0.0f;
        for (auto& [move, logit] : move_logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        for (const auto& [move, exp_logit] : move_logits) {
            policy_priors[move] = exp_logit / sum_exp;
        }

        // Write into node
        node->policy_priors = std::move(policy_priors);
        node->Q = value;
        node->value_probs = raw.value_probs;
        node->distQ = raw.value_probs;
        node->compute_variance();
        ++total_gpu_evals_;
    }

    // ─── Recursive search (identical logic to FractionalMCTSSearch) ─────

    coro::task<void> recursive_search(FractionalNode* node, chess::Board& scratch_board,
                                      float N, int alpha, int beta) {
        if (node->is_terminal) co_return;

        if (N <= node->max_N) co_return;
        node->max_N = N;

        // Depth reduction via variance
        constexpr float bin_width = 2.0f / VALUE_NUM_BINS;
        float vp_mean = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
            vp_mean += node->value_probs[i] * center;
        }
        float alt_variance = 0.0f;
        for (int i = alpha; i < VALUE_NUM_BINS; ++i) {
            float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
            float diff = center - vp_mean;
            alt_variance += node->value_probs[i] * diff * diff;
        }
        alt_variance *= 2.0f;

        float effective_variance = std::min(node->variance, alt_variance);
        float N_reduction = effective_variance * 18.0f;
        float effective_N = N * N_reduction;

        if (node->children.empty()) {
            int limit = node->get_limit(config_.policy_coverage_threshold,
                                        config_.single_node_coverage_threshold);
            if (effective_N < static_cast<float>(limit)) co_return;
        }

        float expansion_threshold = effective_N > 0.0f ? 1.0f / effective_N : 1.0f;
        co_await expand_children(node, scratch_board, expansion_threshold);

        if (node->children.empty()) co_return;

        // Scale N by total policy weight of expanded children
        float total_child_weight = 0.0f;
        for (const auto& [move, child] : node->children) {
            total_child_weight += child.P;
        }
        N *= total_child_weight;

        auto allocations = compute_allocations(node, N);

        int child_alpha = (VALUE_NUM_BINS - 1) - beta;
        int child_beta  = (VALUE_NUM_BINS - 1) - alpha;

        // Recurse into children
        for (auto& [move, child] : node->children) {
            auto it = allocations.find(move);
            if (it != allocations.end() && it->second > 0.0f) {
                float N_i = it->second;
                scratch_board.makeMove<true>(move);
                co_await recursive_search(&child, scratch_board, N_i, child_alpha, child_beta);
                scratch_board.unmakeMove(move);
            }
        }

        // Recompute allocations after recursion
        auto second_allocations = compute_allocations(node, N);

        // Update Q and distQ as weighted average of children
        float weighted_sum = 0.0f;
        float total_weight = 0.0f;
        std::array<float, VALUE_NUM_BINS> weighted_distQ{};
        weighted_distQ.fill(0.0f);

        for (auto& [move, child] : node->children) {
            auto it = second_allocations.find(move);
            if (it != second_allocations.end() && it->second > 0.0f) {
                float N_i = it->second;
                weighted_sum += (-child.Q) * N_i;
                for (int j = 0; j < VALUE_NUM_BINS; ++j) {
                    weighted_distQ[VALUE_NUM_BINS - 1 - j] += child.distQ[j] * N_i;
                }
                total_weight += N_i;
            }
        }

        if (total_weight > 0.0f) {
            node->Q = weighted_sum / total_weight;
            float inv_weight = 1.0f / total_weight;
            for (int j = 0; j < VALUE_NUM_BINS; ++j) {
                node->distQ[j] = weighted_distQ[j] * inv_weight;
            }
        }
    }

    // ─── Child expansion ────────────────────────────────────────────────

    coro::task<void> expand_children(FractionalNode* node, chess::Board& scratch_board,
                                     float threshold) {
        std::vector<std::pair<chess::Move, float>> sorted_priors;
        sorted_priors.reserve(node->policy_priors.size());
        for (const auto& [move, prior] : node->policy_priors) {
            sorted_priors.emplace_back(move, prior);
        }
        std::sort(sorted_priors.begin(), sorted_priors.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        int limit = node->get_limit(config_.policy_coverage_threshold,
                                    config_.single_node_coverage_threshold);

        for (size_t i = 0; i < sorted_priors.size(); ++i) {
            const auto& [move, prior] = sorted_priors[i];

            if (node->children.count(move) > 0) continue;

            bool in_limit = static_cast<int>(i) < limit;
            if (prior < threshold && !in_limit) continue;

            FractionalNode child(prior);

            scratch_board.makeMove<true>(move);

            auto [reason, game_result] = scratch_board.isGameOver();
            if (game_result != chess::GameResult::NONE) {
                child.is_terminal = true;
                child.distQ.fill(0.0f);
                if (game_result == chess::GameResult::LOSE) {
                    child.Q = -1.0f;
                    child.distQ[0] = 1.0f;
                } else {
                    child.Q = 0.0f;
                    child.distQ[VALUE_NUM_BINS / 2] = 1.0f;
                }
            } else {
                co_await evaluate_node(&child, scratch_board);
            }

            scratch_board.unmakeMove(move);
            node->children[move] = std::move(child);
        }
    }

    // ─── PUCT budget allocation (pure CPU, no suspension) ───────────────

    [[nodiscard]] static int compute_percentile_bin(
        const std::array<float, VALUE_NUM_BINS>& dist, float percentile) {
        float cumsum = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            cumsum += dist[i];
            if (cumsum >= percentile) return i;
        }
        return VALUE_NUM_BINS - 1;
    }

    std::unordered_map<chess::Move, float, MoveHash>
    compute_allocations(FractionalNode* node, float N) {
        std::unordered_map<chess::Move, float, MoveHash> allocations;
        if (node->children.empty()) return allocations;

        float c_puct = config_.c_puct;
        float sqrt_N = std::sqrt(N);

        std::vector<std::pair<chess::Move, FractionalNode*>> children_vec;
        children_vec.reserve(node->children.size());
        for (auto& [move, child] : node->children) {
            children_vec.emplace_back(move, &child);
        }

        auto compute_allocation = [c_puct, sqrt_N](FractionalNode* child, float K) -> float {
            float denominator = K - (-child->Q);
            if (denominator <= 0.0f) return std::numeric_limits<float>::infinity();
            return c_puct * child->P * sqrt_N / denominator;
        };

        auto sum_allocations = [&children_vec, &compute_allocation](float K) -> float {
            float total = 0.0f;
            for (const auto& [move, child] : children_vec) {
                float alloc = compute_allocation(child, K);
                if (std::isinf(alloc)) return std::numeric_limits<float>::infinity();
                total += alloc;
            }
            return total;
        };

        float max_q = -std::numeric_limits<float>::infinity();
        for (const auto& [move, child] : children_vec) {
            max_q = std::max(max_q, -child->Q);
        }
        float K_low = max_q + 1e-9f;
        float K_high = K_low + 10.0f;

        for (int i = 0; i < 100; ++i) {
            if (sum_allocations(K_high) <= N) break;
            K_high *= 2.0f;
        }

        for (int i = 0; i < 64; ++i) {
            float K_mid = (K_low + K_high) / 2.0f;
            if (sum_allocations(K_mid) > N) {
                K_low = K_mid;
            } else {
                K_high = K_mid;
            }
        }

        float K = (K_low + K_high) / 2.0f;
        for (const auto& [move, child] : children_vec) {
            allocations[move] = compute_allocation(child, K);
        }
        return allocations;
    }

    // ─── Members ────────────────────────────────────────────────────────

    BatchEvaluator& evaluator_;
    FractionalMCTSConfig config_;
    int total_gpu_evals_;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_COROUTINE_SEARCH_HPP
