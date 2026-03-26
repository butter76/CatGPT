/**
 * Coroutine-based Fractional MCTS Search.
 *
 * Coroutine-based Fractional MCTS using PUCT budget allocation,
 * iterative deepening, and binary search for K.  Every GPU
 * evaluation point is a coroutine
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
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <coro/task.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "../engine/fractional_mcts/config.hpp"
#include "../engine/fractional_mcts/node.hpp"
#include "../engine/fractional_mcts/search_stats.hpp"
#include "../engine/policy.hpp"
#include "../engine/search_result.hpp"
#include "../tokenizer.hpp"
#include "batch_evaluator.hpp"
#include "eval_request.hpp"

namespace catgpt {

struct TTEntry {
    float Q = 0.0f;
    std::array<float, VALUE_NUM_BINS> distQ{};
    float max_N = -1.0f;
};

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
    CoroutineSearch(BatchEvaluator& evaluator, const FractionalMCTSConfig& config,
                    std::ostream* stats_out = nullptr)
        : evaluator_(evaluator)
        , config_(config)
        , total_gpu_evals_(0)
        , stats_out_(stats_out)
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

        // ── Stats: root_eval ──
        if (stats_out_) {
            chess::Move initial_best = chess::Move::NO_MOVE;
            float best_prior = -1.0f;
            for (const auto& [move, prior] : root->policy_priors) {
                if (prior > best_prior) {
                    best_prior = prior;
                    initial_best = move;
                }
            }
            int root_cp = static_cast<int>(100.7066f * std::tan(root->Q * 1.5637541897f));
            std::unordered_map<chess::Move, float, MoveHash> empty_allocs;
            print_catgpt_stats(*stats_out_, "root_eval", root.get(), empty_allocs, 0.0f,
                              initial_best, root_cp, total_gpu_evals_, 0);
        }

        // Determine target evals
        int target_evals = config_.min_total_evals;

        // Iterative deepening
        float N = config_.initial_budget;
        float last_used_N = N;
        int iteration = 0;
        int last_stats_iteration = 0;  // Track when we last printed stats
        auto last_stats_time = std::chrono::steady_clock::now();  // Track time since last stats

        while (total_gpu_evals_ < target_evals && N < 250 * target_evals) {
            last_used_N = N;
            int median = compute_percentile_bin(root->distQ, 0.50f);
            int alpha = std::max(0, median);
            int beta  = std::min(VALUE_NUM_BINS - 1, median);
            co_await recursive_search(root.get(), board, N, alpha, beta);

            // ── Stats: print at iterations 0, 5, 12, 23, 39, ... (1.5x + 5 progression)
            //           OR every 5 seconds minimum to prevent web timeouts ──
            int next_stats_threshold = static_cast<int>(1.5f * last_stats_iteration + 5);
            auto now = std::chrono::steady_clock::now();
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                now - last_stats_time).count();
            bool time_triggered = elapsed_seconds >= 5;

            if (stats_out_ && !root->children.empty() &&
                (iteration >= next_stats_threshold || time_triggered)) {
                std::unordered_map<chess::Move, float, MoveHash> allocs;
                float N_adj = 0.0f;
                compute_root_stats_allocations(
                    root.get(), N,
                    [this](FractionalNode* node, float budget) {
                        return compute_allocations(node, budget);
                    },
                    allocs, N_adj);

                chess::Move current_best = best_move_from_allocations(allocs);
                if (current_best != chess::Move::NO_MOVE) {
                    int cp = child_q_to_cp(root->children.at(current_best).Q);
                    auto pv = root->get_pv(
                        [this](FractionalNode* node, float budget) {
                            return compute_allocations(node, budget);
                        });
                    print_catgpt_stats(*stats_out_, "search_update", root.get(), allocs, N_adj,
                                      current_best, cp, total_gpu_evals_, iteration, pv);
                    last_stats_iteration = iteration;
                    last_stats_time = now;
                }
            }

            N *= 1.02f;
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
            result.cp_score = static_cast<int>(100.7066f * std::tan(q * 1.5637541897f));

            // ── Stats: search_complete ──
            if (stats_out_) {
                std::unordered_map<chess::Move, float, MoveHash> allocs;
                float N_adj = 0.0f;
                compute_root_stats_allocations(
                    root.get(), last_used_N,
                    [this](FractionalNode* node, float budget) {
                        return compute_allocations(node, budget);
                    },
                    allocs, N_adj);
                auto pv = root->get_pv(
                    [this](FractionalNode* node, float budget) {
                        return compute_allocations(node, budget);
                    });
                print_catgpt_stats(*stats_out_, "search_complete", root.get(), allocs, N_adj,
                                  best_move, result.cp_score, total_gpu_evals_, iteration, pv);
            }
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
     * Results are cached by zobrist hash to avoid redundant GPU evals.
     */
    coro::task<void> evaluate_node(FractionalNode* node, const chess::Board& pos) {
        uint64_t hash = pos.hash();

        // Check eval cache — reuse previous GPU result for the same position
        auto cache_it = eval_cache_.find(hash);
        RawNNOutput raw;
        if (cache_it != eval_cache_.end()) {
            raw = cache_it->second;
        } else {
            auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);

            // co_await suspends here → GPU thread batches & evaluates
            raw = co_await EvalAwaitable(evaluator_, tokens);
            eval_cache_[hash] = raw;
            ++total_gpu_evals_;
        }

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

        // Softmax at temp 1.0 (standard)
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto& [move, logit] : move_logits) {
            max_logit = std::max(max_logit, logit);
        }

        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        {
            float sum_exp = 0.0f;
            std::vector<std::pair<chess::Move, float>> exps;
            exps.reserve(move_logits.size());
            for (const auto& [move, logit] : move_logits) {
                float e = std::exp(logit - max_logit);
                exps.emplace_back(move, e);
                sum_exp += e;
            }
            for (const auto& [move, e] : exps) {
                policy_priors[move] = e / sum_exp;
            }
        }

        // Softmax at temp 1.2 (warmer — used only for PUCT allocation)
        constexpr float alloc_temp = 1.2f;
        constexpr float inv_alloc_temp = 1.0f / alloc_temp;
        std::unordered_map<chess::Move, float, MoveHash> policy_priors_alloc;
        {
            float max_scaled = max_logit * inv_alloc_temp;
            float sum_exp = 0.0f;
            std::vector<std::pair<chess::Move, float>> exps;
            exps.reserve(move_logits.size());
            for (const auto& [move, logit] : move_logits) {
                float e = std::exp(logit * inv_alloc_temp - max_scaled);
                exps.emplace_back(move, e);
                sum_exp += e;
            }
            for (const auto& [move, e] : exps) {
                policy_priors_alloc[move] = e / sum_exp;
            }
        }

        // Optimistic policy (from NN's optimistic_policy head, temp 1.2)
        std::unordered_map<chess::Move, float, MoveHash> policy_priors_optimistic;
        if (raw.has_optimistic_policy) {
            std::vector<std::pair<chess::Move, float>> opt_move_logits;
            opt_move_logits.reserve(moves.size());

            for (const auto& move : moves) {
                auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
                int flat_idx = policy_flat_index(from_idx, to_idx);
                float logit = raw.optimistic_policy[flat_idx];
                opt_move_logits.emplace_back(move, logit);
            }

            float opt_max_logit = -std::numeric_limits<float>::infinity();
            for (const auto& [move, logit] : opt_move_logits) {
                opt_max_logit = std::max(opt_max_logit, logit);
            }

            float opt_max_scaled = opt_max_logit * inv_alloc_temp;
            float sum_exp = 0.0f;
            std::vector<std::pair<chess::Move, float>> exps;
            exps.reserve(opt_move_logits.size());
            for (const auto& [move, logit] : opt_move_logits) {
                float e = std::exp(logit * inv_alloc_temp - opt_max_scaled);
                exps.emplace_back(move, e);
                sum_exp += e;
            }
            for (const auto& [move, e] : exps) {
                policy_priors_optimistic[move] = e / sum_exp;
            }
        }

        // Write into node
        node->policy_priors = std::move(policy_priors);
        node->policy_priors_alloc = std::move(policy_priors_alloc);
        node->policy_priors_optimistic = std::move(policy_priors_optimistic);
        node->Q = value;
        node->value_probs = raw.value_probs;
        node->distQ = raw.value_probs;
        node->compute_variance();
    }

    // ─── Recursive search ─────────────────────────────────────────────

    coro::task<void> recursive_search(FractionalNode* node, chess::Board& scratch_board,
                                      float N, int alpha, int beta) {
        if (node->is_terminal) co_return;

        if (N <= node->max_N) co_return;
        node->max_N = N;

        uint64_t pos_hash = scratch_board.hash();
        {
            auto tt_it = tt_.find(pos_hash);
            if (tt_it != tt_.end() && tt_it->second.max_N >= N) {
                node->Q = tt_it->second.Q;
                node->distQ = tt_it->second.distQ;
                node->max_N = tt_it->second.max_N;
                co_return;
            }
        }

        float effective_variance = node->variance;
        float N_reduction = effective_variance * 12.0f;
        float effective_N = N * N_reduction;

        if (node->children.empty()) {
            int limit = node->get_limit(config_.policy_coverage_threshold,
                                        config_.single_node_coverage_threshold);
            if (effective_N < static_cast<float>(limit)) co_return;
        }

        float expansion_N = N * node->variance * 12.0f;
        co_await expand_children(node, scratch_board, expansion_N);

        if (node->children.empty()) co_return;

        // Scale N by total policy weight of expanded children
        float total_child_weight = 0.0f;
        for (const auto& [move, child] : node->children) {
            total_child_weight += child.P;
        }
        N *= total_child_weight;

        int child_alpha = (VALUE_NUM_BINS - 1) - beta;
        int child_beta  = (VALUE_NUM_BINS - 1) - alpha;

        // Recurse into children with clamped allocation loop.
        // If any child's allocation exceeds its limit we cap it, recurse,
        // then recompute allocations (Q values and max_N have changed) and
        // repeat until allocations stabilise or we hit the iteration cap.
        // After the first iteration, only recurse into children that would be clamped.
        // Use optimistic policy for exploration during the recursion.
        for (int clamp_iter = 0; clamp_iter < 100; ++clamp_iter) {
            auto allocations = compute_allocations(node, N, /*use_optimistic=*/true);

            // Identify which children would be clamped and apply the clamp
            std::vector<chess::Move> clamped_moves;
            for (auto& [move, child] : node->children) {
                auto it = allocations.find(move);
                if (it == allocations.end() || it->second <= 0.0f) continue;

                float limit = std::max(child.max_N, 0.1f) * 1.1f;
                if (it->second > limit) {
                    it->second = limit;
                    clamped_moves.push_back(move);
                }
            }

            if (clamp_iter == 0) {
                // First iteration: recurse into all children
                for (auto& [move, child] : node->children) {
                    auto it = allocations.find(move);
                    if (it != allocations.end() && it->second > 0.0f) {
                        float N_i = it->second;
                        scratch_board.makeMove<true>(move);
                        co_await recursive_search(&child, scratch_board, N_i, child_alpha, child_beta);
                        scratch_board.unmakeMove(move);
                    }
                }
            } else {
                // Subsequent iterations: only recurse into clamped children
                if (clamped_moves.empty()) break;

                for (const auto& move : clamped_moves) {
                    auto& child = node->children.at(move);
                    float N_i = allocations.at(move);
                    scratch_board.makeMove<true>(move);
                    co_await recursive_search(&child, scratch_board, N_i, child_alpha, child_beta);
                    scratch_board.unmakeMove(move);
                }
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

        auto& tt_entry = tt_[pos_hash];
        if (node->max_N > tt_entry.max_N) {
            tt_entry.Q = node->Q;
            tt_entry.distQ = node->distQ;
            tt_entry.max_N = node->max_N;
        }
    }

    // ─── Child expansion ────────────────────────────────────────────────

    coro::task<void> expand_children(FractionalNode* node, chess::Board& scratch_board,
                                     float N) {
        if (N <= 0.0f) co_return;

        // Build list of ALL legal moves with optimistic policy, sorted descending
        struct MoveInfo {
            chess::Move move;
            float P_optimistic;
            float Q_effective;
            bool is_expanded;
        };
        std::vector<MoveInfo> all_moves;
        all_moves.reserve(node->policy_priors.size());

        for (const auto& [move, prior] : node->policy_priors) {
            float p_opt = prior;
            auto opt_it = node->policy_priors_optimistic.find(move);
            if (opt_it != node->policy_priors_optimistic.end()) {
                p_opt = opt_it->second;
            }
            all_moves.push_back({move, p_opt, 0.0f, false});
        }
        std::sort(all_moves.begin(), all_moves.end(),
                  [](const auto& a, const auto& b) { return a.P_optimistic > b.P_optimistic; });

        // Assign Q values: actual Q for expanded, FPU for unexpanded
        float cumulative_policy = 0.0f;
        for (auto& info : all_moves) {
            auto child_it = node->children.find(info.move);
            if (child_it != node->children.end()) {
                info.Q_effective = -child_it->second.Q;
                info.is_expanded = true;
            } else {
                info.Q_effective = node->Q - config_.fpu_reduction * std::sqrt(cumulative_policy);
            }
            cumulative_policy += info.P_optimistic;
        }

        // Binary search for K such that sum of allocations = N
        float c_puct = config_.c_puct;
        float N_exp = std::sqrt(N);

        auto compute_alloc = [c_puct, N_exp](const MoveInfo& info, float K) -> float {
            float denom = K - info.Q_effective;
            if (denom <= 0.0f) return std::numeric_limits<float>::infinity();
            return c_puct * info.P_optimistic * N_exp / denom;
        };

        auto sum_allocs = [&all_moves, &compute_alloc](float K) -> float {
            float total = 0.0f;
            for (const auto& info : all_moves) {
                float a = compute_alloc(info, K);
                if (std::isinf(a)) return std::numeric_limits<float>::infinity();
                total += a;
            }
            return total;
        };

        float max_q = -std::numeric_limits<float>::infinity();
        for (const auto& info : all_moves) {
            max_q = std::max(max_q, info.Q_effective);
        }
        float K_low = max_q + 1e-9f;
        float K_high = K_low + 10.0f;

        for (int i = 0; i < 100; ++i) {
            if (sum_allocs(K_high) <= N) break;
            K_high *= 2.0f;
        }
        for (int i = 0; i < 64; ++i) {
            float K_mid = (K_low + K_high) / 2.0f;
            if (sum_allocs(K_mid) > N) {
                K_low = K_mid;
            } else {
                K_high = K_mid;
            }
        }
        float K = (K_low + K_high) / 2.0f;

        // Determine which moves are in the policy-coverage limit (by standard prior)
        int limit = node->get_limit(config_.policy_coverage_threshold,
                                    config_.single_node_coverage_threshold);
        std::vector<std::pair<chess::Move, float>> sorted_standard;
        sorted_standard.reserve(node->policy_priors.size());
        for (const auto& [move, prior] : node->policy_priors) {
            sorted_standard.emplace_back(move, prior);
        }
        std::sort(sorted_standard.begin(), sorted_standard.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::unordered_map<chess::Move, bool, MoveHash> in_limit;
        for (int i = 0; i < limit && i < static_cast<int>(sorted_standard.size()); ++i) {
            in_limit[sorted_standard[i].first] = true;
        }

        // Expand unexpanded children with allocation >= 1, or within the limit
        for (const auto& info : all_moves) {
            if (info.is_expanded) continue;
            bool force = in_limit.count(info.move) > 0;
            float alloc = compute_alloc(info, K);
            if (alloc < 1.0f && !force) continue;

            float prior = node->policy_priors.at(info.move);
            float prior_alloc = prior;
            auto alloc_it = node->policy_priors_alloc.find(info.move);
            if (alloc_it != node->policy_priors_alloc.end()) {
                prior_alloc = alloc_it->second;
            }
            float prior_optimistic = prior_alloc;
            auto opt_it = node->policy_priors_optimistic.find(info.move);
            if (opt_it != node->policy_priors_optimistic.end()) {
                prior_optimistic = opt_it->second;
            }
            FractionalNode child(prior, prior_alloc, prior_optimistic);

            scratch_board.makeMove<true>(info.move);

            bool is_twofold = scratch_board.isRepetition(1);
            auto [reason, game_result] = scratch_board.isGameOver();

            if (is_twofold || game_result != chess::GameResult::NONE) {
                child.is_terminal = true;
                child.distQ.fill(0.0f);
                if (!is_twofold && game_result == chess::GameResult::LOSE) {
                    child.Q = -1.0f;
                    child.distQ[0] = 1.0f;
                } else {
                    child.Q = 0.0f;
                    child.distQ[VALUE_NUM_BINS / 2] = 1.0f;
                }
            } else {
                co_await evaluate_node(&child, scratch_board);
            }

            scratch_board.unmakeMove(info.move);
            node->children[info.move] = std::move(child);
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
    compute_allocations(FractionalNode* node, float N, bool use_optimistic = false) {
        std::unordered_map<chess::Move, float, MoveHash> allocations;
        if (node->children.empty()) return allocations;

        float c_puct = config_.c_puct;
        float N_exp = std::pow(N, 0.666f) / 2.5f;

        std::vector<std::pair<chess::Move, FractionalNode*>> children_vec;
        children_vec.reserve(node->children.size());
        for (auto& [move, child] : node->children) {
            children_vec.emplace_back(move, &child);
        }

        auto compute_allocation = [c_puct, N_exp, use_optimistic](FractionalNode* child, float K) -> float {
            float denominator = K - (-child->Q);
            if (denominator <= 0.0f) return std::numeric_limits<float>::infinity();
            float P = use_optimistic ? child->P_optimistic : child->P_alloc;
            return c_puct * P * N_exp / denominator;
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
    std::unordered_map<uint64_t, RawNNOutput> eval_cache_;
    std::unordered_map<uint64_t, TTEntry> tt_;
    std::ostream* stats_out_;  // If non-null, JSON stats lines are written here during search
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_COROUTINE_SEARCH_HPP
