/**
 * Fractional MCTS Search Algorithm.
 *
 * A novel MCTS variant where:
 * - Visit counts N are fractional (floats) rather than integers
 * - Search uses iterative deepening with increasing budget N
 * - Budget is allocated to children by solving the PUCT equation for equal "urgency"
 *
 * The algorithm:
 * 1. Initialize root with GPU eval (gets policy priors and initial Q)
 * 2. Run iterative deepening with N = initial * (multiplier ^ iteration)
 * 3. Each iteration recursively allocates budget to children
 * 4. Stop when total GPU evals >= min_total_evals
 *
 * For a node with budget N:
 * - Compute "limit" = number of children covering 80% of policy mass
 * - If N < limit: return cached Q (base case, no expansion)
 * - Otherwise: expand children with P >= 1/N, allocate budget via binary search,
 *   recurse, then update Q as weighted average of children's Q values
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <print>
#include <unordered_map>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../search_algo.hpp"
#include "../trt_evaluator.hpp"
#include "config.hpp"
#include "node.hpp"
#include "search_stats.hpp"

namespace catgpt {

/**
 * Fractional MCTS search with iterative deepening.
 */
class FractionalMCTSSearch : public SearchAlgo {
public:
    /**
     * Construct Fractional MCTS search with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     * @param config Fractional MCTS configuration.
     */
    FractionalMCTSSearch(std::shared_ptr<TrtEvaluator> evaluator, FractionalMCTSConfig config = {})
        : evaluator_(std::move(evaluator))
        , config_(config)
        , board_(STARTPOS_FEN)
        , stop_flag_(false)
        , total_gpu_evals_(0)
        , stats_out_(nullptr)
    {}

    /**
     * Enable search stats printing.
     * When set, JSON stats lines are written to `out` during search().
     */
    void set_stats_output(std::ostream& out) {
        stats_out_ = &out;
    }

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
        root_.reset();
        total_gpu_evals_ = 0;
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
        root_.reset();  // Invalidate tree after move
        total_gpu_evals_ = 0;
    }

    SearchResult search(const SearchLimits& limits) override {
        stop_flag_.store(false, std::memory_order_relaxed);
        total_gpu_evals_ = 0;

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

        // Initialize root node and evaluate it
        root_ = std::make_unique<FractionalNode>();
        evaluate_node(root_.get(), board_);

        // ── Stats: root_eval (initial NN evaluation, raw policy) ──
        if (stats_out_) {
            // Best move = highest prior (no search yet)
            chess::Move initial_best = chess::Move::NO_MOVE;
            float best_prior = -1.0f;
            for (const auto& [move, prior] : root_->policy_priors) {
                if (prior > best_prior) {
                    best_prior = prior;
                    initial_best = move;
                }
            }
            int root_cp = static_cast<int>(90.0f * std::tan(root_->Q * 1.5637541897f));
            std::unordered_map<chess::Move, float, MoveHash> empty_allocs;
            print_catgpt_stats(*stats_out_, "root_eval", root_.get(), empty_allocs, 0.0f,
                              initial_best, root_cp, total_gpu_evals_, 0);
        }

        // Determine target evals (can be limited by search limits)
        int target_evals = config_.min_total_evals;
        if (limits.nodes.has_value()) {
            target_evals = std::min(target_evals, static_cast<int>(limits.nodes.value()));
        }

        // Run iterative deepening
        float N = config_.initial_budget;
        float last_used_N = N;
        int iteration = 0;
        chess::Move stats_prev_best = chess::Move::NO_MOVE;

        while (total_gpu_evals_ < target_evals && iteration < 10000) {
            if (stop_flag_.load(std::memory_order_relaxed)) {
                break;
            }

            // Check time limit
            if (limits.movetime.has_value()) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
                if (elapsed.count() >= limits.movetime.value()) {
                    break;
                }
            }

            last_used_N = N;
            int alpha = compute_percentile_bin(root_->distQ, 0.16f);
            int beta  = compute_percentile_bin(root_->distQ, 0.84f);
            recursive_search(root_.get(), board_, N, alpha, beta);

            // ── Stats: check if best move changed ──
            if (stats_out_ && !root_->children.empty()) {
                std::unordered_map<chess::Move, float, MoveHash> allocs;
                float N_adj = 0.0f;
                compute_root_stats_allocations(
                    root_.get(), N,
                    [this](FractionalNode* node, float budget) {
                        return compute_allocations(node, budget);
                    },
                    allocs, N_adj);

                chess::Move current_best = best_move_from_allocations(allocs);
                if (current_best != stats_prev_best && current_best != chess::Move::NO_MOVE) {
                    stats_prev_best = current_best;
                    int cp = child_q_to_cp(root_->children.at(current_best).Q);
                    print_catgpt_stats(*stats_out_, "search_update", root_.get(), allocs, N_adj,
                                      current_best, cp, total_gpu_evals_, iteration);
                }
            }

            N += 1.0f;
            ++iteration;
        }

        // ── Stats: search_complete ──
        float stats_N_adj_final = 0.0f;
        std::unordered_map<chess::Move, float, MoveHash> stats_allocs_final;
        if (stats_out_ && !root_->children.empty()) {
            compute_root_stats_allocations(
                root_.get(), last_used_N,
                [this](FractionalNode* node, float budget) {
                    return compute_allocations(node, budget);
                },
                stats_allocs_final, stats_N_adj_final);
        }

        // Select best move by allocation (using the last N that was actually used)
        auto final_allocations = compute_allocations(root_.get(), last_used_N);

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

            // Get Q for score reporting from the chosen child
            auto& chosen_child = root_->children.at(best_move);
            float q = -chosen_child.Q;
            // Convert Q from [-1, 1] to centipawns using tangent scaling
            int cp = static_cast<int>(90.0f * std::tan(q * 1.5637541897f));
            result.score = Score::cp(cp);

            // ── Stats: search_complete ──
            if (stats_out_) {
                print_catgpt_stats(*stats_out_, "search_complete", root_.get(),
                                  stats_allocs_final, stats_N_adj_final,
                                  best_move, cp, total_gpu_evals_, iteration);
            }
        } else {
            // Fallback (shouldn't happen)
            result.best_move = moves[0];
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = iteration;
        result.nodes = total_gpu_evals_;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (total_gpu_evals_ * 1000) / elapsed.count();
        }

        // Build PV from best-Q path
        result.pv = root_->get_pv();

        return result;
    }

    void stop() override {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

    /**
     * Get the root node for analysis (after search).
     */
    [[nodiscard]] const FractionalNode* root() const {
        return root_.get();
    }

    /**
     * Get total GPU evaluations from last search.
     */
    [[nodiscard]] int total_evals() const {
        return total_gpu_evals_;
    }

private:
    /**
     * Result of neural network evaluation.
     */
    struct EvalResult {
        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        float value;  // [-1, 1]
        std::array<float, VALUE_NUM_BINS> value_probs;
    };

    /**
     * Evaluate a node with the neural network.
     * Populates node->policy_priors, node->Q, and node->value_probs.
     */
    void evaluate_node(FractionalNode* node, const chess::Board& pos) {
        auto eval = run_neural_network(pos);
        node->policy_priors = std::move(eval.policy_priors);
        node->Q = eval.value;
        node->value_probs = eval.value_probs;
        node->distQ = eval.value_probs;  // distQ starts as value_probs
        node->compute_variance();
        ++total_gpu_evals_;
    }

    /**
     * Run the neural network on a position.
     */
    EvalResult run_neural_network(const chess::Board& pos) {
        // Tokenize
        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);

        // Run neural network
        auto nn_output = evaluator_->evaluate(tokens);

        // Convert value from [0, 1] to [-1, 1]
        float value = 2.0f * nn_output.value - 1.0f;

        // Extract policy priors for legal moves
        bool flip = pos.sideToMove() == chess::Color::BLACK;

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, pos);

        // Collect logits for legal moves
        std::vector<std::pair<chess::Move, float>> move_logits;
        move_logits.reserve(moves.size());

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = nn_output.policy[flat_idx];
            move_logits.emplace_back(move, logit);
        }

        // Softmax over legal moves only
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto& [move, logit] : move_logits) {
            max_logit = std::max(max_logit, logit);
        }

        float sum_exp = 0.0f;
        for (auto& [move, logit] : move_logits) {
            logit = std::exp(logit - max_logit);  // Numerical stability
            sum_exp += logit;
        }

        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        for (const auto& [move, exp_logit] : move_logits) {
            policy_priors[move] = exp_logit / sum_exp;
        }

        return {std::move(policy_priors), value, nn_output.value_probs};
    }

    /**
     * Compute the percentile bin from a distQ distribution.
     * Returns the smallest bin index where the CDF first reaches >= percentile.
     */
    [[nodiscard]] static int compute_percentile_bin(const std::array<float, VALUE_NUM_BINS>& dist, float percentile) {
        float cumsum = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            cumsum += dist[i];
            if (cumsum >= percentile) {
                return i;
            }
        }
        return VALUE_NUM_BINS - 1;
    }

    /**
     * Recursively search from a node with budget N.
     *
     * @param alpha Lower bound bin index (16th percentile of root distQ, from this node's perspective).
     * @param beta  Upper bound bin index (84th percentile of root distQ, from this node's perspective).
     *              When recursing to children, these are negated (B -> VALUE_NUM_BINS-1-B) and swapped.
     */
    void recursive_search(FractionalNode* node, chess::Board& scratch_board, float N, int alpha, int beta) {
        // Terminal nodes: Q is already set
        if (node->is_terminal) {
            return;
        }

        // Early return: if we've already searched this node with a higher budget,
        // its Q is already at least as refined — skip re-searching.
        if (N <= node->max_N) {
            return;
        }
        node->max_N = N;

        // Depth reduction: more certain nodes (low variance) get reduced effective N,
        // more uncertain nodes (high variance) get amplified effective N.
        // Compute alt_variance: partial variance from alpha..80, ignoring the
        // "irrelevant" tail below alpha. Scaled by 2 to compensate for partial sum.
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
            // Compute limit: how many children cover policy threshold
            int limit = node->get_limit(config_.policy_coverage_threshold,
                                        config_.single_node_coverage_threshold);

            // Base case: effective_N < limit, don't expand further
            // Q is already set from prior evaluation
            if (effective_N < static_cast<float>(limit)) {
                return;
            }
        }

        // Expansion case: expand children with P >= 1/effective_N
        float expansion_threshold = effective_N > 0.0f ? 1.0f / effective_N : 1.0f;
        expand_children(node, scratch_board, expansion_threshold);

        // If no children were expanded, nothing to do
        if (node->children.empty()) {
            return;
        }

        // Scale N by total policy weight of expanded children
        float total_child_weight = 0.0f;
        for (const auto& [move, child] : node->children) {
            total_child_weight += child.P;
        }
        N *= total_child_weight;

        // Compute budget allocations via binary search
        // TODO: Could be replaced later with a cached value
        auto allocations = compute_allocations(node, N);

        // Negate + swap alpha/beta for children (perspective flip)
        int child_alpha = (VALUE_NUM_BINS - 1) - beta;
        int child_beta  = (VALUE_NUM_BINS - 1) - alpha;

        // Recurse into children
        for (auto& [move, child] : node->children) {
            auto it = allocations.find(move);
            if (it != allocations.end() && it->second > 0.0f) {
                float N_i = it->second;
                scratch_board.makeMove<true>(move);
                recursive_search(&child, scratch_board, N_i, child_alpha, child_beta);
                scratch_board.unmakeMove(move);
            }
        }

        // Recompute allocations
        auto second_allocations = compute_allocations(node, N);

        // Update Q and distQ as weighted average of children's values
        // Note: negate child.Q because it's from opponent's perspective
        // Note: flip child.distQ bins (reverse) because it's from opponent's perspective
        float weighted_sum = 0.0f;
        float total_weight = 0.0f;
        std::array<float, VALUE_NUM_BINS> weighted_distQ{};
        weighted_distQ.fill(0.0f);

        for (auto& [move, child] : node->children) {
            auto it = second_allocations.find(move);
            if (it != second_allocations.end() && it->second > 0.0f) {
                float N_i = it->second;
                weighted_sum += (-child.Q) * N_i;
                // Accumulate flipped distQ: child's bin j maps to parent's bin (NUM_BINS-1-j)
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

    /**
     * Expand children with prior probability >= threshold.
     * Children in the first 'limit' positions (by descending prior) are always
     * expanded regardless of threshold.
     */
    void expand_children(FractionalNode* node, chess::Board& scratch_board, float threshold) {
        // Sort policy priors by descending prior
        std::vector<std::pair<chess::Move, float>> sorted_priors;
        sorted_priors.reserve(node->policy_priors.size());
        for (const auto& [move, prior] : node->policy_priors) {
            sorted_priors.emplace_back(move, prior);
        }
        std::sort(sorted_priors.begin(), sorted_priors.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // Get limit (number of children covering policy coverage threshold)
        int limit = node->get_limit(config_.policy_coverage_threshold,
                                    config_.single_node_coverage_threshold);

        for (size_t i = 0; i < sorted_priors.size(); ++i) {
            const auto& [move, prior] = sorted_priors[i];

            // Skip if already expanded
            if (node->children.count(move) > 0) {
                continue;
            }

            // Skip if below threshold AND not in first 'limit' nodes
            bool in_limit = static_cast<int>(i) < limit;
            if (prior < threshold && !in_limit) {
                continue;
            }

            // Create child node
            FractionalNode child(prior);

            // Check for terminal state
            scratch_board.makeMove<true>(move);

            auto [reason, game_result] = scratch_board.isGameOver();
            if (game_result != chess::GameResult::NONE) {
                child.is_terminal = true;
                child.distQ.fill(0.0f);
                if (game_result == chess::GameResult::LOSE) {
                    // Side to move is checkmated (they lost)
                    child.Q = -1.0f;
                    child.distQ[0] = 1.0f;  // All mass at bin 0 (loss)
                } else {
                    // Draw
                    child.Q = 0.0f;
                    child.distQ[VALUE_NUM_BINS / 2] = 1.0f;  // All mass at center (draw)
                }
            } else {
                // Evaluate child position
                evaluate_node(&child, scratch_board);
            }

            scratch_board.unmakeMove(move);
            node->children[move] = std::move(child);
        }
    }

    /**
     * Compute budget allocations for children via binary search.
     *
     * Finds K such that for all children i:
     *   N_i = c_puct * P_i * sqrt(N) / (K - Q_i)
     * and sum(N_i) = N.
     */
    std::unordered_map<chess::Move, float, MoveHash>
    compute_allocations(FractionalNode* node, float N) {
        std::unordered_map<chess::Move, float, MoveHash> allocations;

        if (node->children.empty()) {
            return allocations;
        }

        float c_puct = config_.c_puct;
        float sqrt_N = std::sqrt(N);

        // Collect children info
        std::vector<std::pair<chess::Move, FractionalNode*>> children_vec;
        children_vec.reserve(node->children.size());
        for (auto& [move, child] : node->children) {
            children_vec.emplace_back(move, &child);
        }

        // Lambda to compute allocation for a child given K
        auto compute_allocation = [c_puct, sqrt_N](FractionalNode* child, float K) -> float {
            // N_i = c_puct * P_i * sqrt(N) / (K - Q_i)
            // Q_i is from parent's perspective, so we use -child.Q
            float denominator = K - (-child->Q);
            if (denominator <= 0.0f) {
                return std::numeric_limits<float>::infinity();
            }
            return c_puct * child->P * sqrt_N / denominator;
        };

        // Lambda to compute sum of allocations for a given K
        auto sum_allocations = [&children_vec, &compute_allocation](float K) -> float {
            float total = 0.0f;
            for (const auto& [move, child] : children_vec) {
                float alloc = compute_allocation(child, K);
                if (std::isinf(alloc)) {
                    return std::numeric_limits<float>::infinity();
                }
                total += alloc;
            }
            return total;
        };

        // Find K bounds
        // K must be > max(-child.Q) = max(Q from parent's perspective)
        float max_q = -std::numeric_limits<float>::infinity();
        for (const auto& [move, child] : children_vec) {
            max_q = std::max(max_q, -child->Q);
        }
        float K_low = max_q + 1e-9f;
        float K_high = K_low + 10.0f;

        // Expand K_high until sum < N
        for (int i = 0; i < 100; ++i) {
            float s = sum_allocations(K_high);
            if (s <= N) {
                break;
            }
            K_high *= 2.0f;
        }

        // Binary search for K
        for (int i = 0; i < 64; ++i) {
            float K_mid = (K_low + K_high) / 2.0f;
            float s = sum_allocations(K_mid);
            if (s > N) {
                K_low = K_mid;
            } else {
                K_high = K_mid;
            }
        }

        float K = (K_low + K_high) / 2.0f;

        // Compute final allocations
        for (const auto& [move, child] : children_vec) {
            allocations[move] = compute_allocation(child, K);
        }

        return allocations;
    }

    std::shared_ptr<TrtEvaluator> evaluator_;
    FractionalMCTSConfig config_;
    chess::Board board_;
    std::unique_ptr<FractionalNode> root_;
    std::atomic<bool> stop_flag_;
    int total_gpu_evals_;
    std::ostream* stats_out_;  // If non-null, JSON stats lines are written here during search
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_HPP
