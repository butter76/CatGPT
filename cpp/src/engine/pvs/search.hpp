/**
 * PVS (Principal Variation Search) Algorithm.
 *
 * Alpha-beta search with neural network evaluation, using the PVS/NegaScout
 * optimization: after searching the first (principal) move with a full window,
 * remaining moves are searched with a null/zero window. If the null-window
 * search fails high, a re-search with the full window is performed.
 *
 * Key design:
 *   - Fractional depth: iterative deepening increases by depth_step (0.2).
 *   - Child depth = parent_depth + ln(child_policy). High-policy moves
 *     lose less depth, so they are searched deeper.
 *   - 'limit' = number of children covering policy_coverage (75%) of policy
 *     mass, clamped so limit != 1 unless top child has >= 90% policy.
 *   - If depth <= ln(limit): leaf node, return NN value (no expansion).
 *   - If depth > ln(limit): force-expand the first 'limit' children, plus
 *     any additional children whose child_depth >= 0.
 *   - Every node is NN-evaluated at creation time (1 GPU eval per node).
 *
 * GPU evaluations are budgeted: search stops once the configured maximum
 * number of NN evaluations is reached.
 */

#ifndef CATGPT_ENGINE_PVS_SEARCH_HPP
#define CATGPT_ENGINE_PVS_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <print>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../search_algo.hpp"
#include "../trt_evaluator.hpp"
#include "config.hpp"
#include "node.hpp"

namespace catgpt {

/**
 * Principal Variation Search with neural network evaluation.
 *
 * Uses alpha-beta pruning with the PVS null-window optimization.
 * Move ordering is guided by the policy network's priors.
 * Leaf nodes are evaluated by the value network.
 */
class PVSSearch : public SearchAlgo {
public:
    /**
     * Construct PVS search with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     * @param config PVS configuration.
     */
    PVSSearch(std::shared_ptr<TrtEvaluator> evaluator, PVSConfig config = {})
        : evaluator_(std::move(evaluator))
        , config_(config)
        , board_(STARTPOS_FEN)
        , stop_flag_(false)
        , gpu_evals_(0)
    {}

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
        root_.reset();
        gpu_evals_ = 0;
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
        root_.reset();
    }

    SearchResult search(const SearchLimits& limits) override {
        stop_flag_.store(false, std::memory_order_relaxed);
        gpu_evals_ = 0;

        auto start_time = std::chrono::steady_clock::now();

        // Generate legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);

        SearchResult result;

        if (moves.empty()) {
            result.best_move = chess::Move::NO_MOVE;
            if (board_.inCheck()) {
                result.score = Score::mate(0);
            } else {
                result.score = Score::cp(0);
            }
            return result;
        }

        if (moves.size() == 1) {
            result.best_move = moves[0];
            result.depth = 1;
            result.nodes = 1;
            return result;
        }

        // Initialize root with NN evaluation
        root_ = std::make_unique<PVSNode>();
        evaluate_node(root_.get(), board_);

        // Iterative deepening with fractional depth steps
        chess::Move best_move = chess::Move::NO_MOVE;
        float best_score = 0.0f;
        float max_depth_reached = 0.0f;

        for (float depth = config_.depth_step; depth <= max_depth_; depth += config_.depth_step) {
            if (stop_flag_.load(std::memory_order_relaxed)) {
                break;
            }

            if (gpu_evals_ >= config_.max_gpu_evals) {
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

            // Check depth limit (compare fractional depth against integer limit)
            if (limits.depth.has_value() && depth > static_cast<float>(limits.depth.value())) {
                break;
            }

            float alpha = -1.0f;
            float beta = 1.0f;

            float score = pvs(board_, root_.get(), depth, alpha, beta);

            // Only update best move if we completed this depth (or have no move yet)
            if (!stop_flag_.load(std::memory_order_relaxed) || best_move == chess::Move::NO_MOVE) {
                if (gpu_evals_ < config_.max_gpu_evals) {
                    best_score = score;
                    max_depth_reached = depth;

                    // PV move is at front of children (swapped there during search)
                    if (!root_->children.empty()) {
                        best_move = root_->children[0].first;
                    }
                }
            }
        }

        result.best_move = best_move;

        // Convert value from [-1, 1] to centipawns using tangent scaling
        int cp = static_cast<int>(90.0f * std::tan(best_score * 1.5637541897f));
        result.score = Score::cp(cp);

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = static_cast<int>(std::ceil(max_depth_reached));
        result.nodes = gpu_evals_;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (gpu_evals_ * 1000) / elapsed.count();
        }

        // Build PV
        result.pv = extract_pv(root_.get());

        return result;
    }

    void stop() override {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

    [[nodiscard]] const PVSNode* root() const {
        return root_.get();
    }

    [[nodiscard]] int gpu_evals_used() const noexcept {
        return gpu_evals_;
    }

private:
    /**
     * Compute 'limit' for a node: the number of children whose cumulative
     * policy mass reaches policy_coverage (75%).
     *
     * Clamped so limit != 1 unless the top child has >= min_single_policy (90%).
     * Precondition: node->policy is sorted by decreasing probability.
     */
    [[nodiscard]] int compute_limit(const PVSNode* node) const {
        if (node->policy.empty()) {
            return 0;
        }

        float cumulative = 0.0f;
        int limit = 0;

        for (const auto& [move, prob] : node->policy) {
            cumulative += prob;
            ++limit;
            if (cumulative >= config_.policy_coverage) {
                break;
            }
        }

        // Prevent limit == 1 unless top child has >= 90% policy
        if (limit == 1 && node->policy[0].second < config_.min_single_policy) {
            limit = std::min(2, static_cast<int>(node->policy.size()));
        }

        return limit;
    }

    /**
     * Principal Variation Search with fractional depth.
     *
     * @param pos Current board position.
     * @param node Current node (already NN-evaluated).
     * @param depth Remaining fractional depth.
     * @param alpha Lower bound [-1, 1].
     * @param beta Upper bound [-1, 1].
     * @return Value from the side-to-move's perspective in [-1, 1].
     */
    float pvs(chess::Board& pos, PVSNode* node, float depth, float alpha, float beta) {
        // Check termination conditions
        if (stop_flag_.load(std::memory_order_relaxed)) {
            return node->value;
        }

        // Terminal nodes always return their value
        if (node->is_terminal) {
            return node->terminal_value;
        }

        // Compute limit and check leaf condition
        int limit = compute_limit(node);

        if (limit == 0) {
            return node->value;
        }

        float ln_limit = std::log(static_cast<float>(limit));

        // Depth reduction from value uncertainty: low variance → large reduction (prune more)
        float depth_reduction = std::log(node->U + 1e-6f);

        // Leaf: depth is not deep enough to justify expansion
        if (depth + depth_reduction <= ln_limit) {
            return node->value;
        }

        // Determine which children to search from the policy vector:
        //   - Force the first 'limit' children (by decreasing policy)
        //   - Also include any child beyond 'limit' whose child_depth >= 0
        int num_policy = static_cast<int>(node->policy.size());
        int search_up_to = std::min(limit, num_policy);

        // Check for bonus children beyond limit with child_depth >= 0
        for (int i = limit; i < num_policy; ++i) {
            float child_depth = depth + std::log(node->policy[i].second);
            if (child_depth + depth_reduction >= 0.0f) {
                ++search_up_to;
            } else {
                break;  // Policy is sorted descending, so remaining will also be < 0
            }
        }

        // PVS search — expand children on demand
        float best_score = -2.0f;  // Worst possible (loss)
        bool first_child = true;
        float best_depth = 0.0f;   // Search depth of the best child
        chess::Move best_move = chess::Move::NO_MOVE;

        for (int i = 0; i < search_up_to; ++i) {
            auto [move, policy_prior] = node->policy[i];
            float child_depth = depth + std::log(policy_prior);

            pos.makeMove<true>(move);

            // Lazily expand this child (creates + NN-evaluates if new)
            // If budget is exhausted and child doesn't exist yet, skip it
            PVSNode* child = find_child(node, move);
            if (!child) {
                child = ensure_child(node, pos, move);
            }

            float score;
            if (first_child) {
                // Search principal variation with full window
                score = -pvs(pos, child, child_depth, -beta, -alpha);
                first_child = false;
                best_depth = child_depth;
            } else {
                while (true) {
                    // Null-window search
                    score = -pvs(pos, child, child_depth, -alpha - 0.0001f, -alpha);

                    // Re-search with full window if null-window failed high
                    if (score > alpha) {
                        if (child_depth >= best_depth) {
                            score = -pvs(pos, child, child_depth, -beta, -alpha);
                            break;
                        }
                        child_depth = std::min(child_depth + 0.2f, best_depth);
                    } else {
                        break;
                    }
                }
            }

            pos.unmakeMove(move);

            if (score > best_score) {
                best_score = score;
                best_move = move;
            }

            alpha = std::max(alpha, score);

            // Beta cutoff
            if (alpha >= beta) {
                break;
            }

            if (stop_flag_.load(std::memory_order_relaxed)) {
                break;
            }
        }

        // Promote search-proven best move in the policy vector:
        // Give it the weight of the previous top-policy move, then normalize.
        if (best_move != chess::Move::NO_MOVE && best_move != node->policy[0].first) {
            float top_weight = node->policy[0].second;
            for (auto& [m, p] : node->policy) {
                if (m == best_move) {
                    p = top_weight + 0.001f;
                    break;
                }
            }

            // Normalize so policy sums to 1
            float sum = 0.0f;
            for (const auto& [m, p] : node->policy) {
                sum += p;
            }
            for (auto& [m, p] : node->policy) {
                p /= sum;
            }

            // Re-sort by decreasing probability
            std::sort(node->policy.begin(), node->policy.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        }

        // Move best child to front of children for PV extraction
        if (best_move != chess::Move::NO_MOVE) {
            for (std::size_t i = 1; i < node->children.size(); ++i) {
                if (node->children[i].first == best_move) {
                    std::swap(node->children[0], node->children[i]);
                    break;
                }
            }
            return best_score;
        } else {
            return node->value;
        }
    }

    /**
     * Find an existing child for the given move.
     * Returns nullptr if the child does not exist yet.
     */
    [[nodiscard]] PVSNode* find_child(PVSNode* node, const chess::Move& move) {
        for (auto& [m, child] : node->children) {
            if (m == move) {
                return &child;
            }
        }
        return nullptr;
    }

    /**
     * Create and evaluate a new child for the given move.
     * The board position must already reflect the move having been made.
     * Caller must ensure the child does not already exist (use find_child first).
     *
     * @param node Parent node.
     * @param pos Board position AFTER the move.
     * @param move The move that leads to this child.
     * @return Pointer to the newly created child node.
     */
    PVSNode* ensure_child(PVSNode* node, const chess::Board& pos, const chess::Move& move) {
        PVSNode child;

        auto [reason, game_result] = pos.isGameOver();
        if (game_result != chess::GameResult::NONE) {
            // Terminal node — no GPU eval needed
            child.is_terminal = true;
            if (game_result == chess::GameResult::LOSE) {
                child.terminal_value = -1.0f;
                child.value = -1.0f;
            } else {
                child.terminal_value = 0.0f;
                child.value = 0.0f;
            }
        } else {
            evaluate_node(&child, pos);
        }

        node->children.emplace_back(move, std::move(child));
        return &node->children.back().second;
    }

    /**
     * Evaluate a node with the neural network.
     * Populates value, value_probs, and policy directly on the node.
     */
    void evaluate_node(PVSNode* node, const chess::Board& pos) {
        ++gpu_evals_;

        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);
        auto nn_output = evaluator_->evaluate(tokens);

        // Store value: convert from NN [0, 1] to [-1, 1]
        node->value = 2.0f * nn_output.value - 1.0f;
        node->value_probs = nn_output.value_probs;

        // Compute U = variance of value_probs in [-1, 1] space
        // Bin i has center (2*i - (N-1)) / N, i.e. -80/81, -78/81, ..., 78/81, 80/81
        {
            constexpr float N = static_cast<float>(VALUE_NUM_BINS);
            float mean = 0.0f;
            float mean_sq = 0.0f;
            for (int i = 0; i < VALUE_NUM_BINS; ++i) {
                float center = (2.0f * static_cast<float>(i) - (N - 1.0f)) / N;
                float p = nn_output.value_probs[i];
                mean += p * center;
                mean_sq += p * center * center;
            }
            node->U = mean_sq - mean * mean;
        }

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

        // Softmax over legal moves
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto& [move, logit] : move_logits) {
            max_logit = std::max(max_logit, logit);
        }

        float sum_exp = 0.0f;
        for (auto& [move, logit] : move_logits) {
            logit = std::exp(logit - max_logit);
            sum_exp += logit;
        }

        // Build policy vector sorted by decreasing probability
        node->policy.clear();
        node->policy.reserve(move_logits.size());
        for (const auto& [move, exp_logit] : move_logits) {
            node->policy.emplace_back(move, exp_logit / sum_exp);
        }

        std::sort(node->policy.begin(), node->policy.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    /**
     * Extract the principal variation from the search tree.
     */
    [[nodiscard]] std::vector<chess::Move> extract_pv(const PVSNode* node, int max_depth = 20) const {
        std::vector<chess::Move> pv;
        const PVSNode* current = node;

        for (int i = 0; i < max_depth && current && !current->children.empty(); ++i) {
            // PV move is always at the front (swapped there during search)
            pv.push_back(current->children[0].first);
            current = &current->children[0].second;
        }

        return pv;
    }

    static constexpr float max_depth_ = 50.0f;

    std::shared_ptr<TrtEvaluator> evaluator_;
    PVSConfig config_;
    chess::Board board_;
    std::unique_ptr<PVSNode> root_;
    std::atomic<bool> stop_flag_;
    int gpu_evals_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_PVS_SEARCH_HPP
