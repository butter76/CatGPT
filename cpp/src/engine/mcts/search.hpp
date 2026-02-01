/**
 * MCTS Search Algorithm with Soft-Minimax Q Calculation.
 *
 * Monte Carlo Tree Search using PUCT selection (AlphaZero/Leela Chess Zero style).
 *
 * The search proceeds in five phases:
 *   1. SELECT: Traverse tree using PUCT until reaching a leaf
 *   2. EXPAND: Create children for the leaf with priors from policy network
 *   3. EVALUATE: Get value estimate (origQ) from value network
 *   4. BACKPROPAGATE: Update N (visit count) along path from leaf to root
 *   5. CALC Q: Recursively recompute Q values using soft-minimax
 *
 * Soft-minimax Q calculation:
 *   - Sort children by descending N (most visited first)
 *   - Track running max of -child.Q as we iterate
 *   - Q(node) = (origQ * 1 + sum(max_neg_child_q * child.N)) / N
 *
 * This propagates good scores from well-explored children to less-explored
 * siblings, creating a blend between standard MCTS averaging and minimax.
 *
 * After search, the move with highest visit count is selected.
 */

#ifndef CATGPT_ENGINE_MCTS_SEARCH_HPP
#define CATGPT_ENGINE_MCTS_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
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
 * MCTS search algorithm using PUCT selection.
 *
 * PUCT formula:
 *   U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
 *
 * Where:
 *   - Q(s,a): Mean value of taking action a from state s
 *   - P(s,a): Prior probability from policy network
 *   - N(s,a): Visit count for this action
 *   - N_parent: Total visits to parent node, not including itself
 *   - c_puct: Exploration constant
 */
class MCTSSearch : public SearchAlgo {
public:
    /**
     * Construct MCTS search with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     * @param config MCTS configuration.
     */
    MCTSSearch(std::shared_ptr<TrtEvaluator> evaluator, MCTSConfig config = {})
        : evaluator_(std::move(evaluator))
        , config_(config)
        , board_(STARTPOS_FEN)
        , stop_flag_(false)
        , total_gpu_evals_(0)
        , debug_(false)
    {}

    /**
     * Enable or disable debug output.
     */
    void set_debug(bool enabled) { debug_ = enabled; }

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
        root_.reset();
        total_gpu_evals_ = 0;
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
        root_.reset();  // Invalidate tree after move
    }

    SearchResult search(const SearchLimits& limits) override {
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

        // Initialize root node
        root_ = std::make_unique<MCTSNode>();
        total_gpu_evals_ = 0;

        // Determine target number of GPU evaluations
        int target_evals = config_.min_total_evals;
        if (limits.nodes.has_value()) {
            target_evals = std::min(target_evals, static_cast<int>(limits.nodes.value()));
        }

        // Run simulations until we reach the minimum GPU evaluations
        std::int64_t total_nodes = 0;
        while (total_gpu_evals_ < target_evals && total_nodes < config_.max_simulations) {
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

            run_simulation();
            ++total_nodes;
        }

        // Select move with highest visit count
        auto best = root_->best_child_by_visits();
        if (best.has_value()) {
            result.best_move = best->first;

            // Q from root's perspective (negate child's Q since it's opponent's view)
            float q = -best->second->Q();
            // Convert Q from [-1, 1] to centipawns using tangent scaling
            int cp = static_cast<int>(90.0f * std::tan(q * 1.5637541897f));
            result.score = Score::cp(cp);
        } else {
            // Fallback (shouldn't happen)
            result.best_move = moves[0];
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = 1;  // MCTS doesn't have traditional depth
        result.nodes = total_nodes;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (total_nodes * 1000) / elapsed.count();
        }

        // Build PV from most-visited path
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
    [[nodiscard]] const MCTSNode* root() const {
        return root_.get();
    }

private:
    /**
     * Run a single MCTS simulation.
     */
    void run_simulation() {
        MCTSNode* node = root_.get();
        std::vector<MCTSNode*> path = {node};
        chess::Board scratch_board = board_;
        bool at_least = true;  // Root's children selection is "at least"

        // SELECT: traverse tree using upside-based selection until we reach a leaf
        while (node->is_expanded() && !node->is_terminal) {
            auto [move, child] = select_child(node, at_least);
            scratch_board.makeMove<true>(move);
            node = child;
            path.push_back(node);
            at_least = !at_least;  // Flip perspective each level
        }

        // EXPAND & EVALUATE
        if (!node->is_terminal) {
            // Expand the leaf (this sets node->origQ)
            expand_and_evaluate(node, scratch_board);
        }
        // Terminal nodes already have origQ set when created

        // BACKPROPAGATE: update visit counts
        backpropagate(path);

        // CALC Q: recursively recompute Q values for entire tree
        calcQ(root_.get());

        // CALC UPSIDE: probability of being at least as good as current estimate
        float upside_threshold = std::min(root_->cached_Q + 2.0f / 81.0f, 0.999f);
        calcUpside(root_.get(), upside_threshold);

        // Debug: show all visited children at root
        if (debug_) {
            std::print("  POST-CALC (root.N={}, root.Q={:.4f}, upside_threshold={:.4f}):\n",
                       root_->N, root_->cached_Q, upside_threshold);
            std::print("    {:6} {:>5} {:>8} {:>8} {:>8} {:>8}\n",
                       "move", "N", "P", "Q", "-Q", "upside");
            for (const auto& [move, child] : root_->children) {
                if (child.N > 0) {
                    std::print("    {:6} {:5} {:8.4f} {:8.4f} {:8.4f} {:8.4f}\n",
                               chess::uci::moveToUci(move), child.N, child.P,
                               child.cached_Q, -child.cached_Q, child.cached_upside);
                }
            }
            std::print("\n");
        }
    }

    /**
     * Select child with highest exploration score based on upside.
     *
     * Children are sorted by (N desc, P desc). For each child:
     * - Expanded (N > 0): U = remainingUpside * upside_factor * n_parent / (N + 1)
     * - Unexpanded (N == 0): U = remainingUpside * P / (N + 1)
     *
     * For "at least" nodes: upside_factor = cached_upside
     * For "at most" nodes: upside_factor = 1 - cached_upside
     *
     * remainingUpside decays as we iterate through expanded children,
     * giving more exploration budget to higher-visited children.
     *
     * @param node The parent node to select from
     * @param at_least True if selecting from root's perspective, false for opponent
     */
    std::pair<chess::Move, MCTSNode*> select_child(MCTSNode* node, bool at_least) {
        // n_parent = N - 1 (current visit in progress)
        int n_parent = node->N > 1 ? node->N - 1 : 0;

        // Create indices sorted by (N desc, P desc)
        std::vector<std::size_t> indices(node->children.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
            const auto& ca = node->children[a].second;
            const auto& cb = node->children[b].second;
            if (ca.N != cb.N) return ca.N > cb.N;  // Decreasing N
            return ca.P > cb.P;  // Decreasing P (tie-breaker)
        });

        float best_score = -std::numeric_limits<float>::infinity();
        chess::Move best_move = chess::Move::NO_MOVE;
        MCTSNode* best_child = nullptr;

        float remainingUpside = 1.0f;

        if (debug_) {
            std::print("  SELECT (at_least={}, n_parent={}):\n", at_least, n_parent);
        }

        for (std::size_t idx : indices) {
            auto& [move, child] = node->children[idx];

            float U;
            if (child.N > 0) {
                // Expanded node: use upside
                float upside_factor = at_least ? child.cached_upside : (1.0f - child.cached_upside);
                U = remainingUpside * upside_factor / (child.N + 1);

                if (debug_) {
                    std::print("    {} N={} upside={:.4f} upside_factor={:.4f} remainingUp={:.4f} => U={:.4f}\n",
                               chess::uci::moveToUci(move), child.N, child.cached_upside,
                               upside_factor, remainingUpside, U);
                }

                // Update remainingUpside for next iteration
                if (at_least) {
                    remainingUpside *= (1.0f - child.cached_upside);
                } else {
                    remainingUpside *= child.cached_upside;
                }
            } else {
                // Unexpanded node: use policy
                U = remainingUpside * child.P;  // N=0, so denominator is 1

                if (debug_) {
                    std::print("    {} N=0 P={:.4f} remainingUp={:.4f} => U={:.4f}\n",
                               chess::uci::moveToUci(move), child.P, remainingUpside, U);
                }
            }

            if (U > best_score) {
                best_score = U;
                best_move = move;
                best_child = &child;
            }
        }

        if (debug_) {
            std::print("    => selected: {} (U={:.4f})\n", chess::uci::moveToUci(best_move), best_score);
        }

        return {best_move, best_child};
    }

    /**
     * Expand a leaf node and return value estimate.
     */
    float expand_and_evaluate(MCTSNode* node, chess::Board& scratch_board) {
        // Get policy and value from neural network
        auto eval = evaluate_position(scratch_board);

        // Create children for all legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, scratch_board);

        node->children.reserve(moves.size());

        for (const auto& move : moves) {
            float prior = 0.0f;
            auto it = eval.policy_priors.find(move);
            if (it != eval.policy_priors.end()) {
                prior = it->second;
            }

            MCTSNode child(prior);

            // Check for terminal states
            scratch_board.makeMove<true>(move);

            if (scratch_board.isGameOver().second != chess::GameResult::NONE) {
                auto [reason, game_result] = scratch_board.isGameOver();

                if (game_result == chess::GameResult::LOSE) {
                    // The side to move at this position is checkmated (they lost)
                    child.is_terminal = true;
                    child.terminal_value = -1.0f;
                    child.origQ = -1.0f;
                } else if (game_result == chess::GameResult::DRAW) {
                    child.is_terminal = true;
                    child.terminal_value = 0.0f;
                    child.origQ = 0.0f;
                }
            }

            scratch_board.unmakeMove(move);

            node->children.emplace_back(move, std::move(child));
        }

        // Sort children by decreasing policy (highest P first)
        std::sort(node->children.begin(), node->children.end(),
                  [](const auto& a, const auto& b) { return a.second.P > b.second.P; });

        // Store original NN evaluation for recursive Q calculation
        node->origQ = eval.value;
        node->value_probs = eval.value_probs;

        return eval.value;
    }

    /**
     * Result of neural network evaluation.
     */
    struct EvalResult {
        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        float value;  // [-1, 1]
        std::array<float, VALUE_NUM_BINS> value_probs;
    };

    /**
     * Evaluate a position with the neural network.
     */
    EvalResult evaluate_position(const chess::Board& pos) {
        ++total_gpu_evals_;

        std::string fen = pos.getFen();

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
     * Update visit counts along the path from leaf to root.
     */
    void backpropagate(std::vector<MCTSNode*>& path) {
        for (MCTSNode* node : path) {
            node->N += 1;
        }
    }

    /**
     * Compute upside probability from value distribution.
     *
     * @param probs Value distribution (81 bins uniformly over [-1, 1])
     * @param threshold The Q value threshold to compare against
     * @param at_least If true, compute P(value >= threshold); if false, P(value <= threshold)
     * @return Probability mass meeting the condition
     */
    static float compute_upside_from_probs(
        const std::array<float, VALUE_NUM_BINS>& probs,
        float threshold,
        bool at_least
    ) {
        float sum = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            // Bucket i center: uniformly spaced in [-1, 1]
            float bucket_center = -1.0f + (2.0f * i + 1.0f) / VALUE_NUM_BINS;
            bool include;
            if (at_least) {
                // Include if bucket_center >= threshold, OR last bucket (captures values up to +1)
                include = (bucket_center >= threshold) || (i == VALUE_NUM_BINS - 1);
            } else {
                // Include if bucket_center <= threshold, OR first bucket (captures values down to -1)
                include = (bucket_center <= threshold) || (i == 0);
            }
            if (include) sum += probs[i];
        }
        return sum;
    }

    /**
     * Recursively compute Q values for the entire tree using soft-minimax.
     *
     * Q(node) = (origQ * 1 + sum(max_neg_child_q * child.N)) / N
     *
     * Where max_neg_child_q is the running maximum of -child.Q as we iterate
     * through children sorted by descending visit count. This propagates good
     * scores from well-explored children to less-explored siblings.
     *
     * This is called after each simulation to update cached_Q values.
     */
    void calcQ(MCTSNode* node) {
        // Terminal nodes: Q is just the terminal value
        if (node->is_terminal) {
            node->cached_Q = node->terminal_value.value();
            return;
        }

        // Unexpanded leaf: Q is just origQ (shouldn't happen after backprop)
        if (!node->is_expanded()) {
            node->cached_Q = node->origQ;
            return;
        }

        // Recursively compute Q for all children first
        for (auto& [move, child] : node->children) {
            calcQ(&child);
        }

        // Create indices sorted by descending N (most visited first)
        std::vector<std::size_t> indices(node->children.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
            return node->children[a].second.N > node->children[b].second.N;
        });

        // Soft-minimax weighted average:
        // Track running max of -child.Q as we go through by descending N
        float sum = node->origQ;  // Weight 1 for own evaluation
        float max_neg_child_q = -std::numeric_limits<float>::infinity();

        for (std::size_t idx : indices) {
            const auto& child = node->children[idx].second;
            float neg_child_q = -child.cached_Q;
            max_neg_child_q = std::max(max_neg_child_q, neg_child_q);
            sum += max_neg_child_q * static_cast<float>(child.N);
        }

        node->cached_Q = sum / static_cast<float>(node->N);
    }

    /**
     * Recursively compute upside values for the entire tree.
     *
     * Upside represents the probability that a node's true value (from root's
     * perspective) is at least as good as the root's current estimate.
     *
     * For leaf nodes: upside = P(value >= threshold) using value_probs distribution
     * For non-leaf nodes: soft-max weighted average similar to calcQ
     *
     * @param node The node to compute upside for
     * @param threshold The Q value threshold (root's cachedQ)
     * @param at_least If true, compute P(value >= threshold); flips on each level
     */
    void calcUpside(MCTSNode* node, float threshold, bool at_least = true) {
        // Terminal nodes: deterministic 0 or 1
        if (node->is_terminal) {
            float val = node->terminal_value.value();
            bool meets_condition = at_least ? (val >= threshold) : (val <= threshold);
            node->cached_upside = meets_condition ? 1.0f : 0.0f;
            return;
        }

        // Unexpanded leaf: use value_probs directly
        if (!node->is_expanded()) {
            node->cached_upside = compute_upside_from_probs(node->value_probs, threshold, at_least);
            return;
        }

        // Recursively compute for children (flip perspective!)
        for (auto& [move, child] : node->children) {
            calcUpside(&child, -threshold, !at_least);
        }

        // Own upside contribution (weight 1)
        float own_upside = compute_upside_from_probs(node->value_probs, threshold, at_least);

        // Sort children by descending N (most visited first) - same as calcQ
        std::vector<std::size_t> indices(node->children.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
            return node->children[a].second.N > node->children[b].second.N;
        });

        // Soft-max weighted average (similar to calcQ)
        float sum = own_upside;
        float max_upside = 0.0f;  // Running max (upside is [0,1] so start at 0)

        for (std::size_t idx : indices) {
            const auto& child = node->children[idx].second;
            max_upside = child.cached_upside;
            sum += max_upside * static_cast<float>(child.N);
        }

        node->cached_upside = sum / static_cast<float>(node->N);
    }

    std::shared_ptr<TrtEvaluator> evaluator_;
    MCTSConfig config_;
    chess::Board board_;
    std::unique_ptr<MCTSNode> root_;
    std::atomic<bool> stop_flag_;
    int total_gpu_evals_;
    bool debug_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_SEARCH_HPP
