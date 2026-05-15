/**
 * Fractional MCTS Node data structure.
 *
 * Each node represents a position reached by playing a move from the parent.
 * Unlike traditional MCTS, this node stores policy priors for its children
 * and has a Q value that gets updated during search.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_NODE_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_NODE_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../move_hash.hpp"
#include "../nn_constants.hpp"

namespace catgpt {

/**
 * A node in the Fractional MCTS search tree.
 *
 * Key differences from traditional MCTS node:
 * - policy_priors: Maps THIS node's legal moves to prior probabilities
 * - P: This move's prior probability (from parent's policy)
 * - Q: Current value estimate, updated during search
 *
 * The invariant is that before recursing into a node, it has been GPU-evaluated
 * (or is terminal), so Q and policy_priors are populated.
 */
class FractionalNode {
public:
    FractionalNode() = default;

    explicit FractionalNode(float prior, float prior_alloc = 0.0f)
        : P(prior), P_alloc(prior_alloc) {}

    /**
     * Compute how many children cover the given fraction of policy mass.
     *
     * @param coverage_threshold Fraction of policy mass to cover (e.g., 0.75).
     * @param single_node_threshold Minimum prior for limit=1 (e.g., 0.90).
     *        If the top child's prior is below this, limit is forced to >= 2.
     * @return Number of children needed to cover that fraction.
     */
    [[nodiscard]] int get_limit(float coverage_threshold, float single_node_threshold) const {
        if (policy_priors.empty()) {
            return 0;
        }

        // Collect and sort priors descending
        std::vector<float> priors;
        priors.reserve(policy_priors.size());
        for (const auto& [move, prior] : policy_priors) {
            priors.push_back(prior);
        }
        std::sort(priors.begin(), priors.end(), std::greater<float>());

        float cumsum = 0.0f;
        int limit = static_cast<int>(priors.size());
        for (size_t i = 0; i < priors.size(); ++i) {
            cumsum += priors[i];
            if (cumsum >= coverage_threshold) {
                limit = static_cast<int>(i + 1);
                break;
            }
        }

        // Special case: limit=1 requires the top prior to exceed single_node_threshold
        if (limit == 1 && priors[0] < single_node_threshold) {
            return 2;
        }

        return limit;
    }

    /**
     * Return the child with the highest Q value (from parent's perspective).
     * Note: We negate child's Q since it's from opponent's perspective.
     */
    [[nodiscard]] std::optional<std::pair<chess::Move, FractionalNode*>> best_child_by_q() {
        if (children.empty()) {
            return std::nullopt;
        }

        chess::Move best_move = chess::Move::NO_MOVE;
        FractionalNode* best_child = nullptr;
        float best_q = -std::numeric_limits<float>::infinity();

        for (auto& [move, child] : children) {
            float q_from_parent = -child.Q;  // Negate for parent's perspective
            if (q_from_parent > best_q) {
                best_q = q_from_parent;
                best_move = move;
                best_child = &child;
            }
        }

        return std::make_pair(best_move, best_child);
    }

    /**
     * Compute the variance of the value distribution in [-1, 1] scale.
     *
     * The 81 bins from the HL-Gauss head are over [0, 1]. We map them to
     * [-1, 1] by treating bin i's center as 2*(i+0.5)/81 - 1, then compute
     * the variance of the distribution under value_probs.
     */
    void compute_variance() {
        // Bin centers in [-1, 1]
        constexpr float bin_width = 2.0f / VALUE_NUM_BINS;

        // Mean: μ = Σ p_i * c_i
        float mean = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
            mean += value_probs[i] * center;
        }

        // Variance: σ² = Σ p_i * (c_i - μ)²
        float var = 0.0f;
        for (int i = 0; i < VALUE_NUM_BINS; ++i) {
            float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
            float diff = center - mean;
            var += value_probs[i] * diff * diff;
        }

        variance = var;
    }

    /**
     * Get principal variation using allocation-based move selection.
     *
     * At each depth, computes PUCT allocations (the same mechanism used for
     * actual move selection) and follows the highest-allocation child.
     *
     * @param compute_allocs_fn  A callable(FractionalNode*, float) -> allocations map.
     *                           Typically the search class's compute_allocations method.
     * @param max_depth          Maximum PV length.
     */
    template <typename AllocFn>
    [[nodiscard]] std::vector<chess::Move> get_pv(AllocFn&& compute_allocs_fn, int max_depth = 100) {
        std::vector<chess::Move> pv;
        FractionalNode* node = this;

        for (int i = 0; i < max_depth; ++i) {
            if (node->children.empty() || node->max_N <= 0.0f) break;

            // Replicate the budget scaling from recursive_search:
            // N_adjusted = max_N * sum(child.P for expanded children)
            float total_child_weight = 0.0f;
            for (const auto& [move, child] : node->children) {
                total_child_weight += child.P;
            }
            float N_adjusted = node->max_N * total_child_weight;
            if (N_adjusted <= 0.0f) break;

            auto allocs = compute_allocs_fn(node, N_adjusted);

            chess::Move best = chess::Move::NO_MOVE;
            float best_alloc = -1.0f;
            for (const auto& [move, alloc] : allocs) {
                if (alloc > best_alloc) {
                    best_alloc = alloc;
                    best = move;
                }
            }
            if (best == chess::Move::NO_MOVE) break;

            pv.push_back(best);
            node = &node->children.at(best);
        }

        return pv;
    }

    // Policy priors for THIS node's children (from evaluating this position)
    // Maps legal moves to their prior probabilities (sums to 1.0)
    std::unordered_map<chess::Move, float, MoveHash> policy_priors;

    // Warm policy priors (higher temperature) used only for PUCT allocation.
    // Empty when not used (baseline search ignores this).
    std::unordered_map<chess::Move, float, MoveHash> policy_priors_alloc;

    // Prior probability of this move (from parent's policy output)
    float P = 0.0f;

    // Warm prior for PUCT allocation (from parent's policy_priors_alloc).
    // Falls back to P when not explicitly set.
    float P_alloc = 0.0f;

    // Q value: initially from NN evaluation, updated after recursion
    // Range: [-1, 1] where -1=loss, 0=draw, 1=win (from this node's perspective)
    float Q = 0.0f;

    // Value distribution from HL-Gauss head (81 bins over [0, 1])
    std::array<float, VALUE_NUM_BINS> value_probs{};

    // Distributional Q: starts as value_probs from NN eval, then updated
    // during search as the weighted average of children's flipped distQ.
    // Bins are over [0, 1] from this node's perspective.
    std::array<float, VALUE_NUM_BINS> distQ{};

    // Variance of the value distribution in [-1, 1] scale (computed from value_probs)
    float variance = 0.0f;

    // Highest budget N this node was searched with (for early-return optimization)
    float max_N = -1.0f;

    // Terminal state info
    bool is_terminal = false;

    // Children indexed by move (only expanded children are stored)
    std::unordered_map<chess::Move, FractionalNode, MoveHash> children;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_NODE_HPP
