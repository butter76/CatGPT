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
#include <memory>
#include <unordered_map>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../mcts/node.hpp"  // For MoveHash

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

    explicit FractionalNode(float prior) : P(prior) {}

    /**
     * Compute how many children cover the given fraction of policy mass.
     *
     * @param coverage_threshold Fraction of policy mass to cover (e.g., 0.80).
     * @return Number of children needed to cover that fraction.
     */
    [[nodiscard]] int get_limit(float coverage_threshold = 0.80f) const {
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
        for (size_t i = 0; i < priors.size(); ++i) {
            cumsum += priors[i];
            if (cumsum >= coverage_threshold) {
                return static_cast<int>(i + 1);
            }
        }

        return static_cast<int>(priors.size());
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
     * Get principal variation (best path by Q from this node).
     */
    [[nodiscard]] std::vector<chess::Move> get_pv(int max_depth = 10) {
        std::vector<chess::Move> pv;
        FractionalNode* node = this;

        for (int i = 0; i < max_depth; ++i) {
            auto best = node->best_child_by_q();
            if (!best.has_value()) {
                break;
            }
            auto [move, child] = best.value();
            pv.push_back(move);
            node = child;
        }

        return pv;
    }

    // Policy priors for THIS node's children (from evaluating this position)
    // Maps legal moves to their prior probabilities (sums to 1.0)
    std::unordered_map<chess::Move, float, MoveHash> policy_priors;

    // Prior probability of this move (from parent's policy output)
    float P = 0.0f;

    // Q value: initially from NN evaluation, updated after recursion
    // Range: [-1, 1] where -1=loss, 0=draw, 1=win (from this node's perspective)
    float Q = 0.0f;

    // Terminal state info
    bool is_terminal = false;

    // Children indexed by move (only expanded children are stored)
    std::unordered_map<chess::Move, FractionalNode, MoveHash> children;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_NODE_HPP
