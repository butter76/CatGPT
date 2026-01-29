/**
 * MCTS Node data structure.
 *
 * Each node represents a position reached by playing a move from the parent.
 * The root node has move=NO_MOVE and parent=nullptr.
 */

#ifndef CATGPT_ENGINE_MCTS_NODE_HPP
#define CATGPT_ENGINE_MCTS_NODE_HPP

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"

namespace catgpt {

// Custom hash for chess::Move to use in unordered_map
struct MoveHash {
    std::size_t operator()(const chess::Move& move) const noexcept {
        return std::hash<std::uint16_t>{}(move.move());
    }
};

/**
 * A node in the MCTS search tree.
 *
 * Statistics:
 *   N: Visit count - how many times this node was visited during search.
 *   W: Total value sum - accumulated value from all visits.
 *   P: Prior probability - from policy network, used in PUCT formula.
 *
 * The mean value Q = W/N represents the expected outcome from this position.
 */
class MCTSNode {
public:
    MCTSNode() = default;

    explicit MCTSNode(float prior) : P(prior) {}

    /**
     * Mean value (expected outcome from this position).
     * Returns 0.0 for unvisited nodes.
     */
    [[nodiscard]] float Q() const noexcept {
        return N > 0 ? W / static_cast<float>(N) : 0.0f;
    }

    /**
     * Whether this node has been expanded (children created).
     */
    [[nodiscard]] bool is_expanded() const noexcept {
        return !children.empty();
    }

    /**
     * Return the child with the highest visit count.
     * Returns nullopt if no children.
     */
    [[nodiscard]] std::optional<std::pair<chess::Move, MCTSNode*>> best_child_by_visits() {
        if (children.empty()) {
            return std::nullopt;
        }

        chess::Move best_move = chess::Move::NO_MOVE;
        MCTSNode* best_child = nullptr;
        int best_visits = -1;

        for (auto& [move, child] : children) {
            if (child.N > best_visits) {
                best_visits = child.N;
                best_move = move;
                best_child = &child;
            }
        }

        return std::make_pair(best_move, best_child);
    }

    /**
     * Get normalized visit counts (policy after search).
     */
    [[nodiscard]] std::unordered_map<chess::Move, float, MoveHash> get_visit_distribution() const {
        std::unordered_map<chess::Move, float, MoveHash> dist;

        if (children.empty()) {
            return dist;
        }

        int total_visits = 0;
        for (const auto& [move, child] : children) {
            total_visits += child.N;
        }

        if (total_visits == 0) {
            for (const auto& [move, child] : children) {
                dist[move] = 0.0f;
            }
            return dist;
        }

        for (const auto& [move, child] : children) {
            dist[move] = static_cast<float>(child.N) / static_cast<float>(total_visits);
        }

        return dist;
    }

    /**
     * Get principal variation (most visited path from this node).
     */
    [[nodiscard]] std::vector<chess::Move> get_pv(int max_depth = 10) {
        std::vector<chess::Move> pv;
        MCTSNode* node = this;

        for (int i = 0; i < max_depth; ++i) {
            auto best = node->best_child_by_visits();
            if (!best.has_value()) {
                break;
            }
            auto [move, child] = best.value();
            pv.push_back(move);
            node = child;
        }

        return pv;
    }

    // Statistics
    int N = 0;       // Visit count
    float W = 0.0f;  // Total value sum
    float P = 0.0f;  // Prior probability

    // Terminal state info
    bool is_terminal = false;
    std::optional<float> terminal_value;  // -1=loss, 0=draw, 1=win (from this node's side-to-move)

    // Children ordered by decreasing policy (highest P first)
    std::vector<std::pair<chess::Move, MCTSNode>> children;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_NODE_HPP
