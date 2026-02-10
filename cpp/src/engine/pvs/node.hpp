/**
 * PVS Node data structure.
 *
 * Each node represents a position in the PVS search tree.
 * Every node stores its neural network evaluation (value + policy)
 * directly as member fields — evaluated once at creation time.
 * Terminal nodes store a fixed value instead.
 */

#ifndef CATGPT_ENGINE_PVS_NODE_HPP
#define CATGPT_ENGINE_PVS_NODE_HPP

#include <array>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../mcts/node.hpp"  // For MoveHash
#include "../trt_evaluator.hpp"

namespace catgpt {

/**
 * A node in the PVS search tree.
 *
 * Every non-terminal node has its NN evaluation populated at creation time.
 * Fields:
 *   - value: Evaluation in [-1, 1] from side-to-move's perspective
 *   - value_probs: HL-Gauss value distribution (81 bins)
 *   - policy: Softmaxed policy priors for each legal move
 *   - is_terminal / terminal_value: For checkmate/stalemate positions
 *   - children: Expanded child nodes (sorted by decreasing policy)
 */
class PVSNode {
public:
    PVSNode() = default;

    // NN evaluation (populated at creation time for non-terminal nodes)
    float value = 0.0f;                                // Eval in [-1, 1] from side-to-move (-1=loss, 0=draw, 1=win)
    std::array<float, VALUE_NUM_BINS> value_probs{};   // Value distribution (81 bins)
    std::vector<std::pair<chess::Move, float>> policy;  // Legal moves with softmaxed priors (sorted by decreasing P)

    // Terminal state info
    bool is_terminal = false;
    float terminal_value = 0.0f;  // From side-to-move: -1=loss, 0=draw, 1=win

    // Children (populated during search, sorted by decreasing policy)
    std::vector<std::pair<chess::Move, PVSNode>> children;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_PVS_NODE_HPP
