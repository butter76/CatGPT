/**
 * MCTS Configuration.
 *
 * Configuration parameters for Monte Carlo Tree Search.
 */

#ifndef CATGPT_ENGINE_MCTS_CONFIG_HPP
#define CATGPT_ENGINE_MCTS_CONFIG_HPP

#include <cstdint>

namespace catgpt {

/**
 * Configuration for MCTS search.
 */
struct MCTSConfig {
    /**
     * Exploration constant for PUCT formula.
     * Higher values encourage more exploration of less-visited moves.
     * Typical values: 1.0-2.5. Leela Chess Zero uses ~1.75.
     */
    float c_puct = 1.75f;

    /**
     * Minimum total GPU evaluations before stopping search.
     * Search continues until this many neural network evaluations are done.
     */
    int min_total_evals = 800;

    /**
     * First Play Urgency reduction (Leela-style). For unvisited nodes,
     * the Q value is computed as:
     *     parent.Q - fpu_reduction * sqrt(visited_policy)
     * where visited_policy is the fraction of policy mass of visited children.
     * Default 0.330 matches Leela Chess Zero.
     */
    float fpu_reduction = 0.330f;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_CONFIG_HPP
