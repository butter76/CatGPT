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
     * Number of MCTS simulations to run per move.
     * More simulations = stronger play but slower.
     */
    int num_simulations = 800;

    /**
     * First Play Urgency - the Q value assigned to unvisited nodes.
     * -1.0 (default) means unvisited nodes are treated as losses,
     * encouraging exploration of all moves before deep exploitation.
     */
    float fpu_value = -1.0f;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_CONFIG_HPP
