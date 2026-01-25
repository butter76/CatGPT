/**
 * Fractional MCTS Configuration.
 *
 * Configuration parameters for Fractional MCTS with iterative deepening.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_CONFIG_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_CONFIG_HPP

namespace catgpt {

/**
 * Configuration for Fractional MCTS search.
 *
 * This variant uses fractional visit counts and iterative deepening rather than
 * traditional simulations. The search allocates budget N across children using
 * the PUCT formula solved for equal "urgency" K.
 */
struct FractionalMCTSConfig {
    /**
     * Exploration constant for PUCT formula.
     * Higher values encourage more exploration of less-visited moves.
     * Typical values: 1.0-2.5.
     */
    float c_puct = 1.75f;

    /**
     * Fraction of policy mass that determines the "limit" for expansion.
     * If N < limit (number of children covering this fraction), we don't expand.
     */
    float policy_coverage_threshold = 0.80f;

    /**
     * Minimum total GPU evaluations before stopping search.
     * Search continues until an iteration completes with total >= this.
     */
    int min_total_evals = 400;

    /**
     * Starting budget N for iterative deepening.
     */
    float initial_budget = 1.0f;

    /**
     * Factor to multiply N by each iteration.
     */
    float budget_multiplier = 1.2f;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_CONFIG_HPP
