/**
 * PVS Configuration.
 *
 * Configuration parameters for Principal Variation Search.
 */

#ifndef CATGPT_ENGINE_PVS_CONFIG_HPP
#define CATGPT_ENGINE_PVS_CONFIG_HPP

#include <cstdint>

namespace catgpt {

/**
 * Configuration for PVS search.
 */
struct PVSConfig {
    /**
     * Maximum number of GPU evaluations allowed per search.
     * The search must stop once this budget is exhausted.
     */
    int max_gpu_evals = 400;

    /**
     * Iterative deepening step size.
     * Depth increases by this amount each iteration.
     */
    float depth_step = 0.2f;

    /**
     * Cumulative policy threshold for computing 'limit'.
     * limit = number of children (by decreasing policy) whose cumulative
     * policy mass reaches this fraction.
     */
     float policy_coverage = 0.0f;

     /**
      * Minimum policy weight the top child must have for limit to be 1.
      * If the top child's policy is below this, limit is clamped to >= 2.
      */
     float min_single_policy = 0.75f;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_PVS_CONFIG_HPP
