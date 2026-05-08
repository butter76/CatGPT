/**
 * Lightweight neural-network constants shared across the engine.
 *
 * This header has no CUDA / TensorRT / libfork dependencies, so it is
 * safe to include from lightweight headers (eval_request.hpp, MCTS
 * node/search_stats headers, etc.) that need only the value-distribution
 * bin count without dragging in the full TRT runtime.
 */

#ifndef CATGPT_ENGINE_NN_CONSTANTS_HPP
#define CATGPT_ENGINE_NN_CONSTANTS_HPP

namespace catgpt {

// Number of bins in the HL-Gauss value distribution (BestQ head).
constexpr int VALUE_NUM_BINS = 81;

}  // namespace catgpt

#endif  // CATGPT_ENGINE_NN_CONSTANTS_HPP
