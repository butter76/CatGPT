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

// Upper bound on the number of legal moves in any chess position. The
// theoretical max is 218 (constructed positions); typical positions have
// 30-45. The GPU evaluator gathers policy logits at MAX_LEGAL_MOVES
// caller-supplied indices per request (padded with 0), and the C++
// search softmaxes only the leading `num_legal` entries.
constexpr int MAX_LEGAL_MOVES = 218;

}  // namespace catgpt

#endif  // CATGPT_ENGINE_NN_CONSTANTS_HPP
