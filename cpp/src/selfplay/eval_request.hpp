/**
 * Evaluation Request — the libfork-coroutine-to-GPU bridge.
 *
 * An EvalRequest holds the input tokens for a position and an output slot
 * for the neural network result.  When a search coroutine needs a GPU
 * evaluation it creates an EvalAwaitable, which:
 *   1. Packs the tokens into an EvalRequest
 *   2. Submits the request to the BatchEvaluator queue
 *   3. Suspends the coroutine, handing libfork's submit_handle to the
 *      evaluator so the GPU thread can later wake it via
 *      pool.schedule(handle).
 *   4. On resumption (libfork worker resumes via lf::resume(handle))
 *      returns the result.
 *
 * Lifetime: EvalAwaitable is a temporary materialized in the coroutine
 * frame by `co_await`, so it (and the embedded EvalRequest) is alive for
 * the entire suspension. The GPU thread reads `tokens` and writes `result`
 * while the coroutine is suspended; the coroutine reads `result` on resume.
 */

#ifndef CATGPT_SELFPLAY_EVAL_REQUEST_HPP
#define CATGPT_SELFPLAY_EVAL_REQUEST_HPP

#include <array>
#include <cmath>
#include <cstdint>
#include <libfork/core.hpp>

#include "../engine/nn_constants.hpp"
#include "../engine/policy.hpp"

namespace catgpt {

// Forward declaration — defined in batch_evaluator.hpp
class BatchEvaluator;

/** Number of WDL classes: [Win, Draw, Loss] from current side to move. */
inline constexpr std::size_t WDL_NUM_CLASSES = 3;

/**
 * Q in [-1, 1] from raw WDL logits. Stable softmax in fp32; equivalent to
 * 2 * (P(W) + 0.5*P(D)) - 1 = (P(W) - P(L)) on the simplex.
 */
inline float wdl_logits_to_q(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    const float m = std::max({logits[0], logits[1], logits[2]});
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float inv_z = 1.0f / (ew + ed + el);
    return (ew - el) * inv_z;
}

/**
 * WDL-derived scalar in [0, 1]: P(W) + 0.5*P(D).
 */
inline float wdl_logits_to_value(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    return 0.5f * (wdl_logits_to_q(logits) + 1.0f);
}

/**
 * Raw neural-network output for a single position.
 * This is what the GPU thread writes after batched inference.
 *
 * The model exports:
 *   wdl_logits   — raw WDL logits [W, D, L] (softmax + Q on host)
 *   value_probs  — BestQ HL-Gauss distribution (81 bins)
 *   legal_policy — Optimistic policy logits gathered at the caller-supplied
 *                  legal-move indices, padded to MAX_LEGAL_MOVES. Only the
 *                  first `num_legal` (== caller's Movelist.size()) entries
 *                  are meaningful; the rest are dont-care padding from the
 *                  GPU gather and must NOT be read.
 */
struct RawNNOutput {
    std::array<float, WDL_NUM_CLASSES> wdl_logits;        // raw WDL logits [W, D, L]
    std::array<float, VALUE_NUM_BINS> value_probs;        // BestQ distribution (81 bins)
    std::array<float, MAX_LEGAL_MOVES> legal_policy;      // gathered policy logits
};

/**
 * A single evaluation request.
 *
 * Allocated inside the coroutine frame (via EvalAwaitable).
 * The GPU thread reads `tokens` + `legal_indices` and writes `result`, then
 * resumes the suspended task via `pool->schedule(continuation)`.
 */
struct EvalRequest {
    // --- Inputs (written by coroutine before suspend) ---

    // Position tokens for the transformer input.
    std::array<std::int32_t, 64> tokens;

    // Flat policy indices (into the model's (64, 73)=4672 optimistic-policy
    // tensor) of every legal move at this position, padded to MAX_LEGAL_MOVES
    // with 0 (any in-range index works; the C++ softmax only reads the first
    // `num_legal` entries, set by the caller's Movelist.size()).
    std::array<std::int32_t, MAX_LEGAL_MOVES> legal_indices;

    // --- Output (written by GPU thread before resuming coroutine) ---
    RawNNOutput result;

    // --- Libfork submit handle (set by await_suspend, consumed by
    //     BatchEvaluator::process_batch via pool->schedule). ---
    lf::submit_handle continuation = nullptr;
};

/**
 * Awaitable that submits an eval request to the batch evaluator and
 * suspends the calling coroutine until the GPU result is ready.
 *
 * Conforms to lf::context_switcher (await_suspend takes lf::submit_handle,
 * not std::coroutine_handle<>) so it can be co_awaited inside any
 * lf::task.
 *
 * Usage inside a libfork task:
 *   RawNNOutput output = co_await EvalAwaitable(evaluator, tokens, legal_indices);
 *
 * `legal_indices` carries pre-computed flat policy indices for every legal
 * move, padded to MAX_LEGAL_MOVES with 0. The first `num_legal` entries must
 * be valid encodings (`policy_flat_index(encode_move_to_policy_index(...))`);
 * the rest are dont-care so long as they're in [0, POLICY_SIZE).
 */
class EvalAwaitable {
   public:
    EvalAwaitable(BatchEvaluator& evaluator,
                  const std::array<std::uint8_t, 64>& tokens,
                  const std::array<std::int32_t, MAX_LEGAL_MOVES>& legal_indices)
        : evaluator_(evaluator) {
        // Convert uint8 tokens to int32 (TRT input format)
        for (int i = 0; i < 64; ++i) {
            request_.tokens[i] = static_cast<std::int32_t>(tokens[i]);
        }
        request_.legal_indices = legal_indices;
    }

    bool await_ready() const noexcept { return false; }

    // Defined after BatchEvaluator is complete (bottom of batch_evaluator.hpp).
    void await_suspend(lf::submit_handle h) noexcept;

    RawNNOutput await_resume() noexcept { return request_.result; }

   private:
    BatchEvaluator& evaluator_;
    EvalRequest request_;
};

static_assert(lf::context_switcher<EvalAwaitable>,
              "EvalAwaitable must satisfy lf::context_switcher so it can "
              "be co_awaited inside an lf::task");

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_EVAL_REQUEST_HPP
