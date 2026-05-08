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
#include <cstdint>

#include <libfork/core.hpp>

#include "../engine/nn_constants.hpp"
#include "../engine/policy.hpp"

namespace catgpt {

// Forward declaration — defined in batch_evaluator.hpp
class BatchEvaluator;

/**
 * Raw neural-network output for a single position.
 * This is what the GPU thread writes after batched inference.
 *
 * The model exports:
 *   value            — WDL-derived Q value P(W)+0.5*P(D) in [0, 1]
 *   value_probs      — BestQ HL-Gauss distribution (81 bins)
 *   wdl              — Win/Draw/Loss probabilities [W, D, L]
 *   policy           — Move distribution logits (4672)
 *   optimistic_policy — Optimistic policy logits (4672), trained with value-surprise weighting
 */
struct RawNNOutput {
    float value;                                    // WDL-derived Q value [0, 1]
    std::array<float, VALUE_NUM_BINS> value_probs;  // BestQ distribution (81 bins)
    std::array<float, 3> wdl;                       // Win/Draw/Loss probabilities [W, D, L]
    std::array<float, POLICY_SIZE> policy;          // Policy logits (4672)
    std::array<float, POLICY_SIZE> optimistic_policy;  // Optimistic policy logits (4672)
    bool has_optimistic_policy = false;             // Whether optimistic policy is available
};

/**
 * A single evaluation request.
 *
 * Allocated inside the coroutine frame (via EvalAwaitable).
 * The GPU thread reads `tokens` and writes `result`, then resumes the
 * suspended task via `pool->schedule(continuation)`.
 */
struct EvalRequest {
    // --- Input (written by coroutine before suspend) ---
    std::array<std::int32_t, 64> tokens;

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
 *   RawNNOutput output = co_await EvalAwaitable(evaluator, tokens);
 */
class EvalAwaitable {
   public:
    EvalAwaitable(BatchEvaluator& evaluator,
                  const std::array<std::uint8_t, 64>& tokens)
        : evaluator_(evaluator) {
        // Convert uint8 tokens to int32 (TRT input format)
        for (int i = 0; i < 64; ++i) {
            request_.tokens[i] = static_cast<std::int32_t>(tokens[i]);
        }
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
