/**
 * Evaluation Request — the coroutine-to-GPU bridge (libcoro legacy).
 *
 * An EvalRequest holds the input tokens for a position and an output slot
 * for the neural network result.  When a search coroutine needs a GPU
 * evaluation it creates an EvalAwaitable, which:
 *   1. Packs the tokens into an EvalRequest
 *   2. Submits the request to the BatchEvaluator queue
 *   3. Suspends the coroutine (returning its handle for later resumption)
 *   4. On resumption (by the GPU thread → thread pool), returns the result
 *
 * Lifetime: EvalAwaitable is a temporary materialized in the coroutine
 * frame by `co_await`, so it (and the embedded EvalRequest) is alive for
 * the entire suspension.  The GPU thread reads `tokens` and writes `result`
 * while the coroutine is suspended; the coroutine reads `result` on resume.
 *
 * Lives in `catgpt::legacy` to coexist with the libfork canonical version
 * (`catgpt::EvalAwaitable` etc.) at cpp/src/selfplay/eval_request.hpp. The
 * subnamespace is `legacy` rather than `coro` so it does not shadow the
 * libcoro `::coro::` namespace inside `namespace catgpt { ... }`.
 */

#ifndef CATGPT_SELFPLAY_LEGACY_EVAL_REQUEST_HPP
#define CATGPT_SELFPLAY_LEGACY_EVAL_REQUEST_HPP

#include <array>
#include <cmath>
#include <coroutine>
#include <cstdint>

#include "../../engine/nn_constants.hpp"
#include "../../engine/policy.hpp"

namespace catgpt::legacy {

// Forward declaration — defined in batch_evaluator.hpp
class BatchEvaluator;

inline constexpr std::size_t WDL_NUM_CLASSES = 3;

inline float wdl_logits_to_q(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    const float m = std::max({logits[0], logits[1], logits[2]});
    const float ew = std::exp(logits[0] - m);
    const float ed = std::exp(logits[1] - m);
    const float el = std::exp(logits[2] - m);
    const float inv_z = 1.0f / (ew + ed + el);
    return (ew - el) * inv_z;
}

inline float wdl_logits_to_value(const std::array<float, WDL_NUM_CLASSES>& logits) noexcept
{
    return 0.5f * (wdl_logits_to_q(logits) + 1.0f);
}

/**
 * Raw neural-network output for a single position.
 * This is what the GPU thread writes after batched inference.
 *
 * The model exports:
 *   wdl_logits          — raw WDL logits [W, D, L]
 *   value_probs         — BestQ HL-Gauss distribution (81 bins)
 *   policy              — Move distribution logits (4672)
 *   optimistic_policy   — Optimistic policy logits (4672)
 */
struct RawNNOutput {
    std::array<float, WDL_NUM_CLASSES> wdl_logits;
    std::array<float, VALUE_NUM_BINS> value_probs;
    std::array<float, POLICY_SIZE> policy;
    std::array<float, POLICY_SIZE> optimistic_policy;
};

/**
 * A single evaluation request.
 *
 * Allocated inside the coroutine frame (via EvalAwaitable).
 * The GPU thread reads `tokens` and writes `result`, then resumes
 * the coroutine via `continuation`.
 */
struct EvalRequest {
    // --- Input (written by coroutine before suspend) ---
    std::array<std::int32_t, 64> tokens;

    // --- Output (written by GPU thread before resuming coroutine) ---
    RawNNOutput result;

    // --- Coroutine handle (set by await_suspend, used by GPU thread) ---
    std::coroutine_handle<> continuation;
};

/**
 * Awaitable that submits an eval request to the batch evaluator
 * and suspends the calling coroutine until the GPU result is ready.
 *
 * Usage inside a coroutine:
 *   RawNNOutput output = co_await EvalAwaitable(evaluator, tokens);
 */
class EvalAwaitable {
public:
    EvalAwaitable(BatchEvaluator& evaluator,
                  const std::array<std::uint8_t, 64>& tokens)
        : evaluator_(evaluator)
    {
        // Convert uint8 tokens to int32 (TRT input format)
        for (int i = 0; i < 64; ++i) {
            request_.tokens[i] = static_cast<std::int32_t>(tokens[i]);
        }
    }

    // Never immediately ready — always go through the GPU.
    bool await_ready() const noexcept { return false; }

    // Submit the request and suspend.  Defined after BatchEvaluator
    // is complete (see bottom of batch_evaluator.hpp).
    void await_suspend(std::coroutine_handle<> h) noexcept;

    // Called when the coroutine resumes — the GPU thread has already
    // filled request_.result.
    RawNNOutput await_resume() noexcept {
        return request_.result;
    }

private:
    BatchEvaluator& evaluator_;
    EvalRequest request_;
};

}  // namespace catgpt::legacy

#endif  // CATGPT_SELFPLAY_LEGACY_EVAL_REQUEST_HPP
