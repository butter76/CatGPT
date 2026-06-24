/**
 * LKS search.
 *
 * Lazy-SMP-shaped chess search backed by a TensorRT BatchEvaluator per
 * worker, sharing a lock-free SearchArena, with libfork (continuation
 * stealing, segmented stacks) running the per-worker fan-out.
 *
 * Threading + ownership model:
 *
 *   - `LksSearch` is constructed with a TRT engine path and the desired
 *     fan-out (num_workers, coros_per_worker, max_batch_size). The
 *     constructor builds N persistent `WorkerSearch`es. Each owns:
 *
 *         * a `lf::lazy_pool` (1 worker thread, sleeps when no work),
 *         * a `BatchEvaluator` (own engine, own CUDA stream, own GPU
 *           thread, own pinned/device buffers),
 *         * a `LfAsyncSemaphore` (caps concurrently-alive
 *           `recursive_search` invocations to 4*K).
 *
 *     The TRT engine load and CUDA buffer allocation happen ONCE here.
 *     They live until `LksSearch` is destroyed.
 *
 *   - `search(cfg)` spawns the `worker_main` jthread and returns
 *     immediately. `worker_main` resets the per-search atomics, spawns
 *     N short-lived `runner` jthreads (one per worker_search), then
 *     loops on aggregate stats + UCI emission.
 *
 *   - Each `runner` runs its own iterative-deepening loop, with the
 *     starting depth Lazy-SMP-staggered across workers. Each iteration
 *     is a single `recursive_search` from the root, dispatched onto the
 *     worker's lazy_pool via `lf::sync_wait`. Intra-iteration batching
 *     comes from libfork fork-join over RecurseThenRead children, with
 *     the per-worker semaphore gating in-flight recursive_search
 *     invocations.
 *
 *   - `quit()` triggers `worker_main`'s stop_token. `worker_main` flips
 *     each worker's `stop` atomic and joins the runners. Persistent
 *     evaluators stay alive (their GPU threads park on empty queues)
 *     and are reused on the next `search()`.
 *
 *   - Only at `LksSearch` destruction do the evaluators' GPU threads
 *     and CUDA resources get torn down.
 *
 * Permit pattern (libfork-specific, replaces the libcoro `eval_sem`):
 *
 *   - Every entry to `recursive_search` owns one `Permit` on entry.
 *     The Permit is move-only; its destructor releases the slot back
 *     to the per-worker `LfAsyncSemaphore`.
 *
 *   - The eval-on-miss path no longer acquires/releases a separate
 *     semaphore; the entry permit covers the GPU eval.
 *
 *   - At fan-out, the FIRST RecurseThenRead child inherits the parent's
 *     permit via `lf::fork(self)(... std::move(permit) ...)`; every
 *     subsequent sibling does `co_await sem.acquire()` first to get its
 *     own permit. This is the producer-side backpressure: when permits
 *     are exhausted, the parent suspends in acquire() before any new
 *     child frame is allocated.
 *
 *   - K = LfAsyncSemaphore.count = "max simultaneously-alive
 *     recursive_search invocations per worker" (was "max in-flight GPU
 *     evals" under libcoro). Tighter bound; orchestration frames
 *     (TT-traversal / pre-pass) are bounded too.
 *
 * Search algorithm:
 *
 *   - Log-scale iterative deepening. `depth = log(N)`; each ID step
 *     bumps `depth` by `delta_depth` (default 0.2). Workers stagger
 *     their starting depth: worker `i` of `N` starts at
 *     `start_depth + (i / N) * delta_depth`.
 *
 *   - The TT (lock-free `SearchArena`) is shared across workers.
 *     `qd_packed` stores `(Q, max_depth)`.
 *
 *   - `recursive_search(board, depth)` follows the "caller-skip"
 *     contract: caller has already verified the position is not
 *     terminal-by-MoveInfo, not a path-repetition, not a 50-move
 *     draw, and not TT-cached at depth. Inside:
 *
 *       1. TT-probe (`find`).
 *       2. If miss: GPU eval, enumerate legal moves + per-move
 *          POSITION-ONLY terminal_kind detection (repetitions and
 *          50-move draws are NOT in terminal_kind — they are
 *          path-dependent), alloc node_info, fill MoveInfo, then
 *          `find_or_claim` and publish. CAS losers (peers that
 *          published during our eval await) adopt the peer's entry
 *          and orphan their bytes.
 *       3. Re-deepen check: if `rec_depth` >= 196 or the entry's
 *          max_depth >= depth, return (handing the TT (Q, max_depth)
 *          back to the caller via the `Plan* out` argument).
 *       4. Pass 1: classify children into a local Plan vector
 *          (terminal / path-dep draw / TT-hit / Unexpanded). Halley
 *          allocator fills `alloc` in place.
 *       5. Pass 2: Expanded children fork when `alloc > p.depth`;
 *          Unexpanded children expand only on iter 0 via force
 *          (top `hdr->force_expand` priors — a per-position dynamic
 *          count computed at first-eval from a temp-1.0 policy via
 *          a 95%-of-modified-entropy rule, capped at 8 — or all
 *          Unexpanded when `depth > depth_floor +
 *          log(force_all_unexpanded_log_arg)`).
 *          Pre-mark `Mode::Expanded`
 *          and spawn a child `recursive_search` via `lf::fork`, passing
 *          `&plan` as the out-param. First fork
 *          inherits the parent's permit; subsequent forks each
 *          `co_await sem.acquire()`. `lf::join` waits for all forked
 *          children. Each child writes its rolled-up `(Q, depth)`
 *          straight back into the parent's plan row — no post-join TT
 *          re-read.
 *       6. Rollup: parent's Q is the negamax of children's Q
 *          (`Q_new = max_i(-child_Q_i)`) over Expanded plans
 *          (terminal / path-dep draws contribute fixed Q; forked
 *          children contribute the values their own rollup wrote into
 *          the plan).
 *       7. `update_qd` with the new (Q, depth); also forward (Q, depth)
 *          to our caller through `*out`.
 */

#ifndef CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
#define CATGPT_ENGINE_LKS_LKS_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <exception>

#include <libfork/core.hpp>
#include <libfork/schedule.hpp>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../lf/async_semaphore.hpp"
#include "../../selfplay/batch_evaluator.hpp"
#include "../../selfplay/eval_request.hpp"
#include "../numa_util.hpp"
#include "../../syzygy.hpp"
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../trt_runtime.hpp"
#include "../fractional_mcts/search_stats.hpp"
#include "../fractional_mcts/v2/board_secondary.hpp"
#include "../fractional_mcts/v2/tt_arena.hpp"
#include "compute_allocations.hpp"

namespace catgpt::lks {

namespace fs = std::filesystem;

namespace detail {

/**
 * Search-algorithm tunables piped from `LksSearchConfig` into
 * `recursive_search` via `RecurseContext`. Treated as constant for
 * the duration of a single `search()` call: the active config lives
 * in `worker_main`'s frame for the whole search, so a `const
 * SearchParams*` in the per-iteration context stays valid throughout.
 *
 * Add new descent-time knobs (CPUCT, FPU, ...) here and read them
 * off `ctx->params` at the use site.
 */
struct SearchParams {
    float policy_temp = 1.3f;  // softmax temperature; >1 flattens, <1 sharpens

    // WDL win-vs-loss sharpening temperature for the leaf value. The softmax
    // draw prob D is held fixed; the W:L contrast is sharpened at this
    // temperature. <1 sharpens (more optimism about the favored side, shrinking
    // the selfplay-noise loss tail); 1.0 == plain P(W)-P(L).
    float wl_temp = 0.7f;

    // Root-side-dependent WDL temps. The effective value copied into
    // `wl_temp` at search launch (see LksSearch::search): a White-to-move
    // root uses wl_temp_white, otherwise wl_temp_black.
    float wl_temp_white = 0.5f;
    float wl_temp_black = 0.5f;

    // Heuristic for the initial max_depth stamped onto a freshly evaluated
    // TT entry: default_max_depth = -log(variance * C). Low-variance nodes
    // get a high max_depth so early ID iterations skip re-descending them
    // (trust the NN). C is this constant.
    float default_depth_constant = 12.0f;

    // PUCT allocation constant. Feeds Halley-in-delta dual solve:
    //   log N_i = log P_i + 2d/3 + log(c_puct/3) - log(e^u + Δ_i),
    // where u = log(K - q_max), Δ_i = q_max - q_i, q_i = -Q_i.
    float c_puct = 1.75f;

    // FPU reduction. For unexpanded children we synthesize a parent-POV Q via
    //   Q_eff_parent_pov = parent_Q - fpu_reduction * sqrt(cumulative_P),
    // where cumulative_P sums priors of preceding P-sorted children (expanded
    // or not).
    float fpu_reduction = 0.330f;

    // Clamp loop: cap per-iteration depth growth and break-out tolerance.
    //   - each forked child is dispatched at
    //     min(p.alloc, p.depth + clamp_step * jitter), where `jitter` is
    //     drawn uniformly from [0.5, 1.5] per fork from a per-worker-thread
    //     RNG (see clamp_step_jitter()); this perturbs the per-fork depth
    //     step to give Lazy-SMP workers naturally divergent descent shapes
    //     without changing the long-run allocator fixed point;
    //   - loop breaks (after iter 0) once every plan has
    //     p.alloc - p.depth <= break_eps (i.e. the Halley allocator is at
    //     a fixed point and no child wants meaningfully more depth);
    //   - clamp_max_iters is a defensive cap on the outer loop.
    // Rationale: Halley allocations can swing wildly as FPU-optimistic
    // siblings resolve; small per-iter growth + re-allocation avoids
    // burning depth on children whose Q will collapse on first contact.
    float clamp_step      = 0.4f;
    float break_eps       = 0.1f;
    int   clamp_max_iters = 1024;

    // Iter-0 clamp loop: force-expand every Unexpanded child (not just the
    // top hdr->force_expand priors) when depth > depth_floor + log(this).
    float force_all_unexpanded_log_arg = 40.0f;
};

}  // namespace detail

/**
 * Game-clock time management for a single `go`.
 *
 * Two ways to bound a search by wall-clock time:
 *   - `movetime_ms > 0`: spend exactly that long (soft == hard).
 *   - `wtime_ms`/`btime_ms` (+ optional inc): derive a soft/hard budget
 *     from the side-to-move's remaining clock. `worker_main` reads the
 *     side to move off `board_` and picks the matching clock/inc.
 *
 * The budget model (mirrors the legacy chessbench engine):
 *   reserve_ms is always held back; bank = max(0, time_left - reserve_ms).
 *   soft = soft_pct*bank + inc   (target: stop after the iteration that
 *                                 crosses it)
 *   hard = min(hard_pct*bank + inc, bank)  (force-stop ceiling)
 * Soft is boosted by first_move_pct (game's first move) and surprise_pct
 * (opponent played an unpredicted reply), and extended mid-search by
 * change_bonus_pct (best move changed) / worsen_bonus_pct (root score
 * dropped by more than worsen_threshold_cp), each applied at most once
 * and always clamped to hard.
 *
 * `early_return_margin` is a driver-only tunable (worker_main ignores it);
 * see the early-return chain in lks_uci_main.cpp.
 *
 * Every `*_pct`/`*_ms`/`*_cp` field is an exposed tunable so the search
 * can be retuned without recompiling (see the LKS_TIME_* env vars in
 * lks_uci_main.cpp). A budget is "active" iff movetime_ms > 0 OR
 * wtime_ms > 0 OR btime_ms > 0; otherwise the search is time-unbounded.
 */
struct TimeControl {
    // ── Per-go clock inputs (from the `go` command); <=0 means unset ──
    int64_t wtime_ms    = -1;
    int64_t btime_ms    = -1;
    int64_t winc_ms     = 0;
    int64_t binc_ms     = 0;
    int64_t movetime_ms = -1;

    // ── Cross-move flags (set by the UCI driver) ──
    bool first_move = false;  // first search of the game
    bool surprise   = false;  // opponent's reply differed from our prediction

    // ── Tunable constants (defaults mirror the legacy engine) ──
    int64_t reserve_ms          = 100;     // never spend below this much clock
    float   soft_pct            = 0.035f;   // soft target as fraction of bank
    float   hard_pct            = 0.28f;   // hard ceiling as fraction of bank
    float   first_move_pct      = 0.07f;   // soft floor on the first move
    float   surprise_pct        = 0.05f;   // soft floor on a surprise
    float   change_bonus_pct    = 0.015f;   // soft extension when best move changes
    float   worsen_bonus_pct    = 0.02f;   // soft extension when score drops
    int     worsen_threshold_cp = 10;      // cp drop that counts as "worsened"

    // Early-return ("ponderhit-like") chain margin, in log-depth units.
    // Driver-only field (worker_main never reads it): when the opponent
    // plays our predicted reply and the new root's TT depth is within this
    // margin of the anchored root depth, the UCI driver replies straight
    // from the TT without launching a search. See lks_uci_main.cpp.
    float   early_return_margin = 0.3f;

    // Active iff some clock or an explicit movetime was supplied.
    [[nodiscard]] bool active() const noexcept {
        return movetime_ms > 0 || wtime_ms > 0 || btime_ms > 0;
    }
};

/**
 * Per-search configuration.
 *
 * `on_uci_line` is invoked from `worker_main`, one call per UCI line.
 * The string_view is valid only for the duration of the call. For UCI
 * production install:
 *   `[](std::string_view s){ std::cout << s << '\n'; std::cout.flush(); }`
 * For tests install a recording lambda.
 */
struct LksSearchConfig {
    // Aggregate eval budget across all workers. Enforced with a grace
    // period in `worker_main`: once `sum(w->evals) >= max_evals`, the
    // search keeps running until `min_depth()` strictly advances (i.e.
    // the slowest worker finishes one more ID iteration), then stops.
    // Expect the final `total_evals()` to overshoot this by up to one
    // slow-worker iteration's worth of evals.
    uint64_t max_evals = 800;

    // Iterative-deepening (log-scale). N = e^depth.
    float start_depth = 0.0f;       // worker 0's starting depth
    float delta_depth = 0.2f;       // per-iteration depth step
    float max_depth   = 32.0f;      // absolute depth cap (e^32 ~= 8e13)

    // Stop the search the moment min_depth() across workers reaches this
    // value. `+infinity` (default) disables the check. The UCI driver
    // maps `go depth N` to `N / 100.0f` (centi-depth, matching the
    // encoding used in `info depth ...`).
    float target_min_depth = std::numeric_limits<float>::infinity();

    detail::SearchParams params{};  // descent-time tunables (see SearchParams)

    TimeControl time{};  // wall-clock budget (see TimeControl)

    std::function<void(std::string_view)> on_uci_line;
};

namespace detail {

// (SearchParams is declared earlier in this namespace — above
// LksSearchConfig — so the config can embed it by value.)

/**
 * Paired legal move + its softmax prior. Emitted by
 * `softmax_legal_sorted` in decreasing-P order so downstream MoveInfo
 * fill is a single sequential pass and the arena ends up P-sorted
 * without an in-place shuffle.
 *
 * Stored as the raw `uint16_t` rather than a `chess::Move` because the
 * latter carries an unused 16-bit score field that would bloat this
 * struct.
 */
struct MoveWithPriors {
    uint16_t move;   // 2: chess::Move underlying u16
    uint16_t _pad;   // 2
    float    P;      // 4: policy prior (sort key)
};
static_assert(sizeof(MoveWithPriors) == 8, "MoveWithPriors must be 8 bytes");

/**
 * Tempered softmax over legal-move policy logits — runs one softmax over
 * the GPU-gathered `out.legal_policy[0..num_moves)` at `policy_temp`, then
 * sorts by decreasing `P`. Output order is what downstream descent expects
 * (highest-prior first), and is also the order we want in the arena's
 * MoveInfo[].
 *
 * The flat (4672) policy tensor never reaches the C++ side anymore: the
 * exporter does a Gather inside TRT against the caller-supplied legal_indices,
 * and `out.legal_policy[i]` is already the logit for `legal[i]`. Padding
 * entries past `legal.size()` are dont-care.
 *
 * `policy_temp` divides logits before the max-subtract / exp; T > 1 flattens,
 * T < 1 sharpens. T == 1.0 reproduces the plain softmax. The division is
 * folded into `inv_temp` and applied in the first pass so the existing
 * max-subtract / exp / normalize loop is unchanged.
 */
inline void softmax_legal_sorted(const RawNNOutput& out,
                                 const chess::Movelist& legal,
                                 float policy_temp,
                                 std::vector<MoveWithPriors>& moves)
{
    const int n = legal.size();
    moves.resize(static_cast<size_t>(n));

    const float inv_temp = 1.0f / policy_temp;

    // Pre-pass: copy legal moves and scaled logits. Track the max for the
    // LSE max-subtract.
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        const float scaled = out.legal_policy[i] * inv_temp;
        moves[i].move  = static_cast<uint16_t>(legal[i].move());
        moves[i]._pad  = 0;
        moves[i].P     = scaled;
        max_logit      = std::max(max_logit, scaled);
    }

    // exp(scaled - max), accumulating the sum.
    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        moves[i].P = std::exp(moves[i].P - max_logit);
        sum_exp   += moves[i].P;
    }

    // Normalize. If TRT produced non-finite logits we'd otherwise get NaN
    // priors and poison every subsequent allocator call; fall back to uniform
    // priors over legal moves so a misbehaving NN can't brick search.
    const float inv_sum =
        (std::isfinite(sum_exp) && sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    if (inv_sum == 0.0f) {
        const float uniform = (n > 0) ? (1.0f / static_cast<float>(n)) : 0.0f;
        for (int i = 0; i < n; ++i) moves[i].P = uniform;
    } else {
        for (int i = 0; i < n; ++i) moves[i].P *= inv_sum;
    }

    // Sort by descending P.
    std::sort(moves.begin(), moves.end(),
        [](const MoveWithPriors& a, const MoveWithPriors& b) {
            return a.P > b.P;
        });
}

/**
 * Build the (padded) flat-policy-index array consumed by the GPU's
 * end-of-policy-head gather. Mirrors `softmax_legal_sorted`'s legal-move
 * iteration order — `legal[i]` ↔ `out.legal_policy[i]` ↔ `indices[i]`.
 *
 * Padding slots beyond `legal.size()` are filled with 0; the GPU still
 * gathers them but the C++ softmax only reads the first `legal.size()`
 * entries. Any in-range index works as padding.
 */
inline void build_legal_indices(
    const chess::Movelist& legal,
    bool flip_for_black,
    std::array<std::int32_t, MAX_LEGAL_MOVES>& indices) noexcept
{
    const int n = legal.size();
    for (int i = 0; i < n; ++i) {
        const auto [from_idx, to_idx] =
            encode_move_to_policy_index(legal[i], flip_for_black);
        indices[i] = static_cast<std::int32_t>(
            policy_flat_index(from_idx, to_idx));
    }
    for (int i = n; i < MAX_LEGAL_MOVES; ++i) indices[i] = 0;
}

/**
 * Variance of the NN value distribution, measured in the same
 * [-1, 1] scale as Q from WDL logits. The HL-Gauss head's 81
 * bins are taken as equal-width over [-1, 1] with the i-th center
 * at `-1 + (i + 0.5) * (2 / VALUE_NUM_BINS)`.
 *
 * Mirrors `FractionalNode::compute_variance` in
 * `engine/fractional_mcts/node.hpp` so LKS agrees on scale with
 * the prototype search.
 */
inline float compute_value_variance(
    const std::array<float, VALUE_NUM_BINS>& value_probs) noexcept
{
    constexpr float bin_width = 2.0f / static_cast<float>(VALUE_NUM_BINS);
    float mean = 0.0f;
    for (int i = 0; i < VALUE_NUM_BINS; ++i) {
        const float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        mean += value_probs[i] * center;
    }
    float var = 0.0f;
    for (int i = 0; i < VALUE_NUM_BINS; ++i) {
        const float center = -1.0f + (static_cast<float>(i) + 0.5f) * bin_width;
        const float diff = center - mean;
        var += value_probs[i] * diff * diff;
    }
    return var;
}

/**
 * Per-position iter-0 force-expand count, computed once at first
 * evaluation from the GPU's legal-move policy logits.
 *
 * Intuition: rather than a fixed `force_expand_count` knob, we let
 * each position's policy decide how many top-prior children deserve
 * an unconditional iter-0 expansion. Sharp tactical positions (one
 * dominant move) get a small count; quiet, diffuse positions (many
 * plausible replies) get more.
 *
 * Algorithm:
 *   1. Temp-1.0 softmax over `out.legal_policy[0..n)` -> p[i].
 *   2. Subtract a noise floor of 1/300 from every entry, capped at 0:
 *        q_i = max(0, p_i - 1/300)
 *      Renormalize q to sum to 1 (when the sum is > 0).
 *   3. Entropy H = -sum(q * log q) of the renormalized q.
 *   4. Sort q descending; pick the smallest k whose top-k partial
 *      sum of -q*log(q) is >= 0.95 * H.
 *   5. Clamp to [1, 8] when n > 0; return 0 when n == 0.
 *
 * The 1/300 floor strips out long-tail near-zero priors so they
 * don't pad the entropy, and the 95% cover keeps the count
 * concentrated on moves that meaningfully share the policy mass.
 */
inline uint16_t compute_force_expand(const RawNNOutput& out, int num_moves) noexcept {
    constexpr uint16_t kMinForce  = 1;
    // Not generally load-bearing: clamping here typically moves nodes evaluated
    // by only ~3%. A generous max is for pathological subtrees only.
    constexpr uint16_t kMaxForce  = 8;
    constexpr float    kNoiseFloor = 1.0f / 300.0f;
    constexpr float    kCover      = 0.95f;

    if (num_moves <= 0) return 0;

    // Temp-1.0 softmax (LSE max-subtract for numerical stability).
    std::array<float, MAX_LEGAL_MOVES> q{};
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_moves; ++i) {
        max_logit = std::max(max_logit, out.legal_policy[i]);
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < num_moves; ++i) {
        const float e = std::exp(out.legal_policy[i] - max_logit);
        q[i] = e;
        sum_exp += e;
    }
    if (!std::isfinite(sum_exp) || sum_exp <= 0.0f) {
        // Degenerate logits (NaN / -inf max). Match softmax_legal_sorted's
        // uniform fallback so the floor kicks in.
        const float uniform = 1.0f / static_cast<float>(num_moves);
        for (int i = 0; i < num_moves; ++i) q[i] = uniform;
    } else {
        const float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < num_moves; ++i) q[i] *= inv_sum;
    }

    // Noise-floor subtract + renormalize.
    float s = 0.0f;
    for (int i = 0; i < num_moves; ++i) {
        const float v = q[i] - kNoiseFloor;
        q[i] = (v > 0.0f) ? v : 0.0f;
        s += q[i];
    }
    if (s <= 0.0f) {
        return std::min<uint16_t>(kMinForce, static_cast<uint16_t>(num_moves));
    }
    const float inv_s = 1.0f / s;
    for (int i = 0; i < num_moves; ++i) q[i] *= inv_s;

    // Sort descending (only the first `num_moves` entries are valid).
    std::sort(q.begin(), q.begin() + num_moves, std::greater<float>());

    // Entropy of the renormalized q.
    float H = 0.0f;
    for (int i = 0; i < num_moves; ++i) {
        if (q[i] > 0.0f) H -= q[i] * std::log(q[i]);
    }
    if (H <= 0.0f) {
        return std::min<uint16_t>(kMinForce, static_cast<uint16_t>(num_moves));
    }

    // Smallest k with top-k entropy mass >= 0.95 * H.
    const float target = kCover * H;
    float partial = 0.0f;
    int k = 0;
    for (int i = 0; i < num_moves; ++i) {
        if (q[i] > 0.0f) partial -= q[i] * std::log(q[i]);
        ++k;
        if (partial >= target) break;
    }

    const int clamped = std::clamp(k, static_cast<int>(kMinForce),
                                      static_cast<int>(kMaxForce));
    return static_cast<uint16_t>(std::min(clamped, num_moves));
}

/**
 * Position-only terminal classification for a child move at expansion
 * time. Path-dependent terminal conditions are NOT considered here:
 *   - repetitions: handled at descent time via `isRepetition(1)`
 *     (plain 2-fold repetition draw; not cached in terminal_kind).
 *   - 50-move rule: handled at descent time via `isHalfMoveDraw()`.
 * Both depend on data (path history / half-move clock) that is not
 * part of the Zobrist key, so two transpositions to the same key can
 * disagree on them. Only stable, position-only draws (insufficient
 * material), mate/stalemate, and Syzygy-resolved WDL (theoretical,
 * rule50=0) go into the TT-stored terminal_kind.
 *
 * Pre: `child_board` is the board AFTER `makeMove<true>(move)`.
 *
 * `syz` is optional; when non-null and the child board is Syzygy-eligible
 * (piece-count + no-castling) the WDL probe runs after the natural-
 * terminal checks. probe_wdl uses rule50=0 internally, so the resulting
 * kind is path-independent (safe to cache in TT). Cursed/blessed are
 * conservatively folded to Draw — under the 50-move rule they are
 * theoretical draws and treating them as wins/losses risks the search
 * trusting near-50-ply outcomes that won't actually materialize.
 */
inline v2::TerminalKind classify_terminal(chess::Board& child_board,
                                          const catgpt::SyzygyProber* syz)
{
    chess::Movelist child_legal;
    chess::movegen::legalmoves(child_legal, child_board);
    if (child_legal.empty()) {
        // No legal replies: checkmate => loss for child; stalemate => draw.
        return child_board.inCheck() ? v2::kTerminalLossForChild
                                     : v2::kTerminalDraw;
    }
    if (child_board.isInsufficientMaterial()) {
        return v2::kTerminalDraw;
    }
    if (syz != nullptr && syz->is_eligible(child_board)) {
        if (auto wdl = syz->probe_wdl(child_board); wdl) {
            switch (*wdl) {
                case catgpt::SyzygyWDL::WIN:
                    return v2::kTerminalWinForChild;
                case catgpt::SyzygyWDL::LOSS:
                    return v2::kTerminalLossForChild;
                case catgpt::SyzygyWDL::DRAW:
                case catgpt::SyzygyWDL::CURSED_WIN:
                case catgpt::SyzygyWDL::BLESSED_LOSS:
                    return v2::kTerminalDraw;
            }
        }
    }
    return v2::kTerminalNone;
}

/**
 * Per-worker_search state.
 *
 * Persistent across searches:
 *   - pool, evaluator, sem
 *
 * Per-search (reset by worker_main at the start of every search):
 *   - stop, evals, tt_claims, depth
 *
 * Declaration order matters for safe destruction:
 *   sem        -> destroyed first  (no waiters by then; runners joined)
 *   evaluator  -> destroyed second (joins GPU thread; no more pool->schedule)
 *   pool       -> destroyed last   (safe now; GPU thread won't call into it)
 */
struct WorkerSearch {
    std::unique_ptr<lf::lazy_pool>            pool;       // 1 worker
    std::unique_ptr<BatchEvaluator>           evaluator;
    std::unique_ptr<lfsync::LfAsyncSemaphore> sem;        // permits = coros_per_worker

    int device_id = 0;  // CUDA device this worker's evaluator is bound to.

    // NUMA placement, filled in by the LksSearch ctor before the per-worker
    // init thread runs. `numa_node` is the OS node id local to this worker's
    // GPU; `compute_cpu` is the physical core the lazy_pool descent thread is
    // pinned to; `gpu_cpu` is the SMT sibling the BatchEvaluator GPU thread
    // is pinned to. All -1 when pinning is unavailable (no /sys topology).
    int numa_node   = 0;
    int compute_cpu = -1;
    int gpu_cpu     = -1;

    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> evals{0};
    std::atomic<uint64_t> tt_claims{0};
    std::atomic<float>    depth{0.0f};

    std::jthread runner;

    WorkerSearch() = default;
    WorkerSearch(const WorkerSearch&) = delete;
    WorkerSearch& operator=(const WorkerSearch&) = delete;
    WorkerSearch(WorkerSearch&&) = delete;
    WorkerSearch& operator=(WorkerSearch&&) = delete;

    // Cooperative stop check, hit at every `co_await` boundary so a search
    // drains promptly once `worker_main` flips the flag. The aggregate eval
    // ceiling (`cfg.max_evals`) is enforced centrally by `worker_main`'s
    // grace-period state machine, not here — individual workers do NOT cap
    // themselves on eval count.
    [[nodiscard]] bool should_abort() const noexcept {
        return stop.load(std::memory_order_relaxed);
    }
};

/**
 * Stable bag of per-worker dependencies passed by pointer into the
 * recursive_search lambda (libfork lambdas can't capture, so all
 * dependencies must be arguments). Constructed once per ID iteration
 * in run_iteration().
 */
struct RecurseContext {
    v2::SearchArena*          arena;
    BatchEvaluator*           evaluator;
    lfsync::LfAsyncSemaphore* sem;
    WorkerSearch*             w;
    const SearchParams*       params;  // descent-time tunables; lives in worker_main's cfg
    // Optional Syzygy tablebase prober. nullptr when no Syzygy path was
    // supplied to LksSearch. SyzygyProber::probe_wdl is thread-safe under
    // Fathom's default compile (no TB_NO_THREADS), so the same instance
    // feeds every worker's RecurseContext.
    const catgpt::SyzygyProber* syzygy;
};

/**
 * A TT Q is stored without repetition/50-move history (Zobrist excludes it),
 * so a position scored as losing may actually be a forceable draw on THIS
 * path. When the side to move at `b` can force an upcoming repetition, floor a
 * losing (Q<0) reading up to a draw: that side can always take the repetition
 * instead of walking into the losing line. A winning side would never opt into
 * the draw, so winning readings are left untouched. Path-dependent: applied to
 * the consumed value only, never written back to the shared arena. The `q < 0`
 * test short-circuits the cuckoo walk on the common (non-losing) case.
 * Uses the plain 2-fold variant (`upcomingRepetitionTwoFold`): any reachable
 * repetition can be forced, independent of the search root.
 */
[[nodiscard]] inline float clamp_q_upcoming_rep(
    float q, const chess::Board& b) noexcept {
    return (q < 0.0f && b.upcomingRepetitionTwoFold()) ? 0.0f : q;
}

/**
 * Recursive descent (libfork lambda).
 *
 * INVARIANT: every entry owns exactly one `Permit` on entry. The permit
 * covers any GPU eval. At fan-out the FIRST forked child of the FIRST
 * clamp-loop iteration inherits the parent's permit via std::move; every
 * other fork (later iter-0 siblings, and every fork in iter 1+) does
 * co_await sem->acquire() for its own permit. After iter 0's lf::join
 * the parent is permit-less for the remainder of the call (same state
 * the original single-shot code was in during rollup); iter 1+ forks
 * acquire on demand. On exit any permit still held is released by RAII.
 *
 * This preserves the property that K = sem.count bounds "concurrently
 * alive recursive_search frames that hold a permit", NOT "concurrently
 * alive frames" — so descent depth is not constrained by K. A
 * parent-retains-permit scheme would deadlock at depth K (parent holds
 * one, fork-child acquires the K-th, child can't fan out because all
 * permits are held by ancestor parents up the spine).
 *
 * `out` is the caller's Plan row for this child (null at the root). On
 * the normal-return / re-deepen-exit paths this function writes the
 * child's final `(Q, depth)` to `*out` so the parent can roll up
 * without a second TT probe. Abort paths intentionally leave `*out`
 * untouched — the parent's own `should_abort()` check fires in the
 * same iteration and skips its rollup before the stale row is read.
 */
inline constexpr uint32_t kRecursiveDepthLimit = 196;

// Per-fork multiplier in [0.5, 1.5] applied to `clamp_step` when computing
// a forked child's target depth. The RNG is `thread_local`, and each
// `WorkerSearch` owns a 1-thread `lazy_pool`, so every worker draws from
// an independent stream — giving Lazy-SMP workers naturally divergent
// per-iteration depth steps. The interval is symmetric around 1, so the
// expected step is unchanged from the deterministic clamp_step; only the
// per-fork shape is perturbed.
inline float clamp_step_jitter() noexcept {
    thread_local std::mt19937 rng{std::random_device{}()};
    thread_local std::uniform_real_distribution<float> dist(0.5f, 1.5f);
    return dist(rng);
}

inline constexpr auto recursive_search =
    [](auto recursive_search,
       RecurseContext* ctx,
       lfsync::Permit permit,
       chess::Board board,
       float depth,
       uint32_t rec_depth,
       bool pv_mode,
       Plan* out) -> lf::task<void>
{
    if (ctx->w->should_abort()) co_return;

    const uint64_t key = board.hash();
    const uint32_t sec = v2::secondary_hash(board);
    v2::TTEntry* entry = ctx->arena->find(key, sec);

    if (entry == nullptr) {
        // ── unexpanded: eval, fill bytes, then claim ────────────────
        // We already hold a Permit (entry invariant) — no separate
        // eval-sem acquire/release, and therefore no yield point
        // between the find() above and the find_or_claim() below
        // where a peer could publish without us noticing. The CAS in
        // find_or_claim is the sole source of truth for ownership;
        // the eval-await window's race is handled there.
        //
        // Legal-move generation moves to BEFORE the eval submit so the
        // GPU can gather only the legal-move logits (cuts policy D2H from
        // POLICY_SIZE=4672 to MAX_LEGAL_MOVES=218 floats per request).
        // The encoded flat indices ride along on the EvalRequest; the
        // gathered logits arrive in `out.legal_policy[0..num_moves)`,
        // already aligned with `legal[]`.
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        const uint16_t num_moves = static_cast<uint16_t>(legal.size());

        std::array<std::int32_t, MAX_LEGAL_MOVES> legal_indices;
        const bool flip = board.sideToMove() == chess::Color::BLACK;
        build_legal_indices(legal, flip, legal_indices);

        auto tokens = catgpt::tokenize<BatchEvaluator::SEQ_LENGTH>(
            board, NO_HALFMOVE_CONFIG);
        RawNNOutput out = co_await EvalAwaitable(
            *ctx->evaluator, tokens, legal_indices);
        ctx->w->evals.fetch_add(1, std::memory_order_relaxed);

        // Sort moves by decreasing P up front (on compact 8-byte
        // pairs) so the arena fill below is a single pass that
        // writes each MoveInfo exactly once in the order descent
        // will consume them.
        std::vector<MoveWithPriors> sorted_moves;
        softmax_legal_sorted(out, legal, ctx->params->policy_temp, sorted_moves);

        // alloc + fill BEFORE attempting to claim — these bytes
        // are privately owned until the CAS, and orphaned if the
        // CAS loses.
        const uint64_t off = ctx->arena->alloc_node_info(num_moves);
        v2::MoveInfo* mi = ctx->arena->moves_at(off);
        for (uint16_t i = 0; i < num_moves; ++i) {
            const chess::Move m{sorted_moves[i].move};

            // Per-move position-only terminal detection. With a Syzygy
            // prober wired in, TB-resolved WDL also lands here (cached
            // in the child's MoveInfo so re-visits skip the probe).
            board.makeMove<true>(m);
            const v2::TerminalKind tk = classify_terminal(board, ctx->syzygy);
            board.unmakeMove(m);

            mi[i] = v2::MoveInfo::pack(sorted_moves[i].move,
                                       sorted_moves[i].P,
                                       tk);
        }

        // Per-node variance for this fresh TT entry. alloc_node_info
        // pre-fills variance=0; overwrite with the real value before
        // publish so any reader observing this node sees it.
        const float variance = std::max(compute_value_variance(out.value_probs), 0.25f / ctx->params->default_depth_constant);
        v2::NodeInfoHeader* hdr_w = ctx->arena->info_at(off);
        hdr_w->variance = variance;
        hdr_w->force_expand = compute_force_expand(out, static_cast<int>(num_moves));

        const float Q = catgpt::wdl_logits_to_q_wl_tempered(
            out.wdl_logits, ctx->params->wl_temp);

        // Heuristic initial max_depth for this fresh TT entry:
        //   default_max_depth = -log(variance * C)
        // Low-variance nodes get a HIGH max_depth so early ID iterations
        // skip re-descending them via the re-deepen check below; trust
        // the NN value. High-variance nodes get a LOW (possibly negative)
        // max_depth so the very first descent proceeds and the rollup
        // refines Q. Edge case `variance == 0` produces -inf/+inf, which
        // is safe: the re-deepen check uses `<=` and `update_qd` is
        // monotonic and will overwrite -inf on the first rollup that
        // completes a descent.
        const float default_max_depth = -std::log(
            variance * ctx->params->default_depth_constant);

        // Single 128-bit CAS installs (key, qd_packed(Q, default_max_depth))
        // atomically, so any reader observing key == K necessarily
        // sees the matching qd. Then a single 128-bit release-store
        // of Cell B publishes (origQ, key_secondary, info_offset).
        auto [ce, claimed] = ctx->arena->find_or_claim(
            key, sec, Q, /*max_depth=*/default_max_depth);
        if (claimed) {
            v2::SearchArena::publish_info(
                ce, /*origQ=*/Q, /*key_secondary=*/sec, off);
            ctx->w->tt_claims.fetch_add(1, std::memory_order_relaxed);
        } else {
            // A peer published this key while we were awaiting the
            // GPU eval. Adopt their entry; our `off` bytes are
            // orphaned (forever).
        }

        // Intentionally let the thread finish putting the entry
        // into the TT/arena before aborting.
        if (ctx->w->should_abort()) co_return;

        entry = ce;
    }

    // ── load node info (move list + variance + expanded flag) ──────
    // Cell B is provably published by the time we reach this point:
    //   * `find(key, sec)` only returns non-null after validating
    //     the secondary, which requires Cell B to be published
    //     (the validation acquire-loads it).
    //   * On the fresh-claim path we called `publish_info`
    //     ourselves before assigning `entry`.
    //   * On the lost-CAS / adopt-peer path, find_or_claim's
    //     (key, secondary) match goes through `secondary_matches`,
    //     which only returns true after acquire-observing the
    //     winning peer's Cell B publish — same HB chain as `find`.
    // So a plain `load_info` is sufficient — no spin needed.
    //
    // This used to live AFTER the re-deepen check; it moved up so the
    // PV-mode waiver can read `hdr->expanded` to decide whether to
    // skip the depth-monotonic short-circuit. The cost is one extra
    // cache line touched on the re-deepen-skip path; in exchange the
    // PV-mode descent never replays a stale TT (Q, max_depth) snapshot
    // for a node that was never actually rolled up here.
    const v2::InfoCell info_cell = v2::SearchArena::load_info(entry);
    assert(info_cell.info_offset != v2::kNoInfoOffset
           && "Cell B unpublished after find/find_or_claim returned a "
              "validated entry; invariant broken");
    const v2::NodeInfoHeader* hdr = ctx->arena->info_at(info_cell.info_offset);
    const uint16_t num_moves = hdr->num_moves;
    const v2::MoveInfo* moves = ctx->arena->moves_at(info_cell.info_offset);

    // ── re-deepen check (works for both fresh-claim and TT-shared) ──
    // PV-mode waiver: when this descent runs under pv_mode and the
    // entry has at least one prior real rollup (`hdr->expanded == 1`),
    // we ignore the depth-monotonic gate and descend anyway. This is
    // what lets the PV walk a real chain of recurse-driven moves at
    // every ply instead of replaying TT (Q, max_depth) snapshots that
    // can come from a different path (and may secretly hide a
    // repetition along *this* PV).
    //
    // The recursive-depth cap is NEVER waived — runaway descent depth
    // is a real risk even on the PV.
    {
        // Cell A is atomic with the key match (find / find_or_claim
        // both ensure we observe a key from a successful CAS), so qd
        // is never torn here.
        auto [cur_q, cur_max_d] = v2::unpack_qd(
            v2::SearchArena::load_qd(entry).qd_packed);

        // PV modes ignore the TT depth, bounded by -4.0f
        // Also we should always descend if the current Q is positive
        // to prevent blindness to repetitions.
        const bool waive_depth_check =
            pv_mode &&
            std::atomic_ref<uint8_t>(hdr->expanded)
                .load(std::memory_order_relaxed) != 0 && (depth > -4.0f || cur_q > 0.2f);
        if (rec_depth >= kRecursiveDepthLimit ||
            (!waive_depth_check && depth <= cur_max_d)) {
            // Hand the already-unpacked TT (Q, max_depth) back to the
            // caller — same value the deleted post-join re-read would
            // have produced. At the recursion cap, skip Pass 1+ as if
            // re-deepened. The parent's `out->isPV` flag is left
            // alone here: no fresh descent happened, so any prior PV
            // claim on this row remains the freshest data the parent
            // has.
            if (out) {
                out->Q = clamp_q_upcoming_rep(cur_q, board);
                out->depth = cur_max_d;
            }
            co_return;
        }
    }

    // We're committed to descending past the re-deepen check. Mark this
    // node as "has had at least one real rollup (or is about to)" so
    // future PV-mode visitors may waive the depth-monotonic gate. Plain
    // byte storage with `std::atomic_ref` for relaxed atomicity — every
    // writer stores the same value (1), so concurrent races are
    // idempotent and a missed write only delays the waiver by one
    // iteration.
    std::atomic_ref<uint8_t>(hdr->expanded)
        .store(1, std::memory_order_relaxed);

    // Parent Q (from the same TT entry we just validated) feeds the FPU
    // estimate for unexpanded children. For a freshly-claimed entry this
    // is the NN's 2*value-1; for a TT-shared entry it's whatever the
    // last rollup wrote. Both are well-defined and acquire-safe here.
    const float parent_Q = [entry]() {
        auto [q, _] = v2::unpack_qd(v2::SearchArena::load_qd(entry).qd_packed);
        (void)_;
        return q;
    }();

    // Variance-scaled depth floor: threshold for iter-0 force-all
    // Unexpanded expansion (`depth > depth_floor +
    // log(force_all_unexpanded_log_arg)`). Same
    // formula as default_max_depth on fresh TT entries.
    const float depth_floor = -std::log(
        hdr->variance * ctx->params->default_depth_constant);
    assert(std::isfinite(depth_floor) &&
           "depth_floor non-finite: hdr->variance * default_depth_constant must be > 0");
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();

    // ── Pass 1: classify children, populate Plan rows (no co_await) ──
    std::vector<Plan> plans;
    plans.reserve(num_moves);

    const float fpu_reduction = ctx->params->fpu_reduction;
    float cumulative_P = 0.0f;
    constexpr float kPosInf = std::numeric_limits<float>::infinity();

    // Set true if any of the first `hdr->force_expand` (highest-prior)
    // moves is a path-dependent draw — a 2-fold repetition draw
    // (`isRepetition(1)`) / 50-move draw on this path. When a top move only
    // draws, iter-0 force-expands ALL unexpanded children (see Pass 2's
    // force_all_unexpanded) so the descent looks past the draw for a
    // non-drawing alternative.
    bool top_move_pathdraw = false;

    for (uint16_t i = 0; i < num_moves; ++i) {
        const auto& m = moves[i];
        const v2::TerminalKind m_tk = m.terminal_kind();
        const float m_P = m.P();

        // Position-only terminals: Expanded with fixed Q and depth=+inf
        // (so the "alloc > depth" gate never triggers a re-recurse).
        // Q is stored in child-STM convention (same as TT): a child-loss
        // is Q=-1 ⇒ parent's rollup -Q = +1 ⇒ we win.
        //
        // isPV=true on all depth=+inf rows: terminal Q is exact (no
        // refinement possible) AND we must NEVER let the iter > 0 PV
        // force-fork dispatch a recursive_search on a terminal child
        // (caller-skip contract requires the parent to filter out
        // terminals before recursing). Pre-marking isPV=true makes
        // the break gate satisfiable when the best move is terminal,
        // and combined with the `std::isfinite(p.depth)` guard on
        // is_pv_force below, no terminal child is ever forked.
        if (m_tk == v2::kTerminalDraw) {
            plans.push_back({Mode::Expanded, m_P, /*Q=*/0.0f, kPosInf,
                             /*alloc=*/0.0f, /*isPV=*/true});
            cumulative_P += m_P;
            continue;
        }
        if (m_tk == v2::kTerminalLossForChild) {
            plans.push_back({Mode::Expanded, m_P, /*Q=*/-1.0f, kPosInf,
                             /*alloc=*/0.0f, /*isPV=*/true});
            cumulative_P += m_P;
            continue;
        }
        if (m_tk == v2::kTerminalWinForChild) {
            // Reachable only via Syzygy: child-STM is in a TB-won position,
            // so child_Q=+1 ⇒ parent rollup -Q = -1 ⇒ this move loses for us.
            plans.push_back({Mode::Expanded, m_P, /*Q=*/+1.0f, kPosInf,
                             /*alloc=*/0.0f, /*isPV=*/true});
            cumulative_P += m_P;
            continue;
        }

        chess::Board cb = board;
        cb.makeMove<true>(chess::Move{m.move});

        // Path-dependent draw handling. None of this is in terminal_kind:
        // different paths reaching the same parent key may disagree on
        // repetition / half-move clock, so the draw can't be promoted to a
        // path-independent terminal. All draw rows are Expanded with
        // depth=+inf (never re-recurse) and pre-marked isPV=true for the
        // same reason as terminal_kind rows (see comment above).
        //
        //   50-move rule: a plain path-local draw — drawn for THIS caller
        //   only.
        //
        //   repetition: a path-local draw for the discoverer under a plain
        //   2-fold repetition check (`isRepetition(1)`): any position seen once
        //   before within the half-move window draws.
        bool draw_here = false;
        if (cb.isHalfMoveDraw()) {
            draw_here = true;
        } else if (cb.isRepetition(1)) {
            draw_here = true;
        }
        if (draw_here) {
            // A path-dependent draw among the top-prior moves triggers the
            // wider iter-0 force-expand (see Pass 2). `hdr->force_expand`
            // is the same count Pass 2 gates per-index force-expansion on.
            if (i < hdr->force_expand) top_move_pathdraw = true;
            plans.push_back({Mode::Expanded, m_P, /*Q=*/0.0f, kPosInf,
                             /*alloc=*/0.0f, /*isPV=*/true});
            cumulative_P += m_P;
            continue;
        }

        if (v2::TTEntry* ce =
                ctx->arena->find(cb.hash(), v2::secondary_hash(cb))) {
            // TT hit: Expanded, real Q + real depth from the child's entry.
            // `find` only returns non-null after `secondary_matches`
            // observed Cell B publish with matching key_secondary, so
            // qd is consistent (Cell A is atomic with the key match).
            auto [q, child_max_d] = v2::unpack_qd(
                v2::SearchArena::load_qd(ce).qd_packed);
            q = clamp_q_upcoming_rep(q, cb);
            plans.push_back({Mode::Expanded, m_P, q, child_max_d,
                             /*alloc=*/0.0f});
            cumulative_P += m_P;
            continue;
        }

        // TT miss: Unexpanded. FPU (Leela-style):
        //   Q_eff_parent_pov = parent_Q - fpu_reduction * sqrt(cumulative_P)
        // Stored in child-STM convention as -Q_eff_parent_pov so the
        // rollup's -Q yields back Q_eff_parent_pov.
        const float Q_eff_parent_pov =
            parent_Q - fpu_reduction * std::sqrt(cumulative_P);
        plans.push_back({Mode::Unexpanded, m_P, /*Q=*/-Q_eff_parent_pov,
                         /*depth=*/kNegInf, /*alloc=*/0.0f});
        cumulative_P += m_P;
    }

    // ── Pass 2: clamped allocate+fork loop ──────────────────────────
    // Halley allocations can swing wildly as FPU-optimistic siblings
    // resolve. To avoid burning depth on a child whose Q will collapse
    // on first contact, we iterate:
    //
    //   1. compute_log_allocations on the current (Q, depth) state.
    //   2. If iter > 0 and every Expanded plan satisfies
    //      alloc - depth <= break_eps, break (in pv_mode the break is
    //      additionally gated on PV convergence — see below).
    //   3. Expanded plans fork when alloc > p.depth at
    //      min(alloc, p.depth + clamp_step * jitter), where jitter is
    //      drawn uniformly from [0.5, 1.5] per fork via
    //      clamp_step_jitter() (Lazy-SMP variation). Unexpanded plans never
    //      use the alloc gate; on iter 0 only, force-expand the first
    //      `hdr->force_expand` Unexpanded priors (per-position dynamic
    //      count, computed at first-eval from a temp-1.0 policy), or
    //      all Unexpanded when `depth > depth_floor +
    //      log(force_all_unexpanded_log_arg)`, when in pv_mode, or when
    //      any of the first `hdr->force_expand` moves is a path-dependent
    //      draw (`top_move_pathdraw`), at recursion depth p.alloc.
    //      Children write (Q, depth) back through `&p`.
    //
    // PV-mode overlay (only when this coroutine runs with pv_mode=true):
    //   - On every iter > 0, identify pv_idx = argmax_i(-plans[i].Q)
    //     over Expanded plans (the parent-POV best child) and
    //     unconditionally force-fork that child under
    //     pv_mode_child=true, even if its alloc gate (`alloc >
    //     p.depth`) doesn't fire. The re-deepen waiver in the child's
    //     recursive_search lets the recursion proceed even when
    //     target_depth doesn't strictly grow depth.
    //   - All non-PV-forced forks use pv_mode_child=false. Children
    //     of non-PV parents always run with pv_mode=false.
    //   - At each fork dispatch the parent eagerly assigns
    //     `p.isPV = pv_mode_child`, so the row's PV claim tracks the
    //     mode of the most recent rollup that overwrote it. Plans
    //     not re-forked this iter retain their previous isPV value.
    //   - The break-after-break_eps gate is augmented: we additionally
    //     require some Expanded isPV=true plan to be within kPvQEps
    //     (parent-POV) of the current best score. This prevents an
    //     early break before the PV move has been refined under
    //     pv_mode_child=true at all.
    //
    // Permits: only iter 0's FIRST fork inherits this coroutine's
    // entry permit (so K bounds permit-holding frames, not alive
    // frames — keeping descent depth independent of K). Every other
    // fork (later iter-0 siblings and every fork in iter 1+) does
    // co_await sem->acquire(). After iter 0's lf::join this coroutine
    // is permit-less for the rest of the call, same state the
    // original single-shot code was in during rollup.
    //
    // Pre-marking: a forked child OVERWRITES p.Q / p.depth (via the
    // `&p` out-param) before this coroutine resumes from lf::join, so
    // we flip the row to Expanded immediately before forking. Abort
    // paths leave the row stale but the post-iter should_abort()
    // check below skips later rollup before any stale row is observed.
    {
        const float clamp_step  = ctx->params->clamp_step;
        const float break_eps   = ctx->params->break_eps;
        const float force_all_log_arg =
            ctx->params->force_all_unexpanded_log_arg;
        const int   max_iters   = ctx->params->clamp_max_iters;
        const int   force_count = static_cast<int>(hdr->force_expand);
        assert(force_count >= 1 &&
               "force_expand should be >= 1 at first-eval; a count < 1 leaves iter-0 "
               "force-expand dependent solely on the force_all_unexpanded path");
        // PV convergence tolerance: loop in pv_mode only breaks once
        // some isPV=true plan's parent-POV score is within this margin
        // of the best Expanded child's score. Magnitude matches a
        // typical Q-noise floor; tight enough that the break_eps gate
        // remains the primary terminator on most positions.
        constexpr float kPvQEps = 0.04f;

        int iter = 0;
        for (; iter < max_iters; ++iter) {
            compute_log_allocations(plans.data(),
                                    static_cast<int>(plans.size()),
                                    depth, ctx->params->c_puct);

            // PV pick (computed once per iter, used by both the break
            // gate and the per-plan force-fork). Only meaningful at
            // iter > 0 in pv_mode; otherwise stays -1.
            int pv_idx = -1;
            if (pv_mode && iter > 0) {
                float best_neg_q =
                    -std::numeric_limits<float>::infinity();
                for (int i = 0; i < static_cast<int>(num_moves); ++i) {
                    const Plan& p = plans[i];
                    if (p.mode != Mode::Expanded) continue;
                    const float neg_q = -p.Q;
                    if (neg_q > best_neg_q) {
                        best_neg_q = neg_q;
                        pv_idx = i;
                    }
                }
            }

            if (iter > 0) {
                bool any_unsaturated = false;
                for (const Plan& p : plans) {
                    if (p.mode != Mode::Expanded) continue;
                    if (p.alloc - p.depth > break_eps) {
                        any_unsaturated = true;
                        break;
                    }
                }
                bool pv_break_ok = !pv_mode;
                if (pv_mode) {
                    if (pv_idx < 0) {
                        // No Expanded plans at all; PV concept is
                        // vacuous, let the regular gate proceed.
                        pv_break_ok = true;
                    } else {
                        const float best_neg_q = -plans[pv_idx].Q;
                        for (const Plan& p : plans) {
                            if (p.mode != Mode::Expanded || !p.isPV)
                                continue;
                            if (std::abs(-p.Q - best_neg_q) <= kPvQEps) {
                                pv_break_ok = true;
                                break;
                            }
                        }
                    }
                }
                if (!any_unsaturated && pv_break_ok) break;
            }

            const bool force_all_unexpanded =
                (iter == 0)
                && ((depth > depth_floor + std::log(force_all_log_arg))
                    || pv_mode
                    || top_move_pathdraw);

            bool first_fork = (iter == 0);
            bool any_fork   = false;
            for (uint16_t i = 0; i < num_moves; ++i) {
                Plan& p = plans[i];

                bool is_pv_force = false;
                if (p.mode == Mode::Unexpanded) {
                    if (iter != 0) continue;
                    const bool force_expand =
                        (static_cast<int>(i) < force_count)
                        || force_all_unexpanded;
                    if (!force_expand) continue;
                } else {
                    // Expanded: normal alloc gate, with a PV-mode
                    // override on the current best child. The override
                    // fires every iter > 0 in pv_mode (no `!p.isPV`
                    // gate) so we keep refining the PV move every
                    // iteration; idempotent because target_depth and
                    // depth_floor stabilize and the rollup writeback
                    // is monotonic in (Q, depth) on the PV chain.
                    //
                    // The `std::isfinite(p.depth)` guard excludes
                    // terminal / path-dep-draw plans (depth=+inf):
                    // recursive_search's caller-skip contract forbids
                    // descending into a terminal child, and there's
                    // nothing to refine anyway (Q is exact). Such
                    // rows were pre-marked isPV=true at Pass 1, so
                    // the break gate is still satisfiable when the
                    // best move is terminal.
                    is_pv_force =
                        pv_mode && iter > 0
                        && static_cast<int>(i) == pv_idx
                        && std::isfinite(p.depth);
                    if (!(p.alloc > p.depth) && !is_pv_force) continue;
                }

                const bool pv_mode_child = is_pv_force;

                const float target_depth =
                    (p.mode == Mode::Unexpanded)
                        ? p.alloc
                        : std::min(p.alloc,
                                   p.depth + clamp_step * clamp_step_jitter());

                chess::Board cb = board;
                cb.makeMove<true>(chess::Move{moves[i].move});

                lfsync::Permit child_permit = first_fork
                    ? std::move(permit)
                    : co_await ctx->sem->acquire();
                if (ctx->w->should_abort()) {
                    break;
                }
                first_fork = false;
                any_fork   = true;

                p.mode = Mode::Expanded;
                // Eagerly mark this plan's PV state from the child's
                // dispatch mode: PV-forced child ⇒ isPV=true, normal
                // re-fork ⇒ isPV=false. This naturally invalidates
                // any prior PV claim on a plan that's now being
                // re-evaluated under a non-PV descent, with no
                // back-propagation needed from the child.
                p.isPV = pv_mode_child;
                co_await lf::fork[recursive_search](
                    ctx, std::move(child_permit), std::move(cb),
                    target_depth, rec_depth + 1, pv_mode_child, &p);
            }

            if (any_fork) {
                co_await lf::join;
            } else if (iter > 0) {
                // Nothing past its depth gate this iter — iter 1+ where
                // every plan that could grow is within break_eps and
                // (in pv_mode) PV convergence is already satisfied.
                // Either way, no later iter would change anything,
                // so fall through to rollup. Without this guard the
                // loop would spin doing no work.
                break;
            }

            if (ctx->w->should_abort()) co_return;
        }
        assert(iter < max_iters &&
               "clamp loop hit clamp_max_iters without converging; "
               "falling through to rollup on an unconverged allocation state");
    }

    // ── Rollup: negamax over Expanded plans ─────────────────────────
    // Parent's Q is the best move's value in parent-POV:
    //   Q_new = max over Expanded plans of (-p.Q)
    // (children store Q in child-STM convention; negation flips to
    // parent POV.) Unexpanded children that never cleared the gate
    // are dropped entirely.
    float best = -std::numeric_limits<float>::infinity();
    int  best_idx = -1;
    bool any_expanded = false;
    // Track the best child that was actually descended under pv_mode_child
    // (isPV=true) separately from the global best. These usually coincide,
    // but the PV-convergence gate only guarantees an isPV plan within
    // kPvQEps of the best — so the global argmax can be a sibling that was
    // never refined on the PV. We return Q from the global best but persist
    // pv_child from the best *isPV* plan, so the display PV walk only ever
    // follows children that were genuinely explored as PV.
    float best_pv = -std::numeric_limits<float>::infinity();
    int   best_pv_idx = -1;
    for (int i = 0; i < static_cast<int>(plans.size()); ++i) {
        const Plan& p = plans[i];
        if (p.mode != Mode::Expanded) continue;
        any_expanded = true;
        const float v = -p.Q;
        if (v > best) { best = v; best_idx = i; }
        if (p.isPV && v > best_pv) { best_pv = v; best_pv_idx = i; }
    }
    assert(any_expanded &&
           "rollup found no Expanded plan; falling back to TT (Q, depth) for parent");
    if (any_expanded) {
        const float Q_new = best;
        // Persist the child index for the PV / bestmove display walk.
        // Prefer the best isPV plan (the move actually descended on the PV);
        // fall back to the global best_idx when no isPV plan exists (e.g. a
        // non-pv_mode rollup). plans[i] is built 1:1 with moves[i] in Pass 1,
        // so the index is a valid MoveInfo index. Relaxed last-writer-wins
        // store (display only); mirrors the `expanded` flag's
        // weak-consistency model.
        const int pv_store_idx = (best_pv_idx >= 0) ? best_pv_idx : best_idx;
        std::atomic_ref<uint16_t>(hdr->pv_child)
            .store(static_cast<uint16_t>(pv_store_idx), std::memory_order_relaxed);
        if (pv_mode) {
            // PV-mode rollup: the re-deepen waiver lets us descend on
            // an entry that already has a deeper TT (Q, max_depth)
            // snapshot, so the depth-monotonic gate in `update_qd`
            // would silently drop our rollup. Use the unconditional
            // writer instead, and floor the written depth at
            // `depth_floor` so a PV-mode visit at very low depth
            // doesn't ratchet TT depth below the NN's
            // confidence-derived floor.
            const float d_write = std::max(depth, depth_floor);
            v2::SearchArena::store_qd_force(entry, Q_new, d_write);
            if (out) { out->Q = Q_new; out->depth = d_write; }
        } else {
            v2::SearchArena::update_qd(entry, Q_new, depth);
            // Hand the rolled-up (Q, depth) back to the parent's plan
            // row. If update_qd above lost the CAS to a peer with a
            // deeper entry, the parent still gets our local rollup —
            // a fresh, self-consistent estimate from the same
            // children we just descended into.
            if (out) { out->Q = Q_new; out->depth = depth; }
        }
    } else if (out) {
        // No Expanded child contributed (everything was Unexpanded
        // and gated out of fan-out). Fall back to whatever TT holds
        // for `entry` — typically the NN-Q from our own
        // find_or_claim publish — so the parent's rollup doesn't
        // inherit the FPU stand-in we pre-marked above.
        auto [q_tt, d_tt] = v2::unpack_qd(
            v2::SearchArena::load_qd(entry).qd_packed);
        out->Q     = q_tt;
        out->depth = d_tt;
    }
};

/**
 * Thin root entry: acquires the very first permit from the per-worker
 * semaphore, then descends into recursive_search. Keeps the
 * recursive_search invariant ("entry owns a permit") clean.
 */
inline constexpr auto root_search =
    [](auto /*self*/,
       RecurseContext* ctx,
       chess::Board board,
       float depth) -> lf::task<void>
{
    lfsync::Permit p = co_await ctx->sem->acquire();
    co_await lf::call[recursive_search](
        ctx, std::move(p), std::move(board), depth,
        /*rec_depth=*/0u, /*pv_mode=*/true, /*out=*/nullptr);
    co_await lf::join;
};

/**
 * One-shot task whose only job is to pin the lazy_pool's (single) worker
 * thread to a physical core. Each `WorkerSearch` owns an `lf::lazy_pool(1)`
 * whose worker thread is created internally by libfork, so we cannot set
 * its affinity from the outside; running this task via `lf::sync_wait`
 * executes it *on* that worker thread, letting it pin itself. NUMA pinning
 * is a best-effort performance hint (failures are swallowed in
 * `pin_this_thread_to_cpu`), so this never affects correctness.
 */
inline constexpr auto pin_pool_worker =
    [](auto /*self*/, int cpu) -> lf::task<void> {
    catgpt::numa::pin_this_thread_to_cpu(cpu);
    co_return;
};

}  // namespace detail

class LksSearch {
public:
    using WorkerSearch = detail::WorkerSearch;

    /**
     * @param trt_engine_path     Path to the serialized TensorRT engine.
     *                            Loaded once per worker_search at ctor time.
     * @param lifetime_max_evals  Capacity for the shared SearchArena (TT
     *                            entries reachable across all workers'
     *                            lifetime). Sized once; never grows.
     * @param workers_per_gpu     Number of WorkerSearch instances PER
     *                            CUDA device. Total worker count is
     *                            `workers_per_gpu * cudaGetDeviceCount()`.
     *                            Each worker owns its own engine + GPU
     *                            thread + stream for natural pipelining.
     *                            Block-partitioned: workers
     *                            `[i*workers_per_gpu, (i+1)*workers_per_gpu)`
     *                            are pinned to GPU `i`.
     * @param coros_per_worker    Tuning knob for the per-worker semaphore:
     *                            each worker permits up to
     *                            `coros_per_worker` simultaneously-alive
     *                            recursive_search invocations. Bounds
     *                            both orchestration frames and GPU evals.
     * @param max_batch_size      Cap on positions per GPU batch.
     * @param syzygy_path         Filesystem path to a directory containing
     *                            Syzygy .rtbw/.rtbz endgame tablebase files.
     *                            Empty (the default) disables Syzygy entirely;
     *                            non-empty constructs a single per-LksSearch
     *                            `SyzygyProber`. Fathom holds global state,
     *                            so at most one `LksSearch` per process
     *                            should be constructed with a non-empty path.
     */
    explicit LksSearch(fs::path trt_engine_path,
                       uint64_t lifetime_max_evals = (1ULL << 20),
                       int workers_per_gpu = 1,
                       int coros_per_worker = 32,
                       int max_batch_size = 32,
                       fs::path syzygy_path = {})
        : trt_engine_path_(std::move(trt_engine_path))
        , lifetime_max_evals_(lifetime_max_evals)
        , workers_per_gpu_(workers_per_gpu > 0 ? workers_per_gpu : 1)
        , coros_per_worker_(coros_per_worker > 0 ? coros_per_worker : 1)
        , max_batch_size_(max_batch_size > 0 ? max_batch_size : 1)
        , board_(chess::constants::STARTPOS)
    {
        if (!syzygy_path.empty()) {
            // SyzygyProber ctor calls Fathom's tb_init (global state).
            // If load fails, prober.is_available() will be false and the
            // shortcut path silently disables itself.
            syzygy_ = std::make_unique<catgpt::SyzygyProber>(
                syzygy_path.string());
        }

        // Discover GPU topology. CUDA_VISIBLE_DEVICES is the right knob
        // to restrict which devices participate; this just uses every
        // visible one.
        int num_gpus = 0;
        CATGPT_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
        if (num_gpus <= 0) {
            throw std::runtime_error(
                "LksSearch: no CUDA devices visible (cudaGetDeviceCount==0)");
        }
        num_gpus_    = num_gpus;
        num_workers_ = workers_per_gpu_ * num_gpus_;

        std::println(stderr,
                     "[LksSearch] {} device(s) x {} workers_per_gpu = {} total workers",
                     num_gpus_, workers_per_gpu_, num_workers_);

        // ── NUMA topology + per-worker core assignment ──────────────────
        // Discover the host's NUMA layout once and decide, per worker, which
        // physical core runs its libfork descent (compute_cpu) and which SMT
        // sibling runs its GPU thread (gpu_cpu), on the NUMA node local to
        // the worker's GPU. A per-node round-robin hands out distinct cores;
        // assignment degrades to -1 (no pinning) when /sys topology is
        // unavailable (e.g. a local dev box). Computed serially here so the
        // parallel init threads below just consume their slot.
        topo_ = numa::host_topology();
        {
            std::vector<std::size_t> node_rr(
                static_cast<std::size_t>(std::max(1, topo_.num_nodes())), 0);
            w_node_.assign(num_workers_, topo_.num_nodes() > 0 ? topo_.node_ids[0] : 0);
            w_compute_cpu_.assign(num_workers_, -1);
            w_gpu_cpu_.assign(num_workers_, -1);
            for (int i = 0; i < num_workers_; ++i) {
                const int device_id = i / workers_per_gpu_;
                const int os_node   = gpu_numa_node(device_id, topo_);
                const int nidx      = topo_.index_of_node(os_node);
                if (nidx < 0) continue;
                w_node_[i] = os_node;
                const auto& phys = topo_.node_phys_cpus[nidx];
                if (phys.empty()) continue;
                const int core = phys[node_rr[nidx] % phys.size()];
                ++node_rr[nidx];
                w_compute_cpu_[i] = core;
                w_gpu_cpu_[i] =
                    (core >= 0 && core < static_cast<int>(topo_.sibling_of.size()))
                        ? topo_.sibling_of[core]
                        : core;
            }
        }
        std::println(stderr,
                     "[LksSearch] NUMA: {} node(s){}",
                     topo_.num_nodes(),
                     topo_.num_nodes() <= 1 ? " (pinning disabled)" : "");
        for (int i = 0; i < num_workers_; ++i) {
            std::println(stderr,
                         "[LksSearch] worker {} -> gpu {} numa {} compute_cpu {} gpu_cpu {}",
                         i, i / workers_per_gpu_, w_node_[i],
                         w_compute_cpu_[i], w_gpu_cpu_[i]);
        }

        root_key_ = board_.hash();

        // Persistent workers: build pool + evaluator + sem per worker once.
        // The slow part is BatchEvaluator construction (TRT engine load,
        // CUDA buffer alloc, graph capture); fan it out across one
        // jthread per worker so total init time is dominated by the
        // slowest single worker rather than the sum. Each thread pins
        // itself to its assigned device — `cudaSetDevice` is per-thread
        // state, so siblings on different devices don't interfere, and
        // the BatchEvaluator ctor's CUDA calls all land on the right GPU.
        const int eval_permits = coros_per_worker_;
        workers_.resize(num_workers_);
        std::vector<std::exception_ptr> errs(num_workers_);
        {
            std::vector<std::jthread> init_threads;
            init_threads.reserve(num_workers_);
            for (int i = 0; i < num_workers_; ++i) {
                init_threads.emplace_back([this, i, eval_permits, &errs] {
                    try {
                        const int device_id = i / workers_per_gpu_;
                        CATGPT_CUDA_CHECK(cudaSetDevice(device_id));
                        // Pin this init thread to the GPU-local NUMA node
                        // BEFORE constructing the BatchEvaluator: its ctor's
                        // cudaMallocHost pinned transfer buffers first-touch
                        // on the calling thread's node, so this keeps them on
                        // the node closest to the GPU.
                        if (w_compute_cpu_[i] >= 0) {
                            numa::pin_this_thread_to_node(topo_, w_node_[i]);
                        }
                        auto w = std::make_unique<WorkerSearch>();
                        w->device_id   = device_id;
                        w->numa_node   = w_node_[i];
                        w->compute_cpu = w_compute_cpu_[i];
                        w->gpu_cpu     = w_gpu_cpu_[i];
                        // 1 worker per pool: per-WorkerSearch isolation
                        // is the model. lazy_pool means the worker thread
                        // sleeps when no work is ready, woken on
                        // pool->schedule(handle) by the GPU thread or
                        // by a release()'d semaphore waiter.
                        w->pool = std::make_unique<lf::lazy_pool>(/*n=*/1);
                        // Pin the pool's internal worker thread (the descent
                        // thread) to its physical core. libfork creates that
                        // thread internally, so we run a one-shot task on it
                        // via sync_wait to let it set its own affinity.
                        if (w->compute_cpu >= 0) {
                            lf::sync_wait(*w->pool, detail::pin_pool_worker,
                                          w->compute_cpu);
                        }
                        w->evaluator = std::make_unique<BatchEvaluator>(
                            trt_engine_path_, w->pool.get(),
                            max_batch_size_, device_id, w->gpu_cpu);
                        w->sem = std::make_unique<lfsync::LfAsyncSemaphore>(
                            eval_permits, *w->pool);
                        workers_[i] = std::move(w);
                    } catch (...) {
                        errs[i] = std::current_exception();
                    }
                });
            }
        }  // jthreads join here
        for (auto& e : errs) {
            if (e) std::rethrow_exception(e);
        }

        // Allocate the shared SearchArena LAST, after the evaluators are up.
        // Each BatchEvaluator ctor transiently loads the serialized network
        // (~GB per worker) into host RAM, uploads it to the GPU, then frees
        // the host blob. Allocating the (potentially hundreds-of-GiB) arena
        // first would prefault all available RAM and leave no headroom for
        // those transient weight blobs -> OOM during engine load. Building
        // evaluators first lets the arena mmap/prefault into the RAM that is
        // reclaimed once the weights are resident on the GPU.
        {
            const auto t0 = std::chrono::steady_clock::now();
            arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.6,
                           /*avg_moves_per_node=*/40, &topo_);
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            std::println(stderr,
                         "[LksSearch] arena allocated (interleaved, parallel prefault): TT {} B + node arena {} B in {} ms",
                         arena_->table_bytes(), arena_->arena_capacity_bytes(), ms);
        }
    }

    ~LksSearch() {
        // Make sure no worker_main outlives us.
        quit();
        // workers_ vector destructs here:
        //   - each WorkerSearch's `runner` jthread is non-joinable
        //     (worker_main joined it before exiting).
        //   - each WorkerSearch's `sem` drops first (no waiters; runners
        //     joined).
        //   - each WorkerSearch's `evaluator` drops next →
        //     BatchEvaluator dtor calls shutdown() → GPU thread joined
        //     → CUDA resources freed. After this no more pool->schedule.
        //   - each WorkerSearch's `pool` drops last → lazy_pool dtor
        //     joins its worker thread.
    }

    LksSearch(const LksSearch&) = delete;
    LksSearch& operator=(const LksSearch&) = delete;
    LksSearch(LksSearch&&) = delete;
    LksSearch& operator=(LksSearch&&) = delete;

    // ── Synchronous lifecycle (must not be called while a search runs) ──

    void reset() {
        assert(!is_searching() && "reset() called while a search is in flight");
        // Rebuild arena in place: munmap the old buffers, mmap + parallel
        // NUMA-interleaved prefault on fresh ones (see SearchArena). Workers
        // and evaluators are NOT rebuilt. The parallel init keeps the
        // per-`ucinewgame` reallocation off the critical path for huge TTs.
        const auto t0 = std::chrono::steady_clock::now();
        arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.6,
                       /*avg_moves_per_node=*/40, &topo_);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        std::println(stderr, "[LksSearch] arena reallocated in {} ms", ms);
        searched_since_reset_ = false;
    }

    void setBoard(const chess::Board& board) {
        assert(!is_searching() && "setBoard() called while a search is in flight");
        if (searched_since_reset_) {
            reset();
        }
        board_ = board;
        root_key_ = board_.hash();
    }

    void makemove(const chess::Move& move) {
        assert(!is_searching() && "makemove() called while a search is in flight");
        // Tree reuse: arena and TT preserved.
        board_.makeMove<true>(move);
        root_key_ = board_.hash();
    }

    // ── Asynchronous search ─────────────────────────────────────────────

    /**
     * Launches the worker_main jthread (which spawns N runner jthreads)
     * and returns immediately. Throws std::logic_error if a search is
     * already in flight.
     */
    void search(LksSearchConfig config) {
        if (is_searching()) {
            throw std::logic_error("LksSearch::search called while a search is already in flight");
        }
        // Resolve the effective WDL temp from the root side to move: a
        // White-to-move root sharpens harder (wl_temp_white) than a
        // Black-to-move root (wl_temp_black). Applied once per search so a
        // single wl_temp is used uniformly across all leaf evals.
        config.params.wl_temp =
            (board_.sideToMove() == chess::Color::WHITE)
                ? config.params.wl_temp_white
                : config.params.wl_temp_black;
        // Reap a previous worker_main that exited but is still joinable.
        if (worker_.joinable()) {
            worker_.join();
        }
        searched_since_reset_ = true;
        running_.store(true, std::memory_order_release);
        worker_ = std::jthread(
            [this, cfg = std::move(config)](std::stop_token st) mutable {
                worker_main(std::move(st), std::move(cfg));
            });
    }

    /**
     * Stops any in-flight search and joins worker_main (which joins all
     * runners). Idempotent. Safe to call from a thread other than
     * `search()`. Does NOT shutdown the persistent BatchEvaluators.
     */
    void quit() {
        if (!worker_.joinable()) return;
        worker_.request_stop();
        worker_.join();
    }

    [[nodiscard]] bool is_searching() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    /** Total worker count (== workers_per_gpu * cudaGetDeviceCount()). */
    [[nodiscard]] int num_workers() const noexcept { return num_workers_; }
    [[nodiscard]] int workers_per_gpu() const noexcept { return workers_per_gpu_; }
    [[nodiscard]] int num_gpus() const noexcept { return num_gpus_; }

    [[nodiscard]] const chess::Board& board() const noexcept { return board_; }
    [[nodiscard]] uint64_t root_key() const noexcept { return root_key_; }
    [[nodiscard]] const v2::SearchArena& arena() const noexcept { return *arena_; }

    /**
     * How the most recent `worker_main` run terminated, for cross-move time
     * management in the UCI driver:
     *   - kSoftCompleted: the soft deadline expired and the slowest worker
     *     then finished one more ID iteration (the "non-abort" path).
     *   - kHardAborted:   a game-clock search force-stopped at the hard
     *     deadline before the soft grace completed (the "abort" path).
     *   - kOther:         any other exit (stop, max-depth, target depth,
     *     eval budget, root shortcut, or a `movetime` search).
     * Only meaningful after the search has finished (read post-quit()/join).
     */
    enum class Completion { kOther, kSoftCompleted, kHardAborted };
    [[nodiscard]] Completion last_completion() const noexcept {
        return last_completion_;
    }

    /**
     * Root TT entry's max_depth (log-scale), or nullopt when the current
     * root has no published TT entry. Used by the early-return chain to
     * compare a reused subtree's depth against the anchored root depth.
     */
    [[nodiscard]] std::optional<float> root_depth() const {
        const v2::TTEntry* root =
            arena_->find(root_key_, v2::secondary_hash(board_));
        if (root == nullptr) return std::nullopt;
        auto [q, d] = v2::unpack_qd(
            v2::SearchArena::load_qd(root).qd_packed);
        (void)q;
        return d;
    }

    /**
     * Root TT entry's score in centipawns (side-to-move POV), or nullopt
     * when the current root has no published TT entry. Mirrors the score
     * field emitted by `worker_main`'s `info` lines.
     */
    [[nodiscard]] std::optional<int> root_cp() const {
        const v2::TTEntry* root =
            arena_->find(root_key_, v2::secondary_hash(board_));
        if (root == nullptr) return std::nullopt;
        auto [q, d] = v2::unpack_qd(
            v2::SearchArena::load_qd(root).qd_packed);
        (void)d;
        return q_to_cp(q);
    }

    /**
     * Whether a SyzygyProber was constructed AND successfully loaded at
     * least one tablebase file. False when no path was provided, when
     * the path was empty of .rtbw/.rtbz files, or when tb_init failed.
     */
    [[nodiscard]] bool syzygy_available() const noexcept {
        return syzygy_ && syzygy_->is_available();
    }

    /**
     * Aggregated count of NN evals performed across all workers in the
     * most recent (or in-flight) search. Cleared at the start of each
     * `search()` call.
     */
    [[nodiscard]] uint64_t total_evals() const noexcept {
        return total_evals_.load(std::memory_order_relaxed);
    }

    /**
     * `min(depth)` across workers — i.e. the depth fully completed by
     * every worker. Reset to `cfg.start_depth` at the start of each
     * `search()` and monotonically advanced.
     */
    [[nodiscard]] float min_depth() const noexcept {
        float m = std::numeric_limits<float>::infinity();
        for (const auto& w : workers_) {
            m = std::min(m, w->depth.load(std::memory_order_relaxed));
        }
        return std::isfinite(m) ? m : 0.0f;
    }

    /**
     * `max(depth)` across workers — the deepest iteration any worker has
     * completed. Used by tests to verify the per-worker depths are within
     * one `delta_depth` of each other.
     */
    [[nodiscard]] float max_depth() const noexcept {
        float m = -std::numeric_limits<float>::infinity();
        for (const auto& w : workers_) {
            m = std::max(m, w->depth.load(std::memory_order_relaxed));
        }
        return std::isfinite(m) ? m : 0.0f;
    }

    /**
     * Sum of `BatchEvaluator::total_evals()` across persistent workers,
     * across the lifetime of LksSearch (not reset between searches).
     * Useful for verifying that the GPU actually ran inference.
     */
    [[nodiscard]] uint64_t lifetime_gpu_evals() const noexcept {
        uint64_t total = 0;
        for (const auto& w : workers_) {
            total += static_cast<uint64_t>(w->evaluator->total_evals());
        }
        return total;
    }

    /**
     * Sum of `BatchEvaluator::total_batches()` across persistent workers.
     * Combined with `lifetime_gpu_evals()` this gives the average batch
     * size achieved on the GPU.
     */
    [[nodiscard]] uint64_t lifetime_gpu_batches() const noexcept {
        uint64_t total = 0;
        for (const auto& w : workers_) {
            total += static_cast<uint64_t>(w->evaluator->total_batches());
        }
        return total;
    }

    /**
     * Best root move, computed on the fly as the negamax argmax over the
     * root's direct (depth-1) children: score = -child_Q from our POV, with
     * tiebreaks (child max_depth, then prior P). Each child's Q mirrors the
     * Pass-1 classification — terminal_kind drives a fixed Q;
     * `isRepetition(1)` / `isHalfMoveDraw()` are checked path-dependently on
     * `board_`'s own history and scored as draws; everything else reads the
     * child's TT Q.
     *
     * This deliberately does NOT use the persisted `pv_child`: an abort can
     * leave the root's `pv_child` stale or unset (e.g. a rollup that lost the
     * race, or no rollup at all), whereas recomputing the one-ply negamax
     * here always reflects the latest child TT state at decision time. The PV
     * *display* (`principal_variation`) still walks `pv_child`.
     *
     * Children never reached during search are scored as "we lose" so any
     * TT-evaluated sibling outranks them; if NO child has a TT entry, the
     * highest-P move wins via the P tiebreak. Falls back to `legal[0]` when
     * the root itself was never expanded (UCI still needs a move), and to
     * `chess::Move::NO_MOVE` only when there are no legal moves.
     *
     * Safe to call concurrently with a search in flight (acquire TT reads);
     * intended to be called after `quit()`/runners join.
     */
    [[nodiscard]] chess::Move bestmove() const {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        if (legal.empty()) return chess::Move::NO_MOVE;

        const v2::TTEntry* root =
            arena_->find(board_.hash(), v2::secondary_hash(board_));
        if (root == nullptr) return legal[0];

        // A non-null `find` guarantees Cell B is published with a matching
        // secondary, so a plain `load_info` is sufficient (no spin).
        const v2::InfoCell info = v2::SearchArena::load_info(root);
        assert(info.info_offset != v2::kNoInfoOffset
               && "find returned a non-null entry whose Cell B is "
                  "unpublished; SearchArena invariant broken");

        const v2::NodeInfoHeader* hdr = arena_->info_at(info.info_offset);
        const uint16_t num_moves = hdr->num_moves;
        const v2::MoveInfo* moves = arena_->moves_at(info.info_offset);
        if (num_moves == 0) return legal[0];

        chess::Move best       = chess::Move{moves[0].move};
        float       best_score = -std::numeric_limits<float>::infinity();
        float       best_depth = -std::numeric_limits<float>::infinity();
        float       best_P     = -1.0f;

        for (uint16_t i = 0; i < num_moves; ++i) {
            const v2::MoveInfo& mi = moves[i];
            const chess::Move m{mi.move};
            const v2::TerminalKind tk = mi.terminal_kind();
            const float mi_P = mi.P();

            float child_Q;
            float child_depth = std::numeric_limits<float>::infinity();

            if (tk == v2::kTerminalDraw) {
                child_Q = 0.0f;
            } else if (tk == v2::kTerminalLossForChild) {
                child_Q = -1.0f;  // child loses ⇒ we win
            } else if (tk == v2::kTerminalWinForChild) {
                // Syzygy: child is a TB win for STM-at-child ⇒ we lose.
                child_Q = +1.0f;
            } else {
                chess::Board cb = board_;
                cb.makeMove<true>(m);
                // Plain 2-fold repetition draw on `board_`'s own history.
                if (cb.isRepetition(1) || cb.isHalfMoveDraw()) {
                    child_Q = 0.0f;
                } else {
                    const v2::TTEntry* ce =
                        arena_->find(cb.hash(), v2::secondary_hash(cb));
                    if (ce == nullptr) {
                        // Never reached during search (or a 64-bit collision
                        // masking the entry): score as "we lose" so any
                        // TT-evaluated sibling outranks it. The P tiebreak
                        // still picks the highest-prior move when no child
                        // has a TT entry.
                        child_Q     = +1.0f;
                        child_depth = -std::numeric_limits<float>::infinity();
                    } else {
                        auto [q, d] = v2::unpack_qd(
                            v2::SearchArena::load_qd(ce).qd_packed);
                        child_Q     = detail::clamp_q_upcoming_rep(q, cb);
                        child_depth = d;
                    }
                }
            }

            const float score = -child_Q;
            const bool better =
                   score > best_score
                || (score == best_score && child_depth > best_depth)
                || (score == best_score && child_depth == best_depth
                    && mi_P > best_P);
            if (better) {
                best_score = score;
                best_depth = child_depth;
                best_P     = mi_P;
                best       = m;
            }
        }
        return best;
    }

    /**
     * Walk the persisted `pv_child` chain from the root to produce the
     * principal variation. At each ply we play the node's recorded best
     * child (`pv_move_at`) and descend into the resulting position.
     *
     * The line ends — without emitting a move from the current position —
     * when `pv_move_at` returns `NO_MOVE`, i.e. the position has no TT
     * entry, no children, or no rollup recorded a best child (`pv_child ==
     * kNoPvChild`). Because `pv_child` is only set by a real rollup, this
     * naturally stops at the deepest node the search actually evaluated —
     * no separate "is the top child expanded?" heuristic needed.
     *
     * The line additionally ends — AFTER emitting the latest move — when
     * the resulting position draws by a plain 2-fold repetition
     * (`isRepetition(1)`), 50-move draw (`isHalfMoveDraw()`),
     * insufficient material, or after
     * `max_len` plies (defensive cap). These run on the board after
     * `makeMove<true>(m)`, so the board's own path history disambiguates
     * sequence-dependent repetitions correctly.
     *
     * Safe to call concurrently with a search in flight; intended for use
     * after `quit()`/runners join, when the TT is stable.
     */
    [[nodiscard]] std::vector<chess::Move>
    principal_variation(int max_len = 256) const {
        std::vector<chess::Move> pv;
        if (max_len <= 0) return pv;
        pv.reserve(static_cast<size_t>(max_len));

        chess::Board b = board_;
        for (int i = 0; i < max_len; ++i) {
            const chess::Move m = pv_move_at(b);
            if (m == chess::Move::NO_MOVE) break;
            pv.push_back(m);
            b.makeMove<true>(m);
            // A plain 2-fold repetition (or 50-move / insufficient material)
            // ends the line. Runs on `b`'s own path history.
            if (b.isRepetition(1) || b.isHalfMoveDraw()
                || b.isInsufficientMaterial()) {
                break;
            }
        }
        return pv;
    }

private:
    /**
     * Return the move recorded as `b`'s best child by the search's rollup
     * (`NodeInfoHeader::pv_child`), or `chess::Move::NO_MOVE` when there is
     * nothing recorded to follow: no TT entry for `b`, unpublished info, no
     * children, or `pv_child == kNoPvChild` (no rollup yet). The index is a
     * direct offset into the node's P-sorted MoveInfo array.
     *
     * Relaxed load of `pv_child` — display-only; a value from a slightly
     * older rollup is fine. A non-null `find` already guarantees Cell B is
     * published with a matching secondary, so a plain `load_info` suffices.
     */
    [[nodiscard]] chess::Move pv_move_at(const chess::Board& b) const {
        const v2::TTEntry* e =
            arena_->find(b.hash(), v2::secondary_hash(b));
        if (e == nullptr) return chess::Move::NO_MOVE;

        const v2::InfoCell info = v2::SearchArena::load_info(e);
        assert(info.info_offset != v2::kNoInfoOffset
               && "find returned a non-null entry whose Cell B is "
                  "unpublished; SearchArena invariant broken");

        const v2::NodeInfoHeader* hdr = arena_->info_at(info.info_offset);
        const uint16_t num_moves = hdr->num_moves;
        const uint16_t pc =
            std::atomic_ref<uint16_t>(hdr->pv_child)
                .load(std::memory_order_relaxed);
        if (pc == v2::kNoPvChild || pc >= num_moves) {
            return chess::Move::NO_MOVE;
        }
        const v2::MoveInfo* moves = arena_->moves_at(info.info_offset);
        return chess::Move{moves[pc].move};
    }

    /**
     * One of the "no search required" cases at the root:
     *   - `move == NO_MOVE`            => no legal moves; emit `bestmove 0000`.
     *   - `tb` empty, real move        => single-legal-move forced reply.
     *   - `tb` set, real move          => Syzygy DTZ resolution.
     *   - `repetition_draw`, real move => losing root TT entry; insta-play a
     *                                     repetition-causing move to claim a draw.
     *
     * The caller is responsible for emitting the appropriate `info` /
     * `bestmove` lines and skipping the rest of `worker_main`.
     */
    struct RootShortcut {
        chess::Move                          move;
        std::optional<catgpt::SyzygyRootResult> tb;
        // Set when `move` was chosen by the negative-Q repetition insta-play
        // (a 3-fold- or 2-fold-causing move that salvages a draw). Mutually
        // exclusive with `tb`; drives a distinct emit in emit_root_shortcut.
        bool repetition_draw = false;
    };

    /**
     * Detect whether the root can be resolved without spinning up the
     * search workers. Order is fixed:
     *   1. No legal moves (mate / stalemate) — terminal.
     *   2. Exactly one legal move — forced reply.
     *   3. Syzygy DTZ root probe (gated on prober availability +
     *      piece-count + no-castling). The probe itself is cheap.
     *   4. Losing-root repetition insta-play: if the root has a published TT
     *      entry whose Q < 0 (side-to-move is losing per the persisted eval),
     *      claim a draw by insta-playing a repetition-causing move — a 3-fold-
     *      causing move if one exists, else any 2-fold-causing move. Placed
     *      after Syzygy so it only fires on non-TB roots.
     * Returns std::nullopt if a real search is needed.
     *
     * Must be called from `worker_main` only (single-threaded); Fathom's
     * `tb_probe_root` is documented not-thread-safe.
     */
    [[nodiscard]] std::optional<RootShortcut> try_root_shortcut() const {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);

        if (legal.empty()) {
            return RootShortcut{chess::Move::NO_MOVE, std::nullopt};
        }
        if (legal.size() == 1) {
            return RootShortcut{legal[0], std::nullopt};
        }
        if (syzygy_ && syzygy_->is_eligible(board_)) {
            if (auto r = syzygy_->probe_root_dtz(board_); r) {
                return RootShortcut{r->move, std::move(r)};
            }
        }

        // Losing-root repetition insta-play. The root TT Q is read exactly as
        // `root_cp()` does; a negative Q means the side to move is losing, so a
        // forceable repetition draw strictly improves the outcome. Scan the
        // legal moves once, preferring a 3-fold-causing move (claimable draw)
        // and falling back to the first 2-fold-causing move. `makeMove<true>`
        // updates the hash/repetition history so `isRepetition` sees the new
        // position (same convention as `bestmove()`).
        const v2::TTEntry* root =
            arena_->find(board_.hash(), v2::secondary_hash(board_));
        if (root != nullptr) {
            auto [q, d] = v2::unpack_qd(
                v2::SearchArena::load_qd(root).qd_packed);
            (void)d;
            if (q <= 0.0f) {
                chess::Move twofold = chess::Move::NO_MOVE;
                for (const auto& m : legal) {
                    chess::Board cb = board_;
                    cb.makeMove<true>(m);
                    if (cb.isRepetition(2)) {
                        return RootShortcut{m, std::nullopt, true};
                    }
                    if (twofold == chess::Move::NO_MOVE
                        && cb.isRepetition(1)) {
                        twofold = m;
                    }
                }
                if (twofold != chess::Move::NO_MOVE) {
                    return RootShortcut{twofold, std::nullopt, true};
                }
            }
        }
        return std::nullopt;
    }

    /**
     * Stockfish-style mapping from Syzygy WDL to UCI `score cp` value.
     * Cursed/blessed scores are small magnitudes (50cp) to indicate the
     * 50-move rule will defang them. A real win/loss is reported as a
     * large finite cp value rather than a fake `score mate` (DTZ is not
     * DTM, so mate-in-N would mislead GUIs).
     */
    [[nodiscard]] static int syzygy_wdl_to_cp(catgpt::SyzygyWDL w) noexcept {
        switch (w) {
            case catgpt::SyzygyWDL::WIN:          return  12800;
            case catgpt::SyzygyWDL::CURSED_WIN:   return     50;
            case catgpt::SyzygyWDL::DRAW:         return      0;
            case catgpt::SyzygyWDL::BLESSED_LOSS: return    -50;
            case catgpt::SyzygyWDL::LOSS:         return -12800;
        }
        return 0;
    }

    /**
     * Emit the UCI lines for a root shortcut and return. Splits into
     * separate `info` lines because the UCI spec says `string` consumes
     * the rest of the line — keeping it on its own line is robust to
     * future field additions.
     */
    void emit_root_shortcut(const RootShortcut& sc,
                            const std::function<void(std::string_view)>& emit) const
    {
        if (!emit) return;

        if (sc.move == chess::Move::NO_MOVE) {
            emit("bestmove 0000");
            return;
        }

        const std::string uci_move = chess::uci::moveToUci(sc.move);
        char buf[256];

        if (sc.repetition_draw) {
            std::snprintf(buf, sizeof(buf),
                          "info depth 0 score cp 0 pv %s", uci_move.c_str());
            emit(buf);
            emit("info string insta-play repetition draw");
            std::snprintf(buf, sizeof(buf), "bestmove %s", uci_move.c_str());
            emit(buf);
            return;
        }

        if (sc.tb) {
            const int cp = syzygy_wdl_to_cp(sc.tb->wdl);
            std::snprintf(buf, sizeof(buf),
                          "info depth 0 score cp %d pv %s",
                          cp, uci_move.c_str());
            emit(buf);
            const std::string_view wdl_s = catgpt::to_string(sc.tb->wdl);
            std::snprintf(buf, sizeof(buf),
                          "info string syzygy wdl=%.*s dtz=%d",
                          static_cast<int>(wdl_s.size()), wdl_s.data(),
                          sc.tb->dtz);
            emit(buf);
        } else {
            std::snprintf(buf, sizeof(buf),
                          "info depth 0 pv %s", uci_move.c_str());
            emit(buf);
            emit("info string forced move");
        }

        std::snprintf(buf, sizeof(buf), "bestmove %s", uci_move.c_str());
        emit(buf);
    }

    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        // Default termination reason; overwritten on the soft-grace / hard
        // exits below. A root shortcut or any non-time exit leaves kOther.
        last_completion_ = Completion::kOther;

        // ── Root fast-paths ────────────────────────────────────────────
        // Cheap deterministic answers (no legal moves, single legal move,
        // Syzygy-resolvable root) short-circuit before any per-worker
        // counter reset or runner spawn — saves multi-second worker
        // spin-up and the GPU eval that would otherwise happen.
        if (auto sc = try_root_shortcut()) {
            emit_root_shortcut(*sc, cb);
            total_evals_.store(0, std::memory_order_relaxed);
            running_.store(false, std::memory_order_release);
            return;
        }

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // ── Wall-clock budget (see TimeControl) ──────────────────────────
        // soft_deadline: stop after the ID iteration that crosses it (the
        //   target spend; can be extended mid-search on instability).
        // hard_deadline: force-stop ceiling, enforced unconditionally every
        //   tick. Both stay at time_point::max() when neither a clock nor an
        //   explicit movetime was supplied, leaving the search time-unbounded.
        const TimeControl& tc = cfg.time;
        Clock::time_point soft_deadline = Clock::time_point::max();
        Clock::time_point hard_deadline = Clock::time_point::max();
        int64_t time_bank_ms        = 0;      // effective bank, for extensions
        bool    applied_change_bonus = false;
        bool    applied_worsen_bonus = false;
        if (tc.active()) {
            int64_t soft_ms;
            int64_t hard_ms;
            if (tc.movetime_ms > 0) {
                soft_ms = tc.movetime_ms;
                hard_ms = tc.movetime_ms;
            } else {
                const bool    white     = board_.sideToMove() == chess::Color::WHITE;
                const int64_t time_left = white ? tc.wtime_ms : tc.btime_ms;
                const int64_t inc       = white ? tc.winc_ms  : tc.binc_ms;
                const int64_t bank      =
                    std::max<int64_t>(0, time_left - tc.reserve_ms);
                time_bank_ms = bank;
                auto pct = [bank](float p) -> int64_t {
                    const float v = p * static_cast<float>(bank);
                    return v > 0.0f ? static_cast<int64_t>(v) : 0;
                };
                soft_ms = pct(tc.soft_pct) + inc;
                hard_ms = std::min<int64_t>(pct(tc.hard_pct) + inc, bank);
                // Soft floors for a surprise reply / the game's first move.
                // Mark the matching one-shot bonus consumed so the in-loop
                // extension does not stack on top of the floor.
                if (tc.surprise) {
                    soft_ms = std::max<int64_t>(soft_ms, pct(tc.surprise_pct) + inc);
                    applied_change_bonus = true;
                }
                if (tc.first_move) {
                    soft_ms = std::max<int64_t>(soft_ms, pct(tc.first_move_pct));
                    applied_change_bonus = true;
                    applied_worsen_bonus = true;
                }
                if (soft_ms > hard_ms) soft_ms = hard_ms;
            }
            soft_deadline = t0 + std::chrono::milliseconds(soft_ms);
            hard_deadline = t0 + std::chrono::milliseconds(hard_ms);
        }

        // Reset per-search counters. Stagger each worker's starting depth.
        // No per-worker eval slice is set: `cfg.max_evals` is enforced as
        // a single aggregate budget by the loop below.
        total_evals_.store(0, std::memory_order_relaxed);
        for (int i = 0; i < num_workers_; ++i) {
            auto& w = *workers_[i];
            w.stop.store(false, std::memory_order_relaxed);
            w.evals.store(0, std::memory_order_relaxed);
            w.tt_claims.store(0, std::memory_order_relaxed);
            const float start = cfg.start_depth
                              + (static_cast<float>(i) / static_cast<float>(num_workers_))
                                * cfg.delta_depth;
            w.depth.store(start, std::memory_order_relaxed);
        }

        // Spawn per-search runner jthreads against the persistent workers.
        for (int i = 0; i < num_workers_; ++i) {
            auto* w = workers_[i].get();
            w->runner = std::jthread([this, w, i, &cfg]() {
                run_worker_search(*w, i, cfg);
            });
        }

        // Aggregator + UCI info loop. Emit `info depth ...` when centi-depth
        // advances during the loop, and once more after runners join (see
        // below) so budget-grace exits still report final PV/score.
        //
        // Termination conditions (any one ends the loop):
        //   1. `stop_token` requested (UCI `stop`, `quit()`, etc.) —
        //      flips immediately at the top of every tick.
        //   2. Every worker's depth has reached `cfg.max_depth` — no
        //      runner has any more iterations to start. Needed because
        //      in-search Syzygy can leave the search spinning on
        //      terminal children that never trigger GPU evals, so an
        //      eval-only termination would never trip.
        //   3. `min_depth() >= cfg.target_min_depth` — explicit depth
        //      target was hit.
        //   4. Aggregate eval budget grace: once `evals_sum >=
        //      cfg.max_evals` is observed we latch the current
        //      `min_depth()` and keep running. Termination fires on the
        //      next tick where `min_depth()` strictly exceeds the
        //      latched value — i.e. the slowest worker has since
        //      finished one more iteration. This evens out termination
        //      depth across Lazy-SMP workers; expect a per-search eval
        //      overshoot of one slow-worker iteration.
        //   5. Wall-clock deadlines (see TimeControl): `hard_deadline`
        //      force-stops every tick; `soft_deadline` uses the same grace
        //      as the eval budget — it latches `min_depth()` once the
        //      (possibly extended) soft deadline passes and stops on the
        //      next tick where `min_depth()` strictly advances, i.e. the
        //      slowest worker has finished its in-flight ID iteration.
        long  last_depth_centi    = std::numeric_limits<long>::min();
        bool  budget_exhausted    = false;
        float min_depth_at_budget =
            -std::numeric_limits<float>::infinity();

        // Soft-deadline grace state (mirrors the eval-budget grace above).
        bool  soft_expired        = false;
        float min_depth_at_soft   =
            -std::numeric_limits<float>::infinity();

        // Mid-search instability tracking (for soft-deadline extension).
        chess::Move prev_best_move = chess::Move::NO_MOVE;
        int       prev_best_cp    = std::numeric_limits<int>::min();

        // Current root score (centipawns, side-to-move POV), or nullopt when
        // the root has no published TT entry yet.
        auto current_root_cp = [&]() -> std::optional<int> {
            const v2::TTEntry* root =
                arena_->find(root_key_, v2::secondary_hash(board_));
            if (root == nullptr) return std::nullopt;
            auto [root_q, _] = v2::unpack_qd(
                v2::SearchArena::load_qd(root).qd_packed);
            return q_to_cp(root_q);
        };

        auto emit_progress = [&](float min_d,
                                 uint64_t evals_sum,
                                 uint64_t claims_sum) {
            const long depth_centi = std::isfinite(min_d)
                ? std::lround(min_d * 100.0f)
                : 0L;
            const auto now = Clock::now();
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            const long long mss = static_cast<long long>(ms);

            std::string pv_field;
            const std::vector<chess::Move> pv_moves = principal_variation();
            if (!pv_moves.empty()) {
                pv_field = " pv";
                for (const chess::Move& m : pv_moves) {
                    pv_field += ' ';
                    pv_field += chess::uci::moveToUci(m);
                }
            }

            char score_field[32];
            if (const v2::TTEntry* root =
                    arena_->find(root_key_, v2::secondary_hash(board_));
                root != nullptr) {
                auto [root_q, _] = v2::unpack_qd(
                    v2::SearchArena::load_qd(root).qd_packed);
                std::snprintf(score_field, sizeof(score_field),
                              " score cp %d", q_to_cp(root_q));
            } else {
                score_field[0] = '\0';
            }

            // Fixed-prefix fields go through a small fixed buffer; the PV
            // (up to 256 plies ≈ 1.5 KB) is appended via std::string so a
            // long line is never truncated by a fixed buffer.
            char prefix[256];
            std::snprintf(prefix, sizeof(prefix),
                "info depth %ld%s nodes %llu tt_claims %llu time %lld nps %lld",
                depth_centi,
                score_field,
                static_cast<unsigned long long>(evals_sum),
                static_cast<unsigned long long>(claims_sum),
                mss,
                mss > 0 ? static_cast<long long>(evals_sum) * 1000 / mss : 0);
            std::string line = prefix;
            line += pv_field;
            emit(line);
            last_depth_centi = depth_centi;
        };

        while (true) {
            if (st.stop_requested()) break;

            const auto now = Clock::now();
            if (now >= hard_deadline) {
                // Force-stop at the hard cap. For a genuine game-clock
                // search (no explicit movetime) this is the "abort" path
                // — the soft grace never completed. `movetime` searches
                // have no soft phase, so they stay kOther.
                if (tc.movetime_ms <= 0) {
                    last_completion_ = Completion::kHardAborted;
                }
                break;
            }

            uint64_t evals_sum = 0;
            uint64_t claims_sum = 0;
            float    min_d = std::numeric_limits<float>::infinity();
            bool     all_at_max_depth = true;
            for (const auto& w : workers_) {
                evals_sum  += w->evals.load(std::memory_order_relaxed);
                claims_sum += w->tt_claims.load(std::memory_order_relaxed);
                const float d = w->depth.load(std::memory_order_relaxed);
                min_d = std::min(min_d, d);
                if (d < cfg.max_depth) all_at_max_depth = false;
            }
            total_evals_.store(evals_sum, std::memory_order_relaxed);

            if (all_at_max_depth) break;
            if (std::isfinite(min_d) && min_d >= cfg.target_min_depth) break;

            // Budget grace: latch min_d the first time we see the
            // aggregate budget exhausted; on a subsequent tick where
            // min_d has strictly advanced, terminate.
            if (!budget_exhausted && evals_sum >= cfg.max_evals) {
                budget_exhausted    = true;
                min_depth_at_budget = std::isfinite(min_d)
                    ? min_d
                    : -std::numeric_limits<float>::infinity();
            }
            if (budget_exhausted && std::isfinite(min_d)
                && min_d > min_depth_at_budget) {
                break;
            }

            // UCI's `depth` field is integer-valued (cutechess parses via
            // QString::toInt and most other GUIs do the same). Encode our
            // log-scale fractional depth as centi-depth so two ID steps of
            // delta=0.2 advance the field by 20 each.
            const long depth_centi = std::isfinite(min_d)
                ? std::lround(min_d * 100.0f)
                : 0L;
            const bool new_iter = (depth_centi != last_depth_centi);
            if (new_iter) {
                emit_progress(min_d, evals_sum, claims_sum);
            }

            // Mid-search soft-deadline extension on instability (port of the
            // legacy engine): on a fresh ID iteration, and only off the game's
            // first move, grant a one-time extension if the best move changed
            // or the root score dropped by more than worsen_threshold_cp.
            if (new_iter && !soft_expired && tc.active() && tc.movetime_ms <= 0
                && !tc.first_move
                && soft_deadline != Clock::time_point::max()) {
                const std::vector<chess::Move> pv = principal_variation();
                const chess::Move cur_best =
                    pv.empty() ? chess::Move::NO_MOVE : pv.front();
                const std::optional<int> cur_cp = current_root_cp();

                const bool changed =
                    prev_best_move != chess::Move::NO_MOVE
                    && cur_best != chess::Move::NO_MOVE
                    && cur_best != prev_best_move;
                const bool worsened =
                    prev_best_cp != std::numeric_limits<int>::min()
                    && cur_cp.has_value()
                    && (prev_best_cp - *cur_cp) > tc.worsen_threshold_cp;

                int64_t extra_ms = 0;
                if (changed && !applied_change_bonus) {
                    extra_ms += static_cast<int64_t>(
                        tc.change_bonus_pct * static_cast<float>(time_bank_ms));
                    applied_change_bonus = true;
                }
                if (worsened && !applied_worsen_bonus) {
                    extra_ms += static_cast<int64_t>(
                        tc.worsen_bonus_pct * static_cast<float>(time_bank_ms));
                    applied_worsen_bonus = true;
                }
                if (extra_ms > 0) {
                    auto new_soft =
                        soft_deadline + std::chrono::milliseconds(extra_ms);
                    if (new_soft > hard_deadline) new_soft = hard_deadline;
                    soft_deadline = new_soft;
                }

                if (cur_best != chess::Move::NO_MOVE) prev_best_move = cur_best;
                if (cur_cp.has_value()) prev_best_cp = *cur_cp;
            }

            // Soft deadline: mirror the eval-budget grace above. Latch min_d
            // the first time the (possibly extended) soft deadline passes,
            // then stop once the slowest worker has finished one more ID
            // iteration (min_d strictly advanced) — letting the in-flight
            // iteration complete instead of force-aborting at soft. `hard`
            // remains the absolute cap if min_d never advances.
            if (!soft_expired && now >= soft_deadline) {
                soft_expired      = true;
                min_depth_at_soft = std::isfinite(min_d)
                    ? min_d
                    : -std::numeric_limits<float>::infinity();
            }
            if (soft_expired && std::isfinite(min_d)
                && min_d > min_depth_at_soft) {
                // The "non-abort": min_depth advanced past the soft latch.
                last_completion_ = Completion::kSoftCompleted;
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Stop fan-out + join runners. Do NOT shutdown the persistent
        // evaluators — they live across searches.
        for (auto& w : workers_) {
            w->stop.store(true, std::memory_order_release);
        }
        for (auto& w : workers_) {
            if (w->runner.joinable()) w->runner.join();
        }

        // Final aggregate snapshot.
        uint64_t evals_sum = 0;
        uint64_t claims_sum = 0;
        for (const auto& w : workers_) {
            evals_sum  += w->evals.load(std::memory_order_relaxed);
            claims_sum += w->tt_claims.load(std::memory_order_relaxed);
        }
        total_evals_.store(evals_sum, std::memory_order_relaxed);

        // Final `info depth` after runners join: stable TT for PV/score and
        // covers budget grace (we break on min_depth advance before the loop
        // body can emit that step) and other early exits at an unchanged
        // centi-depth tick.
        {
            float min_d = std::numeric_limits<float>::infinity();
            for (const auto& w : workers_) {
                min_d = std::min(min_d,
                    w->depth.load(std::memory_order_relaxed));
            }
            const long depth_centi = std::isfinite(min_d)
                ? std::lround(min_d * 100.0f)
                : 0L;
            if (depth_centi != last_depth_centi || budget_exhausted) {
                emit_progress(min_d, evals_sum, claims_sum);
            }
        }

        // Pick the best move from the TT now that all runners have joined.
        const chess::Move best = bestmove();
        char bm[64];
        if (best != chess::Move::NO_MOVE) {
            const std::string uci_move = chess::uci::moveToUci(best);
            std::snprintf(bm, sizeof(bm), "bestmove %s", uci_move.c_str());
        } else {
            std::snprintf(bm, sizeof(bm), "bestmove 0000");
        }
        emit(bm);

        running_.store(false, std::memory_order_release);
    }

    /**
     * Runner body. Spawned per-search by worker_main on its own jthread.
     * Drives the iterative-deepening loop for this worker.
     *
     * The runner has no per-worker eval cap of its own — it keeps
     * grinding iterations until either the cooperative `stop` flag
     * flips (set by `worker_main` when the aggregate budget grace
     * expires, a `stop_token` fires, or any other terminator) or the
     * local ID depth reaches `cfg.max_depth`.
     */
    // Map a CUDA device to its local NUMA node (OS node id) via its PCIe
    // bus id and /sys, with a round-robin fallback when sysfs reports no
    // affinity (numa_node == -1, common in VMs / single-node hosts). Lives
    // here (not in numa_util.hpp) so the CUDA dependency stays out of the
    // header shared with the CUDA-less tt_arena tests.
    static int gpu_numa_node(int device_id, const numa::Topology& topo) {
        char bus[32] = {0};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device_id) != cudaSuccess) {
            return numa::round_robin_node(device_id, topo);
        }
        const int n = numa::numa_node_for_pci_bus(bus);
        if (n < 0 || topo.index_of_node(n) < 0) {
            return numa::round_robin_node(device_id, topo);
        }
        return n;
    }

    void run_worker_search(WorkerSearch& w, int worker_idx,
                           const LksSearchConfig& cfg)
    {
        // The runner mostly blocks in lf::sync_wait while the pool's pinned
        // descent thread does the work, so a loose node-level pin (not a
        // dedicated core) is enough to keep its bookkeeping NUMA-local
        // without stealing a physical core from the hot threads.
        if (w.compute_cpu >= 0) {
            numa::pin_this_thread_to_node(topo_, w.numa_node);
        }
        const float start = cfg.start_depth
                          + (static_cast<float>(worker_idx) / static_cast<float>(num_workers_))
                            * cfg.delta_depth;
        float depth = start;
        w.depth.store(depth, std::memory_order_relaxed);
        while (!w.should_abort() && depth < cfg.max_depth) {
            run_iteration(w, cfg.params, depth);
            depth += cfg.delta_depth;
            w.depth.store(depth, std::memory_order_relaxed);
        }
    }

    /**
     * One ID iteration: build a per-iteration RecurseContext and dispatch
     * the descent onto this worker's lazy_pool. lf::sync_wait blocks the
     * runner thread until the recursion completes.
     *
     * The runner is NOT a libfork worker thread, so calling sync_wait
     * (which would throw schedule_in_worker if called from one) is safe.
     */
    void run_iteration(WorkerSearch& w,
                       const detail::SearchParams& params,
                       float depth) {
        detail::RecurseContext ctx{
            /*arena=*/&*arena_,
            /*evaluator=*/w.evaluator.get(),
            /*sem=*/w.sem.get(),
            /*w=*/&w,
            /*params=*/&params,
            /*syzygy=*/syzygy_.get(),
        };
        lf::sync_wait(*w.pool, detail::root_search, &ctx, board_, depth);
    }

    fs::path trt_engine_path_;
    uint64_t lifetime_max_evals_;
    int      workers_per_gpu_;
    int      num_gpus_ = 0;
    int      num_workers_ = 0;   // workers_per_gpu_ * num_gpus_, set in ctor
    int      coros_per_worker_;
    int      max_batch_size_;

    chess::Board board_;
    uint64_t     root_key_ = 0;
    bool         searched_since_reset_ = false;

    // NUMA topology (scanned once in the ctor) and the per-worker core
    // assignment derived from it. Owned here so the arena's parallel
    // allocation (ctor + reset) reuses it without re-scanning /sys.
    numa::Topology   topo_;
    std::vector<int> w_node_;         // OS node id per worker
    std::vector<int> w_compute_cpu_;  // descent (lazy_pool) core per worker
    std::vector<int> w_gpu_cpu_;      // GPU thread core per worker

    std::optional<v2::SearchArena>             arena_;
    std::vector<std::unique_ptr<WorkerSearch>> workers_;
    std::jthread                               worker_;
    std::atomic<bool>                          running_{false};
    std::atomic<uint64_t>                      total_evals_{0};

    // How the most recent worker_main run ended (see last_completion()).
    // Written only on the worker thread; read by the driver post-join.
    Completion                                 last_completion_ = Completion::kOther;

    // Syzygy tablebase prober — owns Fathom's global tb_init/tb_free
    // lifecycle. Null when no Syzygy path was provided. Probed once at
    // the top of every `worker_main` (before runners spawn) so the
    // not-thread-safe `tb_probe_root` is safe.
    std::unique_ptr<catgpt::SyzygyProber> syzygy_;
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
