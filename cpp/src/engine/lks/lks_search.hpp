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
 *       3. Re-deepen check: if the entry's max_depth >= depth, return.
 *       4. Single-pass classify+fork: each child is either
 *          Skip / RecurseThenRead / ReadOnly. RecurseThenRead spawns a
 *          child `recursive_search` task via `lf::fork`. First fork
 *          inherits the parent's permit; subsequent forks each
 *          `co_await sem.acquire()`.
 *       5. `lf::join` waits for all forked children.
 *       6. Rollup: P-weighted average of -child_Q over the same plan
 *          vector (terminal/repetition/50-move contribute fixed Q;
 *          RecurseThenRead reads the child's TT entry).
 *       7. `update_qd` with the new (Q, depth).
 */

#ifndef CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
#define CATGPT_ENGINE_LKS_LKS_SEARCH_HPP

#include <algorithm>
#include <array>
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
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
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
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../trt_runtime.hpp"
#include "../fractional_mcts/v2/board_secondary.hpp"
#include "../fractional_mcts/v2/tt_arena.hpp"

namespace catgpt::lks {

namespace fs = std::filesystem;

/**
 * Search-algorithm knobs (separate from per-search driver settings
 * like budgets and UCI plumbing).
 *
 * Defaults mirror `coroutine_search.hpp` / `FractionalMCTSConfig`:
 *
 *   - c_puct                : PUCT exploration constant fed into the
 *                             Halley allocation solver.
 *   - fpu_reduction         : First-Play Urgency reduction. Unexpanded
 *                             children's effective Q starts at
 *                             `parent.Q - fpu_reduction * sqrt(cum_P)`
 *                             where `cum_P` is the sum of priors of
 *                             higher-ranked siblings (P-sorted).
 *   - clamp_log_step        : per-iteration clamp cap. Each child's
 *                             allocation `log_n_i` is clamped to
 *                             `recorded_depth_i + clamp_log_step` (a
 *                             recorded depth of -inf yields no clamp).
 *                             Set to log(1.3) ≈ 0.262.
 *   - weight_log_threshold  : at the rollup, children with
 *                             `log_n_i < depth + weight_log_threshold`
 *                             are dropped from the N-weighted
 *                             average. Set to log(0.02) ≈ -3.912.
 *   - max_clamp_iters       : hard cap on the iterative clamp loop.
 *                             The loop also breaks early once no
 *                             non-trivial clamp fires (iter > 0).
 *   - forced_unexpanded     : number of P-sorted MoveInfo slots
 *                             whose UNEXPANDED status maps to
 *                             `depth = -inf` (so the allocator gives
 *                             those slots their full PUCT share on
 *                             iter 0). Position-based: applies to
 *                             slot 0 and slot 1 iff they are
 *                             Unexpanded; if a forced slot is
 *                             TTHit / Terminal it keeps its natural
 *                             depth.
 */
struct LksAlgoConfig {
    float c_puct = 1.75f;
    float fpu_reduction = 0.330f;
    float clamp_log_step = 0.26236426f;        // log(1.3)
    float weight_log_threshold = -3.9120230f;  // log(0.02)
    int   max_clamp_iters = 100;
    int   forced_unexpanded = 2;
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
    uint64_t max_evals = 800;       // total budget across all workers
    int min_info_period_ms = 100;   // throttle for aggregated `info` lines

    // Iterative-deepening (log-scale). N = e^depth.
    float start_depth = 0.0f;       // worker 0's starting depth
    float delta_depth = 0.2f;       // per-iteration depth step
    float max_depth   = 32.0f;      // absolute depth cap (e^32 ~= 8e13)

    LksAlgoConfig algo;             // PUCT / FPU / clamp / rollup knobs

    std::function<void(std::string_view)> on_uci_line;
};

namespace detail {

/**
 * Paired legal move + its softmax prior. Emitted by
 * `softmax_legal_sorted` in decreasing-P order so downstream
 * MoveInfo fill is a single sequential pass and the arena
 * ends up P-sorted without an in-place shuffle.
 *
 * Kept small (8 bytes) and mirrors MoveInfo's `move`/`P` layout:
 * sorting cost is the same as sorting MoveInfos, and we avoid
 * the makeMove/unmakeMove terminal work running in a soon-to-be-
 * thrown-away order. Stored as the raw `uint16_t` rather than a
 * `chess::Move` because the latter carries an unused 16-bit
 * score field that would bloat this struct.
 */
struct MoveWithPrior {
    uint16_t move;  // 2: chess::Move underlying u16
    uint16_t _pad;  // 2
    float    P;     // 4
};
static_assert(sizeof(MoveWithPrior) == 8, "MoveWithPrior must be 8 bytes");

/**
 * Mirror `coroutine_search.hpp::evaluate_node`'s temp-1.0 softmax over
 * legal-move policy logits, sorted by decreasing P. Output order is
 * what downstream descent expects (highest-prior first), and is also
 * the order we want in the arena's MoveInfo[].
 */
inline void softmax_legal_sorted(const RawNNOutput& out,
                                 const chess::Board& board,
                                 const chess::Movelist& legal,
                                 std::vector<MoveWithPrior>& moves)
{
    const bool flip = board.sideToMove() == chess::Color::BLACK;
    const int n = legal.size();
    moves.resize(static_cast<size_t>(n));

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) {
        const auto [from_idx, to_idx] = encode_move_to_policy_index(legal[i], flip);
        const int flat = policy_flat_index(from_idx, to_idx);
        const float logit = out.policy[flat];
        moves[i].move = static_cast<uint16_t>(legal[i].move());
        moves[i]._pad = 0;
        moves[i].P = logit;
        max_logit = std::max(max_logit, logit);
    }
    float sum_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        moves[i].P = std::exp(moves[i].P - max_logit);
        sum_exp += moves[i].P;
    }
    const float inv_sum = sum_exp > 0.0f ? 1.0f / sum_exp : 0.0f;
    for (int i = 0; i < n; ++i) {
        moves[i].P *= inv_sum;
    }

    std::sort(moves.begin(), moves.end(),
        [](const MoveWithPrior& a, const MoveWithPrior& b) {
            return a.P > b.P;
        });
}

/**
 * Variance of the NN value distribution, measured in the same
 * [-1, 1] scale as `Q = 2*out.value - 1`. The HL-Gauss head's 81
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
 * Position-only terminal classification for a child move at expansion
 * time. Path-dependent terminal conditions are NOT considered here:
 *   - repetitions: handled at descent time via `isRepetition(1)`.
 *   - 50-move rule: handled at descent time via `isHalfMoveDraw()`.
 * Both depend on data (path history / half-move clock) that is not
 * part of the Zobrist key, so two transpositions to the same key can
 * disagree on them. Only stable, position-only draws (insufficient
 * material) and mate/stalemate go into the TT-stored terminal_kind.
 *
 * Pre: `child_board` is the board AFTER `makeMove<true>(move)`.
 */
inline v2::TerminalKind classify_terminal(chess::Board& child_board) {
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
    return v2::kTerminalNone;
}

/**
 * Per-worker_search state.
 *
 * Persistent across searches:
 *   - pool, evaluator, sem
 *
 * Per-search (reset by worker_main at the start of every search):
 *   - stop, evals, tt_claims, depth, budget
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

    std::atomic<bool>     stop{false};
    std::atomic<uint64_t> evals{0};
    std::atomic<uint64_t> tt_claims{0};
    std::atomic<uint64_t> budget{0};            // per-worker eval cap for this search
    std::atomic<float>    depth{0.0f};

    std::jthread runner;

    WorkerSearch() = default;
    WorkerSearch(const WorkerSearch&) = delete;
    WorkerSearch& operator=(const WorkerSearch&) = delete;
    WorkerSearch(WorkerSearch&&) = delete;
    WorkerSearch& operator=(WorkerSearch&&) = delete;

    // Single common abort-check for both `stop` and budget exhaustion.
    // Used at every `co_await` boundary so a search drains promptly.
    [[nodiscard]] bool should_abort() const noexcept {
        return stop.load(std::memory_order_relaxed)
            || evals.load(std::memory_order_relaxed)
               >= budget.load(std::memory_order_relaxed);
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
    const LksAlgoConfig*      algo;
};

/**
 * Result of a single `recursive_search` invocation.
 *
 * `Q` is the rolled-up value from THIS node's side-to-move POV (i.e. the
 * same convention used for `qd_packed` Cell A; the parent negates this
 * value when accumulating into its own rollup).
 *
 * `depth` is the log-scale budget that was committed at this node — for
 * a normal completion this equals the `depth` argument that was passed
 * to `recursive_search`; for the early-out re-deepen path it equals the
 * cached `max_depth` already in the TT entry (which is `>= depth`).
 *
 * Returned wrapped in `std::optional` so callers can distinguish a
 * normal completion from an aborted descent (`std::nullopt`). Aborted
 * children propagate up: any sibling returning `nullopt` causes the
 * parent to also return `nullopt` without rolling up or updating the TT.
 */
struct QDResult {
    float Q;
    float depth;
};

/**
 * POC: Halley-method allocation solver in delta-space.
 *
 * Computes log N_i for each child given priors P_i, child Q values
 * Q_i, total budget N = e^depth, and c_puct, by solving for
 * delta = K - q_max in
 *
 *     g(delta) = sum_i w_i / (delta + Delta_i) - N = 0
 *
 * with w_i = c_puct * P_i * N^(2/3)/3 and Delta_i = q_max - q_i,
 * q_i = -Q_i. g is convex-decreasing on (0, +inf) so Halley from the
 * provable upper bound delta_hi = c_puct/(3 N^(1/3)) converges
 * monotonically from above without overshoot. Cubic convergence:
 * ~3 iters cold to log_tol = 1e-6.
 *
 * No transcendentals in the inner loop. ~600ns at M=30 cold per the
 * gym (`compute_alloc_gym.{hpp,cpp}` -- this is the A6c_halley_d_hi
 * solver, ported in-place for the POC).
 *
 * Output is log(N_i) (LKS works in depth = log N space).
 *
 * NOTE: POC. Wired into recursive_search but the result is currently
 * unused -- purpose is to validate the call site, gather realistic
 * (P, Q) inputs from the live TT, and measure per-iteration cost
 * inside the actual search before deciding how to consume the
 * allocations downstream (sub-budgeting, depth shaping, etc.).
 */
struct AllocResult {
    static constexpr std::size_t kMaxChildren = 256;
    std::array<float, kMaxChildren> log_n{};
    uint16_t M = 0;
    uint16_t iters = 0;
    bool converged = false;
};

inline AllocResult compute_allocations_halley(
    const float* P,
    const float* Q,
    uint16_t M,
    float depth,
    float c_puct = 2.0f,
    int max_iters = 12,
    double log_tol = 1e-6)
{
    AllocResult r{};
    if (M == 0 || M > AllocResult::kMaxChildren) return r;
    r.M = M;

    float qmax = -std::numeric_limits<float>::infinity();
    for (uint16_t i = 0; i < M; ++i) {
        const float negq = -Q[i];
        if (negq > qmax) qmax = negq;
    }

    std::array<double, AllocResult::kMaxChildren> Delta{};
    for (uint16_t i = 0; i < M; ++i) {
        Delta[i] = static_cast<double>(qmax) - static_cast<double>(-Q[i]);
    }

    const double N = std::exp(static_cast<double>(depth));
    const double w_factor = static_cast<double>(c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;
    const double tol_g = log_tol * N;

    auto eval = [&](double d) {
        double s = 0.0, s2 = 0.0, s3 = 0.0;
        for (uint16_t i = 0; i < M; ++i) {
            if (P[i] <= 0.0f) continue;
            const double di  = d + Delta[i];
            const double inv = 1.0 / di;
            const double t   = w_factor * static_cast<double>(P[i]) * inv;
            s  += t;
            s2 += t * inv;
            s3 += t * inv * inv;
        }
        return std::tuple{s - N, -s2, 2.0 * s3};
    };

    // Cold-start from the provable upper bound on delta*.
    double d = static_cast<double>(c_puct) / (3.0 * std::cbrt(N));

    int it = 0;
    for (; it < max_iters; ++it) {
        auto [g, gp, gpp] = eval(d);
        if (std::fabs(g) <= tol_g) { ++it; break; }
        const double denom = 2.0 * gp * gp - g * gpp;
        double d_new;
        if (gp < 0.0 && std::isfinite(denom) && denom != 0.0) {
            d_new = d - (2.0 * g * gp) / denom;
            if (d_new <= 0.0) d_new = d * 0.5;
        } else {
            d_new = d * 0.5;
        }
        if (std::fabs(d_new - d) <= log_tol * (d + 1e-30)) {
            d = d_new;
            ++it;
            break;
        }
        d = d_new;
    }

    const double bias = (2.0/3.0) * static_cast<double>(depth)
                      + std::log(static_cast<double>(c_puct) / 3.0);
    for (uint16_t i = 0; i < M; ++i) {
        if (P[i] <= 0.0f) {
            r.log_n[i] = -std::numeric_limits<float>::infinity();
        } else {
            const double delta_i = d + Delta[i];
            r.log_n[i] = static_cast<float>(
                std::log(static_cast<double>(P[i])) - std::log(delta_i) + bias);
        }
    }
    r.iters = static_cast<uint16_t>(it);
    r.converged = it < max_iters;
    return r;
}

/**
 * Recursive descent (libfork lambda).
 *
 * INVARIANT: every entry owns exactly one `Permit`. The permit covers
 * this entire invocation including any GPU eval. At fan-out the first
 * RecurseThenRead child inherits via std::move(permit); subsequent
 * children each co_await sem->acquire(). On exit (either co_return at
 * abort, base, or after lf::join) any permit still held is released by
 * RAII.
 */
inline constexpr auto recursive_search =
    [](auto recursive_search,
       RecurseContext* ctx,
       lfsync::Permit permit,
       chess::Board board,
       float depth) -> lf::task<std::optional<QDResult>>
{
    if (ctx->w->should_abort()) co_return std::nullopt;

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
        auto tokens = catgpt::tokenize<BatchEvaluator::SEQ_LENGTH>(
            board, NO_HALFMOVE_CONFIG);
        RawNNOutput out = co_await EvalAwaitable(*ctx->evaluator, tokens);
        ctx->w->evals.fetch_add(1, std::memory_order_relaxed);

        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board);
        const uint16_t num_moves = static_cast<uint16_t>(legal.size());

        // Sort moves by decreasing P up front (on compact 8-byte
        // pairs) so the arena fill below is a single pass that
        // writes each MoveInfo exactly once in the order descent
        // will consume them.
        std::vector<MoveWithPrior> sorted_moves;
        softmax_legal_sorted(out, board, legal, sorted_moves);

        // alloc + fill BEFORE attempting to claim — these bytes
        // are privately owned until the CAS, and orphaned if the
        // CAS loses.
        const uint64_t off = ctx->arena->alloc_node_info(num_moves);
        v2::MoveInfo* mi = ctx->arena->moves_at(off);
        for (uint16_t i = 0; i < num_moves; ++i) {
            const chess::Move m{sorted_moves[i].move};

            // Per-move position-only terminal detection.
            board.makeMove<true>(m);
            const v2::TerminalKind tk = classify_terminal(board);
            board.unmakeMove(m);

            mi[i] = v2::MoveInfo::pack(sorted_moves[i].move,
                                       sorted_moves[i].P,
                                       tk);
        }

        // Per-node variance of the value distribution, in the same
        // [-1, 1] scale as Q. alloc_node_info pre-fills variance=0;
        // overwrite with the real value.
        ctx->arena->info_at(off)->variance =
            compute_value_variance(out.value_probs);

        const float Q = 2.0f * out.value - 1.0f;

        // Single 128-bit CAS installs (key, qd_packed(Q, 0))
        // atomically, so any reader observing key == K necessarily
        // sees the matching qd. Then a single 128-bit release-store
        // of Cell B publishes (origQ, key_secondary, info_offset).
        auto [ce, claimed] = ctx->arena->find_or_claim(
            key, sec, Q, /*max_depth=*/0.0f);
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
        if (ctx->w->should_abort()) co_return std::nullopt;

        entry = ce;
    }

    // ── re-deepen check (works for both fresh-claim and TT-shared) ──
    // Also captures `parent_Q` (this node's current rolled-up Q) for
    // the FPU formula in the unexpanded-children pre-pass below.
    float parent_Q;
    {
        // Cell A is atomic with the key match (find / find_or_claim
        // both ensure we observe a key from a successful CAS), so qd
        // is never torn here.
        auto [cur_q, cur_max_d] = v2::unpack_qd(
            v2::SearchArena::load_qd(entry).qd_packed);
        if (depth <= cur_max_d) co_return QDResult{cur_q, cur_max_d};
        parent_Q = cur_q;
    }

    // ── ChildState pre-pass over MoveInfo ──────────────────────────
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
    const v2::InfoCell info_cell = v2::SearchArena::load_info(entry);
    assert(info_cell.info_offset != v2::kNoInfoOffset
           && "Cell B unpublished after find/find_or_claim returned a "
              "validated entry; invariant broken");
    const v2::NodeInfoHeader* hdr = ctx->arena->info_at(info_cell.info_offset);
    const uint16_t num_moves = hdr->num_moves;
    const v2::MoveInfo* moves = ctx->arena->moves_at(info_cell.info_offset);

    /**
     * Per-child classification (data-shape, distinct from the
     * fork-or-not control decision computed below).
     *
     *  - `TerminalDraw`         : MoveInfo terminal_kind == Draw.
     *                             Q = 0 (child-POV), depth = +inf.
     *  - `TerminalLossForChild` : MoveInfo terminal_kind == LossForChild.
     *                             Q = -1 (child-POV: child loses),
     *                             depth = +inf.
     *  - `TerminalRepetition`   : path-dependent 2-fold detected
     *                             after `makeMove`. Q = 0, depth = +inf.
     *  - `TerminalHalfMove`     : path-dependent 50-move clock draw.
     *                             Q = 0, depth = +inf.
     *  - `TTHit`                : child key found in TT. Q = TT q
     *                             (child-POV), depth = TT max_depth.
     *  - `Unexpanded`           : child key not in TT. Q = FPU
     *                             value in child-POV (= -(parent_Q -
     *                             fpu_reduction * sqrt(cum_P))),
     *                             depth = 0, except slots 0 and 1
     *                             of the P-sorted MoveInfo are
     *                             forced to depth = -inf when those
     *                             slots are Unexpanded (the
     *                             `forced_unexpanded` rule — a
     *                             position-based override on
     *                             indices, NOT a "first 2 among
     *                             unexpanded children" rule).
     */
    enum class ChildStatus : uint8_t {
        TerminalDraw,
        TerminalLossForChild,
        TerminalRepetition,
        TerminalHalfMove,
        TTHit,
        Unexpanded,
    };
    auto is_terminal = [](ChildStatus s) noexcept {
        return s == ChildStatus::TerminalDraw
            || s == ChildStatus::TerminalLossForChild
            || s == ChildStatus::TerminalRepetition
            || s == ChildStatus::TerminalHalfMove;
    };

    /**
     * One slot per MoveInfo entry, populated unconditionally by the
     * pre-pass and consumed by both the decision/fork loop and the
     * rollup.
     *
     * `Q` is in **child-POV** (matches TT `qd_packed` semantics and
     * the Halley solver's input convention). Parent-POV values used
     * for the rollup are obtained as `-Q`.
     *
     * `cb`, `child_key`, `child_sec` are populated for every
     * non-MoveInfo-terminal child (i.e. `cb` is junk for
     * TerminalDraw / TerminalLossForChild). The path-dependent
     * terminal kinds (Repetition / HalfMove) DO require `makeMove`
     * for detection so their `cb` is also valid; we just won't
     * recurse into them.
     *
     * `was_forked` is a transient flag set by the clamp-iteration
     * fork pass to direct (a) the abort-propagation scan and (b) the
     * (Q, depth) writeback from `child_result`. Reset to false at
     * the top of every clamp iter.
     *
     * `child_result` is the return slot written by `lf::fork` into
     * this child's `recursive_search`. Default `nullopt` for
     * unforked children and for forks that aborted (the child
     * coroutine returned `nullopt`).
     */
    struct ChildState {
        chess::Move    move;
        float          P;
        ChildStatus    status;
        float          Q;       // child-POV
        float          depth;   // recorded log-N for the child
        chess::Board   cb;      // post-makeMove (junk for position-only terminals)
        uint64_t       child_key;
        uint32_t       child_sec;
        bool           was_forked;
        std::optional<QDResult> child_result;
    };

    std::vector<ChildState> states;
    states.reserve(num_moves);

    constexpr float kInf = std::numeric_limits<float>::infinity();
    const int forced_unexpanded = ctx->algo->forced_unexpanded;
    const float fpu_reduction = ctx->algo->fpu_reduction;

    float cumulative_P = 0.0f;
    for (uint16_t i = 0; i < num_moves; ++i) {
        const auto& m = moves[i];
        const v2::TerminalKind m_tk = m.terminal_kind();
        const float m_P = m.P();

        ChildState cs;
        cs.move = chess::Move{m.move};
        cs.P = m_P;
        cs.child_key = 0;
        cs.child_sec = 0;
        cs.was_forked = false;
        cs.child_result = std::nullopt;

        if (m_tk == v2::kTerminalDraw) {
            cs.status = ChildStatus::TerminalDraw;
            cs.Q = 0.0f;
            cs.depth = kInf;
        } else if (m_tk == v2::kTerminalLossForChild) {
            cs.status = ChildStatus::TerminalLossForChild;
            cs.Q = -1.0f;
            cs.depth = kInf;
        } else {
            // makeMove<true> is the strict-EP template overload; the
            // local-then-move dance avoids the parser ambiguity that
            // arises calling `cs.cb.makeMove<true>(...)` through a
            // member access inside the auto-typed libfork lambda.
            chess::Board cb = board;
            cb.makeMove<true>(cs.move);
            if (cb.isRepetition(1)) {
                cs.status = ChildStatus::TerminalRepetition;
                cs.Q = 0.0f;
                cs.depth = kInf;
                cs.cb = std::move(cb);
            } else if (cb.isHalfMoveDraw()) {
                cs.status = ChildStatus::TerminalHalfMove;
                cs.Q = 0.0f;
                cs.depth = kInf;
                cs.cb = std::move(cb);
            } else {
                cs.child_key = cb.hash();
                cs.child_sec = v2::secondary_hash(cb);
                if (v2::TTEntry* ce = ctx->arena->find(cs.child_key, cs.child_sec)) {
                    // (key, secondary) match verified by find. Cell A's
                    // qd is atomic with the primary match — no spin.
                    auto [q, child_max_d] = v2::unpack_qd(
                        v2::SearchArena::load_qd(ce).qd_packed);
                    cs.status = ChildStatus::TTHit;
                    cs.Q = q;
                    cs.depth = child_max_d;
                } else {
                    cs.status = ChildStatus::Unexpanded;
                    // FPU reduction lives in PARENT-POV; we store
                    // child-POV by negating: child-POV Q = -(parent
                    // Q - reduction * sqrt(cum_P)).
                    const float fpu_parent_pov =
                        parent_Q - fpu_reduction * std::sqrt(cumulative_P);
                    cs.Q = -fpu_parent_pov;
                    // Position-based forced override on indices 0 and 1
                    // (only when the slot ends up Unexpanded — a TTHit
                    // or terminal slot at index 0/1 keeps its natural
                    // depth).
                    cs.depth = (i < static_cast<uint16_t>(forced_unexpanded))
                        ? -kInf : 0.0f;
                }
                cs.cb = std::move(cb);
            }
        }

        cumulative_P += m_P;
        states.push_back(std::move(cs));
    }

    // ── Halley-driven allocation + clamped iteration loop ─────────
    // Each iter:
    //   1. Build (P[], Q[]) from current ChildState and run the
    //      log-space Halley solver to get log_n[i].
    //   2. Per-child compute the clamp limit and the recurse
    //      decision:
    //          clamp_limit_i = (depth_i == -inf) ? +inf
    //                                            : depth_i + clamp_log_step
    //          clamped_i     = min(log_n_i, clamp_limit_i)
    //          needs_recurse_i = (log_n_i > depth_i)         (strict)
    //          clamp_fired_i   = (log_n_i > clamp_limit_i)   (strict)
    //   3. Fork every needs_recurse_i child with depth =
    //      clamped_i; lf::join.
    //   4. Abort propagation: any forked child returning nullopt
    //      causes us to also co_return nullopt (no rollup).
    //   5. Copy (Q, depth) from each fork's QDResult back into
    //      the corresponding ChildState slot. This avoids any TT
    //      re-find and feeds the next iteration's Halley.
    //   6. Termination: break the loop if `clamp_iter > 0` and no
    //      child's `log_n_i` exceeded its clamp limit (i.e. every
    //      forked child got its full requested allocation — the
    //      clamping is no longer binding).
    //
    // Permit handling: parent enters with one permit. Iter 0's
    // first fork inherits via std::move; every other fork
    // (including iter ≥ 1's first fork) does its own
    // `co_await sem.acquire()`. Tracked via `permit_held` so the
    // inheritance fires at most once across all iterations.
    assert(num_moves <= AllocResult::kMaxChildren
           && "num_moves > kMaxChildren — chess legal-move count "
              "should never exceed 256");
    std::array<float, AllocResult::kMaxChildren> P_buf{};
    std::array<float, AllocResult::kMaxChildren> Q_buf{};
    std::array<float, AllocResult::kMaxChildren> clamped_log_n{};
    std::array<bool,  AllocResult::kMaxChildren> needs_recurse{};

    bool permit_held = true;
    const auto& algo = *ctx->algo;
    const int max_iters = algo.max_clamp_iters;

    for (int clamp_iter = 0; clamp_iter < max_iters; ++clamp_iter) {
        // Reset per-iter transient flags. `child_result` is
        // value-reset so a slot's prior-iter value can't masquerade
        // as this iter's return.
        for (ChildState& cs : states) {
            cs.was_forked = false;
            cs.child_result = std::nullopt;
        }

        for (uint16_t i = 0; i < num_moves; ++i) {
            P_buf[i] = states[i].P;
            Q_buf[i] = states[i].Q;  // child-POV — matches Halley input
        }
        AllocResult alloc = compute_allocations_halley(
            P_buf.data(), Q_buf.data(), num_moves, depth, algo.c_puct);

        bool any_clamp_fired = false;
        for (uint16_t i = 0; i < num_moves; ++i) {
            const float log_n_i = alloc.log_n[i];
            const float d_i     = states[i].depth;
            const float clamp_limit = (std::isinf(d_i) && d_i < 0.0f)
                ? kInf
                : d_i + algo.clamp_log_step;
            if (log_n_i > clamp_limit) any_clamp_fired = true;
            clamped_log_n[i] = std::min(log_n_i, clamp_limit);
            needs_recurse[i] = log_n_i > d_i;
        }

        // Fork pass.
        bool any_fork = false;
        for (uint16_t i = 0; i < num_moves; ++i) {
            ChildState& cs = states[i];
            if (is_terminal(cs.status)) continue;  // d_i = +inf, never recurses
            if (!needs_recurse[i])      continue;  // alloc <= recorded depth

            lfsync::Permit child_permit = permit_held
                ? std::move(permit)
                : co_await ctx->sem->acquire();
            permit_held = false;
            any_fork    = true;
            cs.was_forked = true;

            // Copy cs.cb (rather than std::move) so the original
            // survives the next clamp iteration — `cs.cb` is
            // re-used unchanged across iters since the move it
            // encodes doesn't depend on which iter we're in.
            co_await lf::fork(&cs.child_result, recursive_search)(
                ctx, std::move(child_permit), chess::Board(cs.cb),
                clamped_log_n[i]);
        }

        if (any_fork) {
            co_await lf::join;
        }

        if (ctx->w->should_abort()) co_return std::nullopt;

        // Abort propagation: any forked child that returned
        // `nullopt` is the unambiguous signal that the descent
        // bailed out partway. Propagate up — no rollup, no TT
        // commit — so partial / stale work doesn't pollute the TT.
        for (const ChildState& cs : states) {
            if (cs.was_forked && !cs.child_result.has_value()) {
                co_return std::nullopt;
            }
        }

        // Pull (Q, depth) from each fork's QDResult into ChildState
        // (no TT re-find). For the next clamp iter, cs.depth is
        // exactly the depth the sub-recurse committed to (= the
        // clamped allocation we just passed in).
        //
        // Also promote `Unexpanded` -> `TTHit` for any forked child:
        // after the descent committed (Q, depth) to the TT, this slot
        // carries a real Q estimate, not an FPU value, so the final
        // rollup's "drop Unexpanded" filter must no longer skip it.
        // (Without this promotion, the forced_unexpanded slots — which
        // always fork at iter 0 — would silently be excluded from the
        // parent's Q rollup despite holding the largest allocations.)
        for (ChildState& cs : states) {
            if (cs.was_forked) {
                cs.Q     = cs.child_result->Q;
                cs.depth = cs.child_result->depth;
                if (cs.status == ChildStatus::Unexpanded) {
                    cs.status = ChildStatus::TTHit;
                }
            }
        }

        // Termination: only after iter 0 (we always run iter 0 at
        // least once). If no child's allocation exceeded its clamp
        // limit, every forked child got the full allocation it
        // wanted and the clamping is no longer binding — converged.
        if (clamp_iter > 0 && !any_clamp_fired) break;
    }

    // ── final allocation + N-weighted rollup ───────────────────────
    // After the clamp loop converges, run one more Halley pass on
    // the (now-stable) ChildState (P, Q, depth) to get the final
    // log_n[i] used both for the weight-threshold filter and for the
    // weights themselves. The rollup is a weighted average of -cs.Q
    // (parent-POV) in N-scale (weights = N_i = exp(log_n_i)), which
    // equivalently can be computed in log-space via a logsumexp-style
    // reduction. Two filters apply:
    //   1. Drop Unexpanded children — they only have FPU values, not
    //      real Q estimates, so they shouldn't poison the rollup.
    //   2. Drop children below the log-N threshold
    //      (`log_n_i < depth + weight_log_threshold`, default
    //      log(0.02)) so vanishingly small allocations don't drag
    //      the average around.
    // If everything is filtered out (degenerate node — only
    // Unexpanded children with zero alloc, or all children below
    // threshold), fall back to whatever the TT currently holds for
    // this node and skip the `update_qd` (nothing meaningful to
    // commit).
    for (uint16_t i = 0; i < num_moves; ++i) {
        P_buf[i] = states[i].P;
        Q_buf[i] = states[i].Q;
    }
    AllocResult final_alloc = compute_allocations_halley(
        P_buf.data(), Q_buf.data(), num_moves, depth, algo.c_puct);

    const float weight_floor_log = depth + algo.weight_log_threshold;
    // Numerically-stable weighted average: shift weights by the max
    // log_n among contributing children, sum exp(log_n - max), then
    // the max factor cancels in num/den.
    float max_log_n = -std::numeric_limits<float>::infinity();
    for (uint16_t i = 0; i < num_moves; ++i) {
        const ChildState& cs = states[i];
        if (cs.status == ChildStatus::Unexpanded) continue;
        const float log_n_i = final_alloc.log_n[i];
        if (log_n_i < weight_floor_log) continue;
        if (log_n_i > max_log_n) max_log_n = log_n_i;
    }

    float rolled_q;
    if (std::isfinite(max_log_n)) {
        float num = 0.0f;
        float den = 0.0f;
        for (uint16_t i = 0; i < num_moves; ++i) {
            const ChildState& cs = states[i];
            if (cs.status == ChildStatus::Unexpanded) continue;
            const float log_n_i = final_alloc.log_n[i];
            if (log_n_i < weight_floor_log) continue;
            const float w = std::exp(log_n_i - max_log_n);
            num += w * (-cs.Q);
            den += w;
        }
        rolled_q = num / den;  // den > 0 since max_log_n is finite
        v2::SearchArena::update_qd(entry, rolled_q, depth);
    } else {
        // Degenerate: no contributing children. Fall back to whatever
        // Cell A currently holds (typically the freshly-claimed
        // origQ); skip update_qd.
        rolled_q = v2::unpack_qd(
            v2::SearchArena::load_qd(entry).qd_packed).first;
    }

    co_return QDResult{rolled_q, depth};
};

/**
 * Thin root entry: acquires the very first permit from the per-worker
 * semaphore, then descends into recursive_search. Keeps the
 * recursive_search invariant ("entry owns a permit") clean. Forwards
 * the descent's `std::optional<QDResult>` back to the caller of
 * `lf::sync_wait` (`run_iteration`).
 */
inline constexpr auto root_search =
    [](auto /*self*/,
       RecurseContext* ctx,
       chess::Board board,
       float depth) -> lf::task<std::optional<QDResult>>
{
    lfsync::Permit p = co_await ctx->sem->acquire();
    std::optional<QDResult> result;
    co_await lf::call(&result, recursive_search)(
        ctx, std::move(p), std::move(board), depth);
    co_await lf::join;
    co_return result;
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
     */
    explicit LksSearch(fs::path trt_engine_path,
                       uint64_t lifetime_max_evals = (1ULL << 20),
                       int workers_per_gpu = 1,
                       int coros_per_worker = 32,
                       int max_batch_size = 32)
        : trt_engine_path_(std::move(trt_engine_path))
        , lifetime_max_evals_(lifetime_max_evals)
        , workers_per_gpu_(workers_per_gpu > 0 ? workers_per_gpu : 1)
        , coros_per_worker_(coros_per_worker > 0 ? coros_per_worker : 1)
        , max_batch_size_(max_batch_size > 0 ? max_batch_size : 1)
        , board_(chess::constants::STARTPOS)
    {
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

        arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.5,
                       /*avg_moves_per_node=*/40);
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
                        auto w = std::make_unique<WorkerSearch>();
                        w->device_id = device_id;
                        // 1 worker per pool: per-WorkerSearch isolation
                        // is the model. lazy_pool means the worker thread
                        // sleeps when no work is ready, woken on
                        // pool->schedule(handle) by the GPU thread or
                        // by a release()'d semaphore waiter.
                        w->pool = std::make_unique<lf::lazy_pool>(/*n=*/1);
                        w->evaluator = std::make_unique<BatchEvaluator>(
                            trt_engine_path_, w->pool.get(),
                            max_batch_size_, device_id);
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
        // Rebuild arena in place: two delete[]s on the old buffers, two new[]s
        // on fresh ones. Workers and evaluators are NOT rebuilt.
        arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.5,
                       /*avg_moves_per_node=*/40);
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
        // Reap a previous worker_main that exited but is still joinable.
        if (worker_.joinable()) {
            worker_.join();
        }
        // Capture the algo knobs so post-search inspectors (notably
        // `bestmove()`) can reproduce the same Halley allocation
        // (same c_puct etc.) the workers used during the search.
        last_algo_ = config.algo;
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
     * Pick the root move by argmax of the final allocation
     * `log_n[i]` returned by the Halley solver — same selection rule
     * as `coroutine_search.hpp`'s `compute_allocations(root, ...)` +
     * argmax-of-allocation.
     *
     * Pre-pass classification mirrors `detail::recursive_search`:
     *   - `kTerminalDraw`             -> Q = 0
     *   - `kTerminalLossForChild`     -> Q = -1 (child-POV)
     *   - path-dependent draw         -> Q = 0
     *   - TT-hit (non-terminal child) -> Q from child's qd_packed
     *   - TT-miss (Unexpanded)        -> EXCLUDED from the allocation
     *                                    argmax (matches
     *                                    coroutine_search, which
     *                                    only walks expanded
     *                                    children for selection).
     *
     * The Halley solver runs at the root's recorded `max_depth`
     * (i.e. the deepest budget any worker rolled up at root) using
     * the algo `c_puct` captured at `search()`-time.
     *
     * Falls back to:
     *   - `chess::Move::NO_MOVE` if there are no legal moves;
     *   - `legal[0]` if the root has no TT entry, no MoveInfo, or
     *     a degenerate `num_moves == 0`;
     *   - the highest-P MoveInfo entry if every child is Unexpanded
     *     (nothing got searched, so the prior is the only signal).
     *
     * Safe to call concurrently with a search in flight (uses
     * acquire loads on `qd_packed` / `info_offset`); intended to be
     * called after `quit()`/runners join, when the result is stable.
     */
    [[nodiscard]] chess::Move bestmove() const {
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        if (legal.empty()) return chess::Move::NO_MOVE;

        const v2::TTEntry* root =
            arena_->find(root_key_, v2::secondary_hash(board_));
        if (root == nullptr) return legal[0];

        // A non-null `find` already guarantees Cell B is published with
        // a matching key_secondary (the secondary check is gated on
        // observing the release-store), so a plain `load_info` is
        // sufficient — no spin, no kNoInfoOffset to handle here.
        const v2::InfoCell info = v2::SearchArena::load_info(root);
        assert(info.info_offset != v2::kNoInfoOffset
               && "find returned a non-null entry whose Cell B is "
                  "unpublished; SearchArena invariant broken");

        const v2::NodeInfoHeader* hdr = arena_->info_at(info.info_offset);
        const uint16_t num_moves = hdr->num_moves;
        const v2::MoveInfo* moves = arena_->moves_at(info.info_offset);
        if (num_moves == 0) return legal[0];

        const auto root_qd = v2::unpack_qd(
            v2::SearchArena::load_qd(root).qd_packed);
        const float root_depth = root_qd.second;

        // Build (P[], Q[]) for every classifiable child — i.e.
        // everything except TT-miss "Unexpanded" slots. The
        // `move_buf` parallel array tracks which root MoveInfo slot
        // each (P, Q) entry came from for the post-Halley argmax.
        std::array<float, detail::AllocResult::kMaxChildren> P_buf{};
        std::array<float, detail::AllocResult::kMaxChildren> Q_buf{};
        std::array<chess::Move, detail::AllocResult::kMaxChildren> move_buf{};
        uint16_t active = 0;

        chess::Move highest_P_move = chess::Move{moves[0].move};
        float       highest_P      = moves[0].P();

        for (uint16_t i = 0; i < num_moves; ++i) {
            const v2::MoveInfo& mi = moves[i];
            const chess::Move m{mi.move};
            const v2::TerminalKind tk = mi.terminal_kind();
            const float mi_P = mi.P();

            if (mi_P > highest_P) {
                highest_P = mi_P;
                highest_P_move = m;
            }

            float child_Q;
            bool include = true;

            if (tk == v2::kTerminalDraw) {
                child_Q = 0.0f;
            } else if (tk == v2::kTerminalLossForChild) {
                child_Q = -1.0f;
            } else {
                chess::Board cb = board_;
                cb.makeMove<true>(m);
                if (cb.isRepetition(1) || cb.isHalfMoveDraw()) {
                    child_Q = 0.0f;
                } else {
                    const uint64_t child_key = cb.hash();
                    const uint32_t child_sec = v2::secondary_hash(cb);
                    const v2::TTEntry* ce = arena_->find(child_key, child_sec);
                    if (ce == nullptr) {
                        // Unexpanded at root — exclude from allocation
                        // argmax (matches coroutine_search.hpp).
                        include = false;
                        child_Q = 0.0f;  // unused
                    } else {
                        // Cell A's qd is atomic with the (key, secondary)
                        // match.
                        auto [q, _] = v2::unpack_qd(
                            v2::SearchArena::load_qd(ce).qd_packed);
                        (void)_;
                        child_Q = q;
                    }
                }
            }

            if (include) {
                P_buf[active]    = mi_P;
                Q_buf[active]    = child_Q;
                move_buf[active] = m;
                ++active;
            }
        }

        if (active == 0) {
            // Nothing was searched (every child Unexpanded). The
            // prior is the only signal we have.
            return highest_P_move;
        }

        const detail::AllocResult alloc = detail::compute_allocations_halley(
            P_buf.data(), Q_buf.data(), active, root_depth, last_algo_.c_puct);

        chess::Move best  = move_buf[0];
        float best_log_n  = alloc.log_n[0];
        for (uint16_t i = 1; i < active; ++i) {
            if (alloc.log_n[i] > best_log_n) {
                best_log_n = alloc.log_n[i];
                best       = move_buf[i];
            }
        }
        return best;
    }

private:
    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // Reset per-search counters. Stagger each worker's starting depth.
        const uint64_t per_worker_budget =
            (cfg.max_evals + static_cast<uint64_t>(num_workers_) - 1)
            / static_cast<uint64_t>(num_workers_);
        total_evals_.store(0, std::memory_order_relaxed);
        for (int i = 0; i < num_workers_; ++i) {
            auto& w = *workers_[i];
            w.stop.store(false, std::memory_order_relaxed);
            w.evals.store(0, std::memory_order_relaxed);
            w.tt_claims.store(0, std::memory_order_relaxed);
            w.budget.store(per_worker_budget, std::memory_order_relaxed);
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

        // Aggregator + UCI info loop.
        auto last_info = t0;
        while (true) {
            const bool stop_requested = st.stop_requested();
            bool all_done = true;
            for (const auto& w : workers_) {
                if (w->runner.joinable()
                    && w->evals.load(std::memory_order_relaxed)
                       < w->budget.load(std::memory_order_relaxed)) {
                    all_done = false;
                    break;
                }
            }
            if (stop_requested || all_done) break;

            std::this_thread::sleep_for(std::chrono::milliseconds(5));

            const auto now = Clock::now();
            const auto since_info = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_info).count();
            if (since_info < cfg.min_info_period_ms) continue;

            uint64_t evals_sum = 0;
            uint64_t claims_sum = 0;
            float    min_d = std::numeric_limits<float>::infinity();
            for (const auto& w : workers_) {
                evals_sum  += w->evals.load(std::memory_order_relaxed);
                claims_sum += w->tt_claims.load(std::memory_order_relaxed);
                min_d = std::min(min_d, w->depth.load(std::memory_order_relaxed));
            }
            total_evals_.store(evals_sum, std::memory_order_relaxed);

            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            const long long mss = static_cast<long long>(ms);
            // UCI's `depth` field is integer-valued (cutechess parses via
            // QString::toInt and most other GUIs do the same). Encode our
            // log-scale fractional depth as centi-depth so two ID steps of
            // delta=0.2 advance the field by 20 each.
            const long depth_centi = std::isfinite(min_d)
                ? std::lround(min_d * 100.0f)
                : 0L;
            char buf[224];
            std::snprintf(buf, sizeof(buf),
                "info depth %ld nodes %llu tt_claims %llu time %lld nps %lld",
                depth_centi,
                static_cast<unsigned long long>(evals_sum),
                static_cast<unsigned long long>(claims_sum),
                mss,
                mss > 0 ? static_cast<long long>(evals_sum) * 1000 / mss : 0);
            emit(buf);
            last_info = now;
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
        (void)claims_sum;

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
     */
    void run_worker_search(WorkerSearch& w, int worker_idx,
                           const LksSearchConfig& cfg)
    {
        const float start = cfg.start_depth
                          + (static_cast<float>(worker_idx) / static_cast<float>(num_workers_))
                            * cfg.delta_depth;
        float depth = start;
        w.depth.store(depth, std::memory_order_relaxed);
        while (!w.should_abort() && depth < cfg.max_depth) {
            run_iteration(w, depth, cfg.algo);
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
    void run_iteration(WorkerSearch& w, float depth, const LksAlgoConfig& algo) {
        detail::RecurseContext ctx{
            /*arena=*/&*arena_,
            /*evaluator=*/w.evaluator.get(),
            /*sem=*/w.sem.get(),
            /*w=*/&w,
            /*algo=*/&algo,
        };
        // root_search returns the descent's (Q, depth) as
        // std::optional<QDResult>; nullopt indicates an aborted
        // root-level search (stop fired or budget exhausted before any
        // useful work). The driver loop keys progress off w.depth and
        // the TT, so we don't need the value here — discard it.
        auto result = lf::sync_wait(
            *w.pool, detail::root_search, &ctx, board_, depth);
        (void)result;
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

    // Captured at the start of every `search()` call so post-search
    // inspectors (notably `bestmove()`'s argmax-of-allocation) can
    // re-run the Halley solver with the same knobs the workers used.
    LksAlgoConfig last_algo_{};

    std::optional<v2::SearchArena>             arena_;
    std::vector<std::unique_ptr<WorkerSearch>> workers_;
    std::jthread                               worker_;
    std::atomic<bool>                          running_{false};
    std::atomic<uint64_t>                      total_evals_{0};
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
