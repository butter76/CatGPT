/**
 * LKS search.
 *
 * Lazy-SMP-shaped chess search backed by a TensorRT BatchEvaluator per
 * worker, sharing a lock-free SearchArena.
 *
 * Threading + ownership model:
 *
 *   - `LksSearch` is constructed with a TRT engine path and the desired
 *     fan-out (num_workers, coros_per_worker, max_batch_size). The
 *     constructor builds N persistent `WorkerSearch`es. Each owns:
 *
 *         * a `coro::thread_pool` (1 thread, just resumes coroutines),
 *         * a `BatchEvaluator` (own engine, own CUDA stream, own GPU
 *           thread, own pinned/device buffers),
 *         * a `coro::semaphore` (caps in-flight GPU evals to 4*K).
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
 *     is a single `recursive_search` from the root; intra-iteration
 *     batching comes from `coro::when_all` over children, with the
 *     per-worker eval semaphore gating in-flight GPU evals.
 *
 *   - `quit()` triggers `worker_main`'s stop_token. `worker_main` flips
 *     each worker's `stop` atomic and joins the runners. Persistent
 *     evaluators stay alive (their GPU threads park on empty queues)
 *     and are reused on the next `search()`.
 *
 *   - Only at `LksSearch` destruction do the evaluators' GPU threads
 *     and CUDA resources get torn down.
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
 *     terminal-by-MoveInfo, not a path-repetition, and not TT-cached
 *     at depth. Inside:
 *
 *       1. TT-probe (`find`).
 *       2. If miss: acquire eval-sem permit, then RE-PROBE the TT — a
 *          peer worker may have published the same key while we queued
 *          on the semaphore. On a re-probe hit, release the permit and
 *          adopt the peer's entry. On a re-probe miss, GPU eval,
 *          enumerate legal moves + per-move POSITION-ONLY terminal_kind
 *          detection (no repetitions in terminal_kind — those are
 *          path-dependent), alloc node_info, fill MoveInfo, then
 *          `find_or_claim` and publish. CAS losers orphan their bytes.
 *       3. Re-deepen check: if the entry's max_depth >= depth, return.
 *       4. Pre-pass over children to classify them as Skip /
 *          RecurseThenRead / ReadOnly. RecurseThenRead spawns a child
 *          `recursive_search` task; all are run in parallel via
 *          `coro::when_all`.
 *       5. Rollup: P-weighted average of -child_Q over the same plan
 *          vector (terminal/repetition contribute fixed Q;
 *          RecurseThenRead reads the child's TT entry).
 *       6. `update_qd` with the new (Q, depth).
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
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <coro/semaphore.hpp>
#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>
#include <coro/when_all.hpp>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../selfplay/batch_evaluator.hpp"
#include "../../selfplay/eval_request.hpp"
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../fractional_mcts/v2/board_secondary.hpp"
#include "../fractional_mcts/v2/tt_arena.hpp"

namespace catgpt::lks {

namespace fs = std::filesystem;

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
    int max_evals = 800;            // total budget across all workers
    int min_info_period_ms = 100;   // throttle for aggregated `info` lines

    // Iterative-deepening (log-scale). N = e^depth.
    float start_depth = 0.0f;       // worker 0's starting depth
    float delta_depth = 0.2f;       // per-iteration depth step
    float max_depth   = 32.0f;      // absolute depth cap (e^32 ~= 8e13)

    std::function<void(std::string_view)> on_uci_line;
};

class LksSearch {
public:
    /**
     * @param trt_engine_path     Path to the serialized TensorRT engine.
     *                            Loaded once per worker_search at ctor time.
     * @param lifetime_max_evals  Capacity for the shared SearchArena (TT
     *                            entries reachable across all workers'
     *                            lifetime). Sized once; never grows.
     * @param num_workers         Number of WorkerSearch instances. Each
     *                            owns its own engine + GPU thread + stream
     *                            for natural pipelining across the GPU.
     * @param coros_per_worker    Tuning knob for the eval semaphore: each
     *                            worker permits up to `4 * coros_per_worker`
     *                            in-flight GPU evals concurrently. Drives
     *                            achieved batch sizes.
     * @param max_batch_size      Cap on positions per GPU batch.
     */
    explicit LksSearch(fs::path trt_engine_path,
                       uint64_t lifetime_max_evals = (1ULL << 20),
                       int num_workers = 2,
                       int coros_per_worker = 8,
                       int max_batch_size = 32)
        : trt_engine_path_(std::move(trt_engine_path))
        , lifetime_max_evals_(lifetime_max_evals)
        , num_workers_(num_workers > 0 ? num_workers : 1)
        , coros_per_worker_(coros_per_worker > 0 ? coros_per_worker : 1)
        , max_batch_size_(max_batch_size > 0 ? max_batch_size : 1)
        , board_(chess::constants::STARTPOS)
    {
        arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.5,
                       /*avg_moves_per_node=*/40);
        root_key_ = board_.hash();

        // Persistent workers: build pool + evaluator + eval_sem per worker once.
        const std::ptrdiff_t eval_permits =
            std::min<std::ptrdiff_t>(4 * coros_per_worker_, kMaxEvalSemValue);
        workers_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; ++i) {
            auto w = std::make_unique<WorkerSearch>();
            w->pool = coro::thread_pool::make_shared(coro::thread_pool::options{
                .thread_count = 1,
            });
            w->evaluator = std::make_unique<BatchEvaluator>(
                trt_engine_path_, w->pool, max_batch_size_);
            w->eval_sem = std::make_unique<coro::semaphore<kMaxEvalSemValue>>(eval_permits);
            workers_.push_back(std::move(w));
        }
    }

    ~LksSearch() {
        // Make sure no worker_main outlives us.
        quit();
        // workers_ vector destructs here:
        //   - each WorkerSearch's `runner` jthread is non-joinable
        //     (worker_main joined it before exiting).
        //   - each WorkerSearch's `evaluator` unique_ptr drops →
        //     BatchEvaluator dtor calls shutdown() → GPU thread joined
        //     → CUDA resources freed.
        //   - each WorkerSearch's `pool` shared_ptr drops →
        //     coro::thread_pool dtor joins its worker thread.
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
     * Pick the root move with the best score, where score = -child_Q from
     * our perspective. Mirrors the plan classification used in
     * `recursive_search`: terminal_kind drives fixed Q for terminal
     * children, `isRepetition(1)` is checked path-dependently, and
     * non-terminal children's Q is read from their TT entry.
     *
     * Tiebreak (lex): score, then child max_depth, then prior P.
     *
     * Falls back to:
     *   - `chess::Move::NO_MOVE` if there are no legal moves,
     *   - `legal[0]` if the root has not yet been expanded (TT miss or
     *     `info_offset == kNoInfoOffset`).
     *
     * Children that were never expanded during search (e.g. budget
     * exhausted before they were reached) are scored as "we lose" so
     * any move with TT-evaluated children outranks them; if NO child
     * has a TT entry, the highest-P move wins via the P tiebreak.
     *
     * Safe to call concurrently with a search in flight (uses acquire
     * loads on `qd_packed` / `info_offset`); intended to be called
     * after `quit()`/runners join, when the result is stable.
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
            } else {
                chess::Board cb = board_;
                cb.makeMove<true>(m);
                if (cb.isRepetition(1)) {
                    child_Q = 0.0f;
                } else {
                    const uint64_t child_key = cb.hash();
                    const uint32_t child_sec = v2::secondary_hash(cb);
                    const v2::TTEntry* ce = arena_->find(child_key, child_sec);
                    if (ce == nullptr) {
                        // Unreached during search (or genuine 64-bit
                        // collision masking the entry): score as "we
                        // lose" so any TT-evaluated sibling outranks
                        // it. P tiebreak still picks the highest-prior
                        // unreached move when no children have TT
                        // entries.
                        child_Q     = +1.0f;
                        child_depth = -std::numeric_limits<float>::infinity();
                    } else {
                        // Cell A's qd is atomic with the (key, secondary)
                        // match — no need to wait on Cell B again
                        // (we don't need moveInfo, and find already
                        // verified the secondary).
                        auto [q, d] = v2::unpack_qd(
                            v2::SearchArena::load_qd(ce).qd_packed);
                        child_Q     = q;
                        child_depth = d;
                    }
                }
            }

            const float score = -child_Q;
            const bool better =
                   score > best_score
                || (score == best_score && child_depth > best_depth)
                || (score == best_score && child_depth == best_depth && mi_P > best_P);
            if (better) {
                best_score = score;
                best_depth = child_depth;
                best_P     = mi_P;
                best       = m;
            }
        }
        return best;
    }

private:
    // Compile-time max for the per-worker eval semaphore. Runtime starting
    // value is `4 * coros_per_worker_`, capped here.
    static constexpr std::ptrdiff_t kMaxEvalSemValue = 256;

    /**
     * Per-worker_search state.
     *
     * Persistent across searches:
     *   - pool, evaluator, eval_sem
     *
     * Per-search (reset by worker_main at the start of every search):
     *   - stop, evals, tt_claims, depth, budget
     */
    struct WorkerSearch {
        std::shared_ptr<coro::thread_pool>                  pool;
        std::unique_ptr<BatchEvaluator>                     evaluator;
        std::unique_ptr<coro::semaphore<kMaxEvalSemValue>>  eval_sem;

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

    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // Reset per-search counters. Stagger each worker's starting depth.
        const uint64_t per_worker_budget =
            (static_cast<uint64_t>(cfg.max_evals) + num_workers_ - 1)
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
            char buf[224];
            std::snprintf(buf, sizeof(buf),
                "info depth %.2f nodes %llu tt_claims %llu time %lld nps %lld",
                std::isfinite(min_d) ? min_d : 0.0f,
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
            coro::sync_wait(run_iteration(w, depth));
            depth += cfg.delta_depth;
            w.depth.store(depth, std::memory_order_relaxed);
        }
    }

    /**
     * One ID iteration: schedule onto the worker's pool and descend
     * from the root at the given log-scale depth.
     */
    coro::task<void> run_iteration(WorkerSearch& w, float depth) {
        co_await w.pool->schedule();
        co_await recursive_search(w, board_, depth);
    }

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
    static void softmax_legal_sorted(const RawNNOutput& out,
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
    static float compute_value_variance(
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
     * time. Repetitions are NOT considered terminal here — they are
     * path-dependent and handled at descent time via `isRepetition(1)`.
     *
     * Pre: `child_board` is the board AFTER `makeMove<true>(move)`.
     */
    static v2::TerminalKind classify_terminal(chess::Board& child_board) {
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
        if (child_board.isHalfMoveDraw()) {
            return v2::kTerminalDraw;
        }
        return v2::kTerminalNone;
    }

    /**
     * Recursive search, log-scale, claim-after-eval.
     *
     * Caller-skip protocol: caller has already verified that this
     * position is NOT a path-repetition and is NOT TT-cached at depth.
     * Caller-skip for terminals is enforced at the parent's MoveInfo
     * level (terminal_kind != None children are never recursed into).
     *
     * `board` is taken by value: it lives in this coroutine's frame for
     * the entire descent, so children can capture references to it
     * during the parallel fan-out without lifetime hazards.
     */
    coro::task<void> recursive_search(WorkerSearch& w, chess::Board board, float depth) {
        if (w.should_abort()) co_return;

        const uint64_t key = board.hash();
        const uint32_t sec = v2::secondary_hash(board);
        v2::TTEntry* entry = arena_->find(key, sec);

        if (entry == nullptr) {
            // ── unexpanded: eval, fill bytes, then claim ────────────────
            {
                auto sem_res = co_await w.eval_sem->acquire();
                if (sem_res == coro::semaphore_acquire_result::shutdown) {
                    co_return;
                }
            }

            // After acquire we MUST release the permit on every exit
            // path, including stop/budget aborts that bypass the GPU
            // eval.
            if (w.should_abort()) {
                w.eval_sem->release();
                co_return;
            }

            // Post-acquire re-probe: a peer worker may have evaluated and
            // published this key while we were queued on the semaphore.
            // Adopting their entry skips tokenize + GPU eval + movegen +
            // softmax + alloc + per-move terminal classification, all of
            // which would otherwise be wasted (the find_or_claim CAS would
            // lose and our arena bytes would be permanently orphaned).
            if (v2::TTEntry* e = arena_->find(key, sec); e != nullptr) {
                w.eval_sem->release();
                entry = e;
            } else {
                auto tokens = catgpt::tokenize<BatchEvaluator::SEQ_LENGTH>(
                    board, NO_HALFMOVE_CONFIG);
                RawNNOutput out = co_await EvalAwaitable(*w.evaluator, tokens);
                w.eval_sem->release();
                w.evals.fetch_add(1, std::memory_order_relaxed);

                chess::Movelist legal;
                chess::movegen::legalmoves(legal, board);
                const uint16_t num_moves = static_cast<uint16_t>(legal.size());

                // Sort moves by decreasing P up front (on compact
                // 8-byte pairs) so the arena fill below is a single
                // pass that writes each MoveInfo exactly once in the
                // order descent will consume them.
                std::vector<MoveWithPrior> sorted_moves;
                softmax_legal_sorted(out, board, legal, sorted_moves);

                // alloc + fill BEFORE attempting to claim — these bytes are
                // privately owned until the CAS, and orphaned if the CAS loses.
                const uint64_t off = arena_->alloc_node_info(num_moves);
                v2::MoveInfo* mi = arena_->moves_at(off);
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

                // Per-node variance of the value distribution, in the
                // same [-1, 1] scale as Q. alloc_node_info pre-fills
                // variance=0; overwrite with the real value.
                arena_->info_at(off)->variance =
                    compute_value_variance(out.value_probs);

                const float Q = 2.0f * out.value - 1.0f;

                // Single 128-bit CAS installs (key, qd_packed(Q, 0))
                // atomically, so any reader observing key == K
                // necessarily sees the matching qd. Then a single 128-bit
                // release-store of Cell B publishes
                // (origQ, key_secondary, info_offset).
                auto [ce, claimed] = arena_->find_or_claim(
                    key, sec, Q, /*max_depth=*/0.0f);
                if (claimed) {
                    v2::SearchArena::publish_info(
                        ce, /*origQ=*/Q, /*key_secondary=*/sec, off);
                    w.tt_claims.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Lost the race despite the post-acquire re-probe:
                    // someone published in the gap between our re-probe
                    // and our find_or_claim. Our `off` bytes are orphaned
                    // (forever).
                }

                // Intentionally let the thread finish putting the entry
                // into the TT/arena before aborting
                if (w.should_abort()) co_return;

                entry = ce;
            }
        }

        // ── re-deepen check (works for both fresh-claim and TT-shared) ──
        {
            // Cell A is atomic with the key match (find / find_or_claim
            // both ensure we observe a key from a successful CAS), so qd
            // is never torn here.
            auto [_, cur_max_d] = v2::unpack_qd(
                v2::SearchArena::load_qd(entry).qd_packed);
            (void)_;
            if (depth <= cur_max_d) co_return;
        }

        // ── pre-pass: classify each child for fan-out + rollup ──────────
        // Cell B is provably published by the time we reach this point:
        //   * `find(key, sec)` only returns non-null after validating
        //     the secondary, which requires Cell B to be published
        //     (the validation acquire-loads it).
        //   * On the fresh-claim path we called `publish_info`
        //     ourselves before assigning `entry`.
        //   * On the post-acquire-re-probe path the same `find(key, sec)`
        //     guarantee applies.
        // So a plain `load_info` is sufficient — no spin needed.
        const v2::InfoCell info_cell = v2::SearchArena::load_info(entry);
        assert(info_cell.info_offset != v2::kNoInfoOffset
               && "Cell B unpublished after find/find_or_claim returned a "
                  "validated entry; invariant broken");
        const v2::NodeInfoHeader* hdr = arena_->info_at(info_cell.info_offset);
        const uint16_t num_moves = hdr->num_moves;
        const v2::MoveInfo* moves = arena_->moves_at(info_cell.info_offset);

        enum class Mode : uint8_t { Skip, RecurseThenRead, ReadOnly };
        struct Plan {
            Mode     mode;
            float    P;
            float    fixed_Q;     // valid for ReadOnly
            uint64_t child_key;   // valid for RecurseThenRead
            uint32_t child_sec;   // valid for RecurseThenRead (cached so
                                  // the post-fan-out rollup re-find can
                                  // run without re-deriving the board)
        };
        std::vector<Plan> plans;
        plans.reserve(num_moves);

        std::vector<coro::task<void>> child_tasks;
        child_tasks.reserve(num_moves);

        for (uint16_t i = 0; i < num_moves; ++i) {
            const auto& m = moves[i];
            const v2::TerminalKind m_tk = m.terminal_kind();
            const float m_P = m.P();

            // Depth floor — applies to EVERY child regardless of kind.
            // With child_depth = depth + log(P), each level of recursion
            // strictly decreases depth. Without a floor at 0 the recursion
            // would run forever as priors compound below e^0 = 1 visit.
            // Cuts terminal_kind contributions too: a terminal child below
            // the floor doesn't matter at this iteration's resolution.
            const float child_depth = (m_P > 0.0f)
                ? depth + std::log(std::max(m_P, 1e-9f))
                : -std::numeric_limits<float>::infinity();
            if (child_depth < 0.0f) {
                plans.push_back({Mode::Skip, m_P, 0.0f, 0, 0});
                continue;
            }

            if (m_tk == v2::kTerminalDraw) {
                plans.push_back({Mode::ReadOnly, m_P, /*Q=*/0.0f, 0, 0});
                continue;
            }
            if (m_tk == v2::kTerminalLossForChild) {
                plans.push_back({Mode::ReadOnly, m_P, /*Q=*/-1.0f, 0, 0});
                continue;
            }

            chess::Board cb = board;
            cb.makeMove<true>(chess::Move{m.move});

            // Path-dependent: a 2-fold along this path is NOT in
            // terminal_kind (different paths hashing to the same key
            // may not be repetitions). Treat as a draw at this caller.
            if (cb.isRepetition(1)) {
                plans.push_back({Mode::ReadOnly, m_P, /*Q=*/0.0f, 0, 0});
                continue;
            }

            const uint64_t child_key = cb.hash();
            const uint32_t child_sec = v2::secondary_hash(cb);

            if (v2::TTEntry* ce = arena_->find(child_key, child_sec)) {
                // (key, secondary) match verified by find — find only
                // returns non-null after `secondary_matches` observed
                // Cell B published with a matching key_secondary, so
                // the child entry is fully published. Cell A's qd is
                // atomic with the primary match; we never spin on the
                // child's moveInfo from here, and we don't need it for
                // the ReadOnly path (just q + max_depth).
                auto [q, child_max_d] = v2::unpack_qd(
                    v2::SearchArena::load_qd(ce).qd_packed);
                if (child_depth <= child_max_d) {
                    plans.push_back({Mode::ReadOnly, m_P, q, child_key, child_sec});
                    continue;
                }
            }

            plans.push_back({Mode::RecurseThenRead, m_P, 0.0f, child_key, child_sec});
            child_tasks.emplace_back(
                recursive_search(w, std::move(cb), child_depth));
        }

        if (!child_tasks.empty()) {
            co_await coro::when_all(std::move(child_tasks));
        }

        if (w.should_abort()) co_return;

        // ── rollup: P-weighted average of -child_Q ──────────────────────
        float num = 0.0f;
        float den = 0.0f;
        for (const Plan& p : plans) {
            if (p.mode == Mode::Skip) continue;
            float child_Q;
            if (p.mode == Mode::ReadOnly) {
                child_Q = p.fixed_Q;
            } else {
                v2::TTEntry* ce = arena_->find(p.child_key, p.child_sec);
                if (!ce) continue;       // child never claimed (e.g. stop fired)
                auto [q, _] = v2::unpack_qd(
                    v2::SearchArena::load_qd(ce).qd_packed);
                (void)_;
                child_Q = q;
            }
            num += p.P * (-child_Q);
            den += p.P;
        }
        if (den > 0.0f) {
            v2::SearchArena::update_qd(entry, num / den, depth);
        }
    }

    fs::path trt_engine_path_;
    uint64_t lifetime_max_evals_;
    int      num_workers_;
    int      coros_per_worker_;
    int      max_batch_size_;

    chess::Board board_;
    uint64_t     root_key_ = 0;
    bool         searched_since_reset_ = false;

    std::optional<v2::SearchArena>             arena_;
    std::vector<std::unique_ptr<WorkerSearch>> workers_;
    std::jthread                               worker_;
    std::atomic<bool>                          running_{false};
    std::atomic<uint64_t>                      total_evals_{0};
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
