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
 *         * a `coro::thread_pool` (1 thread, just resumes coroutines), and
 *         * a `BatchEvaluator` (own engine, own CUDA stream, own GPU
 *           thread, own pinned/device buffers).
 *
 *     The TRT engine load and CUDA buffer allocation happen ONCE here.
 *     They live until `LksSearch` is destroyed.
 *
 *   - `search(cfg)` spawns the `worker_main` jthread and returns
 *     immediately. `worker_main` resets the per-search atomics, spawns
 *     N short-lived `runner` jthreads (one per worker_search), then
 *     loops on aggregate stats + UCI emission.
 *
 *   - Each `runner` does `coro::sync_wait(coro::when_all(K coros))`
 *     where each coro is a `descent_coro` that schedules itself onto the
 *     worker's pool, picks a synthetic position, `co_await`s an
 *     `EvalAwaitable` against the worker's evaluator, and writes the
 *     result into the shared `SearchArena`.
 *
 *   - `quit()` triggers `worker_main`'s stop_token. `worker_main` flips
 *     each worker's `stop` atomic and joins the runners. Runners join
 *     once their K coros all see the stop and exit. The persistent
 *     evaluators stay alive (their GPU threads park on empty queues)
 *     and are reused on the next `search()`.
 *
 *   - Only at `LksSearch` destruction do the evaluators' GPU threads
 *     and CUDA resources get torn down.
 */

#ifndef CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
#define CATGPT_ENGINE_LKS_LKS_SEARCH_HPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>
#include <coro/when_all.hpp>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../selfplay/batch_evaluator.hpp"
#include "../../selfplay/eval_request.hpp"
#include "../../tokenizer.hpp"
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
     * @param coros_per_worker    Parallel descent coroutines per worker.
     *                            Drives batch sizes: K coros suspended
     *                            → up to K positions per GPU batch.
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

        // Persistent workers: build pool + evaluator per worker once.
        workers_.reserve(num_workers_);
        for (int i = 0; i < num_workers_; ++i) {
            auto w = std::make_unique<WorkerSearch>();
            w->pool = coro::thread_pool::make_shared(coro::thread_pool::options{
                .thread_count = 1,
            });
            w->evaluator = std::make_unique<BatchEvaluator>(
                trt_engine_path_, w->pool, max_batch_size_);
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

private:
    /**
     * Per-worker_search state.
     *
     * Persistent across searches:
     *   - pool, evaluator, runner-spawn slot
     *
     * Per-search (reset by worker_main at the start of every search):
     *   - stop, evals, tt_claims
     */
    struct WorkerSearch {
        std::shared_ptr<coro::thread_pool> pool;
        std::unique_ptr<BatchEvaluator>    evaluator;

        std::atomic<bool>     stop{false};
        std::atomic<uint64_t> evals{0};
        std::atomic<uint64_t> tt_claims{0};

        std::jthread runner;

        WorkerSearch() = default;
        WorkerSearch(const WorkerSearch&) = delete;
        WorkerSearch& operator=(const WorkerSearch&) = delete;
        WorkerSearch(WorkerSearch&&) = delete;
        WorkerSearch& operator=(WorkerSearch&&) = delete;
    };

    static uint64_t splitmix64(uint64_t& x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // Reset per-search counters.
        total_evals_.store(0, std::memory_order_relaxed);
        for (auto& w : workers_) {
            w->stop.store(false, std::memory_order_relaxed);
            w->evals.store(0, std::memory_order_relaxed);
            w->tt_claims.store(0, std::memory_order_relaxed);
        }

        // Pick a placeholder best-move so the final `bestmove` line is well-formed.
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        chess::Move best = legal.empty() ? chess::Move::NO_MOVE : legal[0];

        const uint64_t per_worker_budget =
            (static_cast<uint64_t>(cfg.max_evals) + num_workers_ - 1)
            / static_cast<uint64_t>(num_workers_);

        // Spawn per-search runner jthreads against the persistent workers.
        for (int i = 0; i < num_workers_; ++i) {
            auto* w = workers_[i].get();
            w->runner = std::jthread([this, w, i, per_worker_budget]() {
                run_worker_search(*w, i, per_worker_budget);
            });
        }

        // Aggregator + UCI info loop.
        auto last_info = t0;
        while (true) {
            const bool stop_requested = st.stop_requested();
            bool all_done = true;
            for (const auto& w : workers_) {
                if (w->runner.joinable()
                    && w->evals.load(std::memory_order_relaxed) < per_worker_budget) {
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
            for (const auto& w : workers_) {
                evals_sum += w->evals.load(std::memory_order_relaxed);
                claims_sum += w->tt_claims.load(std::memory_order_relaxed);
            }
            total_evals_.store(evals_sum, std::memory_order_relaxed);

            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            const long long mss = static_cast<long long>(ms);
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "info nodes %llu tt_claims %llu time %lld nps %lld",
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
            evals_sum += w->evals.load(std::memory_order_relaxed);
            claims_sum += w->tt_claims.load(std::memory_order_relaxed);
        }
        total_evals_.store(evals_sum, std::memory_order_relaxed);
        (void)claims_sum;

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
     * Builds K descent coros and blocks until they all complete.
     */
    void run_worker_search(WorkerSearch& w, int worker_idx, uint64_t per_worker_budget) {
        std::vector<coro::task<void>> tasks;
        tasks.reserve(coros_per_worker_);
        for (int i = 0; i < coros_per_worker_; ++i) {
            tasks.emplace_back(descent_coro(w, worker_idx, i, per_worker_budget));
        }
        coro::sync_wait(coro::when_all(std::move(tasks)));
    }

    /**
     * One descent coroutine. Runs entirely on `w.pool`. Each iteration:
     *
     *   1. Walk a few random legal moves from `board_` to make a synthetic
     *      reachable position.
     *   2. Tokenize.
     *   3. co_await EvalAwaitable on the worker's BatchEvaluator. (This
     *      is the suspension point that lets K coros queue up batched
     *      requests.)
     *   4. Eval-first batch-allocate against the shared SearchArena;
     *      claim the TT slot for this position; publish.
     *
     * No real algorithm yet — this is the plumbing test.
     */
    coro::task<void> descent_coro(WorkerSearch& w,
                                  int worker_idx,
                                  int coro_id,
                                  uint64_t per_worker_budget) {
        co_await w.pool->schedule();

        uint64_t rng = root_key_
                     ^ (uint64_t(worker_idx + 1) * 0x9E3779B97F4A7C15ULL)
                     ^ (uint64_t(coro_id + 1)   * 0xBF58476D1CE4E5B9ULL);

        chess::Board scratch;
        while (!w.stop.load(std::memory_order_relaxed)
               && w.evals.load(std::memory_order_relaxed) < per_worker_budget) {
            // Synthesise a random reachable position from the root.
            scratch = board_;
            for (int d = 0; d < 4; ++d) {
                chess::Movelist legal;
                chess::movegen::legalmoves(legal, scratch);
                if (legal.empty()) break;
                uint64_t r = splitmix64(rng);
                scratch.makeMove<true>(legal[r % legal.size()]);
            }

            auto tokens = catgpt::tokenize<BatchEvaluator::SEQ_LENGTH>(
                scratch, NO_HALFMOVE_CONFIG);

            // Suspension point. The K-1 other coros may be in various
            // stages here; the GPU thread picks them all up as one batch.
            RawNNOutput out = co_await EvalAwaitable(*w.evaluator, tokens);

            // Stop fast-path check after resume.
            if (w.stop.load(std::memory_order_relaxed)) break;

            const uint64_t key = scratch.hash();
            constexpr uint16_t kNumMoves = 30;
            const uint64_t off = arena_->alloc_node_info(kNumMoves);
            v2::MoveInfo* moves = arena_->moves_at(off);
            for (uint16_t j = 0; j < kNumMoves; ++j) {
                moves[j].move = static_cast<uint16_t>(j);
                moves[j].terminal_kind = 0;
                moves[j]._pad = 0;
                moves[j].P = 1.0f / kNumMoves;
                moves[j].P_alloc = moves[j].P;
                moves[j].P_optimistic = moves[j].P;
            }

            auto [entry, claimed] = arena_->find_or_claim(key);
            if (claimed) {
                const float Q = 2.0f * out.value - 1.0f;
                v2::SearchArena::set_initial_qn(entry, Q, /*max_N=*/1.0f);
                v2::SearchArena::publish_info(entry, off);
                w.tt_claims.fetch_add(1, std::memory_order_relaxed);
            }
            w.evals.fetch_add(1, std::memory_order_relaxed);
        }
        co_return;
    }

    fs::path trt_engine_path_;
    uint64_t lifetime_max_evals_;
    int      num_workers_;
    int      coros_per_worker_;
    int      max_batch_size_;

    chess::Board board_;
    uint64_t     root_key_ = 0;
    bool         searched_since_reset_ = false;

    std::optional<v2::SearchArena>            arena_;
    std::vector<std::unique_ptr<WorkerSearch>> workers_;
    std::jthread                              worker_;
    std::atomic<bool>                         running_{false};
    std::atomic<uint64_t>                     total_evals_{0};
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
