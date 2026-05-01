/**
 * LKS search.
 *
 * `LksSearch` is the next-generation, Lazy-SMP-shaped search class.
 *
 * Threading model:
 *
 *   - `search(cfg)` spawns the `worker_main` jthread and returns
 *     immediately. `worker_main` owns the UCI emission contract and is the
 *     ONLY thread that calls `cfg.on_uci_line`.
 *
 *   - `worker_main` spawns N `WorkerSearch` jthreads (`num_workers`
 *     parameter). Each WorkerSearch is otherwise independent: its own
 *     thread, its own stop flag, its own per-worker stats. They share
 *     exactly one thing: the lock-free `SearchArena` (TT + bump arena).
 *
 *   - `quit()` triggers worker_main's stop_token. worker_main fans the
 *     stop out to every WorkerSearch and joins them, then emits a final
 *     `bestmove` line and exits. After `quit()` returns,
 *     `is_searching()` is false.
 *
 * This milestone:
 *
 *   - WorkerSearch loops simulate "fake search work": sleep briefly,
 *     synthesize a key, eval-first batch-allocate a NodeInfo, claim the
 *     TT slot via `find_or_claim`, publish. No real algorithm, no GPU.
 *   - The point is to exercise multi-worker stop propagation, periodic
 *     aggregated UCI info emission from worker_main, and concurrent
 *     lock-free TT writes across workers.
 *
 * State carried by `LksSearch`:
 *
 *   - `chess::Board board_`        — current root position.
 *   - `SearchArena arena_`         — TT + NodeInfo bump arena. Sized once
 *                                    at construction; preserved across
 *                                    `makemove(...)` for tree reuse.
 *   - `bool searched_since_reset_` — if true, `setBoard(b)` calls
 *                                    `reset()` first.
 *   - `int num_workers_`           — number of WorkerSearch threads to
 *                                    spawn per search.
 */

#ifndef CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
#define CATGPT_ENGINE_LKS_LKS_SEARCH_HPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
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

#include "../../../external/chess-library/include/chess.hpp"
#include "../fractional_mcts/v2/tt_arena.hpp"

namespace catgpt::lks {

/**
 * Per-search configuration.
 *
 * `on_uci_line` is invoked from `worker_main`, one call per UCI line.
 * The string_view is valid only for the duration of the call; the worker
 * does not retain it. For UCI production install:
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
     * @param lifetime_max_evals  Capacity for the shared SearchArena (TT
     *                            entries reachable across all workers'
     *                            lifetime). Sized once; never grows.
     * @param num_workers         Number of WorkerSearch threads spawned
     *                            per search() call. Defaults to 2 to
     *                            illustrate the Lazy-SMP shape; tune to
     *                            your hardware.
     */
    explicit LksSearch(uint64_t lifetime_max_evals = (1ULL << 20),
                       int num_workers = 2)
        : lifetime_max_evals_(lifetime_max_evals)
        , num_workers_(num_workers > 0 ? num_workers : 1)
        , board_(chess::constants::STARTPOS)
    {
        arena_.emplace(lifetime_max_evals_, /*load_factor=*/0.5,
                       /*avg_moves_per_node=*/40);
        root_key_ = board_.hash();
    }

    ~LksSearch() {
        // Make sure no worker outlives us.
        quit();
    }

    LksSearch(const LksSearch&) = delete;
    LksSearch& operator=(const LksSearch&) = delete;
    LksSearch(LksSearch&&) = delete;
    LksSearch& operator=(LksSearch&&) = delete;

    // ── Synchronous lifecycle (must not be called while a search runs) ──

    void reset() {
        assert(!is_searching() && "reset() called while a search is in flight");
        // Rebuild arena in place: two delete[]s on the old buffers, two new[]s
        // on fresh ones. This is the only allocation point in the API.
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
        // Tree reuse: the arena and TT are preserved. The new root is just
        // whatever zobrist `board_` lands on after this move.
        board_.makeMove<true>(move);
        root_key_ = board_.hash();
    }

    // ── Asynchronous search ─────────────────────────────────────────────

    /**
     * Launches the worker_main jthread (which in turn spawns N
     * WorkerSearches) and returns immediately. Throws std::logic_error
     * if a search is already in flight.
     */
    void search(LksSearchConfig config) {
        if (is_searching()) {
            throw std::logic_error("LksSearch::search called while a search is already in flight");
        }
        // If a previous worker is joinable but already exited (running_=false),
        // join it now so we can replace it cleanly.
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
     * WorkerSearches before exiting). Idempotent. Safe to call from a
     * different thread than `search()`.
     */
    void quit() {
        if (!worker_.joinable()) return;
        worker_.request_stop();
        worker_.join();
        // `running_` was already cleared by worker_main before exit.
    }

    [[nodiscard]] bool is_searching() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const chess::Board& board() const noexcept { return board_; }
    [[nodiscard]] uint64_t root_key() const noexcept { return root_key_; }
    [[nodiscard]] const v2::SearchArena& arena() const noexcept { return *arena_; }

    /**
     * Aggregated count of fake evals performed across all workers in the
     * most recent (or in-flight) search. Cleared at the start of each
     * search().
     */
    [[nodiscard]] uint64_t total_evals() const noexcept {
        return total_evals_.load(std::memory_order_relaxed);
    }

private:
    /**
     * Per-worker_search state. Non-movable (jthread + atomics + back-ptr).
     * Owned by worker_main via `std::vector<std::unique_ptr<WorkerSearch>>`.
     */
    struct WorkerSearch {
        std::atomic<bool>     stop{false};
        std::atomic<uint64_t> evals{0};
        std::atomic<uint64_t> tt_claims{0};
        std::jthread          runner;

        WorkerSearch() = default;
        WorkerSearch(const WorkerSearch&) = delete;
        WorkerSearch& operator=(const WorkerSearch&) = delete;
        WorkerSearch(WorkerSearch&&) = delete;
        WorkerSearch& operator=(WorkerSearch&&) = delete;
    };

    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // Reset aggregate counters at the start of every search.
        total_evals_.store(0, std::memory_order_relaxed);

        // Pick a placeholder best-move from the legal list so the final
        // `bestmove` line is well-formed.
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        chess::Move best = legal.empty() ? chess::Move::NO_MOVE : legal[0];

        // Spawn N worker_searches. Each owns its own jthread and gets
        // a worker-local rng seed derived from its index.
        std::vector<std::unique_ptr<WorkerSearch>> workers;
        workers.reserve(num_workers_);
        const uint64_t per_worker_budget =
            (cfg.max_evals + num_workers_ - 1) / num_workers_;
        const uint64_t root_seed = root_key_;
        for (int i = 0; i < num_workers_; ++i) {
            workers.push_back(std::make_unique<WorkerSearch>());
            auto* w = workers.back().get();
            w->runner = std::jthread([this, w, i, root_seed, per_worker_budget]() {
                worker_search_loop(*w, i, root_seed, per_worker_budget);
            });
        }

        // worker_main loop: aggregate stats, emit periodic info lines,
        // and watch for stop. We also watch for "all workers finished
        // their per-worker budget" (each worker_search exits cleanly
        // when it hits per_worker_budget evals).
        auto last_info = t0;
        while (true) {
            const bool stop_requested = st.stop_requested();
            // Are all workers done?
            bool all_done = true;
            for (const auto& w : workers) {
                if (w->runner.joinable() && w->evals.load(std::memory_order_relaxed)
                                            < per_worker_budget) {
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

            // Aggregate stats across workers.
            uint64_t evals_sum = 0;
            uint64_t claims_sum = 0;
            for (const auto& w : workers) {
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

        // Stop fan-out. Even if we exited the loop because all workers
        // already finished, flipping the flag is harmless (idempotent).
        for (auto& w : workers) {
            w->stop.store(true, std::memory_order_release);
        }
        for (auto& w : workers) {
            if (w->runner.joinable()) w->runner.join();
        }

        // Final aggregate snapshot.
        uint64_t evals_sum = 0;
        uint64_t claims_sum = 0;
        for (const auto& w : workers) {
            evals_sum += w->evals.load(std::memory_order_relaxed);
            claims_sum += w->tt_claims.load(std::memory_order_relaxed);
        }
        total_evals_.store(evals_sum, std::memory_order_relaxed);

        // Always emit a bestmove line, even if we were stopped early.
        char bm[64];
        if (best != chess::Move::NO_MOVE) {
            const std::string uci_move = chess::uci::moveToUci(best);
            std::snprintf(bm, sizeof(bm), "bestmove %s", uci_move.c_str());
        } else {
            std::snprintf(bm, sizeof(bm), "bestmove 0000");
        }
        emit(bm);

        // Flip running_ false *after* the bestmove callback returns so
        // callers can rely on: `!is_searching()` implies bestmove already
        // emitted.
        running_.store(false, std::memory_order_release);
    }

    /**
     * Single WorkerSearch's main loop. Synthesises fake "search work" so
     * we have something concrete to stop, count, and write to the TT.
     *
     * Each iteration:
     *   1. Check stop flag; bail if set.
     *   2. Sleep ~1ms (simulates: descent + GPU eval + post-process).
     *   3. Eval-first batch-allocate against the shared SearchArena.
     *   4. find_or_claim a synthetic key derived from {root, worker_idx,
     *      iter}. Publish on win.
     *   5. Increment per-worker counters.
     */
    void worker_search_loop(WorkerSearch& w,
                            int worker_idx,
                            uint64_t root_seed,
                            uint64_t per_worker_budget) {
        // Per-worker rng so each worker hits a (with overwhelming
        // probability) disjoint key stream. `root_seed` ties the stream to
        // the current root for determinism within one search.
        uint64_t state = root_seed
                       ^ (uint64_t(worker_idx + 1) * 0x9E3779B97F4A7C15ULL);

        while (!w.stop.load(std::memory_order_relaxed)
               && w.evals.load(std::memory_order_relaxed) < per_worker_budget) {
            // Simulate: descent + GPU eval + post-process.
            std::this_thread::sleep_for(std::chrono::microseconds(1000));

            // Synthetic key.
            state += 0x9E3779B97F4A7C15ULL;
            uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            uint64_t key = z ^ (z >> 31);

            // Eval-first batch-allocate workflow against the shared TT.
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
                v2::SearchArena::set_initial_qn(entry, /*Q=*/0.0f, /*max_N=*/1.0f);
                v2::SearchArena::publish_info(entry, off);
                w.tt_claims.fetch_add(1, std::memory_order_relaxed);
            }
            w.evals.fetch_add(1, std::memory_order_relaxed);
        }
    }

    uint64_t lifetime_max_evals_;
    int      num_workers_;
    chess::Board board_;
    uint64_t root_key_ = 0;
    bool     searched_since_reset_ = false;

    std::optional<v2::SearchArena> arena_;
    std::jthread                   worker_;
    std::atomic<bool>              running_{false};
    std::atomic<uint64_t>          total_evals_{0};
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
