/**
 * LKS search — skeleton.
 *
 * `LksSearch` is the next-generation search class. This milestone is
 * API-only: it exercises the lifecycle (reset / setBoard / makemove /
 * search / quit) and locks down the threading + callback contract before
 * the real algorithm is ported in.
 *
 * Threading model:
 *
 *   - `search(cfg)` spawns a `std::jthread` and returns immediately.
 *   - The worker thread invokes `cfg.on_uci_line(line)` once per UCI line
 *     it wants to emit (lines have no trailing newline; the consumer is
 *     responsible for terminating them).
 *   - `quit()` sets the worker's stop_token and joins. After `quit()`
 *     returns, the worker has already emitted its final `bestmove` line
 *     and `is_searching()` is false.
 *
 * The point of the callback is that the UCI main thread is never blocked:
 * it stays parked on stdin, and the worker thread writes lines via the
 * callback (which, in production, just does `cout << ... << '\n'`).
 *
 * State carried by `LksSearch`:
 *
 *   - `chess::Board board_`        — current root position.
 *   - `SearchArena arena_`         — TT + NodeInfo bump arena. Sized once
 *                                    at construction; preserved across
 *                                    `makemove(...)` for tree reuse.
 *   - `bool searched_since_reset_` — if true, `setBoard(b)` calls
 *                                    `reset()` first.
 */

#ifndef CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
#define CATGPT_ENGINE_LKS_LKS_SEARCH_HPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "../../../external/chess-library/include/chess.hpp"
#include "../fractional_mcts/v2/tt_arena.hpp"

namespace catgpt::lks {

/**
 * Per-search configuration.
 *
 * `on_uci_line` is invoked from the worker thread, one call per UCI line.
 * The string_view is valid only for the duration of the call; the worker
 * does not retain it. For UCI production install:
 *   `[](std::string_view s){ std::cout << s << '\n'; std::cout.flush(); }`
 * For tests install a recording lambda (mind the move-out semantics — pass
 * a callable that captures by reference into a vector you own).
 */
struct LksSearchConfig {
    int max_evals = 800;            // budget for this search
    int min_info_period_ms = 100;   // throttle for `info` lines (skeleton)

    std::function<void(std::string_view)> on_uci_line;
};

class LksSearch {
public:
    explicit LksSearch(uint64_t lifetime_max_evals = (1ULL << 20))
        : lifetime_max_evals_(lifetime_max_evals)
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
     * Launches the search worker and returns immediately. Throws
     * std::logic_error if a search is already in flight.
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
     * Stops any in-flight search and joins the worker thread. Idempotent
     * and safe to call from a thread other than the one that called
     * `search()`. Returns once the worker has emitted its final
     * `bestmove` line and exited.
     */
    void quit() {
        if (!worker_.joinable()) return;
        worker_.request_stop();
        worker_.join();
        // `running_` was already cleared by the worker before exit.
    }

    [[nodiscard]] bool is_searching() const noexcept {
        return running_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const chess::Board& board() const noexcept { return board_; }

    [[nodiscard]] uint64_t root_key() const noexcept { return root_key_; }

    [[nodiscard]] const v2::SearchArena& arena() const noexcept { return *arena_; }

private:
    void worker_main(std::stop_token st, LksSearchConfig cfg) {
        auto& cb = cfg.on_uci_line;
        auto emit = [&](std::string_view s) { if (cb) cb(s); };

        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();

        // Pick a placeholder best-move from the legal list so the final
        // `bestmove` line is well-formed.
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, board_);
        chess::Move best = legal.empty() ? chess::Move::NO_MOVE : legal[0];

        int evals = 0;
        int depth = 0;
        auto last_info = t0;

        while (!st.stop_requested() && evals < cfg.max_evals) {
            // Pretend a "step" runs here. The sleep gives quit() something
            // to race against in tests.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ++evals;
            if (evals % 32 == 0) ++depth;

            const auto now = Clock::now();
            const auto since_info = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_info).count();
            if (since_info < cfg.min_info_period_ms) continue;

            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - t0).count();
            char buf[160];
            const long long mss = static_cast<long long>(ms);
            std::snprintf(buf, sizeof(buf),
                "info depth %d nodes %d time %lld nps %lld",
                depth, evals, mss,
                mss > 0 ? static_cast<long long>(evals) * 1000 / mss : 0);
            emit(buf);
            last_info = now;
        }

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

    uint64_t lifetime_max_evals_;
    chess::Board board_;
    uint64_t root_key_ = 0;
    bool searched_since_reset_ = false;

    std::optional<v2::SearchArena> arena_;
    std::jthread worker_;
    std::atomic<bool> running_{false};
};

}  // namespace catgpt::lks

#endif  // CATGPT_ENGINE_LKS_LKS_SEARCH_HPP
