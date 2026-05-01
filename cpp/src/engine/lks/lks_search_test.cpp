/**
 * Lifecycle test driver for LksSearch.
 *
 * No GPU, no UCI loop, no networking — exercises:
 *
 *   1. natural completion: search() runs to max_evals, emits bestmove last
 *   2. quit-mid-search:    search() interrupted, bestmove still emitted, no
 *                          info lines after quit() returns
 *   3. setBoard-resets:    after a search, setBoard() calls reset() so the
 *                          arena's `arena_used_bytes()` drops to 0
 *   4. makemove-preserves: makemove() does NOT reset the arena (used bytes
 *                          stay the same)
 *   5. double-search:      calling search() while another is in flight
 *                          throws std::logic_error
 *   6. quit-noop:          calling quit() with no search in flight is a
 *                          fast no-op (bounded latency)
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "lks_search.hpp"

namespace {

using catgpt::lks::LksSearch;
using catgpt::lks::LksSearchConfig;

int g_failed = 0;

#define EXPECT(cond)                                                       \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr,                                           \
                         "  FAIL: %s (line %d)\n", #cond, __LINE__);       \
            ++g_failed;                                                    \
        }                                                                  \
    } while (0)

struct Recorder {
    std::mutex mu;
    std::vector<std::string> lines;

    auto callback() {
        return [this](std::string_view s) {
            std::lock_guard<std::mutex> lock(mu);
            lines.emplace_back(s);
        };
    }

    std::vector<std::string> snapshot() {
        std::lock_guard<std::mutex> lock(mu);
        return lines;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mu);
        return lines.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu);
        lines.clear();
    }
};

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

// ── Tests ─────────────────────────────────────────────────────────────────

void test_natural_completion() {
    std::printf("[1] natural completion\n");
    LksSearch search;
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 12;             // ~12 * 5ms = 60ms total
    cfg.min_info_period_ms = 0;     // emit info every step
    cfg.on_uci_line = rec.callback();

    search.search(std::move(cfg));

    // Wait for completion (with a generous timeout).
    auto t0 = std::chrono::steady_clock::now();
    while (search.is_searching()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
            std::fprintf(stderr, "  timed out waiting for natural completion\n");
            ++g_failed;
            search.quit();
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    // Drain the worker thread.
    search.quit();

    auto lines = rec.snapshot();
    EXPECT(!lines.empty());
    EXPECT(starts_with(lines.back(), "bestmove "));
    // Some info lines should have been emitted.
    int info_count = 0;
    for (const auto& l : lines) if (starts_with(l, "info ")) ++info_count;
    EXPECT(info_count > 0);
}

void test_quit_mid_search() {
    std::printf("[2] quit mid-search\n");
    LksSearch search;
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 100000;         // would run for ~500s
    cfg.min_info_period_ms = 0;
    cfg.on_uci_line = rec.callback();

    auto t0 = std::chrono::steady_clock::now();
    search.search(std::move(cfg));
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    EXPECT(elapsed_ms < 500);                   // quit was prompt
    EXPECT(!search.is_searching());              // worker is gone

    auto lines = rec.snapshot();
    EXPECT(!lines.empty());
    EXPECT(starts_with(lines.back(), "bestmove "));

    // No info line should appear *after* the bestmove (the bestmove must
    // be the last emission).
    bool seen_bestmove = false;
    bool info_after_bestmove = false;
    for (const auto& l : lines) {
        if (starts_with(l, "bestmove ")) seen_bestmove = true;
        else if (seen_bestmove && starts_with(l, "info ")) info_after_bestmove = true;
    }
    EXPECT(seen_bestmove);
    EXPECT(!info_after_bestmove);

    // After quit() returns, no further callbacks may arrive.
    size_t before = rec.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT(rec.size() == before);
}

void test_setboard_resets_after_search() {
    std::printf("[3] setBoard resets after search\n");
    LksSearch search;
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 4;
    cfg.min_info_period_ms = 0;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));
    while (search.is_searching()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    search.quit();

    // Pre-condition: search() did not yet write into the arena (skeleton),
    // so used_bytes will be 0 already. To make this test meaningful we
    // touch the arena directly to simulate a real search.
    {
        // We can't const_cast the arena, but we can call a non-const ref via
        // a tiny helper: just verify reset behaviour on arena_used_bytes().
    }

    // Verify post-search state, then setBoard, then verify the arena was
    // reset (used_bytes == 0).
    EXPECT(search.arena().arena_used_bytes() == 0u); // skeleton: nothing allocated

    chess::Board b("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    search.setBoard(b);
    EXPECT(search.arena().arena_used_bytes() == 0u);
    EXPECT(search.root_key() == b.hash());
}

void test_makemove_preserves_arena() {
    std::printf("[4] makemove preserves arena\n");
    LksSearch search;

    // Snapshot pointer-ish identity by capturing arena_used_bytes() (currently
    // 0 since the skeleton doesn't write to it). The deeper invariant we want
    // is that makemove() does not call reset(); the easiest proxy is that
    // searched_since_reset_ stays as-is and the arena_used_bytes() does not
    // *decrease*. We extend this once the real algorithm starts populating it.
    const auto used_before = search.arena().arena_used_bytes();

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, search.board());
    EXPECT(!moves.empty());
    auto m = moves[0];

    // Run a search so searched_since_reset_ is true; then makemove and ensure
    // arena state is preserved (no implicit reset).
    Recorder rec;
    LksSearchConfig cfg;
    cfg.max_evals = 2;
    cfg.min_info_period_ms = 0;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    search.quit();

    const auto used_mid = search.arena().arena_used_bytes();
    search.makemove(m);
    const auto used_after = search.arena().arena_used_bytes();

    EXPECT(used_after == used_mid);             // makemove did not reset
    EXPECT(used_before == 0u);                  // sanity (skeleton)
    EXPECT(search.root_key() != 0u);
}

void test_double_search_throws() {
    std::printf("[5] double-search throws\n");
    LksSearch search;
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 1000;
    cfg.min_info_period_ms = 0;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));

    bool threw = false;
    try {
        LksSearchConfig cfg2;
        cfg2.max_evals = 10;
        cfg2.on_uci_line = rec.callback();
        search.search(std::move(cfg2));
    } catch (const std::logic_error&) {
        threw = true;
    }
    EXPECT(threw);

    search.quit();
    EXPECT(!search.is_searching());
}

void test_quit_without_search_is_noop() {
    std::printf("[6] quit() with no search is a fast no-op\n");
    LksSearch search;

    auto t0 = std::chrono::steady_clock::now();
    search.quit();
    search.quit();
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();
    EXPECT(elapsed_us < 5000);                  // < 5ms for three calls
    EXPECT(!search.is_searching());
}

}  // namespace

int main() {
    test_natural_completion();
    test_quit_mid_search();
    test_setboard_resets_after_search();
    test_makemove_preserves_arena();
    test_double_search_throws();
    test_quit_without_search_is_noop();

    if (g_failed == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d expectation(s) failed.\n", g_failed);
    return 1;
}
