/**
 * Lifecycle + multi-worker tests for LksSearch (real GPU path).
 *
 * Each LksSearch instance spins up N persistent BatchEvaluators, each of
 * which loads a TensorRT engine and allocates CUDA buffers — a few seconds
 * total. To keep this test suite fast we use workers_per_gpu=1 for the
 * lifecycle tests that don't need parallelism.
 *
 * Tests:
 *   1. natural completion: search() runs to max_evals, emits bestmove last
 *   2. quit-mid-search:    search() interrupted, bestmove still emitted,
 *                          no info lines after quit() returns
 *   3. setBoard-resets:    after a search, setBoard() drops arena_used_bytes to 0
 *   4. makemove-preserves: makemove() does NOT reset the arena (bytes preserved)
 *   5. double-search:      calling search() while one is running throws
 *   6. quit-noop:          calling quit() with no search is a fast no-op
 *   7. multi-worker scaling: 2 workers complete more evals than 1 worker
 *   8. stop-while-busy:    quit() is prompt with N busy workers
 *   9. concurrent TT writes: arena allocations consistent across workers
 *  10. real GPU evals:     LksSearch::lifetime_gpu_evals() matches total_evals()
 *  11. workers reused:     two back-to-back search() calls don't reload engines
 *
 * The TRT engine path can be overridden via the CATGPT_TRT_ENGINE env var.
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "lks_search.hpp"

namespace fs = std::filesystem;

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

fs::path resolve_engine_path() {
    if (const char* env = std::getenv("CATGPT_TRT_ENGINE")) {
        return env;
    }
    return "/home/shadeform/CatGPT/main.trt";
}

const fs::path& engine_path() {
    static const fs::path p = resolve_engine_path();
    return p;
}

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
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 40;
    cfg.on_uci_line = rec.callback();

    search.search(std::move(cfg));

    auto t0 = std::chrono::steady_clock::now();
    while (search.is_searching()) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) {
            std::fprintf(stderr, "  timed out waiting for natural completion\n");
            ++g_failed;
            search.quit();
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    search.quit();

    auto lines = rec.snapshot();
    EXPECT(!lines.empty());
    EXPECT(starts_with(lines.back(), "bestmove "));
}

void test_quit_mid_search() {
    std::printf("[2] quit mid-search\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 1'000'000;          // unbounded for the test window
    cfg.on_uci_line = rec.callback();

    auto t0 = std::chrono::steady_clock::now();
    search.search(std::move(cfg));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    EXPECT(elapsed_ms < 1500);                  // generous: GPU batch can be slow cold
    EXPECT(!search.is_searching());

    auto lines = rec.snapshot();
    EXPECT(!lines.empty());
    EXPECT(starts_with(lines.back(), "bestmove "));

    // No info line should appear *after* the bestmove.
    bool seen_bestmove = false;
    bool info_after_bestmove = false;
    for (const auto& l : lines) {
        if (starts_with(l, "bestmove ")) seen_bestmove = true;
        else if (seen_bestmove && starts_with(l, "info ")) info_after_bestmove = true;
    }
    EXPECT(seen_bestmove);
    EXPECT(!info_after_bestmove);

    size_t before = rec.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT(rec.size() == before);
}

void test_setboard_resets_after_search() {
    std::printf("[3] setBoard resets after search\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 16;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));
    while (search.is_searching()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    search.quit();

    EXPECT(search.arena().arena_used_bytes() > 0u);

    chess::Board b("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    search.setBoard(b);
    EXPECT(search.arena().arena_used_bytes() == 0u);
    EXPECT(search.root_key() == b.hash());
}

void test_makemove_preserves_arena() {
    std::printf("[4] makemove preserves arena\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, search.board());
    EXPECT(!moves.empty());
    auto m = moves[0];

    EXPECT(search.arena().arena_used_bytes() == 0u);

    Recorder rec;
    LksSearchConfig cfg;
    cfg.max_evals = 16;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    search.quit();

    const auto used_mid = search.arena().arena_used_bytes();
    EXPECT(used_mid > 0u);

    search.makemove(m);
    const auto used_after = search.arena().arena_used_bytes();

    EXPECT(used_after == used_mid);
    EXPECT(search.root_key() != 0u);
}

void test_double_search_throws() {
    std::printf("[5] double-search throws\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 1'000'000;
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
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/1);

    auto t0 = std::chrono::steady_clock::now();
    search.quit();
    search.quit();
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        t1 - t0).count();
    EXPECT(elapsed_us < 5000);
    EXPECT(!search.is_searching());
}

// ── Multi-worker tests ────────────────────────────────────────────────────

void test_multi_worker_throughput() {
    std::printf("[7] multi-worker throughput (2 workers > 1 worker)\n");
    constexpr int kRunMs = 500;
    auto run_for = [](int workers_per_gpu, int run_ms) -> uint64_t {
        LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 20),
                         workers_per_gpu, /*coros_per_worker=*/8);
        Recorder rec;
        LksSearchConfig cfg;
        cfg.max_evals = 1'000'000;
        cfg.on_uci_line = rec.callback();
        search.search(std::move(cfg));
        std::this_thread::sleep_for(std::chrono::milliseconds(run_ms));
        search.quit();
        return search.total_evals();
    };

    uint64_t evals_1 = run_for(1, kRunMs);
    uint64_t evals_2 = run_for(2, kRunMs);

    std::printf("    1-worker evals=%llu  2-worker evals=%llu  (run=%dms)\n",
                (unsigned long long)evals_1,
                (unsigned long long)evals_2,
                kRunMs);

    EXPECT(evals_1 > 0);
    EXPECT(evals_2 > evals_1);
    // 2 workers should pipeline the GPU. Realistically we observe ~1.25x in
    // practice (the GPU is already mostly saturated by a single worker at
    // K=8 coros); demand at least 1.15x.
    EXPECT(evals_2 * 100 >= 115 * evals_1);
}

void test_stop_with_busy_workers() {
    std::printf("[8] quit() with N busy workers is prompt\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 20),
                     /*workers_per_gpu=*/2, /*coros_per_worker=*/8);
    Recorder rec;
    LksSearchConfig cfg;
    cfg.max_evals = 1'000'000;
    cfg.on_uci_line = rec.callback();

    auto t0 = std::chrono::steady_clock::now();
    search.search(std::move(cfg));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    auto t_before_quit = std::chrono::steady_clock::now();
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    auto quit_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t_before_quit).count();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();

    std::printf("    quit_latency=%lldms  total=%lldms  total_evals=%llu\n",
                (long long)quit_ms, (long long)total_ms,
                (unsigned long long)search.total_evals());

    // Bound: at most one in-flight GPU batch must drain.
    EXPECT(quit_ms < 500);
    EXPECT(!search.is_searching());

    auto lines = rec.snapshot();
    EXPECT(!lines.empty());
    EXPECT(starts_with(lines.back(), "bestmove "));
}

void test_concurrent_tt_writes() {
    std::printf("[9] concurrent TT writes — arena_used bounded by num_moves and eval count\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/2, /*coros_per_worker=*/8);
    Recorder rec;
    LksSearchConfig cfg;
    cfg.max_evals = 256;
    cfg.on_uci_line = rec.callback();

    search.search(std::move(cfg));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    search.quit();

    const auto& arena = search.arena();
    // Real movegen has variable num_moves per position. Claim-after-eval also
    // permits CAS-loser orphaned bytes (multiple coros expanding the same key
    // simultaneously). The only invariants left:
    //   - each eval claimed at most one block of bytes (218 = max legal moves)
    //   - the arena holds AT LEAST node_info_bytes(2) per evaluated key (a node
    //     with two legal moves is the minimum observable in self-play).
    const uint64_t bytes_min2  = catgpt::v2::SearchArena::node_info_bytes(/*num_moves=*/2);
    const uint64_t bytes_max218 = catgpt::v2::SearchArena::node_info_bytes(/*num_moves=*/218);
    const uint64_t arena_used = arena.arena_used_bytes();
    const uint64_t evals = search.total_evals();

    std::printf("    total_evals=%llu  arena_used=%llu  bounds=[%llu, %llu]\n",
                (unsigned long long)evals,
                (unsigned long long)arena_used,
                (unsigned long long)(evals * bytes_min2),
                (unsigned long long)(evals * bytes_max218));

    EXPECT(evals > 0);
    EXPECT(arena_used <= evals * bytes_max218);
    EXPECT(arena_used >= bytes_min2);   // at least the root was expanded
}

void test_real_gpu_evals() {
    std::printf("[10] real GPU evals — lifetime_gpu_evals matches total_evals\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 20),
                     /*workers_per_gpu=*/2, /*coros_per_worker=*/8);
    Recorder rec;
    LksSearchConfig cfg;
    cfg.max_evals = 200;
    cfg.on_uci_line = rec.callback();

    search.search(std::move(cfg));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    search.quit();

    uint64_t total = search.total_evals();
    uint64_t gpu_total = search.lifetime_gpu_evals();
    std::printf("    total_evals=%llu  lifetime_gpu_evals=%llu (delta=%lld)\n",
                (unsigned long long)total,
                (unsigned long long)gpu_total,
                (long long)gpu_total - (long long)total);

    // gpu_total counts every GPU completion; total_evals counts only those
    // for which the descent coroutine got past its post-`co_await` stop
    // check before incrementing. When stop is flipped, up to `K` coros
    // per worker can have an in-flight GPU eval (the eval semaphore's
    // permit count, equal to coros_per_worker); their results land but
    // those coros may exit without incrementing `evals`. So
    // gpu_total >= total and
    //   gpu_total - total <= num_workers * coros_per_worker.
    EXPECT(gpu_total >= total);
    // num_workers * coros_per_worker; num_workers scales with #GPUs.
    EXPECT(gpu_total - total <=
           static_cast<uint64_t>(search.num_workers()) * 8);
    EXPECT(total > 50);
}

void test_workers_reused() {
    std::printf("[11] workers reused across back-to-back searches\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 20),
                     /*workers_per_gpu=*/1, /*coros_per_worker=*/8);

    // Search A: short.
    Recorder rec_a;
    LksSearchConfig cfg_a;
    cfg_a.max_evals = 64;
    cfg_a.on_uci_line = rec_a.callback();
    auto t_a0 = std::chrono::steady_clock::now();
    search.search(std::move(cfg_a));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto t_a1 = std::chrono::steady_clock::now();
    auto a_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_a1 - t_a0).count();
    uint64_t evals_after_a = search.total_evals();
    uint64_t gpu_after_a = search.lifetime_gpu_evals();

    EXPECT(evals_after_a > 0);

    // Search B: same fan-out. Should start producing evals quickly because
    // the engines are already loaded; no engine reload visible in latency.
    Recorder rec_b;
    LksSearchConfig cfg_b;
    cfg_b.max_evals = 64;
    cfg_b.on_uci_line = rec_b.callback();
    auto t_b0 = std::chrono::steady_clock::now();
    search.search(std::move(cfg_b));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto t_b1 = std::chrono::steady_clock::now();
    auto b_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_b1 - t_b0).count();
    uint64_t evals_after_b = search.total_evals();
    uint64_t gpu_after_b = search.lifetime_gpu_evals();

    std::printf("    search A: %lldms  evals=%llu (gpu=%llu)\n",
                (long long)a_ms,
                (unsigned long long)evals_after_a,
                (unsigned long long)gpu_after_a);
    std::printf("    search B: %lldms  evals=%llu (gpu=%llu, +%llu vs after A)\n",
                (long long)b_ms,
                (unsigned long long)evals_after_b,
                (unsigned long long)gpu_after_b,
                (unsigned long long)(gpu_after_b - gpu_after_a));

    // total_evals is per-search and resets at the start of each search.
    EXPECT(evals_after_b > 0);
    EXPECT(evals_after_b <= evals_after_a + 64 + 32);  // sanity bound

    // lifetime_gpu_evals accumulates: search B's GPU evals stack on top of A.
    EXPECT(gpu_after_b > gpu_after_a);

    // Reuse signal: search B latency should be similar to A's (no engine
    // reload). Worst-case allow 4x A's runtime as a safety margin against
    // jitter, but flag if it's catastrophically slower.
    if (a_ms > 0) {
        EXPECT(b_ms <= 4 * a_ms + 500);
    }
}

void test_id_depth_advances() {
    std::printf("[12] iterative-deepening depth advances + stays in sync\n");
    LksSearch search(engine_path(), /*lifetime_max_evals=*/(1ULL << 18),
                     /*workers_per_gpu=*/2, /*coros_per_worker=*/8);
    Recorder rec;

    LksSearchConfig cfg;
    cfg.max_evals = 64;
    cfg.start_depth = 0.0f;
    cfg.delta_depth = 0.2f;
    cfg.on_uci_line = rec.callback();
    search.search(std::move(cfg));
    while (search.is_searching()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    search.quit();

    // Per-worker depths are always start + k*delta for some integer k >= 0.
    // The per-worker starts are staggered: worker 0 -> 0.0, worker 1 -> 0.1
    // (cfg.delta_depth / num_workers, where num_workers = workers_per_gpu * #GPUs).
    const float min_d = search.min_depth();
    const float max_d = search.max_depth();
    std::printf("    min_depth=%.3f max_depth=%.3f spread=%.3f\n",
                min_d, max_d, max_d - min_d);

    EXPECT(min_d >= 0.0f);
    EXPECT(max_d >= min_d);
    // Each worker advances independently; the spread between fastest and
    // slowest worker should never exceed two ID steps in steady state.
    EXPECT((max_d - min_d) < 2.0f * 0.2f + 1e-3f);

    // At least one ID iteration should have completed by either worker.
    EXPECT(max_d > 0.0f);

    // info lines should carry a `depth` field with a non-negative float,
    // and at least one should carry a `score cp` field sourced from the
    // root TT entry.
    bool saw_depth_field = false;
    bool saw_score_cp = false;
    for (const auto& l : rec.snapshot()) {
        if (starts_with(l, "info depth ")) saw_depth_field = true;
        if (l.find(" score cp ") != std::string::npos) saw_score_cp = true;
    }
    EXPECT(saw_depth_field);
    EXPECT(saw_score_cp);
}

}  // namespace

int main() {
    std::printf("Using TRT engine: %s\n", engine_path().c_str());

    test_natural_completion();
    test_quit_mid_search();
    test_setboard_resets_after_search();
    test_makemove_preserves_arena();
    test_double_search_throws();
    test_quit_without_search_is_noop();
    test_multi_worker_throughput();
    test_stop_with_busy_workers();
    test_concurrent_tt_writes();
    test_real_gpu_evals();
    test_workers_reused();
    test_id_depth_advances();

    if (g_failed == 0) {
        std::printf("\nAll tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d expectation(s) failed.\n", g_failed);
    return 1;
}
