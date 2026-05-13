/**
 * Throughput vs batch-size sweep for LksSearch.
 *
 * Sweeps `(num_workers, max_batch_size)` and reports:
 *   - elapsed wall time
 *   - total NN evaluations (lifetime_gpu_evals)
 *   - throughput (evals/sec)
 *   - total GPU batches dispatched
 *   - average batch size (evals / batches)
 *
 * Intent: understand how throughput scales with M (max GPU batch size,
 * the engine knob controlling bucket selection) and N (workers, which
 * controls GPU pipelining via separate streams).
 *
 * Saturation rule:
 *
 *   With the GPU thread eagerly draining `BatchEvaluator::queue_`, the
 *   per-worker steady-state batch size is min(K - B, max_bucket(K)),
 *   where K = coros_per_worker. The equilibrium is B = K/2 unless K is
 *   large enough that K - B ≥ M (the largest bucket ≤ max_batch_size),
 *   i.e. K ≥ 2*M. So to actually measure the GPU-saturated throughput
 *   at a given max_batch_size M, you need K significantly above 2*M.
 *
 *   Empirically (RTX 5090, FP16 main.trt) batches saturate cleanly at
 *   K ≈ 2*M and headline throughput peaks around K = 3*M to 4*M; above
 *   that, queue depth grows without buying more batch size and per-
 *   coro latency just inflates. This bench picks K = 4*M.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "lks_search.hpp"

namespace fs = std::filesystem;
using catgpt::lks::LksSearch;
using catgpt::lks::LksSearchConfig;

namespace {

fs::path engine_path() {
    if (const char* env = std::getenv("CATGPT_TRT_ENGINE")) return env;
    return "/home/shadeform/CatGPT/main.trt";
}

struct Result {
    int workers_per_gpu;
    int coros_per_worker;
    int max_batch_size;
    double elapsed_ms;
    uint64_t evals;
    uint64_t batches;
};

Result run_one(int workers_per_gpu, int coros_per_worker, int max_batch_size, int run_ms) {
    LksSearch search(engine_path(),
                     /*lifetime_max_evals=*/(1ULL << 22),
                     workers_per_gpu,
                     coros_per_worker,
                     max_batch_size);

    LksSearchConfig cfg;
    cfg.max_evals = 1'000'000'000;
    cfg.on_uci_line = [](std::string_view) {};

    // Warmup: ~50ms to let TRT build optimization caches.
    search.search(cfg);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    search.quit();

    // Reset evals_ tally implicitly happens at next search() — but
    // BatchEvaluator's lifetime counters keep accumulating. Sample now.
    uint64_t pre_evals = search.lifetime_gpu_evals();

    // Aggregate per-worker batches across workers (we'll need a delta).
    auto total_batches = [&]() -> uint64_t {
        uint64_t b = 0;
        for (int i = 0; i < workers_per_gpu; ++i) {
            // No public getter for per-worker batches; we'll use a method
            // we'll add below.
        }
        return b;
    };
    (void)total_batches;
    uint64_t pre_batches = search.lifetime_gpu_batches();

    auto t0 = std::chrono::steady_clock::now();
    LksSearchConfig cfg2 = cfg;
    cfg2.on_uci_line = [](std::string_view) {};
    search.search(std::move(cfg2));
    std::this_thread::sleep_for(std::chrono::milliseconds(run_ms));
    search.quit();
    auto t1 = std::chrono::steady_clock::now();

    Result r;
    r.workers_per_gpu = workers_per_gpu;
    r.coros_per_worker = coros_per_worker;
    r.max_batch_size = max_batch_size;
    r.elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    r.evals = search.lifetime_gpu_evals() - pre_evals;
    r.batches = search.lifetime_gpu_batches() - pre_batches;
    return r;
}

void print_header() {
    std::printf("%-7s %-6s %-9s %-10s %-10s %-12s %-10s %-9s\n",
                "wpg", "coros", "max_batch",
                "elapsed", "evals", "evals/s", "batches", "avg_batch");
}

void print_row(const Result& r) {
    double evals_per_sec = r.evals / (r.elapsed_ms / 1000.0);
    double avg_batch = r.batches > 0 ? double(r.evals) / r.batches : 0.0;
    std::printf("%-7d %-6d %-9d %-9.1fms %-10llu %-12.0f %-10llu %-9.2f\n",
                r.workers_per_gpu,
                r.coros_per_worker,
                r.max_batch_size,
                r.elapsed_ms,
                (unsigned long long)r.evals,
                evals_per_sec,
                (unsigned long long)r.batches,
                avg_batch);
}

}  // namespace

int main(int argc, char** argv) {
    int run_ms = 1000;
    if (argc > 1) run_ms = std::atoi(argv[1]);

    // Pick coros_per_worker large enough to break the K/2 equilibrium
    // and keep BatchEvaluator's queue saturated at the target bucket.
    constexpr int kCoroFactor = 4;

    std::printf("LKS throughput sweep: engine=%s, run_ms=%d per cell, "
                "coros_per_worker = %d * max_batch_size\n\n",
                engine_path().c_str(), run_ms, kCoroFactor);

    print_header();

    // max_batch_size sweep at workers_per_gpu=1 over the engine's bucket set.
    // (Bucket sizes mirror BatchEvaluator::kBucketSizes.)
    std::printf("# max_batch sweep (1 worker per gpu)\n");
    for (int M : {1, 2, 4, 6, 8, 12, 18, 26, 36, 56, 112}) {
        auto r = run_one(/*workers_per_gpu=*/1,
                         /*coros_per_worker=*/kCoroFactor * M,
                         /*max_batch_size=*/M, run_ms);
        print_row(r);
    }
    std::printf("\n");

    // max_batch_size sweep at workers_per_gpu=2.
    std::printf("# max_batch sweep (2 workers per gpu)\n");
    for (int M : {1, 2, 4, 6, 8, 12, 18, 26, 36, 56, 112}) {
        auto r = run_one(/*workers_per_gpu=*/2,
                         /*coros_per_worker=*/kCoroFactor * M,
                         /*max_batch_size=*/M, run_ms);
        print_row(r);
    }
    std::printf("\n");

    // workers_per_gpu sweep at max_batch_size=112 (the largest bucket).
    constexpr int kMaxBatch = 112;
    std::printf("# workers_per_gpu sweep (max_batch=%d, coros_per_worker=%d)\n",
                kMaxBatch, kCoroFactor * kMaxBatch);
    for (int N : {1, 2, 3, 4, 6, 8}) {
        auto r = run_one(/*workers_per_gpu=*/N,
                         /*coros_per_worker=*/kCoroFactor * kMaxBatch,
                         /*max_batch_size=*/kMaxBatch, run_ms);
        print_row(r);
    }
    return 0;
}
