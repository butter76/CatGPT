/**
 * Synthetic microbenchmark for SearchArena (TT + NodeInfo arena).
 *
 * Phases (per K):
 *   A. Fill        — insert K random keys, alloc per-node info, fill priors.
 *   B. Traversal   — repeated random lookups + walking MoveInfo[] + child
 *                    lookups derived from `key XOR move`. Mixes hits and
 *                    misses to model real search behaviour.
 *   C. Tear-down   — destroy SearchArena and time the dtor (two delete[]s).
 *
 * Reports: ns/op, hit/miss split, avg+p99 probe length, bytes used, tear-down ns.
 *
 * Once the SearchArena ctor returns, a global `operator new` guard triggers
 * abort if anyone allocates from the heap during the timed phases. This
 * proves that the only frees on tear-down are the two `delete[]`s.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <random>
#include <vector>

#include "tt_arena.hpp"

using catgpt::v2::SearchArena;
using catgpt::v2::TTEntry;
using catgpt::v2::NodeInfoHeader;
using catgpt::v2::MoveInfo;
using catgpt::v2::kEmptyKey;
using catgpt::v2::kNoInfoOffset;

// ── Allocation guard ───────────────────────────────────────────────────────
//
// When `g_no_alloc` is true, any heap allocation aborts. We turn this on after
// the SearchArena and all bench-side buffers are allocated, and turn it off
// before sorting/printing.
namespace {
std::atomic<bool> g_no_alloc{false};
std::atomic<uint64_t> g_alloc_count{0};
}  // namespace

void* operator new(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    if (g_no_alloc.load(std::memory_order_relaxed)) {
        std::fprintf(stderr,
                     "[tt_arena_bench] FATAL: heap allocation of %zu bytes "
                     "during a no-alloc phase\n", n);
        std::abort();
    }
    void* p = std::malloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}

void* operator new[](std::size_t n) {
    return ::operator new(n);
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// ── Helpers ────────────────────────────────────────────────────────────────

namespace {

using Clock = std::chrono::steady_clock;
using ns = std::chrono::nanoseconds;

double ns_per_op(ns total, uint64_t ops) {
    return static_cast<double>(total.count()) / static_cast<double>(ops);
}

uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct PhaseTimings {
    ns fill;
    ns traverse;
    ns teardown;
    uint64_t fill_ops;
    uint64_t traverse_lookups;
    uint64_t traverse_hits;
    uint64_t traverse_misses;
    double avg_probe_len_hit;
    uint64_t p99_probe_len_hit;
    uint64_t max_probe_len_hit;
    uint64_t arena_used;
    uint64_t arena_capacity;
    uint64_t table_bytes;
    uint64_t alloc_count_during_fill;
    uint64_t alloc_count_during_traverse;
};

void run_one(uint64_t K, PhaseTimings& out) {
    constexpr double kLoadFactor = 0.5;
    constexpr uint64_t kAvgMovesPerNode = 40;

    // Pre-allocate everything we'll need before turning the guard on.
    std::vector<uint64_t> keys(K);
    {
        std::mt19937_64 rng(0xC0FFEEULL ^ K);
        for (uint64_t i = 0; i < K; ++i) {
            uint64_t k = rng();
            // Make sure no zero keys (canonicalize_key would handle it but
            // we want truly distinct test keys).
            if (k == 0) k = 1;
            keys[i] = k;
        }
    }

    // Each node will have num_moves drawn from U[20, 60].
    std::vector<uint16_t> move_counts(K);
    {
        std::mt19937_64 rng(0xBADF00DULL ^ K);
        std::uniform_int_distribution<int> dist(20, 60);
        for (uint64_t i = 0; i < K; ++i) {
            move_counts[i] = static_cast<uint16_t>(dist(rng));
        }
    }

    // Probe-length samples for hits during traversal.
    const uint64_t kProbeSamples = std::min<uint64_t>(K, 1ULL << 18);
    std::vector<uint32_t> hit_probes;
    hit_probes.reserve(kProbeSamples);

    // The arena under test.
    SearchArena arena(K, kLoadFactor, kAvgMovesPerNode);
    out.table_bytes = arena.table_bytes();
    out.arena_capacity = arena.arena_capacity_bytes();

    // ── Phase A: fill ──────────────────────────────────────────────────────
    g_alloc_count.store(0, std::memory_order_relaxed);
    g_no_alloc.store(true, std::memory_order_relaxed);

    auto t0 = Clock::now();
    for (uint64_t i = 0; i < K; ++i) {
        const uint16_t nm = move_counts[i];

        // Eval-first batch-allocate: write the arena bytes BEFORE claiming
        // the TT slot. (In the synthetic single-thread case the order
        // doesn't matter for correctness but matches the production
        // workflow.)
        const uint64_t off = arena.alloc_node_info(nm);
        auto* hdr = arena.info_at(off);
        hdr->variance = 0.05f;

        MoveInfo* moves = arena.moves_at(off);
        uint64_t s = keys[i];
        for (uint16_t j = 0; j < nm; ++j) {
            uint64_t r = splitmix64(s);
            moves[j].move = static_cast<uint16_t>(r);
            moves[j].terminal_kind = 0;
            moves[j]._pad = 0;
            moves[j].P = 1.0f / static_cast<float>(nm);
            moves[j].P_alloc = moves[j].P;
            moves[j].P_optimistic = moves[j].P;
        }

        auto [e, claimed] = arena.find_or_claim(keys[i]);
        // Distinct keys -> we always claim.
        SearchArena::set_initial_qd(e, /*Q=*/0.123f, /*max_depth=*/0.0f);
        SearchArena::publish_info(e, off);
    }
    auto t1 = Clock::now();
    out.fill = std::chrono::duration_cast<ns>(t1 - t0);
    out.fill_ops = K;
    out.alloc_count_during_fill = g_alloc_count.load(std::memory_order_relaxed);
    out.arena_used = arena.arena_used_bytes();

    // ── Phase B: traversal (random lookups + child lookups) ───────────────
    g_alloc_count.store(0, std::memory_order_relaxed);

    const uint64_t kLookups = 10ULL * K;
    uint64_t hits = 0, misses = 0;
    // Use a simple LCG-style rng without std::uniform_int_distribution to avoid
    // surprises; keep it deterministic.
    uint64_t rng_state = 0xDEADBEEFCAFEULL ^ K;

    auto t2 = Clock::now();
    uint64_t accum = 0;  // anti-DCE
    for (uint64_t i = 0; i < kLookups; ++i) {
        uint64_t r = splitmix64(rng_state);
        // 80% of lookups are derived from a real key (parent), 20% are
        // child-derived (mostly miss). This roughly mirrors a search where
        // most TT touches are revisits along established branches.
        const bool use_real = (r & 0x7) != 0;
        uint64_t parent_idx = (r >> 3) % K;
        uint64_t key = keys[parent_idx];

        TTEntry* e = arena.find(key);
        if (e) {
            ++hits;
            // Sample a probe length on hits, capped to keep storage bounded.
            if (hit_probes.size() < kProbeSamples) {
                hit_probes.push_back(static_cast<uint32_t>(arena.probe_length(key)));
            }
            // Walk a few moves to touch arena.
            uint64_t off = e->info_offset.load(std::memory_order_acquire);
            if (off != kNoInfoOffset) {
                const NodeInfoHeader* hdr = arena.info_at(off);
                const MoveInfo* mv = arena.moves_at(off);
                accum += hdr->num_moves;
                accum += mv[r % hdr->num_moves].move;

                if (!use_real) {
                    // Synthesize a "child" key that's almost certainly a miss.
                    uint64_t child_key = key ^ (mv[r % hdr->num_moves].move
                                              | 0x100000000ULL);
                    TTEntry* ce = arena.find(child_key);
                    if (ce) ++hits; else ++misses;
                }
            }
        } else {
            ++misses;
        }
    }
    auto t3 = Clock::now();
    out.traverse = std::chrono::duration_cast<ns>(t3 - t2);
    out.traverse_lookups = kLookups;
    out.traverse_hits = hits;
    out.traverse_misses = misses;
    out.alloc_count_during_traverse = g_alloc_count.load(std::memory_order_relaxed);

    // Side-effect to defeat DCE.
    if (accum == 0xFFFFFFFFFFFFFFFFULL) std::printf(" ");

    // Probe-length stats — disable the guard so we can sort.
    g_no_alloc.store(false, std::memory_order_relaxed);
    if (!hit_probes.empty()) {
        std::sort(hit_probes.begin(), hit_probes.end());
        uint64_t sum = 0;
        for (uint32_t x : hit_probes) sum += x;
        out.avg_probe_len_hit = static_cast<double>(sum) / hit_probes.size();
        size_t p99_idx = (hit_probes.size() * 99) / 100;
        if (p99_idx >= hit_probes.size()) p99_idx = hit_probes.size() - 1;
        out.p99_probe_len_hit = hit_probes[p99_idx];
        out.max_probe_len_hit = hit_probes.back();
    } else {
        out.avg_probe_len_hit = 0.0;
        out.p99_probe_len_hit = 0;
        out.max_probe_len_hit = 0;
    }

    // ── Phase C: tear-down ────────────────────────────────────────────────
    // Build a second SearchArena via placement-new so we can time *just* its
    // destructor in isolation (the function-scope `arena` will also be
    // destroyed at function exit, but outside the timer).
    {
        alignas(SearchArena) static thread_local unsigned char
            arena_storage[sizeof(SearchArena)];
        SearchArena* ap = new (arena_storage) SearchArena(K, kLoadFactor, kAvgMovesPerNode);
        for (uint64_t i = 0; i < std::min<uint64_t>(K, 1024); ++i) {
            uint64_t off = ap->alloc_node_info(move_counts[i]);
            auto [e, claimed] = ap->find_or_claim(keys[i]);
            (void)claimed;
            SearchArena::publish_info(e, off);
        }

        g_no_alloc.store(true, std::memory_order_relaxed);
        auto td0 = Clock::now();
        ap->~SearchArena();
        auto td1 = Clock::now();
        g_no_alloc.store(false, std::memory_order_relaxed);
        out.teardown = std::chrono::duration_cast<ns>(td1 - td0);
    }
}

void print_row(uint64_t K, const PhaseTimings& t) {
    auto MB = [](uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); };
    std::printf("\n=== K = %lu ===\n", static_cast<unsigned long>(K));
    std::printf("  table:  %.1f MB (%lu entries)\n",
                MB(t.table_bytes),
                static_cast<unsigned long>(t.table_bytes / sizeof(TTEntry)));
    std::printf("  arena:  %.1f MB used / %.1f MB cap   (%.1f bytes/node)\n",
                MB(t.arena_used), MB(t.arena_capacity),
                static_cast<double>(t.arena_used) / static_cast<double>(K));
    std::printf("  fill:        %.2f ns/op   (%lu ops, %.2f ms total)\n",
                ns_per_op(t.fill, t.fill_ops),
                static_cast<unsigned long>(t.fill_ops),
                static_cast<double>(t.fill.count()) / 1e6);
    std::printf("  traverse:    %.2f ns/op   (%lu lookups, %lu hits, %lu misses)\n",
                ns_per_op(t.traverse, t.traverse_lookups),
                static_cast<unsigned long>(t.traverse_lookups),
                static_cast<unsigned long>(t.traverse_hits),
                static_cast<unsigned long>(t.traverse_misses));
    std::printf("  probe len:   avg=%.3f  p99=%lu  max=%lu  (hit lookups)\n",
                t.avg_probe_len_hit,
                static_cast<unsigned long>(t.p99_probe_len_hit),
                static_cast<unsigned long>(t.max_probe_len_hit));
    std::printf("  tear-down:   %.3f ms\n",
                static_cast<double>(t.teardown.count()) / 1e6);
    std::printf("  allocs during fill=%lu  during traverse=%lu (must be 0)\n",
                static_cast<unsigned long>(t.alloc_count_during_fill),
                static_cast<unsigned long>(t.alloc_count_during_traverse));
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<uint64_t> Ks;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            Ks.push_back(static_cast<uint64_t>(std::strtoull(argv[i], nullptr, 0)));
        }
    } else {
        Ks = {1ULL << 16, 1ULL << 18, 1ULL << 20, 1ULL << 22};
    }

    std::printf("tt_arena_bench: TTEntry=%zuB, NodeInfoHeader=%zuB, MoveInfo=%zuB\n",
                sizeof(TTEntry), sizeof(NodeInfoHeader), sizeof(MoveInfo));

    for (uint64_t K : Ks) {
        PhaseTimings t{};
        run_one(K, t);
        print_row(K, t);
    }

    return 0;
}
