/**
 * Multi-threaded stress + bench for the lock-free SearchArena.
 *
 * Phases (each independent, self-contained):
 *
 *   P1. Correctness stress:
 *       N threads each insert a disjoint random key set. After joining,
 *       every key must be findable, every slot must be published, and
 *       its qd_packed.max_depth must equal the *largest* max_depth any writer
 *       ever attempted via update_qd for that key.
 *
 *   P2. Hot insert race:
 *       M threads all try to insert the SAME small set of keys (K_HOT)
 *       repeatedly. Each thread also does a stream of update_qd calls
 *       on those keys with monotonically increasing max_depth. After joining,
 *       each key has exactly one slot, and each slot's max_depth is the global
 *       max attempted across all threads.
 *
 *   P3. Reader-during-writer:
 *       One writer inserts at a slow cadence; N readers continuously
 *       probe and call wait_published. Verify no torn / inconsistent
 *       reads (every successful wait_published yields a slot whose
 *       NodeInfoHeader.num_moves matches what the writer recorded).
 *
 *   P4. Throughput sweep:
 *       T in {1, 2, 4, 8, 16}: each thread runs the eval-first batch
 *       workflow against a private random key stream. Report inserts/sec
 *       and lookups/sec, plus the scaling factor vs T=1.
 *
 *   P5. Torn-claim regression (the bug the 128-bit Cell A fix targets):
 *       1 writer holds a single hot key and re-claims it (via a fresh
 *       SearchArena per generation) while encoding a generation tag in
 *       qd_packed.q. N readers `find()` the key and `load_qd()` it; for
 *       any successful key match, the loaded qd_packed must encode the
 *       same generation as the writer was on at that moment (or one
 *       neighbouring generation). Pre-fix this would race: a reader
 *       could observe the new key with a still-zero qd_packed. With
 *       the 128-bit CAS in Cell A the race is structurally impossible.
 *
 * No TRT / libcoro / chess-library deps. Pure C++23 + pthread.
 */

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tt_arena.hpp"

using catgpt::v2::SearchArena;
using catgpt::v2::TTEntry;
using catgpt::v2::NodeInfoHeader;
using catgpt::v2::MoveInfo;
using catgpt::v2::ClaimResult;
using catgpt::v2::kNoInfoOffset;
using catgpt::v2::pack_qd;
using catgpt::v2::unpack_qd;

namespace {

using Clock = std::chrono::steady_clock;

uint64_t splitmix64(uint64_t& x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

int g_failed = 0;

#define EXPECT(cond) do {                                                  \
    if (!(cond)) {                                                         \
        std::fprintf(stderr, "  FAIL: %s (line %d)\n", #cond, __LINE__);   \
        ++g_failed;                                                        \
    }                                                                      \
} while (0)

// ── P1 — Correctness stress ───────────────────────────────────────────────

void run_p1_correctness(uint64_t per_thread_keys, int num_threads) {
    std::printf("\n[P1] correctness stress: %d threads x %lu disjoint keys\n",
                num_threads, (unsigned long)per_thread_keys);
    const uint64_t K = static_cast<uint64_t>(num_threads) * per_thread_keys;
    SearchArena arena(K, 0.5, 40);

    // Pre-generate disjoint key streams.
    std::vector<std::vector<uint64_t>> keys_per_thread(num_threads);
    std::vector<std::vector<float>> max_d_per_thread(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        keys_per_thread[t].resize(per_thread_keys);
        max_d_per_thread[t].resize(per_thread_keys);
        std::mt19937_64 rng(0xA1B2C3D4ULL ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL));
        for (uint64_t i = 0; i < per_thread_keys; ++i) {
            // Mix thread id into upper bits to keep streams disjoint with
            // overwhelming probability.
            uint64_t k = rng();
            k = (k & 0x000FFFFFFFFFFFFFULL) | (uint64_t(t + 1) << 52);
            keys_per_thread[t][i] = k;
            max_d_per_thread[t][i] = static_cast<float>((rng() % 10000) + 1);
        }
    }

    std::barrier sync(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            sync.arrive_and_wait();
            const auto& keys = keys_per_thread[t];
            const auto& max_ds = max_d_per_thread[t];
            for (uint64_t i = 0; i < keys.size(); ++i) {
                const uint16_t nm = 30;
                uint64_t off = arena.alloc_node_info(nm);
                MoveInfo* moves = arena.moves_at(off);
                for (uint16_t j = 0; j < nm; ++j) {
                    moves[j] = MoveInfo::pack(j, 1.0f / nm,
                                              catgpt::v2::kTerminalNone);
                }

                auto [e, claimed] = arena.find_or_claim(keys[i],
                                                        /*Q=*/0.0f,
                                                        /*max_depth=*/max_ds[i]);
                EXPECT(claimed); // disjoint keys -> always our claim
                SearchArena::publish_info(e, /*origQ=*/0.0f, off);
            }
        });
    }
    for (auto& w : workers) w.join();

    // Verify: every key is findable, published, with the right max_depth and num_moves.
    for (int t = 0; t < num_threads; ++t) {
        const auto& keys = keys_per_thread[t];
        const auto& max_ds = max_d_per_thread[t];
        for (uint64_t i = 0; i < keys.size(); ++i) {
            TTEntry* e = arena.find(keys[i]);
            EXPECT(e != nullptr);
            if (!e) continue;
            const catgpt::v2::InfoCell info = SearchArena::load_info(e);
            EXPECT(info.info_offset != kNoInfoOffset);
            const auto* hdr = arena.info_at(info.info_offset);
            EXPECT(hdr->num_moves == 30);
            auto [q, md] = unpack_qd(SearchArena::load_qd(e).qd_packed);
            (void)q;
            EXPECT(md == max_ds[i]);
        }
    }
}

// ── P2 — Hot insert race ──────────────────────────────────────────────────

void run_p2_hot_race(int num_threads, uint64_t k_hot, uint64_t iters_per_thread) {
    std::printf("\n[P2] hot insert race: %d threads x %lu iterations on %lu hot keys\n",
                num_threads, (unsigned long)iters_per_thread, (unsigned long)k_hot);

    SearchArena arena(k_hot, 0.5, 40);

    std::vector<uint64_t> hot_keys(k_hot);
    {
        std::mt19937_64 rng(0xCAFE0001ULL);
        for (uint64_t i = 0; i < k_hot; ++i) {
            hot_keys[i] = rng() | 1;
        }
    }

    std::atomic<uint64_t> claim_wins{0};
    std::barrier sync(num_threads);
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            sync.arrive_and_wait();
            uint64_t rng = 0xFACE0000ULL ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL);
            uint64_t my_wins = 0;
            for (uint64_t it = 0; it < iters_per_thread; ++it) {
                uint64_t r = splitmix64(rng);
                uint64_t key = hot_keys[r % k_hot];

                auto [e, claimed] = arena.find_or_claim(key, /*Q=*/0.0f, /*max_depth=*/0.0f);
                if (claimed) {
                    ++my_wins;
                    uint64_t off = arena.alloc_node_info(30);
                    MoveInfo* moves = arena.moves_at(off);
                    for (uint16_t j = 0; j < 30; ++j) {
                        moves[j] = MoveInfo::pack(j, 1.0f / 30.0f,
                                                  catgpt::v2::kTerminalNone);
                    }
                    SearchArena::publish_info(e, /*origQ=*/0.0f, off);
                }

                // Independent: every thread CAS-bumps max_depth to a thread-and-iter
                // dependent value. Final max_depth for this key should equal the
                // global max attempted by any thread.
                float new_md = static_cast<float>((r >> 8) & 0xFFFFF);
                SearchArena::update_qd(e, /*Q=*/0.0f, new_md);
            }
            claim_wins.fetch_add(my_wins, std::memory_order_relaxed);
        });
    }
    for (auto& w : workers) w.join();

    // Exactly one thread won the claim per hot key.
    EXPECT(claim_wins.load() == k_hot);

    // Each hot key has a published slot, and its max_depth is the global max
    // across all threads' attempted updates (which is deterministic given
    // the rng seeds).
    std::unordered_map<uint64_t, float> expected_max_d;
    for (uint64_t key : hot_keys) expected_max_d[key] = 0.0f;
    for (int t = 0; t < num_threads; ++t) {
        uint64_t rng = 0xFACE0000ULL ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL);
        for (uint64_t it = 0; it < iters_per_thread; ++it) {
            uint64_t r = splitmix64(rng);
            uint64_t key = hot_keys[r % k_hot];
            float candidate = static_cast<float>((r >> 8) & 0xFFFFF);
            float& cur = expected_max_d[key];
            if (candidate > cur) cur = candidate;
        }
    }

    for (uint64_t key : hot_keys) {
        TTEntry* e = arena.find(key);
        EXPECT(e != nullptr);
        if (!e) continue;
        const catgpt::v2::InfoCell info = SearchArena::load_info(e);
        EXPECT(info.info_offset != kNoInfoOffset);
        auto [q, md] = unpack_qd(SearchArena::load_qd(e).qd_packed);
        (void)q;
        EXPECT(md == expected_max_d[key]);
    }
}

// ── P3 — Reader-during-writer ─────────────────────────────────────────────

void run_p3_reader_writer(int num_readers, uint64_t inserts, int writer_pause_us) {
    std::printf("\n[P3] reader-during-writer: %d readers, %lu writes (writer pauses %dus between)\n",
                num_readers, (unsigned long)inserts, writer_pause_us);

    SearchArena arena(inserts, 0.5, 40);

    std::vector<uint64_t> keys(inserts);
    {
        std::mt19937_64 rng(0x12345678ULL);
        for (uint64_t i = 0; i < inserts; ++i) keys[i] = rng() | 1;
    }

    std::atomic<bool> writer_done{false};
    std::atomic<uint64_t> torn_reads{0};
    std::atomic<uint64_t> wait_timeouts{0};
    std::atomic<uint64_t> successful_waits{0};

    std::thread writer([&]() {
        for (uint64_t i = 0; i < inserts; ++i) {
            uint16_t nm = static_cast<uint16_t>(20 + (i % 30));
            uint64_t off = arena.alloc_node_info(nm);
            MoveInfo* moves = arena.moves_at(off);
            for (uint16_t j = 0; j < nm; ++j) {
                moves[j] = MoveInfo::pack(j, 1.0f / nm,
                                          catgpt::v2::kTerminalNone);
            }
            // Encode num_moves into max_depth for round-trip checking.
            auto [e, claimed] = arena.find_or_claim(keys[i],
                                                    /*Q=*/0.0f,
                                                    /*max_depth=*/static_cast<float>(nm));
            (void)claimed;
            SearchArena::publish_info(e, /*origQ=*/0.0f, off);
            if (writer_pause_us > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(writer_pause_us));
            }
        }
        writer_done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    readers.reserve(num_readers);
    for (int r = 0; r < num_readers; ++r) {
        readers.emplace_back([&, r]() {
            uint64_t rng_state = 0xBEEFULL ^ (uint64_t(r) * 0x9E3779B97F4A7C15ULL);
            while (!writer_done.load(std::memory_order_acquire)) {
                uint64_t idx = splitmix64(rng_state) % keys.size();
                TTEntry* e = arena.find(keys[idx]);
                if (!e) continue;
                auto info_opt = SearchArena::wait_published(e, 4096);
                if (!info_opt) {
                    wait_timeouts.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                successful_waits.fetch_add(1, std::memory_order_relaxed);
                const auto* hdr = arena.info_at(info_opt->info_offset);
                auto [q, md] = unpack_qd(SearchArena::load_qd(e).qd_packed);
                (void)q;
                if (static_cast<float>(hdr->num_moves) != md) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    writer.join();
    for (auto& t : readers) t.join();

    std::printf("    successful_waits=%lu  wait_timeouts=%lu  torn_reads=%lu\n",
                (unsigned long)successful_waits.load(),
                (unsigned long)wait_timeouts.load(),
                (unsigned long)torn_reads.load());
    EXPECT(torn_reads.load() == 0);
}

// ── P4 — Throughput sweep ─────────────────────────────────────────────────

struct ThroughputResult {
    int threads;
    double inserts_per_sec;
    double lookups_per_sec;
    double seconds;
};

ThroughputResult run_p4_one(int num_threads, uint64_t per_thread_inserts) {
    constexpr uint64_t kBatchSize = 32;
    constexpr uint16_t kMovesPerNode = 30;

    const uint64_t K = static_cast<uint64_t>(num_threads) * per_thread_inserts;
    SearchArena arena(K, 0.5, kMovesPerNode);

    std::barrier sync(num_threads);
    std::atomic<uint64_t> total_lookups{0};

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    auto t_begin = std::atomic<int64_t>{0};
    auto t_end = std::atomic<int64_t>{0};

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            uint64_t rng_state = 0xDEAD0000ULL ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL);

            // Pre-generate keys and decisions to avoid measuring rng cost.
            std::vector<uint64_t> keys(per_thread_inserts);
            for (uint64_t i = 0; i < per_thread_inserts; ++i) {
                uint64_t k = splitmix64(rng_state);
                k = (k & 0x000FFFFFFFFFFFFFULL) | (uint64_t(t + 1) << 52);
                keys[i] = k;
            }

            sync.arrive_and_wait();
            if (t == 0) {
                t_begin.store(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch()).count(),
                    std::memory_order_relaxed);
            }

            // Eval-first batch workflow.
            const uint64_t per_node_bytes = SearchArena::node_info_bytes(kMovesPerNode);
            uint64_t lookups = 0;

            for (uint64_t i = 0; i < per_thread_inserts; i += kBatchSize) {
                const uint64_t this_batch =
                    std::min<uint64_t>(kBatchSize, per_thread_inserts - i);

                // 1. one fetch_add for the whole batch
                uint64_t base = arena.alloc_raw(per_node_bytes * this_batch);

                // 2. fill arena bytes for each position
                for (uint64_t j = 0; j < this_batch; ++j) {
                    uint64_t off = base + j * per_node_bytes;
                    auto* hdr = arena.info_at(off);
                    hdr->variance = 0.05f;
                    hdr->num_moves = kMovesPerNode;
                    hdr->flags = 0;
                    MoveInfo* moves = arena.moves_at(off);
                    for (uint16_t m = 0; m < kMovesPerNode; ++m) {
                        moves[m] = MoveInfo::pack(m, 1.0f / kMovesPerNode,
                                                  catgpt::v2::kTerminalNone);
                    }
                }

                // 3. claim + publish each
                for (uint64_t j = 0; j < this_batch; ++j) {
                    uint64_t off = base + j * per_node_bytes;
                    auto [e, claimed] = arena.find_or_claim(keys[i + j],
                                                            /*Q=*/0.0f,
                                                            /*max_depth=*/1.0f);
                    (void)claimed;
                    SearchArena::publish_info(e, /*origQ=*/0.0f, off);
                }

                // 4. a small lookup burst per batch (search reads are mostly hits)
                for (uint64_t j = 0; j < this_batch; ++j) {
                    uint64_t r = splitmix64(rng_state) % (i + this_batch);
                    TTEntry* e = arena.find(keys[r]);
                    asm volatile("" : : "r"(e) : "memory");
                    ++lookups;
                }
            }

            sync.arrive_and_wait();
            if (t == 0) {
                t_end.store(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        Clock::now().time_since_epoch()).count(),
                    std::memory_order_relaxed);
            }
            total_lookups.fetch_add(lookups, std::memory_order_relaxed);
        });
    }
    for (auto& w : workers) w.join();

    int64_t ns = t_end.load() - t_begin.load();
    double sec = double(ns) / 1e9;
    return ThroughputResult{
        num_threads,
        double(K) / sec,
        double(total_lookups.load()) / sec,
        sec,
    };
}

void run_p4_sweep(uint64_t per_thread_inserts) {
    std::printf("\n[P4] throughput sweep: each thread does %lu inserts (eval-first batch)\n",
                (unsigned long)per_thread_inserts);
    std::printf("    %4s  %14s  %14s  %14s  %8s\n",
                "T", "inserts/s", "lookups/s", "ns/insert", "scale");

    std::vector<int> Ts = {1, 2, 4, 8};
    if (std::thread::hardware_concurrency() >= 16) Ts.push_back(16);

    double base_ips = 0.0;
    for (int T : Ts) {
        auto r = run_p4_one(T, per_thread_inserts);
        if (T == 1) base_ips = r.inserts_per_sec;
        double scale = base_ips > 0 ? r.inserts_per_sec / base_ips : 1.0;
        std::printf("    %4d  %14.0f  %14.0f  %14.2f  %7.2fx\n",
                    r.threads, r.inserts_per_sec, r.lookups_per_sec,
                    1e9 / r.inserts_per_sec, scale);
    }
}

// ── P5 — Torn-claim regression ────────────────────────────────────────────
//
// Writer claims a stream of distinct keys, encoding a per-key generation
// tag in qd_packed.q. Readers race the writer: any successful `find(key)`
// must yield a `load_qd` whose qd_packed.q equals the tag the writer
// committed for that key (because the 128-bit CAS in Cell A binds key
// and qd_packed atomically). Pre-fix (key + qd as separate atomics) the
// reader could observe key == K but qd_packed == 0; post-fix that is
// structurally impossible.
//
// We deliberately skip `publish_info` here — this test exercises Cell A
// only.

void run_p5_torn_claim(int num_readers, uint64_t num_rounds) {
    std::printf("\n[P5] torn-claim regression: %d readers vs 1 writer, %lu rounds\n",
                num_readers, (unsigned long)num_rounds);

    SearchArena arena(num_rounds, 0.5, 40);

    constexpr uint64_t kKeyBase = 0xCAFEBABEDEADBEEFULL;

    std::atomic<uint64_t> rounds_done{0};
    std::atomic<bool>     writer_done{false};
    std::atomic<uint64_t> torn_reads{0};
    std::atomic<uint64_t> successful_reads{0};

    std::thread writer([&]() {
        for (uint64_t r = 0; r < num_rounds; ++r) {
            const uint64_t key = kKeyBase ^ r;
            // Strictly non-zero so a torn read of qd_packed (== 0) is
            // distinguishable from the published value.
            const float tag = static_cast<float>(r) + 1.0f;
            auto [e, claimed] = arena.find_or_claim(key, /*Q=*/tag, /*max_depth=*/0.0f);
            EXPECT(claimed);
            (void)e;
            rounds_done.store(r + 1, std::memory_order_release);
        }
        writer_done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    readers.reserve(num_readers);
    for (int t = 0; t < num_readers; ++t) {
        readers.emplace_back([&, t]() {
            uint64_t rng = 0xC001D00DULL ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL);
            while (!writer_done.load(std::memory_order_acquire)) {
                const uint64_t hi = rounds_done.load(std::memory_order_acquire);
                if (hi == 0) continue;
                const uint64_t r = splitmix64(rng) % hi;
                const uint64_t key = kKeyBase ^ r;
                TTEntry* e = arena.find(key);
                if (!e) continue;
                const catgpt::v2::KeyQd kq = SearchArena::load_qd(e);
                if (kq.key != key) {
                    // Probe found a slot with this key; load_qd must
                    // observe at least the same key. (This branch should
                    // never fire — kept for safety.)
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                auto [q, md] = unpack_qd(kq.qd_packed);
                (void)md;
                const float expected = static_cast<float>(r) + 1.0f;
                if (q != expected) {
                    torn_reads.fetch_add(1, std::memory_order_relaxed);
                } else {
                    successful_reads.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    writer.join();
    for (auto& tt : readers) tt.join();

    std::printf("    successful_reads=%lu  torn_reads=%lu\n",
                (unsigned long)successful_reads.load(),
                (unsigned long)torn_reads.load());
    EXPECT(torn_reads.load() == 0);
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t p1_per_thread = 1ULL << 14;   // 16k
    int      p1_threads    = 8;
    int      p2_threads    = 8;
    uint64_t p2_k_hot      = 1ULL << 10;   // 1024
    uint64_t p2_iters      = 1ULL << 14;   // 16k per thread
    int      p3_readers    = 4;
    uint64_t p3_inserts    = 1ULL << 12;   // 4k
    int      p3_pause_us   = 0;
    uint64_t p4_per_thread = 1ULL << 16;   // 64k
    int      p5_readers    = 4;
    uint64_t p5_rounds     = 1ULL << 16;   // 64k claims

    if (argc > 1) {
        // Allow overriding p4_per_thread for quick smoke runs.
        p4_per_thread = std::strtoull(argv[1], nullptr, 0);
    }

    std::printf("tt_arena_concurrent_bench: hardware_concurrency=%u  TTEntry=%zuB\n",
                std::thread::hardware_concurrency(), sizeof(TTEntry));

    run_p1_correctness(p1_per_thread, p1_threads);
    run_p2_hot_race(p2_threads, p2_k_hot, p2_iters);
    run_p3_reader_writer(p3_readers, p3_inserts, p3_pause_us);
    run_p4_sweep(p4_per_thread);
    run_p5_torn_claim(p5_readers, p5_rounds);

    if (g_failed == 0) {
        std::printf("\nAll concurrency assertions passed.\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d expectation(s) failed.\n", g_failed);
    return 1;
}
