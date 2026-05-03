/**
 * Lock-free Transposition Table + immutable NodeInfo arena.
 *
 * Two pre-allocated buffers, owned by one `SearchArena`. Both are sized once
 * from `K` (the bound on GPU evals) and never grow. "Free" = destroy the
 * `SearchArena` (two `delete[]`s, no per-node frees).
 *
 *   - `table_` : open-addressed, linear-probed hash of `TTEntry`s. Each
 *                slot's hot mutable per-node state (key, Q, max_depth,
 *                info_offset) lives in three `std::atomic<uint64_t>`
 *                fields so concurrent readers and writers never observe a
 *                torn slot. Capacity = `next_pow2(ceil(K / load_factor))`.
 *
 *   - `arena_` : single bump-allocated byte arena holding cold immutable
 *                per-node info (NodeInfoHeader + MoveInfo[]). The head
 *                pointer is a `std::atomic<uint64_t>`, advanced via
 *                `fetch_add`. Indexed by `uint64_t` byte offset so we are
 *                safe past 2^32 nodes and past 4 GiB. The first
 *                `kArenaReservedBytes` of the buffer are reserved so that
 *                `alloc_raw` never returns 0 — that lets us use offset 0
 *                as the `kNoInfoOffset` "not yet published" sentinel and
 *                get the right state on a fresh TTEntry for free via
 *                value-init.
 *
 * Concurrency protocol (eval-first, batch-allocate workflow):
 *
 *   Writer (post-GPU, batched):
 *     1.  fetch_add `total_bytes` for the whole batch (one global atomic).
 *     2.  Plain stores into the (privately-owned) bytes for each position.
 *     3.  For each position: probe linearly, CAS `key` 0 -> K (acq_rel).
 *         On CAS success, relaxed-store qd_packed, then release-store
 *         info_offset. On CAS failure with key == K, another writer beat
 *         us; we skip (the bytes we wrote are wasted but harmless). On
 *         CAS failure with key == K' != K, keep probing.
 *
 *   Reader:
 *     1.  Probe linearly, comparing `key` (relaxed).
 *     2.  On match, acquire-load info_offset. If kNoInfoOffset the writer
 *         is between its CAS and its release-store; the reader can either
 *         spin briefly (`wait_published`) or treat as miss.
 *     3.  After acquire of info_offset, the header + MoveInfo[] writes
 *         and the writer's relaxed qd_packed store are visible.
 *
 * Terminal children do NOT consume a TT entry or NodeInfo: they live as a
 * `MoveInfo.terminal_kind` byte in the parent. This preserves the invariant
 * `#TTEntries == #GPUevals`.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP

#include <atomic>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>  // _mm_pause
#define CATGPT_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CATGPT_CPU_RELAX() asm volatile("yield" ::: "memory")
#else
#define CATGPT_CPU_RELAX() ((void)0)
#endif

namespace catgpt::v2 {

inline constexpr uint64_t kEmptyKey     = 0ULL;
// Offset 0 is reserved at the start of `arena_` (see `kArenaReservedBytes`
// below) so `alloc_raw` never hands it out. That lets us use 0 as the
// "not yet published" sentinel and have value-init of a `TTEntry` give
// every slot the correct empty-and-unpublished state for free.
inline constexpr uint64_t kNoInfoOffset = 0ULL;

/**
 * Pack/unpack (Q, max_depth) into a single uint64_t for atomic CAS updates.
 * Layout: low 32 bits = Q (float), high 32 bits = max_depth (float).
 *
 * `max_depth` is log-scale: max_depth == log(N) where N is the budget
 * historically requested at this node. All search-side calculations work
 * directly in log-scale to stay numerically safe at large N.
 */
inline uint64_t pack_qd(float q, float max_depth) noexcept {
    uint32_t a;
    uint32_t b;
    std::memcpy(&a, &q, 4);
    std::memcpy(&b, &max_depth, 4);
    return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 32);
}

inline std::pair<float, float> unpack_qd(uint64_t v) noexcept {
    uint32_t a = static_cast<uint32_t>(v);
    uint32_t b = static_cast<uint32_t>(v >> 32);
    float q;
    float md;
    std::memcpy(&q, &a, 4);
    std::memcpy(&md, &b, 4);
    return {q, md};
}

/**
 * The "fresh entry" sentinel for max_depth: lower than any conceivable
 * `start_depth`, so the first descent always wins the
 * `depth > max_depth` gate. Use as the second arg to `set_initial_qd`
 * when a writer claims a slot but hasn't yet decided on a depth (rare;
 * the normal claim-after-eval path stores the real depth directly).
 */
inline constexpr float kFreshMaxDepth = -1.0e30f;

/**
 * Hot mutable per-node entry. Cacheline-sized (32 bytes -> two-per-line on
 * 64B caches; we still align(32) so each slot is aligned).
 *
 * `key == 0` is reserved as the empty-slot sentinel; the (astronomical)
 * zero-zobrist case is handled by `SearchArena::canonicalize_key`.
 */
struct alignas(32) TTEntry {
    std::atomic<uint64_t> key;          // 8: 0 == empty; CAS target for claim
    std::atomic<uint64_t> qd_packed;    // 8: pack_qd(Q, max_depth); CAS-loop updates
    std::atomic<uint64_t> info_offset;  // 8: kNoInfoOffset until published
    uint64_t              _reserved;    // 8
};
static_assert(sizeof(TTEntry) == 32, "TTEntry must be 32 bytes");
static_assert(alignof(TTEntry) == 32, "TTEntry must be 32-byte aligned");
static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "std::atomic<uint64_t> must be lock-free for the lock-free TT to make sense");

/**
 * Per-node header at the start of each NodeInfo block in the arena.
 * Followed in memory by `MoveInfo[num_moves]`. These bytes are written
 * before the slot's `info_offset` is release-published, so they are
 * visible to any reader who acquire-loads `info_offset`.
 */
struct NodeInfoHeader {
    float    variance;   // 4: computed once from NN value distribution
    uint16_t num_moves;  // 2: <= 218 in chess
    uint16_t flags;      // 2: reserved bits
};
static_assert(sizeof(NodeInfoHeader) == 8, "NodeInfoHeader must be 8 bytes");

/**
 * Bytes reserved at the start of `arena_` so that `alloc_raw` never
 * returns offset 0 (which is `kNoInfoOffset`). Sized to a NodeInfoHeader
 * so that subsequent allocations stay 4-byte aligned for the float/u16
 * members of NodeInfoHeader and MoveInfo.
 */
inline constexpr uint64_t kArenaReservedBytes = sizeof(NodeInfoHeader);

/**
 * Per-move slot. Holds the move itself, a terminal-kind byte that coalesces
 * terminal children (no TT entry / NodeInfo for them), and the three priors.
 */
struct MoveInfo {
    uint16_t move;          // 2: chess::Move underlying u16
    uint8_t  terminal_kind; // 1: 0=none, 1=draw/twofold, 2=loss_for_child
    uint8_t  _pad;          // 1
    float    P;             // 4: policy1 (standard, temp 1.0)
    float    P_alloc;       // 4: policy2 (temp 1.3)
    float    P_optimistic;  // 4: policy3 (optimistic head)
};
static_assert(sizeof(MoveInfo) == 16, "MoveInfo must be 16 bytes");

enum TerminalKind : uint8_t {
    kTerminalNone = 0,
    kTerminalDraw = 1,
    kTerminalLossForChild = 2,
};

/**
 * Result of a `find_or_claim` probe. `entry` is never null. `claimed_by_us`
 * is true iff this thread won the CAS to claim a previously-empty slot
 * (and is therefore on the hook to call `publish_info` after writing the
 * arena bytes).
 */
struct ClaimResult {
    TTEntry* entry;
    bool     claimed_by_us;
};

/**
 * Owns the TT and the NodeInfo arena. Sizing is purely a function of `K`,
 * the (a-priori) bound on GPU evals across the lifetime of one arena.
 *
 * Memory:
 *   table_  : capacity_ * sizeof(TTEntry) bytes          (one new[])
 *   arena_  : arena_capacity_bytes_ bytes                (one new[])
 *
 * Tear-down cost = two `delete[]`s, regardless of how many nodes lived.
 */
class SearchArena {
public:
    /**
     * @param k_max_evals          Upper bound on number of GPU evals (== TT entries).
     * @param load_factor          TT load factor target (e.g. 0.5).
     * @param avg_moves_per_node   Used to size the arena. p99 in chess is ~80;
     *                             40 is comfortably above the median (~30).
     */
    explicit SearchArena(uint64_t k_max_evals,
                         double load_factor = 0.5,
                         uint64_t avg_moves_per_node = 40)
        : capacity_(compute_capacity(k_max_evals, load_factor))
        , arena_capacity_bytes_(compute_arena_bytes(k_max_evals, avg_moves_per_node)
                                + kArenaReservedBytes)
    {
        // Value-init zeros each TTEntry: key = kEmptyKey, qd_packed = 0,
        // info_offset = kNoInfoOffset. With both sentinels fixed at 0,
        // every slot starts in the correct "empty + unpublished" state
        // without an extra init pass. Readers must still gate any
        // qd_packed read on `info_offset != kNoInfoOffset` because the
        // writer publishes qd_packed via a relaxed store and hands off
        // visibility through the release-store of info_offset.
        table_ = new TTEntry[capacity_]();
        arena_ = new uint8_t[arena_capacity_bytes_];
    }

    ~SearchArena() {
        delete[] table_;
        delete[] arena_;
    }

    SearchArena(const SearchArena&) = delete;
    SearchArena& operator=(const SearchArena&) = delete;
    SearchArena(SearchArena&&) = delete;
    SearchArena& operator=(SearchArena&&) = delete;

    // ── TT ops ───────────────────────────────────────────────────────────

    /**
     * Lock-free linear-probed find-or-claim.
     *
     * Returns `{entry, claimed_by_us}`. `entry` is never null. If
     * `claimed_by_us` is true, the caller has just won the CAS for this
     * key and is responsible for calling `publish_info(entry, offset)`
     * after writing the arena bytes for the node. If false, another
     * thread either has already claimed and possibly published the slot,
     * or is mid-publish (caller can use `wait_published` to spin).
     *
     * The caller must never insert more than `k_max_evals` distinct keys
     * (the table won't resize and load factor would degrade).
     */
    [[nodiscard]] ClaimResult find_or_claim(uint64_t key) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            TTEntry& e = table_[idx];
            uint64_t observed = e.key.load(std::memory_order_relaxed);
            if (observed == kEmptyKey) {
                uint64_t expected = kEmptyKey;
                if (e.key.compare_exchange_strong(expected, key,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
                {
                    return {&e, /*claimed_by_us=*/true};
                }
                // CAS failed: someone else just claimed this slot. Re-check
                // the now-installed key: if it's ours, we share; otherwise
                // continue probing.
                if (expected == key) {
                    return {&e, /*claimed_by_us=*/false};
                }
                // Fall through: probe next slot.
            } else if (observed == key) {
                return {&e, /*claimed_by_us=*/false};
            }
            idx = (idx + 1) & mask;
        }
    }

    /**
     * Read-only probe. Returns nullptr on miss. Does NOT spin on
     * info_offset; caller decides what to do with an unpublished slot.
     */
    [[nodiscard]] TTEntry* find(uint64_t key) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            TTEntry& e = table_[idx];
            uint64_t observed = e.key.load(std::memory_order_relaxed);
            if (observed == key) return &e;
            if (observed == kEmptyKey) return nullptr;
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] const TTEntry* find(uint64_t key) const noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            const TTEntry& e = table_[idx];
            uint64_t observed = e.key.load(std::memory_order_relaxed);
            if (observed == key) return &e;
            if (observed == kEmptyKey) return nullptr;
            idx = (idx + 1) & mask;
        }
    }

    /**
     * Spin until `entry->info_offset` is published, or `max_spins` pause
     * iterations elapse. Returns true if the slot is published.
     *
     * This is a busy-spin: the writer's window between claiming and
     * publishing is ~tens of ns (one relaxed store + one release store),
     * so spin durations are typically far below `max_spins`.
     */
    static bool wait_published(const TTEntry* entry, uint64_t max_spins = 1024) noexcept {
        for (uint64_t i = 0; i < max_spins; ++i) {
            if (entry->info_offset.load(std::memory_order_acquire) != kNoInfoOffset) {
                return true;
            }
            CATGPT_CPU_RELAX();
        }
        return entry->info_offset.load(std::memory_order_acquire) != kNoInfoOffset;
    }

    /**
     * Number of probes a `find` would perform for `key` (for benchmarks).
     * Returns 1 when the key is found in its home slot. Linear probing.
     */
    [[nodiscard]] uint64_t probe_length(uint64_t key) const noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;
        uint64_t probes = 1;
        while (true) {
            const TTEntry& e = table_[idx];
            uint64_t observed = e.key.load(std::memory_order_relaxed);
            if (observed == key) return probes;
            if (observed == kEmptyKey) return probes;
            idx = (idx + 1) & mask;
            ++probes;
        }
    }

    // ── Per-slot mutators ────────────────────────────────────────────────

    /**
     * Release-publish `info_offset` for a slot whose arena bytes have been
     * fully written. This must be called exactly once per `find_or_claim`
     * that returned `claimed_by_us == true`; do not call with a different
     * offset later.
     */
    static void publish_info(TTEntry* entry, uint64_t info_offset) noexcept {
        assert(info_offset != kNoInfoOffset);
        entry->info_offset.store(info_offset, std::memory_order_release);
    }

    /**
     * Initial qd_packed store (relaxed). Use immediately after a winning
     * `find_or_claim`, *before* `publish_info`. Visibility is established
     * by the subsequent release-store of `info_offset`.
     */
    static void set_initial_qd(TTEntry* entry, float q, float max_depth) noexcept {
        entry->qd_packed.store(pack_qd(q, max_depth), std::memory_order_relaxed);
    }

    /**
     * CAS loop on qd_packed. Updates iff `new_max_depth` strictly exceeds
     * the currently-stored max_depth. Returns true if our value won,
     * false if a concurrent or prior writer already stored a >= max_depth.
     */
    static bool update_qd(TTEntry* entry, float new_q, float new_max_depth) noexcept {
        uint64_t expected = entry->qd_packed.load(std::memory_order_relaxed);
        const uint64_t desired = pack_qd(new_q, new_max_depth);
        while (true) {
            auto [_, old_max_depth] = unpack_qd(expected);
            (void)_;
            if (!(new_max_depth > old_max_depth)) return false;
            if (entry->qd_packed.compare_exchange_weak(expected, desired,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_relaxed))
            {
                return true;
            }
            // expected was reloaded by compare_exchange_weak; loop.
        }
    }

    // ── Arena ops ────────────────────────────────────────────────────────

    /**
     * Reserve `bytes` of arena space and return the base byte offset.
     * One global `fetch_add`. Caller is responsible for laying out
     * NodeInfoHeader + MoveInfo[] within the returned region.
     */
    [[nodiscard]] uint64_t alloc_raw(uint64_t bytes) noexcept {
        const uint64_t offset = arena_head_.fetch_add(bytes, std::memory_order_relaxed);
        assert(offset + bytes <= arena_capacity_bytes_ && "arena overflow");
        return offset;
    }

    /**
     * Convenience for single-node allocations: reserve a NodeInfoHeader +
     * `num_moves` * MoveInfo and pre-fill the header. Caller still needs
     * to fill the MoveInfo[]. Returns the base byte offset.
     */
    [[nodiscard]] uint64_t alloc_node_info(uint16_t num_moves) noexcept {
        const uint64_t bytes = node_info_bytes(num_moves);
        const uint64_t offset = alloc_raw(bytes);
        auto* header = std::launder(reinterpret_cast<NodeInfoHeader*>(arena_ + offset));
        header->variance = 0.0f;
        header->num_moves = num_moves;
        header->flags = 0;
        return offset;
    }

    [[nodiscard]] static constexpr uint64_t node_info_bytes(uint16_t num_moves) noexcept {
        return static_cast<uint64_t>(sizeof(NodeInfoHeader))
             + static_cast<uint64_t>(num_moves) * sizeof(MoveInfo);
    }

    [[nodiscard]] NodeInfoHeader* info_at(uint64_t offset) noexcept {
        assert(offset != kNoInfoOffset);
        return std::launder(reinterpret_cast<NodeInfoHeader*>(arena_ + offset));
    }

    [[nodiscard]] const NodeInfoHeader* info_at(uint64_t offset) const noexcept {
        assert(offset != kNoInfoOffset);
        return std::launder(reinterpret_cast<const NodeInfoHeader*>(arena_ + offset));
    }

    [[nodiscard]] MoveInfo* moves_at(uint64_t offset) noexcept {
        assert(offset != kNoInfoOffset);
        return std::launder(reinterpret_cast<MoveInfo*>(
            arena_ + offset + sizeof(NodeInfoHeader)));
    }

    [[nodiscard]] const MoveInfo* moves_at(uint64_t offset) const noexcept {
        assert(offset != kNoInfoOffset);
        return std::launder(reinterpret_cast<const MoveInfo*>(
            arena_ + offset + sizeof(NodeInfoHeader)));
    }

    // ── Sizing accessors (for benchmarks / sanity checks) ────────────────

    [[nodiscard]] uint64_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] uint64_t arena_capacity_bytes() const noexcept { return arena_capacity_bytes_; }
    [[nodiscard]] uint64_t arena_used_bytes() const noexcept {
        // Subtract the reserved-prefix bookkeeping so callers see "0 means
        // fresh, positive means user-visible bytes consumed".
        return arena_head_.load(std::memory_order_relaxed) - kArenaReservedBytes;
    }
    [[nodiscard]] uint64_t table_bytes() const noexcept { return capacity_ * sizeof(TTEntry); }

private:
    static uint64_t compute_capacity(uint64_t k_max_evals, double load_factor) {
        if (load_factor <= 0.0 || load_factor >= 1.0) load_factor = 0.5;
        const double target = static_cast<double>(k_max_evals) / load_factor;
        uint64_t want = (target < 16.0) ? 16ULL : static_cast<uint64_t>(target) + 1ULL;
        return std::bit_ceil(want);
    }

    static uint64_t compute_arena_bytes(uint64_t k_max_evals, uint64_t avg_moves_per_node) {
        const uint64_t per_node = sizeof(NodeInfoHeader)
                                + avg_moves_per_node * sizeof(MoveInfo);
        return k_max_evals * per_node;
    }

    /**
     * Force the (astronomically rare) `key == 0` case to a non-zero value so
     * we can use 0 as the empty-slot sentinel. We OR in a high bit; this
     * destroys at most one bit of entropy and only ever fires for the literal
     * zero key.
     */
    static uint64_t canonicalize_key(uint64_t k) noexcept {
        return k == 0ULL ? 0x8000000000000000ULL : k;
    }

    TTEntry*              table_      = nullptr;
    uint8_t*              arena_      = nullptr;
    uint64_t              capacity_   = 0;            // pow2
    // Skip the reserved prefix so the first `alloc_raw` returns an offset
    // > 0 and offset 0 stays available as the `kNoInfoOffset` sentinel.
    std::atomic<uint64_t> arena_head_{kArenaReservedBytes};
    uint64_t              arena_capacity_bytes_ = 0;
};

}  // namespace catgpt::v2

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP
