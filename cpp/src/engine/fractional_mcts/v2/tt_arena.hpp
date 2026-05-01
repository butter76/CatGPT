/**
 * Lock-free Transposition Table + immutable NodeInfo arena.
 *
 * Two pre-allocated buffers, owned by one `SearchArena`. Both are sized once
 * from `K` (the bound on GPU evals) and never grow. "Free" = destroy the
 * `SearchArena` (two `delete[]`s, no per-node frees).
 *
 *   - `table_` : open-addressed, linear-probed hash of `TTEntry`s. Each
 *                slot's hot mutable per-node state (key, Q, max_N,
 *                info_offset) lives in three `std::atomic<uint64_t>`
 *                fields so concurrent readers and writers never observe a
 *                torn slot. Capacity = `next_pow2(ceil(K / load_factor))`.
 *
 *   - `arena_` : single bump-allocated byte arena holding cold immutable
 *                per-node info (NodeInfoHeader + MoveInfo[]). The head
 *                pointer is a `std::atomic<uint64_t>`, advanced via
 *                `fetch_add`. Indexed by `uint64_t` byte offset so we are
 *                safe past 2^32 nodes and past 4 GiB.
 *
 * Concurrency protocol (eval-first, batch-allocate workflow):
 *
 *   Writer (post-GPU, batched):
 *     1.  fetch_add `total_bytes` for the whole batch (one global atomic).
 *     2.  Plain stores into the (privately-owned) bytes for each position.
 *     3.  For each position: probe linearly, CAS `key` 0 -> K (acq_rel).
 *         On CAS success, relaxed-store qn_packed, then release-store
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
 *         and the writer's relaxed qn_packed store are visible.
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

inline constexpr uint64_t kEmptyKey = 0ULL;
inline constexpr uint64_t kNoInfoOffset = std::numeric_limits<uint64_t>::max();

/**
 * Pack/unpack (Q, max_N) into a single uint64_t for atomic CAS updates.
 * Layout: low 32 bits = Q (float), high 32 bits = max_N (float).
 */
inline uint64_t pack_qn(float q, float max_n) noexcept {
    uint32_t a;
    uint32_t b;
    std::memcpy(&a, &q, 4);
    std::memcpy(&b, &max_n, 4);
    return static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 32);
}

inline std::pair<float, float> unpack_qn(uint64_t v) noexcept {
    uint32_t a = static_cast<uint32_t>(v);
    uint32_t b = static_cast<uint32_t>(v >> 32);
    float q;
    float mn;
    std::memcpy(&q, &a, 4);
    std::memcpy(&mn, &b, 4);
    return {q, mn};
}

inline constexpr uint64_t kInitialQnPacked =
    // Q = 0.0f, max_N = -1.0f
    (0ULL) | (static_cast<uint64_t>(0xBF800000U) << 32);
static_assert((static_cast<uint32_t>(kInitialQnPacked) == 0U),
              "Q half of kInitialQnPacked must be 0.0f bits");

/**
 * Hot mutable per-node entry. Cacheline-sized (32 bytes -> two-per-line on
 * 64B caches; we still align(32) so each slot is aligned).
 *
 * `key == 0` is reserved as the empty-slot sentinel; the (astronomical)
 * zero-zobrist case is handled by `SearchArena::canonicalize_key`.
 */
struct alignas(32) TTEntry {
    std::atomic<uint64_t> key;          // 8: 0 == empty; CAS target for claim
    std::atomic<uint64_t> qn_packed;    // 8: pack_qn(Q, max_N); CAS-loop updates
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
        , arena_capacity_bytes_(compute_arena_bytes(k_max_evals, avg_moves_per_node))
    {
        // value-init zeros each TTEntry: key = 0, qn_packed = 0, info_offset = 0.
        // qn_packed = 0 means (Q=0.0, max_N=0.0), but readers should only consult
        // qn_packed for slots whose info_offset is published (release/acquire), at
        // which point the writer has already stored the real qn_packed. Empty slots
        // are identified by key == 0, not by qn_packed.
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
     * Initial qn_packed store (relaxed). Use immediately after a winning
     * `find_or_claim`, *before* `publish_info`. Visibility is established
     * by the subsequent release-store of `info_offset`.
     */
    static void set_initial_qn(TTEntry* entry, float q, float max_n) noexcept {
        entry->qn_packed.store(pack_qn(q, max_n), std::memory_order_relaxed);
    }

    /**
     * CAS loop on qn_packed. Updates iff `new_max_n` strictly exceeds the
     * currently-stored max_N. Returns true if our value won, false if a
     * concurrent or prior writer already stored a >= max_N.
     */
    static bool update_qn(TTEntry* entry, float new_q, float new_max_n) noexcept {
        uint64_t expected = entry->qn_packed.load(std::memory_order_relaxed);
        const uint64_t desired = pack_qn(new_q, new_max_n);
        while (true) {
            auto [_, old_max_n] = unpack_qn(expected);
            (void)_;
            if (!(new_max_n > old_max_n)) return false;
            if (entry->qn_packed.compare_exchange_weak(expected, desired,
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
        return arena_head_.load(std::memory_order_relaxed);
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
    std::atomic<uint64_t> arena_head_{0};
    uint64_t              arena_capacity_bytes_ = 0;
};

}  // namespace catgpt::v2

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP
