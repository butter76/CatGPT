/**
 * Transposition Table + immutable NodeInfo arena (PoC).
 *
 * Two pre-allocated buffers, owned by one `SearchArena`. Both are sized once
 * from `K` (the bound on GPU evals) and never grow. "Free" = destroy the
 * `SearchArena` (two `delete[]`s, no per-node frees).
 *
 *   - `table_` : open-addressed, linear-probed hash of `TTEntry`s. Hot mutable
 *                per-node state (Q, max_N, info_offset). Capacity is
 *                `next_pow2(ceil(K / load_factor))`.
 *
 *   - `arena_` : single bump-allocated byte arena holding cold immutable
 *                per-node info: a `NodeInfoHeader` followed by
 *                `MoveInfo[num_moves]`. Indexed by `uint64_t` byte offset
 *                so we are safe past 2^32 nodes and past 4 GiB.
 *
 * Terminal children do NOT consume a TT entry or NodeInfo: they live as a
 * `MoveInfo.terminal_kind` byte in the parent. This preserves the invariant
 * `#TTEntries == #GPUevals`.
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>

namespace catgpt::v2 {

inline constexpr uint64_t kEmptyKey = 0ULL;
inline constexpr uint64_t kNoInfoOffset = std::numeric_limits<uint64_t>::max();

/**
 * Hot mutable per-node entry. Cacheline-sized (32 bytes -> two-per-line on
 * 64B caches; we still align(32) so probes stay neat under linear probing).
 *
 * `key == 0` is reserved as the empty-slot sentinel; the (astronomical)
 * zero-zobrist case is handled by `SearchArena::canonicalize_key`.
 */
struct alignas(32) TTEntry {
    uint64_t key;          // 8: zobrist; 0 == empty slot
    float    Q;            // 4: mutable, side-to-move POV
    float    max_N;        // 4: mutable, highest budget searched
    uint64_t info_offset;  // 8: byte offset into NodeInfoArena (kNoInfoOffset = unset)
    uint64_t _reserved;    // 8: reserved (gen counter / lock bit later)
};
static_assert(sizeof(TTEntry) == 32, "TTEntry must be 32 bytes");
static_assert(alignof(TTEntry) == 32, "TTEntry must be 32-byte aligned");

/**
 * Per-node header at the start of each NodeInfo block in the arena.
 * Followed in memory by `MoveInfo[num_moves]`.
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
 * Owns the TT and the NodeInfo arena. Sizing is purely a function of `K`,
 * the (a-priori) bound on GPU evals for one search.
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
        // Single allocation each; value-init zeros the TT (key=0 -> empty).
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
     * Find an existing entry by key, or insert a new empty one.
     * On insert the returned entry has key=k, Q=0, max_N=-1, info_offset=kNoInfoOffset.
     * `inserted` is set to true iff a new slot was claimed.
     *
     * Caller is responsible for never inserting more than `k_max_evals`
     * distinct keys (we don't check; load factor would degrade otherwise).
     */
    [[nodiscard]] TTEntry* find_or_insert(uint64_t key, bool& inserted) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            TTEntry& e = table_[idx];
            if (e.key == kEmptyKey) {
                e.key = key;
                e.Q = 0.0f;
                e.max_N = -1.0f;
                e.info_offset = kNoInfoOffset;
                e._reserved = 0;
                inserted = true;
                return &e;
            }
            if (e.key == key) {
                inserted = false;
                return &e;
            }
            idx = (idx + 1) & mask;
        }
    }

    /**
     * Find an existing entry by key. Returns nullptr on miss.
     */
    [[nodiscard]] TTEntry* find(uint64_t key) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            TTEntry& e = table_[idx];
            if (e.key == key) return &e;
            if (e.key == kEmptyKey) return nullptr;
            idx = (idx + 1) & mask;
        }
    }

    /**
     * Compute the number of probes performed for a given key (for benchmarks).
     * Returns 1 if the key is found in its home slot, etc. Returns the number
     * of slots inspected before hitting an empty slot if the key is missing.
     */
    [[nodiscard]] uint64_t probe_length(uint64_t key) const noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;
        uint64_t probes = 1;
        while (true) {
            const TTEntry& e = table_[idx];
            if (e.key == key) return probes;
            if (e.key == kEmptyKey) return probes;
            idx = (idx + 1) & mask;
            ++probes;
        }
    }

    // ── Arena ops ────────────────────────────────────────────────────────

    /**
     * Bump-allocate a NodeInfoHeader + `num_moves` * MoveInfo. Returns the
     * byte offset (suitable for storing in `TTEntry::info_offset`). The header
     * is zeroed except for `num_moves`; MoveInfos are uninitialized (caller
     * must fill them).
     */
    [[nodiscard]] uint64_t alloc_node_info(uint16_t num_moves) noexcept {
        const uint64_t bytes = static_cast<uint64_t>(sizeof(NodeInfoHeader))
                             + static_cast<uint64_t>(num_moves) * sizeof(MoveInfo);
        const uint64_t offset = arena_head_;
        assert(offset + bytes <= arena_capacity_bytes_ && "arena overflow");
        arena_head_ = offset + bytes;

        auto* header = std::launder(reinterpret_cast<NodeInfoHeader*>(arena_ + offset));
        header->variance = 0.0f;
        header->num_moves = num_moves;
        header->flags = 0;
        return offset;
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
    [[nodiscard]] uint64_t arena_used_bytes() const noexcept { return arena_head_; }
    [[nodiscard]] uint64_t table_bytes() const noexcept { return capacity_ * sizeof(TTEntry); }

private:
    static uint64_t compute_capacity(uint64_t k_max_evals, double load_factor) {
        // capacity = next_pow2(ceil(K / load_factor)); minimum 16 for safety.
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

    TTEntry* table_ = nullptr;
    uint8_t* arena_ = nullptr;
    uint64_t capacity_ = 0;            // pow2
    uint64_t arena_head_ = 0;
    uint64_t arena_capacity_bytes_ = 0;
};

}  // namespace catgpt::v2

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_V2_TT_ARENA_HPP
