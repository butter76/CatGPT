/**
 * Lock-free Transposition Table + immutable NodeInfo arena.
 *
 * Two pre-allocated buffers, owned by one `SearchArena`. Both are sized once
 * from `K` (the bound on GPU evals) and never grow. "Free" = destroy the
 * `SearchArena` (two `delete[]`s, no per-node frees).
 *
 *   - `table_` : open-addressed, linear-probed hash of `TTEntry`s. Each
 *                slot is two 16-byte cells, each backed by a 128-bit
 *                lock-free atomic:
 *                  * `key_qd`  (Cell A, mutable):
 *                       low  8 B = key (0 == empty),
 *                       high 8 B = qd_packed (Q + max_depth).
 *                       Claimed via a single 128-bit CAS so any reader
 *                       observing key == K necessarily observes the
 *                       matching qd_packed from the same atomic.
 *                  * `info`    (Cell B, immutable post-publish):
 *                       low  4 B = origQ (NN value at expansion time),
 *                       next 4 B = reserved,
 *                       high 8 B = info_offset (kNoInfoOffset until
 *                       published). Set exactly once with a 128-bit
 *                       release-store after the writer has filled the
 *                       arena bytes for this node.
 *                Capacity = `next_pow2(ceil(K / load_factor))`.
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
 *     3.  For each position: probe linearly, 128-bit CAS Cell A from
 *         (kEmptyKey, 0) to (key, pack_qd(Q, max_depth)) (acq_rel). On
 *         success, 128-bit release-store Cell B with (origQ, info_offset).
 *         On CAS failure with key half == K, another writer beat us; we
 *         skip (the bytes we wrote are wasted but harmless). On CAS
 *         failure with key half == K' != K, keep probing.
 *
 *   Reader (Q + max_depth path — never spins):
 *     1.  Probe linearly, comparing the key half of `key_qd` (relaxed).
 *     2.  On match, acquire-load Cell A and unpack qd_packed. The key
 *         match implies qd_packed is from the same 128-bit CAS the
 *         claimer issued, so there is no torn-read window.
 *
 *   Reader (moveInfo path — may spin):
 *     1.  As above, find the matching slot.
 *     2.  Acquire-load Cell B (or `wait_published`). If
 *         `info_offset == kNoInfoOffset`, the writer is between its
 *         claim CAS and its publish release-store; spin briefly.
 *     3.  After acquire of Cell B with a non-sentinel info_offset, the
 *         arena bytes (NodeInfoHeader + MoveInfo[]) and origQ are
 *         visible.
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
#include <optional>
#include <type_traits>
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

// ── 128-bit lock-free atomic ──────────────────────────────────────────────
//
// Thin wrapper over `unsigned __int128` with `alignas(16)`, using the GCC
// `__atomic_*` builtins so the compiler emits inline `cmpxchg16b` on
// x86_64 (`-mcx16`, which g++-14 does NOT auto-enable under `-march=native`
// on AMD EPYC even though the ISA bit is set — pass it explicitly).
//
// We gate on `__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16` rather than
// `__atomic_always_lock_free(16, …)`: the latter returns 0 on gcc even
// with `-mcx16` because gcc reserves "always lock-free" for sizes up to
// 8 bytes. The macro is the canonical signal that the toolchain will
// emit inline `cmpxchg16b` for 16-byte atomic ops.
#if !defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_16)
#error "tt_arena.hpp requires lock-free 16-byte CAS (build with -mcx16; "       \
       "this is needed for the lock-free TTEntry layout)."
#endif

namespace detail {
__extension__ using atomic128_storage_t = unsigned __int128;
}  // namespace detail

template <typename T>
struct alignas(16) Atomic128 {
    static_assert(sizeof(T) == 16, "Atomic128<T> requires sizeof(T) == 16");
    static_assert(std::is_trivially_copyable_v<T>,
                  "Atomic128<T> requires trivially copyable T");

    using Storage = detail::atomic128_storage_t;
    static_assert(sizeof(Storage) == 16);

    Atomic128() noexcept : storage_(0) {}
    explicit Atomic128(T v) noexcept { store(v, std::memory_order_relaxed); }

    Atomic128(const Atomic128&)            = delete;
    Atomic128& operator=(const Atomic128&) = delete;

    [[nodiscard]] T load(std::memory_order o) const noexcept {
        Storage s = __atomic_load_n(const_cast<Storage*>(&storage_),
                                    static_cast<int>(o));
        return from_storage(s);
    }

    void store(T v, std::memory_order o) noexcept {
        __atomic_store_n(&storage_, to_storage(v), static_cast<int>(o));
    }

    bool compare_exchange_strong(T& expected, T desired,
                                 std::memory_order succ,
                                 std::memory_order fail) noexcept {
        Storage e = to_storage(expected);
        const Storage d = to_storage(desired);
        const bool ok = __atomic_compare_exchange_n(
            &storage_, &e, d, /*weak=*/false,
            static_cast<int>(succ), static_cast<int>(fail));
        if (!ok) expected = from_storage(e);
        return ok;
    }

    bool compare_exchange_weak(T& expected, T desired,
                               std::memory_order succ,
                               std::memory_order fail) noexcept {
        Storage e = to_storage(expected);
        const Storage d = to_storage(desired);
        const bool ok = __atomic_compare_exchange_n(
            &storage_, &e, d, /*weak=*/true,
            static_cast<int>(succ), static_cast<int>(fail));
        if (!ok) expected = from_storage(e);
        return ok;
    }

private:
    static T from_storage(Storage s) noexcept {
        T t;
        std::memcpy(&t, &s, sizeof(T));
        return t;
    }
    static Storage to_storage(T v) noexcept {
        Storage s = 0;
        std::memcpy(&s, &v, sizeof(T));
        return s;
    }

    Storage storage_;
};

// ── TT cell layouts ──────────────────────────────────────────────────────

inline constexpr uint64_t kEmptyKey     = 0ULL;
// Offset 0 is reserved at the start of `arena_` (see `kArenaReservedBytes`
// below) so `alloc_raw` never hands it out. That lets us use 0 as the
// "not yet published" sentinel and have value-init of a `TTEntry` give
// every slot the correct empty-and-unpublished state for free.
inline constexpr uint64_t kNoInfoOffset = 0ULL;

/**
 * Pack/unpack (Q, max_depth) into a single uint64_t so the (Q, max_depth)
 * pair travels as the high 8 bytes of `KeyQd` through one 128-bit atomic.
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
 * `depth > max_depth` gate. Use as the second arg to `find_or_claim`
 * when a writer claims a slot but hasn't yet decided on a depth (rare;
 * the normal claim-after-eval path passes the real depth directly).
 */
inline constexpr float kFreshMaxDepth = -1.0e30f;

/**
 * Cell A of a TTEntry. Mutable: the claim CAS installs (key, qd_packed)
 * atomically; `update_qd` then mutates only the qd_packed half via a
 * 128-bit CAS loop on the same cell.
 *
 * `key == 0` is reserved as the empty-slot sentinel; the (astronomical)
 * zero-zobrist case is handled by `SearchArena::canonicalize_key`.
 */
struct KeyQd {
    uint64_t key;        // 8: 0 == empty
    uint64_t qd_packed;  // 8: pack_qd(Q, max_depth)
};
static_assert(sizeof(KeyQd) == 16);
static_assert(alignof(KeyQd) == 8);
static_assert(std::is_trivially_copyable_v<KeyQd>);

/**
 * Cell B of a TTEntry. Immutable post-publish: set exactly once with a
 * 128-bit release-store after the writer has filled the arena bytes.
 * Readers that need moveInfo acquire-load this cell and (optionally)
 * spin until `info_offset != kNoInfoOffset`.
 *
 * `origQ` is the NN-evaluated Q for this position at expansion time. It
 * is preserved here so it remains accessible after `update_qd` overwrites
 * the rolled-up Q in Cell A.
 */
struct InfoCell {
    float    origQ;         // 4: NN value at expansion (immutable)
    uint32_t _reserved;     // 4: reserved for future use (flags / version)
    uint64_t info_offset;   // 8: kNoInfoOffset until published
};
static_assert(sizeof(InfoCell) == 16);
static_assert(alignof(InfoCell) == 8);
static_assert(std::is_trivially_copyable_v<InfoCell>);

/**
 * One TT slot. Two 16-byte cells, each backed by a 128-bit atomic. Stays
 * 32 bytes / 32-byte aligned so two slots fit per 64-byte cache line.
 */
struct alignas(32) TTEntry {
    Atomic128<KeyQd>     key_qd;  // 16: claim CAS + update_qd CAS-loop
    Atomic128<InfoCell>  info;    // 16: single release-store on publish
};
static_assert(sizeof(TTEntry) == 32, "TTEntry must be 32 bytes");
static_assert(alignof(TTEntry) == 32, "TTEntry must be 32-byte aligned");

/**
 * Per-node header at the start of each NodeInfo block in the arena.
 * Followed in memory by `MoveInfo[num_moves]`. These bytes are written
 * before the slot's `info` cell is release-published, so they are
 * visible to any reader who acquire-loads `info` and observes
 * `info_offset != kNoInfoOffset`.
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
 * so that subsequent allocations stay naturally aligned for the u16/u32
 * members of NodeInfoHeader and MoveInfo.
 */
inline constexpr uint64_t kArenaReservedBytes = sizeof(NodeInfoHeader);

enum TerminalKind : uint8_t {
    kTerminalNone = 0,
    kTerminalDraw = 1,
    kTerminalLossForChild = 2,
};

/**
 * Per-move slot — 4 bytes total. Layout-invisible to callers; go through
 * the `P()` / `terminal_kind()` accessors (and `pack()` to construct).
 *
 * Encoding of `_packed` (viewed as IEEE 754 binary16):
 *   - Sign bit = 0  →  non-terminal.  P = fp16_to_f32(_packed).
 *                      `terminal_kind() == kTerminalNone`.
 *   - Sign bit = 1  →  terminal.      P = fp16_to_f32(|_packed| with the
 *                      least-significant mantissa bit forced to 0).
 *                      Terminal kind is encoded in that same LSB:
 *                          LSB = 0  →  kTerminalDraw
 *                          LSB = 1  →  kTerminalLossForChild
 *
 * Why this works:
 *   - Legal softmax priors are in [0, 1]; the sign bit is otherwise unused
 *     so it's a free flag for "this child is terminal".
 *   - Stealing the mantissa LSB costs ~0.1% of the value near 1.0, which
 *     is far below the other sources of noise in the rollup.
 *   - `_Float16` on x86-64 compiles to F16C VCVTPS2PH/VCVTPH2PS (single
 *     cycle each) under -march=native; on hosts without F16C the compiler
 *     emits a short software sequence. Packing is off the hot path
 *     (once per expansion, done pre-CAS while bytes are privately owned);
 *     unpacking is in the hot descent pre-pass but still cheap.
 *
 * Two MoveInfos per 8-byte word, four per 16B cacheline half: the
 * move-iteration hot loop in search reads these densely.
 */
struct MoveInfo {
    uint16_t move;     // 2: chess::Move underlying u16
    uint16_t _packed;  // 2: opaque — holds P + terminal_kind, see above

    [[nodiscard]] TerminalKind terminal_kind() const noexcept {
        if ((_packed & kSignBit) == 0) return kTerminalNone;
        return (_packed & kKindLSB) ? kTerminalLossForChild : kTerminalDraw;
    }

    [[nodiscard]] float P() const noexcept {
        uint16_t bits = static_cast<uint16_t>(_packed & kMagnitudeMask);
        // For terminals, the kind flag lived in the mantissa LSB; mask
        // it out so the residual magnitude is a clean half-float.
        if ((_packed & kSignBit) != 0) {
            bits = static_cast<uint16_t>(bits & ~static_cast<uint16_t>(kKindLSB));
        }
        _Float16 h = std::bit_cast<_Float16>(bits);
        return static_cast<float>(h);
    }

    /**
     * Construct a MoveInfo from (move, P, terminal_kind). P must be
     * non-negative; anything outside [0, fp16_max] saturates per the
     * usual float-to-half conversion rules.
     */
    [[nodiscard]] static MoveInfo pack(uint16_t move, float P, TerminalKind tk) noexcept {
        assert(P >= 0.0f && "MoveInfo::pack expects non-negative P");
        _Float16 h = static_cast<_Float16>(P);
        uint16_t bits = std::bit_cast<uint16_t>(h);
        // Strip any existing sign (e.g. +0 → -0 edge cases) so the
        // sign bit exclusively signals "terminal".
        bits = static_cast<uint16_t>(bits & kMagnitudeMask);
        switch (tk) {
            case kTerminalNone:
                break;
            case kTerminalDraw:
                bits = static_cast<uint16_t>((bits | kSignBit) & ~static_cast<uint16_t>(kKindLSB));
                break;
            case kTerminalLossForChild:
                bits = static_cast<uint16_t>(bits | kSignBit | kKindLSB);
                break;
        }
        return MoveInfo{move, bits};
    }

private:
    static constexpr uint16_t kSignBit       = 0x8000u;
    static constexpr uint16_t kKindLSB       = 0x0001u;
    static constexpr uint16_t kMagnitudeMask = 0x7FFFu;
};
static_assert(sizeof(MoveInfo) == 4, "MoveInfo must be 4 bytes");
static_assert(alignof(MoveInfo) == 2, "MoveInfo must be 2-byte aligned");
static_assert(std::is_trivially_copyable_v<MoveInfo>,
              "MoveInfo is held in arena bytes; must be trivially copyable");
static_assert(sizeof(_Float16) == 2, "_Float16 must be IEEE 754 binary16");

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
        // Value-init zeros each TTEntry: Cell A = (kEmptyKey, 0) and Cell
        // B = (origQ=0, _reserved=0, info_offset=kNoInfoOffset). With
        // both sentinels fixed at 0, every slot starts in the correct
        // "empty + unpublished" state without an extra init pass.
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
     * Lock-free linear-probed find-or-claim with combined initial qd.
     *
     * Returns `{entry, claimed_by_us}`. `entry` is never null. If
     * `claimed_by_us` is true, the caller has just won the 128-bit CAS
     * for this key (and the (q, max_depth) pair was committed atomically
     * with the key in Cell A) — caller is on the hook to call
     * `publish_info(entry, origQ, info_offset)` after writing the arena
     * bytes for the node. If false, another thread either has already
     * claimed and possibly published the slot, or is mid-publish (caller
     * can use `wait_published` to spin).
     *
     * The caller must never insert more than `k_max_evals` distinct keys
     * (the table won't resize and load factor would degrade).
     */
    [[nodiscard]] ClaimResult find_or_claim(uint64_t key, float q, float max_depth) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;
        const KeyQd desired{key, pack_qd(q, max_depth)};

        while (true) {
            TTEntry& e = table_[idx];
            KeyQd observed = e.key_qd.load(std::memory_order_relaxed);
            if (observed.key == kEmptyKey) {
                KeyQd expected{kEmptyKey, 0ULL};
                if (e.key_qd.compare_exchange_strong(
                        expected, desired,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire))
                {
                    return {&e, /*claimed_by_us=*/true};
                }
                // CAS failed: someone else just claimed this slot. Re-check
                // the now-installed key: if it's ours, we share; otherwise
                // continue probing.
                if (expected.key == key) {
                    return {&e, /*claimed_by_us=*/false};
                }
                // Fall through: probe next slot.
            } else if (observed.key == key) {
                return {&e, /*claimed_by_us=*/false};
            }
            idx = (idx + 1) & mask;
        }
    }

    /**
     * Read-only probe. Returns nullptr on miss. Does NOT spin on the
     * info cell; caller decides what to do with an unpublished slot.
     *
     * Returned entry, if non-null, has Cell A's key already installed.
     * Callers that need (Q, max_depth) should use `load_qd`; callers
     * that need moveInfo should use `wait_published` / `load_info`.
     */
    [[nodiscard]] TTEntry* find(uint64_t key) noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            TTEntry& e = table_[idx];
            KeyQd observed = e.key_qd.load(std::memory_order_relaxed);
            if (observed.key == key) return &e;
            if (observed.key == kEmptyKey) return nullptr;
            idx = (idx + 1) & mask;
        }
    }

    [[nodiscard]] const TTEntry* find(uint64_t key) const noexcept {
        key = canonicalize_key(key);
        const uint64_t mask = capacity_ - 1;
        uint64_t idx = key & mask;

        while (true) {
            const TTEntry& e = table_[idx];
            KeyQd observed = e.key_qd.load(std::memory_order_relaxed);
            if (observed.key == key) return &e;
            if (observed.key == kEmptyKey) return nullptr;
            idx = (idx + 1) & mask;
        }
    }

    /**
     * 128-bit acquire-load of Cell A. Returns (key, qd_packed) atomically
     * — any caller observing `result.key == K` necessarily observes the
     * matching qd_packed from the same atomic. Never spins; this is the
     * correct read for callers that only need (Q, max_depth).
     */
    [[nodiscard]] static KeyQd load_qd(const TTEntry* entry) noexcept {
        return entry->key_qd.load(std::memory_order_acquire);
    }

    /**
     * 128-bit acquire-load of Cell B. If `info_offset == kNoInfoOffset`
     * the slot is claimed but not yet published; caller can either
     * spin (`wait_published`) or treat as unavailable.
     */
    [[nodiscard]] static InfoCell load_info(const TTEntry* entry) noexcept {
        return entry->info.load(std::memory_order_acquire);
    }

    /**
     * Spin until Cell B is published, or `max_spins` pause iterations
     * elapse. Returns the published `InfoCell` on success, `nullopt`
     * on timeout.
     *
     * The writer's window between claiming and publishing is two atomic
     * stores plus the arena fills, so spin durations are typically far
     * below `max_spins`.
     */
    [[nodiscard]] static std::optional<InfoCell> wait_published(
            const TTEntry* entry, uint64_t max_spins = 1024) noexcept
    {
        for (uint64_t i = 0; i < max_spins; ++i) {
            InfoCell c = entry->info.load(std::memory_order_acquire);
            if (c.info_offset != kNoInfoOffset) return c;
            CATGPT_CPU_RELAX();
        }
        InfoCell c = entry->info.load(std::memory_order_acquire);
        if (c.info_offset != kNoInfoOffset) return c;
        return std::nullopt;
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
            KeyQd observed = e.key_qd.load(std::memory_order_relaxed);
            if (observed.key == key) return probes;
            if (observed.key == kEmptyKey) return probes;
            idx = (idx + 1) & mask;
            ++probes;
        }
    }

    // ── Per-slot mutators ────────────────────────────────────────────────

    /**
     * Release-publish Cell B for a slot whose arena bytes have been
     * fully written. This must be called exactly once per `find_or_claim`
     * that returned `claimed_by_us == true`; do not call twice.
     *
     * `origQ` is the NN-evaluated Q at expansion time. It becomes
     * immutable once this store retires.
     */
    static void publish_info(TTEntry* entry, float origQ, uint64_t info_offset) noexcept {
        assert(info_offset != kNoInfoOffset);
        const InfoCell c{origQ, /*_reserved=*/0u, info_offset};
        entry->info.store(c, std::memory_order_release);
    }

    /**
     * 128-bit CAS loop on Cell A. Updates the qd_packed half iff
     * `new_max_depth` strictly exceeds the currently-stored max_depth.
     * The key half is kept fixed (the claim already installed it; this
     * loop just re-asserts it as part of the CAS so we only update slots
     * that are actually claimed).
     *
     * Returns true if our value won, false if a concurrent or prior
     * writer already stored a >= max_depth.
     */
    static bool update_qd(TTEntry* entry, float new_q, float new_max_depth) noexcept {
        KeyQd expected = entry->key_qd.load(std::memory_order_relaxed);
        while (true) {
            auto [_, old_max_depth] = unpack_qd(expected.qd_packed);
            (void)_;
            if (!(new_max_depth > old_max_depth)) return false;
            const KeyQd desired{expected.key, pack_qd(new_q, new_max_depth)};
            if (entry->key_qd.compare_exchange_weak(
                    expected, desired,
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
