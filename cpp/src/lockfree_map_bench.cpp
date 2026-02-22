/**
 * Lock-Free Hash Table Benchmark.
 *
 * Spawns one thread per root move (20 threads from the starting position).
 * Each thread does iterative deepening minimax (depth 1→5) on its subtree,
 * using a shared lock-free open-addressing hash table as a transposition table.
 *
 * The table uses 64-bit Zobrist keys and linear probing.  Slots store a
 * tag (upper key bits for fast rejection) and an atomic Node pointer.
 * Node pointers are write-once; the Node's eval/depth may be updated
 * when a deeper search is found.
 *
 * Total positions covered: all positions within ply 6 of the starting
 * position (~119M leaf nodes at ply 6 by perft, fewer unique positions
 * due to transpositions).
 */

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <print>
#include <thread>
#include <vector>

#include "chess.hpp"

namespace {

using namespace chess;

// ─── Transposition table node ────────────────────────────────────────

struct Node {
    uint64_t key{0};                    // Full Zobrist key for verification
    std::atomic<int32_t> eval{0};       // Best-known eval (cp, side-to-move POV)
    std::atomic<int8_t>  depth{-1};     // Depth at which eval was computed
};

static_assert(sizeof(Node) == 16, "Node should be 16 bytes");

// ─── Node pool (lock-free bump allocator) ────────────────────────────

class NodePool {
public:
    explicit NodePool(size_t capacity)
        : nodes_(capacity), next_(0) {}

    /// Allocate a node pre-stamped with the given Zobrist key.
    Node* allocate(uint64_t key) {
        size_t idx = next_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= nodes_.size()) {
            std::println(stderr, "NodePool exhausted at index {}", idx);
            std::abort();
        }
        nodes_[idx].key = key;
        return &nodes_[idx];
    }

    [[nodiscard]] size_t allocated() const {
        return next_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] size_t capacity() const { return nodes_.size(); }

    [[nodiscard]] size_t memory_bytes() const {
        return nodes_.capacity() * sizeof(Node);
    }

private:
    std::vector<Node> nodes_;
    std::atomic<size_t> next_;
};

// ─── Lock-free open-addressing hash table ────────────────────────────

class LockFreeTable {
public:
    /// @param log2_capacity  Table will have 2^log2_capacity slots.
    explicit LockFreeTable(size_t log2_capacity)
        : mask_((1ULL << log2_capacity) - 1)
        , slots_(1ULL << log2_capacity) {}

    /// Insert a new node or find the existing entry for this key.
    ///
    /// @param key       Full 64-bit Zobrist key.
    /// @param new_node  Pre-allocated node (key already stamped).
    /// @return {node_ptr, true} if we inserted, {existing_ptr, false} if
    ///         the key was already present.
    std::pair<Node*, bool> insert_or_find(uint64_t key, Node* new_node) {
        size_t idx = key & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            Slot& slot = slots_[slot_idx];

            Node* p = slot.ptr.load(std::memory_order_acquire);

            if (p == nullptr) {
                // Slot looks empty — try to claim it.
                if (slot.ptr.compare_exchange_strong(
                        p, new_node,
                        std::memory_order_acq_rel,
                        std::memory_order_acquire)) {
                    // Won the slot.  Write the tag so future probes can
                    // do a fast 8-byte reject without dereferencing ptr.
                    slot.tag.store(key, std::memory_order_release);
                    return {new_node, true};
                }
                // CAS failed — `p` now holds the winner's pointer.
            }

            // Slot is occupied.  Fast-reject via tag.
            uint64_t t = slot.tag.load(std::memory_order_acquire);
            if (t != 0 && t != key) continue;   // Definitely a different key.

            // Tag matches or hasn't been written yet — verify via Node.
            // (p->key is guaranteed visible because the ptr acquire
            //  synchronises-with the release in the winning CAS, and
            //  the winner set node->key before calling insert_or_find.)
            if (p->key == key) return {p, false};
        }

        assert(false && "LockFreeTable: table full — increase capacity");
        __builtin_unreachable();
    }

    /// Probe the table for an existing entry.
    /// @return  Pointer to the Node if found, nullptr otherwise.
    [[nodiscard]] Node* find(uint64_t key) const {
        size_t idx = key & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            const Slot& slot = slots_[slot_idx];

            Node* p = slot.ptr.load(std::memory_order_acquire);
            if (p == nullptr) return nullptr;   // Empty slot → not in table.

            uint64_t t = slot.tag.load(std::memory_order_acquire);
            if (t != 0 && t != key) continue;

            if (p->key == key) return p;
        }

        return nullptr;
    }

    [[nodiscard]] size_t capacity() const { return mask_ + 1; }

    [[nodiscard]] size_t memory_bytes() const {
        return (mask_ + 1) * sizeof(Slot);
    }

private:
    struct Slot {
        std::atomic<uint64_t> tag{0};
        std::atomic<Node*>    ptr{nullptr};
    };

    static_assert(sizeof(Slot) == 16, "Slot should be 16 bytes");

    size_t mask_;
    std::vector<Slot> slots_;
};

// ─── Material evaluation ─────────────────────────────────────────────

int material_eval(const Board& board) {
    constexpr int PAWN_V   = 100;
    constexpr int KNIGHT_V = 320;
    constexpr int BISHOP_V = 330;
    constexpr int ROOK_V   = 500;
    constexpr int QUEEN_V  = 900;

    auto count = [&](PieceType pt, Color c) -> int {
        return board.pieces(pt, c).count();
    };

    int white = PAWN_V   * count(PieceType::PAWN,   Color::WHITE)
              + KNIGHT_V * count(PieceType::KNIGHT, Color::WHITE)
              + BISHOP_V * count(PieceType::BISHOP, Color::WHITE)
              + ROOK_V   * count(PieceType::ROOK,   Color::WHITE)
              + QUEEN_V  * count(PieceType::QUEEN,  Color::WHITE);

    int black = PAWN_V   * count(PieceType::PAWN,   Color::BLACK)
              + KNIGHT_V * count(PieceType::KNIGHT, Color::BLACK)
              + BISHOP_V * count(PieceType::BISHOP, Color::BLACK)
              + ROOK_V   * count(PieceType::ROOK,   Color::BLACK)
              + QUEEN_V  * count(PieceType::QUEEN,  Color::BLACK);

    int eval = white - black;
    return (board.sideToMove() == Color::WHITE) ? eval : -eval;
}

// ─── Per-thread statistics ───────────────────────────────────────────

struct ThreadStats {
    uint64_t positions{0};       // Total minimax calls
    uint64_t table_hits{0};      // Depth-sufficient hits (early return)
    uint64_t table_inserts{0};   // New entries added to table
    uint64_t pool_waste{0};      // Pool allocations wasted by races
};

// ─── Minimax with transposition table ────────────────────────────────

int minimax(Board& board, int depth,
            LockFreeTable& table, NodePool& pool,
            ThreadStats& stats) {
    ++stats.positions;

    uint64_t key = board.hash();

    // ── Probe table ──────────────────────────────────────────────
    Node* node = table.find(key);
    if (node != nullptr) {
        int8_t nd = node->depth.load(std::memory_order_acquire);
        if (nd >= static_cast<int8_t>(depth)) {
            ++stats.table_hits;
            return node->eval.load(std::memory_order_relaxed);
        }
    }

    // ── Generate legal moves (also detects checkmate / stalemate) ─
    Movelist moves;
    movegen::legalmoves(moves, board);

    int eval;
    int store_depth;

    if (moves.empty()) {
        eval = board.inCheck() ? -30000 : 0;
        store_depth = 127;          // Terminal — valid at any depth.
    } else if (depth == 0) {
        eval = material_eval(board);
        store_depth = 0;
    } else {
        int best = -40000;
        for (const auto& move : moves) {
            board.makeMove<true>(move);
            int score = -minimax(board, depth - 1, table, pool, stats);
            board.unmakeMove(move);
            if (score > best) best = score;
        }
        eval = best;
        store_depth = depth;
    }

    // ── Store / update table entry ───────────────────────────────
    //
    // NOTE: The eval+depth update is NOT jointly atomic.  Two threads
    // racing to update the same node could interleave writes.  This is
    // acceptable for a benchmark — worst case is a slightly stale eval
    // is used for one lookup.  A production engine would pack eval and
    // depth into a single atomic<uint64_t> and CAS them together.

    if (node == nullptr) {
        Node* new_node = pool.allocate(key);
        new_node->eval.store(eval, std::memory_order_relaxed);
        new_node->depth.store(static_cast<int8_t>(store_depth),
                              std::memory_order_release);

        auto [existing, inserted] = table.insert_or_find(key, new_node);
        if (inserted) {
            ++stats.table_inserts;
        } else {
            ++stats.pool_waste;
            // Someone else inserted first.  Update if we searched deeper.
            if (store_depth > existing->depth.load(std::memory_order_acquire)) {
                existing->eval.store(eval, std::memory_order_relaxed);
                existing->depth.store(static_cast<int8_t>(store_depth),
                                      std::memory_order_release);
            }
        }
    } else {
        // Node existed but with insufficient depth — update it.
        if (store_depth > node->depth.load(std::memory_order_acquire)) {
            node->eval.store(eval, std::memory_order_relaxed);
            node->depth.store(static_cast<int8_t>(store_depth),
                              std::memory_order_release);
        }
    }

    return eval;
}

}  // anonymous namespace

// ─── main ────────────────────────────────────────────────────────────

int main() {
    using namespace chess;

    constexpr int    MAX_DEPTH  = 5;
    constexpr size_t TABLE_LOG2 = 27;            // 2^27 ≈ 134M slots
    constexpr size_t POOL_SIZE  = 80'000'000;    // 80M nodes

    std::println("╔══════════════════════════════════════════╗");
    std::println("║   Lock-Free Hash Table Benchmark         ║");
    std::println("╚══════════════════════════════════════════╝");
    std::println("");
    std::println("Table capacity : {} slots (2^{})", 1ULL << TABLE_LOG2, TABLE_LOG2);
    std::println("Node pool      : {} nodes", POOL_SIZE);
    std::println("Max depth/thread: {}", MAX_DEPTH);
    std::println("Threads        : 1 per root move (20 for startpos)");
    std::println("");

    // ── Allocate ─────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    LockFreeTable table(TABLE_LOG2);
    NodePool pool(POOL_SIZE);

    auto t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::println("Table memory   : {:.2f} GB", table.memory_bytes() / 1e9);
    std::println("Pool memory    : {:.2f} GB", pool.memory_bytes() / 1e9);
    std::println("Allocation     : {:.0f} ms", alloc_ms);
    std::println("");

    // ── Root position ────────────────────────────────────────────
    Board root;     // Standard starting position
    Movelist root_moves;
    movegen::legalmoves(root_moves, root);

    size_t num_threads = root_moves.size();
    std::println("Root has {} legal moves → spawning {} threads", num_threads, num_threads);
    std::println("");

    // ── Search ───────────────────────────────────────────────────
    std::vector<ThreadStats> per_thread(num_threads);
    std::vector<int>         scores(num_threads, 0);
    std::vector<double>      thread_ms(num_threads, 0.0);
    std::vector<std::thread> threads;

    auto t_search_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&, i]() {
            auto t_start = std::chrono::high_resolution_clock::now();

            Board child = root;
            child.makeMove<true>(root_moves[i]);

            int score = 0;
            for (int depth = 1; depth <= MAX_DEPTH; ++depth) {
                score = -minimax(child, depth, table, pool, per_thread[i]);
            }
            scores[i] = score;

            auto t_end = std::chrono::high_resolution_clock::now();
            thread_ms[i] = std::chrono::duration<double, std::milli>(
                               t_end - t_start).count();
        });
    }

    for (auto& t : threads) t.join();

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(
                           t_search_end - t_search_start).count();

    // ── Aggregate statistics ─────────────────────────────────────
    ThreadStats total{};
    for (const auto& ts : per_thread) {
        total.positions     += ts.positions;
        total.table_hits    += ts.table_hits;
        total.table_inserts += ts.table_inserts;
        total.pool_waste    += ts.pool_waste;
    }

    std::println("═══════════════════════════════════════════");
    std::println("  RESULTS");
    std::println("═══════════════════════════════════════════");
    std::println("Wall time          : {:.1f} ms ({:.3f} s)",
                 search_ms, search_ms / 1000.0);
    std::println("Positions visited  : {:L}", total.positions);
    std::println("Table inserts      : {:L}", total.table_inserts);
    std::println("Table hits         : {:L} ({:.1f}%)",
                 total.table_hits,
                 100.0 * static_cast<double>(total.table_hits)
                       / std::max(uint64_t{1}, total.positions));
    std::println("Pool waste (races) : {:L}", total.pool_waste);
    std::println("Pool used          : {:L} / {:L} ({:.1f}%)",
                 pool.allocated(), pool.capacity(),
                 100.0 * static_cast<double>(pool.allocated())
                       / static_cast<double>(pool.capacity()));
    std::println("Table load factor  : {:.2f}%",
                 100.0 * static_cast<double>(total.table_inserts)
                       / static_cast<double>(table.capacity()));
    std::println("Throughput         : {:.2f} M positions/sec",
                 static_cast<double>(total.positions) / search_ms / 1000.0);
    std::println("");

    // ── Per-thread summary ───────────────────────────────────────
    std::println("Per-thread breakdown:");
    std::println("  {:>3s}  {:>7s}  {:>12s}  {:>10s}  {:>8s}",
                 "#", "time ms", "positions", "hits", "inserts");
    for (size_t i = 0; i < num_threads; ++i) {
        std::println("  {:>3}  {:>7.0f}  {:>12L}  {:>10L}  {:>8L}",
                     i, thread_ms[i],
                     per_thread[i].positions,
                     per_thread[i].table_hits,
                     per_thread[i].table_inserts);
    }
    std::println("");

    // ── Move evaluations ─────────────────────────────────────────
    std::println("Move evaluations (depth {}):", MAX_DEPTH);

    std::vector<size_t> order(num_threads);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return scores[a] > scores[b];
    });

    for (size_t idx : order) {
        std::println("  {:<5s}  {:>+5d} cp",
                     uci::moveToUci(root_moves[idx]),
                     scores[idx]);
    }

    return 0;
}
