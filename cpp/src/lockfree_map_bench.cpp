/**
 * Fractional MCTS Benchmark with Simulated GPU Batching.
 *
 * This is a sequential MCTS-like algorithm using fractional (log-scale) depth:
 *   - Start with initial depth D
 *   - Children get depth: child_depth = parent_depth + ln(probability)
 *   - Nodes with depth > ln(2) ≈ 0.693 can have children
 *   - Force-expand top 2 children; others need child_depth >= 0
 *   - Parent value = weighted average of child values (not minimax)
 *
 * Node structure (no managed memory):
 *   - Pre-computed terminal children info (best terminal value + total terminal policy)
 *   - Top 10 non-terminal children in fixed array
 *   - Remaining moves assumed to have MIN_POLICY (0.30%)
 *   - Remaining moves ignored if depth < ln(1/MIN_POLICY) ≈ 5.81
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <print>
#include <random>
#include <vector>

#include "chess.hpp"

namespace {

using namespace chess;

// ─── Atomic depth+value pair for lock-free updates ───────────────────

struct alignas(8) DepthValue {
    float searched_depth{-1.0f};  // Depth at which this node was last searched
    float value{0.0f};            // Evaluation [-1, 1], from side-to-move
};
static_assert(sizeof(DepthValue) == 8, "DepthValue must be 8 bytes for atomic ops");
static_assert(std::atomic<DepthValue>::is_always_lock_free,
              "DepthValue atomic must be lock-free");

// ─── Constants ───────────────────────────────────────────────────────

constexpr int TOP_CHILDREN_COUNT = 10;
constexpr float MIN_POLICY = 0.003f;  // 0.30% assumed for remaining moves
constexpr float MIN_POLICY_DEPTH_THRESHOLD = 5.809f;  // ln(1/MIN_POLICY) ≈ ln(333.33)
constexpr float LN_2 = 0.693147f;
constexpr float MIN_CHILD_DEPTH = 0.0f;
constexpr int FORCE_EXPAND_COUNT = 2;

// ─── Child entry (fixed size, no heap allocation) ────────────────────

struct ChildEntry {
    Move move{Move::NO_MOVE};
    uint32_t node_idx{UINT32_MAX};  // Index into NodePool (UINT32_MAX = unexpanded)
    float policy{0.0f};

    [[nodiscard]] bool is_valid() const { return policy > 0.0f; }
    [[nodiscard]] bool is_expanded() const { return node_idx != UINT32_MAX; }
};

static_assert(sizeof(ChildEntry) == 12, "ChildEntry should be 12 bytes");

// ─── Node structure (fixed size, no managed memory) ──────────────────

struct Node {
    PackedBoard key{};                  // Exact 24-byte board representation

    // === Atomic depth+value (updated together at end of search) ===
    std::atomic<DepthValue> depth_value{DepthValue{LN_2, 0.0f}};

    // === Terminal children (pre-computed during GPU eval) ===
    float best_terminal_value{-2.0f};   // Best terminal value from our POV (-2 = none)
    Move best_terminal_move{Move::NO_MOVE};  // Move achieving best terminal
    float terminal_policy_sum{0.0f};    // Total policy weight of terminal moves

    // === Top 10 non-terminal children ===
    std::array<ChildEntry, TOP_CHILDREN_COUNT> children{};

    // === Extra moves (beyond top 10, lazily allocated) ===
    uint32_t extra_moves_idx{UINT32_MAX};  // Index into ExtraMovesPool (UINT32_MAX = not allocated)
    uint16_t num_extra_moves{0};           // Count of extra moves (set during GPU eval)

    // INVARIANT: Any node in the TT or a children array is already evaluated.

    // Atomic accessors
    [[nodiscard]] DepthValue load_depth_value(std::memory_order order = std::memory_order_acquire) const {
        return depth_value.load(order);
    }
    void store_depth_value(DepthValue dv, std::memory_order order = std::memory_order_release) {
        depth_value.store(dv, order);
    }

    [[nodiscard]] bool has_terminal_win() const { return best_terminal_value > 0.5f; }
    [[nodiscard]] bool has_terminal() const { return best_terminal_value > -1.5f; }
    [[nodiscard]] bool has_extra_moves_allocated() const { return extra_moves_idx != UINT32_MAX; }
};

static_assert(sizeof(Node) <= 192, "Node should be at most 192 bytes");

// ─── Extra moves pool (bump allocator for ChildEntry arrays) ─────────

class ExtraMovesPool {
public:
    explicit ExtraMovesPool(size_t capacity)
        : entries_(capacity), next_(0) {}

    /// Allocate a contiguous block of ChildEntry slots.
    /// @return Starting index into the pool.
    uint32_t allocate(uint16_t count) {
        uint32_t idx = next_;
        next_ += count;
        if (next_ > entries_.size()) {
            std::println(stderr, "ExtraMovesPool exhausted at index {}", next_);
            std::abort();
        }
        return idx;
    }

    [[nodiscard]] ChildEntry& get(uint32_t idx) { return entries_[idx]; }
    [[nodiscard]] const ChildEntry& get(uint32_t idx) const { return entries_[idx]; }

    [[nodiscard]] size_t allocated() const { return next_; }
    [[nodiscard]] size_t capacity() const { return entries_.size(); }
    [[nodiscard]] size_t memory_bytes() const {
        return entries_.capacity() * sizeof(ChildEntry);
    }

private:
    std::vector<ChildEntry> entries_;
    uint32_t next_;
};

// ─── Node pool (bump allocator) ──────────────────────────────────────

class NodePool {
public:
    explicit NodePool(size_t capacity)
        : nodes_(capacity), next_(0) {}

    uint32_t allocate(const PackedBoard& key) {
        uint32_t idx = next_++;
        if (idx >= nodes_.size()) {
            std::println(stderr, "NodePool exhausted at index {}", idx);
            std::abort();
        }
        nodes_[idx].key = key;
        return idx;
    }

    [[nodiscard]] Node& get(uint32_t idx) { return nodes_[idx]; }
    [[nodiscard]] const Node& get(uint32_t idx) const { return nodes_[idx]; }

    [[nodiscard]] size_t allocated() const { return next_; }
    [[nodiscard]] size_t capacity() const { return nodes_.size(); }
    [[nodiscard]] size_t memory_bytes() const {
        return nodes_.capacity() * sizeof(Node);
    }

private:
    std::vector<Node> nodes_;
    uint32_t next_;
};

// ─── Transposition table (inline keys, no node-pool chase) ───────────

struct TTSlot {
    PackedBoard key{};               // Inline key for comparison without chasing into NodePool
    uint32_t node_idx{UINT32_MAX};   // UINT32_MAX = empty slot

    [[nodiscard]] bool is_empty() const { return node_idx == UINT32_MAX; }
};

static_assert(sizeof(TTSlot) == 28, "TTSlot should be 28 bytes (24-byte key + 4-byte index)");

class TranspositionTable {
public:
    explicit TranspositionTable(size_t log2_capacity)
        : mask_((1ULL << log2_capacity) - 1)
        , slots_(1ULL << log2_capacity) {}

    /// Hash a PackedBoard (24 bytes) down to a uint64_t for slot indexing.
    static uint64_t hash_key(const PackedBoard& key) {
        uint64_t h0, h1, h2;
        std::memcpy(&h0, key.data(), 8);
        std::memcpy(&h1, key.data() + 8, 8);
        std::memcpy(&h2, key.data() + 16, 8);
        return h0 ^ (h1 * 0x9e3779b97f4a7c15ULL) ^ (h2 * 0x517cc1b727220a95ULL);
    }

    /// Insert a new node index or find existing entry for this key.
    /// @return {node_idx, true} if inserted, {existing_idx, false} if found
    std::pair<uint32_t, bool> insert_or_find(const PackedBoard& key, uint32_t new_idx) {
        size_t idx = hash_key(key) & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            TTSlot& slot = slots_[slot_idx];

            if (slot.is_empty()) {
                slot.key = key;
                slot.node_idx = new_idx;
                return {new_idx, true};
            }

            if (slot.key == key) {
                return {slot.node_idx, false};
            }
        }

        assert(false && "TranspositionTable: table full");
        __builtin_unreachable();
    }

    [[nodiscard]] uint32_t find(const PackedBoard& key) const {
        size_t idx = hash_key(key) & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            const TTSlot& slot = slots_[slot_idx];

            if (slot.is_empty()) return UINT32_MAX;
            if (slot.key == key) return slot.node_idx;
        }

        return UINT32_MAX;
    }

    [[nodiscard]] size_t capacity() const { return mask_ + 1; }
    [[nodiscard]] size_t memory_bytes() const {
        return (mask_ + 1) * sizeof(TTSlot);
    }

private:
    size_t mask_;
    std::vector<TTSlot> slots_;
};

// ─── Simulated GPU evaluator ─────────────────────────────────────────

struct EvalRequest {
    uint32_t parent_node_idx;  // UINT32_MAX for root nodes
    PackedBoard key;           // PackedBoard of the child to create
    uint16_t child_index;      // 0-9: top children, 10+: extra moves
};

/**
 * Simulates GPU batch inference.
 * Pre-computes terminal children and top-10 policy children.
 */
class SimulatedGPU {
public:
    explicit SimulatedGPU(NodePool& pool, TranspositionTable& table, ExtraMovesPool& extra_pool,
                          unsigned seed = 42)
        : pool_(pool), table_(table), extra_pool_(extra_pool), rng_(seed), noise_(0.0f, 1.0f) {}

    void queue(uint32_t parent_node_idx, const PackedBoard& key, uint16_t child_index) {
        pending_.push_back({parent_node_idx, key, child_index});
    }

    size_t flush() {
        size_t count = pending_.size();
        ++batch_count_;

        for (auto& req : pending_) {
            evaluate_single(req);
        }

        pending_.clear();
        total_evals_ += count;
        return count;
    }

    [[nodiscard]] size_t total_evals() const { return total_evals_; }
    [[nodiscard]] size_t batch_count() const { return batch_count_; }
    [[nodiscard]] size_t pending_count() const { return pending_.size(); }

private:
    void evaluate_single(EvalRequest& req) {
        // Allocate node in pool and decode board from packed representation
        uint32_t new_idx = pool_.allocate(req.key);
        Node& node = pool_.get(new_idx);
        Board board = Board::Compact::decode(req.key);

        // ─── 1. Compute value from material ──────────────────────
        constexpr float PAWN_V   = 1.0f;
        constexpr float KNIGHT_V = 3.2f;
        constexpr float BISHOP_V = 3.3f;
        constexpr float ROOK_V   = 5.0f;
        constexpr float QUEEN_V  = 9.0f;

        auto count = [&](PieceType pt, Color c) -> float {
            return static_cast<float>(board.pieces(pt, c).count());
        };

        float white = PAWN_V   * count(PieceType::PAWN,   Color::WHITE)
                    + KNIGHT_V * count(PieceType::KNIGHT, Color::WHITE)
                    + BISHOP_V * count(PieceType::BISHOP, Color::WHITE)
                    + ROOK_V   * count(PieceType::ROOK,   Color::WHITE)
                    + QUEEN_V  * count(PieceType::QUEEN,  Color::WHITE);

        float black = PAWN_V   * count(PieceType::PAWN,   Color::BLACK)
                    + KNIGHT_V * count(PieceType::KNIGHT, Color::BLACK)
                    + BISHOP_V * count(PieceType::BISHOP, Color::BLACK)
                    + ROOK_V   * count(PieceType::ROOK,   Color::BLACK)
                    + QUEEN_V  * count(PieceType::QUEEN,  Color::BLACK);

        float material_diff = (white - black) / 10.0f;
        float value = std::tanh(material_diff);
        if (board.sideToMove() == Color::BLACK) {
            value = -value;
        }
        // Store initial value at LN_2 depth (leaf node, no children searched yet)
        node.store_depth_value({LN_2, value});

        // ─── 2. Generate legal moves ─────────────────────────────
        Movelist moves;
        movegen::legalmoves(moves, board);

        // Nodes are never created for terminal positions (handled as terminal children)
        assert(!moves.empty() && "Node created for terminal position");

        // ─── 3. Compute policy scores via softmax ────────────────
        std::vector<std::pair<Move, float>> move_scores;
        move_scores.reserve(moves.size());

        for (const auto& move : moves) {
            float score = 1.0f;
            if (board.isCapture(move)) score += 3.0f;
            if (move.typeOf() == Move::PROMOTION) score += 4.0f;
            score += noise_(rng_) * 0.3f;
            move_scores.emplace_back(move, score);
        }

        // Softmax with temperature
        constexpr float temperature = 0.5f;
        float max_score = -1e9f;
        for (const auto& [m, s] : move_scores) {
            max_score = std::max(max_score, s);
        }
        float sum_exp = 0.0f;
        for (auto& [m, s] : move_scores) {
            s = std::exp((s - max_score) / temperature);
            sum_exp += s;
        }
        for (auto& [m, s] : move_scores) {
            s /= sum_exp;
        }

        // ─── 4. Categorize moves: terminal vs non-terminal ───────
        struct MoveInfo {
            Move move;
            float policy;
            bool is_terminal;
            float terminal_value;  // Only valid if is_terminal
        };
        std::vector<MoveInfo> move_infos;
        move_infos.reserve(moves.size());

        for (const auto& [move, policy] : move_scores) {
            board.makeMove<true>(move);
            auto [reason, result] = board.isGameOver();
            board.unmakeMove(move);

            MoveInfo info{move, policy, false, 0.0f};

            if (result != GameResult::NONE) {
                // Skip 3-fold repetitions entirely
                if (reason == GameResultReason::THREEFOLD_REPETITION) {
                    continue;
                }

                info.is_terminal = true;
                if (result == GameResult::LOSE) {
                    // Opponent just lost = we win (from our perspective after making move)
                    info.terminal_value = 1.0f;
                } else {
                    // Draw
                    info.terminal_value = 0.0f;
                }
            }

            move_infos.push_back(info);
        }

        // ─── 5. Process terminal moves ───────────────────────────
        node.terminal_policy_sum = 0.0f;
        node.best_terminal_value = -2.0f;
        node.best_terminal_move = Move::NO_MOVE;

        std::vector<MoveInfo> non_terminal_moves;
        non_terminal_moves.reserve(move_infos.size());

        for (const auto& info : move_infos) {
            if (info.is_terminal) {
                node.terminal_policy_sum += info.policy;
                if (info.terminal_value > node.best_terminal_value) {
                    node.best_terminal_value = info.terminal_value;
                    node.best_terminal_move = info.move;
                }
            } else {
                non_terminal_moves.push_back(info);
            }
        }

        // ─── 6. Sort non-terminal moves by policy (descending) ───
        std::sort(non_terminal_moves.begin(), non_terminal_moves.end(),
                  [](const auto& a, const auto& b) { return a.policy > b.policy; });

        // ─── 7. Fill top 10 children array ───────────────────────
        for (size_t i = 0; i < TOP_CHILDREN_COUNT; ++i) {
            if (i < non_terminal_moves.size()) {
                node.children[i].move = non_terminal_moves[i].move;
                node.children[i].policy = non_terminal_moves[i].policy;
                node.children[i].node_idx = UINT32_MAX;  // Unexpanded
            } else {
                node.children[i] = {};  // Zero-initialize (invalid entry)
            }
        }

        // ─── 8. Count extra moves (beyond top 10) ───────────────
        node.num_extra_moves = static_cast<uint16_t>(
            std::max(0, static_cast<int>(non_terminal_moves.size()) - TOP_CHILDREN_COUNT));

        // ─── 9. Insert into TT and link to parent ───────────────
        auto [tt_idx, inserted] = table_.insert_or_find(req.key, new_idx);
        uint32_t result_idx = tt_idx;

        if (req.parent_node_idx != UINT32_MAX) {
            Node& parent = pool_.get(req.parent_node_idx);
            if (req.child_index < TOP_CHILDREN_COUNT) {
                parent.children[req.child_index].node_idx = result_idx;
            } else {
                uint16_t extra_i = req.child_index - TOP_CHILDREN_COUNT;
                extra_pool_.get(parent.extra_moves_idx + extra_i).node_idx = result_idx;
            }
        }
    }

    NodePool& pool_;
    TranspositionTable& table_;
    ExtraMovesPool& extra_pool_;
    std::vector<EvalRequest> pending_;
    size_t total_evals_{0};
    size_t batch_count_{0};

    std::mt19937 rng_;
    std::uniform_real_distribution<float> noise_;
};

// ─── Fractional MCTS Search ──────────────────────────────────────────

class FractionalSearch {
public:
    FractionalSearch(TranspositionTable& table, NodePool& pool, SimulatedGPU& gpu,
                     ExtraMovesPool& extra_pool)
        : table_(table), pool_(pool), gpu_(gpu), extra_pool_(extra_pool) {}

    float search(Board& board, float initial_depth) {
        uint32_t root_idx = get_or_create_evaluated_node(board);
        const Node& root = pool_.get(root_idx);

        // Check for immediate winning terminal move
        if (root.has_terminal_win()) {
            return root.best_terminal_value;
        }

        recursive_search(root_idx, initial_depth);

        return pool_.get(root_idx).load_depth_value().value;
    }

    [[nodiscard]] uint32_t get_root_idx(const Board& board) const {
        return table_.find(Board::Compact::encode(board));
    }

    [[nodiscard]] size_t table_hits() const { return table_hits_; }
    [[nodiscard]] size_t new_nodes() const { return new_nodes_; }

private:
    /// Get an existing evaluated node from TT, or create+evaluate+insert a new one.
    /// Guarantees the returned node is evaluated (maintains invariant).
    uint32_t get_or_create_evaluated_node(const Board& board) {
        PackedBoard key = Board::Compact::encode(board);

        // Check TT first - if found, it's guaranteed to be evaluated
        uint32_t existing = table_.find(key);
        if (existing != UINT32_MAX) {
            ++table_hits_;
            return existing;
        }

        // Queue for GPU eval (no parent)
        gpu_.queue(UINT32_MAX, key, 0);
        gpu_.flush();
        ++new_nodes_;

        // GPU allocated, evaluated, and inserted into TT
        return table_.find(key);
    }

    void recursive_search(uint32_t node_idx, float depth) {
        if (!search_expand(node_idx, depth)) return;
        search_recurse(node_idx, depth);
        search_finalize(node_idx, depth);
    }

    /// Preamble + Phase 1: Check depth, allocate extra moves, expand children.
    /// Returns false if search was skipped (depth already sufficient).
    bool search_expand(uint32_t node_idx, float depth) {
        Node& node = pool_.get(node_idx);

        // Load current depth+value atomically
        DepthValue current = node.load_depth_value();
        if (current.searched_depth >= depth) return false;

        // Reconstruct board from packed representation
        Board board = Board::Compact::decode(node.key);

        // Allocate extra moves if depth exceeds threshold and not yet allocated
        if (depth >= MIN_POLICY_DEPTH_THRESHOLD &&
            node.num_extra_moves > 0 &&
            !node.has_extra_moves_allocated()) {
            allocate_extra_moves(node_idx, board);
        }

        // Expand children (top 10 + extra moves)
        expand_children(node_idx, board, depth);

        return true;
    }

    /// Phase 2 + Phase 3: Recurse into all expanded children.
    void search_recurse(uint32_t node_idx, float depth) {
        const Node& node = pool_.get(node_idx);
        [[maybe_unused]] Board board = Board::Compact::decode(node.key);

        // ─── Recurse into expanded top children ──────────────────
        // No make/unmake needed — each child stores its own PackedBoard key
        for (const auto& child_entry : node.children) {
            if (!child_entry.is_valid()) break;
            if (!child_entry.is_expanded()) continue;

            float child_depth = depth + std::log(child_entry.policy);
            recursive_search(child_entry.node_idx, child_depth);
        }

        // ─── Recurse into expanded extra moves ──────────────────
        if (node.has_extra_moves_allocated()) {
            float extra_depth = depth + std::log(MIN_POLICY);
            for (uint16_t i = 0; i < node.num_extra_moves; ++i) {
                const ChildEntry& entry = extra_pool_.get(node.extra_moves_idx + i);
                if (!entry.is_expanded()) continue;
                recursive_search(entry.node_idx, extra_depth);
            }
        }
    }

    /// Phase 4: Compute weighted average value and store atomically.
    void search_finalize(uint32_t node_idx, float depth) {
        Node& node = pool_.get(node_idx);

        DepthValue current = node.load_depth_value();
        float new_value = compute_value(node_idx, current.value);
        node.store_depth_value({depth, new_value});
    }

    void expand_children(uint32_t node_idx, Board& board, float depth) {
        Node& node = pool_.get(node_idx);
        size_t queued = 0;

        // ─── Process top 10 children ─────────────────────────────
        for (size_t i = 0; i < TOP_CHILDREN_COUNT; ++i) {
            ChildEntry& entry = node.children[i];
            if (!entry.is_valid()) break;
            if (entry.is_expanded()) continue;

            float child_depth = depth + std::log(entry.policy);

            // Force expand first FORCE_EXPAND_COUNT children
            bool force_expand = static_cast<int>(i) < FORCE_EXPAND_COUNT;
            if (!force_expand && child_depth < MIN_CHILD_DEPTH) continue;

            board.makeMove<true>(entry.move);
            PackedBoard key = Board::Compact::encode(board);
            board.unmakeMove(entry.move);

            uint32_t existing = table_.find(key);
            if (existing != UINT32_MAX) {
                ++table_hits_;
                entry.node_idx = existing;
            } else {
                gpu_.queue(node_idx, key, static_cast<uint16_t>(i));
                ++queued;
            }
        }

        // ─── Process extra moves (if allocated and depth is high enough) ───
        if (node.has_extra_moves_allocated()) {
            float extra_depth = depth + std::log(MIN_POLICY);
            if (extra_depth >= MIN_CHILD_DEPTH) {
                for (uint16_t i = 0; i < node.num_extra_moves; ++i) {
                    ChildEntry& entry = extra_pool_.get(node.extra_moves_idx + i);
                    if (entry.is_expanded()) continue;

                    board.makeMove<true>(entry.move);
                    PackedBoard key = Board::Compact::encode(board);
                    board.unmakeMove(entry.move);

                    uint32_t existing = table_.find(key);
                    if (existing != UINT32_MAX) {
                        ++table_hits_;
                        entry.node_idx = existing;
                    } else {
                        gpu_.queue(node_idx, key, static_cast<uint16_t>(TOP_CHILDREN_COUNT + i));
                        ++queued;
                    }
                }
            }
        }

        // ─── Batch evaluate all pending nodes ────────────────────
        if (queued > 0) {
            gpu_.flush();
            new_nodes_ += queued;
        }
    }

    /// Allocate and populate extra moves array for a node.
    /// Called once when depth first exceeds MIN_POLICY_DEPTH_THRESHOLD.
    void allocate_extra_moves(uint32_t node_idx, Board& board) {
        Node& node = pool_.get(node_idx);
        if (node.num_extra_moves == 0) return;

        // Generate all legal moves
        Movelist moves;
        movegen::legalmoves(moves, board);

        // Collect top 10 moves for filtering
        std::array<Move, TOP_CHILDREN_COUNT> top_moves{};
        for (size_t i = 0; i < TOP_CHILDREN_COUNT; ++i) {
            if (node.children[i].is_valid()) {
                top_moves[i] = node.children[i].move;
            }
        }

        // Collect extra moves (not in top 10, not terminal)
        std::vector<Move> extra_moves;
        extra_moves.reserve(node.num_extra_moves);

        for (const auto& move : moves) {
            // Skip if in top 10
            bool in_top = false;
            for (const auto& top_move : top_moves) {
                if (move == top_move) {
                    in_top = true;
                    break;
                }
            }
            if (in_top) continue;

            // Skip terminal moves (already handled in GPU eval)
            board.makeMove<true>(move);
            auto [reason, result] = board.isGameOver();
            board.unmakeMove(move);

            if (result != GameResult::NONE) continue;

            extra_moves.push_back(move);
        }

        // Allocate in ExtraMovesPool
        uint32_t idx = extra_pool_.allocate(static_cast<uint16_t>(extra_moves.size()));
        node.extra_moves_idx = idx;

        // Fill entries with MIN_POLICY
        for (size_t i = 0; i < extra_moves.size(); ++i) {
            ChildEntry& entry = extra_pool_.get(idx + i);
            entry.move = extra_moves[i];
            entry.policy = MIN_POLICY;
            entry.node_idx = UINT32_MAX;  // Unexpanded
        }

        ++extra_allocs_;
    }

    /// Compute weighted average value from children.
    /// @param node_idx Index of the node to compute value for
    /// @param current_value Fallback value if no children are expanded
    /// @return The computed value
    [[nodiscard]] float compute_value(uint32_t node_idx, float current_value) {
        const Node& node = pool_.get(node_idx);

        float weighted_sum = 0.0f;
        float total_weight = 0.0f;

        // Include terminal children
        if (node.has_terminal()) {
            weighted_sum += node.best_terminal_value * node.terminal_policy_sum;
            total_weight += node.terminal_policy_sum;
        }

        // Include top 10 children
        for (const auto& entry : node.children) {
            if (!entry.is_valid()) break;
            if (!entry.is_expanded()) continue;

            const Node& child = pool_.get(entry.node_idx);
            float child_value = -child.load_depth_value().value;
            weighted_sum += child_value * entry.policy;
            total_weight += entry.policy;
        }

        // Include extra moves (if allocated)
        if (node.has_extra_moves_allocated()) {
            for (uint16_t i = 0; i < node.num_extra_moves; ++i) {
                const ChildEntry& entry = extra_pool_.get(node.extra_moves_idx + i);
                if (!entry.is_expanded()) continue;

                const Node& child = pool_.get(entry.node_idx);
                float child_value = -child.load_depth_value().value;
                weighted_sum += child_value * entry.policy;
                total_weight += entry.policy;
            }
        }

        if (total_weight > 0.0f) {
            return weighted_sum / total_weight;
        }
        return current_value;
    }

    TranspositionTable& table_;
    NodePool& pool_;
    SimulatedGPU& gpu_;
    ExtraMovesPool& extra_pool_;

    size_t table_hits_{0};
    size_t new_nodes_{0};
    size_t extra_allocs_{0};
};

}  // anonymous namespace

// ─── main ────────────────────────────────────────────────────────────

int main() {
    using namespace chess;

    constexpr float  INITIAL_DEPTH = 17.5f;       // Higher depth to exercise extra moves
    constexpr size_t TABLE_LOG2    = 24;          // 2^24 = 16M slots
    constexpr size_t POOL_SIZE     = 10'000'000;  // 10M nodes
    constexpr size_t EXTRA_POOL_SIZE = 1'000'000; // 1M extra move entries

    std::println("╔══════════════════════════════════════════╗");
    std::println("║   Fractional MCTS Benchmark (v3)         ║");
    std::println("╚══════════════════════════════════════════╝");
    std::println("");
    std::println("Initial depth      : {:.1f}", INITIAL_DEPTH);
    std::println("Expansion rule     : depth > ln(2) ≈ {:.3f}", LN_2);
    std::println("Force expand       : top {} children", FORCE_EXPAND_COUNT);
    std::println("Min child depth    : {:.1f}", MIN_CHILD_DEPTH);
    std::println("Top children       : {}", TOP_CHILDREN_COUNT);
    std::println("Min policy         : {:.2f}% (threshold depth: {:.2f})",
                 MIN_POLICY * 100.0f, MIN_POLICY_DEPTH_THRESHOLD);
    std::println("Table capacity     : {} slots (2^{})", 1ULL << TABLE_LOG2, TABLE_LOG2);
    std::println("Node pool          : {} nodes", POOL_SIZE);
    std::println("Extra moves pool   : {} entries", EXTRA_POOL_SIZE);
    std::println("Node size          : {} bytes", sizeof(Node));
    std::println("");

    // ── Allocate ─────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    TranspositionTable table(TABLE_LOG2);
    NodePool pool(POOL_SIZE);
    ExtraMovesPool extra_pool(EXTRA_POOL_SIZE);
    SimulatedGPU gpu(pool, table, extra_pool);

    auto t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::println("Table memory       : {:.2f} MB", table.memory_bytes() / 1e6);
    std::println("Pool memory        : {:.2f} MB", pool.memory_bytes() / 1e6);
    std::println("Extra pool memory  : {:.2f} MB", extra_pool.memory_bytes() / 1e6);
    std::println("Allocation         : {:.0f} ms", alloc_ms);
    std::println("");

    // ── Search from starting position ────────────────────────────
    Board root;
    std::println("Starting search from: {}", root.getFen());
    std::println("");

    auto t_search_start = std::chrono::high_resolution_clock::now();

    FractionalSearch search(table, pool, gpu, extra_pool);
    float root_value = search.search(root, INITIAL_DEPTH);

    auto t_search_end = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration<double, std::milli>(
                           t_search_end - t_search_start).count();

    // ── Results ──────────────────────────────────────────────────
    std::println("═══════════════════════════════════════════");
    std::println("  RESULTS");
    std::println("═══════════════════════════════════════════");
    std::println("Wall time          : {:.1f} ms ({:.3f} s)",
                 search_ms, search_ms / 1000.0);
    std::println("Root value         : {:+.4f}", root_value);

    int cp = static_cast<int>(90.0f * std::tan(root_value * 1.5637541897f));
    std::println("Root eval (cp)     : {:+d}", cp);

    std::println("GPU evals          : {:L}", gpu.total_evals());
    std::println("GPU batches        : {:L}", gpu.batch_count());
    std::println("Avg batch size     : {:.1f}",
                 gpu.batch_count() > 0
                     ? static_cast<double>(gpu.total_evals()) / gpu.batch_count()
                     : 0.0);
    std::println("New nodes          : {:L}", search.new_nodes());
    std::println("Table hits         : {:L}", search.table_hits());
    std::println("Pool used          : {:L} / {:L} ({:.1f}%)",
                 pool.allocated(), pool.capacity(),
                 100.0 * static_cast<double>(pool.allocated())
                       / static_cast<double>(pool.capacity()));
    std::println("Extra pool used    : {:L} / {:L} ({:.1f}%)",
                 extra_pool.allocated(), extra_pool.capacity(),
                 100.0 * static_cast<double>(extra_pool.allocated())
                       / static_cast<double>(extra_pool.capacity()));
    std::println("Throughput         : {:.2f} K evals/sec",
                 static_cast<double>(gpu.total_evals()) / search_ms);
    std::println("");

    // ── Move evaluations ─────────────────────────────────────────
    std::println("Top moves:");

    uint32_t root_idx = search.get_root_idx(root);
    if (root_idx != UINT32_MAX) {
        const Node& root_node = pool.get(root_idx);

        // Show terminal info
        if (root_node.has_terminal()) {
            std::println("  Terminal: best={:+.2f}, move={}, policy_sum={:.1f}%",
                         root_node.best_terminal_value,
                         uci::moveToUci(root_node.best_terminal_move),
                         root_node.terminal_policy_sum * 100.0f);
        }

        std::println("  Extra moves: {} (each {:.2f}%, allocated: {})",
                     root_node.num_extra_moves, MIN_POLICY * 100.0f,
                     root_node.has_extra_moves_allocated() ? "yes" : "no");
        std::println("");

        // Collect and sort children by value
        std::vector<std::tuple<Move, float, float, bool>> move_info;

        for (const auto& entry : root_node.children) {
            if (!entry.is_valid()) break;

            float child_value = 0.0f;
            bool expanded = entry.is_expanded();

            if (expanded) {
                const Node& child = pool.get(entry.node_idx);
                child_value = -child.load_depth_value().value;  // Negate for parent's perspective
            }

            move_info.emplace_back(entry.move, entry.policy, child_value, expanded);
        }

        std::sort(move_info.begin(), move_info.end(),
                  [](const auto& a, const auto& b) {
                      return std::get<2>(a) > std::get<2>(b);
                  });

        std::println("  {:>5s}  {:>7s}  {:>7s}  {:>6s}  {:>8s}",
                     "move", "policy", "value", "cp", "expanded");
        for (const auto& [move, prob, value, expanded] : move_info) {
            int child_cp = static_cast<int>(90.0f * std::tan(value * 1.5637541897f));
            std::println("  {:<5s}  {:>6.1f}%  {:>+7.4f}  {:>+6d}  {:>8s}",
                         uci::moveToUci(move),
                         prob * 100.0f,
                         value,
                         child_cp,
                         expanded ? "yes" : "no");
        }
    }

    return 0;
}
