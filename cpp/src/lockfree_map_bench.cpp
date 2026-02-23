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
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <print>
#include <random>
#include <vector>

#include "chess.hpp"

namespace {

using namespace chess;

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
    uint64_t key{0};                    // Zobrist key for verification
    float value{0.0f};                  // Evaluation [-1, 1], from side-to-move
    float searched_depth{-1.0f};        // Depth at which this node was last searched

    // === Terminal children (pre-computed during GPU eval) ===
    float best_terminal_value{-2.0f};   // Best terminal value from our POV (-2 = none)
    Move best_terminal_move{Move::NO_MOVE};  // Move achieving best terminal
    float terminal_policy_sum{0.0f};    // Total policy weight of terminal moves

    // === Top 10 non-terminal children ===
    std::array<ChildEntry, TOP_CHILDREN_COUNT> children{};

    // === Remaining moves (beyond top 10) ===
    uint16_t num_remaining_moves{0};    // Count of moves beyond top 10 (each assumed MIN_POLICY)
    uint16_t total_moves{0};            // Total legal moves for this position

    // INVARIANT: Any node in the TT or a children array is already evaluated.
    // No `evaluated` flag needed.

    [[nodiscard]] bool has_terminal_win() const { return best_terminal_value > 0.5f; }
    [[nodiscard]] bool has_terminal() const { return best_terminal_value > -1.5f; }
};

static_assert(sizeof(Node) <= 192, "Node should be at most 192 bytes");

// ─── Node pool (bump allocator) ──────────────────────────────────────

class NodePool {
public:
    explicit NodePool(size_t capacity)
        : nodes_(capacity), next_(0) {}

    uint32_t allocate(uint64_t key) {
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

// ─── Transposition table (sequential, no atomics) ────────────────────

class TranspositionTable {
public:
    explicit TranspositionTable(size_t log2_capacity)
        : mask_((1ULL << log2_capacity) - 1)
        , slots_(1ULL << log2_capacity, UINT32_MAX) {}

    /// Insert a new node index or find existing entry for this key.
    /// @return {node_idx, true} if inserted, {existing_idx, false} if found
    std::pair<uint32_t, bool> insert_or_find(uint64_t key, uint32_t new_idx, const NodePool& pool) {
        size_t idx = key & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            uint32_t& slot = slots_[slot_idx];

            if (slot == UINT32_MAX) {
                slot = new_idx;
                return {new_idx, true};
            }

            if (pool.get(slot).key == key) {
                return {slot, false};
            }
        }

        assert(false && "TranspositionTable: table full");
        __builtin_unreachable();
    }

    [[nodiscard]] uint32_t find(uint64_t key, const NodePool& pool) const {
        size_t idx = key & mask_;

        for (size_t probe = 0; probe <= mask_; ++probe) {
            size_t slot_idx = (idx + probe) & mask_;
            uint32_t slot = slots_[slot_idx];

            if (slot == UINT32_MAX) return UINT32_MAX;
            if (pool.get(slot).key == key) return slot;
        }

        return UINT32_MAX;
    }

    [[nodiscard]] size_t capacity() const { return mask_ + 1; }
    [[nodiscard]] size_t memory_bytes() const {
        return (mask_ + 1) * sizeof(uint32_t);
    }

private:
    size_t mask_;
    std::vector<uint32_t> slots_;
};

// ─── Simulated GPU evaluator ─────────────────────────────────────────

struct EvalRequest {
    uint32_t node_idx;
    Board board;
};

/**
 * Simulates GPU batch inference.
 * Pre-computes terminal children and top-10 policy children.
 */
class SimulatedGPU {
public:
    explicit SimulatedGPU(NodePool& pool, unsigned seed = 42)
        : pool_(pool), rng_(seed), noise_(0.0f, 1.0f) {}

    void queue(uint32_t node_idx, const Board& board) {
        pending_.push_back({node_idx, board});
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
        Node& node = pool_.get(req.node_idx);
        Board& board = req.board;

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
        node.value = value;

        // ─── 2. Generate legal moves ─────────────────────────────
        Movelist moves;
        movegen::legalmoves(moves, board);
        node.total_moves = static_cast<uint16_t>(moves.size());

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

        // ─── 8. Count remaining moves ────────────────────────────
        node.num_remaining_moves = static_cast<uint16_t>(
            std::max(0, static_cast<int>(non_terminal_moves.size()) - TOP_CHILDREN_COUNT));
    }

    NodePool& pool_;
    std::vector<EvalRequest> pending_;
    size_t total_evals_{0};
    size_t batch_count_{0};

    std::mt19937 rng_;
    std::uniform_real_distribution<float> noise_;
};

// ─── Fractional MCTS Search ──────────────────────────────────────────

class FractionalSearch {
public:
    FractionalSearch(TranspositionTable& table, NodePool& pool, SimulatedGPU& gpu)
        : table_(table), pool_(pool), gpu_(gpu) {}

    float search(Board& board, float initial_depth) {
        uint32_t root_idx = get_or_create_evaluated_node(board);
        const Node& root = pool_.get(root_idx);

        // Check for immediate winning terminal move
        if (root.has_terminal_win()) {
            return root.best_terminal_value;
        }

        recursive_search(root_idx, board, initial_depth);

        return pool_.get(root_idx).value;
    }

    [[nodiscard]] uint32_t get_root_idx(const Board& board) const {
        return table_.find(board.hash(), pool_);
    }

    [[nodiscard]] size_t table_hits() const { return table_hits_; }
    [[nodiscard]] size_t new_nodes() const { return new_nodes_; }

private:
    /// Get an existing evaluated node from TT, or create+evaluate+insert a new one.
    /// Guarantees the returned node is evaluated (maintains invariant).
    uint32_t get_or_create_evaluated_node(const Board& board) {
        uint64_t key = board.hash();

        // Check TT first - if found, it's guaranteed to be evaluated
        uint32_t existing = table_.find(key, pool_);
        if (existing != UINT32_MAX) {
            ++table_hits_;
            return existing;
        }

        // Allocate, evaluate, then insert into TT
        uint32_t new_idx = pool_.allocate(key);
        gpu_.queue(new_idx, board);
        gpu_.flush();

        // Now insert into TT (node is evaluated)
        table_.insert_or_find(key, new_idx, pool_);
        ++new_nodes_;

        return new_idx;
    }

    void recursive_search(uint32_t node_idx, Board& board, float depth) {
        Node& node = pool_.get(node_idx);

        if (node.searched_depth >= depth) return;
        node.searched_depth = depth;

        // Can this node have children? (depth > ln(2))
        if (depth <= LN_2) return;

        // ─── Phase 1: Expand children ────────────────────────────
        expand_children(node_idx, board, depth);

        // ─── Phase 2: Recurse into expanded children ─────────────
        Node& node_after_expand = pool_.get(node_idx);  // Re-fetch after potential reallocation

        for (auto& child_entry : node_after_expand.children) {
            if (!child_entry.is_valid()) break;  // No more valid children
            if (!child_entry.is_expanded()) continue;  // Not expanded

            float child_depth = depth + std::log(child_entry.policy);
            if (child_depth >= MIN_CHILD_DEPTH) {
                board.makeMove<true>(child_entry.move);
                recursive_search(child_entry.node_idx, board, child_depth);
                board.unmakeMove(child_entry.move);
            }
        }

        // ─── Phase 3: Handle remaining moves if depth is high enough
        if (depth >= MIN_POLICY_DEPTH_THRESHOLD && node_after_expand.num_remaining_moves > 0) {
            check_remaining_moves(node_idx, board, depth);
        }

        // ─── Phase 4: Compute weighted average ───────────────────
        compute_value(node_idx);
    }

    void expand_children(uint32_t node_idx, Board& board, float depth) {
        // Track children that need evaluation (not yet in TT)
        struct PendingChild {
            size_t child_array_idx;
            uint32_t node_idx;
            uint64_t key;
        };
        std::vector<PendingChild> pending;

        // ─── Pass 1: Check TT, queue new nodes for GPU ───────────
        {
            Node& node = pool_.get(node_idx);

            for (size_t i = 0; i < TOP_CHILDREN_COUNT; ++i) {
                ChildEntry& entry = node.children[i];
                if (!entry.is_valid()) break;
                if (entry.is_expanded()) continue;

                float child_depth = depth + std::log(entry.policy);

                // Force expand first FORCE_EXPAND_COUNT children
                bool force_expand = static_cast<int>(i) < FORCE_EXPAND_COUNT;
                if (!force_expand && child_depth < MIN_CHILD_DEPTH) continue;

                board.makeMove<true>(entry.move);
                uint64_t key = board.hash();

                // Check TT - if found, node is already evaluated
                uint32_t existing = table_.find(key, pool_);
                if (existing != UINT32_MAX) {
                    ++table_hits_;
                    pool_.get(node_idx).children[i].node_idx = existing;
                } else {
                    // Allocate and queue for GPU (don't add to TT yet)
                    uint32_t child_idx = pool_.allocate(key);
                    gpu_.queue(child_idx, board);
                    pending.push_back({i, child_idx, key});
                }

                board.unmakeMove(entry.move);
            }
        }

        // ─── Batch evaluate all pending nodes ────────────────────
        if (!pending.empty()) {
            gpu_.flush();

            // ─── Pass 2: Insert evaluated nodes into TT and set children indices
            for (const auto& p : pending) {
                table_.insert_or_find(p.key, p.node_idx, pool_);
                pool_.get(node_idx).children[p.child_array_idx].node_idx = p.node_idx;
                ++new_nodes_;
            }
        }
    }

    void check_remaining_moves(uint32_t node_idx, Board& board, float depth) {
        // For remaining moves (beyond top 10), check TT for existing evaluations
        // These moves are assumed to have MIN_POLICY weight

        Node& node = pool_.get(node_idx);
        if (node.num_remaining_moves == 0) return;

        float remaining_depth = depth + std::log(MIN_POLICY);
        if (remaining_depth < MIN_CHILD_DEPTH) return;

        // Generate all moves to find the remaining ones
        Movelist moves;
        movegen::legalmoves(moves, board);

        // Collect moves that are in top 10
        std::array<Move, TOP_CHILDREN_COUNT> top_moves{};
        for (size_t i = 0; i < TOP_CHILDREN_COUNT; ++i) {
            if (node.children[i].is_valid()) {
                top_moves[i] = node.children[i].move;
            }
        }

        // Check remaining moves in TT
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

            // Check if terminal (skip those)
            board.makeMove<true>(move);
            auto [reason, result] = board.isGameOver();

            if (result == GameResult::NONE) {
                // Non-terminal: check TT
                uint64_t child_key = board.hash();
                uint32_t child_idx = table_.find(child_key, pool_);

                if (child_idx != UINT32_MAX) {
                    // Found in TT - this child's value will be included in compute_value
                    // via the remaining_moves mechanism
                    ++tt_remaining_hits_;
                }
            }

            board.unmakeMove(move);
        }
    }

    void compute_value(uint32_t node_idx) {
        Node& node = pool_.get(node_idx);

        float weighted_sum = 0.0f;
        float total_weight = 0.0f;

        // Include terminal children
        if (node.has_terminal()) {
            // Use best terminal value weighted by total terminal policy
            weighted_sum += node.best_terminal_value * node.terminal_policy_sum;
            total_weight += node.terminal_policy_sum;
        }

        // Include top 10 children
        for (const auto& entry : node.children) {
            if (!entry.is_valid()) break;
            if (!entry.is_expanded()) continue;

            const Node& child = pool_.get(entry.node_idx);
            // Child value is from opponent's perspective, so negate
            float child_value = -child.value;
            weighted_sum += child_value * entry.policy;
            total_weight += entry.policy;
        }

        // Note: remaining moves are currently not included in the average
        // (they would need TT lookup which we've counted but not stored)

        if (total_weight > 0.0f) {
            node.value = weighted_sum / total_weight;
        }
    }

    TranspositionTable& table_;
    NodePool& pool_;
    SimulatedGPU& gpu_;

    size_t table_hits_{0};
    size_t new_nodes_{0};
    size_t tt_remaining_hits_{0};
};

}  // anonymous namespace

// ─── main ────────────────────────────────────────────────────────────

int main() {
    using namespace chess;

    constexpr float  INITIAL_DEPTH = 10.0f;       // Higher depth to exercise remaining moves
    constexpr size_t TABLE_LOG2    = 24;          // 2^24 = 16M slots
    constexpr size_t POOL_SIZE     = 10'000'000;  // 10M nodes

    std::println("╔══════════════════════════════════════════╗");
    std::println("║   Fractional MCTS Benchmark (v2)         ║");
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
    std::println("Node size          : {} bytes", sizeof(Node));
    std::println("");

    // ── Allocate ─────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    TranspositionTable table(TABLE_LOG2);
    NodePool pool(POOL_SIZE);
    SimulatedGPU gpu(pool);

    auto t1 = std::chrono::high_resolution_clock::now();
    double alloc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::println("Table memory       : {:.2f} MB", table.memory_bytes() / 1e6);
    std::println("Pool memory        : {:.2f} MB", pool.memory_bytes() / 1e6);
    std::println("Allocation         : {:.0f} ms", alloc_ms);
    std::println("");

    // ── Search from starting position ────────────────────────────
    Board root;
    std::println("Starting search from: {}", root.getFen());
    std::println("");

    auto t_search_start = std::chrono::high_resolution_clock::now();

    FractionalSearch search(table, pool, gpu);
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

        std::println("  Remaining moves: {} (each assumed {:.2f}%)",
                     root_node.num_remaining_moves, MIN_POLICY * 100.0f);
        std::println("");

        // Collect and sort children by value
        std::vector<std::tuple<Move, float, float, bool>> move_info;

        for (const auto& entry : root_node.children) {
            if (!entry.is_valid()) break;

            float child_value = 0.0f;
            bool expanded = entry.is_expanded();

            if (expanded) {
                const Node& child = pool.get(entry.node_idx);
                child_value = -child.value;  // Negate for parent's perspective
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
