/**
 * Challenger MCTS Search (coroutine-based).
 *
 * This is a COPY of CoroutineMCTS (coroutine_mcts.hpp) that you can
 * freely modify to test MCTS search algorithm variations.  The baseline
 * (CoroutineMCTS) stays unchanged as the control.
 *
 * The SelfPlayRunner pits ChallengerMCTS (engine A) against
 * CoroutineMCTS (engine B) when search_type is MCTS, playing each
 * opening twice with swapped colors.
 *
 * To experiment: edit the search logic in THIS file only.
 * The baseline in coroutine_mcts.hpp should not be touched.
 *
 * MCTS proceeds in simulations:
 *   1. SELECT: Traverse tree using PUCT until reaching a leaf
 *   2. EXPAND: Create children for the leaf with priors from policy network
 *   3. EVALUATE: Get value estimate (origQ) from value network (co_await)
 *   4. BACKPROPAGATE: Update N (visit count) along path from leaf to root
 *   5. CALC Q: Recursively recompute Q values for the tree
 *
 * Q values are computed as:
 *   Q(node) = (origQ * 1 + sum(-child.Q * child.N)) / N
 *
 * After search, the move with highest visit count is selected.
 */

#ifndef CATGPT_SELFPLAY_CHALLENGER_MCTS_HPP
#define CATGPT_SELFPLAY_CHALLENGER_MCTS_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include <coro/task.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "../engine/mcts/config.hpp"
#include "../engine/mcts/node.hpp"
#include "../engine/move_hash.hpp"
#include "../engine/policy.hpp"
#include "../tokenizer.hpp"
#include "batch_evaluator.hpp"
#include "coroutine_search.hpp"  // For MoveResult
#include "eval_request.hpp"

namespace catgpt {

/**
 * Challenger MCTS — edit this to test MCTS variations.
 *
 * Not a long-lived object — create one per search_move() call.
 * This keeps the design simple (no tree reuse between moves).
 */
class ChallengerMCTS {
public:
    ChallengerMCTS(BatchEvaluator& evaluator, const MCTSConfig& config)
        : evaluator_(evaluator)
        , config_(config)
        , total_gpu_evals_(0)
    {}

    /**
     * Search for the best move from the given position.
     * This is a coroutine — it will suspend whenever it needs a GPU eval.
     *
     * @param board The current position (will NOT be modified).
     * @return MoveResult with the best move and evaluation.
     */
    coro::task<MoveResult> search_move(chess::Board board) {
        MoveResult result;

        // Generate legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);

        if (moves.empty()) {
            result.best_move = chess::Move::NO_MOVE;
            result.cp_score = board.inCheck() ? -32000 : 0;
            co_return result;
        }

        // Single legal move — return immediately
        if (moves.size() == 1) {
            result.best_move = moves[0];
            result.gpu_evals = 0;
            co_return result;
        }

        // Initialize root node
        auto root = std::make_unique<MCTSNode>();
        total_gpu_evals_ = 0;

        // Determine target number of GPU evaluations
        int target_evals = config_.min_total_evals;

        // Run simulations until we reach the minimum GPU evaluations
        int total_simulations = 0;
        while (total_gpu_evals_ < target_evals &&
               total_simulations < 25 * target_evals) {
            co_await run_simulation(root.get(), board);
            ++total_simulations;
        }

        // Select move with highest visit count
        auto best = root->best_child_by_visits();
        if (best.has_value()) {
            result.best_move = best->first;

            // Q from root's perspective (negate child's Q since it's opponent's view)
            float q = -best->second->Q();
            // Convert Q from [-1, 1] to centipawns using tangent scaling
            result.cp_score = static_cast<int>(90.0f * std::tan(q * 1.5637541897f));
        } else {
            // Fallback (shouldn't happen)
            result.best_move = moves[0];
        }

        result.gpu_evals = total_gpu_evals_;
        result.iterations = total_simulations;
        co_return result;
    }

private:
    // ─── Single MCTS simulation ─────────────────────────────────────────

    /**
     * Run a single MCTS simulation: SELECT → EXPAND → EVALUATE → BACKPROP → CALC Q.
     */
    coro::task<void> run_simulation(MCTSNode* root, const chess::Board& board) {
        MCTSNode* node = root;
        std::vector<MCTSNode*> path = {node};
        chess::Board scratch_board = board;

        // SELECT: traverse tree using PUCT until we reach a leaf
        while (node->is_expanded() && !node->is_terminal) {
            auto [move, child] = select_child(node);
            scratch_board.makeMove<true>(move);
            node = child;
            path.push_back(node);
        }

        // EXPAND & EVALUATE
        if (!node->is_terminal) {
            co_await expand_and_evaluate(node, scratch_board);
        }
        // Terminal nodes already have origQ set when created

        // BACKPROPAGATE + UPDATE Q: walk leaf→root, increment N and recompute cached_Q
        backpropagate_and_update_q(path);
    }

    // ─── PUCT selection ─────────────────────────────────────────────────

    /**
     * Select child with highest PUCT score.
     *
     * Uses Leela-style FPU where unvisited nodes get:
     *     fpu = parent.Q - fpu_reduction * sqrt(visited_policy)
     */
    std::pair<chess::Move, MCTSNode*> select_child(MCTSNode* node) {
        // Use N-1 since current visit is in progress; safe to be 0 due to FPU
        float sqrt_n_parent = node->N > 1
            ? std::sqrt(static_cast<float>(node->N - 1)) : 0.0f;

        float best_score = -std::numeric_limits<float>::infinity();
        chess::Move best_move = chess::Move::NO_MOVE;
        MCTSNode* best_child = nullptr;

        // Cumulative policy of children before current child (in decreasing P order)
        float cumulative_policy = 0.0f;

        for (auto& [move, child] : node->children) {
            // Determine Q value from parent's perspective
            // Child's Q/terminal_value is from child's side-to-move (our opponent)
            // So we negate to get our perspective
            float q;
            if (child.is_terminal) {
                q = -child.terminal_value.value();
            } else if (child.N == 0) {
                // Per-child FPU: based on policy of higher-ranked siblings
                float fpu = node->Q() - config_.fpu_reduction * std::sqrt(cumulative_policy);
                q = fpu;
            } else {
                q = -child.Q();
            }

            // PUCT formula
            float u = q + config_.c_puct * child.P * sqrt_n_parent / (1.0f + child.N);

            if (u > best_score) {
                best_score = u;
                best_move = move;
                best_child = &child;
            }

            // Add this child's policy to cumulative sum for next iteration
            cumulative_policy += child.P;
        }

        return {best_move, best_child};
    }

    // ─── Expansion & NN evaluation (the suspension point) ───────────────

    /**
     * Expand a leaf node: create children for all legal moves, evaluate
     * the position via the GPU (co_await), and set origQ.
     */
    coro::task<void> expand_and_evaluate(MCTSNode* node, chess::Board& scratch_board) {
        // Get policy and value from neural network
        auto [policy_priors, value, value_probs] =
            co_await evaluate_position(scratch_board);

        // Create children for all legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, scratch_board);

        node->children.reserve(moves.size());

        for (const auto& move : moves) {
            float prior = 0.0f;
            auto it = policy_priors.find(move);
            if (it != policy_priors.end()) {
                prior = it->second;
            }

            MCTSNode child(prior);

            // Check for terminal states
            scratch_board.makeMove<true>(move);

            // Treat 2-fold repetition as a draw (isGameOver only checks 3-fold)
            bool is_twofold = scratch_board.isRepetition(1);
            auto [reason, game_result] = scratch_board.isGameOver();

            if (is_twofold || game_result != chess::GameResult::NONE) {
                child.is_terminal = true;
                if (!is_twofold && game_result == chess::GameResult::LOSE) {
                    // The side to move at this position is checkmated (they lost)
                    child.terminal_value = -1.0f;
                    child.origQ = -1.0f;
                } else {
                    // Draw (stalemate, repetition, insufficient material, etc.)
                    child.terminal_value = 0.0f;
                    child.origQ = 0.0f;
                }
            }

            scratch_board.unmakeMove(move);

            node->children.emplace_back(move, std::move(child));
        }

        // Sort children by decreasing policy (highest P first)
        std::sort(node->children.begin(), node->children.end(),
                  [](const auto& a, const auto& b) { return a.second.P > b.second.P; });

        // Store original NN evaluation for recursive Q calculation
        node->origQ = value;
        node->value_probs = value_probs;
    }

    /**
     * Result of neural network evaluation.
     */
    struct EvalResult {
        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        float value;  // [-1, 1]
        std::array<float, VALUE_NUM_BINS> value_probs;
    };

    /**
     * Evaluate a position with the neural network (the suspension point).
     * Suspends the coroutine until batched inference completes.
     * Results are cached by zobrist hash to avoid redundant GPU evals.
     */
    coro::task<EvalResult> evaluate_position(const chess::Board& pos) {
        uint64_t hash = pos.hash();

        // Check eval cache — reuse previous GPU result for the same position
        RawNNOutput raw;
        auto cache_it = eval_cache_.find(hash);
        if (cache_it != eval_cache_.end()) {
            raw = cache_it->second;
        } else {
            auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);

            // co_await suspends here → GPU thread batches & evaluates
            raw = co_await EvalAwaitable(evaluator_, tokens);
            eval_cache_[hash] = raw;
            ++total_gpu_evals_;
        }

        // Convert value from [0, 1] to [-1, 1]
        float value = 2.0f * raw.value - 1.0f;

        // Extract policy priors for legal moves via softmax
        bool flip = pos.sideToMove() == chess::Color::BLACK;

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, pos);

        std::vector<std::pair<chess::Move, float>> move_logits;
        move_logits.reserve(moves.size());

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = raw.policy[flat_idx];
            move_logits.emplace_back(move, logit);
        }

        // Softmax over legal moves only
        float max_logit = -std::numeric_limits<float>::infinity();
        for (const auto& [move, logit] : move_logits) {
            max_logit = std::max(max_logit, logit);
        }

        float sum_exp = 0.0f;
        for (auto& [move, logit] : move_logits) {
            logit = std::exp(logit - max_logit);  // Numerical stability
            sum_exp += logit;
        }

        std::unordered_map<chess::Move, float, MoveHash> policy_priors;
        for (const auto& [move, exp_logit] : move_logits) {
            policy_priors[move] = exp_logit / sum_exp;
        }

        co_return EvalResult{std::move(policy_priors), value, raw.value_probs};
    }

    // ─── Backpropagation + Q update (merged) ──────────────────────────

    /**
     * Walk the path from leaf to root, incrementing N and recomputing
     * cached_Q at each node.
     *
     * Only nodes on the path have a changed N, so only their cached_Q
     * values are stale.  Processing leaf→root ensures the child on the
     * path is already up-to-date before its parent is recomputed.
     *
     * Complexity: O(path_length × avg_branching_factor) per simulation,
     * instead of O(tree_size) for the old recursive calc_q.
     */
    static void backpropagate_and_update_q(std::vector<MCTSNode*>& path) {
        for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            MCTSNode* node = path[i];
            node->N += 1;

            if (node->is_terminal) {
                node->cached_Q = node->terminal_value.value();
            } else if (!node->is_expanded()) {
                node->cached_Q = node->origQ;
            } else {
                // Q = (origQ * 1 + sum(-child.cached_Q * child.N)) / N
                float sum = node->origQ;
                for (const auto& [move, child] : node->children) {
                    sum += -child.cached_Q * static_cast<float>(child.N);
                }
                node->cached_Q = sum / static_cast<float>(node->N);
            }
        }
    }

    // ─── Members ────────────────────────────────────────────────────────

    BatchEvaluator& evaluator_;
    MCTSConfig config_;
    int total_gpu_evals_;
    std::unordered_map<uint64_t, RawNNOutput> eval_cache_;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_CHALLENGER_MCTS_HPP
