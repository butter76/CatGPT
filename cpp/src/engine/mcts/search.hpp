/**
 * MCTS Search Algorithm.
 *
 * Monte Carlo Tree Search using PUCT selection (AlphaZero/Leela Chess Zero style).
 *
 * The search proceeds in four phases:
 *   1. SELECT: Traverse tree using PUCT until reaching a leaf
 *   2. EXPAND: Create children for the leaf with priors from policy network
 *   3. EVALUATE: Get value estimate from value network
 *   4. BACKPROPAGATE: Update N, W along path from leaf to root
 *
 * After search, the move with highest visit count is selected.
 */

#ifndef CATGPT_ENGINE_MCTS_SEARCH_HPP
#define CATGPT_ENGINE_MCTS_SEARCH_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <print>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../../tokenizer.hpp"
#include "../policy.hpp"
#include "../search_algo.hpp"
#include "../trt_evaluator.hpp"
#include "config.hpp"
#include "node.hpp"

namespace catgpt {

/**
 * MCTS search algorithm using PUCT selection.
 *
 * PUCT formula:
 *   U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))
 *
 * Where:
 *   - Q(s,a): Mean value of taking action a from state s
 *   - P(s,a): Prior probability from policy network
 *   - N(s,a): Visit count for this action
 *   - N_parent: Total visits to parent node
 *   - c_puct: Exploration constant
 */
class MCTSSearch : public SearchAlgo {
public:
    /**
     * Construct MCTS search with a TensorRT evaluator.
     *
     * @param evaluator Shared pointer to TensorRT evaluator.
     * @param config MCTS configuration.
     */
    MCTSSearch(std::shared_ptr<TrtEvaluator> evaluator, MCTSConfig config = {})
        : evaluator_(std::move(evaluator))
        , config_(config)
        , board_(STARTPOS_FEN)
        , stop_flag_(false)
        , total_gpu_evals_(0)
    {}

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
        root_.reset();
        total_gpu_evals_ = 0;
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
        root_.reset();  // Invalidate tree after move
    }

    SearchResult search(const SearchLimits& limits) override {
        stop_flag_.store(false, std::memory_order_relaxed);

        auto start_time = std::chrono::steady_clock::now();

        // Generate legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board_);

        SearchResult result;

        if (moves.empty()) {
            // No legal moves - checkmate or stalemate
            result.best_move = chess::Move::NO_MOVE;
            if (board_.inCheck()) {
                result.score = Score::mate(0);  // We are checkmated
            } else {
                result.score = Score::cp(0);  // Stalemate
            }
            return result;
        }

        // Single legal move - return immediately
        if (moves.size() == 1) {
            result.best_move = moves[0];
            result.depth = 1;
            result.nodes = 1;
            return result;
        }

        // Initialize root node
        root_ = std::make_unique<MCTSNode>();
        total_gpu_evals_ = 0;

        // Determine target number of GPU evaluations
        int target_evals = config_.min_total_evals;
        if (limits.nodes.has_value()) {
            target_evals = std::min(target_evals, static_cast<int>(limits.nodes.value()));
        }

        // Run simulations until we reach the minimum GPU evaluations
        std::int64_t total_nodes = 0;
        while (total_gpu_evals_ < target_evals) {
            if (stop_flag_.load(std::memory_order_relaxed)) {
                break;
            }

            // Check time limit
            if (limits.movetime.has_value()) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
                if (elapsed.count() >= limits.movetime.value()) {
                    break;
                }
            }

            run_simulation();
            ++total_nodes;
        }

        // Select move with highest visit count
        auto best = root_->best_child_by_visits();
        if (best.has_value()) {
            result.best_move = best->first;

            // Q from root's perspective (negate child's Q since it's opponent's view)
            float q = -best->second->Q();
            // Convert Q from [-1, 1] to centipawns (rough approximation)
            int cp = static_cast<int>(q * 100.0f);
            result.score = Score::cp(cp);
        } else {
            // Fallback (shouldn't happen)
            result.best_move = moves[0];
        }

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        result.depth = 1;  // MCTS doesn't have traditional depth
        result.nodes = total_nodes;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (total_nodes * 1000) / elapsed.count();
        }

        // Build PV from most-visited path
        result.pv = root_->get_pv();

        return result;
    }

    void stop() override {
        stop_flag_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

    /**
     * Get the root node for analysis (after search).
     */
    [[nodiscard]] const MCTSNode* root() const {
        return root_.get();
    }

private:
    /**
     * Run a single MCTS simulation.
     */
    void run_simulation() {
        MCTSNode* node = root_.get();
        std::vector<MCTSNode*> path = {node};
        chess::Board scratch_board = board_;

        // SELECT: traverse tree using PUCT until we reach a leaf
        while (node->is_expanded() && !node->is_terminal) {
            auto [move, child] = select_child(node);
            scratch_board.makeMove<true>(move);
            node = child;
            path.push_back(node);
        }

        // EXPAND & EVALUATE
        float value;
        if (node->is_terminal) {
            // Terminal node - use stored value
            value = node->terminal_value.value();
        } else {
            // Expand the leaf and get value
            value = expand_and_evaluate(node, scratch_board);
        }

        // BACKPROPAGATE
        backpropagate(path, value);
    }

    /**
     * Select child with highest PUCT score.
     *
     * Uses Leela-style FPU where unvisited nodes get:
     *     fpu = parent.Q - fpu_reduction * sqrt(visited_policy)
     */
    std::pair<chess::Move, MCTSNode*> select_child(MCTSNode* node) {
        float sqrt_n_parent = node->N > 0 ? std::sqrt(static_cast<float>(node->N)) : 1.0f;

        // Compute Leela-style FPU for unvisited nodes
        // visited_policy = sum of P for children with N > 0
        float visited_policy = 0.0f;
        for (const auto& [move, child] : node->children) {
            if (child.N > 0) {
                visited_policy += child.P;
            }
        }
        float fpu = node->Q() - config_.fpu_reduction * std::sqrt(visited_policy);

        float best_score = -std::numeric_limits<float>::infinity();
        chess::Move best_move = chess::Move::NO_MOVE;
        MCTSNode* best_child = nullptr;

        for (auto& [move, child] : node->children) {
            // Determine Q value from parent's perspective
            // Child's Q/terminal_value is from child's side-to-move (our opponent)
            // So we negate to get our perspective
            float q;
            if (child.is_terminal) {
                q = -child.terminal_value.value();
            } else if (child.N == 0) {
                // Leela-style FPU for unvisited nodes
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
        }

        return {best_move, best_child};
    }

    /**
     * Expand a leaf node and return value estimate.
     */
    float expand_and_evaluate(MCTSNode* node, chess::Board& scratch_board) {
        // Get policy and value from neural network
        auto [policy_priors, value] = evaluate_position(scratch_board);

        // Create children for all legal moves
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, scratch_board);

        for (const auto& move : moves) {
            float prior = 0.0f;
            auto it = policy_priors.find(move);
            if (it != policy_priors.end()) {
                prior = it->second;
            }

            MCTSNode child(prior);

            // Check for terminal states
            scratch_board.makeMove<true>(move);

            if (scratch_board.isGameOver().second != chess::GameResult::NONE) {
                auto [reason, game_result] = scratch_board.isGameOver();

                if (game_result == chess::GameResult::LOSE) {
                    // The side to move at this position is checkmated (they lost)
                    child.is_terminal = true;
                    child.terminal_value = -1.0f;
                } else if (game_result == chess::GameResult::DRAW) {
                    child.is_terminal = true;
                    child.terminal_value = 0.0f;
                }
            }

            scratch_board.unmakeMove(move);

            node->children[move] = std::move(child);
        }

        return value;
    }

    /**
     * Evaluate a position with the neural network.
     */
    std::pair<std::unordered_map<chess::Move, float, MoveHash>, float>
    evaluate_position(const chess::Board& pos) {
        ++total_gpu_evals_;

        std::string fen = pos.getFen();

        // Tokenize
        auto tokens = tokenize<TrtEvaluator::SEQ_LENGTH>(pos, NO_HALFMOVE_CONFIG);

        // Run neural network
        auto nn_output = evaluator_->evaluate(tokens);

        // Convert value from [0, 1] to [-1, 1]
        float value = 2.0f * nn_output.value - 1.0f;

        // Extract policy priors for legal moves
        bool flip = pos.sideToMove() == chess::Color::BLACK;

        chess::Movelist moves;
        chess::movegen::legalmoves(moves, pos);

        // Collect logits for legal moves
        std::vector<std::pair<chess::Move, float>> move_logits;
        move_logits.reserve(moves.size());

        for (const auto& move : moves) {
            auto [from_idx, to_idx] = encode_move_to_policy_index(move, flip);
            int flat_idx = policy_flat_index(from_idx, to_idx);
            float logit = nn_output.policy[flat_idx];
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

        return {policy_priors, value};
    }

    /**
     * Update statistics along the path from leaf to root.
     */
    void backpropagate(std::vector<MCTSNode*>& path, float value) {
        // Walk back from leaf to root
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            MCTSNode* node = *it;
            node->N += 1;
            node->W += value;
            // Flip perspective for parent (opponent's view)
            value = -value;
        }
    }

    std::shared_ptr<TrtEvaluator> evaluator_;
    MCTSConfig config_;
    chess::Board board_;
    std::unique_ptr<MCTSNode> root_;
    std::atomic<bool> stop_flag_;
    int total_gpu_evals_;
};

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_SEARCH_HPP
