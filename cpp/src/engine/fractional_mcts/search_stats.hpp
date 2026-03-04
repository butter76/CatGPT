/**
 * Shared JSON stats printing for Fractional MCTS search.
 *
 * Used by both the synchronous (UCI / standalone) and coroutine (selfplay)
 * search implementations.  Outputs one JSON object per line that the web
 * backend can parse and stream to the browser.
 *
 * Event types:
 *   "root_eval"       — after NN evaluation of the root position
 *   "search_update"   — when the highest-allocation child changes
 *   "search_complete" — after the search loop finishes
 *
 * Each event contains:
 *   distQ   — root's 81-bin distributional Q
 *   policy  — modified policy weights (allocation/N_adjusted for expanded,
 *             raw prior for unexpanded moves)
 *   bestMove, cp, nodes, iteration — search state
 */

#ifndef CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_STATS_HPP
#define CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_STATS_HPP

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../trt_evaluator.hpp"  // VALUE_NUM_BINS
#include "node.hpp"

namespace catgpt {

/**
 * Print a JSON stats line for one search event.
 *
 * @param out          Output stream (stdout for standalone binary, stderr for selfplay).
 * @param type         Event type string.
 * @param root         Root node (must be non-null).
 * @param allocations  Budget allocations for expanded children.
 *                     Pass empty map for "root_eval" (before any children are expanded).
 * @param N_adjusted   N * total_expanded_child_weight.  Used to compute modified policy
 *                     weights for expanded children.  Ignored when allocations is empty.
 * @param best_move    Current best move.
 * @param cp           Centipawn evaluation.
 * @param nodes        Total GPU evaluations so far.
 * @param iteration    Current iteration number.
 */
inline void print_catgpt_stats(
    std::ostream& out,
    const char* type,
    const FractionalNode* root,
    const std::unordered_map<chess::Move, float, MoveHash>& allocations,
    float N_adjusted,
    chess::Move best_move,
    int cp,
    int nodes,
    int iteration
) {
    std::ostringstream json;
    json << std::setprecision(6);

    json << "{\"type\":\"" << type << "\"";
    json << ",\"bestMove\":\"" << chess::uci::moveToUci(best_move) << "\"";
    json << ",\"cp\":" << cp;
    json << ",\"nodes\":" << nodes;
    json << ",\"iteration\":" << iteration;

    // ── distQ (81 bins) ──────────────────────────────────────────
    json << ",\"distQ\":[";
    for (int i = 0; i < VALUE_NUM_BINS; ++i) {
        if (i > 0) json << ",";
        json << root->distQ[i];
    }
    json << "]";

    // ── Modified policy weights ──────────────────────────────────
    //
    // For expanded children:  allocation_i / N_adjusted
    // For unexpanded moves:   raw prior P_i
    //
    // Collect all legal moves from policy_priors, compute weight,
    // sort descending for readability.

    struct Entry {
        std::string move_uci;
        float weight;
    };
    std::vector<Entry> entries;
    entries.reserve(root->policy_priors.size());

    for (const auto& [move, prior] : root->policy_priors) {
        float weight;
        auto child_it = root->children.find(move);
        if (child_it != root->children.end() && !allocations.empty()) {
            // Expanded child: use search-refined allocation
            auto alloc_it = allocations.find(move);
            if (alloc_it != allocations.end() && N_adjusted > 0.0f) {
                weight = alloc_it->second / N_adjusted;
            } else {
                weight = prior;
            }
        } else {
            // Unexpanded: raw prior
            weight = prior;
        }
        entries.push_back({chess::uci::moveToUci(move), weight});
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.weight > b.weight; });

    json << ",\"policy\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) json << ",";
        json << "{\"move\":\"" << entries[i].move_uci
             << "\",\"weight\":" << entries[i].weight << "}";
    }
    json << "]";

    json << "}";

    out << json.str() << "\n";
    out.flush();
}

/**
 * Compute N_adjusted and allocations for the root, suitable for stats printing.
 *
 * @param root              Root node with expanded children.
 * @param N                 Raw budget from iterative deepening.
 * @param compute_allocs_fn A callable(FractionalNode*, float) -> allocations map.
 *                          (Pass the class's compute_allocations method.)
 * @param[out] allocs       Output allocations.
 * @param[out] N_adjusted   Output adjusted N.
 */
template <typename AllocFn>
void compute_root_stats_allocations(
    FractionalNode* root,
    float N,
    AllocFn&& compute_allocs_fn,
    std::unordered_map<chess::Move, float, MoveHash>& allocs,
    float& N_adjusted
) {
    float total_child_weight = 0.0f;
    for (const auto& [move, child] : root->children) {
        total_child_weight += child.P;
    }
    N_adjusted = N * total_child_weight;
    allocs = compute_allocs_fn(root, N_adjusted);
}

/**
 * Find the best move from allocations (highest allocation).
 */
inline chess::Move best_move_from_allocations(
    const std::unordered_map<chess::Move, float, MoveHash>& allocations
) {
    chess::Move best = chess::Move::NO_MOVE;
    float best_alloc = -1.0f;
    for (const auto& [move, alloc] : allocations) {
        if (alloc > best_alloc) {
            best_alloc = alloc;
            best = move;
        }
    }
    return best;
}

/**
 * Convert a child's Q to centipawns (from parent's perspective).
 */
inline int child_q_to_cp(float child_Q) {
    float q = -child_Q;  // negate: child's Q is from opponent's perspective
    return static_cast<int>(90.0f * std::tan(q * 1.5637541897f));
}

}  // namespace catgpt

#endif  // CATGPT_ENGINE_FRACTIONAL_MCTS_SEARCH_STATS_HPP
