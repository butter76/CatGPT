/**
 * Shared JSON stats printing for MCTS search.
 *
 * Analogous to fractional_mcts/search_stats.hpp but for MCTSNode.
 * Outputs one JSON object per line that the web backend can parse
 * and stream to the browser via SSE.
 *
 * Event types:
 *   "root_eval"       — after root expansion (first simulation)
 *   "search_update"   — periodically during search
 *   "search_complete" — after the search loop finishes
 *
 * Each event contains:
 *   distQ   — root's 81-bin value distribution (raw NN output)
 *   policy  — allocation-based weights for expanded children,
 *             raw prior P for unexpanded children
 *   bestMove, cp, nodes, iteration — search state
 */

#ifndef CATGPT_ENGINE_MCTS_SEARCH_STATS_HPP
#define CATGPT_ENGINE_MCTS_SEARCH_STATS_HPP

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "../../../external/chess-library/include/chess.hpp"
#include "../move_hash.hpp"
#include "../trt_evaluator.hpp"  // VALUE_NUM_BINS
#include "node.hpp"

namespace catgpt {

/**
 * Print a JSON stats line for one MCTS search event.
 *
 * @param out          Output stream (stdout for standalone binary).
 * @param type         Event type string.
 * @param root         Root node (must be non-null and expanded).
 * @param allocations  Budget allocations for visited children.
 *                     Pass empty map for "root_eval" (before allocations exist).
 * @param best_move    Current best move.
 * @param cp           Centipawn evaluation.
 * @param nodes        Total GPU evaluations so far.
 * @param iteration    Current simulation number.
 * @param pv           Principal variation (empty for root_eval).
 */
inline void print_mcts_stats(
    std::ostream& out,
    const char* type,
    const MCTSNode* root,
    const std::unordered_map<chess::Move, float, MoveHash>& allocations,
    chess::Move best_move,
    int cp,
    int nodes,
    int iteration,
    const std::vector<chess::Move>& pv = {}
) {
    std::ostringstream json;
    json << std::setprecision(6);

    json << "{\"type\":\"" << type << "\"";
    json << ",\"bestMove\":\"" << chess::uci::moveToUci(best_move) << "\"";
    json << ",\"cp\":" << cp;
    json << ",\"nodes\":" << nodes;
    json << ",\"iteration\":" << iteration;

    // distQ (81 bins) — root's value_probs from NN
    json << ",\"distQ\":[";
    for (int i = 0; i < VALUE_NUM_BINS; ++i) {
        if (i > 0) json << ",";
        json << root->value_probs[i];
    }
    json << "]";

    // Policy weights: allocation/total for expanded children, raw P for others
    struct Entry {
        std::string move_uci;
        float weight;
        bool has_q;
        float q;
    };
    std::vector<Entry> entries;
    entries.reserve(root->children.size());

    float total_alloc = 0.0f;
    if (!allocations.empty()) {
        for (const auto& [move, alloc] : allocations) {
            total_alloc += alloc;
        }
    }

    for (const auto& [move, child] : root->children) {
        float weight;
        bool has_q = child.N > 0;
        float q = has_q ? -child.Q() : 0.0f;

        if (!allocations.empty() && total_alloc > 0.0f) {
            auto alloc_it = allocations.find(move);
            if (alloc_it != allocations.end()) {
                weight = alloc_it->second / total_alloc;
            } else {
                weight = child.P;
            }
        } else {
            weight = child.P;
        }

        entries.push_back({chess::uci::moveToUci(move), weight, has_q, q});
    }

    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.weight > b.weight; });

    json << ",\"policy\":[";
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) json << ",";
        json << "{\"move\":\"" << entries[i].move_uci
             << "\",\"weight\":" << entries[i].weight;
        if (entries[i].has_q) {
            json << ",\"q\":" << entries[i].q;
        }
        json << "}";
    }
    json << "]";

    // PV (principal variation)
    if (!pv.empty()) {
        json << ",\"pv\":[";
        for (size_t i = 0; i < pv.size(); ++i) {
            if (i > 0) json << ",";
            json << "\"" << chess::uci::moveToUci(pv[i]) << "\"";
        }
        json << "]";
    }

    json << "}";

    out << json.str() << "\n";
    out.flush();
}

}  // namespace catgpt

#endif  // CATGPT_ENGINE_MCTS_SEARCH_STATS_HPP
