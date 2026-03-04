/**
 * CatGPT Search Binary — Standalone Fractional MCTS for web UI.
 *
 * Takes a FEN string and runs Fractional MCTS search, printing JSON stats
 * to stdout as the search progresses.  Designed to be spawned by the web
 * backend and parsed line-by-line.
 *
 * Output format (one JSON object per line):
 *   {"type":"root_eval",      "bestMove":"e2e4", "cp":15, "nodes":1,   "iteration":0,   "distQ":[...], "policy":[...]}
 *   {"type":"search_update",  "bestMove":"d2d4", "cp":20, "nodes":50,  "iteration":10,  "distQ":[...], "policy":[...]}
 *   {"type":"search_complete","bestMove":"d2d4", "cp":22, "nodes":400, "iteration":100, "distQ":[...], "policy":[...]}
 *   bestmove d2d4
 *
 * Usage:
 *   catgpt_search <engine_path> <fen> [nodes]
 *
 * Arguments:
 *   engine_path  Path to TensorRT engine file (.trt)
 *   fen          FEN string (quoted)
 *   nodes        Optional: max GPU evaluations (default: 400)
 */

#include <filesystem>
#include <iostream>
#include <memory>
#include <print>
#include <string>

#include "../external/chess-library/include/chess.hpp"
#include "engine/fractional_mcts/search.hpp"
#include "engine/trt_evaluator.hpp"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Disable stdio synchronization for better performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 3) {
        std::println(stderr, "Usage: {} <engine_path> <fen> [nodes]", argv[0]);
        std::println(stderr, "  engine_path  Path to TensorRT engine (.trt)");
        std::println(stderr, "  fen          FEN string (quoted)");
        std::println(stderr, "  nodes        Max GPU evaluations (default: 400)");
        return 1;
    }

    fs::path engine_path = argv[1];
    std::string fen = argv[2];
    int target_nodes = 400;
    if (argc > 3) {
        target_nodes = std::stoi(argv[3]);
        if (target_nodes < 1) target_nodes = 1;
    }

    // Validate engine file
    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}", engine_path.string());
        return 1;
    }

    try {
        // Load TRT engine (status to stderr so stdout stays clean for JSON)
        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        auto evaluator = std::make_shared<catgpt::TrtEvaluator>(engine_path);
        std::println(stderr, "Engine loaded successfully");
        std::cerr.flush();

        // Create search with stats output to stdout
        catgpt::FractionalMCTSConfig config;
        config.min_total_evals = target_nodes;
        catgpt::FractionalMCTSSearch search(evaluator, config);
        search.set_stats_output(std::cout);

        // Set position
        search.reset(fen);

        // Run search with node limit
        catgpt::SearchLimits limits;
        limits.nodes = target_nodes;
        auto result = search.search(limits);

        // Print bestmove line (protocol compatibility)
        if (result.has_move()) {
            std::cout << "bestmove " << chess::uci::moveToUci(result.best_move) << std::endl;
        } else {
            std::cout << "bestmove 0000" << std::endl;
        }

    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
