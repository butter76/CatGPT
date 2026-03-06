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

#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>

#include "../external/chess-library/include/chess.hpp"
#include "engine/fractional_mcts/config.hpp"
#include "selfplay/batch_evaluator.hpp"
#include "selfplay/coroutine_search.hpp"

namespace fs = std::filesystem;

/**
 * Wrapper coroutine that schedules onto the thread pool and runs the search.
 * This bridges the sync main() with the async CoroutineSearch.
 */
coro::task<catgpt::MoveResult> run_search(
    std::shared_ptr<coro::thread_pool> pool,
    catgpt::BatchEvaluator& evaluator,
    const catgpt::FractionalMCTSConfig& config,
    const std::string& fen)
{
    // Schedule onto the worker thread pool
    co_await pool->schedule();

    // Create search instance with stats output to stdout
    catgpt::CoroutineSearch search(evaluator, config, &std::cout);

    // Parse FEN and run search
    chess::Board board(fen);
    co_return co_await search.search_move(board);
}

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
        // Create thread pool (1 thread is sufficient for single-position search)
        auto pool = coro::thread_pool::make_shared(coro::thread_pool::options{
            .thread_count = 1,
        });

        // Create batch evaluator (starts GPU thread internally)
        // Batch size 1 is fine for single search — no batching benefit, but correct behavior
        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        catgpt::BatchEvaluator evaluator(engine_path, pool, /*max_batch_size=*/1);
        std::println(stderr, "Engine loaded successfully");
        std::cerr.flush();

        // Configure search
        catgpt::FractionalMCTSConfig config;
        config.min_total_evals = target_nodes;

        // Run search synchronously via sync_wait
        auto result = coro::sync_wait(run_search(pool, evaluator, config, fen));

        // Print bestmove line (protocol compatibility)
        if (result.best_move != chess::Move::NO_MOVE) {
            std::cout << "bestmove " << chess::uci::moveToUci(result.best_move) << std::endl;
        } else {
            std::cout << "bestmove 0000" << std::endl;
        }

        // Shutdown evaluator gracefully (BatchEvaluator destructor handles this,
        // but explicit shutdown ensures clean exit before pool destruction)
        evaluator.shutdown();

    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
