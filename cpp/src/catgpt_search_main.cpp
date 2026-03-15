/**
 * CatGPT Search Binary — Standalone MCTS for web UI.
 *
 * Takes a FEN string and runs either Fractional MCTS or standard MCTS search,
 * printing JSON stats to stdout as the search progresses.  Designed to be
 * spawned by the web backend and parsed line-by-line.
 *
 * Output format (one JSON object per line):
 *   {"type":"root_eval",      "bestMove":"e2e4", "cp":15, "nodes":1,   "iteration":0,   "distQ":[...], "policy":[...]}
 *   {"type":"search_update",  "bestMove":"d2d4", "cp":20, "nodes":50,  "iteration":10,  "distQ":[...], "policy":[...]}
 *   {"type":"search_complete","bestMove":"d2d4", "cp":22, "nodes":400, "iteration":100, "distQ":[...], "policy":[...]}
 *   bestmove d2d4
 *
 * Usage:
 *   catgpt_search <engine_path> <fen> [nodes] [--mcts]
 *
 * Arguments:
 *   engine_path  Path to TensorRT engine file (.trt)
 *   fen          FEN string (quoted)
 *   nodes        Optional: max GPU evaluations (default: 400)
 *   --mcts       Optional: use standard MCTS instead of Fractional MCTS
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
#include "engine/mcts/config.hpp"
#include "selfplay/batch_evaluator.hpp"
#include "selfplay/coroutine_mcts.hpp"
#include "selfplay/coroutine_search.hpp"

namespace fs = std::filesystem;

coro::task<catgpt::MoveResult> run_fractional_search(
    std::shared_ptr<coro::thread_pool> pool,
    catgpt::BatchEvaluator& evaluator,
    const catgpt::FractionalMCTSConfig& config,
    const std::string& fen)
{
    co_await pool->schedule();
    catgpt::CoroutineSearch search(evaluator, config, &std::cout);
    chess::Board board(fen);
    co_return co_await search.search_move(board);
}

coro::task<catgpt::MoveResult> run_mcts_search(
    std::shared_ptr<coro::thread_pool> pool,
    catgpt::BatchEvaluator& evaluator,
    const catgpt::MCTSConfig& config,
    const std::string& fen)
{
    co_await pool->schedule();
    catgpt::CoroutineMCTS search(evaluator, config, &std::cout);
    chess::Board board(fen);
    co_return co_await search.search_move(board);
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 3) {
        std::println(stderr, "Usage: {} <engine_path> <fen> [nodes] [--mcts]", argv[0]);
        std::println(stderr, "  engine_path  Path to TensorRT engine (.trt)");
        std::println(stderr, "  fen          FEN string (quoted)");
        std::println(stderr, "  nodes        Max GPU evaluations (default: 400)");
        std::println(stderr, "  --mcts       Use standard MCTS instead of Fractional MCTS");
        return 1;
    }

    fs::path engine_path = argv[1];
    std::string fen = argv[2];
    int target_nodes = 400;
    bool use_mcts = false;

    // Parse remaining args (nodes and --mcts can appear in any order)
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mcts") {
            use_mcts = true;
        } else {
            target_nodes = std::stoi(arg);
            if (target_nodes < 1) target_nodes = 1;
        }
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}", engine_path.string());
        return 1;
    }

    try {
        auto pool = coro::thread_pool::make_shared(coro::thread_pool::options{
            .thread_count = 1,
        });

        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        catgpt::BatchEvaluator evaluator(engine_path, pool, /*max_batch_size=*/1);
        std::println(stderr, "Engine loaded successfully ({})",
                     use_mcts ? "MCTS" : "Fractional MCTS");
        std::cerr.flush();

        catgpt::MoveResult result;
        if (use_mcts) {
            catgpt::MCTSConfig config;
            config.min_total_evals = target_nodes;
            result = coro::sync_wait(run_mcts_search(pool, evaluator, config, fen));
        } else {
            catgpt::FractionalMCTSConfig config;
            config.min_total_evals = target_nodes;
            result = coro::sync_wait(run_fractional_search(pool, evaluator, config, fen));
        }

        if (result.best_move != chess::Move::NO_MOVE) {
            std::cout << "bestmove " << chess::uci::moveToUci(result.best_move) << std::endl;
        } else {
            std::cout << "bestmove 0000" << std::endl;
        }

        evaluator.shutdown();

    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
