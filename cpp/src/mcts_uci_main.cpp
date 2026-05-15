/**
 * CatGPT MCTS UCI Engine - Main Entry Point
 *
 * Uses CoroutineMCTS (AlphaZero/Leela-style MCTS) backed by a BatchEvaluator
 * for GPU inference.  A thin SearchAlgo adapter bridges the coroutine-based
 * search into the synchronous UCI interface via coro::sync_wait.
 *
 * Usage: catgpt_mcts [engine_path]
 *   engine_path: Path to TensorRT engine file (default: ./catgpt.trt)
 */

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <print>

#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>

#include "../external/chess-library/include/chess.hpp"
#include "engine/mcts/config.hpp"
#include "engine/search_algo.hpp"
#include "selfplay/legacy/batch_evaluator.hpp"
#include "selfplay/coroutine_mcts.hpp"
#include "uci/uci_handler.hpp"

namespace fs = std::filesystem;

namespace catgpt {

/**
 * SearchAlgo adapter for CoroutineMCTS.
 *
 * Wraps the coroutine-based CoroutineMCTS into the synchronous SearchAlgo
 * interface used by UCIHandler.  Each search() call creates a fresh
 * CoroutineMCTS, runs it to completion via coro::sync_wait, and converts
 * the MoveResult into a SearchResult.
 */
class CoroutineMCTSAdapter : public SearchAlgo {
public:
    CoroutineMCTSAdapter(std::shared_ptr<coro::thread_pool> pool,
                         std::shared_ptr<legacy::BatchEvaluator> evaluator,
                         MCTSConfig config = {})
        : pool_(std::move(pool))
        , evaluator_(std::move(evaluator))
        , config_(config)
        , board_(STARTPOS_FEN)
    {}

    void reset(std::string_view fen = STARTPOS_FEN) override {
        board_ = chess::Board(fen);
    }

    void makemove(const chess::Move& move) override {
        board_.makeMove<true>(move);
    }

    SearchResult search(const SearchLimits& limits) override {
        auto start_time = std::chrono::steady_clock::now();

        // Apply node limit if specified, otherwise use config default
        MCTSConfig search_config = config_;
        if (limits.nodes.has_value()) {
            search_config.min_total_evals = static_cast<int>(limits.nodes.value());
        }

        // Run the coroutine search synchronously
        auto move_result = coro::sync_wait(
            run_search(pool_, *evaluator_, search_config, board_));

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time);

        // Convert MoveResult → SearchResult
        SearchResult result;
        result.best_move = move_result.best_move;
        result.score = Score::cp(move_result.cp_score);
        result.depth = move_result.iterations;
        result.nodes = move_result.gpu_evals;
        result.time_ms = elapsed.count();
        if (elapsed.count() > 0) {
            result.nps = (move_result.gpu_evals * 1000) / elapsed.count();
        }
        return result;
    }

    void stop() override {
        // CoroutineMCTS doesn't support early termination — the search
        // runs to completion based on min_total_evals.
    }

    [[nodiscard]] const chess::Board& board() const override {
        return board_;
    }

private:
    /**
     * Wrapper coroutine: schedules onto the thread pool and runs the search.
     */
    static coro::task<MoveResult> run_search(
        std::shared_ptr<coro::thread_pool> pool,
        legacy::BatchEvaluator& evaluator,
        const MCTSConfig& config,
        chess::Board board)
    {
        co_await pool->schedule();
        CoroutineMCTS search(evaluator, config);
        co_return co_await search.search_move(board);
    }

    std::shared_ptr<coro::thread_pool> pool_;
    std::shared_ptr<legacy::BatchEvaluator> evaluator_;
    MCTSConfig config_;
    chess::Board board_;
};

}  // namespace catgpt

int main(int argc, char* argv[]) {
    // Disable stdio synchronization for better performance
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // Parse command line arguments
    fs::path engine_path = "/home/shadeform/CatGPT/sample.trt";
    if (argc > 1) {
        engine_path = argv[1];
    }

    // Check if engine file exists
    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}", engine_path.string());
        std::println(stderr, "Usage: {} [engine_path]", argv[0]);
        return 1;
    }

    try {
        // Create thread pool (1 worker thread sufficient for single-game UCI)
        auto pool = coro::thread_pool::make_shared(coro::thread_pool::options{
            .thread_count = 1,
        });

        // Create batch evaluator (starts GPU thread internally)
        std::println(stderr, "Loading TensorRT engine: {}", engine_path.string());
        auto evaluator = std::make_shared<catgpt::legacy::BatchEvaluator>(
            engine_path, pool, /*max_batch_size=*/1);
        std::println(stderr, "Engine loaded successfully");

        // Create UCI handler with CoroutineMCTS adapter
        catgpt::UCIHandler handler([pool, evaluator]() {
            return std::make_unique<catgpt::CoroutineMCTSAdapter>(
                pool, evaluator);
        });

        // Run the UCI loop
        handler.run();

        // Clean shutdown
        evaluator->shutdown();

    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
