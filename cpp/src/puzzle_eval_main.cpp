/**
 * CatGPT Batched Puzzle Evaluation — Main Entry Point
 *
 * Evaluates chess puzzles using batched GPU inference with the same
 * coroutine + BatchEvaluator infrastructure as self-play. Multiple
 * puzzles are solved concurrently, and their GPU eval requests are
 * batched together for efficient TensorRT inference.
 *
 * Input:  Lichess puzzle CSV (PuzzleId, FEN, Moves, Rating, ...)
 * Output: JSON-lines to stdout (one per puzzle + summary)
 *
 * Usage:
 *   catgpt_puzzle_eval <engine_path> <puzzle_csv> [options]
 *
 * Options:
 *   --evals N        Min GPU evals per move (default: 400)
 *   --cpuct F        PUCT exploration constant (default: 1.75)
 *   --concurrent N   Concurrent puzzle slots (default: 128)
 *   --threads N      Search worker threads (default: 8)
 *   --batch N        Max GPU batch size (default: 64)
 *   --max-puzzles N  Limit number of puzzles (default: all)
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <print>
#include <string>

#include "selfplay/puzzle_runner.hpp"

namespace fs = std::filesystem;

void print_usage(const char* argv0) {
    std::println(stderr, "Usage: {} <engine_path> <puzzle_csv> [options]", argv0);
    std::println(stderr, "");
    std::println(stderr, "Concurrency options:");
    std::println(stderr, "  --concurrent N   Concurrent puzzle slots (default: 128)");
    std::println(stderr, "  --threads N      Search worker threads (default: 8)");
    std::println(stderr, "  --batch N        Max GPU batch size (default: 64)");
    std::println(stderr, "");
    std::println(stderr, "Search options:");
    std::println(stderr, "  --evals N        Min GPU evals per move (default: 400)");
    std::println(stderr, "  --cpuct F        PUCT exploration constant (default: 1.75)");
    std::println(stderr, "");
    std::println(stderr, "Data options:");
    std::println(stderr, "  --max-puzzles N  Limit number of puzzles (default: all)");
}

int main(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    catgpt::PuzzleEvalConfig config;
    config.engine_path = argv[1];
    config.puzzle_csv = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        auto next_int = [&]() -> int {
            if (i + 1 >= argc) {
                std::println(stderr, "Error: {} requires an argument", arg);
                std::exit(1);
            }
            return std::stoi(argv[++i]);
        };

        auto next_float = [&]() -> float {
            if (i + 1 >= argc) {
                std::println(stderr, "Error: {} requires an argument", arg);
                std::exit(1);
            }
            return std::stof(argv[++i]);
        };

        if (arg == "--evals") {
            config.search_config.min_total_evals = next_int();
        } else if (arg == "--cpuct") {
            config.search_config.c_puct = next_float();
        } else if (arg == "--concurrent") {
            config.num_concurrent = next_int();
        } else if (arg == "--threads") {
            config.num_search_threads = next_int();
        } else if (arg == "--batch") {
            config.max_batch_size = next_int();
        } else if (arg == "--max-puzzles") {
            config.max_puzzles = next_int();
        } else {
            std::println(stderr, "Unknown option: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate paths
    if (!fs::exists(config.engine_path)) {
        std::println(stderr, "Error: TensorRT engine not found: {}", config.engine_path);
        return 1;
    }
    if (!fs::exists(config.puzzle_csv)) {
        std::println(stderr, "Error: Puzzle CSV not found: {}", config.puzzle_csv);
        return 1;
    }

    std::println(stderr, "╔════════════════════════════════════════════════╗");
    std::println(stderr, "║       CatGPT Batched Puzzle Evaluation        ║");
    std::println(stderr, "╚════════════════════════════════════════════════╝");
    std::println(stderr, "");
    std::println(stderr, "  Engine:      {}", config.engine_path);
    std::println(stderr, "  Puzzles:     {}", config.puzzle_csv);
    if (config.max_puzzles > 0) {
        std::println(stderr, "  Max puzzles: {}", config.max_puzzles);
    }
    std::println(stderr, "  Concurrent:  {}", config.num_concurrent);
    std::println(stderr, "  Threads:     {}", config.num_search_threads);
    std::println(stderr, "  Batch size:  {}", config.max_batch_size);
    std::println(stderr, "  Evals/move:  {}", config.search_config.min_total_evals);
    std::println(stderr, "  PUCT:        {:.2f}", config.search_config.c_puct);
    std::println(stderr, "");

    try {
        catgpt::PuzzleRunner runner(config);
        runner.run();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
