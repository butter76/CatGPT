/**
 * CatGPT Self-Play — Main Entry Point
 *
 * Runs batched self-play games using coroutine-based Fractional MCTS
 * with GPU-batched neural network inference.
 *
 * Usage:
 *   catgpt_selfplay <engine_path> [options]
 *
 * Options:
 *   --games N        Total games to play (default: 1000)
 *   --concurrent N   Concurrent games (default: 32)
 *   --threads N      Search worker threads (default: 8)
 *   --batch N        Max GPU batch size (default: 32)
 *   --evals N        Min GPU evals per move (default: 400)
 *   --openings PATH  Opening book file (EPD/FEN)
 *   --pgn PATH       Output PGN file
 */

#include <filesystem>
#include <iostream>
#include <print>
#include <string>

#include "selfplay/selfplay_runner.hpp"

namespace fs = std::filesystem;

void print_usage(const char* argv0) {
    std::println(stderr, "Usage: {} <engine_path> [options]", argv0);
    std::println(stderr, "");
    std::println(stderr, "Options:");
    std::println(stderr, "  --games N        Total games to play (default: 1000)");
    std::println(stderr, "  --concurrent N   Concurrent games (default: 32)");
    std::println(stderr, "  --threads N      Search worker threads (default: 8)");
    std::println(stderr, "  --batch N        Max GPU batch size (default: 32)");
    std::println(stderr, "  --evals N        Min GPU evals per move (default: 400)");
    std::println(stderr, "  --openings PATH  Opening book file (EPD/FEN)");
    std::println(stderr, "  --pgn PATH       Output PGN file");
    std::println(stderr, "  --cpuct F        PUCT exploration constant (default: 1.75)");
}

int main(int argc, char* argv[]) {
    // Check for --help before anything else
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    catgpt::SelfPlayConfig config;
    config.engine_path = argv[1];

    // Parse optional arguments
    for (int i = 2; i < argc; ++i) {
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

        auto next_string = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::println(stderr, "Error: {} requires an argument", arg);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--games") {
            config.total_games = next_int();
        } else if (arg == "--concurrent") {
            config.num_concurrent_games = next_int();
        } else if (arg == "--threads") {
            config.num_search_threads = next_int();
        } else if (arg == "--batch") {
            config.max_batch_size = next_int();
        } else if (arg == "--evals") {
            config.search_config.min_total_evals = next_int();
        } else if (arg == "--openings") {
            config.openings_path = next_string();
        } else if (arg == "--pgn") {
            config.output_pgn = next_string();
        } else if (arg == "--cpuct") {
            config.search_config.c_puct = next_float();
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::println(stderr, "Unknown option: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Validate engine path
    if (!fs::exists(config.engine_path)) {
        std::println(stderr, "Error: TensorRT engine not found: {}", config.engine_path);
        return 1;
    }

    // Validate openings file
    if (!config.openings_path.empty() && !fs::exists(config.openings_path)) {
        std::println(stderr, "Error: Openings file not found: {}", config.openings_path);
        return 1;
    }

    std::println(stderr, "╔════════════════════════════════════════════════╗");
    std::println(stderr, "║          CatGPT Batched Self-Play             ║");
    std::println(stderr, "╚════════════════════════════════════════════════╝");
    std::println(stderr, "");
    std::println(stderr, "  Engine:      {}", config.engine_path);
    std::println(stderr, "  Games:       {}", config.total_games);
    std::println(stderr, "  Concurrent:  {}", config.num_concurrent_games);
    std::println(stderr, "  Threads:     {}", config.num_search_threads);
    std::println(stderr, "  Batch size:  {}", config.max_batch_size);
    std::println(stderr, "  Evals/move:  {}", config.search_config.min_total_evals);
    std::println(stderr, "  Openings:    {}", config.openings_path.empty() ? "(startpos)" : config.openings_path);
    std::println(stderr, "  PGN output:  {}", config.output_pgn.empty() ? "(none)" : config.output_pgn);
    std::println(stderr, "");

    try {
        catgpt::SelfPlayRunner runner(config);
        runner.run();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
