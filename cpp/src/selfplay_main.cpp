/**
 * CatGPT Self-Play Tournament — Main Entry Point
 *
 * Runs batched tournament games between two search algorithms:
 *   - Baseline  (CoroutineSearch)   — the control
 *   - Challenger (ChallengerSearch)  — your variation to test
 *
 * Each opening is played twice with colors swapped.
 * Statistics are reported from the Challenger's perspective.
 *
 * Usage:
 *   catgpt_selfplay <engine_path> [options]
 *
 * Options:
 *   --pairs N            Game pairs to play (default: 500, each pair = 2 games)
 *   --concurrent N       Concurrent game slots (default: 32)
 *   --threads N          Search worker threads (default: 8)
 *   --batch N            Max GPU batch size (default: 32)
 *   --evals N            Min GPU evals per move for BOTH engines (default: 400)
 *   --baseline-evals N   Override evals for baseline only
 *   --challenger-evals N Override evals for challenger only
 *   --cpuct F            PUCT constant for BOTH engines (default: 1.75)
 *   --baseline-cpuct F   Override cpuct for baseline only
 *   --challenger-cpuct F Override cpuct for challenger only
 *   --openings PATH      Opening book file (EPD/FEN)
 *   --pgn PATH           Output PGN file
 *   --baseline-name S    Label for baseline (default: Baseline)
 *   --challenger-name S  Label for challenger (default: Challenger)
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
    std::println(stderr, "Tournament options:");
    std::println(stderr, "  --pairs N            Game pairs (default: one per opening, each pair = 2 games)");
    std::println(stderr, "  --concurrent N       Concurrent game slots (default: 32)");
    std::println(stderr, "  --threads N          Search worker threads (default: 8)");
    std::println(stderr, "  --batch N            Max GPU batch size (default: 32)");
    std::println(stderr, "");
    std::println(stderr, "Search options (apply to both engines unless overridden):");
    std::println(stderr, "  --evals N            Min GPU evals per move (default: 400)");
    std::println(stderr, "  --cpuct F            PUCT exploration constant (default: 1.75)");
    std::println(stderr, "");
    std::println(stderr, "Per-engine overrides:");
    std::println(stderr, "  --baseline-evals N   Evals for baseline only");
    std::println(stderr, "  --challenger-evals N Evals for challenger only");
    std::println(stderr, "  --baseline-cpuct F   PUCT for baseline only");
    std::println(stderr, "  --challenger-cpuct F PUCT for challenger only");
    std::println(stderr, "  --baseline-name S    Label for baseline (default: Baseline)");
    std::println(stderr, "  --challenger-name S  Label for challenger (default: Challenger)");
    std::println(stderr, "");
    std::println(stderr, "I/O options:");
    std::println(stderr, "  --openings PATH      Opening book file (EPD/FEN)");
    std::println(stderr, "  --pgn PATH           Output PGN file");
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

    // Track shared values so we can apply them to both engines
    bool shared_evals_set = false;
    int shared_evals = 0;
    bool shared_cpuct_set = false;
    float shared_cpuct = 0.0f;

    bool baseline_evals_set = false;
    int baseline_evals = 0;
    bool challenger_evals_set = false;
    int challenger_evals = 0;

    bool baseline_cpuct_set = false;
    float baseline_cpuct = 0.0f;
    bool challenger_cpuct_set = false;
    float challenger_cpuct = 0.0f;

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

        if (arg == "--pairs") {
            config.total_pairs = next_int();
        } else if (arg == "--concurrent") {
            config.num_concurrent_games = next_int();
        } else if (arg == "--threads") {
            config.num_search_threads = next_int();
        } else if (arg == "--batch") {
            config.max_batch_size = next_int();
        } else if (arg == "--evals") {
            shared_evals = next_int();
            shared_evals_set = true;
        } else if (arg == "--baseline-evals") {
            baseline_evals = next_int();
            baseline_evals_set = true;
        } else if (arg == "--challenger-evals") {
            challenger_evals = next_int();
            challenger_evals_set = true;
        } else if (arg == "--cpuct") {
            shared_cpuct = next_float();
            shared_cpuct_set = true;
        } else if (arg == "--baseline-cpuct") {
            baseline_cpuct = next_float();
            baseline_cpuct_set = true;
        } else if (arg == "--challenger-cpuct") {
            challenger_cpuct = next_float();
            challenger_cpuct_set = true;
        } else if (arg == "--openings") {
            config.openings_path = next_string();
        } else if (arg == "--pgn") {
            config.output_pgn = next_string();
        } else if (arg == "--baseline-name") {
            config.baseline_name = next_string();
        } else if (arg == "--challenger-name") {
            config.challenger_name = next_string();
        } else {
            std::println(stderr, "Unknown option: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Apply shared values first, then per-engine overrides
    if (shared_evals_set) {
        config.baseline_config.min_total_evals = shared_evals;
        config.challenger_config.min_total_evals = shared_evals;
    }
    if (shared_cpuct_set) {
        config.baseline_config.c_puct = shared_cpuct;
        config.challenger_config.c_puct = shared_cpuct;
    }
    if (baseline_evals_set) {
        config.baseline_config.min_total_evals = baseline_evals;
    }
    if (challenger_evals_set) {
        config.challenger_config.min_total_evals = challenger_evals;
    }
    if (baseline_cpuct_set) {
        config.baseline_config.c_puct = baseline_cpuct;
    }
    if (challenger_cpuct_set) {
        config.challenger_config.c_puct = challenger_cpuct;
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

    // Prompt for run name and set PGN output path
    if (config.output_pgn.empty()) {
        std::print(stderr, "Run name: ");
        std::string run_name;
        std::getline(std::cin, run_name);

        // Trim whitespace
        auto s = run_name.find_first_not_of(" \t\r\n");
        auto e = run_name.find_last_not_of(" \t\r\n");
        if (s != std::string::npos) {
            run_name = run_name.substr(s, e - s + 1);
        }

        // Replace spaces with underscores
        for (auto& c : run_name) {
            if (c == ' ') c = '_';
        }

        if (run_name.empty()) {
            run_name = "unnamed";
        }

        // Ensure outputs directory exists
        fs::create_directories("outputs");
        config.output_pgn = "outputs/sprt_" + run_name + ".pgn";
    }

    std::println(stderr, "╔════════════════════════════════════════════════╗");
    std::println(stderr, "║       CatGPT Batched Tournament               ║");
    std::println(stderr, "╚════════════════════════════════════════════════╝");
    std::println(stderr, "");
    std::println(stderr, "  Engine:       {}", config.engine_path);
    if (config.total_pairs > 0) {
        std::println(stderr, "  Pairs:        {} ({} games)", config.total_pairs, config.total_pairs * 2);
    } else {
        std::println(stderr, "  Pairs:        (one per opening)");
    }
    std::println(stderr, "  Concurrent:   {}", config.num_concurrent_games);
    std::println(stderr, "  Threads:      {}", config.num_search_threads);
    std::println(stderr, "  Batch size:   {}", config.max_batch_size);
    std::println(stderr, "  Openings:     {}", config.openings_path.empty() ? "(startpos)" : config.openings_path);
    std::println(stderr, "  PGN output:   {}", config.output_pgn.empty() ? "(none)" : config.output_pgn);
    std::println(stderr, "");
    std::println(stderr, "  {} (challenger):", config.challenger_name);
    std::println(stderr, "    evals={}, cpuct={:.2f}",
                 config.challenger_config.min_total_evals, config.challenger_config.c_puct);
    std::println(stderr, "  {} (baseline):", config.baseline_name);
    std::println(stderr, "    evals={}, cpuct={:.2f}",
                 config.baseline_config.min_total_evals, config.baseline_config.c_puct);
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
