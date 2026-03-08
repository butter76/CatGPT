/**
 * CatGPT Self-Play Tournament — Main Entry Point
 *
 * Runs batched tournament games in four modes:
 *
 * Mode 1 — Search-vs-Search (default):
 *   - Baseline  (CoroutineSearch)   — the control
 *   - Challenger (ChallengerSearch)  — your variation to test
 *
 * Mode 2 — CatGPT-vs-Stockfish (--stockfish):
 *   - Challenger (CoroutineSearch)   — CatGPT
 *   - Baseline   (Stockfish)         — fixed-node UCI opponent
 *
 * Mode 3 — CatGPT-vs-Lc0 (--lc0):
 *   - Challenger (CoroutineSearch)   — CatGPT
 *   - Baseline   (Lc0)              — fixed-node MCTS opponent
 *
 * Mode 4 — Lc0-vs-Stockfish (--lc0 --stockfish):
 *   - Challenger (Lc0)              — fixed-node MCTS
 *   - Baseline   (Stockfish)         — fixed-node UCI opponent
 *   (No GPU / TRT engine required)
 *
 * Each opening is played twice with colors swapped.
 * Statistics are reported from the Challenger's perspective.
 *
 * Usage:
 *   catgpt_selfplay [engine_path] [options]
 *
 * Options:
 *   --pairs N            Game pairs to play (default: 500, each pair = 2 games)
 *   --concurrent N       Concurrent game slots (default: 64)
 *   --threads N          Search worker threads (default: 8)
 *   --batch N            Max GPU batch size (default: 64)
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
 *   --stockfish          Enable Stockfish-as-baseline mode
 *   --stockfish-path S   Path to Stockfish binary (default: stockfish)
 *   --stockfish-nodes N  Fixed node count (default: 10000)
 *   --stockfish-processes N  Concurrent Stockfish subprocesses (default: 8)
 *   --stockfish-threads N    UCI Threads per process (default: 1)
 *   --stockfish-hash N       UCI Hash MB per process (default: 16)
 *   --lc0               Enable Lc0-as-baseline mode
 *   --lc0-path S         Path to Lc0 binary
 *   --lc0-weights S      Path to Lc0 weights file (.pb.gz) [required with --lc0]
 *   --lc0-nodes N        Fixed node count / playouts (default: 800)
 *   --lc0-processes N    Concurrent Lc0 subprocesses (default: 4)
 *   --lc0-threads N      UCI Threads per Lc0 process (default: 1)
 *   --lc0-backend S      Neural-net backend (default: cuda-auto)
 *   --lc0-minibatch N   MinibatchSize for NN computation (default: 0 = backend default)
 *   --search-type S     Search algorithm: "fractional" (default) or "mcts"
 *   --fpu F             FPU reduction for MCTS mode (default: 0.003)
 *   --max-simulations N Max MCTS simulations (default: 10000)
 */

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <print>
#include <string>

#include "selfplay/selfplay_runner.hpp"

namespace fs = std::filesystem;

void print_usage(const char* argv0) {
    std::println(stderr, "Usage: {} [engine_path] [options]", argv0);
    std::println(stderr, "       engine_path is required unless both --stockfish and --lc0 are used");
    std::println(stderr, "");
    std::println(stderr, "Tournament options:");
    std::println(stderr, "  --pairs N            Game pairs (default: one per opening, each pair = 2 games)");
    std::println(stderr, "  --concurrent N       Concurrent game slots (default: 64)");
    std::println(stderr, "  --threads N          Search worker threads (default: 8)");
    std::println(stderr, "  --batch N            Max GPU batch size (default: 64)");
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
    std::println(stderr, "  --syzygy PATH        Syzygy tablebase directory (or $SYZYGY_HOME)");
    std::println(stderr, "  --json-metrics       Emit JSON-lines metrics to stdout");
    std::println(stderr, "");
    std::println(stderr, "Stockfish opponent (replaces baseline with Stockfish):");
    std::println(stderr, "  --stockfish          Enable Stockfish-as-baseline mode");
    std::println(stderr, "  --stockfish-path S   Path to Stockfish binary (default: stockfish)");
    std::println(stderr, "  --stockfish-nodes N  Fixed node count (default: 10000)");
    std::println(stderr, "  --stockfish-processes N  Concurrent SF subprocesses (default: 8)");
    std::println(stderr, "  --stockfish-threads N    UCI Threads per SF process (default: 1)");
    std::println(stderr, "  --stockfish-hash N       UCI Hash MB per SF process (default: 16)");
    std::println(stderr, "");
    std::println(stderr, "Lc0 opponent (replaces baseline with Lc0):");
    std::println(stderr, "  --lc0                Enable Lc0-as-baseline mode");
    std::println(stderr, "  --lc0-path S         Path to Lc0 binary");
    std::println(stderr, "  --lc0-weights S      Path to Lc0 weights file (.pb.gz) [required]");
    std::println(stderr, "  --lc0-nodes N        Fixed node count / playouts (default: 800)");
    std::println(stderr, "  --lc0-processes N    Concurrent Lc0 subprocesses (default: 4)");
    std::println(stderr, "  --lc0-threads N      UCI Threads per Lc0 process (default: 1)");
    std::println(stderr, "  --lc0-backend S      Neural-net backend (default: cuda-auto)");
    std::println(stderr, "  --lc0-minibatch N    MinibatchSize for NN computation (default: 0 = backend default)");
    std::println(stderr, "");
    std::println(stderr, "Search algorithm:");
    std::println(stderr, "  --search-type S      Search algorithm: \"fractional\" (default) or \"mcts\"");
    std::println(stderr, "  --fpu F              FPU reduction for MCTS mode (default: 0.003)");
    std::println(stderr, "  --max-simulations N  Max MCTS simulations per move (default: 10000)");
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

    // Determine if argv[1] is the engine path or an option flag.
    // If it starts with "--" it's an option; otherwise it's the engine path.
    int first_opt = 1;
    if (argc >= 2 && std::string(argv[1]).substr(0, 2) != "--") {
        config.engine_path = argv[1];
        first_opt = 2;
    }

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
    for (int i = first_opt; i < argc; ++i) {
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
        } else if (arg == "--syzygy") {
            config.syzygy_path = next_string();
        } else if (arg == "--json-metrics") {
            config.json_metrics = true;
        } else if (arg == "--stockfish") {
            config.use_stockfish = true;
        } else if (arg == "--stockfish-path") {
            config.stockfish_path = next_string();
        } else if (arg == "--stockfish-nodes") {
            config.stockfish_nodes = next_int();
        } else if (arg == "--stockfish-processes") {
            config.stockfish_processes = next_int();
        } else if (arg == "--stockfish-threads") {
            config.stockfish_threads = next_int();
        } else if (arg == "--stockfish-hash") {
            config.stockfish_hash = next_int();
        } else if (arg == "--lc0") {
            config.use_lc0 = true;
        } else if (arg == "--lc0-path") {
            config.lc0_path = next_string();
        } else if (arg == "--lc0-weights") {
            config.lc0_weights = next_string();
        } else if (arg == "--lc0-nodes") {
            config.lc0_nodes = next_int();
        } else if (arg == "--lc0-processes") {
            config.lc0_processes = next_int();
        } else if (arg == "--lc0-threads") {
            config.lc0_threads = next_int();
        } else if (arg == "--lc0-backend") {
            config.lc0_backend = next_string();
        } else if (arg == "--lc0-minibatch") {
            config.lc0_minibatch_size = next_int();
        } else if (arg == "--search-type") {
            std::string type = next_string();
            if (type == "mcts") {
                config.search_type = catgpt::SearchType::MCTS;
            } else if (type == "fractional") {
                config.search_type = catgpt::SearchType::FRACTIONAL_MCTS;
            } else {
                std::println(stderr, "Error: --search-type must be 'fractional' or 'mcts', got '{}'", type);
                return 1;
            }
        } else if (arg == "--fpu") {
            float fpu = next_float();
            config.baseline_mcts_config.fpu_reduction = fpu;
            config.challenger_mcts_config.fpu_reduction = fpu;
        } else if (arg == "--max-simulations") {
            int max_sims = next_int();
            config.baseline_mcts_config.max_simulations = max_sims;
            config.challenger_mcts_config.max_simulations = max_sims;
        } else {
            std::println(stderr, "Unknown option: {}", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Apply shared values first, then per-engine overrides
    // (applies to both Fractional MCTS and MCTS configs)
    if (shared_evals_set) {
        config.baseline_config.min_total_evals = shared_evals;
        config.challenger_config.min_total_evals = shared_evals;
        config.baseline_mcts_config.min_total_evals = shared_evals;
        config.challenger_mcts_config.min_total_evals = shared_evals;
    }
    if (shared_cpuct_set) {
        config.baseline_config.c_puct = shared_cpuct;
        config.challenger_config.c_puct = shared_cpuct;
        config.baseline_mcts_config.c_puct = shared_cpuct;
        config.challenger_mcts_config.c_puct = shared_cpuct;
    }
    if (baseline_evals_set) {
        config.baseline_config.min_total_evals = baseline_evals;
        config.baseline_mcts_config.min_total_evals = baseline_evals;
    }
    if (challenger_evals_set) {
        config.challenger_config.min_total_evals = challenger_evals;
        config.challenger_mcts_config.min_total_evals = challenger_evals;
    }
    if (baseline_cpuct_set) {
        config.baseline_config.c_puct = baseline_cpuct;
        config.baseline_mcts_config.c_puct = baseline_cpuct;
    }
    if (challenger_cpuct_set) {
        config.challenger_config.c_puct = challenger_cpuct;
        config.challenger_mcts_config.c_puct = challenger_cpuct;
    }

    // Determine if we need GPU (TRT engine)
    bool both_external = config.use_stockfish && config.use_lc0;
    bool needs_gpu = !both_external;

    // Validate engine path (required unless both external engines are used)
    if (needs_gpu && config.engine_path.empty()) {
        std::println(stderr, "Error: engine_path is required unless both --stockfish and --lc0 are used");
        print_usage(argv[0]);
        return 1;
    }

    // Lc0 mode: validate weights path is provided
    if (config.use_lc0 && config.lc0_weights.empty()) {
        std::println(stderr, "Error: --lc0-weights is required when using --lc0");
        return 1;
    }

    // Mode 4: Lc0 vs Stockfish — set sensible defaults
    if (both_external) {
        if (config.challenger_name == "Challenger") {
            config.challenger_name = "Lc0";
        }
        if (config.baseline_name == "Baseline") {
            config.baseline_name = "Stockfish";
        }
    } else if (config.use_stockfish) {
        // Mode 2: CatGPT vs Stockfish
        if (config.challenger_name == "Challenger") {
            config.challenger_name = "CatGPT";
        }
        if (config.baseline_name == "Baseline") {
            config.baseline_name = "Stockfish";
        }
    } else if (config.use_lc0) {
        // Mode 3: CatGPT vs Lc0
        if (config.challenger_name == "Challenger") {
            config.challenger_name = "CatGPT";
        }
        if (config.baseline_name == "Baseline") {
            config.baseline_name = "Lc0";
        }
    }

    // Syzygy: fall back to $SYZYGY_HOME if --syzygy not provided
    if (config.syzygy_path.empty()) {
        if (const char* env = std::getenv("SYZYGY_HOME"); env != nullptr) {
            config.syzygy_path = env;
        }
    }

    // Validate engine path (only needed when GPU search is used)
    if (needs_gpu && !fs::exists(config.engine_path)) {
        std::println(stderr, "Error: TensorRT engine not found: {}", config.engine_path);
        return 1;
    }

    // Validate openings file
    if (!config.openings_path.empty() && !fs::exists(config.openings_path)) {
        std::println(stderr, "Error: Openings file not found: {}", config.openings_path);
        return 1;
    }

    // Prompt for run name and set PGN output path (skip if --json-metrics, the wrapper handles it)
    if (config.output_pgn.empty()) {
        std::string run_name;
        if (config.json_metrics) {
            run_name = "selfplay";
        } else {
            std::print(stderr, "Run name: ");
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
    if (needs_gpu) {
        std::println(stderr, "  Engine:       {}", config.engine_path);
    } else {
        std::println(stderr, "  Engine:       (none — external-vs-external mode)");
    }
    if (config.total_pairs > 0) {
        std::println(stderr, "  Pairs:        {} ({} games)", config.total_pairs, config.total_pairs * 2);
    } else {
        std::println(stderr, "  Pairs:        (one per opening)");
    }
    std::println(stderr, "  Concurrent:   {}", config.num_concurrent_games);
    std::println(stderr, "  Threads:      {}", config.num_search_threads);
    if (needs_gpu) {
        std::println(stderr, "  Batch size:   {}", config.max_batch_size);
    }
    std::println(stderr, "  Openings:     {}", config.openings_path.empty() ? "(startpos)" : config.openings_path);
    std::println(stderr, "  Syzygy:       {}", config.syzygy_path.empty() ? "(disabled)" : config.syzygy_path);
    std::println(stderr, "  PGN output:   {}", config.output_pgn.empty() ? "(none)" : config.output_pgn);
    std::println(stderr, "  Search type:  {}",
                 config.search_type == catgpt::SearchType::MCTS ? "mcts" : "fractional");
    std::println(stderr, "");

    // Challenger info
    if (both_external) {
        // Mode 4: Lc0 is the challenger
        std::println(stderr, "  {} (challenger):", config.challenger_name);
        std::println(stderr, "    nodes={}, processes={}, threads={}, backend={}",
                     config.lc0_nodes, config.lc0_processes,
                     config.lc0_threads, config.lc0_backend);
        std::println(stderr, "    weights={}", config.lc0_weights);
    } else {
        std::println(stderr, "  {} (challenger):", config.challenger_name);
        if (config.search_type == catgpt::SearchType::MCTS) {
            std::println(stderr, "    evals={}, cpuct={:.2f}, fpu={:.4f}, max_sims={}",
                         config.challenger_mcts_config.min_total_evals,
                         config.challenger_mcts_config.c_puct,
                         config.challenger_mcts_config.fpu_reduction,
                         config.challenger_mcts_config.max_simulations);
        } else {
            std::println(stderr, "    evals={}, cpuct={:.2f}",
                         config.challenger_config.min_total_evals, config.challenger_config.c_puct);
        }
    }

    // Baseline info
    if (config.use_stockfish) {
        std::println(stderr, "  {} (baseline):", config.baseline_name);
        std::println(stderr, "    nodes={}, processes={}, threads={}, hash={}MB",
                     config.stockfish_nodes, config.stockfish_processes,
                     config.stockfish_threads, config.stockfish_hash);
    } else if (config.use_lc0) {
        std::println(stderr, "  {} (baseline):", config.baseline_name);
        std::println(stderr, "    nodes={}, processes={}, threads={}, backend={}",
                     config.lc0_nodes, config.lc0_processes,
                     config.lc0_threads, config.lc0_backend);
        std::println(stderr, "    weights={}", config.lc0_weights);
    } else {
        std::println(stderr, "  {} (baseline):", config.baseline_name);
        if (config.search_type == catgpt::SearchType::MCTS) {
            std::println(stderr, "    evals={}, cpuct={:.2f}, fpu={:.4f}, max_sims={}",
                         config.baseline_mcts_config.min_total_evals,
                         config.baseline_mcts_config.c_puct,
                         config.baseline_mcts_config.fpu_reduction,
                         config.baseline_mcts_config.max_simulations);
        } else {
            std::println(stderr, "    evals={}, cpuct={:.2f}",
                         config.baseline_config.min_total_evals, config.baseline_config.c_puct);
        }
    }
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
