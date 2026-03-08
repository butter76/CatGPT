/**
 * Self-Play Configuration.
 *
 * Configuration for the batched self-play system: concurrent games,
 * GPU batching, search parameters, and game adjudication.
 *
 * Two search configs are provided:
 *   - baseline_config:    the control (CoroutineSearch / CoroutineMCTS)
 *   - challenger_config:  the variation being tested (ChallengerSearch)
 *
 * Each opening is played twice with colors swapped so that engine
 * assignment bias is eliminated.
 */

#ifndef CATGPT_SELFPLAY_CONFIG_HPP
#define CATGPT_SELFPLAY_CONFIG_HPP

#include <string>

#include "../engine/fractional_mcts/config.hpp"
#include "../engine/mcts/config.hpp"

namespace catgpt {

/**
 * Which search algorithm to use for the CatGPT engines.
 */
enum class SearchType {
    FRACTIONAL_MCTS,  // Default: iterative deepening fractional MCTS (CoroutineSearch)
    MCTS,             // Traditional MCTS with PUCT selection (CoroutineMCTS)
};

/**
 * Configuration for the batched self-play runner.
 */
struct SelfPlayConfig {
    // === Concurrency ===

    /** Number of game-pairs running simultaneously (each pair = 2 games). */
    int num_concurrent_games = 64;

    /** Number of worker threads for running search coroutines. */
    int num_search_threads = 8;

    /** Maximum GPU batch size (requests are batched up to this limit). */
    int max_batch_size = 64;

    // === Game Limits ===

    /** Total number of game PAIRS to play (each pair = 2 games).
     *  0 = one pair per opening position (default). */
    int total_pairs = 0;

    /** Maximum number of moves (plies) per game before adjudicating as draw. */
    int max_moves = 512;

    // === Draw Adjudication ===

    /** Number of consecutive moves with |cp| < draw_cp_threshold to adjudicate draw. */
    int draw_adjudicate_moves = 10;

    /** Centipawn threshold for draw adjudication. */
    int draw_cp_threshold = 10;

    // === Resign Adjudication ===

    /** Number of consecutive moves with cp < -resign_cp_threshold to adjudicate loss. */
    int resign_adjudicate_moves = 5;

    /** Centipawn threshold for resign adjudication. */
    int resign_cp_threshold = 400;

    // === Search Algorithm ===

    /** Which search algorithm to use for CatGPT engines. */
    SearchType search_type = SearchType::FRACTIONAL_MCTS;

    // === Search (two engines) — Fractional MCTS ===

    /** Baseline search configuration (CoroutineSearch — the control). */
    FractionalMCTSConfig baseline_config{};

    /** Challenger search configuration (ChallengerSearch — your variation). */
    FractionalMCTSConfig challenger_config{};

    // === Search (MCTS mode) ===

    /** Baseline MCTS configuration (used when search_type == MCTS). */
    MCTSConfig baseline_mcts_config{};

    /** Challenger MCTS configuration (used when search_type == MCTS). */
    MCTSConfig challenger_mcts_config{};

    // === Engine Labels (for PGN + logging) ===

    std::string baseline_name = "Baseline";
    std::string challenger_name = "Challenger";

    // === Paths ===

    /** Path to TensorRT engine file. */
    std::string engine_path;

    /** Path to openings file (EPD or FEN format, one position per line). */
    std::string openings_path;

    /** Path to output PGN file. Empty = no PGN output. */
    std::string output_pgn;

    /** Path to Syzygy tablebase directory. Empty = no tablebase adjudication. */
    std::string syzygy_path;

    /** If true, emit JSON-lines metrics to stdout after each game. */
    bool json_metrics = false;

    // === Stockfish opponent ===

    /** If true, replace the baseline engine with Stockfish. */
    bool use_stockfish = false;

    /** Path to the Stockfish binary (or just "stockfish" if on PATH). */
    std::string stockfish_path = "stockfish";

    /** Fixed node count for Stockfish (`go nodes X`). */
    int stockfish_nodes = 10000;

    /** Number of concurrent Stockfish subprocesses. */
    int stockfish_processes = 8;

    /** UCI Threads option per Stockfish process. */
    int stockfish_threads = 1;

    /** UCI Hash option (MB) per Stockfish process. */
    int stockfish_hash = 16;

    // === Lc0 opponent ===

    /** If true, use Lc0 as an opponent.
     *  When use_stockfish is also true, runs Lc0 (challenger) vs Stockfish (baseline). */
    bool use_lc0 = false;

    /** Path to the Lc0 binary. */
    std::string lc0_path = "/home/shadeform/lc0/build/release/lc0";

    /** Path to Lc0 neural network weights file (.pb.gz). Required when use_lc0 is true. */
    std::string lc0_weights;

    /** Fixed node count (MCTS playouts) for Lc0 (`go nodes X`). */
    int lc0_nodes = 800;

    /** Number of concurrent Lc0 subprocesses. */
    int lc0_processes = 4;

    /** UCI Threads option per Lc0 process. */
    int lc0_threads = 1;

    /** Neural-net backend for Lc0 (e.g. "cuda-auto", "eigen"). */
    std::string lc0_backend = "cuda-auto";

    /** MinibatchSize for Lc0 NN computation (0 = backend default). */
    int lc0_minibatch_size = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_CONFIG_HPP
