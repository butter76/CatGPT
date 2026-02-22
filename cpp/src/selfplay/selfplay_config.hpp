/**
 * Self-Play Configuration.
 *
 * Configuration for the batched self-play system: concurrent games,
 * GPU batching, search parameters, and game adjudication.
 *
 * Two search configs are provided:
 *   - baseline_config:    the control (CoroutineSearch)
 *   - challenger_config:  the variation being tested (ChallengerSearch)
 *
 * Each opening is played twice with colors swapped so that engine
 * assignment bias is eliminated.
 */

#ifndef CATGPT_SELFPLAY_CONFIG_HPP
#define CATGPT_SELFPLAY_CONFIG_HPP

#include <string>

#include "../engine/fractional_mcts/config.hpp"

namespace catgpt {

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

    // === Search (two engines) ===

    /** Baseline search configuration (CoroutineSearch — the control). */
    FractionalMCTSConfig baseline_config{};

    /** Challenger search configuration (ChallengerSearch — your variation). */
    FractionalMCTSConfig challenger_config{};

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
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_CONFIG_HPP
