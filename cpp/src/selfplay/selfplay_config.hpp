/**
 * Self-Play Configuration.
 *
 * Configuration for the batched self-play system: concurrent games,
 * GPU batching, search parameters, and game adjudication.
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

    /** Number of games running simultaneously. */
    int num_concurrent_games = 32;

    /** Number of worker threads for running search coroutines. */
    int num_search_threads = 8;

    /** Maximum GPU batch size (requests are batched up to this limit). */
    int max_batch_size = 32;

    // === Game Limits ===

    /** Total number of games to play before stopping. 0 = unlimited. */
    int total_games = 1000;

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

    // === Search ===

    /** Fractional MCTS search configuration. */
    FractionalMCTSConfig search_config{};

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
