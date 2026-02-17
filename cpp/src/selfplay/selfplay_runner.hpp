/**
 * Self-Play Runner — the main orchestrator.
 *
 * Manages N concurrent games, a thread pool for search coroutines,
 * and a BatchEvaluator for GPU inference.  Games are played
 * continuously: when one finishes, a new game immediately starts
 * in that slot.
 *
 * Architecture:
 *   Main thread   → runs the event loop (spawn/collect games)
 *   Thread pool   → runs search coroutines (8 worker threads)
 *   GPU thread    → batches and runs TRT inference
 *
 * Flow for one game:
 *   1. Spawn a coroutine that plays one full game
 *   2. The coroutine alternates: search_move → apply_move → check_game_over
 *   3. Each search_move may suspend many times for GPU evals
 *   4. When the game finishes, the coroutine co_returns the GameRecord
 *   5. The runner collects the result and spawns a new game in that slot
 */

#ifndef CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP
#define CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP

#include <atomic>
#include <chrono>
#include <fstream>
#include <memory>
#include <mutex>
#include <print>
#include <string>
#include <vector>

#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>
#include <coro/when_all.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "batch_evaluator.hpp"
#include "coroutine_search.hpp"
#include "game_slot.hpp"
#include "selfplay_config.hpp"

namespace catgpt {

/**
 * Load opening positions from a FEN/EPD file.
 * Each line is treated as a FEN string (ignoring empty lines and comments).
 */
inline std::vector<std::string> load_openings(const std::string& path) {
    std::vector<std::string> openings;
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open openings file: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Trim whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        auto end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);

        // Skip comments
        if (line.empty() || line[0] == '#') continue;

        // For EPD format: take first 4 fields and add default halfmove/fullmove
        // if they're missing (EPD has 4 fields, FEN has 6)
        auto parts_count = 0;
        for (auto c : line) {
            if (c == ' ') ++parts_count;
        }
        if (parts_count < 4) {
            // Looks like it needs halfmove + fullmove
            line += " 0 1";
        }

        openings.push_back(line);
    }

    return openings;
}

class SelfPlayRunner {
public:
    explicit SelfPlayRunner(const SelfPlayConfig& config)
        : config_(config)
    {
        // Load openings
        if (!config_.openings_path.empty()) {
            openings_ = load_openings(config_.openings_path);
            std::println(stderr, "[SelfPlay] Loaded {} openings", openings_.size());
        }
        if (openings_.empty()) {
            // Fallback: standard starting position
            openings_.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }

        // Create thread pool
        pool_ = coro::thread_pool::make_shared(coro::thread_pool::options{
            .thread_count = static_cast<uint32_t>(config_.num_search_threads),
        });

        // Create batch evaluator (starts GPU thread)
        evaluator_ = std::make_unique<BatchEvaluator>(
            config_.engine_path, pool_, config_.max_batch_size);

        // Open PGN output file
        if (!config_.output_pgn.empty()) {
            pgn_file_.open(config_.output_pgn, std::ios::app);
            if (!pgn_file_) {
                std::println(stderr, "[SelfPlay] WARNING: Failed to open PGN file: {}",
                             config_.output_pgn);
            }
        }
    }

    ~SelfPlayRunner() {
        if (evaluator_) evaluator_->shutdown();
        if (pool_) pool_->shutdown();
    }

    /**
     * Run self-play until the target number of games is reached.
     */
    void run() {
        auto start = std::chrono::steady_clock::now();

        std::println(stderr, "[SelfPlay] Starting: {} concurrent games, {} search threads, "
                     "max_batch={}, target={} games",
                     config_.num_concurrent_games, config_.num_search_threads,
                     config_.max_batch_size,
                     config_.total_games > 0 ? std::to_string(config_.total_games) : "unlimited");

        // Block on the main coroutine
        coro::sync_wait(run_all_games());

        auto elapsed = std::chrono::steady_clock::now() - start;
        double secs = std::chrono::duration<double>(elapsed).count();

        std::println(stderr, "\n[SelfPlay] Done: {} games in {:.1f}s ({:.1f} games/sec)",
                     games_completed_.load(), secs,
                     games_completed_.load() / secs);
        std::println(stderr, "[SelfPlay] GPU evals: {} ({:.0f} evals/sec)",
                     evaluator_->total_evals(),
                     evaluator_->total_evals() / secs);
        print_stats();
    }

private:
    /**
     * Top-level coroutine: spawn N game coroutines, collect results,
     * respawn games until the target is reached.
     */
    coro::task<void> run_all_games() {
        // Schedule onto the thread pool
        co_await pool_->schedule();

        int num_slots = config_.num_concurrent_games;
        int target = config_.total_games;

        // Spawn initial batch of game coroutines
        // We use a simple model: spawn all games, each one plays until done,
        // then we spawn replacement games.
        // With the when_all approach, we spawn all and wait for all.
        // But we want continuous replacement, so instead we use spawn().

        std::atomic<int> games_started{0};

        // Launch concurrent game coroutines
        // Each game_worker plays games in a loop until the target is reached
        std::vector<coro::task<void>> workers;
        workers.reserve(num_slots);

        for (int slot = 0; slot < num_slots; ++slot) {
            workers.push_back(game_worker(slot, games_started, target));
        }

        co_await coro::when_all(std::move(workers));
    }

    /**
     * A single game worker that plays games in a loop.
     * Each worker occupies one "slot" and keeps playing games
     * until the global target is reached.
     */
    coro::task<void> game_worker(int slot_id,
                                 std::atomic<int>& games_started,
                                 int target) {
        // Schedule onto thread pool
        co_await pool_->schedule();

        while (true) {
            // Atomically claim a game number
            int game_num = games_started.fetch_add(1);
            if (target > 0 && game_num >= target) {
                break;  // Target reached
            }

            // Pick an opening (round-robin through the list)
            const auto& opening = openings_[game_num % openings_.size()];

            // Play one full game
            GameRecord record = co_await play_one_game(opening);

            // Record result
            on_game_complete(record, game_num, slot_id);
        }
    }

    /**
     * Play one full game from the given opening, returning the record.
     */
    coro::task<GameRecord> play_one_game(const std::string& opening_fen) {
        GameSlot slot;
        slot.start(opening_fen);

        while (!slot.is_terminated()) {
            // Search for the best move
            CoroutineSearch search(*evaluator_, config_.search_config);
            MoveResult move_result = co_await search.search_move(slot.board());

            if (move_result.best_move == chess::Move::NO_MOVE) {
                // No legal moves — game should already be detected as over
                break;
            }

            // Apply the move
            slot.apply_move(move_result.best_move, move_result.cp_score, move_result.gpu_evals);

            // Check for game over
            slot.check_game_over(config_);
        }

        co_return slot.to_record();
    }

    /**
     * Called when a game completes (thread-safe).
     */
    void on_game_complete(const GameRecord& record, int game_num, int slot_id) {
        int completed = games_completed_.fetch_add(1) + 1;

        // Update stats
        {
            std::lock_guard lock(stats_mutex_);
            switch (record.outcome) {
                case GameOutcome::WHITE_WIN: ++stats_wins_; break;
                case GameOutcome::BLACK_WIN: ++stats_losses_; break;
                case GameOutcome::DRAW:      ++stats_draws_; break;
            }
            stats_total_moves_ += static_cast<int>(record.moves.size());
            stats_total_evals_ += record.total_gpu_evals;
        }

        // Write PGN
        if (pgn_file_.is_open()) {
            write_pgn(record, game_num + 1);
        }

        // Progress logging (every 10 games or at low counts)
        if (completed <= 5 || completed % 10 == 0) {
            std::lock_guard lock(stats_mutex_);
            std::println(stderr, "[SelfPlay] Game #{}: {} in {} moves (slot={}) | "
                         "W/D/L: {}/{}/{} ({} total)",
                         completed, record.result_string(),
                         record.moves.size(), slot_id,
                         stats_wins_, stats_draws_, stats_losses_, completed);
        }
    }

    /**
     * Write a game record as PGN.
     */
    void write_pgn(const GameRecord& record, int round) {
        std::lock_guard lock(pgn_mutex_);
        if (!pgn_file_.is_open()) return;

        pgn_file_ << "[Event \"CatGPT Self-Play\"]\n";
        pgn_file_ << "[Round \"" << round << "\"]\n";
        pgn_file_ << "[White \"CatGPT\"]\n";
        pgn_file_ << "[Black \"CatGPT\"]\n";
        pgn_file_ << "[Result \"" << record.result_string() << "\"]\n";
        pgn_file_ << "[FEN \"" << record.opening_fen << "\"]\n";
        pgn_file_ << "[SetUp \"1\"]\n";

        // Write moves (reconstruct board to get SAN)
        chess::Board board(record.opening_fen);
        int move_num = board.fullMoveNumber();
        bool white_to_move = board.sideToMove() == chess::Color::WHITE;

        pgn_file_ << "\n";
        for (size_t i = 0; i < record.moves.size(); ++i) {
            if (white_to_move) {
                pgn_file_ << move_num << ". ";
            } else if (i == 0) {
                pgn_file_ << move_num << "... ";
            }

            pgn_file_ << chess::uci::moveToSan(board, record.moves[i]) << " ";

            board.makeMove<true>(record.moves[i]);
            if (!white_to_move) ++move_num;
            white_to_move = !white_to_move;
        }

        pgn_file_ << record.result_string() << "\n\n";
        pgn_file_.flush();
    }

    void print_stats() {
        std::lock_guard lock(stats_mutex_);
        int total = stats_wins_ + stats_draws_ + stats_losses_;
        if (total == 0) return;

        float avg_moves = static_cast<float>(stats_total_moves_) / total;
        float avg_evals = static_cast<float>(stats_total_evals_) / total;

        std::println(stderr, "[SelfPlay] Final: W={} D={} L={} ({} games)",
                     stats_wins_, stats_draws_, stats_losses_, total);
        std::println(stderr, "[SelfPlay] Avg moves/game: {:.1f}, Avg GPU evals/game: {:.0f}",
                     avg_moves, avg_evals);
    }

    // ─── Members ────────────────────────────────────────────────────────

    SelfPlayConfig config_;
    std::vector<std::string> openings_;

    std::shared_ptr<coro::thread_pool> pool_;
    std::unique_ptr<BatchEvaluator> evaluator_;

    // PGN output
    std::ofstream pgn_file_;
    std::mutex pgn_mutex_;

    // Statistics
    std::atomic<int> games_completed_{0};
    std::mutex stats_mutex_;
    int stats_wins_ = 0;
    int stats_draws_ = 0;
    int stats_losses_ = 0;
    int stats_total_moves_ = 0;
    int stats_total_evals_ = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP
