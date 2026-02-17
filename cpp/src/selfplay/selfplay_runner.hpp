/**
 * Self-Play Runner — the main orchestrator.
 *
 * Pits ChallengerSearch (engine A) against CoroutineSearch (engine B)
 * in a tournament.  Each opening is played TWICE with colors swapped
 * to eliminate first-move bias:
 *   Game 1: Challenger=White  vs  Baseline=Black
 *   Game 2: Baseline=White   vs  Challenger=Black
 *
 * Statistics are tracked from the Challenger's perspective:
 *   W = Challenger wins, L = Challenger losses, D = draws
 *
 * Architecture:
 *   Main thread   → runs the event loop (spawn/collect game pairs)
 *   Thread pool   → runs search coroutines (N worker threads)
 *   GPU thread    → batches and runs TRT inference
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
#include "challenger_search.hpp"
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
            openings_.push_back("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        }

        // Default total_pairs to number of openings (one pair per opening)
        if (config_.total_pairs <= 0) {
            config_.total_pairs = static_cast<int>(openings_.size());
            std::println(stderr, "[SelfPlay] Pairs defaulting to opening count: {}", config_.total_pairs);
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
     * Run the tournament until the target number of game pairs is reached.
     */
    void run() {
        auto start = std::chrono::steady_clock::now();

        int total_games = config_.total_pairs * 2;
        std::println(stderr, "[SelfPlay] {} vs {}",
                     config_.challenger_name, config_.baseline_name);
        std::println(stderr, "[SelfPlay] Starting: {} concurrent slots, {} search threads, "
                     "max_batch={}, target={} pairs ({} games)",
                     config_.num_concurrent_games, config_.num_search_threads,
                     config_.max_batch_size, config_.total_pairs, total_games);

        coro::sync_wait(run_all_pairs());

        auto elapsed = std::chrono::steady_clock::now() - start;
        double secs = std::chrono::duration<double>(elapsed).count();

        int completed = games_completed_.load();
        std::println(stderr, "\n[SelfPlay] Done: {} games in {:.1f}s ({:.1f} games/sec)",
                     completed, secs, completed / secs);
        std::println(stderr, "[SelfPlay] GPU evals: {} ({:.0f} evals/sec)",
                     evaluator_->total_evals(),
                     evaluator_->total_evals() / secs);
        print_stats();
    }

private:
    // ─── Coroutine orchestration ────────────────────────────────────────

    coro::task<void> run_all_pairs() {
        co_await pool_->schedule();

        int num_slots = config_.num_concurrent_games;
        int target_pairs = config_.total_pairs;

        std::atomic<int> pairs_started{0};

        std::vector<coro::task<void>> workers;
        workers.reserve(num_slots);

        for (int slot = 0; slot < num_slots; ++slot) {
            workers.push_back(pair_worker(slot, pairs_started, target_pairs));
        }

        co_await coro::when_all(std::move(workers));
    }

    /**
     * A worker that plays game pairs in a loop.
     * Each iteration plays one opening twice (colors swapped).
     */
    coro::task<void> pair_worker(int slot_id,
                                 std::atomic<int>& pairs_started,
                                 int target_pairs) {
        co_await pool_->schedule();

        while (true) {
            int pair_num = pairs_started.fetch_add(1);
            if (target_pairs > 0 && pair_num >= target_pairs) {
                break;
            }

            const auto& opening = openings_[pair_num % openings_.size()];

            // Alternate starting color across pairs so the challenger
            // doesn't always get the first move in game 1.
            bool challenger_white_first = (pair_num % 2 == 0);

            // Game 1
            GameRecord game1 = co_await play_one_game(
                opening, /*challenger_is_white=*/challenger_white_first);
            on_game_complete(game1, pair_num * 2, slot_id);

            // Game 2: colors swapped
            GameRecord game2 = co_await play_one_game(
                opening, /*challenger_is_white=*/!challenger_white_first);
            on_game_complete(game2, pair_num * 2 + 1, slot_id);
        }
    }

    /**
     * Play one game.  On each move, the current side-to-move determines
     * which search engine is used.
     *
     * @param challenger_is_white  If true, ChallengerSearch plays White.
     */
    coro::task<GameRecord> play_one_game(const std::string& opening_fen,
                                         bool challenger_is_white) {
        GameSlot slot;
        slot.start(opening_fen);

        // baseline_white is the opposite of challenger_is_white
        bool baseline_white = !challenger_is_white;

        while (!slot.is_terminated()) {
            bool white_to_move = slot.board().sideToMove() == chess::Color::WHITE;
            bool challenger_to_move = (white_to_move == challenger_is_white);

            MoveResult move_result;
            if (challenger_to_move) {
                ChallengerSearch search(*evaluator_, config_.challenger_config);
                move_result = co_await search.search_move(slot.board());
            } else {
                CoroutineSearch search(*evaluator_, config_.baseline_config);
                move_result = co_await search.search_move(slot.board());
            }

            if (move_result.best_move == chess::Move::NO_MOVE) {
                break;
            }

            slot.apply_move(move_result.best_move, move_result.cp_score, move_result.gpu_evals);
            slot.check_game_over(config_);
        }

        auto record = slot.to_record();
        record.baseline_white = baseline_white;
        co_return record;
    }

    // ─── Elo estimation ──────────────────────────────────────────────────

    /**
     * Estimate Elo difference from W/D/L.
     * Uses: score = (W + 0.5*D) / N, then Elo = -400 * log10(1/score - 1).
     * Returns 0 if there are no games or score is exactly 50%.
     * Clamps to ±9999 for extreme scores.
     */
    [[nodiscard]] static float estimate_elo(int wins, int draws, int losses) {
        int total = wins + draws + losses;
        if (total == 0) return 0.0f;
        float score = (static_cast<float>(wins) + 0.5f * draws) / total;
        // Clamp to avoid log(0) / division by zero
        score = std::clamp(score, 0.0001f, 0.9999f);
        return -400.0f * std::log10(1.0f / score - 1.0f);
    }

    // ─── Result tracking ────────────────────────────────────────────────

    void on_game_complete(const GameRecord& record, int game_num, int slot_id) {
        int completed = games_completed_.fetch_add(1) + 1;

        // Track from challenger's perspective
        float score = record.baseline_score();
        // baseline_score() returns score for baseline; invert for challenger
        float challenger_score = 1.0f - score;

        {
            std::lock_guard lock(stats_mutex_);
            if (challenger_score > 0.75f) {
                ++challenger_wins_;
            } else if (challenger_score < 0.25f) {
                ++challenger_losses_;
            } else {
                ++draws_;
            }
            stats_total_moves_ += static_cast<int>(record.moves.size());
            stats_total_evals_ += record.total_gpu_evals;
        }

        // Write PGN
        if (pgn_file_.is_open()) {
            write_pgn(record, game_num + 1);
        }

        // Progress logging
        if (completed <= 5 || completed % 10 == 0) {
            std::lock_guard lock(stats_mutex_);
            // Show which engine won
            std::string winner;
            if (record.outcome == GameOutcome::DRAW) {
                winner = "draw";
            } else {
                bool white_won = (record.outcome == GameOutcome::WHITE_WIN);
                bool challenger_won = (white_won != record.baseline_white);
                winner = challenger_won ? config_.challenger_name : config_.baseline_name;
            }
            float elo = estimate_elo(challenger_wins_, draws_, challenger_losses_);
            std::println(stderr, "[SelfPlay] Game #{}: {} ({}) in {} moves (slot={}) | "
                         "{} W/D/L: {}/{}/{} ({} games, Elo: {:+.0f})",
                         completed, record.result_string(), winner,
                         record.moves.size(), slot_id,
                         config_.challenger_name,
                         challenger_wins_, draws_, challenger_losses_, completed, elo);
        }
    }

    // ─── PGN output ─────────────────────────────────────────────────────

    void write_pgn(const GameRecord& record, int round) {
        std::lock_guard lock(pgn_mutex_);
        if (!pgn_file_.is_open()) return;

        // Determine engine names for White and Black
        const std::string& white_name = record.baseline_white
            ? config_.baseline_name : config_.challenger_name;
        const std::string& black_name = record.baseline_white
            ? config_.challenger_name : config_.baseline_name;

        pgn_file_ << "[Event \"CatGPT Tournament\"]\n";
        pgn_file_ << "[Round \"" << round << "\"]\n";
        pgn_file_ << "[White \"" << white_name << "\"]\n";
        pgn_file_ << "[Black \"" << black_name << "\"]\n";
        pgn_file_ << "[Result \"" << record.result_string() << "\"]\n";
        pgn_file_ << "[FEN \"" << record.opening_fen << "\"]\n";
        pgn_file_ << "[SetUp \"1\"]\n";

        // Write moves
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
        int total = challenger_wins_ + draws_ + challenger_losses_;
        if (total == 0) return;

        float avg_moves = static_cast<float>(stats_total_moves_) / total;
        float avg_evals = static_cast<float>(stats_total_evals_) / total;
        float score_pct = (challenger_wins_ + 0.5f * draws_) / total * 100.0f;
        float elo = estimate_elo(challenger_wins_, draws_, challenger_losses_);

        std::println(stderr, "[SelfPlay] {} vs {}: W={} D={} L={} ({} games, {:.1f}%, Elo: {:+.0f})",
                     config_.challenger_name, config_.baseline_name,
                     challenger_wins_, draws_, challenger_losses_, total, score_pct, elo);
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

    // Statistics (from challenger's perspective)
    std::atomic<int> games_completed_{0};
    std::mutex stats_mutex_;
    int challenger_wins_ = 0;
    int draws_ = 0;
    int challenger_losses_ = 0;
    int stats_total_moves_ = 0;
    int stats_total_evals_ = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP
