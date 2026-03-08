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
 *   GPU thread    → batches and runs TRT inference (skipped in external-vs-external mode)
 */

#ifndef CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP
#define CATGPT_SELFPLAY_SELFPLAY_RUNNER_HPP

#include <atomic>
#include <chrono>
#include <cstdio>
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
#include "challenger_mcts.hpp"
#include "challenger_search.hpp"
#include "coroutine_mcts.hpp"
#include "coroutine_search.hpp"
#include "game_slot.hpp"
#include "selfplay_config.hpp"
#include "lc0_pool.hpp"
#include "stockfish_pool.hpp"
#include "syzygy.hpp"

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

        // Create batch evaluator (starts GPU thread) — skipped in external-vs-external mode
        bool needs_gpu = !(config_.use_stockfish && config_.use_lc0);
        if (needs_gpu) {
            evaluator_ = std::make_unique<BatchEvaluator>(
                config_.engine_path, pool_, config_.max_batch_size);
        }

        // Initialize Stockfish pool (if using Stockfish as opponent)
        if (config_.use_stockfish) {
            stockfish_pool_ = std::make_unique<StockfishPool>(
                config_.stockfish_path,
                config_.stockfish_processes,
                config_.stockfish_nodes,
                config_.stockfish_threads,
                config_.stockfish_hash,
                pool_);
        }

        // Initialize Lc0 pool (if using Lc0 as opponent)
        if (config_.use_lc0) {
            lc0_pool_ = std::make_unique<Lc0Pool>(
                config_.lc0_path,
                config_.lc0_weights,
                config_.lc0_processes,
                config_.lc0_nodes,
                config_.lc0_threads,
                config_.lc0_backend,
                config_.lc0_minibatch_size,
                pool_);
        }

        // Initialize Syzygy tablebases
        if (!config_.syzygy_path.empty()) {
            syzygy_ = std::make_unique<SyzygyProber>(config_.syzygy_path);
        }

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
        if (lc0_pool_) lc0_pool_->shutdown();
        if (stockfish_pool_) stockfish_pool_->shutdown();
        if (evaluator_) evaluator_->shutdown();
        if (pool_) pool_->shutdown();
    }

    /**
     * Run the tournament until the target number of game pairs is reached.
     */
    void run() {
        start_time_ = std::chrono::steady_clock::now();

        int total_games = config_.total_pairs * 2;
        std::println(stderr, "[SelfPlay] {} vs {}",
                     config_.challenger_name, config_.baseline_name);
        std::println(stderr, "[SelfPlay] Starting: {} concurrent slots, {} search threads, "
                     "max_batch={}, target={} pairs ({} games)",
                     config_.num_concurrent_games, config_.num_search_threads,
                     config_.max_batch_size, config_.total_pairs, total_games);

        coro::sync_wait(run_all_pairs());

        auto elapsed = std::chrono::steady_clock::now() - start_time_;
        double secs = std::chrono::duration<double>(elapsed).count();

        int completed = games_completed_.load();
        std::println(stderr, "\n[SelfPlay] Done: {} games in {:.1f}s ({:.1f} games/sec)",
                     completed, secs, completed / secs);
        if (evaluator_) {
            std::println(stderr, "[SelfPlay] GPU evals: {} ({:.0f} evals/sec)",
                         evaluator_->total_evals(),
                         evaluator_->total_evals() / secs);
        }
        print_stats();

        // JSON summary
        if (config_.json_metrics) {
            std::lock_guard lock(stats_mutex_);
            int total = challenger_wins_ + draws_ + challenger_losses_;
            float elo = estimate_elo(challenger_wins_, draws_, challenger_losses_);
            float avg_moves = total > 0 ? static_cast<float>(stats_total_moves_) / total : 0.0f;
            float avg_evals = total > 0 ? static_cast<float>(stats_total_evals_) / total : 0.0f;
            long long total_gpu_evals = evaluator_ ? static_cast<long long>(evaluator_->total_evals()) : 0LL;
            double gpu_evals_per_sec = (evaluator_ && secs > 0) ? evaluator_->total_evals() / secs : 0.0;
            std::printf(
                "{\"type\":\"summary\",\"games\":%d,"
                "\"challenger_wins\":%d,\"draws\":%d,\"challenger_losses\":%d,"
                "\"elo\":%.1f,\"avg_moves\":%.1f,\"avg_gpu_evals\":%.1f,"
                "\"total_secs\":%.1f,\"games_per_sec\":%.2f,"
                "\"total_gpu_evals\":%lld,\"gpu_evals_per_sec\":%.0f}\n",
                total,
                challenger_wins_, draws_, challenger_losses_,
                elo, avg_moves, avg_evals,
                secs, total > 0 ? completed / secs : 0.0,
                total_gpu_evals,
                gpu_evals_per_sec);
            std::fflush(stdout);
        }
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
     * Mode 1 (normal):
     *   Challenger = ChallengerSearch, Baseline = CoroutineSearch
     *
     * Mode 2 (Stockfish):
     *   Challenger = CoroutineSearch (CatGPT), Baseline = Stockfish
     *
     * Mode 3 (Lc0):
     *   Challenger = CoroutineSearch (CatGPT), Baseline = Lc0
     *
     * Mode 4 (Lc0 vs Stockfish):
     *   Challenger = Lc0, Baseline = Stockfish
     *
     * @param challenger_is_white  If true, the challenger plays White.
     */
    coro::task<GameRecord> play_one_game(const std::string& opening_fen,
                                         bool challenger_is_white) {
        GameSlot slot;
        slot.start(opening_fen);

        // baseline_white is the opposite of challenger_is_white
        bool baseline_white = !challenger_is_white;

        // Determine mode once
        bool both_external = config_.use_stockfish && config_.use_lc0;
        bool single_external = !both_external && (config_.use_stockfish || config_.use_lc0);

        while (!slot.is_terminated()) {
            bool white_to_move = slot.board().sideToMove() == chess::Color::WHITE;
            bool challenger_to_move = (white_to_move == challenger_is_white);

            MoveResult move_result;
            if (both_external) {
                // Mode 4: Lc0 (challenger) vs Stockfish (baseline)
                if (challenger_to_move) {
                    move_result = co_await Lc0Awaitable(*lc0_pool_, slot.board());
                } else {
                    move_result = co_await StockfishAwaitable(*stockfish_pool_, slot.board());
                }
            } else if (challenger_to_move) {
                if (single_external) {
                    // External engine mode: CatGPT is the challenger
                    move_result = co_await search_catgpt(
                        config_.challenger_config, config_.challenger_mcts_config,
                        slot.board());
                } else if (config_.search_type == SearchType::MCTS) {
                    // Normal MCTS mode: ChallengerMCTS is the challenger
                    ChallengerMCTS search(*evaluator_, config_.challenger_mcts_config);
                    move_result = co_await search.search_move(slot.board());
                } else {
                    // Normal fractional mode: ChallengerSearch is the challenger
                    ChallengerSearch search(*evaluator_, config_.challenger_config);
                    move_result = co_await search.search_move(slot.board());
                }
            } else {
                if (config_.use_stockfish) {
                    // Stockfish mode: Stockfish is the baseline
                    move_result = co_await StockfishAwaitable(*stockfish_pool_, slot.board());
                } else if (config_.use_lc0) {
                    // Lc0 mode: Lc0 is the baseline
                    move_result = co_await Lc0Awaitable(*lc0_pool_, slot.board());
                } else {
                    // Normal mode: CoroutineSearch or CoroutineMCTS is the baseline
                    move_result = co_await search_catgpt(
                        config_.baseline_config, config_.baseline_mcts_config,
                        slot.board());
                }
            }

            if (move_result.best_move == chess::Move::NO_MOVE) {
                break;
            }

            slot.apply_move(move_result.best_move, move_result.cp_score, move_result.gpu_evals);
            slot.check_game_over(config_, syzygy_.get());
        }

        auto record = slot.to_record();
        record.baseline_white = baseline_white;
        co_return record;
    }

    // ─── CatGPT search dispatch ────────────────────────────────────────

    /**
     * Run CatGPT search using the configured algorithm (Fractional MCTS or MCTS).
     */
    coro::task<MoveResult> search_catgpt(const FractionalMCTSConfig& frac_config,
                                         const MCTSConfig& mcts_config,
                                         const chess::Board& board) {
        if (config_.search_type == SearchType::MCTS) {
            CoroutineMCTS search(*evaluator_, mcts_config);
            co_return co_await search.search_move(board);
        } else {
            CoroutineSearch search(*evaluator_, frac_config);
            co_return co_await search.search_move(board);
        }
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

    [[nodiscard]] static const char* termination_name(GameTermination t) {
        switch (t) {
            case GameTermination::ONGOING:                return "ongoing";
            case GameTermination::CHECKMATE:              return "checkmate";
            case GameTermination::STALEMATE:              return "stalemate";
            case GameTermination::INSUFFICIENT_MATERIAL:  return "insufficient_material";
            case GameTermination::THREEFOLD_REPETITION:   return "threefold_repetition";
            case GameTermination::FIFTY_MOVE_RULE:        return "fifty_move_rule";
            case GameTermination::DRAW_ADJUDICATED:       return "draw_adjudicated";
            case GameTermination::RESIGN_ADJUDICATED:     return "resign_adjudicated";
            case GameTermination::SYZYGY_ADJUDICATED:     return "syzygy_adjudicated";
            case GameTermination::MAX_MOVES:              return "max_moves";
        }
        return "unknown";
    }

    void on_game_complete(const GameRecord& record, int game_num, int slot_id) {
        int completed = games_completed_.fetch_add(1) + 1;

        // Track from challenger's perspective
        float score = record.baseline_score();
        // baseline_score() returns score for baseline; invert for challenger
        float challenger_score = 1.0f - score;

        bool challenger_won = false;
        {
            std::lock_guard lock(stats_mutex_);
            if (challenger_score > 0.75f) {
                ++challenger_wins_;
                challenger_won = true;
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

        // JSON metrics to stdout (for Python wrapper / wandb)
        if (config_.json_metrics) {
            std::lock_guard lock(stats_mutex_);
            float elo = estimate_elo(challenger_wins_, draws_, challenger_losses_);
            int total = challenger_wins_ + draws_ + challenger_losses_;
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time_).count();
            double games_per_sec = elapsed > 0.0 ? completed / elapsed : 0.0;

            // Manual JSON construction (no dependency needed)
            std::printf(
                "{\"type\":\"game\",\"game_num\":%d,"
                "\"challenger_wins\":%d,\"draws\":%d,\"challenger_losses\":%d,"
                "\"games\":%d,\"elo\":%.1f,"
                "\"game_moves\":%d,\"game_gpu_evals\":%d,"
                "\"termination\":\"%s\",\"challenger_won\":%s,"
                "\"avg_moves\":%.1f,\"avg_gpu_evals\":%.1f,"
                "\"games_per_sec\":%.2f,\"elapsed_secs\":%.1f}\n",
                completed,
                challenger_wins_, draws_, challenger_losses_,
                total, elo,
                static_cast<int>(record.moves.size()), record.total_gpu_evals,
                termination_name(record.termination),
                challenger_won ? "true" : (record.outcome == GameOutcome::DRAW ? "false" : "false"),
                total > 0 ? static_cast<float>(stats_total_moves_) / total : 0.0f,
                total > 0 ? static_cast<float>(stats_total_evals_) / total : 0.0f,
                games_per_sec, elapsed);
            std::fflush(stdout);
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
                bool chal_won = (white_won != record.baseline_white);
                winner = chal_won ? config_.challenger_name : config_.baseline_name;
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

            pgn_file_ << chess::uci::moveToSan(board, record.moves[i]);

            // Annotate with eval and GPU evals used
            if (i < record.cp_scores_per_move.size() || i < record.gpu_evals_per_move.size()) {
                pgn_file_ << " {";
                bool need_sep = false;
                if (i < record.cp_scores_per_move.size()) {
                    pgn_file_ << "cp=" << record.cp_scores_per_move[i];
                    need_sep = true;
                }
                if (i < record.gpu_evals_per_move.size()) {
                    if (need_sep) pgn_file_ << ", ";
                    pgn_file_ << "evals=" << record.gpu_evals_per_move[i];
                }
                pgn_file_ << "}";
            }

            pgn_file_ << " ";

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
    std::unique_ptr<StockfishPool> stockfish_pool_;
    std::unique_ptr<Lc0Pool> lc0_pool_;
    std::unique_ptr<SyzygyProber> syzygy_;

    // PGN output
    std::ofstream pgn_file_;
    std::mutex pgn_mutex_;

    // Timing
    std::chrono::steady_clock::time_point start_time_;

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
