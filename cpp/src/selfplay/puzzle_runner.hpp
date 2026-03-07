/**
 * Puzzle Evaluation Runner — batched puzzle solving with coroutines.
 *
 * Uses the same coroutine + BatchEvaluator infrastructure as self-play
 * to evaluate chess puzzles with batched GPU inference. Many puzzles
 * run concurrently as coroutines; when any search needs a GPU eval,
 * it suspends and the request is batched with others.
 *
 * Architecture:
 *   Main thread   → loads puzzles, runs event loop, prints summary
 *   Thread pool   → runs puzzle-solving coroutines (N worker threads)
 *   GPU thread    → batches and runs TRT inference
 *
 * Input:  Lichess puzzle CSV (PuzzleId, FEN, Moves, Rating, ...)
 * Output: JSON-lines to stdout (one per puzzle + summary)
 */

#ifndef CATGPT_SELFPLAY_PUZZLE_RUNNER_HPP
#define CATGPT_SELFPLAY_PUZZLE_RUNNER_HPP

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <print>
#include <sstream>
#include <string>
#include <vector>

#include <coro/sync_wait.hpp>
#include <coro/task.hpp>
#include <coro/thread_pool.hpp>
#include <coro/when_all.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "batch_evaluator.hpp"
#include "coroutine_search.hpp"

namespace catgpt {

// ─── Puzzle data structures ─────────────────────────────────────────────────

struct Puzzle {
    std::string id;
    int rating = 0;
    std::string fen;
    std::vector<std::string> moves;
};

struct PuzzleResult {
    std::string puzzle_id;
    int rating = 0;
    bool solved = false;
    int moves_correct = 0;
    int moves_total = 0;
};

struct PuzzleEvalConfig {
    int num_concurrent = 128;
    int num_search_threads = 8;
    int max_batch_size = 64;
    int max_puzzles = 0;  // 0 = all
    FractionalMCTSConfig search_config{};
    std::string engine_path;
    std::string puzzle_csv;
};

// ─── CSV parsing ────────────────────────────────────────────────────────────

/**
 * Parse a single CSV field, handling quoted fields.
 * Advances pos past the field and its delimiter.
 */
inline std::string parse_csv_field(const std::string& line, size_t& pos) {
    if (pos >= line.size()) return "";

    std::string field;
    if (line[pos] == '"') {
        ++pos;
        while (pos < line.size()) {
            if (line[pos] == '"') {
                if (pos + 1 < line.size() && line[pos + 1] == '"') {
                    field += '"';
                    pos += 2;
                } else {
                    ++pos;  // closing quote
                    break;
                }
            } else {
                field += line[pos++];
            }
        }
        if (pos < line.size() && line[pos] == ',') ++pos;
    } else {
        auto comma = line.find(',', pos);
        if (comma == std::string::npos) {
            field = line.substr(pos);
            pos = line.size();
        } else {
            field = line.substr(pos, comma - pos);
            pos = comma + 1;
        }
    }
    return field;
}

/**
 * Load puzzles from a Lichess-format CSV file.
 *
 * Expected columns: PuzzleId, FEN, Moves, Rating, ...
 * The Moves column is space-separated UCI moves.
 */
inline std::vector<Puzzle> load_puzzles_csv(const std::string& path, int max_puzzles = 0) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open puzzle CSV: " + path);
    }

    // Read and parse header to find column indices
    std::string header_line;
    if (!std::getline(file, header_line)) {
        throw std::runtime_error("Empty puzzle CSV: " + path);
    }

    int col_id = -1, col_fen = -1, col_moves = -1, col_rating = -1;
    {
        size_t pos = 0;
        int col = 0;
        while (pos < header_line.size()) {
            std::string name = parse_csv_field(header_line, pos);
            if (name == "PuzzleId") col_id = col;
            else if (name == "FEN") col_fen = col;
            else if (name == "Moves") col_moves = col;
            else if (name == "Rating") col_rating = col;
            ++col;
        }
    }

    if (col_id < 0 || col_fen < 0 || col_moves < 0 || col_rating < 0) {
        throw std::runtime_error("CSV missing required columns (PuzzleId, FEN, Moves, Rating)");
    }

    std::vector<Puzzle> puzzles;
    std::string line;
    while (std::getline(file, line)) {
        if (max_puzzles > 0 && static_cast<int>(puzzles.size()) >= max_puzzles) break;
        if (line.empty()) continue;

        // Parse all fields
        std::vector<std::string> fields;
        size_t pos = 0;
        while (pos < line.size()) {
            fields.push_back(parse_csv_field(line, pos));
        }

        int max_col = std::max({col_id, col_fen, col_moves, col_rating});
        if (static_cast<int>(fields.size()) <= max_col) continue;

        Puzzle p;
        p.id = fields[col_id];
        p.fen = fields[col_fen];
        p.rating = std::stoi(fields[col_rating]);

        // Split moves on spaces
        std::istringstream iss(fields[col_moves]);
        std::string move;
        while (iss >> move) {
            p.moves.push_back(move);
        }

        if (!p.moves.empty()) {
            puzzles.push_back(std::move(p));
        }
    }

    return puzzles;
}

// ─── Puzzle Runner ──────────────────────────────────────────────────────────

class PuzzleRunner {
public:
    explicit PuzzleRunner(const PuzzleEvalConfig& config)
        : config_(config)
    {
        // Load puzzles
        puzzles_ = load_puzzles_csv(config_.puzzle_csv, config_.max_puzzles);
        std::println(stderr, "[PuzzleEval] Loaded {} puzzles from {}", puzzles_.size(), config_.puzzle_csv);

        if (puzzles_.empty()) {
            throw std::runtime_error("No puzzles loaded");
        }

        // Create thread pool
        pool_ = coro::thread_pool::make_shared(coro::thread_pool::options{
            .thread_count = static_cast<uint32_t>(config_.num_search_threads),
        });

        // Create batch evaluator (starts GPU thread)
        evaluator_ = std::make_unique<BatchEvaluator>(
            config_.engine_path, pool_, config_.max_batch_size);
    }

    ~PuzzleRunner() {
        if (evaluator_) evaluator_->shutdown();
        if (pool_) pool_->shutdown();
    }

    void run() {
        start_time_ = std::chrono::steady_clock::now();

        int num_puzzles = static_cast<int>(puzzles_.size());
        std::println(stderr, "[PuzzleEval] Starting: {} puzzles, {} concurrent, {} threads, batch={}",
                     num_puzzles, config_.num_concurrent, config_.num_search_threads,
                     config_.max_batch_size);
        std::println(stderr, "[PuzzleEval] Search: evals={}, cpuct={:.2f}",
                     config_.search_config.min_total_evals, config_.search_config.c_puct);

        coro::sync_wait(run_all_puzzles());

        auto elapsed = std::chrono::steady_clock::now() - start_time_;
        double secs = std::chrono::duration<double>(elapsed).count();

        int completed = puzzles_completed_.load();
        std::println(stderr, "\n[PuzzleEval] Done: {} puzzles in {:.1f}s ({:.1f} puzzles/sec)",
                     completed, secs, completed / secs);
        std::println(stderr, "[PuzzleEval] GPU evals: {} ({:.0f} evals/sec)",
                     evaluator_->total_evals(),
                     evaluator_->total_evals() / secs);

        print_summary(secs);
    }

private:
    // ─── Coroutine orchestration ────────────────────────────────────────

    coro::task<void> run_all_puzzles() {
        co_await pool_->schedule();

        int num_slots = std::min(config_.num_concurrent, static_cast<int>(puzzles_.size()));

        std::vector<coro::task<void>> workers;
        workers.reserve(num_slots);

        for (int slot = 0; slot < num_slots; ++slot) {
            workers.push_back(puzzle_worker(slot));
        }

        co_await coro::when_all(std::move(workers));
    }

    coro::task<void> puzzle_worker(int /*slot_id*/) {
        co_await pool_->schedule();

        int total = static_cast<int>(puzzles_.size());

        while (true) {
            int idx = next_puzzle_.fetch_add(1);
            if (idx >= total) break;

            PuzzleResult result = co_await solve_puzzle(puzzles_[idx]);
            on_puzzle_complete(result, idx);
        }
    }

    /**
     * Solve a single puzzle: iterate through its moves, using the search
     * engine for "engine turns" (odd-indexed moves) and applying opponent
     * moves directly.
     */
    coro::task<PuzzleResult> solve_puzzle(const Puzzle& puzzle) {
        PuzzleResult result;
        result.puzzle_id = puzzle.id;
        result.rating = puzzle.rating;

        chess::Board board(puzzle.fen);

        for (size_t i = 0; i < puzzle.moves.size(); ++i) {
            const auto& uci_move_str = puzzle.moves[i];

            if (i % 2 == 0) {
                // Opponent's move — just apply it
                auto move = chess::uci::uciToMove(board, uci_move_str);
                if (move == chess::Move::NO_MOVE) break;
                board.makeMove<true>(move);
            } else {
                // Engine must find this move
                result.moves_total += 1;

                CoroutineSearch search(*evaluator_, config_.search_config);
                MoveResult move_result = co_await search.search_move(board);

                auto expected = chess::uci::uciToMove(board, uci_move_str);
                if (move_result.best_move == expected) {
                    result.moves_correct += 1;
                }

                // Apply the correct move to continue the puzzle
                if (expected == chess::Move::NO_MOVE) break;
                board.makeMove<true>(expected);
            }
        }

        result.solved = (result.moves_correct == result.moves_total && result.moves_total > 0);
        co_return result;
    }

    // ─── Result tracking ────────────────────────────────────────────────

    void on_puzzle_complete(const PuzzleResult& result, int /*idx*/) {
        int completed = puzzles_completed_.fetch_add(1) + 1;

        {
            std::lock_guard lock(stats_mutex_);
            total_correct_ += result.moves_correct;
            total_moves_ += result.moves_total;
            if (result.solved) ++total_solved_;
        }

        // JSON line for this puzzle
        std::printf(
            "{\"type\":\"puzzle\",\"id\":\"%s\",\"rating\":%d,"
            "\"solved\":%s,\"moves_correct\":%d,\"moves_total\":%d}\n",
            result.puzzle_id.c_str(), result.rating,
            result.solved ? "true" : "false",
            result.moves_correct, result.moves_total);
        std::fflush(stdout);

        // Progress to stderr
        int total = static_cast<int>(puzzles_.size());
        if (completed <= 5 || completed % 100 == 0 || completed == total) {
            std::lock_guard lock(stats_mutex_);
            float solve_rate = completed > 0
                ? static_cast<float>(total_solved_) / completed : 0.0f;
            float accuracy = total_moves_ > 0
                ? static_cast<float>(total_correct_) / total_moves_ : 0.0f;

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time_).count();
            double per_sec = elapsed > 0.0 ? completed / elapsed : 0.0;

            std::println(stderr, "[PuzzleEval] {}/{} ({:.1f}/s) | acc={:.1f}% solve={:.1f}% ({}/{})",
                         completed, total, per_sec,
                         accuracy * 100.0f, solve_rate * 100.0f, total_solved_, completed);
        }
    }

    void print_summary(double elapsed_secs) {
        std::lock_guard lock(stats_mutex_);

        int total = puzzles_completed_.load();
        float accuracy = total_moves_ > 0
            ? static_cast<float>(total_correct_) / total_moves_ : 0.0f;
        float solve_rate = total > 0
            ? static_cast<float>(total_solved_) / total : 0.0f;

        std::println(stderr, "\n[PuzzleEval] Summary:");
        std::println(stderr, "  Puzzles:       {}", total);
        std::println(stderr, "  Solved:        {} ({:.1f}%)", total_solved_, solve_rate * 100.0f);
        std::println(stderr, "  Move accuracy: {:.1f}% ({}/{})", accuracy * 100.0f, total_correct_, total_moves_);
        std::println(stderr, "  GPU evals:     {}", evaluator_->total_evals());
        std::println(stderr, "  Elapsed:       {:.1f}s ({:.1f} puzzles/sec)",
                     elapsed_secs, total > 0 ? total / elapsed_secs : 0.0);

        // Summary JSON
        std::printf(
            "{\"type\":\"summary\",\"num_puzzles\":%d,"
            "\"num_solved\":%d,\"solve_rate\":%.6f,"
            "\"move_accuracy\":%.6f,\"total_moves\":%d,\"correct_moves\":%d,"
            "\"total_gpu_evals\":%lld,\"elapsed_secs\":%.1f,"
            "\"puzzles_per_sec\":%.2f}\n",
            total, total_solved_, solve_rate, accuracy,
            total_moves_, total_correct_,
            static_cast<long long>(evaluator_->total_evals()),
            elapsed_secs,
            total > 0 ? total / elapsed_secs : 0.0);
        std::fflush(stdout);
    }

    // ─── Members ────────────────────────────────────────────────────────

    PuzzleEvalConfig config_;
    std::vector<Puzzle> puzzles_;

    std::shared_ptr<coro::thread_pool> pool_;
    std::unique_ptr<BatchEvaluator> evaluator_;

    // Timing
    std::chrono::steady_clock::time_point start_time_;

    // Progress
    std::atomic<int> next_puzzle_{0};
    std::atomic<int> puzzles_completed_{0};

    // Aggregate stats
    std::mutex stats_mutex_;
    int total_solved_ = 0;
    int total_correct_ = 0;
    int total_moves_ = 0;
};

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_PUZZLE_RUNNER_HPP
