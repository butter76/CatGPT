/**
 * Lc0 Process Pool — async UCI engine integration for self-play.
 *
 * Manages a pool of Lc0 (Leela Chess Zero) subprocesses, each on a dedicated
 * I/O thread.  Search coroutines submit requests via Lc0Awaitable (mirroring
 * StockfishAwaitable / EvalAwaitable), suspending the coroutine so thread-pool
 * threads are never blocked on Lc0 I/O.
 *
 * Protocol per move (no tree reuse):
 *   ucinewgame
 *   isready         → wait for readyok
 *   position fen <FEN>
 *   go nodes <N>    → read until bestmove
 *
 * During the UCI handshake we set:
 *   WeightsFile  — explicit path (avoids lc0's fragile autodiscover)
 *   ScoreType    — centipawn (so we can reuse parse_info_score from SF)
 *   Threads      — CPU worker threads per process
 *   Backend      — neural-net backend (default: cuda-auto)
 *
 * Each Lc0Process owns one subprocess + pipes; worker threads pull
 * requests from a shared queue and resume coroutines on the thread pool
 * when the result is ready.
 */

#ifndef CATGPT_SELFPLAY_LC0_POOL_HPP
#define CATGPT_SELFPLAY_LC0_POOL_HPP

#include <array>
#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <print>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <coro/thread_pool.hpp>

#include "../../external/chess-library/include/chess.hpp"
#include "coroutine_search.hpp"  // For MoveResult

namespace catgpt {

// ─── Lc0 Request (the coroutine-to-lc0 bridge) ─────────────────────────────

/**
 * A single Lc0 search request.
 *
 * Allocated inside the coroutine frame (via Lc0Awaitable).
 * A worker thread reads `fen`/`nodes`, writes `result`, then resumes
 * the coroutine via `continuation`.
 */
struct Lc0Request {
    // --- Input (written by coroutine before suspend) ---
    std::string fen;
    int nodes = 0;

    // --- Board reference for parsing the UCI move ---
    // (needed by chess::uci::uciToMove)
    chess::Board board;

    // --- Output (written by worker thread before resuming coroutine) ---
    MoveResult result;

    // --- Coroutine handle ---
    std::coroutine_handle<> continuation;
};

// Forward declaration
class Lc0Pool;

// ─── Lc0Awaitable ───────────────────────────────────────────────────────────

/**
 * Awaitable that submits a search request to the Lc0 pool
 * and suspends the calling coroutine until the result is ready.
 *
 * Usage inside a coroutine:
 *   MoveResult result = co_await Lc0Awaitable(pool, board);
 */
class Lc0Awaitable {
public:
    Lc0Awaitable(Lc0Pool& pool, const chess::Board& board)
        : pool_(pool)
    {
        request_.fen = board.getFen();
        request_.board = board;
    }

    bool await_ready() const noexcept { return false; }

    // Defined after Lc0Pool is complete (below).
    void await_suspend(std::coroutine_handle<> h) noexcept;

    MoveResult await_resume() noexcept {
        return request_.result;
    }

private:
    Lc0Pool& pool_;
    Lc0Request request_;
};

// ─── Lc0Process (one subprocess) ────────────────────────────────────────────

/**
 * Manages a single Lc0 subprocess with stdin/stdout pipes.
 *
 * Not thread-safe — each instance is owned by exactly one worker thread.
 */
class Lc0Process {
public:
    Lc0Process(const std::string& binary_path,
               const std::string& weights_path,
               int threads,
               const std::string& backend,
               int minibatch_size)
        : pid_(-1)
    {
        spawn(binary_path);
        uci_handshake(weights_path, threads, backend, minibatch_size);
    }

    ~Lc0Process() {
        shutdown();
    }

    // Non-copyable, non-movable
    Lc0Process(const Lc0Process&) = delete;
    Lc0Process& operator=(const Lc0Process&) = delete;

    /**
     * Execute one search: ucinewgame → isready → position → go nodes → bestmove.
     * Blocks the calling thread until Lc0 returns bestmove.
     */
    MoveResult search(const std::string& fen, int nodes, const chess::Board& board) {
        MoveResult result;

        // Clear the tree
        send_line("ucinewgame");
        send_line("isready");
        wait_for("readyok");

        // Set position
        send_line("position fen " + fen);

        // Start search
        send_line("go nodes " + std::to_string(nodes));

        // Read until bestmove, tracking the last score
        int last_cp = 0;
        bool found_score = false;

        while (true) {
            std::string line = read_line();
            if (line.empty()) continue;

            // Parse info lines for score (works because we set ScoreType=centipawn)
            if (line.starts_with("info ")) {
                parse_info_score(line, last_cp, found_score);
            }

            // Parse bestmove
            if (line.starts_with("bestmove ")) {
                auto move_str = extract_bestmove(line);
                if (!move_str.empty()) {
                    result.best_move = chess::uci::uciToMove(board, move_str);
                }
                break;
            }
        }

        result.cp_score = found_score ? last_cp : 0;
        result.gpu_evals = 0;  // Not counted for external engines
        result.iterations = 0;
        return result;
    }

    void shutdown() {
        if (pid_ <= 0) return;

        // Try graceful quit
        try {
            send_line("quit");
        } catch (...) {}

        // Close pipes
        if (write_fd_ >= 0) { close(write_fd_); write_fd_ = -1; }
        if (read_fd_ >= 0) { close(read_fd_); read_fd_ = -1; }

        // Wait for child (with timeout via SIGKILL fallback)
        int status = 0;
        pid_t ret = waitpid(pid_, &status, WNOHANG);
        if (ret == 0) {
            // Still running — give it a moment then kill
            usleep(100'000);  // 100ms
            ret = waitpid(pid_, &status, WNOHANG);
            if (ret == 0) {
                kill(pid_, SIGKILL);
                waitpid(pid_, &status, 0);
            }
        }
        pid_ = -1;
    }

private:
    void spawn(const std::string& binary_path) {
        int stdin_pipe[2];
        int stdout_pipe[2];

        if (pipe(stdin_pipe) != 0 || pipe(stdout_pipe) != 0) {
            throw std::runtime_error("Lc0Process: pipe() failed: " +
                                     std::string(strerror(errno)));
        }

        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error("Lc0Process: fork() failed: " +
                                     std::string(strerror(errno)));
        }

        if (pid_ == 0) {
            // Child process
            close(stdin_pipe[1]);   // Close write end of stdin pipe
            close(stdout_pipe[0]);  // Close read end of stdout pipe

            dup2(stdin_pipe[0], STDIN_FILENO);
            dup2(stdout_pipe[1], STDOUT_FILENO);
            dup2(stdout_pipe[1], STDERR_FILENO);

            close(stdin_pipe[0]);
            close(stdout_pipe[1]);

            execlp(binary_path.c_str(), binary_path.c_str(), nullptr);
            // If exec fails:
            _exit(127);
        }

        // Parent process
        close(stdin_pipe[0]);   // Close read end of stdin pipe
        close(stdout_pipe[1]);  // Close write end of stdout pipe

        write_fd_ = stdin_pipe[1];
        read_fd_ = stdout_pipe[0];

        // Buffer for reading
        read_buf_.reserve(4096);
    }

    void uci_handshake(const std::string& weights_path, int threads,
                       const std::string& backend, int minibatch_size) {
        send_line("uci");
        wait_for("uciok");

        // Explicit weights path — avoids lc0's fragile autodiscover mechanism
        send_line("setoption name WeightsFile value " + weights_path);

        // Force centipawn score output so we can reuse the same parser as Stockfish
        send_line("setoption name ScoreType value centipawn");

        if (threads > 0) {
            send_line("setoption name Threads value " + std::to_string(threads));
        }

        if (!backend.empty()) {
            send_line("setoption name Backend value " + backend);
        }

        if (minibatch_size > 0) {
            send_line("setoption name MinibatchSize value " + std::to_string(minibatch_size));
        }

        send_line("isready");
        wait_for("readyok");
    }

    void send_line(const std::string& line) {
        std::string data = line + "\n";
        ssize_t written = write(write_fd_, data.c_str(), data.size());
        if (written < 0) {
            throw std::runtime_error("Lc0Process: write() failed: " +
                                     std::string(strerror(errno)));
        }
    }

    /**
     * Read one line from Lc0's stdout.
     * Blocks until a full line is available.
     */
    std::string read_line() {
        while (true) {
            // Check if we already have a complete line in the buffer
            auto pos = read_buf_.find('\n');
            if (pos != std::string::npos) {
                std::string line = read_buf_.substr(0, pos);
                read_buf_.erase(0, pos + 1);
                // Strip trailing \r
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                return line;
            }

            // Read more data
            char buf[4096];
            ssize_t n = read(read_fd_, buf, sizeof(buf));
            if (n <= 0) {
                if (n == 0) {
                    throw std::runtime_error("Lc0Process: EOF on read");
                }
                if (errno == EINTR) continue;
                throw std::runtime_error("Lc0Process: read() failed: " +
                                         std::string(strerror(errno)));
            }
            read_buf_.append(buf, static_cast<size_t>(n));
        }
    }

    /**
     * Read lines until we see one that starts with the expected prefix.
     */
    void wait_for(const std::string& expected) {
        while (true) {
            std::string line = read_line();
            if (line.starts_with(expected)) return;
        }
    }

    /**
     * Parse "info ... score cp X ..." or "info ... score mate X ..." from a line.
     * Works because we set ScoreType=centipawn in the UCI handshake.
     */
    static void parse_info_score(const std::string& line, int& cp_out, bool& found) {
        auto score_pos = line.find(" score ");
        if (score_pos == std::string::npos) return;

        auto after_score = score_pos + 7;  // strlen(" score ")

        if (line.substr(after_score, 3) == "cp ") {
            auto num_start = after_score + 3;
            auto num_end = line.find(' ', num_start);
            std::string num_str = (num_end != std::string::npos)
                ? line.substr(num_start, num_end - num_start)
                : line.substr(num_start);
            try {
                cp_out = std::stoi(num_str);
                found = true;
            } catch (...) {}
        } else if (line.substr(after_score, 5) == "mate ") {
            auto num_start = after_score + 5;
            auto num_end = line.find(' ', num_start);
            std::string num_str = (num_end != std::string::npos)
                ? line.substr(num_start, num_end - num_start)
                : line.substr(num_start);
            try {
                int mate_in = std::stoi(num_str);
                cp_out = mate_in > 0 ? (30000 - mate_in) : (-30000 - mate_in);
                found = true;
            } catch (...) {}
        }
    }

    /**
     * Extract the move string from "bestmove e2e4 ..." or "bestmove e2e4".
     */
    static std::string extract_bestmove(const std::string& line) {
        // "bestmove " is 9 chars
        if (line.size() < 13) return "";  // "bestmove e2e4" = 13 chars minimum

        auto start = 9;
        auto end = line.find(' ', start);
        return (end != std::string::npos)
            ? line.substr(start, end - start)
            : line.substr(start);
    }

    pid_t pid_;
    int write_fd_ = -1;
    int read_fd_ = -1;
    std::string read_buf_;
};

// ─── Lc0Pool ────────────────────────────────────────────────────────────────

/**
 * Pool of Lc0 processes with an async awaitable interface.
 *
 * Worker threads own one Lc0Process each and pull requests from
 * a shared queue.  When the queue is empty, they sleep on a condition
 * variable (zero CPU usage).  When a result is ready, the coroutine
 * is resumed on the coro::thread_pool.
 */
class Lc0Pool {
public:
    /**
     * @param binary_path    Path to lc0 binary.
     * @param weights_path   Path to neural network weights file (.pb.gz).
     * @param num_processes  Number of concurrent Lc0 subprocesses.
     * @param nodes          Fixed node count for `go nodes`.
     * @param threads_per    UCI Threads option per process.
     * @param backend        Neural-net backend (e.g. "cuda-auto", "eigen").
     * @param minibatch_size MinibatchSize for NN computation (0 = backend default).
     * @param thread_pool    Shared thread pool for resuming coroutines.
     */
    Lc0Pool(const std::string& binary_path,
            const std::string& weights_path,
            int num_processes,
            int nodes,
            int threads_per,
            const std::string& backend,
            int minibatch_size,
            std::shared_ptr<coro::thread_pool> thread_pool)
        : nodes_(nodes)
        , thread_pool_(std::move(thread_pool))
        , shutdown_(false)
    {
        std::println(stderr, "[Lc0Pool] Spawning {} Lc0 processes "
                     "(nodes={}, threads={}, backend={}, minibatch={}, weights={})",
                     num_processes, nodes, threads_per, backend, minibatch_size, weights_path);

        // Launch worker threads (each creates its own Lc0Process)
        workers_.reserve(num_processes);
        for (int i = 0; i < num_processes; ++i) {
            workers_.emplace_back([this, binary_path, weights_path, threads_per, backend, minibatch_size, i]() {
                worker_loop(binary_path, weights_path, threads_per, backend, minibatch_size, i);
            });
        }

        std::println(stderr, "[Lc0Pool] All {} workers ready", num_processes);
    }

    ~Lc0Pool() {
        shutdown();
    }

    // Non-copyable, non-movable
    Lc0Pool(const Lc0Pool&) = delete;
    Lc0Pool& operator=(const Lc0Pool&) = delete;

    /**
     * Submit a search request.  Called from Lc0Awaitable::await_suspend.
     */
    void submit(Lc0Request* request) {
        request->nodes = nodes_;
        {
            std::lock_guard lock(queue_mutex_);
            queue_.push_back(request);
        }
        queue_cv_.notify_one();
    }

    /**
     * Shut down all workers and Lc0 processes.
     */
    void shutdown() {
        if (shutdown_.exchange(true)) return;

        queue_cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();
    }

    /** Number of searches completed. */
    [[nodiscard]] int64_t total_searches() const noexcept {
        return total_searches_.load(std::memory_order_relaxed);
    }

private:
    void worker_loop(const std::string& binary_path,
                     const std::string& weights_path,
                     int threads_per,
                     const std::string& backend,
                     int minibatch_size,
                     int worker_id) {
        // Each worker owns its own Lc0 process
        std::unique_ptr<Lc0Process> lc0;
        try {
            lc0 = std::make_unique<Lc0Process>(binary_path, weights_path, threads_per, backend, minibatch_size);
        } catch (const std::exception& e) {
            std::println(stderr, "[Lc0Pool] Worker {} failed to start Lc0: {}",
                         worker_id, e.what());
            return;
        }

        while (true) {
            Lc0Request* request = nullptr;

            // Wait for a request
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() {
                    return !queue_.empty() || shutdown_.load();
                });

                if (shutdown_.load() && queue_.empty()) {
                    break;
                }

                if (!queue_.empty()) {
                    request = queue_.front();
                    queue_.pop_front();
                }
            }

            if (!request) continue;

            // Execute the search (blocks this thread on Lc0 I/O — that's fine,
            // this is a dedicated worker thread, not a coroutine thread-pool thread)
            try {
                request->result = lc0->search(request->fen, request->nodes, request->board);
            } catch (const std::exception& e) {
                std::println(stderr, "[Lc0Pool] Worker {} search error: {}",
                             worker_id, e.what());
                request->result.best_move = chess::Move::NO_MOVE;
                request->result.cp_score = 0;

                // Try to restart Lc0
                lc0.reset();
                try {
                    lc0 = std::make_unique<Lc0Process>(binary_path, weights_path,
                                                        threads_per, backend, minibatch_size);
                    std::println(stderr, "[Lc0Pool] Worker {} restarted Lc0", worker_id);
                } catch (const std::exception& e2) {
                    std::println(stderr, "[Lc0Pool] Worker {} failed to restart: {}",
                                 worker_id, e2.what());
                    // Resume the coroutine anyway to avoid deadlock, then exit
                    thread_pool_->resume(request->continuation);
                    total_searches_.fetch_add(1, std::memory_order_relaxed);
                    break;
                }
            }

            // Resume the coroutine on the worker thread pool
            thread_pool_->resume(request->continuation);
            total_searches_.fetch_add(1, std::memory_order_relaxed);
        }

        // Shutdown the Lc0 process
        if (lc0) lc0->shutdown();
    }

    int nodes_;
    std::shared_ptr<coro::thread_pool> thread_pool_;

    // Request queue
    std::deque<Lc0Request*> queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Worker threads
    std::vector<std::thread> workers_;
    std::atomic<bool> shutdown_;

    // Stats
    std::atomic<int64_t> total_searches_{0};
};

// ─── Lc0Awaitable::await_suspend ────────────────────────────────────────────

inline void Lc0Awaitable::await_suspend(std::coroutine_handle<> h) noexcept {
    request_.continuation = h;
    pool_.submit(&request_);
}

}  // namespace catgpt

#endif  // CATGPT_SELFPLAY_LC0_POOL_HPP
