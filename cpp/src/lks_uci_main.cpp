/**
 * CatGPT LKS UCI Engine - Main Entry Point
 *
 * Bespoke UCI loop on top of LksSearch (Lazy-SMP, log-scale ID, GPU-batched
 * via TensorRT). LksSearch is natively async: search() spawns a worker
 * jthread and returns immediately; quit() joins it. The worker emits
 * info/bestmove lines via on_uci_line, which we wire straight to stdout.
 *
 * Supported commands: uci, isready, ucinewgame, position, go, stop, quit.
 *
 * Tree reuse across `position` commands: if the new (fen, moves) is a
 * strict prefix-extension of the previous one, only the new tail of moves
 * is replayed via makemove(), which preserves the shared TT in the arena.
 *
 * Usage: lks_uci [engine_path]
 *   engine_path: Path to TensorRT engine file
 *                (default: $CATGPT_TRT_ENGINE or /home/shadeform/CatGPT/main.trt)
 *
 * Tuning env vars (defaults match LksSearch ctor defaults):
 *   LKS_NUM_WORKERS         (default 2)
 *   LKS_COROS_PER_WORKER    (default 8)
 *   LKS_MAX_BATCH_SIZE      (default 32)
 *   LKS_LIFETIME_MAX_EVALS  (default 1<<20)
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <print>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../external/chess-library/include/chess.hpp"
#include "engine/lks/lks_search.hpp"

namespace fs = std::filesystem;

namespace catgpt {

inline constexpr std::string_view ENGINE_NAME = "CatGPT-LKS";
inline constexpr std::string_view ENGINE_AUTHOR = "CatGPT Team";

// Canonical startpos FEN; chess::Board(STARTPOS_FEN) accepts this.
inline constexpr std::string_view STARTPOS_FEN =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static std::vector<std::string> tokenize(std::string_view line) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : line) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!token.empty()) {
                tokens.push_back(std::move(token));
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    return tokens;
}

static int env_int(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    try { return std::stoi(s); } catch (...) { return fallback; }
}

static uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    try { return std::stoull(s); } catch (...) { return fallback; }
}

class LksUciDriver {
public:
    LksUciDriver(fs::path engine_path,
                 uint64_t lifetime_max_evals,
                 int num_workers,
                 int coros_per_worker,
                 int max_batch_size)
        : search_(std::move(engine_path), lifetime_max_evals,
                  num_workers, coros_per_worker, max_batch_size)
    {}

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            auto tokens = tokenize(line);
            if (tokens.empty()) continue;
            const auto& cmd = tokens[0];

            if (cmd == "uci") {
                handle_uci();
            } else if (cmd == "isready") {
                handle_isready();
            } else if (cmd == "ucinewgame") {
                handle_ucinewgame();
            } else if (cmd == "position") {
                handle_position(tokens);
            } else if (cmd == "go") {
                handle_go(tokens);
            } else if (cmd == "stop") {
                handle_stop();
            } else if (cmd == "quit") {
                // Cancel watchdog before quitting to avoid a stray quit()
                // call after we've returned. Member dtor order (watchdog
                // first, then search_) makes this safe regardless, but
                // explicit is cheap.
                movetime_watchdog_ = std::jthread{};
                search_.quit();
                return;
            }
            // Unknown commands are silently ignored per UCI spec.
        }
        // EOF: ensure clean shutdown.
        movetime_watchdog_ = std::jthread{};
        search_.quit();
    }

private:
    static void emit_line(std::string_view s) {
        std::cout << s << '\n';
        std::cout.flush();
    }

    void handle_uci() {
        std::println("id name {}", ENGINE_NAME);
        std::println("id author {}", ENGINE_AUTHOR);
        std::println("uciok");
        std::fflush(stdout);
    }

    void handle_isready() {
        // Drain any in-flight search so we can honestly report ready.
        // quit() is idempotent and a fast no-op when nothing is running.
        movetime_watchdog_ = std::jthread{};
        search_.quit();
        std::println("readyok");
        std::fflush(stdout);
    }

    void handle_ucinewgame() {
        movetime_watchdog_ = std::jthread{};
        search_.quit();
        search_.reset();
        prev_fen_.clear();
        prev_moves_.clear();
    }

    void handle_position(const std::vector<std::string>& tokens) {
        movetime_watchdog_ = std::jthread{};
        search_.quit();
        if (tokens.size() < 2) return;

        // Parse out (fen, moves) from the tokens, canonicalizing
        // `startpos` to the literal STARTPOS_FEN so the prefix-equality
        // check below is meaningful.
        std::string fen;
        std::size_t i;
        if (tokens[1] == "startpos") {
            fen = std::string(STARTPOS_FEN);
            i = 2;
        } else if (tokens[1] == "fen") {
            std::string accum;
            i = 2;
            for (; i < tokens.size() && tokens[i] != "moves"; ++i) {
                if (!accum.empty()) accum += ' ';
                accum += tokens[i];
            }
            fen = std::move(accum);
        } else {
            return;
        }

        std::vector<std::string> moves;
        if (i < tokens.size() && tokens[i] == "moves") {
            for (std::size_t j = i + 1; j < tokens.size(); ++j) {
                moves.push_back(tokens[j]);
            }
        }

        // Prefix-based tree reuse: same FEN AND new moves extend the
        // previous move list (or equal it). In that case we only replay
        // the new tail via makemove(), which by contract preserves the
        // shared TT in the arena. The empty-tail case is a no-op.
        const bool prefix_match =
            fen == prev_fen_
            && moves.size() >= prev_moves_.size()
            && std::equal(prev_moves_.begin(), prev_moves_.end(),
                          moves.begin());

        if (prefix_match) {
            for (std::size_t k = prev_moves_.size(); k < moves.size(); ++k) {
                chess::Move m =
                    chess::uci::uciToMove(search_.board(), moves[k]);
                if (m == chess::Move::NO_MOVE) {
                    apply_full_reset(fen, moves);
                    return;
                }
                search_.makemove(m);
            }
            prev_fen_ = std::move(fen);
            prev_moves_ = std::move(moves);
        } else {
            apply_full_reset(fen, moves);
        }
    }

    // Reset path: rebuild the board from the FEN and replay the entire
    // move list. setBoard() drops the arena iff a search has happened
    // since the last reset.
    void apply_full_reset(const std::string& fen,
                          const std::vector<std::string>& moves) {
        search_.setBoard(chess::Board(fen));
        for (const auto& tok : moves) {
            chess::Move m = chess::uci::uciToMove(search_.board(), tok);
            if (m == chess::Move::NO_MOVE) break;
            search_.makemove(m);
        }
        prev_fen_ = fen;
        prev_moves_ = moves;
    }

    void handle_go(const std::vector<std::string>& tokens) {
        // Drain any prior search and cancel any prior movetime watchdog.
        movetime_watchdog_ = std::jthread{};
        search_.quit();

        // Defaults: effectively-infinite eval budget. Overridden by
        // `nodes X`. `infinite` is the same as the default. `movetime X`
        // leaves the eval budget huge and asks the watchdog to call
        // quit() after X ms.
        uint64_t max_evals = 1'000'000'000ULL;
        std::int64_t movetime_ms = -1;

        for (std::size_t i = 1; i < tokens.size(); ++i) {
            const auto& t = tokens[i];
            if (t == "infinite") {
                // already huge
            } else if (t == "nodes" && i + 1 < tokens.size()) {
                try {
                    max_evals = std::stoull(tokens[++i]);
                } catch (...) {}
            } else if (t == "movetime" && i + 1 < tokens.size()) {
                try {
                    movetime_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            }
            // Other tokens (depth/wtime/btime/winc/binc/movestogo/etc.)
            // are silently ignored.
        }

        catgpt::lks::LksSearchConfig cfg;
        cfg.max_evals = static_cast<int>(std::min<uint64_t>(
            max_evals,
            static_cast<uint64_t>(std::numeric_limits<int>::max())));
        cfg.min_info_period_ms = 100;
        cfg.on_uci_line = [](std::string_view s) { emit_line(s); };

        // search() returns immediately; worker_main runs on its own
        // jthread and emits info+bestmove via on_uci_line.
        search_.search(std::move(cfg));

        if (movetime_ms > 0) {
            movetime_watchdog_ = std::jthread(
                [this, ms = movetime_ms](std::stop_token st) {
                    // Sleep in small slices so cancellation is prompt.
                    auto deadline =
                        std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(ms);
                    while (!st.stop_requested()
                           && std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(10));
                    }
                    if (!st.stop_requested()) {
                        search_.quit();
                    }
                });
        }
    }

    void handle_stop() {
        movetime_watchdog_ = std::jthread{};
        search_.quit();
    }

    // Declaration order matters for destruction order: members are
    // destructed in reverse, so movetime_watchdog_ is joined first
    // (canceling any pending quit() callback), then search_ tears down
    // its workers/evaluators/CUDA buffers.
    catgpt::lks::LksSearch search_;
    std::string prev_fen_;
    std::vector<std::string> prev_moves_;
    std::jthread movetime_watchdog_;
};

}  // namespace catgpt

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    fs::path engine_path;
    if (argc > 1) {
        engine_path = argv[1];
    } else if (const char* env = std::getenv("CATGPT_TRT_ENGINE")) {
        engine_path = env;
    } else {
        engine_path = "/home/shadeform/CatGPT/main.trt";
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}",
                     engine_path.string());
        std::println(stderr, "Usage: {} [engine_path]", argv[0]);
        return 1;
    }

    const int num_workers      = catgpt::env_int("LKS_NUM_WORKERS", 2);
    const int coros_per_worker = catgpt::env_int("LKS_COROS_PER_WORKER", 8);
    const int max_batch_size   = catgpt::env_int("LKS_MAX_BATCH_SIZE", 32);
    const uint64_t lifetime_max_evals =
        catgpt::env_u64("LKS_LIFETIME_MAX_EVALS", 1ULL << 20);

    try {
        std::println(stderr, "Loading TensorRT engine: {}",
                     engine_path.string());
        std::println(
            stderr,
            "Config: workers={} coros_per_worker={} max_batch={} arena_capacity={}",
            num_workers, coros_per_worker, max_batch_size, lifetime_max_evals);

        catgpt::LksUciDriver driver(engine_path, lifetime_max_evals,
                                    num_workers, coros_per_worker,
                                    max_batch_size);
        std::println(stderr, "Engine loaded; entering UCI loop");
        driver.run();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
