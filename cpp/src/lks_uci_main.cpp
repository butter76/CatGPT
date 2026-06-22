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
 *                (default: $CATGPT_TRT_ENGINE or ./S4.network)
 *
 * Tuning env vars (UCI defaults tuned for headline GPU saturation;
 * may differ from LksSearch ctor defaults):
 *   LKS_WORKERS_PER_GPU     (default 2; total workers = this * #CUDA devices)
 *   LKS_COROS_PER_WORKER    (default 256)
 *   LKS_MAX_BATCH_SIZE      (default 112)
 *   LKS_LIFETIME_MAX_EVALS  (default 1<<27)
 *   LKS_DELTA_DEPTH         (default 0.2; per-iteration log-scale ID step
 *                            piped into LksSearchConfig::delta_depth on
 *                            every `go`)
 *   LKS_C_PUCT              (default 1.75; PUCT exploration constant piped
 *                            into LksSearchConfig::params.c_puct on every
 *                            `go`)
 *   LKS_WL_TEMP_WHITE       (default 0.5; WDL win-vs-loss sharpening temp
 *                            used when the root is White-to-move)
 *   LKS_WL_TEMP_BLACK       (default 0.5; WDL win-vs-loss sharpening temp
 *                            used when the root is Black-to-move; the
 *                            resolved value is piped into
 *                            LksSearchConfig::params.wl_temp at search
 *                            launch; 1.0 == plain P(W)-P(L))
 *   LKS_MAX_DEPTH           (default 32; log-scale ID ceiling on every
 *                            `go` → LksSearchConfig::max_depth)
 *
 *   Game-clock time management (see LksSearchConfig::TimeControl). All
 *   piped into cfg.time on every `go`; `go wtime/btime[/winc/binc]`
 *   derive a soft/hard budget from the side-to-move's clock, while
 *   `go movetime X` spends exactly X ms. Defaults mirror the legacy
 *   chessbench engine:
 *   LKS_TIME_RESERVE_MS       (default 100;  clock always held back)
 *   LKS_TIME_SOFT_PCT         (default 0.01; soft target = pct*bank + inc)
 *   LKS_TIME_HARD_PCT         (default 0.50; hard cap    = pct*bank + inc)
 *   LKS_TIME_FIRST_MOVE_PCT   (default 0.08; soft floor on first move)
 *   LKS_TIME_SURPRISE_PCT     (default 0.04; soft floor on a surprise)
 *   LKS_TIME_CHANGE_BONUS_PCT (default 0.02; soft +=pct*bank if best changes)
 *   LKS_TIME_WORSEN_BONUS_PCT (default 0.02; soft +=pct*bank if score drops)
 *   LKS_TIME_WORSEN_CP        (default 10;   cp drop that counts as worsened)
 *   LKS_TIME_EARLY_RETURN_MARGIN (default 0.3; log-depth margin for the
 *                            TT early-return chain when the opponent plays
 *                            the predicted reply)
 *
 *   LKS_SYZYGY_PATH         (default $SYZYGY_HOME, else "" = disabled)
 *                           Directory of .rtbw/.rtbz Syzygy endgame
 *                           tablebase files. When set, eligible root
 *                           positions are resolved via DTZ probe
 *                           instead of running the GPU search.
 */

#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <print>
#include <set>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../external/chess-library/include/chess.hpp"
#include "catgpt_version.hpp"
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

static float env_float(const char* name, float fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    try { return std::stof(s); } catch (...) { return fallback; }
}

// Estimate the resident footprint of the shared SearchArena (TT + node
// arena) from LKS_LIFETIME_MAX_EVALS and warn if it approaches the TCEC
// per-engine RAM ceiling. Mirrors SearchArena::compute_capacity /
// compute_arena_bytes (load_factor 0.6, avg_moves_per_node 40) and the
// fixed type sizes (sizeof(TTEntry)==32, NodeInfoHeader 12, MoveInfo 4).
// The bit_ceil rounding of the TT can nearly double its size near a
// power-of-two boundary, so it is reported explicitly. TRT weights and
// pinned host buffers are excluded (a few GB total, per GPU).
static void log_arena_footprint(uint64_t k_max_evals) {
    constexpr uint64_t kTtEntryBytes   = 32;
    constexpr uint64_t kPerNodeBytes   = 12 + 40 * 4;  // header + 40 MoveInfo
    // load_factor 0.6 -> target capacity = K / 0.6, then next pow2.
    const double   target = static_cast<double>(k_max_evals) / 0.6;
    const uint64_t want   = target < 16.0
                              ? 16ULL
                              : static_cast<uint64_t>(target) + 1ULL;
    const uint64_t tt_cap = std::bit_ceil(want);
    const uint64_t tt_bytes    = tt_cap * kTtEntryBytes;
    const uint64_t arena_bytes = k_max_evals * kPerNodeBytes;
    const uint64_t total       = tt_bytes + arena_bytes;

    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    // TCEC: max ~256 GiB/engine with multi-threaded init; keep headroom for
    // TRT weights, pinned buffers, and the OS.
    constexpr double kCeilGiB = 240.0;
    const double total_gib = static_cast<double>(total) / kGiB;

    std::println(stderr,
                 "Arena footprint: TT {:.1f} GiB (capacity {} entries) + node arena {:.1f} GiB = {:.1f} GiB (TRT weights + pinned buffers excluded)",
                 static_cast<double>(tt_bytes) / kGiB, tt_cap,
                 static_cast<double>(arena_bytes) / kGiB, total_gib);
    if (total_gib > kCeilGiB) {
        std::println(stderr,
                     "WARNING: estimated arena footprint {:.1f} GiB exceeds the ~{:.0f} GiB/engine budget; lower LKS_LIFETIME_MAX_EVALS (note: TT rounds up to a power of two, so a small reduction can halve it)",
                     total_gib, kCeilGiB);
    }
}

class LksUciDriver {
public:
    LksUciDriver(fs::path engine_path,
                 uint64_t lifetime_max_evals,
                 int workers_per_gpu,
                 int coros_per_worker,
                 int max_batch_size,
                 fs::path syzygy_path,
                 float delta_depth,
                 float c_puct,
                 float wl_temp_white,
                 float wl_temp_black,
                 float max_depth,
                 catgpt::lks::TimeControl time_tunables)
        : search_(std::move(engine_path), lifetime_max_evals,
                  workers_per_gpu, coros_per_worker, max_batch_size,
                  std::move(syzygy_path))
        , delta_depth_(delta_depth)
        , c_puct_(c_puct)
        , wl_temp_white_(wl_temp_white)
        , wl_temp_black_(wl_temp_black)
        , max_depth_(max_depth)
        , time_tunables_(time_tunables)
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
                search_.quit();
                return;
            }
            // Unknown commands are silently ignored per UCI spec.
        }
        // EOF: ensure clean shutdown.
        search_.quit();
    }

private:
    static void emit_line(std::string_view s) {
        std::cout << s << '\n';
        std::cout.flush();
    }

    // Reply straight from the TT for an early-return move (no search ran).
    // Emits an `info` line (depth/score/pv, mirroring worker_main's fields)
    // and the chosen `bestmove`.
    void emit_early_return() {
        const chess::Move best = search_.bestmove();
        const std::vector<chess::Move> pv = search_.principal_variation();

        std::string info = "info";
        if (const auto d = search_.root_depth()) {
            info += " depth ";
            info += std::to_string(std::lround(*d * 100.0f));
        }
        if (const auto cp = search_.root_cp()) {
            info += " score cp ";
            info += std::to_string(*cp);
        }
        if (!pv.empty()) {
            info += " pv";
            for (const chess::Move& m : pv) {
                info += ' ';
                info += chess::uci::moveToUci(m);
            }
        }
        emit_line(info);
        emit_line("info string early return");

        std::string bm = "bestmove ";
        bm += (best != chess::Move::NO_MOVE)
            ? chess::uci::moveToUci(best)
            : std::string{"0000"};
        emit_line(bm);
    }

    void handle_uci() {
        std::println("id name {} {}", ENGINE_NAME, catgpt::version::VERSION_STRING);
        std::println("id author {}", ENGINE_AUTHOR);
        std::println("info string {} {} on branch {}",
                     ENGINE_NAME,
                     catgpt::version::VERSION_STRING,
                     catgpt::version::GIT_BRANCH);
        std::println("uciok");
        std::fflush(stdout);
    }

    void handle_isready() {
        // Drain any in-flight search so we can honestly report ready.
        // quit() is idempotent and a fast no-op when nothing is running.
        search_.quit();
        std::println("readyok");
        std::fflush(stdout);
    }

    void handle_ucinewgame() {
        search_.quit();
        search_.reset();
        prev_fen_.clear();
        prev_moves_.clear();
        // Fresh game: next search is the first move; no prediction yet.
        first_move_ = true;
        have_searched_ = false;
        pending_surprise_ = false;
        predicted_reply_uci_.clear();
        reply_ply_ = kNoReplyPly;
        // Early-return chain state.
        chain_armed_ = false;
        anchor_root_depth_ = 0.0f;
        last_was_real_search_ = false;
    }

    void handle_position(const std::vector<std::string>& tokens) {
        search_.quit();

        // Capture our prediction of the opponent's reply from the just-
        // finished search BEFORE the board is mutated below. The previous
        // search ran on `prev_moves_`, so our best move lands at ply
        // prev_moves_.size() and the predicted reply at the next ply. The
        // PV walks from the still-loaded previous board: pv[0] is our best,
        // pv[1] the predicted reply.
        if (have_searched_) {
            const std::vector<chess::Move> pv = search_.principal_variation(2);
            predicted_reply_uci_ =
                pv.size() > 1 ? chess::uci::moveToUci(pv[1]) : std::string{};
            reply_ply_ = prev_moves_.size() + 1;
        }

        // Consume the just-finished search's outcome for the early-return
        // chain. The board is still at the previous root here, so
        // `root_depth()` reports that root's depth — the anchor. Only a real
        // worker_main run updates this; a chained early return leaves the
        // anchor (and `last_completion`) untouched so the chain stays bounded
        // to the first root's depth.
        if (have_searched_ && last_was_real_search_) {
            switch (search_.last_completion()) {
                case catgpt::lks::LksSearch::Completion::kSoftCompleted:
                    if (const auto d = search_.root_depth()) {
                        anchor_root_depth_ = *d;
                        chain_armed_ = true;
                    } else {
                        chain_armed_ = false;
                    }
                    break;
                case catgpt::lks::LksSearch::Completion::kHardAborted:
                    // A hard abort counts as a surprise for the next move.
                    pending_surprise_ = true;
                    chain_armed_ = false;
                    break;
                case catgpt::lks::LksSearch::Completion::kOther:
                    chain_armed_ = false;
                    break;
            }
        }

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

        // Surprise detection: when the game continues along the same line
        // and the opponent's actual reply (the move at reply_ply_) differs
        // from what we predicted, flag the next search for a soft-time
        // boost. Only meaningful on a prefix extension so the ply indices
        // line up.
        if (prefix_match && !predicted_reply_uci_.empty()
            && reply_ply_ != kNoReplyPly && moves.size() > reply_ply_) {
            if (moves[reply_ply_] != predicted_reply_uci_) {
                pending_surprise_ = true;
            }
        }

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
        // Drain any prior search.
        search_.quit();

        // Defaults: effectively-infinite eval budget. Overridden by
        // `nodes X`. `infinite` is the same as the default. `depth N` is
        // interpreted as centi-depth (N=800 -> stop when min_depth() >=
        // 8.00), matching the encoding used by `info depth ...` in
        // worker_main. Wall-clock control (`movetime`, `wtime`/`btime`
        // [+inc]) is handled entirely inside worker_main via cfg.time
        // (see LksSearchConfig::TimeControl); the tunable constants come
        // from the LKS_TIME_* env vars captured in time_tunables_.
        catgpt::lks::LksSearchConfig cfg;
        cfg.max_evals = 1'000'000'000ULL;
        cfg.delta_depth = delta_depth_;
        cfg.max_depth = max_depth_;
        cfg.params.c_puct = c_puct_;
        cfg.params.wl_temp_white = wl_temp_white_;
        cfg.params.wl_temp_black = wl_temp_black_;
        cfg.on_uci_line = [](std::string_view s) { emit_line(s); };

        cfg.time = time_tunables_;
        cfg.time.first_move = first_move_;
        cfg.time.surprise   = pending_surprise_;

        for (std::size_t i = 1; i < tokens.size(); ++i) {
            const auto& t = tokens[i];
            if (t == "infinite") {
                // already huge
            } else if (t == "nodes" && i + 1 < tokens.size()) {
                try {
                    cfg.max_evals = std::stoull(tokens[++i]);
                } catch (...) {}
            } else if (t == "movetime" && i + 1 < tokens.size()) {
                try {
                    cfg.time.movetime_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            } else if (t == "wtime" && i + 1 < tokens.size()) {
                try {
                    cfg.time.wtime_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            } else if (t == "btime" && i + 1 < tokens.size()) {
                try {
                    cfg.time.btime_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            } else if (t == "winc" && i + 1 < tokens.size()) {
                try {
                    cfg.time.winc_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            } else if (t == "binc" && i + 1 < tokens.size()) {
                try {
                    cfg.time.binc_ms = std::stoll(tokens[++i]);
                } catch (...) {}
            } else if (t == "depth" && i + 1 < tokens.size()) {
                try {
                    const long d_centi = std::stol(tokens[++i]);
                    if (d_centi > 0) {
                        cfg.target_min_depth =
                            static_cast<float>(d_centi) / 100.0f;
                    }
                } catch (...) {}
            }
            // movestogo and other tokens are intentionally ignored.
        }

        // ── Early-return chain ("ponderhit-like") ────────────────────────
        // When the previous (real) search soft-completed and the opponent
        // then played our predicted reply, the reused TT subtree at the new
        // root may already be searched almost as deeply as the anchored
        // root. If so, answer straight from the TT without launching a
        // search. Restricted to genuine game-clock searches (a soft phase
        // must exist); the anchor stays fixed so the chain stays bounded to
        // the first root's depth.
        const bool game_clock_go =
            cfg.time.active() && cfg.time.movetime_ms <= 0;
        if (chain_armed_ && !first_move_ && !pending_surprise_
            && game_clock_go) {
            if (const auto d = search_.root_depth();
                d && *d >= anchor_root_depth_ - cfg.time.early_return_margin) {
                emit_early_return();
                // Anchor + chain_armed_ left unchanged so chaining continues;
                // only the predicted reply (recomputed by the next
                // handle_position) advances.
                pending_surprise_ = false;
                first_move_ = false;
                have_searched_ = true;
                last_was_real_search_ = false;
                return;
            }
            // New root too shallow: fall through to a full search.
            chain_armed_ = false;
        }

        // The surprise boost is one-shot; consume it. After this search
        // launches the game is no longer on its first move.
        pending_surprise_ = false;
        first_move_ = false;
        have_searched_ = true;
        last_was_real_search_ = true;

        // search() returns immediately; worker_main runs on its own
        // jthread and emits info+bestmove via on_uci_line.
        search_.search(std::move(cfg));
    }

    void handle_stop() {
        search_.quit();
    }

    // Declaration order matters for destruction order: members are
    // destructed in reverse, so search_ (which joins its worker thread in
    // its destructor) tears down last.
    catgpt::lks::LksSearch search_;
    float delta_depth_;
    float c_puct_;
    float wl_temp_white_;
    float wl_temp_black_;
    float max_depth_;
    catgpt::lks::TimeControl time_tunables_;
    std::string prev_fen_;
    std::vector<std::string> prev_moves_;

    // ── Cross-move time-management state ──
    static constexpr std::size_t kNoReplyPly =
        std::numeric_limits<std::size_t>::max();
    bool        first_move_       = true;   // next search is the game's first
    bool        have_searched_    = false;  // a search has produced a PV
    bool        pending_surprise_ = false;  // opponent's reply was unpredicted
    std::string predicted_reply_uci_;       // our predicted opponent reply
    std::size_t reply_ply_        = kNoReplyPly;  // ply index of that reply

    // ── Early-return ("ponderhit-like") chain state ──
    // After a soft-completed real search we anchor the root's TT depth and
    // arm the chain. On subsequent moves, when the opponent plays our
    // predicted reply and the reused subtree's depth is within
    // `early_return_margin` of the anchor, we answer straight from the TT
    // without searching. The anchor is held fixed across chained replies so
    // the chain self-terminates once the depth drifts too far below it.
    bool  chain_armed_          = false;  // early return currently possible
    float anchor_root_depth_    = 0.0f;   // anchored root TT depth (valid iff armed)
    bool  last_was_real_search_ = false;  // last move ran worker_main (vs early return)
};

}  // namespace catgpt

// Prints CLI usage. Every flag also has an LKS_* env equivalent; the CLI
// flag wins when both are set, and both override the built-in default.
static void print_usage(const char* prog) {
    std::println("{} {}", catgpt::ENGINE_NAME, catgpt::version::VERSION_STRING);
    std::println(
        "\n"
        "Usage: {} [network] [options]\n"
        "\n"
        "Positional:\n"
        "  network                      TensorRT .network bundle\n"
        "                               (else $CATGPT_TRT_ENGINE, else ./S4.network)\n"
        "\n"
        "Options (CLI > matching LKS_* env var > default):\n"
        "  --network PATH               network bundle path\n"
        "  --syzygy-path PATH           Syzygy tablebase dir  [LKS_SYZYGY_PATH / SYZYGY_HOME]\n"
        "  --workers-per-gpu N          search workers per GPU   [LKS_WORKERS_PER_GPU=2]\n"
        "  --coros-per-worker N         coroutines per worker    [LKS_COROS_PER_WORKER=256]\n"
        "  --max-batch N                max eval batch size      [LKS_MAX_BATCH_SIZE=112]\n"
        "  --max-evals N                lifetime eval/arena cap  [LKS_LIFETIME_MAX_EVALS]\n"
        "  --delta-depth F              [LKS_DELTA_DEPTH=0.2]\n"
        "  --c-puct F                   [LKS_C_PUCT=1.75]\n"
        "  --wl-temp-white F            [LKS_WL_TEMP_WHITE=0.5]\n"
        "  --wl-temp-black F            [LKS_WL_TEMP_BLACK=0.5]\n"
        "  --max-depth F                [LKS_MAX_DEPTH=32]\n"
        "  --time-reserve-ms N          [LKS_TIME_RESERVE_MS]\n"
        "  --time-soft-pct F            [LKS_TIME_SOFT_PCT]\n"
        "  --time-hard-pct F            [LKS_TIME_HARD_PCT]\n"
        "  --time-first-move-pct F      [LKS_TIME_FIRST_MOVE_PCT]\n"
        "  --time-surprise-pct F        [LKS_TIME_SURPRISE_PCT]\n"
        "  --time-change-bonus-pct F    [LKS_TIME_CHANGE_BONUS_PCT]\n"
        "  --time-worsen-bonus-pct F    [LKS_TIME_WORSEN_BONUS_PCT]\n"
        "  --time-worsen-cp N           [LKS_TIME_WORSEN_CP]\n"
        "  --time-early-return-margin F [LKS_TIME_EARLY_RETURN_MARGIN]\n"
        "  -v, --version                print version and exit\n"
        "  -h, --help                   print this help and exit",
        prog);
}

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // ── CLI parsing ──────────────────────────────────────────────────
    // Precedence is CLI flag > LKS_* env var > built-in default. Accepts
    // "--flag value", "--flag=value", and a single positional network path.
    std::unordered_map<std::string, std::string> cli;
    std::optional<std::string> positional;
    static const std::set<std::string_view> kBoolFlags =
        {"--version", "-v", "--help", "-h"};
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a.starts_with("--") || a == "-v" || a == "-h") {
            if (auto eq = a.find('='); eq != std::string_view::npos) {
                cli[std::string(a.substr(0, eq))] = std::string(a.substr(eq + 1));
            } else if (!kBoolFlags.contains(a) && i + 1 < argc
                       && !std::string_view(argv[i + 1]).starts_with("-")) {
                cli[std::string(a)] = argv[++i];
            } else {
                cli[std::string(a)] = "true";
            }
        } else if (!positional) {
            positional = std::string(a);
        }
    }

    if (cli.contains("--version") || cli.contains("-v")) {
        std::println("{} {} on branch {}",
                     catgpt::ENGINE_NAME,
                     catgpt::version::VERSION_STRING,
                     catgpt::version::GIT_BRANCH);
        return 0;
    }
    if (cli.contains("--help") || cli.contains("-h")) {
        print_usage(argv[0]);
        return 0;
    }

    // CLI>env>default resolvers. `cli_str` exposes a raw flag value;
    // numeric resolvers parse strictly and abort loudly on bad input
    // (explicit operator intent), unlike the tolerant env_* fallbacks.
    auto cli_str = [&](const char* flag) -> std::optional<std::string> {
        if (auto it = cli.find(flag); it != cli.end()) return it->second;
        return std::nullopt;
    };
    auto cli_int = [&](const char* flag, const char* env, int def) -> int {
        if (auto v = cli_str(flag)) {
            try { return std::stoi(*v); }
            catch (...) { std::println(stderr, "Error: bad integer for {}: {}", flag, *v); std::exit(2); }
        }
        return catgpt::env_int(env, def);
    };
    auto cli_u64 = [&](const char* flag, const char* env, uint64_t def) -> uint64_t {
        if (auto v = cli_str(flag)) {
            try { return std::stoull(*v); }
            catch (...) { std::println(stderr, "Error: bad integer for {}: {}", flag, *v); std::exit(2); }
        }
        return catgpt::env_u64(env, def);
    };
    auto cli_float = [&](const char* flag, const char* env, float def) -> float {
        if (auto v = cli_str(flag)) {
            try { return std::stof(*v); }
            catch (...) { std::println(stderr, "Error: bad number for {}: {}", flag, *v); std::exit(2); }
        }
        return catgpt::env_float(env, def);
    };

    std::println(stderr, "{} {} on branch {}",
                 catgpt::ENGINE_NAME,
                 catgpt::version::VERSION_STRING,
                 catgpt::version::GIT_BRANCH);

    // Engine path: positional > --network > $CATGPT_TRT_ENGINE > default.
    fs::path engine_path;
    if (positional) {
        engine_path = *positional;
    } else if (auto v = cli_str("--network")) {
        engine_path = *v;
    } else if (const char* env = std::getenv("CATGPT_TRT_ENGINE")) {
        engine_path = env;
    } else {
        engine_path = "./S4.network";
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}",
                     engine_path.string());
        std::println(stderr, "Run '{} --help' for usage.", argv[0]);
        return 1;
    }

    const int workers_per_gpu  = cli_int("--workers-per-gpu", "LKS_WORKERS_PER_GPU", 2);
    const int coros_per_worker = cli_int("--coros-per-worker", "LKS_COROS_PER_WORKER", 256);
    const int max_batch_size   = cli_int("--max-batch", "LKS_MAX_BATCH_SIZE", 112);
    const uint64_t lifetime_max_evals =
        cli_u64("--max-evals", "LKS_LIFETIME_MAX_EVALS", 1ULL << 27);
    const float delta_depth = cli_float("--delta-depth", "LKS_DELTA_DEPTH", 0.2f);
    const float c_puct      = cli_float("--c-puct", "LKS_C_PUCT", 1.75f);
    const float wl_temp_white = cli_float("--wl-temp-white", "LKS_WL_TEMP_WHITE", 0.5f);
    const float wl_temp_black = cli_float("--wl-temp-black", "LKS_WL_TEMP_BLACK", 0.5f);
    const float max_depth   = cli_float("--max-depth", "LKS_MAX_DEPTH", 32.0f);

    // Game-clock time-management tunables (see LksSearchConfig::TimeControl).
    // Only the constant knobs are loaded here; per-go clock inputs and the
    // first-move/surprise flags are filled in by the driver at search time.
    catgpt::lks::TimeControl time_tunables;
    time_tunables.reserve_ms =
        cli_int("--time-reserve-ms", "LKS_TIME_RESERVE_MS",
                static_cast<int>(time_tunables.reserve_ms));
    time_tunables.soft_pct =
        cli_float("--time-soft-pct", "LKS_TIME_SOFT_PCT", time_tunables.soft_pct);
    time_tunables.hard_pct =
        cli_float("--time-hard-pct", "LKS_TIME_HARD_PCT", time_tunables.hard_pct);
    time_tunables.first_move_pct =
        cli_float("--time-first-move-pct", "LKS_TIME_FIRST_MOVE_PCT", time_tunables.first_move_pct);
    time_tunables.surprise_pct =
        cli_float("--time-surprise-pct", "LKS_TIME_SURPRISE_PCT", time_tunables.surprise_pct);
    time_tunables.change_bonus_pct =
        cli_float("--time-change-bonus-pct", "LKS_TIME_CHANGE_BONUS_PCT", time_tunables.change_bonus_pct);
    time_tunables.worsen_bonus_pct =
        cli_float("--time-worsen-bonus-pct", "LKS_TIME_WORSEN_BONUS_PCT", time_tunables.worsen_bonus_pct);
    time_tunables.worsen_threshold_cp =
        cli_int("--time-worsen-cp", "LKS_TIME_WORSEN_CP", time_tunables.worsen_threshold_cp);
    time_tunables.early_return_margin =
        cli_float("--time-early-return-margin", "LKS_TIME_EARLY_RETURN_MARGIN",
                  time_tunables.early_return_margin);

    // Syzygy path: --syzygy-path > $LKS_SYZYGY_PATH > $SYZYGY_HOME; empty = disabled.
    fs::path syzygy_path;
    if (auto v = cli_str("--syzygy-path"); v && !v->empty()) {
        syzygy_path = *v;
    } else if (const char* p = std::getenv("LKS_SYZYGY_PATH"); p && *p) {
        syzygy_path = p;
    } else if (const char* p = std::getenv("SYZYGY_HOME"); p && *p) {
        syzygy_path = p;
    }

    try {
        std::println(stderr, "Loading TensorRT engine: {}",
                     engine_path.string());
        std::println(
            stderr,
            "Config: workers_per_gpu={} coros_per_worker={} max_batch={} arena_capacity={} delta_depth={} max_depth={} c_puct={} wl_temp_white={} wl_temp_black={} syzygy_path={}",
            workers_per_gpu, coros_per_worker, max_batch_size, lifetime_max_evals,
            delta_depth, max_depth, c_puct, wl_temp_white, wl_temp_black,
            syzygy_path.empty() ? std::string{"<disabled>"} : syzygy_path.string());
        catgpt::log_arena_footprint(lifetime_max_evals);
        std::println(
            stderr,
            "Time: reserve_ms={} soft_pct={} hard_pct={} first_move_pct={} surprise_pct={} change_bonus_pct={} worsen_bonus_pct={} worsen_cp={} early_return_margin={}",
            time_tunables.reserve_ms, time_tunables.soft_pct, time_tunables.hard_pct,
            time_tunables.first_move_pct, time_tunables.surprise_pct,
            time_tunables.change_bonus_pct, time_tunables.worsen_bonus_pct,
            time_tunables.worsen_threshold_cp, time_tunables.early_return_margin);

        catgpt::LksUciDriver driver(engine_path, lifetime_max_evals,
                                    workers_per_gpu, coros_per_worker,
                                    max_batch_size, std::move(syzygy_path),
                                    delta_depth, c_puct, wl_temp_white,
                                    wl_temp_black, max_depth,
                                    time_tunables);
        std::println(stderr, "Engine loaded; entering UCI loop");
        driver.run();
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
