/**
 * CatGPT Search Binary (LKS-backed) — Standalone search for the web UI.
 *
 * Wraps `catgpt::lks::LksSearch` (Lazy-SMP, log-scale ID, TRT-batched,
 * lock-free TT) and emits a JSON-per-line stats stream that the web
 * backend parses. Replaces the older libcoro `CoroutineSearch` binary.
 *
 * Output format (one JSON object per line):
 *   {"type":"root_eval",      "bestMove":"e2e4", "cp":15, "nodes":1,   "iteration":0,  "policy":[...]}
 *   {"type":"search_update",  "bestMove":"d2d4", "cp":20, "nodes":50,  "iteration":40, "policy":[...], "pv":[...]}
 *   {"type":"search_complete","bestMove":"d2d4", "cp":22, "nodes":400, "iteration":80, "policy":[...], "pv":[...]}
 *   bestmove d2d4
 *
 * Fields:
 *   bestMove   UCI move chosen by LksSearch::bestmove().
 *   cp         Centipawn eval of the root, derived from the TT's root Q
 *              via q_to_cp.
 *   nodes      Cumulative NN evals across all workers (LksSearch::total_evals).
 *   iteration  Centi-depth (round(max_depth() * 100)) — there is no global
 *              iteration counter in LKS; this is the closest analog and
 *              keeps the field roughly comparable to the old protocol.
 *   policy     One entry per legal move at the root:
 *                {"move": <uci>, "weight": <softmax(log_alloc)>, "q"?: <parent-POV Q>}
 *              Weights are softmax-normalized exp(plan.alloc) from the
 *              Halley allocator at the current root depth. `q` is omitted
 *              for never-evaluated children (its FPU stand-in would
 *              mislead the chart).
 *   pv         Greedy best-child walk from the root using the same
 *              scoring rule as LksSearch::bestmove (-child_Q, tiebreak
 *              child max_depth, tiebreak prior). Capped at 100 plies and
 *              cut short at terminals, repetitions, 50-move rule, or
 *              TT misses (see LksSearch::principal_variation).
 *
 * Note: there is intentionally no `distQ` field — `v2::TTEntry` only
 * stores scalar (Q, max_depth), and the per-eval `value_probs` from the
 * BatchEvaluator are not persisted anywhere in the search arena.
 *
 * Usage:
 *   catgpt_search <engine_path> <fen> [nodes]
 *
 * Arguments:
 *   engine_path  Path to TensorRT engine file (.network or .trt)
 *   fen          FEN string (quoted)
 *   nodes        Optional: max GPU evaluations (default: 400)
 *
 * Tuning env vars (mirror lks_uci_main, except workers_per_gpu stays at
 * 1 so a single web request doesn't saturate a multi-GPU host):
 *   LKS_WORKERS_PER_GPU     (default 1; single Lazy-SMP worker per GPU)
 *   LKS_COROS_PER_WORKER    (default 112)
 *   LKS_MAX_BATCH_SIZE      (default 56)
 *   LKS_LIFETIME_MAX_EVALS  (default 1<<27)
 *   LKS_SYZYGY_PATH         (default $SYZYGY_HOME, else "" = disabled)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../external/chess-library/include/chess.hpp"
#include "engine/fractional_mcts/search_stats.hpp"  // q_to_cp
#include "engine/fractional_mcts/v2/board_secondary.hpp"
#include "engine/fractional_mcts/v2/tt_arena.hpp"
#include "engine/lks/compute_allocations.hpp"
#include "engine/lks/lks_search.hpp"

namespace fs = std::filesystem;

namespace {

using catgpt::lks::LksSearch;
using catgpt::lks::LksSearchConfig;
using catgpt::lks::detail::Mode;
using catgpt::lks::detail::Plan;
using catgpt::lks::detail::compute_log_allocations;

constexpr float kPosInf = std::numeric_limits<float>::infinity();

int env_int(const char* name, int fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    try { return std::stoi(s); } catch (...) { return fallback; }
}

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* s = std::getenv(name);
    if (!s || !*s) return fallback;
    try { return std::stoull(s); } catch (...) { return fallback; }
}

// ── Snapshot types ─────────────────────────────────────────────────

/**
 * Per-child classification result, mirroring the cases handled by
 * `recursive_search` Pass 1 in lks_search.hpp. `expanded` distinguishes
 * "we have a real Q to report" from "child has never been touched"
 * (TT miss, no terminal shortcut). Q is stored in child-STM convention,
 * same as the TT.
 *
 * Used by `snapshot_root` to build per-child Plans for the policy
 * weight chart; the PV walk now goes through
 * `LksSearch::principal_variation` directly.
 */
struct ChildClass {
    bool  expanded;     // false ⇒ TT miss with no terminal shortcut
    float Q_child;      // child-STM Q; ignored when !expanded
    float child_depth;  // child's max_depth; ignored when !expanded
    bool  is_terminal;  // for PV walk: never descend past this
};

/**
 * Classify a single child of `parent` exactly the way `recursive_search`
 * Pass 1 does. Returns `expanded=false` for plain TT misses; FPU Q is
 * NOT synthesized here — that's the snapshotter's job.
 */
ChildClass classify_child(const chess::Board& parent,
                          chess::Move m,
                          catgpt::v2::TerminalKind tk,
                          const catgpt::v2::SearchArena& arena)
{
    using namespace catgpt::v2;
    if (tk == kTerminalDraw) {
        return {true, 0.0f, kPosInf, true};
    }
    if (tk == kTerminalLossForChild) {
        return {true, -1.0f, kPosInf, true};
    }
    if (tk == kTerminalWinForChild) {
        return {true, +1.0f, kPosInf, true};
    }
    chess::Board cb = parent;
    cb.makeMove<true>(m);
    // Plain 2-fold repetition draw on the child's path history.
    if (cb.isRepetition(1) || cb.isHalfMoveDraw()) {
        return {true, 0.0f, kPosInf, true};
    }
    if (const TTEntry* ce = arena.find(cb.hash(), secondary_hash(cb))) {
        auto [q, d] = unpack_qd(SearchArena::load_qd(ce).qd_packed);
        return {true, q, d, false};
    }
    return {false, 0.0f, -kPosInf, false};
}

struct PolicyEntry {
    chess::Move move;
    float       weight;  // softmax-normalized over all legal moves
    bool        has_q;
    float       q;       // parent-POV (positive == good for STM-at-root)
};

struct RootSnapshot {
    std::vector<PolicyEntry> entries;
    std::vector<chess::Move> pv;
    chess::Move best_move = chess::Move::NO_MOVE;
    int         cp = 0;
    uint64_t    total_evals = 0;
    int         depth_centi = 0;
};

/**
 * Build a `RootSnapshot` from the live TT. Returns false if the root
 * has not yet been evaluated (no TT entry, or entry's NodeInfo not
 * published). Safe to call concurrently with a search in flight — all
 * arena reads go through `find` / `load_*` which provide the necessary
 * acquire semantics.
 *
 * `c_puct` and `fpu_reduction` should match the values supplied in
 * LksSearchConfig.params so the displayed allocations track what the
 * descent is actually doing.
 */
bool snapshot_root(const LksSearch& s,
                   float c_puct,
                   float fpu_reduction,
                   RootSnapshot& out)
{
    using namespace catgpt::v2;
    const TTEntry* root = s.arena().find(s.root_key(), secondary_hash(s.board()));
    if (!root) return false;
    const InfoCell info = SearchArena::load_info(root);
    if (info.info_offset == kNoInfoOffset) return false;

    const NodeInfoHeader* hdr = s.arena().info_at(info.info_offset);
    const uint16_t num = hdr->num_moves;
    const MoveInfo* mi = s.arena().moves_at(info.info_offset);

    auto [root_q, root_d] = unpack_qd(SearchArena::load_qd(root).qd_packed);
    (void)root_d;

    out.cp          = catgpt::q_to_cp(root_q);
    out.total_evals = s.total_evals();
    {
        const float md = s.max_depth();
        out.depth_centi = std::isfinite(md)
            ? static_cast<int>(std::lround(md * 100.0f))
            : 0;
    }

    // Build per-child plans for the Halley allocator, mirroring
    // recursive_search Pass 1 exactly (including FPU for unexpanded
    // children, which we feed into the allocator but DO NOT surface
    // as `q` on the JSON entry).
    std::vector<Plan>        plans;
    std::vector<ChildClass>  klass;
    plans.reserve(num);
    klass.reserve(num);

    const chess::Board parent = s.board();
    float cumulative_P = 0.0f;
    for (uint16_t i = 0; i < num; ++i) {
        const float m_P = mi[i].P();
        ChildClass c = classify_child(parent, chess::Move{mi[i].move},
                                      mi[i].terminal_kind(), s.arena());
        if (c.expanded) {
            plans.push_back({Mode::Expanded, m_P, c.Q_child, c.child_depth, 0.0f});
        } else {
            // FPU stand-in, stored as -Q_eff_parent_pov so rollup's -Q
            // recovers Q_eff_parent_pov. Same formula as the descent.
            const float Q_eff_parent_pov =
                root_q - fpu_reduction * std::sqrt(cumulative_P);
            plans.push_back({Mode::Unexpanded, m_P,
                             /*Q=*/-Q_eff_parent_pov,
                             /*depth=*/-kPosInf,
                             /*alloc=*/0.0f});
        }
        klass.push_back(c);
        cumulative_P += m_P;
    }

    // Halley allocator needs a positive depth; clamp very-early values
    // away from zero so the dual solve doesn't degenerate.
    const float md = s.max_depth();
    const float depth_for_alloc =
        (std::isfinite(md) && md > 0.1f) ? md : 0.1f;
    compute_log_allocations(plans.data(), static_cast<int>(plans.size()),
                            depth_for_alloc, c_puct);

    // Softmax-normalize the log allocations into [0, 1] weights. Drop
    // -inf plans (e.g. P == 0) from the sum but keep them with weight 0.
    float max_log = -kPosInf;
    for (const Plan& p : plans) {
        if (std::isfinite(p.alloc) && p.alloc > max_log) max_log = p.alloc;
    }

    std::vector<double> w(plans.size(), 0.0);
    double sum = 0.0;
    if (std::isfinite(max_log)) {
        for (size_t i = 0; i < plans.size(); ++i) {
            if (!std::isfinite(plans[i].alloc)) continue;
            w[i] = std::exp(static_cast<double>(plans[i].alloc - max_log));
            sum += w[i];
        }
    }

    out.entries.clear();
    out.entries.reserve(num);
    for (uint16_t i = 0; i < num; ++i) {
        PolicyEntry e;
        e.move   = chess::Move{mi[i].move};
        e.weight = sum > 0.0 ? static_cast<float>(w[i] / sum) : mi[i].P();
        e.has_q  = klass[i].expanded;
        e.q      = -klass[i].Q_child;  // child-STM Q -> parent-STM
        out.entries.push_back(e);
    }

    out.best_move = s.bestmove();
    out.pv = s.principal_variation(/*max_len=*/256);
    return true;
}

void emit_json(std::string_view type, const RootSnapshot& snap) {
    std::ostringstream json;
    json.precision(6);

    json << "{\"type\":\"" << type << "\"";
    json << ",\"bestMove\":\""
         << chess::uci::moveToUci(snap.best_move) << "\"";
    json << ",\"cp\":" << snap.cp;
    json << ",\"nodes\":" << snap.total_evals;
    json << ",\"iteration\":" << snap.depth_centi;

    json << ",\"policy\":[";
    for (size_t i = 0; i < snap.entries.size(); ++i) {
        if (i > 0) json << ",";
        const auto& e = snap.entries[i];
        json << "{\"move\":\"" << chess::uci::moveToUci(e.move)
             << "\",\"weight\":" << e.weight;
        if (e.has_q) json << ",\"q\":" << e.q;
        json << "}";
    }
    json << "]";

    if (!snap.pv.empty()) {
        json << ",\"pv\":[";
        for (size_t i = 0; i < snap.pv.size(); ++i) {
            if (i > 0) json << ",";
            json << "\"" << chess::uci::moveToUci(snap.pv[i]) << "\"";
        }
        json << "]";
    }

    json << "}";
    std::cout << json.str() << "\n";
    std::cout.flush();
}

}  // namespace

int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 3) {
        std::println(stderr,
            "Usage: {} <engine_path> <fen> [nodes]", argv[0]);
        std::println(stderr,
            "  engine_path  Path to TensorRT engine (.network or .trt)");
        std::println(stderr, "  fen          FEN string (quoted)");
        std::println(stderr,
            "  nodes        Max GPU evaluations (default: 400)");
        return 1;
    }

    fs::path engine_path = argv[1];
    std::string fen = argv[2];
    uint64_t target_nodes = 400;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mcts") {
            std::println(stderr,
                "[catgpt_search] --mcts is no longer supported under LKS; ignoring");
            continue;
        }
        try {
            const uint64_t n = std::stoull(a);
            if (n >= 1) target_nodes = n;
        } catch (...) {
            // Ignore unparseable positional args (mirrors prior leniency).
        }
    }

    if (!fs::exists(engine_path)) {
        std::println(stderr, "Error: TensorRT engine file not found: {}",
                     engine_path.string());
        return 1;
    }

    const int workers_per_gpu  = env_int("LKS_WORKERS_PER_GPU", 1);
    const int coros_per_worker = env_int("LKS_COROS_PER_WORKER", 112);
    const int max_batch_size   = env_int("LKS_MAX_BATCH_SIZE", 56);
    const uint64_t lifetime_max_evals =
        env_u64("LKS_LIFETIME_MAX_EVALS", 1ULL << 27);

    fs::path syzygy_path;
    if (const char* p = std::getenv("LKS_SYZYGY_PATH"); p && *p) {
        syzygy_path = p;
    } else if (const char* p = std::getenv("SYZYGY_HOME"); p && *p) {
        syzygy_path = p;
    }

    try {
        std::println(stderr, "Loading TensorRT engine: {}",
                     engine_path.string());
        std::println(stderr,
            "Config: workers_per_gpu={} coros_per_worker={} max_batch={} "
            "arena_capacity={} syzygy={}",
            workers_per_gpu, coros_per_worker, max_batch_size,
            lifetime_max_evals,
            syzygy_path.empty() ? std::string{"<disabled>"}
                                : syzygy_path.string());
        std::cerr.flush();

        LksSearch search(engine_path, lifetime_max_evals,
                         workers_per_gpu, coros_per_worker,
                         max_batch_size, std::move(syzygy_path));

        std::println(stderr,
            "Engine loaded; searching up to {} evals from FEN", target_nodes);
        std::cerr.flush();

        search.setBoard(chess::Board(fen));

        LksSearchConfig cfg;
        cfg.max_evals = target_nodes;
        // Discard LKS's UCI-formatted info lines — we emit our own JSON.
        cfg.on_uci_line = [](std::string_view) {};

        // Capture descent-time tunables BEFORE moving cfg into search().
        const float c_puct        = cfg.params.c_puct;
        const float fpu_reduction = cfg.params.fpu_reduction;

        search.search(std::move(cfg));

        // ── Phase 1: spin until the root TT entry exists ─────────────
        // The first GPU eval populates it; with workers_per_gpu=1 that's
        // ~one batched inference after spin-up. 60s is a generous cap
        // for the pathological "TRT engine warm-up + first eval".
        RootSnapshot snap;
        const auto t_phase1 = std::chrono::steady_clock::now();
        while (search.is_searching()) {
            if (snapshot_root(search, c_puct, fpu_reduction, snap)) break;
            if (std::chrono::steady_clock::now() - t_phase1
                > std::chrono::seconds(60)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!snap.entries.empty()) {
            emit_json("root_eval", snap);
        }

        // ── Phase 2: stream search_update lines ──────────────────────
        // Emit when centi-depth or eval-count changes, throttled to
        // 250ms; heartbeat every 2s to keep the SSE channel warm.
        int      last_depth_centi = snap.depth_centi;
        uint64_t last_evals       = snap.total_evals;
        auto     last_emit_t      = std::chrono::steady_clock::now();
        while (search.is_searching()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            RootSnapshot tmp;
            if (!snapshot_root(search, c_puct, fpu_reduction, tmp)) continue;
            const auto now      = std::chrono::steady_clock::now();
            const auto since    = now - last_emit_t;
            const bool changed  = tmp.depth_centi != last_depth_centi
                               || tmp.total_evals != last_evals;
            const bool throttle = since < std::chrono::milliseconds(250);
            const bool heart    = since >= std::chrono::seconds(2);
            if ((changed && !throttle) || heart) {
                emit_json("search_update", tmp);
                last_depth_centi = tmp.depth_centi;
                last_evals       = tmp.total_evals;
                last_emit_t      = now;
            }
        }

        // ── Phase 3: final snapshot + bestmove ──────────────────────
        RootSnapshot final_snap;
        if (snapshot_root(search, c_puct, fpu_reduction, final_snap)) {
            emit_json("search_complete", final_snap);
        }

        const chess::Move best = search.bestmove();
        if (best != chess::Move::NO_MOVE) {
            std::cout << "bestmove " << chess::uci::moveToUci(best)
                      << std::endl;
        } else {
            std::cout << "bestmove 0000" << std::endl;
        }
    } catch (const std::exception& e) {
        std::println(stderr, "Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
