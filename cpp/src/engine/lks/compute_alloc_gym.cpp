/**
 * compute-allocations gym driver.
 *
 * Sweeps a grid of synthetic instances:
 *   - M           ∈ {2, 5, 12, 30, 60}        (num children)
 *   - depth       ∈ {0, 4, 10, 16, 22}        (N from 1 to ~3.6e9)
 *   - P shape     ∈ {uniform, zipf(0.5), one-hot(0.9), two-hot(0.45)}
 *   - Q baseline  ∈ {-0.9, 0, +0.9}
 *   - Q spread    ∈ {0.01, 0.1, 0.5}          (std-dev around baseline, clipped to [-1,1])
 *
 *   = 5 * 5 * 4 * 3 * 3 = 900 instances per seed; default 4 seeds = 3600 instances.
 *
 * For each instance:
 *   1. Compute the long-double reference.
 *   2. Run each solver in registry().
 *   3. Record:
 *        - max |log_n_solver - log_n_ref|       (per-child log accuracy)
 *        - |residual_log_sum|                   (budget conservation)
 *        - iters + extra_iters                  (convergence cost)
 *   4. Time each solver (median of N_TIMING_REPS calls).
 *
 * Output:
 *   - Per-cell summary (M × depth) of median accuracy + iters + ns/call
 *   - Global leaderboard: (solver, mean ns/call, p99 ns/call, max log error)
 *   - CSV dump if --csv=PATH is passed
 *
 * Standalone build: no TRT, no chess-library, no libfork. Just C++23.
 *
 * Usage:
 *   ./compute_alloc_gym              # default sweep + leaderboard
 *   ./compute_alloc_gym --reps=200   # timing reps per instance (default 100)
 *   ./compute_alloc_gym --seeds=8    # number of RNG seeds (default 4)
 *   ./compute_alloc_gym --csv=out.csv
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "compute_alloc_gym.hpp"

namespace gym = catgpt::lks::gym;

// ─── Instance generation ──────────────────────────────────────────────────

enum class PShape  : int { Uniform, Zipf05, OneHot09, TwoHot045 };
enum class QSpread : int { S001, S01, S05 };

constexpr float kQSpreadVals[] = {0.01f, 0.1f, 0.5f};

struct InstanceSpec {
    int     M;
    float   depth;
    PShape  p_shape;
    float   q_base;
    QSpread q_spread;
    uint64_t seed;
};

inline std::vector<float> gen_priors(PShape s, int M, std::mt19937_64& rng) {
    std::vector<float> P(M);
    switch (s) {
        case PShape::Uniform: {
            for (int i = 0; i < M; ++i) P[i] = 1.0f;
            break;
        }
        case PShape::Zipf05: {
            // p_i ∝ 1/(i+1)^0.5, then shuffled.
            for (int i = 0; i < M; ++i) P[i] = 1.0f / std::sqrt(static_cast<float>(i + 1));
            std::shuffle(P.begin(), P.end(), rng);
            break;
        }
        case PShape::OneHot09: {
            // 0.9 mass on one child, 0.1 spread uniform among the rest.
            const int hot = static_cast<int>(rng() % static_cast<uint64_t>(M));
            const float others = (M > 1) ? 0.1f / static_cast<float>(M - 1) : 0.0f;
            for (int i = 0; i < M; ++i) P[i] = (i == hot) ? 0.9f : others;
            // Already sums to 1.
            return P;
        }
        case PShape::TwoHot045: {
            const int h1 = static_cast<int>(rng() % static_cast<uint64_t>(M));
            int h2 = static_cast<int>(rng() % static_cast<uint64_t>(M));
            if (M > 1) while (h2 == h1) h2 = static_cast<int>(rng() % static_cast<uint64_t>(M));
            const float others = (M > 2) ? 0.1f / static_cast<float>(M - 2) : 0.0f;
            for (int i = 0; i < M; ++i) {
                P[i] = (i == h1 || i == h2) ? 0.45f : others;
            }
            if (M == 1) P[0] = 1.0f;
            return P;
        }
    }
    // Normalise (cases that fall through).
    double s_total = 0.0;
    for (float v : P) s_total += v;
    if (s_total > 0.0) for (auto& v : P) v = static_cast<float>(v / s_total);
    return P;
}

inline std::vector<float> gen_q(float base, float spread, int M, std::mt19937_64& rng) {
    std::vector<float> Q(M);
    std::normal_distribution<float> dist(0.0f, spread);
    for (int i = 0; i < M; ++i) {
        float q = base + dist(rng);
        if (q > 1.0f) q = 1.0f;
        if (q < -1.0f) q = -1.0f;
        Q[i] = q;
    }
    return Q;
}

inline gym::Instance make_instance(const InstanceSpec& s) {
    std::mt19937_64 rng(s.seed);
    gym::Instance inst;
    inst.depth = s.depth;
    inst.c_puct = 2.0f;
    inst.P = gen_priors(s.p_shape, s.M, rng);
    inst.Q = gen_q(s.q_base, kQSpreadVals[static_cast<int>(s.q_spread)], s.M, rng);
    return inst;
}

// ─── Driver ───────────────────────────────────────────────────────────────

struct CellAgg {
    // Per (solver, M, depth) bucket.
    int    n = 0;
    double sum_max_log_err = 0.0;
    double max_max_log_err = 0.0;
    double sum_iters       = 0.0;
    double sum_extra       = 0.0;
    double sum_ns          = 0.0;
    double max_ns          = 0.0;
};

struct GlobalAgg {
    int    n = 0;
    double sum_ns = 0.0;
    std::vector<double> ns_samples; // for percentiles
    double max_log_err = 0.0;
    double sum_iters = 0.0;
};

static double max_log_err(const gym::Result& a, const gym::Result& ref) {
    double m = 0.0;
    const std::size_t M = ref.log_n.size();
    for (std::size_t i = 0; i < M; ++i) {
        if (!std::isfinite(ref.log_n[i]) && !std::isfinite(a.log_n[i])) continue;
        const double e = std::fabs(static_cast<double>(a.log_n[i])
                                 - static_cast<double>(ref.log_n[i]));
        if (e > m) m = e;
    }
    return m;
}

int main(int argc, char** argv) {
    int reps = 100;
    int seeds = 4;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a.starts_with("--reps="))      reps  = std::atoi(argv[i] + 7);
        else if (a.starts_with("--seeds=")) seeds = std::atoi(argv[i] + 8);
        else if (a.starts_with("--csv="))   csv_path = std::string(a.substr(6));
        else if (a == "-h" || a == "--help") {
            std::puts("Usage: compute_alloc_gym [--reps=N] [--seeds=N] [--csv=PATH]");
            return 0;
        }
    }

    const std::vector<int>     Ms      = {2, 5, 12, 30, 60};
    const std::vector<float>   depths  = {0.0f, 4.0f, 10.0f, 16.0f, 22.0f};
    const std::vector<PShape>  shapes  = {PShape::Uniform, PShape::Zipf05,
                                          PShape::OneHot09, PShape::TwoHot045};
    const std::vector<float>   qbases  = {-0.9f, 0.0f, 0.9f};
    const std::vector<QSpread> qspreads= {QSpread::S001, QSpread::S01, QSpread::S05};

    const auto& solvers = gym::registry();

    // Aggregations.
    // Cell key: (solver_idx * |M| * |depth|) + m_i * |depth| + d_i
    auto cell_idx = [&](int s_i, int m_i, int d_i) {
        return (s_i * static_cast<int>(Ms.size()) + m_i) * static_cast<int>(depths.size()) + d_i;
    };
    std::vector<CellAgg> cells(solvers.size() * Ms.size() * depths.size());
    std::vector<GlobalAgg> globals(solvers.size());

    std::ofstream csv;
    if (!csv_path.empty()) {
        csv.open(csv_path);
        csv << "solver,M,depth,p_shape,q_base,q_spread,seed,"
               "max_log_err,iters,extra_iters,residual_log_sum,ns_per_call\n";
    }

    long long total_instances = 0;

    for (int seed_i = 0; seed_i < seeds; ++seed_i) {
    for (std::size_t m_i = 0; m_i < Ms.size(); ++m_i) {
    for (std::size_t d_i = 0; d_i < depths.size(); ++d_i) {
    for (std::size_t p_i = 0; p_i < shapes.size(); ++p_i) {
    for (float qb : qbases) {
    for (std::size_t qs_i = 0; qs_i < qspreads.size(); ++qs_i) {
        InstanceSpec spec{
            .M       = Ms[m_i],
            .depth   = depths[d_i],
            .p_shape = shapes[p_i],
            .q_base  = qb,
            .q_spread= qspreads[qs_i],
            .seed    = static_cast<uint64_t>(0xC47C7717ULL)
                     ^ (static_cast<uint64_t>(seed_i)   * 0x9E3779B97F4A7C15ULL)
                     ^ (static_cast<uint64_t>(m_i)      * 0xBF58476D1CE4E5B9ULL)
                     ^ (static_cast<uint64_t>(d_i)      * 0x94D049BB133111EBULL)
                     ^ (static_cast<uint64_t>(p_i)      * 0xD1B54A32D192ED03ULL)
                     ^ static_cast<uint64_t>(static_cast<int>(qb * 10) & 0xFF)
                     ^ (static_cast<uint64_t>(qs_i)     * 0xCBF29CE484222325ULL),
        };
        gym::Instance inst = make_instance(spec);
        gym::Result ref = gym::solve_reference(inst);

        for (std::size_t s_i = 0; s_i < solvers.size(); ++s_i) {
            gym::Options opts;
            opts.max_iters = 64;
            opts.log_tol   = 1e-6;
            gym::Result out = solvers[s_i].solve(inst, opts);

            const double err = max_log_err(out, ref);

            // Time it: median of `reps` runs. Touch a sink so the
            // optimizer can't elide.
            volatile double sink = 0.0;
            std::vector<double> times_ns;
            times_ns.reserve(reps);
            for (int r = 0; r < reps; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                gym::Result rr = solvers[s_i].solve(inst, opts);
                auto t1 = std::chrono::steady_clock::now();
                sink += static_cast<double>(rr.log_n.empty() ? 0.0 : rr.log_n[0]);
                const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                times_ns.push_back(ns);
            }
            (void)sink;
            std::sort(times_ns.begin(), times_ns.end());
            const double median_ns = times_ns[times_ns.size() / 2];

            CellAgg& c = cells[cell_idx(static_cast<int>(s_i),
                                        static_cast<int>(m_i),
                                        static_cast<int>(d_i))];
            c.n += 1;
            c.sum_max_log_err += err;
            if (err > c.max_max_log_err) c.max_max_log_err = err;
            c.sum_iters += out.iters;
            c.sum_extra += out.extra_iters;
            c.sum_ns    += median_ns;
            if (median_ns > c.max_ns) c.max_ns = median_ns;

            GlobalAgg& g = globals[s_i];
            g.n += 1;
            g.sum_ns += median_ns;
            g.ns_samples.push_back(median_ns);
            if (err > g.max_log_err) g.max_log_err = err;
            g.sum_iters += out.iters + out.extra_iters;

            if (csv.is_open()) {
                csv << solvers[s_i].name << ',' << spec.M << ',' << spec.depth
                    << ',' << static_cast<int>(spec.p_shape)
                    << ',' << spec.q_base
                    << ',' << kQSpreadVals[static_cast<int>(spec.q_spread)]
                    << ',' << spec.seed
                    << ',' << err
                    << ',' << out.iters
                    << ',' << out.extra_iters
                    << ',' << out.residual_log_sum
                    << ',' << median_ns << '\n';
            }
        }
        ++total_instances;
    }}}}}}

    // ── Per-cell report (M × depth) per solver ────────────────────────────
    std::printf("\n=== Per-cell summary (mean ns/call · mean iters · mean log err) ===\n");
    std::printf("(reps=%d, seeds=%d, total instances per solver=%lld)\n",
                reps, seeds, total_instances);
    for (std::size_t s_i = 0; s_i < solvers.size(); ++s_i) {
        std::printf("\n[solver: %s]\n", solvers[s_i].name);
        std::printf("                ");
        for (float d : depths) std::printf("  d=%5.1f          ", static_cast<double>(d));
        std::printf("\n");
        for (std::size_t m_i = 0; m_i < Ms.size(); ++m_i) {
            std::printf("  M=%3d        ", Ms[m_i]);
            for (std::size_t d_i = 0; d_i < depths.size(); ++d_i) {
                const CellAgg& c = cells[cell_idx(static_cast<int>(s_i),
                                                  static_cast<int>(m_i),
                                                  static_cast<int>(d_i))];
                if (c.n == 0) {
                    std::printf("  --                ");
                    continue;
                }
                std::printf("  %6.0fns/%4.1fit/%.1e",
                            c.sum_ns / c.n,
                            (c.sum_iters + c.sum_extra) / c.n,
                            c.sum_max_log_err / c.n);
            }
            std::printf("\n");
        }
    }

    // ── Global leaderboard ────────────────────────────────────────────────
    std::printf("\n=== Global leaderboard ===\n");
    std::printf("%-22s | mean ns | p50 ns | p99 ns | mean iters | max |Δlog n|\n",
                "solver");
    std::printf("---------------------- -+---------+--------+--------+------------+----------------\n");
    for (std::size_t s_i = 0; s_i < solvers.size(); ++s_i) {
        GlobalAgg& g = globals[s_i];
        if (g.ns_samples.empty()) continue;
        std::sort(g.ns_samples.begin(), g.ns_samples.end());
        const double p50 = g.ns_samples[g.ns_samples.size() / 2];
        const double p99 = g.ns_samples[std::min<std::size_t>(g.ns_samples.size() - 1,
                                                              g.ns_samples.size() * 99 / 100)];
        std::printf("%-22s | %7.0f | %6.0f | %6.0f | %10.2f | %14.2e\n",
                    solvers[s_i].name,
                    g.sum_ns / g.n, p50, p99,
                    g.sum_iters / g.n, g.max_log_err);
    }

    // ── Warm-start chained sweep ──────────────────────────────────────────
    // Emulates LKS's ID loop: walk depth from 0 to 22 in steps of 0.2,
    // threading u* between consecutive solves on the same instance.
    // Compares "cold" (no warm start) vs "warm" (warm_u = previous u*)
    // for each Newton-class solver. Reports mean iters + ns/call across
    // all (instance, depth) pairs after the very first depth (which has
    // no predecessor).
    //
    // This is the production-realistic regime: production never solves a
    // single fresh instance — it solves a sequence indexed by depth.
    std::printf("\n=== Warm-start chained sweep (depth 0 -> 22, step 0.2) ===\n");
    std::printf("Cold = no warm start each step. Warm = thread previous u* into next.\n\n");
    struct WarmStat { double sum_ns_cold=0, sum_ns_warm=0;
                      double sum_it_cold=0, sum_it_warm=0;
                      int n=0; };
    std::vector<WarmStat> warm_stats(solvers.size());

    for (int seed_i = 0; seed_i < seeds; ++seed_i) {
    for (int M : Ms) {
    for (PShape ps : shapes) {
    for (float qb : qbases) {
    for (QSpread qsv : qspreads) {
        // Use the same seed scheme (without depth in the mix) so that
        // priors/Q stay constant along the depth track.
        const uint64_t seed_base =
              static_cast<uint64_t>(0xC47C7717ULL)
            ^ (static_cast<uint64_t>(seed_i)        * 0x9E3779B97F4A7C15ULL)
            ^ (static_cast<uint64_t>(M)             * 0xBF58476D1CE4E5B9ULL)
            ^ (static_cast<uint64_t>(static_cast<int>(ps)) * 0xD1B54A32D192ED03ULL)
            ^ static_cast<uint64_t>(static_cast<int>(qb * 10) & 0xFF)
            ^ (static_cast<uint64_t>(static_cast<int>(qsv)) * 0xCBF29CE484222325ULL);

        // Solver-specific previous u*.
        std::vector<double> prev_u(solvers.size(), 0.0);
        std::vector<bool>   has_prev(solvers.size(), false);

        for (float depth = 0.0f; depth <= 22.0f + 1e-3f; depth += 0.2f) {
            InstanceSpec spec{M, depth, ps, qb, qsv, seed_base};
            gym::Instance inst = make_instance(spec);

            for (std::size_t s_i = 0; s_i < solvers.size(); ++s_i) {
                gym::Options cold_opts;
                cold_opts.max_iters = 64;
                cold_opts.log_tol   = 1e-6;

                gym::Options warm_opts = cold_opts;
                warm_opts.has_warm_u = has_prev[s_i];
                warm_opts.warm_u     = prev_u[s_i];

                auto time_one = [&](const gym::Options& o) {
                    volatile double sink = 0.0;
                    std::vector<double> times_ns;
                    times_ns.reserve(reps);
                    int iters_last = 0;
                    for (int r = 0; r < reps; ++r) {
                        auto t0 = std::chrono::steady_clock::now();
                        gym::Result rr = solvers[s_i].solve(inst, o);
                        auto t1 = std::chrono::steady_clock::now();
                        sink += static_cast<double>(rr.log_n.empty() ? 0.0 : rr.log_n[0]);
                        times_ns.push_back(
                            std::chrono::duration<double, std::nano>(t1 - t0).count());
                        iters_last = rr.iters + rr.extra_iters;
                    }
                    (void)sink;
                    std::sort(times_ns.begin(), times_ns.end());
                    return std::pair{times_ns[times_ns.size()/2],
                                     static_cast<double>(iters_last)};
                };

                auto [ns_cold, it_cold] = time_one(cold_opts);
                auto [ns_warm, it_warm] = time_one(warm_opts);

                if (has_prev[s_i]) {
                    WarmStat& w = warm_stats[s_i];
                    w.sum_ns_cold += ns_cold;
                    w.sum_ns_warm += ns_warm;
                    w.sum_it_cold += it_cold;
                    w.sum_it_warm += it_warm;
                    w.n += 1;
                }

                // Update predecessor for next depth using a fresh solve
                // (the `warm` timing run already computed it, but its
                // result wasn't returned through `time_one`). One more
                // call — cheap.
                gym::Result final_r = solvers[s_i].solve(inst, warm_opts);
                if (final_r.converged) {
                    prev_u[s_i] = final_r.u_star;
                    has_prev[s_i] = true;
                }
            }
        }
    }}}}}

    std::printf("%-22s | cold iters | warm iters | cold ns | warm ns | speedup\n",
                "solver");
    std::printf("---------------------- -+------------+------------+---------+---------+--------\n");
    for (std::size_t s_i = 0; s_i < solvers.size(); ++s_i) {
        const WarmStat& w = warm_stats[s_i];
        if (w.n == 0) continue;
        const double cold_ns = w.sum_ns_cold / w.n;
        const double warm_ns = w.sum_ns_warm / w.n;
        std::printf("%-22s | %10.2f | %10.2f | %7.0f | %7.0f | %5.2fx\n",
                    solvers[s_i].name,
                    w.sum_it_cold / w.n,
                    w.sum_it_warm / w.n,
                    cold_ns, warm_ns,
                    cold_ns / std::max(warm_ns, 1.0));
    }

    if (csv.is_open()) std::printf("\nCSV: %s\n", csv_path.c_str());
    return 0;
}
