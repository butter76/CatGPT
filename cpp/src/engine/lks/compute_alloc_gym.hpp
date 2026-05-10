/**
 * Compute-allocations gym: core types, shared math, and solver implementations.
 *
 * Problem (recap of coroutine_search.hpp::compute_allocations):
 *
 *   Given M children with priors P_i (sum 1) and child-Q values Q_i in [-1,1],
 *   total budget N, and constant c_puct, find the dual K such that
 *
 *       sum_i N_i = N,    N_i = c_puct * P_i * N^(2/3) / [3 (K - q_i)]
 *
 *   with q_i = -Q_i. K is unique on (q_max, +inf) (LHS is strictly decreasing
 *   in K). Output: log(N_i) for each child. We work in log space because LKS
 *   carries depth = log(N) everywhere, and so individual N_i may exceed
 *   ~4e9 (i.e. 2^32) but log N_i stays bounded.
 *
 * Internal reformulation used by every non-baseline solver:
 *
 *   u       = log(K - q_max)     (scalar dual; well-conditioned across N)
 *   Delta_i = q_max - q_i        (>= 0, computed once)
 *   delta_i = e^u + Delta_i      (>= e^u > 0, no cancellation)
 *   alpha_i = log P_i - log delta_i
 *   f(u)    = lse_i(alpha_i) - T_log,    T_log = d/3 + log(3/c_puct)
 *   log N_i = alpha_i + 2d/3 + log(c_puct/3)
 *
 *   f is strictly decreasing in u; f(-inf)=+inf, f(+inf)=-T_log.
 *
 * Why this works for N up to 4e9 (depth ~ 22.3):
 *   - K - q_i shrinks like N^(-1/3) ~ 6e-4 at the top end, but we never form
 *     it as a difference. delta_i = e^u + Delta_i has each term independently
 *     bounded.
 *   - lse on alpha_i avoids overflow in the sum.
 *   - log N_i is a sum of bounded terms; result stays in float range.
 *
 * Reference solver: long-double bisection in u to ~1e-15.
 *
 * Solvers are pure functions taking an Instance + Options and returning
 * a Result. They are header-only and zero-allocation per call after the
 * caller provides a Scratch (separate header) — for now we accept one
 * std::vector resize per call to keep the API simple.
 */

#ifndef CATGPT_ENGINE_LKS_COMPUTE_ALLOC_GYM_HPP
#define CATGPT_ENGINE_LKS_COMPUTE_ALLOC_GYM_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace catgpt::lks::gym {

// ─── Types ────────────────────────────────────────────────────────────────

struct Instance {
    std::vector<float> P;      // length M, nonneg, sum to 1 (we tolerate slop)
    std::vector<float> Q;      // length M, in [-1, 1]
    float depth = 0.0f;        // d = log(N)
    float c_puct = 1.0f;
};

struct Result {
    std::vector<float> log_n;  // length M, log of allocation per child
    double u_star = 0.0;       // log(K - q_max) at convergence
    int    iters = 0;          // primary iteration count for the solver
    int    extra_iters = 0;    // bracketing/expansion iters (if any)
    bool   converged = false;
    // Diagnostics (filled by finalize()):
    double residual_log_sum = 0.0;  // lse_i(log N_i) - d  (should be ~0)
};

struct Options {
    int    max_iters = 64;
    double log_tol   = 1e-6;   // |f(u)| <= log_tol or |Δu| relative <= log_tol
    bool   has_warm_u = false; // optional warm-start in u-space
    double warm_u    = 0.0;
};

// ─── Shared math helpers ──────────────────────────────────────────────────

inline float q_max_of(const Instance& inst) noexcept {
    float m = -std::numeric_limits<float>::infinity();
    for (float Q : inst.Q) m = std::max(m, -Q);
    return m;
}

// Build Delta_i = q_max - q_i (>= 0).
inline void build_Delta(const Instance& inst, float qmax,
                        std::vector<double>& Delta_out)
{
    Delta_out.resize(inst.Q.size());
    for (std::size_t i = 0; i < inst.Q.size(); ++i) {
        Delta_out[i] = static_cast<double>(qmax) - static_cast<double>(-inst.Q[i]);
    }
}

// Evaluate f(u) and f'(u) (and optionally f''(u)) in double precision,
// using the log-domain reformulation. Returns:
//   f          = log_S - T_log
//   f_prime    = -e^u * (sum_i a_i / delta_i) / S       (always <= 0)
//   f_prime2   = e^u * (S' - e^u * sum a/d^2 / S - (S')^2 / e^u) ... see code
// Caller supplies Delta + scratch alpha vector; we don't allocate.
struct FEval {
    double f;
    double f_prime;
    double f_prime2;
};

inline FEval eval_f(double u,
                    const std::vector<double>& P,    // priors (double)
                    const std::vector<double>& Delta,// nonneg gaps
                    double T_log,
                    std::vector<double>& alpha_scratch)
{
    const std::size_t M = P.size();
    alpha_scratch.resize(M);
    const double eu = std::exp(u);

    double max_alpha = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < M; ++i) {
        if (P[i] <= 0.0) {
            alpha_scratch[i] = -std::numeric_limits<double>::infinity();
            continue;
        }
        const double delta = eu + Delta[i];
        alpha_scratch[i] = std::log(P[i]) - std::log(delta);
        if (alpha_scratch[i] > max_alpha) max_alpha = alpha_scratch[i];
    }
    if (!std::isfinite(max_alpha)) {
        // No active children; degenerate.
        return {-T_log, 0.0, 0.0};
    }

    // Compute S (scaled by exp(-max_alpha) for stability) and the moments
    //   M1 = sum_i a_i / delta_i,    M2 = sum_i a_i / delta_i^2
    // (also scaled by exp(-max_alpha)). Ratios M1/S, M2/S are scale-free.
    double S = 0.0, M1 = 0.0, M2 = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        if (!std::isfinite(alpha_scratch[i])) continue;
        const double a = std::exp(alpha_scratch[i] - max_alpha);
        const double delta = eu + Delta[i];
        S  += a;
        M1 += a / delta;
        M2 += a / (delta * delta);
    }
    const double log_S = max_alpha + std::log(S);
    const double r1 = M1 / S;          // = (sum a/delta) / (sum a)
    const double r2 = M2 / S;          // = (sum a/delta^2) / (sum a)

    // f'(u): chain rule on log_S w.r.t. u, with d alpha_i / du = -e^u/delta_i.
    //   f'(u) = sum_i p_i * (-e^u/delta_i) = -e^u * r1.
    const double fp = -eu * r1;

    // f''(u): differentiate fp = -e^u * r1.
    //   r1' (w.r.t. u) = -e^u * (r2 - r1^2)        (variance-like; >=0)
    //   so fp' = -e^u * r1 - e^u * r1'
    //          = -e^u * r1 + e^{2u} * (r2 - r1^2)
    const double fpp = -eu * r1 + eu * eu * (r2 - r1 * r1);

    return {log_S - T_log, fp, fpp};
}

// Once u* is known, materialise log_n_i and the diagnostic residual.
inline void finalize(const Instance& inst,
                     const std::vector<double>& P,
                     const std::vector<double>& Delta,
                     double u_star,
                     Result& r)
{
    const std::size_t M = inst.P.size();
    const double eu = std::exp(u_star);
    const double bias = (2.0/3.0) * static_cast<double>(inst.depth)
                      + std::log(static_cast<double>(inst.c_puct) / 3.0);

    r.log_n.resize(M);
    double log_max = -std::numeric_limits<double>::infinity();
    std::vector<double> log_n_d(M, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < M; ++i) {
        if (P[i] <= 0.0) {
            r.log_n[i] = -std::numeric_limits<float>::infinity();
            continue;
        }
        const double delta = eu + Delta[i];
        log_n_d[i] = std::log(P[i]) - std::log(delta) + bias;
        r.log_n[i] = static_cast<float>(log_n_d[i]);
        if (log_n_d[i] > log_max) log_max = log_n_d[i];
    }
    double s = 0.0;
    for (double x : log_n_d) {
        if (std::isfinite(x)) s += std::exp(x - log_max);
    }
    const double lse = std::isfinite(log_max) ? log_max + std::log(s)
                                              : -std::numeric_limits<double>::infinity();
    r.u_star = u_star;
    r.residual_log_sum = lse - static_cast<double>(inst.depth);
}

// Analytic init for u: u_apx = -d/3 + log(c_puct/3). This corresponds to
// "all q_i collapsed to q_max" (Delta_i == 0 everywhere); since real
// Delta_i >= 0 only shrinks the LHS sum, real u* <= u_apx always.
// Hence u_apx is a valid upper bound for bracketing.
inline double analytic_u_upper(const Instance& inst) noexcept {
    return -static_cast<double>(inst.depth) / 3.0
         + std::log(static_cast<double>(inst.c_puct) / 3.0);
}

// Cast inst.P/inst.Q -> double and build Delta. One-shot per call.
struct InstanceD {
    std::vector<double> P;
    std::vector<double> Delta;
    double T_log;
    float qmax;
};

inline InstanceD lift(const Instance& inst) {
    InstanceD d;
    const std::size_t M = inst.P.size();
    d.P.resize(M);
    for (std::size_t i = 0; i < M; ++i) d.P[i] = inst.P[i];
    d.qmax = q_max_of(inst);
    build_Delta(inst, d.qmax, d.Delta);
    d.T_log = static_cast<double>(inst.depth) / 3.0
            + std::log(3.0 / static_cast<double>(inst.c_puct));
    return d;
}

// ─── Reference solver: long-double bisection in u to ~1e-15 ───────────────

inline Result solve_reference(const Instance& inst) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    // Long-double precomputation.
    std::vector<long double> P_ld(M), Delta_ld(M);
    long double qmax = -std::numeric_limits<long double>::infinity();
    for (std::size_t i = 0; i < M; ++i) {
        const long double q = -static_cast<long double>(inst.Q[i]);
        if (q > qmax) qmax = q;
    }
    for (std::size_t i = 0; i < M; ++i) {
        P_ld[i] = static_cast<long double>(inst.P[i]);
        Delta_ld[i] = qmax - (-static_cast<long double>(inst.Q[i]));
    }
    const long double T_log = static_cast<long double>(inst.depth) / 3.0L
                            + std::log(3.0L / static_cast<long double>(inst.c_puct));

    auto f_ld = [&](long double u) -> long double {
        const long double eu = std::exp(u);
        long double max_alpha = -std::numeric_limits<long double>::infinity();
        for (std::size_t i = 0; i < M; ++i) {
            if (P_ld[i] <= 0.0L) continue;
            const long double a = std::log(P_ld[i]) - std::log(eu + Delta_ld[i]);
            if (a > max_alpha) max_alpha = a;
        }
        if (!std::isfinite(static_cast<double>(max_alpha))) return -T_log;
        long double S = 0.0L;
        for (std::size_t i = 0; i < M; ++i) {
            if (P_ld[i] <= 0.0L) continue;
            const long double a = std::log(P_ld[i]) - std::log(eu + Delta_ld[i]);
            S += std::exp(a - max_alpha);
        }
        return max_alpha + std::log(S) - T_log;
    };

    // Bracket. Start from analytic upper, walk down for the lower.
    long double u_hi = static_cast<long double>(analytic_u_upper(inst));
    // Guarantee f(u_hi) <= 0 (it should equal 0 in the all-q-equal case;
    // otherwise it's negative). Walk up if not.
    for (int i = 0; i < 200 && f_ld(u_hi) > 0.0L; ++i) u_hi += 1.0L;
    long double u_lo = u_hi - 1.0L;
    for (int i = 0; i < 400 && f_ld(u_lo) < 0.0L; ++i) u_lo -= 1.0L;

    for (int i = 0; i < 256; ++i) {
        const long double mid = 0.5L * (u_lo + u_hi);
        if (f_ld(mid) > 0.0L) u_lo = mid; else u_hi = mid;
        if ((u_hi - u_lo) <= 1e-15L * (std::fabs(u_hi) + 1.0L)) break;
    }
    const long double u_star = 0.5L * (u_lo + u_hi);

    auto inst_d = lift(inst);
    r.iters = 256;
    r.converged = true;
    finalize(inst, inst_d.P, inst_d.Delta, static_cast<double>(u_star), r);
    return r;
}

// ─── A1: bisection in K (faithful port of coroutine_search.hpp) ──────────
// Same structure as production: 100 doublings to find K_high, then 64
// fixed bisection iterations. Honors max_iters (cap on the bisection
// phase only; doublings counted in extra_iters).

inline Result solve_bisection_K(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    const float qmax = q_max_of(inst);
    const double N = std::exp(static_cast<double>(inst.depth));
    const double w_factor = static_cast<double>(inst.c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;

    auto sum_alloc = [&](double K) {
        double s = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            const double d = K - static_cast<double>(-inst.Q[i]);
            if (d <= 0.0) return std::numeric_limits<double>::infinity();
            s += w_factor * static_cast<double>(inst.P[i]) / d;
        }
        return s;
    };

    double K_lo = static_cast<double>(qmax) + 1e-9;
    double K_hi = K_lo + 10.0;
    int expand = 0;
    for (; expand < 100; ++expand) {
        if (sum_alloc(K_hi) <= N) break;
        K_hi *= 2.0;
    }

    const int cap = std::min(opts.max_iters, 64);
    int it = 0;
    for (; it < cap; ++it) {
        const double K_mid = 0.5 * (K_lo + K_hi);
        if (sum_alloc(K_mid) > N) K_lo = K_mid; else K_hi = K_mid;
    }
    const double K = 0.5 * (K_lo + K_hi);
    const double u = std::log(K - static_cast<double>(qmax));

    auto inst_d = lift(inst);
    r.iters = it;
    r.extra_iters = expand;
    r.converged = true;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A1b: bisection in K with early exit at log_tol ─────────────────────
// Same as A1 but checks |Δlog n| (proxied by relative bracket width on K
// in log-space) per iteration and exits early. Production-realistic
// "what if A1 were tightened up to honor a tolerance".

inline Result solve_bisection_K_early(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    const double qmax = q_max_of(inst);
    const double N = std::exp(static_cast<double>(inst.depth));
    const double w_factor = static_cast<double>(inst.c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;

    auto sum_alloc = [&](double K) {
        double s = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            const double d = K - static_cast<double>(-inst.Q[i]);
            if (d <= 0.0) return std::numeric_limits<double>::infinity();
            s += w_factor * static_cast<double>(inst.P[i]) / d;
        }
        return s;
    };

    double K_lo = qmax + 1e-9;
    double K_hi = K_lo + 10.0;
    int expand = 0;
    for (; expand < 100; ++expand) {
        if (sum_alloc(K_hi) <= N) break;
        K_hi *= 2.0;
    }

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        const double K_mid = 0.5 * (K_lo + K_hi);
        if (sum_alloc(K_mid) > N) K_lo = K_mid; else K_hi = K_mid;
        // Early exit: bracket on u = log(K - qmax) narrowed below tol.
        // (K_hi - K_lo) / (K_mid - qmax) ≈ Δu, the spec we care about.
        const double width_u = (K_hi - K_lo) / std::max(K_mid - qmax, 1e-30);
        if (width_u <= opts.log_tol) { ++it; break; }
    }
    const double K = 0.5 * (K_lo + K_hi);
    const double u = std::log(K - qmax);

    auto inst_d = lift(inst);
    r.iters = it;
    r.extra_iters = expand;
    r.converged = it < opts.max_iters;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A2: bisection in u = log(K - q_max) ─────────────────────────────────
// No bracket-doubling needed: analytic_u_upper is a true upper bound;
// lower bound walks down by halving until f(u_lo) >= 0.

inline Result solve_bisection_u(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    std::vector<double> alpha_scratch;

    auto f = [&](double u) {
        return eval_f(u, inst_d.P, inst_d.Delta, inst_d.T_log, alpha_scratch).f;
    };

    double u_hi = analytic_u_upper(inst);
    // f(u_hi) should be <= 0 by construction; walk up if numerical fuzz.
    int extra = 0;
    while (f(u_hi) > 0.0 && extra < 64) { u_hi += 1.0; ++extra; }
    double u_lo = u_hi - 1.0;
    while (f(u_lo) < 0.0 && extra < 256) { u_lo -= 1.0; ++extra; }

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        const double mid = 0.5 * (u_lo + u_hi);
        const double fm = f(mid);
        if (fm > 0.0) u_lo = mid; else u_hi = mid;
        if (std::fabs(fm) <= opts.log_tol) { ++it; break; }
        if ((u_hi - u_lo) <= opts.log_tol * (std::fabs(u_hi) + 1.0)) { ++it; break; }
    }
    const double u_star = 0.5 * (u_lo + u_hi);
    r.iters = it;
    r.extra_iters = extra;
    r.converged = std::fabs(f(u_star)) <= opts.log_tol;
    finalize(inst, inst_d.P, inst_d.Delta, u_star, r);
    return r;
}

// ─── A3: safeguarded Newton in u ─────────────────────────────────────────
// Newton step on f(u); if step exits bracket, fall back to bisection.
// Bracket maintained from sign of f at each visited point.

inline Result solve_newton_u(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    std::vector<double> alpha_scratch;

    auto eval = [&](double u) {
        return eval_f(u, inst_d.P, inst_d.Delta, inst_d.T_log, alpha_scratch);
    };

    double u_hi = analytic_u_upper(inst);
    int extra = 0;
    {
        FEval e_hi = eval(u_hi);
        while (e_hi.f > 0.0 && extra < 64) { u_hi += 1.0; ++extra; e_hi = eval(u_hi); }
    }
    double u_lo = u_hi - 1.0;
    {
        FEval e_lo = eval(u_lo);
        while (e_lo.f < 0.0 && extra < 256) { u_lo -= 1.0; ++extra; e_lo = eval(u_lo); }
    }

    double u = opts.has_warm_u ? opts.warm_u
             : 0.5 * (u_lo + u_hi);
    if (u <= u_lo || u >= u_hi) u = 0.5 * (u_lo + u_hi);

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        FEval e = eval(u);
        if (std::fabs(e.f) <= opts.log_tol) { ++it; break; }
        if (e.f > 0.0) u_lo = u; else u_hi = u;

        double u_new;
        if (e.f_prime < 0.0 && std::isfinite(e.f_prime)) {
            u_new = u - e.f / e.f_prime;
            if (!(u_new > u_lo && u_new < u_hi)) {
                u_new = 0.5 * (u_lo + u_hi);
            }
        } else {
            u_new = 0.5 * (u_lo + u_hi);
        }
        if (std::fabs(u_new - u) <= opts.log_tol * (std::fabs(u) + 1.0)) {
            u = u_new;
            ++it;
            break;
        }
        u = u_new;
    }

    r.iters = it;
    r.extra_iters = extra;
    r.converged = std::fabs(eval(u).f) <= opts.log_tol;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A5: safeguarded Halley in u ─────────────────────────────────────────
// Cubic convergence using f, f', f'' from the existing eval_f. Same
// safeguarding scheme as A3 (fall back to bisection if step exits bracket).

inline Result solve_halley_u(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    std::vector<double> alpha_scratch;

    auto eval = [&](double u) {
        return eval_f(u, inst_d.P, inst_d.Delta, inst_d.T_log, alpha_scratch);
    };

    double u_hi = analytic_u_upper(inst);
    int extra = 0;
    {
        FEval e_hi = eval(u_hi);
        while (e_hi.f > 0.0 && extra < 64) { u_hi += 1.0; ++extra; e_hi = eval(u_hi); }
    }
    double u_lo = u_hi - 1.0;
    {
        FEval e_lo = eval(u_lo);
        while (e_lo.f < 0.0 && extra < 256) { u_lo -= 1.0; ++extra; e_lo = eval(u_lo); }
    }

    double u = opts.has_warm_u ? opts.warm_u : 0.5 * (u_lo + u_hi);
    if (u <= u_lo || u >= u_hi) u = 0.5 * (u_lo + u_hi);

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        FEval e = eval(u);
        if (std::fabs(e.f) <= opts.log_tol) { ++it; break; }
        if (e.f > 0.0) u_lo = u; else u_hi = u;

        // Halley: u_new = u - 2 f f' / (2 f'^2 - f f'')
        double u_new;
        const double denom = 2.0 * e.f_prime * e.f_prime - e.f * e.f_prime2;
        if (e.f_prime < 0.0 && std::isfinite(denom) && denom != 0.0) {
            u_new = u - (2.0 * e.f * e.f_prime) / denom;
            if (!(u_new > u_lo && u_new < u_hi)) {
                u_new = 0.5 * (u_lo + u_hi);
            }
        } else {
            u_new = 0.5 * (u_lo + u_hi);
        }
        if (std::fabs(u_new - u) <= opts.log_tol * (std::fabs(u) + 1.0)) {
            u = u_new;
            ++it;
            break;
        }
        u = u_new;
    }

    r.iters = it;
    r.extra_iters = extra;
    r.converged = std::fabs(eval(u).f) <= opts.log_tol;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A6: Newton in delta = K - q_max (no transcendentals in inner loop) ──
// Solves g(delta) = sum w_i / (delta + Delta_i) - N = 0 directly.
// Inner kernel: M divisions + M (mul + add). No log/exp.
// Init: delta = c_puct / (3 N^(1/3)) (Δ_i=0 closed form, upper bound).
// Safeguard: bracket [delta_lo, delta_hi] maintained from sign of g.
//
// Tolerance: |g(delta)| <= log_tol * N (since g is in raw-N units, scale
// by N to make the threshold comparable to A3/A5's |f(u)| <= log_tol).

inline Result solve_newton_delta(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    const double N = std::exp(static_cast<double>(inst.depth));
    const double w_factor = static_cast<double>(inst.c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;
    const double tol_g = opts.log_tol * N;

    auto eval = [&](double d) {
        // Returns (g, g').
        double s = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            if (inst_d.P[i] <= 0.0) continue;
            const double di = d + inst_d.Delta[i];
            const double inv = 1.0 / di;
            const double t = w_factor * inst_d.P[i] * inv;
            s  += t;
            s2 += t * inv;            // = w_i / di^2
        }
        return std::pair{s - N, -s2};
    };

    double d_hi;
    if (opts.has_warm_u) {
        d_hi = std::exp(opts.warm_u);
    } else {
        // Closed form from Δ_i = 0 (provable upper bound on δ*).
        d_hi = static_cast<double>(inst.c_puct) / (3.0 * std::cbrt(N));
    }
    int extra = 0;
    while (extra < 64) {
        auto [g_hi, _] = eval(d_hi);
        (void)_;
        if (g_hi <= 0.0) break;          // delta_hi big enough
        d_hi *= 2.0;
        ++extra;
    }
    double d_lo = d_hi * 0.5;
    while (extra < 256) {
        auto [g_lo, _] = eval(d_lo);
        (void)_;
        if (g_lo >= 0.0) break;
        d_lo *= 0.5;
        if (d_lo < 1e-300) { d_lo = 1e-300; break; }
        ++extra;
    }

    double d = (opts.has_warm_u && d_hi >= std::exp(opts.warm_u)
                                && d_lo <= std::exp(opts.warm_u))
             ? std::exp(opts.warm_u)
             : 0.5 * (d_lo + d_hi);

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        auto [g, gp] = eval(d);
        if (std::fabs(g) <= tol_g) { ++it; break; }
        if (g > 0.0) d_lo = d; else d_hi = d;

        double d_new;
        if (gp < 0.0 && std::isfinite(gp)) {
            d_new = d - g / gp;
            if (!(d_new > d_lo && d_new < d_hi)) {
                d_new = 0.5 * (d_lo + d_hi);
            }
        } else {
            d_new = 0.5 * (d_lo + d_hi);
        }
        if (std::fabs(d_new - d) <= opts.log_tol * (d + 1e-30)) {
            d = d_new;
            ++it;
            break;
        }
        d = d_new;
    }

    const double u = std::log(d);
    r.iters = it;
    r.extra_iters = extra;
    r.converged = it < opts.max_iters;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A6b: Newton-in-delta with d_hi init (no bracket walk-down) ──────────
// Key observation: g(δ) = Σ w_i/(δ+Δ_i) - N is strictly convex and
// decreasing on (0, ∞) (g' < 0, g'' > 0). Starting Newton from any
// δ_0 > δ* (where g(δ_0) < 0), the Newton step is
//     δ_new = δ_0 - g(δ_0)/g'(δ_0)   with g, g' both negative
// so δ_new < δ_0; convexity guarantees δ_new > δ* (the tangent at δ_0
// lies below the chord to (δ*, 0)). Therefore Newton from δ_hi
// converges *monotonically from above*, never overshoots below δ*, and
// needs no lower-bracket safeguard. The proven upper bound
//     δ_hi = c_puct / (3 N^{1/3})           (Δ_i = 0 closed form)
// is a valid starting point.
//
// Inner kernel identical to A6: M divisions + M (mul + add) per iter,
// no transcendentals.

inline Result solve_newton_delta_hi(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    const double N = std::exp(static_cast<double>(inst.depth));
    const double w_factor = static_cast<double>(inst.c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;
    const double tol_g = opts.log_tol * N;

    auto eval = [&](double d) {
        double s = 0.0, s2 = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            if (inst_d.P[i] <= 0.0) continue;
            const double di = d + inst_d.Delta[i];
            const double inv = 1.0 / di;
            const double t = w_factor * inst_d.P[i] * inv;
            s  += t;
            s2 += t * inv;
        }
        return std::pair{s - N, -s2};
    };

    const double d_hi_proven = static_cast<double>(inst.c_puct) / (3.0 * std::cbrt(N));

    double d;
    if (opts.has_warm_u) {
        // Warm start: respect the warm value but clamp to the proven upper
        // bound so we keep the monotone-from-above guarantee.
        d = std::min(std::exp(opts.warm_u), d_hi_proven);
        // If warm came in below δ*, the first Newton step could overshoot
        // upward into the safe region; not catastrophic but adds an iter.
    } else {
        d = d_hi_proven;
    }

    // Hard floor only — convexity guarantees Newton stays >= δ*.
    int it = 0;
    for (; it < opts.max_iters; ++it) {
        auto [g, gp] = eval(d);
        if (std::fabs(g) <= tol_g) { ++it; break; }
        if (gp >= 0.0 || !std::isfinite(gp)) {
            // Shouldn't happen; defensive halve.
            d *= 0.5;
            continue;
        }
        double d_new = d - g / gp;
        if (d_new <= 0.0) d_new = d * 0.5;
        if (std::fabs(d_new - d) <= opts.log_tol * (d + 1e-30)) {
            d = d_new;
            ++it;
            break;
        }
        d = d_new;
    }

    const double u = std::log(d);
    r.iters = it;
    r.extra_iters = 0;
    r.converged = it < opts.max_iters;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── A6c: Halley in delta (cubic, no transcendentals in inner loop) ─────
// Same monotone-from-above setup as A6b, but Halley step:
//     δ_new = δ - 2 g g' / (2 g'^2 - g g'')
// per-iter cost: M (1 div + 3 mul + 2 add) for g, g', g''. ~1.5x A6b's
// per-iter cost; if it saves ≥1 iter on average, it wins.

inline Result solve_halley_delta_hi(const Instance& inst, const Options& opts) {
    Result r{};
    const std::size_t M = inst.P.size();
    if (M == 0) { r.converged = true; return r; }

    auto inst_d = lift(inst);
    const double N = std::exp(static_cast<double>(inst.depth));
    const double w_factor = static_cast<double>(inst.c_puct)
                          * std::pow(N, 2.0/3.0) / 3.0;
    const double tol_g = opts.log_tol * N;

    auto eval = [&](double d) {
        double s = 0.0, s2 = 0.0, s3 = 0.0;
        for (std::size_t i = 0; i < M; ++i) {
            if (inst_d.P[i] <= 0.0) continue;
            const double di = d + inst_d.Delta[i];
            const double inv = 1.0 / di;
            const double t = w_factor * inst_d.P[i] * inv;
            s  += t;
            s2 += t * inv;
            s3 += t * inv * inv;
        }
        // g, g', g''
        return std::tuple{s - N, -s2, 2.0 * s3};
    };

    const double d_hi_proven = static_cast<double>(inst.c_puct) / (3.0 * std::cbrt(N));
    double d = opts.has_warm_u
             ? std::min(std::exp(opts.warm_u), d_hi_proven)
             : d_hi_proven;

    int it = 0;
    for (; it < opts.max_iters; ++it) {
        auto [g, gp, gpp] = eval(d);
        if (std::fabs(g) <= tol_g) { ++it; break; }
        const double denom = 2.0 * gp * gp - g * gpp;
        double d_new;
        if (gp < 0.0 && std::isfinite(denom) && denom != 0.0) {
            d_new = d - (2.0 * g * gp) / denom;
            if (d_new <= 0.0) d_new = d * 0.5;
        } else {
            d_new = d * 0.5;
        }
        if (std::fabs(d_new - d) <= opts.log_tol * (d + 1e-30)) {
            d = d_new;
            ++it;
            break;
        }
        d = d_new;
    }

    const double u = std::log(d);
    r.iters = it;
    r.extra_iters = 0;
    r.converged = it < opts.max_iters;
    finalize(inst, inst_d.P, inst_d.Delta, u, r);
    return r;
}

// ─── Solver registry ─────────────────────────────────────────────────────

struct Solver {
    const char* name;
    Result (*solve)(const Instance&, const Options&);
};

inline const std::vector<Solver>& registry() {
    static const std::vector<Solver> r = {
        {"A1_bisection_K",   &solve_bisection_K},
        {"A1b_bisection_K_e",&solve_bisection_K_early},
        {"A2_bisection_u",   &solve_bisection_u},
        {"A3_newton_u_safe", &solve_newton_u},
        {"A5_halley_u_safe", &solve_halley_u},
        {"A6_newton_delta",  &solve_newton_delta},
        {"A6b_newton_d_hi",  &solve_newton_delta_hi},
        {"A6c_halley_d_hi",  &solve_halley_delta_hi},
    };
    return r;
}

}  // namespace catgpt::lks::gym

#endif  // CATGPT_ENGINE_LKS_COMPUTE_ALLOC_GYM_HPP
