/**
 * Per-child PUCT allocation for the LKS descent.
 *
 * Problem (recap, mirroring coroutine_search.hpp::compute_allocations):
 *
 *   Given M children with priors P_i (sum 1) and child-Q values Q_i in
 *   [-1, 1] (TT-stored, child-STM convention), total log-budget d = log N,
 *   and constant c_puct, find the dual K such that
 *
 *     sum_i N_i = N,  N_i = c_puct * P_i * N^(2/3) / [3 (K - q_i)]
 *
 *   with q_i = -Q_i (parent-loss POV). K is unique on (q_max, +inf) — LHS
 *   is strictly decreasing in K. Output is log N_i per child.
 *
 * Solver: Halley in delta = K - q_max, monotone-from-above starting at
 *   delta_hi = c_puct / (3 N^(1/3))
 * (the Delta_i = 0 closed form, which is a proven upper bound on delta*).
 * g(delta) = sum_i w_i / (delta + Delta_i) - N is strictly convex and
 * decreasing on (0, infinity), so the Halley step
 *   delta_new = delta - 2 g g' / (2 g'^2 - g g'')
 * stays above delta* throughout; no bracket safeguard needed. Inner
 * kernel: M (1 div + 3 mul + 2 add) per iter, no transcendentals.
 *
 * This is the "A6c" winner from the compute_alloc_gym referenced in the
 * design commit (e13d41c4). Per-call iteration count typically <= 4 at
 * 1e-6 log-tol; capped at 16 defensively.
 *
 * Lives next to lks_search.hpp because Plan / Mode are the same types
 * the descent uses end-to-end. The solver only reads Plan.P / Plan.Q
 * and writes Plan.alloc; it never touches Plan.mode / .depth / .child_*.
 */

#ifndef CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP
#define CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP

#include <cmath>
#include <cstdint>
#include <limits>

namespace catgpt::lks::detail {

/**
 * Per-child plan row built during the descent's pass 1 (classification),
 * mutated by the Halley allocator (fills `alloc`), then by pass 3
 * (re-read TT updates `Q`/`depth`/`mode` for forked children), and
 * consumed by the rollup.
 *
 * Mode semantics:
 *   - Expanded: a usable `Q` is available for the rollup. Sources are
 *     a TT-hit child, a position-only terminal (draw / loss / win), or
 *     a path-dependent draw (3-fold / 50-move along this path). After
 *     pass 3, any unexpanded child we successfully recursed on is
 *     promoted to Expanded.
 *   - Unexpanded: no real Q yet; `Q` is the FPU stand-in
 *     (Q_eff_parent_pov = parent_Q - fpu_reduction * sqrt(cumulative_P),
 *      stored as -Q_eff_parent_pov in child-STM convention). Used only
 *     to bias the Halley allocation. If the alloc never exceeds the
 *     depth gate, the child stays Unexpanded and is dropped from the
 *     final rollup.
 *
 * `depth` is the per-child "current effort" gate consulted by pass 2:
 *   - Expanded TT entry → child's stored max_depth
 *   - Terminal / path-dep draw → +infinity (never re-recurses)
 *   - Unexpanded → depth_floor (any alloc > floor triggers expansion)
 *
 * `child_key` / `child_sec` are cached so the post-join re-read doesn't
 * have to re-derive the child board. Zeroed for terminal / path-dep
 * plans where no recursion will ever happen.
 */
enum class Mode : uint8_t { Expanded, Unexpanded };

struct Plan {
    Mode     mode;
    float    P;
    float    Q;
    float    depth;
    float    alloc;
    uint64_t child_key;
    uint32_t child_sec;
};

/**
 * Halley-in-delta dual solve. Reads plans[i].P / plans[i].Q, writes
 * plans[i].alloc = log N_i in place. M == 0 is a no-op.
 *
 * Numerics: inner loop is double-precision; cast to float only at the
 * final `alloc` write. Convergence: |g(delta)| <= 1e-6 * N or relative
 * step <= 1e-6.
 */
inline void compute_log_allocations(Plan* plans, int M,
                                    float depth, float c_puct) noexcept
{
    if (M <= 0) return;

    // q_max in parent-loss POV is -min(Q_i).
    double qmax_neg_q = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < M; ++i) {
        const double neg_q = -static_cast<double>(plans[i].Q);
        if (neg_q > qmax_neg_q) qmax_neg_q = neg_q;
    }

    const double N        = std::exp(static_cast<double>(depth));
    const double cbrt_N   = std::cbrt(N);
    const double w_factor = static_cast<double>(c_puct)
                          * cbrt_N * cbrt_N / 3.0;  // c_puct*N^(2/3)/3
    const double tol_g    = 1.0e-6 * N;
    const double log_w    = std::log(static_cast<double>(c_puct) / 3.0);
    const double bias     = (2.0 / 3.0) * static_cast<double>(depth) + log_w;

    // g(d), g'(d), g''(d) at delta = d. Δ_i = qmax_neg_q - (-Q_i) >= 0.
    auto eval = [plans, M, w_factor, qmax_neg_q, N]
                (double d, double& g, double& gp, double& gpp) {
        double s = 0.0, s2 = 0.0, s3 = 0.0;
        for (int i = 0; i < M; ++i) {
            const double p = static_cast<double>(plans[i].P);
            if (p <= 0.0) continue;
            const double Delta_i = qmax_neg_q - (-static_cast<double>(plans[i].Q));
            const double di      = d + Delta_i;
            const double inv     = 1.0 / di;
            const double t       = w_factor * p * inv;
            s  += t;
            s2 += t * inv;
            s3 += t * inv * inv;
        }
        g   = s - N;
        gp  = -s2;
        gpp = 2.0 * s3;
    };

    // delta_hi = c_puct / (3 N^(1/3)): proven upper bound from Δ_i = 0.
    double d = static_cast<double>(c_puct) / (3.0 * cbrt_N);

    constexpr int kMaxIters = 16;
    for (int it = 0; it < kMaxIters; ++it) {
        double g, gp, gpp;
        eval(d, g, gp, gpp);
        if (std::fabs(g) <= tol_g) break;

        const double denom = 2.0 * gp * gp - g * gpp;
        double d_new;
        if (gp < 0.0 && std::isfinite(denom) && denom != 0.0) {
            d_new = d - (2.0 * g * gp) / denom;
            if (d_new <= 0.0) d_new = d * 0.5;
        } else {
            // Shouldn't happen on the monotone-from-above path; defensive halve.
            d_new = d * 0.5;
        }
        if (std::fabs(d_new - d) <= 1.0e-6 * (d + 1.0e-30)) {
            d = d_new;
            break;
        }
        d = d_new;
    }

    // log N_i = log P_i + 2d/3 + log(c_puct/3) - log(e^u + Δ_i),
    // and since u = log(d), e^u + Δ_i = d + Δ_i directly.
    for (int i = 0; i < M; ++i) {
        const double p = static_cast<double>(plans[i].P);
        if (p <= 0.0) {
            plans[i].alloc = -std::numeric_limits<float>::infinity();
            continue;
        }
        const double Delta_i = qmax_neg_q - (-static_cast<double>(plans[i].Q));
        const double log_ni  = std::log(p) - std::log(d + Delta_i) + bias;
        plans[i].alloc       = static_cast<float>(log_ni);
    }
}

}  // namespace catgpt::lks::detail

#endif  // CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP
