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
 * Solver: bracketed Halley in delta = K - q_max. The bracket is
 *   [lo, hi] = [0, c_puct / (3 N^(1/3))]
 * (lo is exclusive; hi is the Delta_i = 0 closed form, a proven upper
 * bound on delta*). Each iteration evaluates g at the current delta,
 * tightens the bracket by the sign of g (g is strictly monotone
 * decreasing), and either accepts the Halley step
 *   delta_new = delta - 2 g g' / (2 g'^2 - g g'')
 * if it lands strictly inside the bracket, or falls back to bisection
 * (midpoint). This preserves Halley's cubic convergence on well-
 * conditioned distributions while guaranteeing convergence in the
 * pathological "one large-prior child far from q_max + the rest
 * clustered at q_max" case, where the prior monotone-from-above-only
 * iteration would halve d unboundedly toward zero and overshoot the
 * true delta* by 4+ orders of magnitude. Inner kernel: M (1 div + 3 mul
 * + 2 add) per iter, no transcendentals.
 *
 * The "A6c" winner from the compute_alloc_gym referenced in the design
 * commit (e13d41c4) was the unbracketed Halley variant; the bracket and
 * bisection fallback were added later after the dual collapse was
 * traced to that path. Per-call iteration count typically <= 4 at
 * 1e-6 log-tol on healthy inputs; capped at 32 defensively (each
 * bisection halves the bracket, so 32 iterations is 1e-9 of the
 * starting width).
 *
 * Lives next to lks_search.hpp because Plan / Mode are the same types
 * the descent uses end-to-end. The solver only reads Plan.P and Plan.Q
 * and writes Plan.alloc; it never touches Plan.mode / .depth.
 */

#ifndef CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP
#define CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace catgpt::lks::detail {

/**
 * Per-child plan row built during the descent's pass 1 (classification),
 * mutated by the Halley allocator (fills `alloc`), then — for forked
 * children — overwritten in place by the child's `recursive_search`
 * (which writes back `Q` / `depth` through its `Plan* out` argument)
 * before the parent's rollup runs.
 *
 * Mode semantics:
 *   - Expanded: a usable `Q` is available for the rollup. Sources are
 *     a TT-hit child, a position-only terminal (draw / loss / win), a
 *     path-dependent draw (3-fold / 50-move along this path), or a
 *     just-recursed child whose `recursive_search` wrote its rolled-up
 *     `(Q, depth)` back through the parent's `&plans[i]`. Pass 2
 *     pre-marks Expanded immediately before forking; abort paths leave
 *     the row stale but the parent's own should_abort() check skips
 *     rollup before any stale row is observed.
 *   - Unexpanded: no real Q yet; `Q` is the FPU stand-in
 *     (Q_eff_parent_pov = parent_Q - fpu_reduction * sqrt(cumulative_P),
 *      stored as -Q_eff_parent_pov in child-STM convention). Used only
 *     to bias the Halley allocation. Unexpanded children expand only
 *     via iter-0 force expansion in pass 2; otherwise they stay
 *     Unexpanded and are dropped from the final rollup.
 *
 * `depth` is the per-child "current effort" gate consulted by pass 2:
 *   - Expanded TT entry → child's stored max_depth
 *   - Terminal / path-dep draw → +infinity (never re-recurses)
 *   - Unexpanded → -infinity (force-expand only on iter 0)
 */
enum class Mode : uint8_t { Expanded, Unexpanded };

struct Plan {
    Mode  mode;
    float P;
    float Q;
    float depth;
    float alloc;
};

/**
 * Bracketed-Halley dual solve. Reads plans[i].Q / plans[i].P and writes
 * plans[i].alloc = log N_i in place. M == 0 is a no-op.
 *
 * Bracketing leverages the structural properties of g(d):
 *   - g is continuous and strictly monotone decreasing on (0, infinity),
 *     since g'(d) = -sum w_i / (d + Δ_i)^2 < 0;
 *   - g(0+) > 0 (>= 0 when no Δ_i is zero) and g(+inf) = -N < 0, so the
 *     unique root d* lies in (0, +inf);
 *   - delta_hi = c_puct / (3 N^(1/3)) is a proven upper bound on d*
 *     (the Δ_i = 0 closed form; LHS is strictly decreasing in K, and any
 *     positive Δ_i only shrinks the sum).
 * So [lo=0, hi=delta_hi] is a guaranteed sign-changing bracket. Each
 * iteration evaluates g at the current d, tightens [lo, hi] by the sign,
 * and accepts the Halley step ONLY if it lands strictly inside the
 * bracket; otherwise it falls back to bisection (midpoint of [lo, hi]).
 *
 * This is the cubic-when-it-works / bisection-when-it-doesn't pattern from
 * boost::math::tools::halley_iterate. It preserves Halley's fast
 * convergence on well-conditioned distributions while guaranteeing the
 * pathological case (one large-prior child far from q_max + the rest
 * clustered at q_max, where denom = 2 g'^2 - g g'' collapses or flips
 * sign and the old fallback halved d unboundedly toward zero) converges
 * within log2(delta_hi / eps) ~= 30 bisection steps even if Halley is
 * rejected every iteration.
 *
 * Numerics: inner loop is double-precision; cast to float only at the
 * final `alloc` write. Convergence: |g(delta)| <= 1e-6 * N or bracket
 * width below 1e-12 relative to hi.
 *
 * Debug-only post-condition: when the solver converges, sum(exp(alloc_i))
 * must equal N within a generous relative tolerance. NDEBUG-gated to
 * keep the release-build hot path free of the extra exp() loop.
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

    // Guaranteed sign-changing bracket. lo=0 is exclusive (g may be +inf
    // there when some Δ_i == 0); we never evaluate at lo. hi is the
    // delta_hi upper bound, where g(hi) <= 0 by the proof above.
    double lo = 0.0;
    double hi = static_cast<double>(c_puct) / (3.0 * cbrt_N);
    double d  = hi;

    constexpr int kMaxIters = 32;
    for (int it = 0; it < kMaxIters; ++it) {
        double g, gp, gpp;
        eval(d, g, gp, gpp);
        if (std::fabs(g) <= tol_g) break;

        // Tighten bracket by sign of g (g is strictly monotone decreasing,
        // so g > 0 means root is to the right of d, g < 0 means to the left).
        if (g > 0.0) lo = d;
        else         hi = d;
        if (hi - lo <= 1.0e-12 * (hi + 1.0e-30)) break;

        // Halley step: cubic-convergence when well-conditioned (denom > 0,
        // gp < 0 always). Accept only if it lands strictly inside the
        // bracket; otherwise fall back to bisection. This is what catches
        // the pathological "one dominant child + cluster at q_max" case
        // where denom collapses or the step overshoots below 0.
        const double denom = 2.0 * gp * gp - g * gpp;
        double d_new;
        if (gp < 0.0 && std::isfinite(denom) && denom > 0.0) {
            d_new = d - (2.0 * g * gp) / denom;
            if (!(d_new > lo && d_new < hi)) {
                d_new = 0.5 * (lo + hi);
            }
        } else {
            d_new = 0.5 * (lo + hi);
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

#ifndef NDEBUG
    // Post-condition: sum exp(alloc_i) ~= N. Tolerance is generous (1%
    // relative + 1.0 absolute) — anything that materially exceeds this
    // means the dual solve failed and downstream depths will be unbounded.
    // The 1.0 absolute slack handles the small-N regime where 1% of N is
    // sub-floating-point noise.
    double sum_ni = 0.0;
    for (int i = 0; i < M; ++i) {
        const double a = static_cast<double>(plans[i].alloc);
        if (std::isfinite(a)) sum_ni += std::exp(a);
    }
    assert(std::fabs(sum_ni - N) <= 1.0e-2 * N + 1.0
           && "compute_log_allocations: sum exp(alloc_i) deviates from N "
              "by more than 1% — Halley/bisection solver failed to converge");
#endif
}

}  // namespace catgpt::lks::detail

#endif  // CATGPT_ENGINE_LKS_COMPUTE_ALLOCATIONS_HPP
