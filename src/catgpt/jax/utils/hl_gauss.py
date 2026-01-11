"""HL-Gauss: Histogram Loss with Gaussian targets.

Implementation based on:
"Stop Regressing: Training Value Functions via Classification for Scalable Deep RL"
(Farebrother et al., 2024) - https://arxiv.org/abs/2403.03950

HL-Gauss converts scalar regression targets into soft categorical distributions
using a Gaussian kernel, enabling training with cross-entropy loss instead of MSE.
This approach:
- Reduces overfitting by spreading probability mass to neighboring bins
- Exploits ordinal structure of the regression problem
- Scales better with larger networks
"""

import jax
import jax.numpy as jnp
import jax.scipy.special


def hl_gauss_transform(
    min_value: float = 0.0,
    max_value: float = 1.0,
    num_bins: int = 81,
    sigma_ratio: float = 0.75,
) -> tuple[jax.Array, callable, callable]:
    """Create HL-Gauss transform functions.

    Args:
        min_value: Minimum value of the target range.
        max_value: Maximum value of the target range.
        num_bins: Number of bins for the categorical distribution.
        sigma_ratio: Ratio of sigma to bin_width. The actual sigma is computed as
            sigma = sigma_ratio * bin_width. Default 0.75 spreads mass to ~5 bins.

    Returns:
        Tuple of (support, transform_to_probs, transform_from_probs):
        - support: Array of bin edges (shape: [num_bins + 1])
        - transform_to_probs: Function to convert scalars to distributions
        - transform_from_probs: Function to convert distributions to scalars
    """
    # Bin edges (num_bins + 1 values)
    support = jnp.linspace(min_value, max_value, num_bins + 1, dtype=jnp.float32)

    # Compute sigma from bin width and ratio
    bin_width = (max_value - min_value) / num_bins
    sigma = sigma_ratio * bin_width

    def to_probs(target: jax.Array) -> jax.Array:
        """Convert scalar target(s) to HL-Gauss probability distribution.

        Args:
            target: Scalar target value(s). Can be any shape, the last dimension
                will be expanded for the bin probabilities.

        Returns:
            Probability distribution over bins. Shape: (*target.shape, num_bins)
        """
        # Expand target for broadcasting: (...,) -> (..., 1)
        target_expanded = target[..., None]

        # Compute CDF at each bin edge using the error function
        # erf(x) = 2/sqrt(pi) * integral(0, x, exp(-t^2) dt)
        # CDF of N(mu, sigma) at x = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
        # We use just erf since we normalize anyway
        cdf_evals = jax.scipy.special.erf(
            (support - target_expanded) / (jnp.sqrt(2.0) * sigma)
        )

        # Probability in each bin = CDF(right) - CDF(left)
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]

        # Normalize to ensure probabilities sum to 1
        # (handles edge cases where target is near boundaries)
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = bin_probs / z[..., None]

        return bin_probs

    def from_probs(probs: jax.Array) -> jax.Array:
        """Convert probability distribution to expected scalar value.

        Args:
            probs: Probability distribution over bins. Shape: (..., num_bins)

        Returns:
            Expected value (weighted sum of bin centers). Shape: (...)
        """
        # Bin centers
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers, axis=-1)

    return support, to_probs, from_probs


# Convenience functions with default parameters for win probability (0-1 range)


def transform_to_probs(
    target: jax.Array,
    num_bins: int = 81,
    sigma_ratio: float = 0.75,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> jax.Array:
    """Convert scalar target(s) to HL-Gauss probability distribution.

    Convenience function that creates the transform on-the-fly.
    For repeated use, prefer hl_gauss_transform() to avoid recomputation.

    Args:
        target: Scalar target value(s).
        num_bins: Number of bins for the categorical distribution.
        sigma_ratio: Ratio of sigma to bin_width.
        min_value: Minimum value of the target range.
        max_value: Maximum value of the target range.

    Returns:
        Probability distribution over bins. Shape: (*target.shape, num_bins)
    """
    _, to_probs, _ = hl_gauss_transform(min_value, max_value, num_bins, sigma_ratio)
    return to_probs(target)


def transform_from_probs(
    probs: jax.Array,
    num_bins: int = 81,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> jax.Array:
    """Convert probability distribution to expected scalar value.

    Convenience function that creates the transform on-the-fly.
    For repeated use, prefer hl_gauss_transform() to avoid recomputation.

    Args:
        probs: Probability distribution over bins. Shape: (..., num_bins)
        num_bins: Number of bins (must match probs.shape[-1]).
        min_value: Minimum value of the target range.
        max_value: Maximum value of the target range.

    Returns:
        Expected value. Shape: probs.shape[:-1]
    """
    # sigma_ratio doesn't matter for from_probs, just use default
    _, _, from_probs = hl_gauss_transform(min_value, max_value, num_bins, sigma_ratio=0.75)
    return from_probs(probs)
