"""Sequential Probability Ratio Test (SPRT) calculator for chess engine testing.

SPRT is the standard method for determining if a chess engine change provides
a statistically significant strength improvement. It tests two hypotheses:

- H0: The Elo difference is elo0 (typically 0, meaning no improvement)
- H1: The Elo difference is elo1 (typically 5-10, meaning meaningful improvement)

The test continues until the Log Likelihood Ratio (LLR) crosses either the
upper bound (accept H1) or lower bound (accept H0).

References:
- https://www.chessprogramming.org/Sequential_Probability_Ratio_Test
- https://tests.stockfishchess.org/sprt_calc
"""

import math
from dataclasses import dataclass
from enum import Enum


class SPRTStatus(Enum):
    """Status of an SPRT test."""

    CONTINUE = "continue"  # Test not yet conclusive
    H0_ACCEPTED = "H0_accepted"  # Null hypothesis accepted (no improvement)
    H1_ACCEPTED = "H1_accepted"  # Alt hypothesis accepted (improvement confirmed)


@dataclass
class SPRTResult:
    """Result of an SPRT calculation after a set of games."""

    # Current log-likelihood ratio
    llr: float

    # LLR bounds for decision making
    lower_bound: float  # Cross this → accept H0
    upper_bound: float  # Cross this → accept H1

    # Game statistics
    games: int
    wins: int  # Engine A wins
    losses: int  # Engine A losses
    draws: int

    # Derived statistics
    score: float  # Win rate: (wins + draws/2) / games
    elo_estimate: float  # Estimated Elo difference
    elo_error: float  # 95% confidence interval

    # Test status
    status: SPRTStatus

    @property
    def win_rate(self) -> float:
        """Win rate (W/games)."""
        return self.wins / self.games if self.games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw rate (D/games)."""
        return self.draws / self.games if self.games > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        """Loss rate (L/games)."""
        return self.losses / self.games if self.games > 0 else 0.0


class SPRTCalculator:
    """SPRT calculator using the logistic Elo model.

    Example:
        sprt = SPRTCalculator(elo0=0, elo1=10, alpha=0.05, beta=0.05)

        # After each game pair
        result = sprt.update(wins=50, losses=45, draws=105)

        if result.status == SPRTStatus.H1_ACCEPTED:
            print("Engine A is stronger!")
        elif result.status == SPRTStatus.H0_ACCEPTED:
            print("No significant difference")
    """

    def __init__(
        self,
        elo0: float = 0.0,
        elo1: float = 10.0,
        alpha: float = 0.05,
        beta: float = 0.05,
    ) -> None:
        """Initialize SPRT calculator.

        Args:
            elo0: Null hypothesis Elo difference (typically 0).
            elo1: Alternative hypothesis Elo difference (typically 5-10).
            alpha: Type I error rate (false positive). Default 0.05 = 5%.
            beta: Type II error rate (false negative). Default 0.05 = 5%.
        """
        if elo0 >= elo1:
            raise ValueError(f"elo0 ({elo0}) must be less than elo1 ({elo1})")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if not (0 < beta < 1):
            raise ValueError(f"beta must be in (0, 1), got {beta}")

        self.elo0 = elo0
        self.elo1 = elo1
        self.alpha = alpha
        self.beta = beta

        # Calculate LLR bounds using Wald's formula
        # Lower bound: log(β / (1 - α))
        # Upper bound: log((1 - β) / α)
        self.lower_bound = math.log(beta / (1 - alpha))
        self.upper_bound = math.log((1 - beta) / alpha)

    @staticmethod
    def elo_to_score(elo: float) -> float:
        """Convert Elo difference to expected score using logistic model.

        The logistic Elo model: score = 1 / (1 + 10^(-elo/400))

        Args:
            elo: Elo difference (positive = player A stronger).

        Returns:
            Expected score in [0, 1].
        """
        return 1.0 / (1.0 + math.pow(10.0, -elo / 400.0))

    @staticmethod
    def score_to_elo(score: float) -> float:
        """Convert score to Elo difference using logistic model.

        Inverse of elo_to_score: elo = -400 * log10((1/score) - 1)

        Args:
            score: Score in (0, 1).

        Returns:
            Elo difference.
        """
        if score <= 0.0:
            return -1000.0  # Cap at -1000 Elo
        if score >= 1.0:
            return 1000.0  # Cap at +1000 Elo
        return -400.0 * math.log10(1.0 / score - 1.0)

    def _calculate_llr(self, wins: int, losses: int, draws: int) -> float:
        """Calculate Log Likelihood Ratio for current game results.

        Uses the trinomial model (wins, draws, losses) with logistic Elo.

        The LLR measures how much more likely the observed results are under
        H1 (elo = elo1) compared to H0 (elo = elo0).

        Args:
            wins: Number of wins for Engine A.
            losses: Number of losses for Engine A.
            draws: Number of draws.

        Returns:
            Log Likelihood Ratio value.
        """
        total = wins + losses + draws
        if total == 0:
            return 0.0

        # Observed score
        score = (wins + draws / 2.0) / total

        # Expected scores under each hypothesis
        score0 = self.elo_to_score(self.elo0)
        score1 = self.elo_to_score(self.elo1)

        # For the trinomial model, we need win/draw/loss probabilities
        # We use the Davidson model approximation where draw rate is estimated
        # from the score: draw_rate ≈ 2 * sqrt(score * (1 - score)) * draw_elo_factor
        #
        # Simplified approach: use the BayesElo/pentanomial approximation
        # LLR ≈ n * (score - score0) * log(score1/score0) + n * (1 - score - (1 - score0)) * log((1-score1)/(1-score0))
        #
        # Even simpler: use the binomial approximation which works well in practice
        # LLR = W*log(p1/p0) + L*log((1-p1)/(1-p0)) where p = win probability

        # Binomial approximation (treating draws as 0.5 wins + 0.5 losses)
        # This is the approach used by most SPRT implementations
        w = wins + draws / 2.0  # Effective wins
        l = losses + draws / 2.0  # Effective losses

        # Avoid log(0)
        eps = 1e-10
        score0 = max(eps, min(1 - eps, score0))
        score1 = max(eps, min(1 - eps, score1))

        # LLR = sum over games of log(P(result|H1) / P(result|H0))
        # For binomial: LLR = W*log(p1/p0) + L*log((1-p1)/(1-p0))
        llr = w * math.log(score1 / score0) + l * math.log((1 - score1) / (1 - score0))

        return llr

    def _calculate_elo_error(self, wins: int, losses: int, draws: int) -> float:
        """Calculate 95% confidence interval for Elo estimate.

        Uses the normal approximation to the binomial distribution.

        Args:
            wins: Number of wins.
            losses: Number of losses.
            draws: Number of draws.

        Returns:
            95% confidence interval half-width in Elo.
        """
        total = wins + losses + draws
        if total < 2:
            return float("inf")

        score = (wins + draws / 2.0) / total

        # Variance of score estimate (binomial)
        variance = score * (1 - score) / total

        # Standard error
        se = math.sqrt(variance) if variance > 0 else 0.0

        # 95% CI is approximately ±1.96 * se
        # Convert to Elo: derivative of score_to_elo at current score
        if 0.001 < score < 0.999:
            # d(elo)/d(score) = 400 / (score * (1-score) * ln(10))
            deriv = 400.0 / (score * (1 - score) * math.log(10))
            elo_error = 1.96 * se * abs(deriv)
        else:
            elo_error = float("inf")

        return elo_error

    def update(self, wins: int, losses: int, draws: int) -> SPRTResult:
        """Update SPRT with current game statistics.

        Args:
            wins: Total wins for Engine A.
            losses: Total losses for Engine A.
            draws: Total draws.

        Returns:
            SPRTResult with current LLR, bounds, and test status.
        """
        total = wins + losses + draws

        # Calculate LLR
        llr = self._calculate_llr(wins, losses, draws)

        # Calculate score and Elo estimate
        if total > 0:
            score = (wins + draws / 2.0) / total
            elo_estimate = self.score_to_elo(score)
        else:
            score = 0.5
            elo_estimate = 0.0

        # Calculate Elo error (95% CI)
        elo_error = self._calculate_elo_error(wins, losses, draws)

        # Determine test status
        if llr >= self.upper_bound:
            status = SPRTStatus.H1_ACCEPTED
        elif llr <= self.lower_bound:
            status = SPRTStatus.H0_ACCEPTED
        else:
            status = SPRTStatus.CONTINUE

        return SPRTResult(
            llr=llr,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            games=total,
            wins=wins,
            losses=losses,
            draws=draws,
            score=score,
            elo_estimate=elo_estimate,
            elo_error=elo_error,
            status=status,
        )

    def games_estimate(self, true_elo: float | None = None) -> int:
        """Estimate number of games needed for the test to conclude.

        This is a rough estimate based on the SPRT formula. The actual number
        of games depends on the true Elo difference.

        Args:
            true_elo: Assumed true Elo difference. If None, uses midpoint of bounds.

        Returns:
            Estimated number of games.
        """
        if true_elo is None:
            true_elo = (self.elo0 + self.elo1) / 2

        # Expected score under true Elo
        score = self.elo_to_score(true_elo)
        score0 = self.elo_to_score(self.elo0)
        score1 = self.elo_to_score(self.elo1)

        # Information per game (KL divergence approximation)
        # This is the expected log-likelihood ratio per game
        info_per_game = score * math.log(score1 / score0) + (1 - score) * math.log(
            (1 - score1) / (1 - score0)
        )

        if abs(info_per_game) < 1e-10:
            return 100000  # Very long test expected

        # Expected games to reach a bound
        # Average of upper and lower bound magnitudes
        avg_bound = (self.upper_bound - self.lower_bound) / 2
        estimated = int(abs(avg_bound / info_per_game))

        return max(100, estimated)  # At least 100 games
