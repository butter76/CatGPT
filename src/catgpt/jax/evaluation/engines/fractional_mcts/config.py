"""Fractional MCTS configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FractionalMCTSConfig:
    """Configuration for Fractional MCTS search with iterative deepening.

    This variant uses fractional visit counts and iterative deepening rather than
    traditional simulations. The search allocates budget N across children using
    the PUCT formula solved for equal "urgency" K.

    Attributes:
        c_puct: Exploration constant for PUCT formula. Higher values encourage
            more exploration of less-visited moves. Typical values: 1.0-2.5.
        policy_coverage_threshold: Fraction of policy mass that determines the
            "limit" for expansion. If N < limit, we don't expand further.
        min_total_evals: Minimum total GPU evaluations before stopping search.
            Search continues until an iteration completes with total >= this.
        initial_budget: Starting budget N for iterative deepening.
        budget_multiplier: Factor to multiply N by each iteration.
    """

    c_puct: float = 1.75
    policy_coverage_threshold: float = 0.80
    min_total_evals: int = 400
    initial_budget: float = 1.0
    budget_multiplier: float = 1.2
