"""MCTS configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for MCTS search.

    Attributes:
        c_puct: Exploration constant for PUCT formula. Higher values encourage
            more exploration of less-visited moves. Typical values: 1.0-2.5.
            Leela Chess Zero uses ~1.75.
        num_simulations: Number of MCTS simulations to run per move.
            More simulations = stronger play but slower.
        fpu_value: First Play Urgency - the Q value assigned to unvisited nodes.
            -1.0 (default) means unvisited nodes are treated as losses,
            encouraging exploration of all moves before deep exploitation.
    """

    c_puct: float = 1.75
    num_simulations: int = 800
    fpu_value: float = -1.0
