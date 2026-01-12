"""Base engine protocol for chess move selection."""

from typing import Protocol

import chess


class Engine(Protocol):
    """Protocol for chess engines that can select moves.

    Engines evaluate chess positions and select the best move according to
    their strategy (e.g., value network, search, policy network).

    All engines should be stateless with respect to game history - they
    receive a board position and return a move without maintaining internal
    game state.
    """

    @property
    def name(self) -> str:
        """Return the name of the engine for logging/display."""
        ...

    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Select the best move in the given position.

        Args:
            board: Current chess position. The engine should not modify this board.

        Returns:
            The selected move, or None if no legal moves exist.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (for engines with memory/caching).

        Most engines are stateless and this is a no-op.
        """
        ...
