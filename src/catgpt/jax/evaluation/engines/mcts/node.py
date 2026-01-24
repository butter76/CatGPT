"""MCTS node data structure."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class MCTSNode:
    """A node in the MCTS search tree.

    Each node represents a position reached by playing `move` from the parent.
    The root node has move=None and parent=None.

    Statistics:
        N: Visit count - how many times this node was visited during search.
        W: Total value sum - accumulated value from all visits.
        P: Prior probability - from policy network, used in PUCT formula.

    The mean value Q = W/N represents the expected outcome from this position.
    """

    parent: MCTSNode | None = None
    move: chess.Move | None = None
    children: dict[chess.Move, MCTSNode] = field(default_factory=dict)

    # Statistics
    N: int = 0
    W: float = 0.0
    P: float = 0.0

    # Terminal state info
    is_terminal: bool = False
    terminal_value: float | None = None  # -1=loss, 0=draw, 1=win (from this node's side-to-move)

    @property
    def Q(self) -> float:
        """Mean value (expected outcome from this position).

        Returns 0.0 for unvisited nodes.
        """
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (children created)."""
        return len(self.children) > 0

    def best_child_by_visits(self) -> tuple[chess.Move, MCTSNode] | None:
        """Return the child with the highest visit count.

        Returns:
            Tuple of (move, child_node) or None if no children.
        """
        if not self.children:
            return None
        return max(self.children.items(), key=lambda kv: kv[1].N)

    def get_visit_distribution(self) -> dict[chess.Move, float]:
        """Get normalized visit counts (policy after search).

        Returns:
            Dictionary mapping moves to their visit proportions.
        """
        if not self.children:
            return {}

        total_visits = sum(child.N for child in self.children.values())
        if total_visits == 0:
            return {move: 0.0 for move in self.children}

        return {move: child.N / total_visits for move, child in self.children.items()}

    def get_pv(self, max_depth: int = 10) -> list[chess.Move]:
        """Get principal variation (most visited path from this node).

        Args:
            max_depth: Maximum depth to traverse.

        Returns:
            List of moves forming the principal variation.
        """
        pv: list[chess.Move] = []
        node = self

        for _ in range(max_depth):
            best = node.best_child_by_visits()
            if best is None:
                break
            move, child = best
            pv.append(move)
            node = child

        return pv

    def __repr__(self) -> str:
        move_str = self.move.uci() if self.move else "root"
        return f"MCTSNode({move_str}, N={self.N}, Q={self.Q:.3f}, P={self.P:.3f})"
