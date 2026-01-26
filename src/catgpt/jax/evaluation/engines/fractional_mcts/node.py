"""Fractional MCTS node data structure."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass
class FractionalNode:
    """A node in the Fractional MCTS search tree.

    Each node represents a position reached by playing `move` from the parent.
    The root node has move=None.

    Unlike traditional MCTS nodes which track integer visit counts, this node
    stores:
    - policy_priors: The policy distribution over this node's children (from NN)
    - P: This move's prior probability (from parent's policy)
    - Q: The current value estimate, updated during search

    The invariant is that before recursing into a node, it has been GPU-evaluated
    (or is terminal), so Q and policy_priors are populated.
    """

    move: chess.Move | None = None
    children: dict[chess.Move, FractionalNode] = field(default_factory=dict)

    # Policy priors for THIS node's children (from evaluating this position)
    # Maps legal moves to their prior probabilities (sums to 1.0)
    policy_priors: dict[chess.Move, float] = field(default_factory=dict)

    # Prior probability of this move (from parent's policy output)
    P: float = 0.0

    # Q value: initially from NN evaluation, updated after recursion
    # Range: [-1, 1] where -1=loss, 0=draw, 1=win (from this node's perspective)
    Q: float = 0.0

    # Terminal state info
    is_terminal: bool = False

    def get_limit(self, coverage_threshold: float = 0.80) -> int:
        """Compute how many children cover the given fraction of policy mass.

        Args:
            coverage_threshold: Fraction of policy mass to cover (e.g., 0.80).

        Returns:
            Number of children needed to cover that fraction.
        """
        if not self.policy_priors:
            return 0

        # Sort priors descending
        sorted_priors = sorted(self.policy_priors.values(), reverse=True)

        cumsum = 0.0
        for i, p in enumerate(sorted_priors):
            cumsum += p
            if cumsum >= coverage_threshold:
                return i + 1

        return len(sorted_priors)

    def best_child_by_prior(self) -> tuple[chess.Move, FractionalNode] | None:
        """Return the child with the highest prior probability.

        Returns:
            Tuple of (move, child_node) or None if no children.
        """
        if not self.children:
            return None
        return max(self.children.items(), key=lambda kv: kv[1].P)

    def best_child_by_q(self) -> tuple[chess.Move, FractionalNode] | None:
        """Return the child with the highest Q value (from parent's perspective).

        Note: We negate child's Q since it's from opponent's perspective.

        Returns:
            Tuple of (move, child_node) or None if no children.
        """
        if not self.children:
            return None
        return max(self.children.items(), key=lambda kv: -kv[1].Q)

    def get_pv(self, max_depth: int = 10) -> list[chess.Move]:
        """Get principal variation (best path by Q from this node).

        Args:
            max_depth: Maximum depth to traverse.

        Returns:
            List of moves forming the principal variation.
        """
        pv: list[chess.Move] = []
        node = self

        for _ in range(max_depth):
            best = node.best_child_by_q()
            if best is None:
                break
            move, child = best
            pv.append(move)
            node = child

        return pv

    def __repr__(self) -> str:
        move_str = self.move.uci() if self.move else "root"
        n_children = len(self.children)
        return f"FractionalNode({move_str}, Q={self.Q:.3f}, P={self.P:.3f}, children={n_children})"
