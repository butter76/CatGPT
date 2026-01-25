"""Fractional MCTS engine with iterative deepening."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import chess
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from catgpt.core.utils import encode_move_to_policy_index
from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize
from catgpt.jax.evaluation.checkpoint import LoadedCheckpoint, load_checkpoint
from catgpt.jax.evaluation.engines.fractional_mcts.config import FractionalMCTSConfig
from catgpt.jax.evaluation.engines.fractional_mcts.node import FractionalNode

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from catgpt.jax.configs import JaxTokenizerConfig
    from catgpt.jax.models.transformer import BidirectionalTransformer


class FractionalMCTSEngine:
    """Fractional MCTS engine with iterative deepening.

    This engine uses a novel MCTS variant where:
    - Visit counts N are fractional (floats) rather than integers
    - Search uses iterative deepening with increasing budget N
    - Budget is allocated to children by solving the PUCT equation for equal "urgency"

    The algorithm:
    1. Initialize root with GPU eval (gets policy priors and initial Q)
    2. Run iterative deepening with N = initial * (multiplier ^ iteration)
    3. Each iteration recursively allocates budget to children
    4. Stop when total GPU evals >= min_total_evals

    For a node with budget N:
    - Compute "limit" = number of children covering 80% of policy mass
    - If N < limit: return cached Q (base case, no expansion)
    - Otherwise: expand children with P >= 1/N, allocate budget via binary search,
      recurse, then update Q as weighted average of children's Q values
    """

    def __init__(
        self,
        model: "BidirectionalTransformer",
        params: dict,
        tokenizer_config: "JaxTokenizerConfig",
        config: FractionalMCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> None:
        """Initialize the FractionalMCTSEngine.

        Args:
            model: The BidirectionalTransformer model (must have policy head enabled).
            params: Model parameters.
            tokenizer_config: Configuration for FEN tokenization.
            config: Fractional MCTS configuration. Uses defaults if None.
            batch_size: Fixed batch size for neural network evaluation.
            compute_dtype: Dtype for intermediate computations.
        """
        self.model = model
        self.params = params
        self.tokenizer_config = tokenizer_config
        self.config = config or FractionalMCTSConfig()
        self.batch_size = batch_size
        self._seq_length = tokenizer_config.sequence_length

        # Resolve compute dtype
        if compute_dtype is None:
            compute_dtype = jnp.float32
        self.compute_dtype = compute_dtype

        # JIT compile the model application
        self._apply_fn = jax.jit(
            lambda params, x: model.apply(params, x, train=False, compute_dtype=compute_dtype)
        )

        # Create tokenizer config
        self._tok_config = TokenizerConfig(
            sequence_length=tokenizer_config.sequence_length,
            include_halfmove=tokenizer_config.include_halfmove,
        )

        # Warmup JIT compilation
        logger.debug(f"FractionalMCTSEngine: warming up JIT with batch_size={batch_size}")
        dummy_input = jnp.zeros((batch_size, self._seq_length), dtype=jnp.int32)
        outputs = self._apply_fn(params, dummy_input)
        jax.block_until_ready(outputs)

        # Verify policy head is enabled
        if "policy_logit" not in outputs:
            raise ValueError(
                "FractionalMCTSEngine requires a model with policy_head enabled. "
                "The model output does not contain 'policy_logit'."
            )

        logger.debug(
            f"FractionalMCTSEngine initialized with min_total_evals={self.config.min_total_evals}, "
            f"c_puct={self.config.c_puct}, budget_multiplier={self.config.budget_multiplier}"
        )

        # Tree state (reset on each select_move call)
        self._root: FractionalNode | None = None
        self._total_gpu_evals: int = 0

    @property
    def name(self) -> str:
        """Return the engine name."""
        return f"FractionalMCTS(evals={self.config.min_total_evals})"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        config: FractionalMCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "FractionalMCTSEngine":
        """Create a FractionalMCTSEngine from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            config: Fractional MCTS configuration.
            batch_size: Fixed batch size for neural network evaluation.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Initialized FractionalMCTSEngine.
        """
        loaded = load_checkpoint(checkpoint_path)
        return cls(
            model=loaded.model,
            params=loaded.params,
            tokenizer_config=loaded.tokenizer_config,
            config=config,
            batch_size=batch_size,
            compute_dtype=compute_dtype,
        )

    @classmethod
    def from_loaded_checkpoint(
        cls,
        checkpoint: LoadedCheckpoint,
        config: FractionalMCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "FractionalMCTSEngine":
        """Create a FractionalMCTSEngine from an already-loaded checkpoint.

        Args:
            checkpoint: Loaded checkpoint containing model and params.
            config: Fractional MCTS configuration.
            batch_size: Fixed batch size for neural network evaluation.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Initialized FractionalMCTSEngine.
        """
        return cls(
            model=checkpoint.model,
            params=checkpoint.params,
            tokenizer_config=checkpoint.tokenizer_config,
            config=config,
            batch_size=batch_size,
            compute_dtype=compute_dtype,
        )

    def reset(self) -> None:
        """Reset engine state (clears the search tree)."""
        self._root = None
        self._total_gpu_evals = 0

    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Run fractional MCTS and return the best move.

        Args:
            board: Current chess position.

        Returns:
            The selected move, or None if no legal moves exist.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Single legal move - return immediately
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Initialize root node and evaluate it
        self._root = FractionalNode()
        self._total_gpu_evals = 0
        self._evaluate_node(self._root, board)

        # Run iterative deepening
        N = self.config.initial_budget
        iteration = 0

        while self._total_gpu_evals < self.config.min_total_evals:
            logger.debug(
                f"Iteration {iteration}: N={N:.2f}, total_evals={self._total_gpu_evals}"
            )
            self._recursive_search(self._root, board, N)
            N *= self.config.budget_multiplier
            iteration += 1

        logger.debug(
            f"Search complete after {iteration} iterations, {self._total_gpu_evals} GPU evals"
        )

        # Select best move by Q value (negated since children are opponent's perspective)
        best = self._root.best_child_by_q()
        if best is None:
            # Fallback (shouldn't happen)
            return legal_moves[0]

        return best[0]

    def _evaluate_node(self, node: FractionalNode, board: chess.Board) -> None:
        """Evaluate a node with the neural network.

        Populates node.policy_priors and node.Q.

        Args:
            node: Node to evaluate.
            board: Position at this node.
        """
        policy_priors, value = self._run_neural_network(board)
        node.policy_priors = policy_priors
        node.Q = value
        self._total_gpu_evals += 1

    def _run_neural_network(
        self, board: chess.Board
    ) -> tuple[dict[chess.Move, float], float]:
        """Run the neural network on a position.

        Args:
            board: Position to evaluate.

        Returns:
            Tuple of (policy_priors, value) where:
                - policy_priors: dict mapping legal moves to prior probabilities
                - value: evaluation for side to move (-1.0=loss, 0.0=draw, 1.0=win)
        """
        fen = board.fen()

        # Tokenize
        tokens = tokenize(fen, self._tok_config)

        # Create padded batch
        padded_tokens = np.zeros((self.batch_size, self._seq_length), dtype=np.uint8)
        padded_tokens[0] = tokens

        # Run model
        tokens_jax = jnp.array(padded_tokens)
        outputs = self._apply_fn(self.params, tokens_jax)

        # Extract value and convert from [0, 1] to [-1, 1]
        raw_value = float(outputs["value"][0])
        value = 2.0 * raw_value - 1.0

        # Extract policy logits and compute priors for legal moves
        policy_flat = np.array(outputs["policy_logit"][0])  # (4672,)
        policy_logits = policy_flat.reshape(64, 73)

        # Determine if we need to flip (tokenizer flips for black to move)
        flip = board.turn == chess.BLACK

        # Get logits for legal moves
        legal_moves = list(board.legal_moves)
        move_logits: list[tuple[chess.Move, float]] = []

        for move in legal_moves:
            from_idx, to_idx = encode_move_to_policy_index(move, flip=flip)
            logit = float(policy_logits[from_idx, to_idx])
            move_logits.append((move, logit))

        # Softmax over legal moves only
        logits_array = np.array([logit for _, logit in move_logits])
        logits_array = logits_array - np.max(logits_array)  # Numerical stability
        exp_logits = np.exp(logits_array)
        probs = exp_logits / np.sum(exp_logits)

        policy_priors = {move: float(prob) for (move, _), prob in zip(move_logits, probs)}

        return policy_priors, value

    def _recursive_search(
        self, node: FractionalNode, board: chess.Board, N: float
    ) -> None:
        """Recursively search from a node with budget N.

        Args:
            node: Current node (must already be evaluated or terminal).
            board: Position at this node.
            N: Budget allocated to this node.
        """
        # Terminal nodes: Q is already set, nothing to do
        if node.is_terminal:
            return

        # Compute limit: how many children cover 80% of policy
        limit = node.get_limit(self.config.policy_coverage_threshold)

        # Base case: N < limit, don't expand further
        # Q is already set from prior evaluation
        if N < limit:
            return

        # Expansion case: expand children with P >= 1/N
        expansion_threshold = 1.0 / N if N > 0 else 1.0
        self._expand_children(node, board, expansion_threshold)

        # If no children were expanded, nothing to do
        if not node.children:
            return

        # Compute budget allocations via binary search
        allocations = self._compute_allocations(node, N)

        # Recurse into children
        for move, child in node.children.items():
            N_i = allocations.get(move, 0.0)
            if N_i > 0:
                board.push(move)
                self._recursive_search(child, board, N_i)
                board.pop()

        # Update Q as weighted average of children's Q values
        # Note: negate child.Q because it's from opponent's perspective
        weighted_sum = 0.0
        total_weight = 0.0
        for move, child in node.children.items():
            N_i = allocations.get(move, 0.0)
            if N_i > 0:
                weighted_sum += (-child.Q) * N_i
                total_weight += N_i

        if total_weight > 0:
            node.Q = weighted_sum / total_weight

    def _expand_children(
        self, node: FractionalNode, board: chess.Board, threshold: float
    ) -> None:
        """Expand children with prior probability >= threshold.

        Only expands children that aren't already expanded.

        Args:
            node: Parent node.
            board: Position at parent.
            threshold: Minimum prior probability to expand.
        """
        for move, prior in node.policy_priors.items():
            if prior >= threshold and move not in node.children:
                # Create child node
                child = FractionalNode(move=move, P=prior)

                # Check for terminal state
                board.push(move)

                if board.is_checkmate():
                    # Side to move is checkmated (they lost)
                    child.is_terminal = True
                    child.Q = -1.0
                elif (
                    board.is_stalemate()
                    or board.is_insufficient_material()
                    or board.can_claim_draw()
                ):
                    child.is_terminal = True
                    child.Q = 0.0
                else:
                    # Evaluate child position
                    self._evaluate_node(child, board)

                board.pop()
                node.children[move] = child

    def _compute_allocations(
        self, node: FractionalNode, N: float
    ) -> dict[chess.Move, float]:
        """Compute budget allocations for children via binary search.

        Finds K such that for all children i:
            N_i = c_puct * P_i * sqrt(N) / (K - Q_i)
        and sum(N_i) = N.

        Args:
            node: Parent node with expanded children.
            N: Total budget to allocate.

        Returns:
            Dictionary mapping moves to their allocated budgets.
        """
        if not node.children:
            return {}

        children = list(node.children.items())
        c_puct = self.config.c_puct
        sqrt_N = math.sqrt(N)

        def compute_allocation(K: float, move: chess.Move, child: FractionalNode) -> float:
            """Compute N_i for a child given K."""
            # N_i = c_puct * P_i * sqrt(N) / (K - Q_i)
            # Note: child.Q is from child's perspective (opponent)
            # In the PUCT formula, we use -child.Q for parent's perspective
            denominator = K - (-child.Q)  # K - Q_i where Q_i is parent's view
            if denominator <= 0:
                return float("inf")
            return c_puct * child.P * sqrt_N / denominator

        def sum_allocations(K: float) -> float:
            """Sum of all N_i for a given K."""
            total = 0.0
            for move, child in children:
                alloc = compute_allocation(K, move, child)
                if alloc == float("inf"):
                    return float("inf")
                total += alloc
            return total

        # Find K bounds
        # K must be > max(-child.Q) = max(Q from parent's perspective)
        max_q = max(-child.Q for _, child in children)
        K_low = max_q + 1e-9

        # Start with a reasonable K_high and expand if needed
        K_high = K_low + 10.0

        # Expand K_high until sum < N (or we hit a reasonable limit)
        for _ in range(100):
            s = sum_allocations(K_high)
            if s <= N:
                break
            K_high *= 2
        else:
            # Fallback: if we can't find good bounds, distribute uniformly
            logger.warning("Binary search bounds failed, using uniform allocation")
            uniform = N / len(children)
            return {move: uniform for move, _ in children}

        # Binary search for K
        for _ in range(64):  # Plenty of iterations for float precision
            K_mid = (K_low + K_high) / 2
            s = sum_allocations(K_mid)
            if s > N:
                K_low = K_mid
            else:
                K_high = K_mid

        K = (K_low + K_high) / 2

        # Compute final allocations
        allocations = {}
        for move, child in children:
            allocations[move] = compute_allocation(K, move, child)

        return allocations

    # -------------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------------

    def get_pv(self) -> list[chess.Move]:
        """Get principal variation (best path by Q from root).

        Must be called after select_move().

        Returns:
            List of moves forming the principal variation.
        """
        if self._root is None:
            return []
        return self._root.get_pv()

    def get_root_policy(self) -> dict[chess.Move, float]:
        """Get policy prior distribution at root.

        Must be called after select_move().

        Returns:
            Dictionary mapping moves to their prior probabilities.
        """
        if self._root is None:
            return {}
        return self._root.policy_priors.copy()

    def get_root_value(self) -> float:
        """Get Q value at root after search.

        Must be called after select_move().

        Returns:
            Value estimate at root position (-1=loss, 0=draw, 1=win).
        """
        if self._root is None:
            return 0.0
        return self._root.Q

    def get_total_evals(self) -> int:
        """Get total GPU evaluations used in last search.

        Returns:
            Number of neural network evaluations.
        """
        return self._total_gpu_evals

    def get_root_stats(self) -> dict[chess.Move, dict[str, float]]:
        """Get statistics for all root children.

        Must be called after select_move().

        Returns:
            Dictionary mapping moves to their statistics (P, Q).
        """
        if self._root is None:
            return {}

        stats: dict[chess.Move, dict[str, float]] = {}

        for move, child in self._root.children.items():
            stats[move] = {
                "P": child.P,
                "Q": -child.Q,  # From root's perspective
                "child_Q": child.Q,  # Raw child Q
            }

        return stats
