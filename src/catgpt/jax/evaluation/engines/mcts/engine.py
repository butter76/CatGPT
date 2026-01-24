"""MCTS engine using CPUCT selection (AlphaZero/Leela Chess Zero style)."""

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
from catgpt.jax.evaluation.engines.mcts.config import MCTSConfig
from catgpt.jax.evaluation.engines.mcts.node import MCTSNode

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from catgpt.jax.configs import JaxTokenizerConfig
    from catgpt.jax.models.transformer import BidirectionalTransformer


class MCTSEngine:
    """MCTS engine using CPUCT selection.

    This engine uses Monte Carlo Tree Search with the PUCT formula for selection:

        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))

    Where:
        - Q(s,a): Mean value of taking action a from state s
        - P(s,a): Prior probability from policy network
        - N(s,a): Visit count for this action
        - N_parent: Total visits to parent node
        - c_puct: Exploration constant

    The search proceeds in four phases:
        1. SELECT: Traverse tree using PUCT until reaching a leaf
        2. EXPAND: Create children for the leaf with priors from policy network
        3. EVALUATE: Get value estimate from value network
        4. BACKPROPAGATE: Update N, W along path from leaf to root

    After search, the move with highest visit count is selected.
    """

    def __init__(
        self,
        model: "BidirectionalTransformer",
        params: dict,
        tokenizer_config: "JaxTokenizerConfig",
        config: MCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> None:
        """Initialize the MCTSEngine.

        Args:
            model: The BidirectionalTransformer model (must have policy head enabled).
            params: Model parameters.
            tokenizer_config: Configuration for FEN tokenization.
            config: MCTS configuration. Uses defaults if None.
            batch_size: Fixed batch size for neural network evaluation.
                For MCTS, typically 1 since we evaluate one position at a time.
            compute_dtype: Dtype for intermediate computations.
        """
        self.model = model
        self.params = params
        self.tokenizer_config = tokenizer_config
        self.config = config or MCTSConfig()
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
        logger.debug(f"MCTSEngine: warming up JIT with batch_size={batch_size}")
        dummy_input = jnp.zeros((batch_size, self._seq_length), dtype=jnp.int32)
        outputs = self._apply_fn(params, dummy_input)
        jax.block_until_ready(outputs)

        # Verify policy head is enabled
        if "policy_logit" not in outputs:
            raise ValueError(
                "MCTSEngine requires a model with policy_head enabled. "
                "The model output does not contain 'policy_logit'."
            )

        logger.debug(
            f"MCTSEngine initialized with num_simulations={self.config.num_simulations}, "
            f"c_puct={self.config.c_puct}"
        )

        # Tree state (reset on each select_move call)
        self._root: MCTSNode | None = None

    @property
    def name(self) -> str:
        """Return the engine name."""
        return f"MCTS(n={self.config.num_simulations})"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        config: MCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "MCTSEngine":
        """Create an MCTSEngine from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            config: MCTS configuration.
            batch_size: Fixed batch size for neural network evaluation.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Initialized MCTSEngine.
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
        config: MCTSConfig | None = None,
        *,
        batch_size: int = 1,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "MCTSEngine":
        """Create an MCTSEngine from an already-loaded checkpoint.

        Args:
            checkpoint: Loaded checkpoint containing model and params.
            config: MCTS configuration.
            batch_size: Fixed batch size for neural network evaluation.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Initialized MCTSEngine.
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

    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Run MCTS and return the move with highest visit count.

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

        # Initialize root node
        self._root = MCTSNode()

        # Run simulations
        for _ in range(self.config.num_simulations):
            self._run_simulation(board)

        # Select move with highest visit count
        best = self._root.best_child_by_visits()
        if best is None:
            # Fallback (shouldn't happen if simulations ran correctly)
            return legal_moves[0]

        return best[0]

    def _run_simulation(self, board: chess.Board) -> None:
        """Run a single MCTS simulation.

        Args:
            board: The root position (not modified).
        """
        node = self._root
        assert node is not None

        path: list[MCTSNode] = [node]
        scratch_board = board.copy()

        # SELECT: traverse tree using PUCT until we reach a leaf
        while node.is_expanded and not node.is_terminal:
            move, node = self._select_child(node)
            scratch_board.push(move)
            path.append(node)

        # EXPAND & EVALUATE
        if node.is_terminal:
            # Terminal node - use stored value
            assert node.terminal_value is not None
            value = node.terminal_value
        else:
            # Expand the leaf and get value
            value = self._expand_and_evaluate(node, scratch_board)

        # BACKPROPAGATE
        self._backpropagate(path, value)

    def _select_child(self, node: MCTSNode) -> tuple[chess.Move, MCTSNode]:
        """Select child with highest PUCT score.

        Args:
            node: Parent node (must be expanded).

        Returns:
            Tuple of (move, child_node) with highest PUCT score.
        """
        sqrt_n_parent = math.sqrt(node.N) if node.N > 0 else 1.0

        best_score = -float("inf")
        best_move: chess.Move | None = None
        best_child: MCTSNode | None = None

        for move, child in node.children.items():
            # Determine Q value from parent's perspective
            # Child's Q/terminal_value is from child's side-to-move (our opponent)
            # So we negate to get our perspective
            if child.is_terminal:
                assert child.terminal_value is not None
                q = -child.terminal_value
            elif child.N == 0:
                q = self.config.fpu_value
            else:
                q = -child.Q

            # PUCT formula
            u = q + self.config.c_puct * child.P * sqrt_n_parent / (1 + child.N)

            if u > best_score:
                best_score = u
                best_move = move
                best_child = child

        assert best_move is not None and best_child is not None
        return best_move, best_child

    def _expand_and_evaluate(self, node: MCTSNode, board: chess.Board) -> float:
        """Expand a leaf node and return value estimate.

        Creates children for all legal moves with prior probabilities from
        the policy network.

        Args:
            node: Leaf node to expand.
            board: Position at this node.

        Returns:
            Value estimate for this position (from side to move's perspective).
        """
        # Get policy and value from neural network
        policy_priors, value = self._evaluate_position(board)

        # Create children for all legal moves
        legal_moves = list(board.legal_moves)

        for move in legal_moves:
            prior = policy_priors.get(move, 0.0)
            child = MCTSNode(parent=node, move=move, P=prior)

            # Check for terminal states
            board.push(move)

            if board.is_checkmate():
                # The side to move at this position is checkmated (they lost)
                child.is_terminal = True
                child.terminal_value = -1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                child.is_terminal = True
                child.terminal_value = 0.0  # Draw
            elif board.can_claim_draw():
                # Repetition or 50-move rule
                child.is_terminal = True
                child.terminal_value = 0.0  # Draw

            board.pop()

            node.children[move] = child

        return value

    def _evaluate_position(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Evaluate a position with the neural network.

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

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        """Update statistics along the path from leaf to root.

        The value alternates perspective at each level since players take turns.

        Args:
            path: List of nodes from root to leaf.
            value: Value at the leaf (from leaf's side to move perspective).
                   Range: [-1, 1] where -1=loss, 0=draw, 1=win.
        """
        # Walk back from leaf to root
        for node in reversed(path):
            node.N += 1
            node.W += value
            # Flip perspective for parent (opponent's view)
            value = -value

    # -------------------------------------------------------------------------
    # Analysis methods
    # -------------------------------------------------------------------------

    def get_pv(self) -> list[chess.Move]:
        """Get principal variation (most visited path from root).

        Must be called after select_move().

        Returns:
            List of moves forming the principal variation.
        """
        if self._root is None:
            return []
        return self._root.get_pv()

    def get_root_policy(self) -> dict[chess.Move, float]:
        """Get normalized visit distribution at root.

        Must be called after select_move().

        Returns:
            Dictionary mapping moves to their visit proportions.
        """
        if self._root is None:
            return {}
        return self._root.get_visit_distribution()

    def get_root_value(self) -> float:
        """Get Q value at root after search.

        Must be called after select_move().

        Returns:
            Mean value at root position (-1=loss, 0=draw, 1=win).
        """
        if self._root is None:
            return 0.0
        return self._root.Q

    def get_root_stats(self) -> dict[chess.Move, dict[str, float]]:
        """Get detailed statistics for all root children.

        Must be called after select_move().

        Returns:
            Dictionary mapping moves to their statistics (N, Q, P, U).
        """
        if self._root is None:
            return {}

        sqrt_n_parent = math.sqrt(self._root.N) if self._root.N > 0 else 1.0
        stats: dict[chess.Move, dict[str, float]] = {}

        for move, child in self._root.children.items():
            # Q from root's perspective (negate child's Q)
            q = -child.Q if child.N > 0 else self.config.fpu_value
            u = self.config.c_puct * child.P * sqrt_n_parent / (1 + child.N)

            stats[move] = {
                "N": child.N,
                "Q": q,  # From root's perspective
                "P": child.P,
                "U": u,
                "score": q + u,
            }

        return stats
