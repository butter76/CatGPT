"""Policy-based chess engine that selects moves with highest policy probability."""

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

if TYPE_CHECKING:
    from jax.typing import DTypeLike

    from catgpt.jax.configs import JaxTokenizerConfig
    from catgpt.jax.models.transformer import BidirectionalTransformer


class PolicyEngine:
    """Chess engine that selects the move with highest policy probability.

    This engine uses the policy head output directly without any search:
    1. Run the model to get policy logits (64*73 = 4672 values)
    2. For each legal move, look up its policy logit
    3. Select the move with the highest logit

    This is a "pure policy" engine useful for evaluating raw policy quality
    without the influence of value-based move ordering or search.

    Note: Requires the model to have policy_head enabled. Will raise an error
    if the model doesn't output policy_logit.
    """

    def __init__(
        self,
        model: "BidirectionalTransformer",
        params: dict,
        tokenizer_config: "JaxTokenizerConfig",
        *,
        batch_size: int = 64,
        compute_dtype: "DTypeLike | None" = None,
    ) -> None:
        """Initialize the PolicyEngine.

        Args:
            model: The BidirectionalTransformer model.
            params: Model parameters.
            tokenizer_config: Configuration for FEN tokenization.
            batch_size: Fixed batch size for evaluation. All batches are padded
                to this size to avoid JIT recompilation.
            compute_dtype: Dtype for intermediate computations (e.g., jnp.float32,
                jnp.bfloat16, jnp.float16). Defaults to jnp.float32.
        """
        self.model = model
        self.params = params
        self.tokenizer_config = tokenizer_config
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

        # Warmup JIT compilation with fixed batch size
        logger.debug(f"PolicyEngine: warming up JIT with batch_size={batch_size}")
        dummy_input = jnp.zeros((batch_size, self._seq_length), dtype=jnp.int32)
        outputs = self._apply_fn(params, dummy_input)
        # Block until compilation is done
        jax.block_until_ready(outputs)

        # Verify policy head is enabled
        if "policy_logit" not in outputs:
            raise ValueError(
                "PolicyEngine requires a model with policy_head enabled. "
                "The model output does not contain 'policy_logit'."
            )

        logger.debug(f"PolicyEngine initialized with batch_size={batch_size}, compute_dtype={compute_dtype}")

    @property
    def name(self) -> str:
        """Return the engine name."""
        return "PolicyEngine"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        batch_size: int = 64,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "PolicyEngine":
        """Create a PolicyEngine from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            batch_size: Fixed batch size for evaluation (padded for JIT efficiency).
            compute_dtype: Dtype for intermediate computations (e.g., jnp.float32,
                jnp.bfloat16, jnp.float16). Defaults to jnp.float32.

        Returns:
            Initialized PolicyEngine.
        """
        loaded = load_checkpoint(checkpoint_path)
        return cls(
            model=loaded.model,
            params=loaded.params,
            tokenizer_config=loaded.tokenizer_config,
            batch_size=batch_size,
            compute_dtype=compute_dtype,
        )

    @classmethod
    def from_loaded_checkpoint(
        cls,
        checkpoint: LoadedCheckpoint,
        *,
        batch_size: int = 64,
        compute_dtype: "DTypeLike | None" = None,
    ) -> "PolicyEngine":
        """Create a PolicyEngine from an already-loaded checkpoint.

        Args:
            checkpoint: Loaded checkpoint containing model and params.
            batch_size: Fixed batch size for evaluation (padded for JIT efficiency).
            compute_dtype: Dtype for intermediate computations (e.g., jnp.float32,
                jnp.bfloat16, jnp.float16). Defaults to jnp.float32.

        Returns:
            Initialized PolicyEngine.
        """
        return cls(
            model=checkpoint.model,
            params=checkpoint.params,
            tokenizer_config=checkpoint.tokenizer_config,
            batch_size=batch_size,
            compute_dtype=compute_dtype,
        )

    def reset(self) -> None:
        """Reset engine state (no-op for this stateless engine)."""
        pass

    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Select the move with highest policy probability.

        Args:
            board: Current chess position.

        Returns:
            The selected move, or None if no legal moves exist.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # If only one legal move, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Get policy logits for this position
        policy_logits = self._get_policy_logits(board)  # (64, 73)

        # Determine if we need to flip (tokenizer flips for black to move)
        flip = board.turn == chess.BLACK

        # Find the move with highest policy logit
        best_move = None
        best_logit = float("-inf")

        for move in legal_moves:
            from_idx, to_idx = encode_move_to_policy_index(move, flip=flip)
            logit = policy_logits[from_idx, to_idx]

            if logit > best_logit:
                best_logit = logit
                best_move = move

        return best_move

    def _get_policy_logits(self, board: chess.Board) -> np.ndarray:
        """Get policy logits for a single position.

        Args:
            board: Chess position to evaluate.

        Returns:
            Policy logits reshaped to (64, 73).
        """
        fen = board.fen()

        # Tokenize the position
        tokens = tokenize(fen, self._tok_config)

        # Create padded batch of size 1 (padded to batch_size for JIT)
        padded_tokens = np.zeros((self.batch_size, self._seq_length), dtype=np.uint8)
        padded_tokens[0] = tokens

        # Run model
        tokens_jax = jnp.array(padded_tokens)
        outputs = self._apply_fn(self.params, tokens_jax)

        # Extract policy logits for first position and reshape
        policy_flat = np.array(outputs["policy_logit"][0])  # (4672,)
        policy_logits = policy_flat.reshape(64, 73)

        return policy_logits

    def get_move_scores(
        self, board: chess.Board
    ) -> list[tuple[chess.Move, float]]:
        """Get policy scores for all legal moves (for analysis/debugging).

        Args:
            board: Current chess position.

        Returns:
            List of (move, policy_logit) tuples sorted by score (best first).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        policy_logits = self._get_policy_logits(board)
        flip = board.turn == chess.BLACK

        move_scores: list[tuple[chess.Move, float]] = []

        for move in legal_moves:
            from_idx, to_idx = encode_move_to_policy_index(move, flip=flip)
            logit = float(policy_logits[from_idx, to_idx])
            move_scores.append((move, logit))

        # Sort by score (higher = better)
        return sorted(move_scores, key=lambda x: x[1], reverse=True)

    def get_move_probabilities(
        self, board: chess.Board
    ) -> list[tuple[chess.Move, float]]:
        """Get softmax probabilities for all legal moves.

        Args:
            board: Current chess position.

        Returns:
            List of (move, probability) tuples sorted by probability (best first).
        """
        move_scores = self.get_move_scores(board)
        if not move_scores:
            return []

        # Apply softmax over legal move logits only
        logits = np.array([score for _, score in move_scores])
        # Subtract max for numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        return [
            (move, float(prob))
            for (move, _), prob in zip(move_scores, probs)
        ]
