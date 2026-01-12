"""Value-based chess engine using 1-move lookahead."""

from pathlib import Path
from typing import TYPE_CHECKING

import chess
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize
from catgpt.jax.evaluation.checkpoint import LoadedCheckpoint, load_checkpoint

if TYPE_CHECKING:
    from catgpt.jax.configs import JaxTokenizerConfig
    from catgpt.jax.models.transformer import BidirectionalTransformer


class ValueEngine:
    """Chess engine using a value network with 1-move lookahead.

    This engine evaluates positions by looking one move ahead:
    1. For each legal move, compute the resulting position
    2. If checkmate: return immediately (winning move)
    3. If stalemate/draw: assign score 0.5
    4. Otherwise: evaluate position with value network

    The engine picks the move that minimizes the opponent's expected score
    (i.e., maximizes our own winning probability).

    The model evaluates positions from the perspective of the side to move,
    returning the probability that the side to move wins. Since we're evaluating
    positions after our move (opponent to move), a lower score for the opponent
    means a better position for us.
    """

    def __init__(
        self,
        model: "BidirectionalTransformer",
        params: dict,
        tokenizer_config: "JaxTokenizerConfig",
        *,
        batch_size: int = 64,
    ) -> None:
        """Initialize the ValueEngine.

        Args:
            model: The BidirectionalTransformer model.
            params: Model parameters.
            tokenizer_config: Configuration for FEN tokenization.
            batch_size: Fixed batch size for evaluation. All batches are padded
                to this size to avoid JIT recompilation.
        """
        self.model = model
        self.params = params
        self.tokenizer_config = tokenizer_config
        self.batch_size = batch_size
        self._seq_length = tokenizer_config.sequence_length

        # JIT compile the model application
        self._apply_fn = jax.jit(
            lambda params, x: model.apply(params, x, train=False)
        )

        # Create tokenizer config
        self._tok_config = TokenizerConfig(
            sequence_length=tokenizer_config.sequence_length,
            include_halfmove=tokenizer_config.include_halfmove,
        )

        # Warmup JIT compilation with fixed batch size
        logger.debug(f"ValueEngine: warming up JIT with batch_size={batch_size}")
        dummy_input = jnp.zeros((batch_size, self._seq_length), dtype=jnp.int32)
        _ = self._apply_fn(params, dummy_input)
        # Block until compilation is done
        jax.block_until_ready(_)

        logger.debug(f"ValueEngine initialized with batch_size={batch_size}")

    @property
    def name(self) -> str:
        """Return the engine name."""
        return "ValueEngine"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        batch_size: int = 64,
    ) -> "ValueEngine":
        """Create a ValueEngine from a checkpoint directory.

        Args:
            checkpoint_path: Path to the checkpoint directory.
            batch_size: Fixed batch size for evaluation (padded for JIT efficiency).

        Returns:
            Initialized ValueEngine.
        """
        loaded = load_checkpoint(checkpoint_path)
        return cls(
            model=loaded.model,
            params=loaded.params,
            tokenizer_config=loaded.tokenizer_config,
            batch_size=batch_size,
        )

    @classmethod
    def from_loaded_checkpoint(
        cls,
        checkpoint: LoadedCheckpoint,
        *,
        batch_size: int = 64,
    ) -> "ValueEngine":
        """Create a ValueEngine from an already-loaded checkpoint.

        Args:
            checkpoint: Loaded checkpoint containing model and params.
            batch_size: Fixed batch size for evaluation (padded for JIT efficiency).

        Returns:
            Initialized ValueEngine.
        """
        return cls(
            model=checkpoint.model,
            params=checkpoint.params,
            tokenizer_config=checkpoint.tokenizer_config,
            batch_size=batch_size,
        )

    def reset(self) -> None:
        """Reset engine state (no-op for this stateless engine)."""
        pass

    def select_move(self, board: chess.Board) -> chess.Move | None:
        """Select the best move using 1-move lookahead.

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

        # Evaluate all moves
        move_scores: list[tuple[chess.Move, float]] = []
        positions_to_eval: list[tuple[chess.Move, str]] = []

        for move in legal_moves:
            board.push(move)

            if board.is_checkmate():
                # Instant win - return immediately
                board.pop()
                return move

            if board.is_stalemate() or board.is_insufficient_material():
                # Draw - score 0.5
                move_scores.append((move, 0.5))
            elif board.can_claim_draw():
                # Claimable draw (repetition, 50-move rule)
                move_scores.append((move, 0.5))
            else:
                # Need to evaluate this position
                positions_to_eval.append((move, board.fen()))

            board.pop()

        # Batch evaluate non-terminal positions
        if positions_to_eval:
            moves_to_eval = [m for m, _ in positions_to_eval]
            fens_to_eval = [f for _, f in positions_to_eval]

            # Evaluate in batches if needed
            values = self._batch_evaluate(fens_to_eval)

            for move, value in zip(moves_to_eval, values):
                move_scores.append((move, value))

        # Pick move with lowest opponent score (best for us)
        # Lower opponent win probability = higher our win probability
        best_move, best_score = min(move_scores, key=lambda x: x[1])

        return best_move

    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a single position.

        Args:
            board: Chess position to evaluate.

        Returns:
            Win probability for the side to move (0.0 to 1.0).
        """
        if board.is_checkmate():
            return 0.0  # Side to move is checkmated
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.5

        values = self._batch_evaluate([board.fen()])
        return values[0]

    def _batch_evaluate(self, fens: list[str]) -> list[float]:
        """Evaluate multiple positions in batches.

        Uses fixed-size batches with padding to avoid JIT recompilation.
        This is critical for performance - variable batch sizes cause
        JAX to recompile for each new shape.

        Args:
            fens: List of FEN strings to evaluate.

        Returns:
            List of win probabilities for the side to move in each position.
        """
        all_values: list[float] = []

        # Process in fixed-size batches
        for i in range(0, len(fens), self.batch_size):
            batch_fens = fens[i : i + self.batch_size]
            actual_size = len(batch_fens)

            # Tokenize all FENs in this batch
            tokens_list = [tokenize(fen, self._tok_config) for fen in batch_fens]

            # Create fixed-size padded array (padding with zeros)
            padded_tokens = np.zeros(
                (self.batch_size, self._seq_length), dtype=np.uint8
            )
            for j, tokens in enumerate(tokens_list):
                padded_tokens[j] = tokens

            # Convert to JAX array (fixed shape = no recompilation)
            tokens_jax = jnp.array(padded_tokens)

            # Run model
            outputs = self._apply_fn(self.params, tokens_jax)

            # Extract only the values for actual positions (not padding)
            values = outputs["value"][:actual_size]
            all_values.extend(values.tolist())

        return all_values

    def get_move_scores(
        self, board: chess.Board
    ) -> list[tuple[chess.Move, float]]:
        """Get scores for all legal moves (for analysis/debugging).

        Args:
            board: Current chess position.

        Returns:
            List of (move, opponent_score) tuples sorted by score (best first).
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return []

        move_scores: list[tuple[chess.Move, float]] = []
        positions_to_eval: list[tuple[chess.Move, str]] = []

        for move in legal_moves:
            board.push(move)

            if board.is_checkmate():
                move_scores.append((move, 0.0))  # Best possible
            elif board.is_stalemate() or board.is_insufficient_material():
                move_scores.append((move, 0.5))
            elif board.can_claim_draw():
                move_scores.append((move, 0.5))
            else:
                positions_to_eval.append((move, board.fen()))

            board.pop()

        if positions_to_eval:
            moves_to_eval = [m for m, _ in positions_to_eval]
            fens_to_eval = [f for _, f in positions_to_eval]
            values = self._batch_evaluate(fens_to_eval)

            for move, value in zip(moves_to_eval, values):
                move_scores.append((move, value))

        # Sort by score (lower = better for us)
        return sorted(move_scores, key=lambda x: x[1])
