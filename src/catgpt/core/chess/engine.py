"""Chess engine implementation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

if TYPE_CHECKING:
    from typing import Protocol

    class ModelProtocol(Protocol):
        """Protocol for neural network models used by ChessEngine."""

        def evaluate(self, board: chess.Board) -> float: ...


@dataclass
class MoveEvaluation:
    """Evaluation result for a chess move."""

    move: chess.Move
    score: float
    depth: int
    principal_variation: list[chess.Move]


class ChessEngine:
    """Neural network-based chess engine.

    This engine uses a trained neural network to evaluate positions
    and search for the best moves. The engine itself is framework-agnostic;
    pass in a model that implements the evaluate() method.
    """

    def __init__(self, model: "ModelProtocol | None" = None) -> None:
        """Initialize the chess engine.

        Args:
            model: Optional neural network model for position evaluation.
                   Must have an evaluate(board) method that returns a float.
        """
        self.board = chess.Board()
        self._model = model

    def reset(self) -> None:
        """Reset the board to the starting position."""
        self.board.reset()

    def set_position(self, fen: str) -> None:
        """Set the board position from a FEN string.

        Args:
            fen: FEN string representing the position.
        """
        self.board.set_fen(fen)

    def make_move(self, move: str | chess.Move) -> bool:
        """Make a move on the board.

        Args:
            move: Move in UCI notation or a chess.Move object.

        Returns:
            True if the move was legal and made, False otherwise.
        """
        if isinstance(move, str):
            try:
                move = chess.Move.from_uci(move)
            except ValueError:
                return False

        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def get_best_move(self, depth: int = 4) -> MoveEvaluation | None:
        """Find the best move in the current position.

        Args:
            depth: Search depth.

        Returns:
            MoveEvaluation for the best move, or None if no legal moves.
        """
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return None

        # TODO: Implement actual search (MCTS or alpha-beta)
        # For now, return first legal move with placeholder evaluation
        best_move = legal_moves[0]
        score = self.evaluate_position()

        return MoveEvaluation(
            move=best_move,
            score=score,
            depth=depth,
            principal_variation=[best_move],
        )

    def evaluate_position(self) -> float:
        """Evaluate the current position.

        Returns:
            Evaluation score from white's perspective.
            Positive = white is better, negative = black is better.
        """
        # Use neural network if available
        if self._model is not None:
            return self._model.evaluate(self.board)

        # Fallback: simple material count
        return self._material_count()

    def _material_count(self) -> float:
        """Simple material-based evaluation."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        score = 0.0
        for piece_type in piece_values:
            white_count = len(self.board.pieces(piece_type, chess.WHITE))
            black_count = len(self.board.pieces(piece_type, chess.BLACK))
            score += piece_values[piece_type] * (white_count - black_count)

        return score
