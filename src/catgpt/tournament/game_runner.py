"""Game runner for chess engine tournaments.

Handles playing individual games between two UCI engines, including
move adjudication and game termination.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

import chess
import chess.syzygy
from loguru import logger

from catgpt.cpp.uci_engine import UCIEngine


class GameTermination(Enum):
    """How a game ended."""

    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    INSUFFICIENT = "insufficient_material"
    FIFTY_MOVE = "fifty_move_rule"
    THREEFOLD = "threefold_repetition"
    DRAW_ADJUDICATED = "draw_adjudicated"
    RESIGN_ADJUDICATED = "resign_adjudicated"
    SYZYGY_ADJUDICATED = "syzygy_adjudicated"
    MAX_MOVES = "max_moves"
    ENGINE_ERROR = "engine_error"


@dataclass
class GameConfig:
    """Configuration for game adjudication."""

    # Maximum moves before forced draw
    adjudicate_draw_moves: int = 200

    # Draw adjudication: if |score| < threshold for N consecutive moves
    adjudicate_draw_score: int = 10  # Centipawns
    adjudicate_draw_count: int = 10  # Consecutive moves

    # Resignation: if score < -threshold for N consecutive moves
    adjudicate_resign_score: int = 1000  # Centipawns (10 pawns)
    adjudicate_resign_count: int = 5  # Consecutive moves

    # Syzygy tablebase adjudication
    syzygy_enabled: bool = False
    syzygy_path: str | None = None  # Path to tablebase files
    syzygy_adjudicate_draw: bool = True  # Adjudicate WDL=0 as draw
    syzygy_adjudicate_win: bool = True  # Adjudicate WDL=±2 as win/loss


@dataclass
class GameResult:
    """Result of a single game."""

    # Result from Engine A's perspective
    result: str  # "1-0" (A wins), "0-1" (A loses), "1/2-1/2" (draw)
    result_value: float  # 1.0, 0.0, or 0.5 for Engine A

    # Game details
    opening_fen: str
    engine_a_white: bool
    moves: list[str]
    termination: GameTermination

    # Metadata
    move_count: int = field(init=False)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self.move_count = len(self.moves)

    @property
    def pgn_result(self) -> str:
        """Result string for PGN."""
        return self.result

    def to_pgn(
        self,
        engine_a_name: str = "Engine_A",
        engine_b_name: str = "Engine_B",
        event: str = "SPRT Tournament",
        round_num: int = 1,
    ) -> str:
        """Generate PGN string for this game."""
        white_name = engine_a_name if self.engine_a_white else engine_b_name
        black_name = engine_b_name if self.engine_a_white else engine_a_name

        # Build PGN header
        lines = [
            f'[Event "{event}"]',
            f'[Site "Local"]',
            f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]',
            f'[Round "{round_num}"]',
            f'[White "{white_name}"]',
            f'[Black "{black_name}"]',
            f'[Result "{self.result}"]',
            f'[FEN "{self.opening_fen}"]',
            f'[SetUp "1"]',
            f'[Termination "{self.termination.value}"]',
            "",
        ]

        # Build move text
        board = chess.Board(self.opening_fen)
        move_text_parts = []

        for i, uci in enumerate(self.moves):
            move = chess.Move.from_uci(uci)

            if board.turn == chess.WHITE:
                move_num = board.fullmove_number
                move_text_parts.append(f"{move_num}.")

            # Use SAN notation
            san = board.san(move)
            move_text_parts.append(san)
            board.push(move)

        move_text = " ".join(move_text_parts)
        if move_text:
            move_text += " "
        move_text += self.result

        lines.append(move_text)
        lines.append("")

        return "\n".join(lines)


class GameRunner:
    """Runs games between two UCI engines."""

    def __init__(self, config: GameConfig | None = None):
        """Initialize game runner.

        Args:
            config: Game adjudication configuration.
        """
        self.config = config or GameConfig()
        self._tablebase: chess.syzygy.Tablebase | None = None

        # Initialize Syzygy tablebases if enabled
        if self.config.syzygy_enabled and self.config.syzygy_path:
            self._init_tablebases()

    def _init_tablebases(self) -> None:
        """Initialize Syzygy tablebase probing."""
        path = Path(self.config.syzygy_path)
        if not path.exists():
            logger.warning(f"Syzygy path not found: {path}")
            return

        try:
            self._tablebase = chess.syzygy.open_tablebase(str(path))
            logger.info(f"Loaded Syzygy tablebases from {path}")

            # Try to add subdirectories (common layout: 3-4-5/, 6/)
            for subdir in path.iterdir():
                if subdir.is_dir():
                    try:
                        count = self._tablebase.add_directory(str(subdir))
                        if count > 0:
                            logger.debug(f"Added {count} tables from {subdir}")
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Failed to load Syzygy tablebases: {e}")
            self._tablebase = None

    def _probe_syzygy(self, board: chess.Board) -> tuple[int | None, int | None]:
        """Probe Syzygy tablebases for WDL and DTZ.

        Args:
            board: Current board position.

        Returns:
            Tuple of (wdl, dtz) where:
            - wdl: 2=win, 1=cursed win, 0=draw, -1=blessed loss, -2=loss (or None)
            - dtz: Distance to zeroing move (or None)
        """
        if self._tablebase is None:
            return None, None

        # Skip positions with castling rights (not in tablebases)
        if board.has_castling_rights(chess.WHITE) or board.has_castling_rights(
            chess.BLACK
        ):
            return None, None

        # Count pieces (tablebases typically support up to 6-7 pieces)
        piece_count = chess.popcount(board.occupied)
        if piece_count > 7:  # Most common tablebase limit
            return None, None

        try:
            wdl = self._tablebase.probe_wdl(board)
            dtz = self._tablebase.probe_dtz(board)
            return wdl, dtz
        except KeyError:
            # Position not in tablebase
            return None, None
        except Exception as e:
            logger.debug(f"Syzygy probe error: {e}")
            return None, None

    def _check_syzygy_adjudication(
        self, board: chess.Board, engine_a_white: bool
    ) -> tuple[GameTermination, str] | None:
        """Check if position can be adjudicated via Syzygy tablebases.

        Args:
            board: Current board position.
            engine_a_white: Whether Engine A is playing White.

        Returns:
            Tuple of (termination, result) if adjudicated, None otherwise.
        """
        if not self.config.syzygy_enabled or self._tablebase is None:
            return None

        wdl, dtz = self._probe_syzygy(board)
        if wdl is None:
            return None

        # WDL values from side-to-move perspective:
        # 2 = winning, 1 = cursed win, 0 = draw, -1 = blessed loss, -2 = losing

        # Adjudicate clear wins/losses (WDL = ±2)
        if self.config.syzygy_adjudicate_win and abs(wdl) == 2:
            if wdl == 2:
                # Side to move is winning
                if board.turn == chess.WHITE:
                    result = "1-0"
                else:
                    result = "0-1"
            else:  # wdl == -2
                # Side to move is losing
                if board.turn == chess.WHITE:
                    result = "0-1"
                else:
                    result = "1-0"

            logger.debug(
                f"Syzygy adjudication: WDL={wdl}, DTZ={dtz}, result={result}"
            )
            return GameTermination.SYZYGY_ADJUDICATED, result

        # Adjudicate draws (WDL = 0)
        if self.config.syzygy_adjudicate_draw and wdl == 0:
            logger.debug(f"Syzygy adjudication: WDL={wdl} (draw)")
            return GameTermination.SYZYGY_ADJUDICATED, "1/2-1/2"

        # Don't adjudicate cursed wins (WDL = 1) or blessed losses (WDL = -1)
        # as these can be drawn with the 50-move rule
        return None

    def close(self) -> None:
        """Close resources (tablebase files)."""
        if self._tablebase is not None:
            self._tablebase.close()
            self._tablebase = None

    def play_game(
        self,
        engine_a: UCIEngine,
        engine_b: UCIEngine,
        opening_fen: str,
        engine_a_white: bool = True,
    ) -> GameResult:
        """Play a single game between two engines.

        Args:
            engine_a: The first engine (the one being tested).
            engine_b: The second engine (the baseline).
            opening_fen: Starting position FEN.
            engine_a_white: If True, Engine A plays White.

        Returns:
            GameResult with the outcome.
        """
        board = chess.Board(opening_fen)
        moves: list[str] = []

        # Assign colors
        white_engine = engine_a if engine_a_white else engine_b
        black_engine = engine_b if engine_a_white else engine_a

        # Track scores for adjudication (not yet implemented in UCI wrapper)
        # For now, we just play until natural termination or move limit

        consecutive_low_score = 0
        consecutive_losing = 0

        while True:
            # Check for game over
            if board.is_game_over():
                termination, result = self._get_natural_termination(board)
                break

            # Check move limit
            if len(moves) >= self.config.adjudicate_draw_moves:
                termination = GameTermination.MAX_MOVES
                result = "1/2-1/2"
                break

            # Check Syzygy tablebase adjudication
            syzygy_result = self._check_syzygy_adjudication(board, engine_a_white)
            if syzygy_result is not None:
                termination, result = syzygy_result
                break

            # Get current engine
            current_engine = white_engine if board.turn == chess.WHITE else black_engine

            # Get move from engine
            try:
                move = current_engine.select_move(board)
            except Exception as e:
                logger.error(f"Engine error: {e}")
                termination = GameTermination.ENGINE_ERROR
                # Engine that errored loses
                if board.turn == chess.WHITE:
                    result = "0-1" if engine_a_white else "1-0"
                else:
                    result = "1-0" if engine_a_white else "0-1"
                break

            if move is None:
                # No legal moves but game_over check failed - shouldn't happen
                logger.warning("Engine returned None but game not over")
                termination = GameTermination.ENGINE_ERROR
                result = "1/2-1/2"
                break

            # Make the move
            board.push(move)
            moves.append(move.uci())

            # Reset engine state periodically to avoid memory issues
            # (commented out - may cause slowdown due to TensorRT reinit)
            # if len(moves) % 50 == 0:
            #     current_engine.reset()

        # Convert result to Engine A's perspective
        result_value = self._result_to_value(result, engine_a_white)

        return GameResult(
            result=result,
            result_value=result_value,
            opening_fen=opening_fen,
            engine_a_white=engine_a_white,
            moves=moves,
            termination=termination,
        )

    def _get_natural_termination(
        self, board: chess.Board
    ) -> tuple[GameTermination, str]:
        """Determine termination type for a naturally ended game.

        Args:
            board: The board in terminal state.

        Returns:
            Tuple of (termination type, result string).
        """
        if board.is_checkmate():
            # The side to move is checkmated
            if board.turn == chess.WHITE:
                return GameTermination.CHECKMATE, "0-1"
            else:
                return GameTermination.CHECKMATE, "1-0"

        if board.is_stalemate():
            return GameTermination.STALEMATE, "1/2-1/2"

        if board.is_insufficient_material():
            return GameTermination.INSUFFICIENT, "1/2-1/2"

        if board.is_fifty_moves():
            return GameTermination.FIFTY_MOVE, "1/2-1/2"

        if board.is_repetition(3):
            return GameTermination.THREEFOLD, "1/2-1/2"

        # Fallback
        return GameTermination.STALEMATE, "1/2-1/2"

    def _result_to_value(self, result: str, engine_a_white: bool) -> float:
        """Convert result string to value from Engine A's perspective.

        Args:
            result: Result string ("1-0", "0-1", "1/2-1/2").
            engine_a_white: Whether Engine A played White.

        Returns:
            1.0 for win, 0.0 for loss, 0.5 for draw.
        """
        if result == "1/2-1/2":
            return 0.5
        elif result == "1-0":
            # White won
            return 1.0 if engine_a_white else 0.0
        else:  # "0-1"
            # Black won
            return 0.0 if engine_a_white else 1.0

    def play_game_pair(
        self,
        engine_a: UCIEngine,
        engine_b: UCIEngine,
        opening_fen: str,
    ) -> tuple[GameResult, GameResult]:
        """Play a pair of games with colors swapped.

        This is the standard way to play in engine matches - each opening
        is played twice with colors swapped to eliminate first-move advantage.

        Args:
            engine_a: The first engine.
            engine_b: The second engine.
            opening_fen: Starting position FEN.

        Returns:
            Tuple of (game1_result, game2_result) where game1 has A as White.
        """
        # Game 1: Engine A plays White
        game1 = self.play_game(engine_a, engine_b, opening_fen, engine_a_white=True)

        # Reset engines between games
        try:
            engine_a.reset()
            engine_b.reset()
        except Exception as e:
            logger.warning(f"Failed to reset engines: {e}")

        # Game 2: Engine A plays Black
        game2 = self.play_game(engine_a, engine_b, opening_fen, engine_a_white=False)

        return game1, game2
