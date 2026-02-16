"""UCI Engine wrapper for communicating with C++ chess engines.

This module provides a Python interface to C++ UCI chess engines via pexpect.
The engine binary is started once and kept running to avoid expensive TensorRT
initialization on each position.

Uses pexpect for reliable interactive communication with the subprocess,
which handles PTY allocation and buffering correctly.
"""

import re
import threading
from pathlib import Path

import chess
import pexpect
from loguru import logger


class UCIEngineError(Exception):
    """Raised when UCI communication fails."""
    pass


class UCIEngine:
    """UCI protocol wrapper for C++ chess engines.

    This class manages a long-lived subprocess running a UCI engine,
    communicating via pexpect. The engine is kept running to avoid
    expensive TensorRT initialization on each position.

    Example:
        engine = UCIEngine("/path/to/catgpt_mcts", "/path/to/model.trt")
        move = engine.select_move(board)
        engine.close()

    Or as a context manager:
        with UCIEngine("/path/to/catgpt_mcts") as engine:
            move = engine.select_move(board)
    """

    def __init__(
        self,
        binary_path: str | Path,
        engine_path: str | Path | None = None,
        *,
        timeout: float = 60.0,
        go_options: dict[str, int | str] | None = None,
    ) -> None:
        """Initialize the UCI engine.

        Args:
            binary_path: Path to the UCI engine executable.
            engine_path: Path to the TensorRT engine file (passed as CLI arg).
            timeout: Timeout in seconds for UCI responses.
            go_options: Options for the 'go' command (e.g., {'nodes': 100, 'movetime': 1000}).
        """
        self.binary_path = Path(binary_path)
        self.engine_path = Path(engine_path) if engine_path else None
        self.timeout = timeout
        self.go_options = go_options or {}

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Engine binary not found: {self.binary_path}")

        if self.engine_path and not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        self._child: pexpect.spawn | None = None
        self._lock = threading.Lock()
        self.last_nodes: int = 0  # GPU evals from the most recent select_move call
        self.last_depth: int = 0  # Search depth from the most recent select_move call
        self._start_engine()

    def _start_engine(self) -> None:
        """Start the UCI engine subprocess."""
        cmd = str(self.binary_path)
        if self.engine_path:
            cmd += f" {self.engine_path}"

        logger.debug(f"Starting UCI engine: {cmd}")

        self._child = pexpect.spawn(
            cmd,
            encoding="utf-8",
            timeout=self.timeout,
        )

        # Initialize UCI protocol
        self._send_command("uci")
        self._wait_for_response("uciok")

        self._send_command("isready")
        self._wait_for_response("readyok")

        logger.debug("UCI engine initialized")

    def _send_command(self, command: str) -> None:
        """Send a command to the engine."""
        if self._child is None:
            raise UCIEngineError("Engine not running")

        logger.trace(f"UCI send: {command}")
        self._child.sendline(command)

    def _wait_for_response(self, expected: str) -> str:
        """Wait for a specific response from the engine.

        Args:
            expected: The response pattern to wait for.

        Returns:
            The matched text.
        """
        if self._child is None:
            raise UCIEngineError("Engine not running")

        try:
            self._child.expect(expected, timeout=self.timeout)
            return self._child.after
        except pexpect.TIMEOUT:
            raise UCIEngineError(f"Timeout waiting for '{expected}'")
        except pexpect.EOF:
            raise UCIEngineError("Engine process terminated unexpectedly")

    def _parse_bestmove(self, text: str) -> chess.Move | None:
        """Parse the bestmove from engine output."""
        # Text contains the matched pattern and surrounding context
        # Look for "bestmove XXXX" pattern
        match = re.search(r"bestmove\s+(\S+)", text)
        if match:
            move_uci = match.group(1)
            if move_uci == "(none)" or move_uci == "0000":
                return None
            try:
                return chess.Move.from_uci(move_uci)
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_info_nodes(text: str) -> int:
        """Parse the node count from UCI info output.

        Looks for the last 'info' line containing 'nodes <N>' in the
        engine output that precedes the bestmove response.

        Args:
            text: Raw engine output before the bestmove line.

        Returns:
            The node count from the last info line, or 0 if not found.
        """
        nodes = 0
        for match in re.finditer(r"info\b.*?\bnodes\s+(\d+)", text):
            nodes = int(match.group(1))
        return nodes

    @staticmethod
    def _parse_info_depth(text: str) -> int:
        """Parse the search depth from UCI info output.

        Looks for the last 'info' line containing 'depth <N>' in the
        engine output that precedes the bestmove response.

        Args:
            text: Raw engine output before the bestmove line.

        Returns:
            The depth from the last info line, or 0 if not found.
        """
        depth = 0
        for match in re.finditer(r"info\b.*?\bdepth\s+(\d+)", text):
            depth = int(match.group(1))
        return depth

    def select_move(
        self,
        board: chess.Board,
        opening_fen: str | None = None,
        moves: list[str] | None = None,
    ) -> chess.Move | None:
        """Select the best move for the given position.

        Args:
            board: Current chess position.
            opening_fen: Starting FEN for this game (enables repetition detection).
            moves: List of UCI moves played since opening_fen.

        Returns:
            The selected move, or None if no legal moves exist.

        Note:
            For proper repetition detection, provide opening_fen and moves.
            This sends "position fen <opening_fen> moves <move1> <move2> ..."
            so the engine can track position history. Without this, the engine
            only sees the current FEN and cannot detect threefold repetition.
        """
        with self._lock:
            # Set up position with move history for repetition detection
            if opening_fen is not None and moves is not None:
                if moves:
                    move_str = " ".join(moves)
                    self._send_command(f"position fen {opening_fen} moves {move_str}")
                else:
                    self._send_command(f"position fen {opening_fen}")
            else:
                # Fallback: just send current FEN (no repetition detection)
                fen = board.fen()
                self._send_command(f"position fen {fen}")

            # Send go command with options
            go_parts = ["go"]
            for key, value in self.go_options.items():
                go_parts.append(f"{key}")
                go_parts.append(str(value))
            self._send_command(" ".join(go_parts))

            # Wait for bestmove — everything before the match (info lines) is
            # available in self._child.before after the expect call.
            self._wait_for_response("bestmove")

            # Get the full line containing bestmove
            if self._child is None:
                return None

            # The text before the "bestmove" match contains info lines
            info_output = self._child.before or ""
            self.last_nodes = self._parse_info_nodes(info_output)
            self.last_depth = self._parse_info_depth(info_output)

            # Read the rest of the line after "bestmove"
            self._child.expect(r"\r?\n")
            bestmove_line = "bestmove" + (self._child.before or "")
            return self._parse_bestmove(bestmove_line)

    def reset(self) -> None:
        """Reset the engine state."""
        with self._lock:
            self._send_command("ucinewgame")
            self._send_command("isready")
            self._wait_for_response("readyok")

    def close(self) -> None:
        """Close the engine subprocess."""
        if self._child is not None:
            try:
                self._send_command("quit")
                self._child.wait()
            except Exception:
                try:
                    self._child.terminate(force=True)
                except Exception:
                    pass
            finally:
                self._child = None

    def __enter__(self) -> "UCIEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __del__(self) -> None:
        """Destructor - ensure process is cleaned up."""
        self.close()

    @property
    def name(self) -> str:
        """Return the engine name."""
        return f"UCI({self.binary_path.name})"


class MCTSEngine(UCIEngine):
    """UCI engine wrapper with MCTS-specific defaults."""

    def __init__(
        self,
        binary_path: str | Path,
        engine_path: str | Path | None = None,
        *,
        num_simulations: int = 400,
        timeout: float = 120.0,
    ) -> None:
        """Initialize MCTS engine.

        Args:
            binary_path: Path to the catgpt_mcts binary.
            engine_path: Path to the TensorRT engine file.
            num_simulations: Number of MCTS simulations (passed as nodes).
            timeout: Timeout in seconds for UCI responses.
        """
        super().__init__(
            binary_path,
            engine_path,
            timeout=timeout,
            go_options={"nodes": num_simulations},
        )
        self._num_simulations = num_simulations

    @property
    def name(self) -> str:
        return f"MCTS(nodes={self._num_simulations})"


class FractionalMCTSEngine(UCIEngine):
    """UCI engine wrapper for Fractional MCTS with iterative deepening."""

    def __init__(
        self,
        binary_path: str | Path,
        engine_path: str | Path | None = None,
        *,
        min_total_evals: int = 400,
        timeout: float = 120.0,
    ) -> None:
        """Initialize Fractional MCTS engine.

        Args:
            binary_path: Path to the catgpt_fractional_mcts binary.
            engine_path: Path to the TensorRT engine file.
            min_total_evals: Minimum total GPU evaluations (passed as nodes).
            timeout: Timeout in seconds for UCI responses.
        """
        super().__init__(
            binary_path,
            engine_path,
            timeout=timeout,
            go_options={"nodes": min_total_evals},
        )
        self._min_total_evals = min_total_evals

    @property
    def name(self) -> str:
        return f"FractionalMCTS(evals={self._min_total_evals})"


class ValueEngine(UCIEngine):
    """UCI engine wrapper for value-based search."""

    def __init__(
        self,
        binary_path: str | Path,
        engine_path: str | Path | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Initialize value engine.

        Args:
            binary_path: Path to the catgpt_value binary.
            engine_path: Path to the TensorRT engine file.
            timeout: Timeout in seconds for UCI responses.
        """
        # Value engine doesn't need go options - it evaluates all moves once
        super().__init__(
            binary_path,
            engine_path,
            timeout=timeout,
            go_options={},
        )

    @property
    def name(self) -> str:
        return "Value(1-ply)"


class PolicyEngine(UCIEngine):
    """UCI engine wrapper for policy-based search."""

    def __init__(
        self,
        binary_path: str | Path,
        engine_path: str | Path | None = None,
        *,
        timeout: float = 60.0,
    ) -> None:
        """Initialize policy engine.

        Args:
            binary_path: Path to the catgpt_policy binary.
            engine_path: Path to the TensorRT engine file.
            timeout: Timeout in seconds for UCI responses.
        """
        # Policy engine just returns highest probability move
        super().__init__(
            binary_path,
            engine_path,
            timeout=timeout,
            go_options={},
        )

    @property
    def name(self) -> str:
        return "Policy"
