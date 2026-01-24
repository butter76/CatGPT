"""Chess puzzle benchmark for evaluating engines."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
from loguru import logger
from tqdm import tqdm

from catgpt.jax.evaluation.benchmarks.base import BenchmarkResult
from catgpt.jax.evaluation.engines.base import Engine


@dataclass
class Puzzle:
    """A chess puzzle from the Lichess puzzle database.

    Attributes:
        id: Unique puzzle identifier.
        rating: Puzzle difficulty rating.
        fen: Starting position in FEN notation.
        moves: List of UCI moves. Even indices (0, 2, 4, ...) are opponent moves,
               odd indices (1, 3, 5, ...) are the moves the engine must find.
    """

    id: str
    rating: int
    fen: str
    moves: list[str]


@dataclass
class PuzzleResult:
    """Result of evaluating a single puzzle.

    Attributes:
        puzzle_id: ID of the puzzle.
        rating: Puzzle difficulty rating.
        solved: True if all engine moves were correct.
        moves_correct: Number of correct engine moves.
        moves_total: Total number of engine moves required.
        predicted_moves: List of moves the engine predicted.
        expected_moves: List of expected correct moves.
    """

    puzzle_id: str
    rating: int
    solved: bool
    moves_correct: int
    moves_total: int
    predicted_moves: list[str]
    expected_moves: list[str]


class PuzzleBenchmark:
    """Benchmark that evaluates engines on chess puzzles.

    Puzzle format (Lichess style):
    - The position is given as a FEN string
    - Moves alternate: opponent first, then engine must respond
    - Moves at indices 0, 2, 4, ... are opponent moves
    - Moves at indices 1, 3, 5, ... are engine moves to find

    Example:
        FEN: 1k6/pnp1N3/1pP3p1/4b2p/1N5P/P3q1P1/1P3PBK/8 b - - 0 1
        Moves: e3f2 b4a6 b8a8 c6b7

        1. Opponent plays e3f2
        2. Engine must find b4a6
        3. Opponent plays b8a8
        4. Engine must find c6b7

    For each puzzle, the engine gets credit for each correct move found.
    A puzzle is "solved" if all engine moves are correct.
    """

    def __init__(
        self,
        csv_path: Path | str,
        name: str | None = None,
        *,
        max_puzzles: int | None = None,
    ) -> None:
        """Initialize the puzzle benchmark.

        Args:
            csv_path: Path to the puzzle CSV file.
            name: Optional name for this benchmark (defaults to filename stem).
            max_puzzles: Optional limit on number of puzzles to load.
        """
        self.csv_path = Path(csv_path)
        self._name = name or self.csv_path.stem
        self.max_puzzles = max_puzzles
        self.puzzles = self._load_puzzles()

        logger.info(f"Loaded {len(self.puzzles)} puzzles from {self.csv_path}")

    @property
    def name(self) -> str:
        """Return the benchmark name."""
        return self._name

    def _load_puzzles(self) -> list[Puzzle]:
        """Load puzzles from the CSV file."""
        puzzles = []

        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if self.max_puzzles is not None and i >= self.max_puzzles:
                    break

                puzzles.append(
                    Puzzle(
                        id=row["PuzzleId"],
                        rating=int(row["Rating"]),
                        fen=row["FEN"],
                        moves=row["Moves"].split(),
                    )
                )

        return puzzles

    def run(
        self,
        engine: Engine,
        *,
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run the benchmark on all puzzles.

        Args:
            engine: Chess engine to evaluate.
            show_progress: Whether to show a progress bar.

        Returns:
            BenchmarkResult with accuracy metrics and per-puzzle details.
        """
        results: list[PuzzleResult] = []

        # Running stats for progress bar
        total_solved = 0
        total_moves_correct = 0
        total_moves = 0

        pbar = None
        if show_progress:
            pbar = tqdm(self.puzzles, desc=f"Evaluating {self.name}", unit="puzzle")
            iterator = pbar
        else:
            iterator = self.puzzles

        for puzzle in iterator:
            result = self._evaluate_puzzle(puzzle, engine)
            results.append(result)

            # Update running stats
            if result.solved:
                total_solved += 1
            total_moves_correct += result.moves_correct
            total_moves += result.moves_total

            # Update progress bar with running stats
            if pbar is not None:
                n_puzzles = len(results)
                solve_rate = total_solved / n_puzzles if n_puzzles > 0 else 0.0
                accuracy = total_moves_correct / total_moves if total_moves > 0 else 0.0
                pbar.set_postfix(
                    acc=f"{accuracy:.1%}",
                    solve=f"{solve_rate:.1%}",
                    solved=f"{total_solved}/{n_puzzles}",
                )

        if pbar is not None:
            pbar.close()

        # Compute aggregate metrics
        metrics = self._compute_metrics(results)

        # Add rating bucket metrics
        bucket_metrics = self._compute_rating_buckets(results)
        metrics.update(bucket_metrics)

        return BenchmarkResult(
            name=self.name,
            metrics=metrics,
            details=[asdict(r) for r in results],
            metadata={
                "csv_path": str(self.csv_path),
                "num_puzzles": len(results),
            },
        )

    def _evaluate_puzzle(self, puzzle: Puzzle, engine: Engine) -> PuzzleResult:
        """Evaluate a single puzzle.

        Args:
            puzzle: The puzzle to evaluate.
            engine: The engine to test.

        Returns:
            PuzzleResult with the evaluation outcome.
        """
        board = chess.Board(puzzle.fen)

        moves_correct = 0
        moves_total = 0
        predicted_moves: list[str] = []
        expected_moves: list[str] = []

        for i, uci_move in enumerate(puzzle.moves):
            if i % 2 == 0:
                # Opponent's move - just apply it
                try:
                    board.push_uci(uci_move)
                except ValueError:
                    logger.warning(
                        f"Invalid opponent move {uci_move} in puzzle {puzzle.id}"
                    )
                    break
            else:
                # Engine must find this move
                moves_total += 1
                expected_moves.append(uci_move)

                # Get engine's move
                predicted = engine.select_move(board)

                if predicted is not None:
                    predicted_uci = predicted.uci()
                    predicted_moves.append(predicted_uci)

                    if predicted_uci == uci_move:
                        moves_correct += 1
                else:
                    predicted_moves.append("")

                # Apply the correct move to continue the puzzle
                try:
                    board.push_uci(uci_move)
                except ValueError:
                    logger.warning(
                        f"Invalid expected move {uci_move} in puzzle {puzzle.id}"
                    )
                    break

        return PuzzleResult(
            puzzle_id=puzzle.id,
            rating=puzzle.rating,
            solved=(moves_correct == moves_total and moves_total > 0),
            moves_correct=moves_correct,
            moves_total=moves_total,
            predicted_moves=predicted_moves,
            expected_moves=expected_moves,
        )

    def _compute_metrics(self, results: list[PuzzleResult]) -> dict[str, float]:
        """Compute aggregate metrics from puzzle results."""
        total_puzzles = len(results)
        solved = sum(1 for r in results if r.solved)
        total_moves = sum(r.moves_total for r in results)
        correct_moves = sum(r.moves_correct for r in results)

        return {
            "num_puzzles": float(total_puzzles),
            "num_solved": float(solved),
            "solve_rate": solved / total_puzzles if total_puzzles > 0 else 0.0,
            "move_accuracy": correct_moves / total_moves if total_moves > 0 else 0.0,
            "total_moves": float(total_moves),
            "correct_moves": float(correct_moves),
            "avg_rating": (
                sum(r.rating for r in results) / total_puzzles
                if total_puzzles > 0
                else 0.0
            ),
        }

    def _compute_rating_buckets(
        self, results: list[PuzzleResult]
    ) -> dict[str, float]:
        """Compute accuracy by rating bucket."""
        buckets: dict[str, tuple[int, int]] = {
            "rating_0_1000": (0, 1000),
            "rating_1000_1500": (1000, 1500),
            "rating_1500_2000": (1500, 2000),
            "rating_2000_2500": (2000, 2500),
            "rating_2500_plus": (2500, 10000),
        }

        bucket_stats: dict[str, dict[str, int]] = {
            k: {"correct": 0, "total": 0, "solved": 0, "puzzles": 0} for k in buckets
        }

        for result in results:
            for bucket_name, (lo, hi) in buckets.items():
                if lo <= result.rating < hi:
                    bucket_stats[bucket_name]["correct"] += result.moves_correct
                    bucket_stats[bucket_name]["total"] += result.moves_total
                    bucket_stats[bucket_name]["puzzles"] += 1
                    if result.solved:
                        bucket_stats[bucket_name]["solved"] += 1
                    break

        metrics: dict[str, float] = {}
        for bucket_name, stats in bucket_stats.items():
            if stats["total"] > 0:
                metrics[f"{bucket_name}_accuracy"] = stats["correct"] / stats["total"]
            else:
                metrics[f"{bucket_name}_accuracy"] = 0.0

            if stats["puzzles"] > 0:
                metrics[f"{bucket_name}_solve_rate"] = (
                    stats["solved"] / stats["puzzles"]
                )
            else:
                metrics[f"{bucket_name}_solve_rate"] = 0.0

            metrics[f"{bucket_name}_count"] = float(stats["puzzles"])

        return metrics


def load_puzzle_benchmarks(
    puzzles_dir: Path | str = Path("puzzles"),
) -> dict[str, PuzzleBenchmark]:
    """Load all available puzzle benchmarks from a directory.

    Args:
        puzzles_dir: Directory containing puzzle CSV files.

    Returns:
        Dictionary mapping benchmark names to PuzzleBenchmark instances.
    """
    puzzles_dir = Path(puzzles_dir)
    benchmarks = {}

    for csv_path in puzzles_dir.glob("*.csv"):
        name = csv_path.stem
        benchmarks[name] = PuzzleBenchmark(csv_path, name=name)

    return benchmarks
