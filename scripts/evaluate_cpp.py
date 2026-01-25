#!/usr/bin/env python3
"""Evaluate C++ engines on chess benchmarks.

This script loads C++ UCI engines (catgpt_mcts, catgpt_value, catgpt_policy)
and evaluates them on chess puzzles. The engine binary is kept running
throughout evaluation to avoid expensive TensorRT initialization.

Usage:
    # Evaluate with defaults (prompts for run name)
    uv run python scripts/evaluate_cpp.py

    # Use MCTS engine with custom simulations
    uv run python scripts/evaluate_cpp.py engine.type=mcts engine.mcts.num_simulations=1600

    # Use value or policy engine
    uv run python scripts/evaluate_cpp.py engine.type=value
    uv run python scripts/evaluate_cpp.py engine.type=policy

    # Run on all benchmarks
    uv run python scripts/evaluate_cpp.py benchmark.names=[puzzles,high_rated_puzzles]

    # Quick test with fewer puzzles
    uv run python scripts/evaluate_cpp.py benchmark.max_puzzles=100

    # Parallel evaluation with 4 workers (each loads own engine)
    uv run python scripts/evaluate_cpp.py engine.num_workers=4

    # Disable W&B logging
    uv run python scripts/evaluate_cpp.py wandb.enabled=false
"""

import csv
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()

# Default benchmark paths
PUZZLE_BENCHMARKS = {
    "puzzles": "puzzles_path",
    "high_rated_puzzles": "high_rated_puzzles_path",
}

# Module-level variable for worker processes
_worker_engine = None


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    import sys

    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def prompt_run_name() -> str:
    """Prompt user for a run name to organize W&B runs.

    Returns:
        The run name entered by the user.
    """
    print("\n" + "=" * 50)
    print("CatGPT C++ Engine Evaluation")
    print("=" * 50)
    run_name = input("Enter run name (for W&B): ").strip()
    if not run_name:
        raise ValueError("Run name cannot be empty")
    # Sanitize: replace spaces with underscores, remove problematic chars
    run_name = run_name.replace(" ", "_")
    run_name = "".join(c for c in run_name if c.isalnum() or c in "_-")
    print(f"W&B run name: {run_name}")
    print("=" * 50 + "\n")
    return run_name


@dataclass
class Puzzle:
    """A chess puzzle from the Lichess puzzle database."""

    id: str
    rating: int
    fen: str
    moves: list[str]


@dataclass
class PuzzleResult:
    """Result of evaluating a single puzzle."""

    puzzle_id: str
    rating: int
    solved: bool
    moves_correct: int
    moves_total: int
    predicted_moves: list[str]
    expected_moves: list[str]


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""

    name: str
    metrics: dict[str, float]
    details: list[dict] | None = None
    metadata: dict | None = None


def load_puzzles(csv_path: Path, max_puzzles: int | None = None) -> list[Puzzle]:
    """Load puzzles from CSV file."""
    puzzles = []

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if max_puzzles is not None and i >= max_puzzles:
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


def evaluate_puzzle(puzzle: Puzzle, engine) -> PuzzleResult:
    """Evaluate a single puzzle.

    Args:
        puzzle: The puzzle to evaluate.
        engine: UCI engine instance.

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
                logger.warning(f"Invalid opponent move {uci_move} in puzzle {puzzle.id}")
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
                logger.warning(f"Invalid expected move {uci_move} in puzzle {puzzle.id}")
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


def compute_metrics(results: list[PuzzleResult]) -> dict[str, float]:
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
            sum(r.rating for r in results) / total_puzzles if total_puzzles > 0 else 0.0
        ),
    }


def compute_rating_buckets(results: list[PuzzleResult]) -> dict[str, float]:
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
            metrics[f"{bucket_name}_solve_rate"] = stats["solved"] / stats["puzzles"]
        else:
            metrics[f"{bucket_name}_solve_rate"] = 0.0

        metrics[f"{bucket_name}_count"] = float(stats["puzzles"])

    return metrics


def _config_to_dict(cfg: DictConfig, project_root: Path) -> dict[str, Any]:
    """Convert config to a picklable dict for multiprocessing."""
    return {
        "cpp_build_dir": str(project_root / cfg.cpp_build_dir),
        "trt_engine": str(project_root / cfg.trt_engine),
        "engine": {
            "type": cfg.engine.type,
            "timeout": cfg.engine.timeout,
            "mcts": {
                "num_simulations": cfg.engine.mcts.num_simulations,
            },
        },
    }


def _worker_init(config_dict: dict[str, Any]) -> None:
    """Initialize worker process with UCI engine.

    Called once per worker process at startup.
    """
    global _worker_engine

    from catgpt.cpp.uci_engine import MCTSEngine, PolicyEngine, ValueEngine

    engine_type = config_dict["engine"]["type"]
    cpp_build_dir = Path(config_dict["cpp_build_dir"])
    trt_engine = Path(config_dict["trt_engine"])
    timeout = config_dict["engine"]["timeout"]

    # Map engine type to binary name
    binary_map = {
        "mcts": "catgpt_mcts",
        "value": "catgpt_value",
        "policy": "catgpt_policy",
    }

    binary_path = cpp_build_dir / binary_map[engine_type]

    if engine_type == "mcts":
        num_simulations = config_dict["engine"]["mcts"]["num_simulations"]
        _worker_engine = MCTSEngine(
            binary_path,
            trt_engine,
            num_simulations=num_simulations,
            timeout=timeout,
        )
    elif engine_type == "value":
        _worker_engine = ValueEngine(binary_path, trt_engine, timeout=timeout)
    else:  # policy
        _worker_engine = PolicyEngine(binary_path, trt_engine, timeout=timeout)

    logger.debug(f"Worker initialized with {engine_type} engine")


def _worker_evaluate_puzzle(puzzle_dict: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single puzzle in a worker process.

    Args:
        puzzle_dict: Dictionary with puzzle data.

    Returns:
        Dictionary with PuzzleResult fields.
    """
    global _worker_engine

    if _worker_engine is None:
        raise RuntimeError("Worker engine not initialized")

    puzzle = Puzzle(
        id=puzzle_dict["id"],
        rating=puzzle_dict["rating"],
        fen=puzzle_dict["fen"],
        moves=puzzle_dict["moves"],
    )

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
                break
        else:
            # Engine must find this move
            moves_total += 1
            expected_moves.append(uci_move)

            # Get engine's move
            predicted = _worker_engine.select_move(board)

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
                break

    return {
        "puzzle_id": puzzle.id,
        "rating": puzzle.rating,
        "solved": (moves_correct == moves_total and moves_total > 0),
        "moves_correct": moves_correct,
        "moves_total": moves_total,
        "predicted_moves": predicted_moves,
        "expected_moves": expected_moves,
    }


def run_benchmark(
    engine,
    puzzles: list[Puzzle],
    name: str,
    *,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Run the benchmark on all puzzles.

    Args:
        engine: UCI engine instance.
        puzzles: List of puzzles to evaluate.
        name: Benchmark name.
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
        pbar = tqdm(puzzles, desc=f"Evaluating {name}", unit="puzzle")
        iterator = pbar
    else:
        iterator = puzzles

    for puzzle in iterator:
        result = evaluate_puzzle(puzzle, engine)
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
    metrics = compute_metrics(results)

    # Add rating bucket metrics
    bucket_metrics = compute_rating_buckets(results)
    metrics.update(bucket_metrics)

    return BenchmarkResult(
        name=name,
        metrics=metrics,
        details=[asdict(r) for r in results],
        metadata={"num_puzzles": len(results)},
    )


def run_benchmark_parallel(
    cfg: DictConfig,
    project_root: Path,
    puzzles: list[Puzzle],
    name: str,
    *,
    num_workers: int,
    show_progress: bool = True,
) -> BenchmarkResult:
    """Run the benchmark in parallel across multiple workers.

    Each worker loads its own UCI engine instance and processes puzzles independently.
    This is useful for expensive engines like MCTS where parallelism can significantly
    speed up evaluation.

    Args:
        cfg: Hydra configuration.
        project_root: Path to the project root directory.
        puzzles: List of puzzles to evaluate.
        name: Benchmark name.
        num_workers: Number of parallel workers.
        show_progress: Whether to show a progress bar.

    Returns:
        BenchmarkResult with accuracy metrics and per-puzzle details.
    """
    logger.info(f"Running parallel evaluation with {num_workers} workers")

    # Prepare puzzle data for workers (convert to dicts for pickling)
    puzzle_data = [
        {
            "id": p.id,
            "rating": p.rating,
            "fen": p.fen,
            "moves": p.moves,
        }
        for p in puzzles
    ]

    # Prepare config dict for workers
    config_dict = _config_to_dict(cfg, project_root)

    # Use spawn context for subprocess isolation
    ctx = mp.get_context("spawn")

    results: list[PuzzleResult] = []
    total_solved = 0
    total_moves_correct = 0
    total_moves = 0

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(config_dict,),
    ) as pool:
        # Use imap_unordered for better progress updates
        work_items = puzzle_data

        if show_progress:
            pbar = tqdm(
                total=len(puzzle_data),
                desc=f"Evaluating {name}",
                unit="puzzle",
            )
        else:
            pbar = None

        for result_dict in pool.imap_unordered(_worker_evaluate_puzzle, work_items):
            result = PuzzleResult(**result_dict)
            results.append(result)

            # Update running stats
            if result.solved:
                total_solved += 1
            total_moves_correct += result.moves_correct
            total_moves += result.moves_total

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
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
    metrics = compute_metrics(results)

    # Add rating bucket metrics
    bucket_metrics = compute_rating_buckets(results)
    metrics.update(bucket_metrics)

    return BenchmarkResult(
        name=name,
        metrics=metrics,
        details=[asdict(r) for r in results],
        metadata={"num_puzzles": len(results), "num_workers": num_workers},
    )


def _get_benchmark_path(cfg: DictConfig, name: str) -> Path:
    """Get the file path for a benchmark by name."""
    if name == "puzzles":
        return Path(cfg.benchmark.puzzles_path)
    elif name == "high_rated_puzzles":
        return Path(cfg.benchmark.high_rated_puzzles_path)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def _print_summary(
    all_results: dict[str, dict[str, BenchmarkResult]],
    benchmark_names: list[str],
) -> None:
    """Print a summary table of all results."""
    console.print("\n")
    console.print("[bold]Summary[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Engine")

    for bench_name in benchmark_names:
        table.add_column(f"{bench_name}\nAccuracy", justify="right")
        table.add_column(f"{bench_name}\nSolve Rate", justify="right")

    for engine_name, engine_results in all_results.items():
        row = [engine_name]
        for bench_name in benchmark_names:
            if bench_name in engine_results:
                result = engine_results[bench_name]
                row.append(f"{result.metrics['move_accuracy']:.2%}")
                row.append(f"{result.metrics['solve_rate']:.2%}")
            else:
                row.extend(["N/A", "N/A"])
        table.add_row(*row)

    console.print(table)


def create_engine(cfg: DictConfig, project_root: Path):
    """Create the appropriate UCI engine based on config.

    Args:
        cfg: Hydra configuration.
        project_root: Path to the project root directory.

    Returns:
        UCI engine instance.
    """
    from catgpt.cpp.uci_engine import MCTSEngine, PolicyEngine, ValueEngine

    engine_type = cfg.engine.type
    cpp_build_dir = project_root / cfg.cpp_build_dir
    trt_engine = project_root / cfg.trt_engine
    timeout = cfg.engine.timeout

    # Map engine type to binary name
    binary_map = {
        "mcts": "catgpt_mcts",
        "value": "catgpt_value",
        "policy": "catgpt_policy",
    }

    if engine_type not in binary_map:
        raise ValueError(f"Unknown engine type: {engine_type}")

    binary_path = cpp_build_dir / binary_map[engine_type]

    if not binary_path.exists():
        raise FileNotFoundError(
            f"Engine binary not found: {binary_path}\n"
            f"Please build with: cd cpp/build && cmake .. && make {binary_map[engine_type]} -j$(nproc)"
        )

    if not trt_engine.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {trt_engine}")

    if engine_type == "mcts":
        num_simulations = cfg.engine.mcts.num_simulations
        logger.info(f"Creating MCTS engine with {num_simulations} simulations")
        return MCTSEngine(
            binary_path,
            trt_engine,
            num_simulations=num_simulations,
            timeout=timeout,
        )
    elif engine_type == "value":
        logger.info("Creating Value engine (1-ply lookahead)")
        return ValueEngine(binary_path, trt_engine, timeout=timeout)
    else:  # policy
        logger.info("Creating Policy engine")
        return PolicyEngine(binary_path, trt_engine, timeout=timeout)


@hydra.main(version_base=None, config_path="../configs", config_name="cpp_eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Prompt for run name first (before any logging setup)
    run_name = prompt_run_name()

    # Get project root (Hydra changes cwd)
    project_root = Path(hydra.utils.get_original_cwd())

    # Setup logging
    setup_logging(cfg.verbose)
    logger.info("Starting CatGPT C++ engine evaluation")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate engine type
    valid_engines = ("value", "policy", "mcts")
    if cfg.engine.type not in valid_engines:
        logger.error(f"Invalid engine type: {cfg.engine.type}")
        logger.info(f"Valid options: {valid_engines}")
        raise SystemExit(1)

    # Validate benchmark names
    benchmark_names = list(cfg.benchmark.names)
    for name in benchmark_names:
        if name not in PUZZLE_BENCHMARKS:
            logger.error(f"Unknown benchmark: {name}")
            logger.info(f"Available benchmarks: {list(PUZZLE_BENCHMARKS.keys())}")
            raise SystemExit(1)
        bench_path = project_root / _get_benchmark_path(cfg, name)
        if not bench_path.exists():
            logger.error(f"Benchmark file not found: {bench_path}")
            raise SystemExit(1)

    logger.info(f"Engine type: {cfg.engine.type}")
    if cfg.engine.type == "mcts":
        logger.info(f"MCTS simulations: {cfg.engine.mcts.num_simulations}")

    num_workers = cfg.engine.num_workers
    use_parallel = num_workers > 1
    logger.info(f"Workers: {num_workers}" + (" (parallel)" if use_parallel else ""))

    # Initialize W&B if requested
    wandb_run = None
    if cfg.wandb.enabled:
        try:
            import wandb as wb

            wandb_run = wb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=run_name,
                tags=list(cfg.wandb.tags),
                config={
                    "trt_engine": str(cfg.trt_engine),
                    "benchmarks": benchmark_names,
                    "max_puzzles": cfg.benchmark.max_puzzles,
                    "engine_type": cfg.engine.type,
                    "mcts_simulations": cfg.engine.mcts.num_simulations if cfg.engine.type == "mcts" else None,
                    "num_workers": num_workers,
                },
            )
            logger.info(f"W&B initialized: {wb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            cfg.wandb.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            cfg.wandb.enabled = False

    # Load benchmarks (puzzles only, engine not loaded yet)
    benchmarks: dict[str, list[Puzzle]] = {}
    for name in benchmark_names:
        csv_path = project_root / _get_benchmark_path(cfg, name)
        puzzles = load_puzzles(csv_path, max_puzzles=cfg.benchmark.max_puzzles)
        benchmarks[name] = puzzles
        logger.info(f"Loaded {len(puzzles)} puzzles for {name}")

    # Create engine only for single-worker mode
    # (parallel mode creates engines in worker processes)
    engine = None
    if not use_parallel:
        try:
            engine = create_engine(cfg, project_root)
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            raise SystemExit(1)

    try:
        # Determine engine name for display
        if use_parallel:
            engine_name = f"{cfg.engine.type.upper()}(workers={num_workers})"
            if cfg.engine.type == "mcts":
                engine_name = f"MCTS(nodes={cfg.engine.mcts.num_simulations}, workers={num_workers})"
        else:
            engine_name = engine.name

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating engine: {engine_name}")
        logger.info(f"{'=' * 60}")

        # Run evaluation
        all_results: dict[str, dict[str, BenchmarkResult]] = {}
        engine_results: dict[str, BenchmarkResult] = {}

        for bench_name, puzzles in benchmarks.items():
            logger.info(f"\nRunning {bench_name} ({len(puzzles)} puzzles)...")

            if use_parallel:
                result = run_benchmark_parallel(
                    cfg,
                    project_root,
                    puzzles,
                    bench_name,
                    num_workers=num_workers,
                    show_progress=True,
                )
            else:
                result = run_benchmark(engine, puzzles, bench_name, show_progress=True)
            engine_results[bench_name] = result

            # Log to console
            console.print(f"\n[bold cyan]{bench_name}[/bold cyan] Results:")
            console.print(f"  Move Accuracy: {result.metrics['move_accuracy']:.2%}")
            console.print(f"  Solve Rate: {result.metrics['solve_rate']:.2%}")
            console.print(f"  Puzzles: {int(result.metrics['num_puzzles'])}")

            # Log to W&B
            if cfg.wandb.enabled and wandb_run:
                import wandb as wb

                for key, value in result.metrics.items():
                    wb.log({f"{bench_name}/{key}": value})

                # Log detailed table
                if result.details:
                    table = wb.Table(
                        columns=[
                            "puzzle_id",
                            "rating",
                            "solved",
                            "moves_correct",
                            "moves_total",
                        ]
                    )
                    for detail in result.details:
                        table.add_data(
                            detail["puzzle_id"],
                            detail["rating"],
                            detail["solved"],
                            detail["moves_correct"],
                            detail["moves_total"],
                        )
                    wb.log({f"{bench_name}/details": table})

        all_results[engine_name] = engine_results

        # Print summary table
        _print_summary(all_results, benchmark_names)

    finally:
        # Close the engine if we created one
        if engine is not None:
            engine.close()
            logger.info("Engine closed")

    # Finish W&B run
    if cfg.wandb.enabled and wandb_run:
        import wandb as wb

        wb.finish()

    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
