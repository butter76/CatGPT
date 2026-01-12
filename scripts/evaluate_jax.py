#!/usr/bin/env python3
"""Evaluate JAX models on chess benchmarks.

This script loads trained checkpoints, wraps them in engines, and evaluates
them on various benchmarks (primarily chess puzzles).

Usage:
    # Evaluate on default puzzles benchmark
    uv run python scripts/evaluate_jax.py checkpoints_jax/best

    # Evaluate on specific benchmarks
    uv run python scripts/evaluate_jax.py checkpoints_jax/best -b puzzles -b high_rated_puzzles

    # Evaluate with W&B logging
    uv run python scripts/evaluate_jax.py checkpoints_jax/best --wandb

    # Quick test with limited puzzles
    uv run python scripts/evaluate_jax.py checkpoints_jax/best --max-puzzles 100

    # Compare multiple checkpoints
    uv run python scripts/evaluate_jax.py checkpoints_jax/epoch_50 checkpoints_jax/epoch_100 checkpoints_jax/best
"""

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="Evaluate JAX chess models on benchmarks.",
    add_completion=False,
)
console = Console()

# Default benchmark paths
PUZZLE_BENCHMARKS = {
    "puzzles": Path("puzzles/puzzles.csv"),
    "high_rated_puzzles": Path("puzzles/high_rated_puzzles.csv"),
}


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


@app.command()
def evaluate(
    checkpoints: Annotated[
        list[Path],
        typer.Argument(help="Path(s) to checkpoint directory(ies)"),
    ],
    benchmark: Annotated[
        list[str],
        typer.Option(
            "--benchmark",
            "-b",
            help="Benchmark(s) to run (puzzles, high_rated_puzzles, all)",
        ),
    ] = ["puzzles"],
    max_puzzles: Annotated[
        int | None,
        typer.Option("--max-puzzles", "-n", help="Limit number of puzzles per benchmark"),
    ] = None,
    wandb: Annotated[
        bool,
        typer.Option("--wandb/--no-wandb", help="Log results to Weights & Biases"),
    ] = True,
    wandb_project: Annotated[
        str,
        typer.Option("--wandb-project", help="W&B project name"),
    ] = "catgpt-puzzles",
    wandb_run_name: Annotated[
        str | None,
        typer.Option("--name", help="W&B run name (default: checkpoint name)"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for model evaluation"),
    ] = 256,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """Evaluate checkpoint(s) on chess benchmarks.

    Loads trained JAX models, wraps them in a ValueEngine (1-move lookahead),
    and evaluates on puzzle benchmarks.
    """
    setup_logging(verbose)

    # Validate checkpoints exist
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            logger.error(f"Checkpoint not found: {checkpoint}")
            raise typer.Exit(1)

    # Determine which benchmarks to run
    if "all" in benchmark:
        benchmark_names = list(PUZZLE_BENCHMARKS.keys())
    else:
        benchmark_names = benchmark

    # Validate benchmark names
    for name in benchmark_names:
        if name not in PUZZLE_BENCHMARKS:
            logger.error(f"Unknown benchmark: {name}")
            logger.info(f"Available benchmarks: {list(PUZZLE_BENCHMARKS.keys())}")
            raise typer.Exit(1)
        if not PUZZLE_BENCHMARKS[name].exists():
            logger.error(f"Benchmark file not found: {PUZZLE_BENCHMARKS[name]}")
            raise typer.Exit(1)

    # Import here to avoid slow startup for --help
    from catgpt.jax.evaluation.benchmarks.puzzles import PuzzleBenchmark
    from catgpt.jax.evaluation.engines.value_engine import ValueEngine

    # Initialize W&B if requested
    wandb_run = None
    if wandb:
        try:
            import wandb as wb

            run_name = wandb_run_name
            if run_name is None and len(checkpoints) == 1:
                run_name = checkpoints[0].name

            wandb_run = wb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "checkpoints": [str(c) for c in checkpoints],
                    "benchmarks": benchmark_names,
                    "max_puzzles": max_puzzles,
                },
            )
            logger.info(f"W&B initialized: {wb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            wandb = False

    # Load benchmarks
    benchmarks = {}
    for name in benchmark_names:
        csv_path = PUZZLE_BENCHMARKS[name]
        benchmarks[name] = PuzzleBenchmark(
            csv_path,
            name=name,
            max_puzzles=max_puzzles,
        )

    # Run evaluation for each checkpoint
    all_results: dict[str, dict[str, any]] = {}

    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.name
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        logger.info(f"{'=' * 60}")

        # Load engine
        try:
            engine = ValueEngine.from_checkpoint(checkpoint_path, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            continue

        checkpoint_results = {}

        for bench_name, bench in benchmarks.items():
            logger.info(f"\nRunning {bench_name} ({len(bench.puzzles)} puzzles)...")

            result = bench.run(engine, show_progress=True)
            checkpoint_results[bench_name] = result

            # Log to console
            console.print(f"\n[bold cyan]{bench_name}[/bold cyan] Results:")
            console.print(f"  Move Accuracy: {result.metrics['move_accuracy']:.2%}")
            console.print(f"  Solve Rate: {result.metrics['solve_rate']:.2%}")
            console.print(f"  Puzzles: {int(result.metrics['num_puzzles'])}")

            # Log to W&B
            if wandb and wandb_run:
                import wandb as wb

                prefix = f"{checkpoint_name}/{bench_name}" if len(checkpoints) > 1 else bench_name

                for key, value in result.metrics.items():
                    wb.log({f"{prefix}/{key}": value})

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
                    wb.log({f"{prefix}/details": table})

        all_results[checkpoint_name] = checkpoint_results

    # Print summary table
    _print_summary(all_results, benchmark_names)

    # Finish W&B run
    if wandb and wandb_run:
        import wandb as wb

        wb.finish()

    console.print("\n[bold green]Evaluation complete![/bold green]")


def _print_summary(
    all_results: dict[str, dict[str, any]],
    benchmark_names: list[str],
) -> None:
    """Print a summary table of all results."""
    console.print("\n")
    console.print("[bold]Summary[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Checkpoint")

    for bench_name in benchmark_names:
        table.add_column(f"{bench_name}\nAccuracy", justify="right")
        table.add_column(f"{bench_name}\nSolve Rate", justify="right")

    for checkpoint_name, checkpoint_results in all_results.items():
        row = [checkpoint_name]
        for bench_name in benchmark_names:
            if bench_name in checkpoint_results:
                result = checkpoint_results[bench_name]
                row.append(f"{result.metrics['move_accuracy']:.2%}")
                row.append(f"{result.metrics['solve_rate']:.2%}")
            else:
                row.extend(["N/A", "N/A"])
        table.add_row(*row)

    console.print(table)


@app.command()
def list_benchmarks() -> None:
    """List available benchmarks."""
    console.print("[bold]Available Benchmarks[/bold]\n")

    for name, path in PUZZLE_BENCHMARKS.items():
        exists = "✓" if path.exists() else "✗"
        status = "[green]found[/green]" if path.exists() else "[red]not found[/red]"
        console.print(f"  {exists} {name}: {path} ({status})")


@app.command()
def inspect(
    checkpoint: Annotated[
        Path,
        typer.Argument(help="Path to checkpoint directory"),
    ],
) -> None:
    """Inspect a checkpoint without running evaluation."""
    setup_logging()

    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(1)

    from catgpt.jax.evaluation.checkpoint import load_checkpoint

    try:
        loaded = load_checkpoint(checkpoint)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Checkpoint: {checkpoint}[/bold]\n")

    console.print("[cyan]Model Config:[/cyan]")
    console.print(f"  Name: {loaded.model_config.name}")
    console.print(f"  Hidden Size: {loaded.model_config.hidden_size}")
    console.print(f"  Num Layers: {loaded.model_config.num_layers}")
    console.print(f"  Num Heads: {loaded.model_config.num_heads}")
    console.print(f"  FF Dim: {loaded.model_config.ff_dim}")
    console.print(f"  Vocab Size: {loaded.model_config.vocab_size}")
    console.print(f"  Seq Length: {loaded.model_config.seq_length}")

    console.print("\n[cyan]Tokenizer Config:[/cyan]")
    console.print(f"  Sequence Length: {loaded.tokenizer_config.sequence_length}")
    console.print(f"  Include Halfmove: {loaded.tokenizer_config.include_halfmove}")

    # Count parameters
    import jax

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(loaded.params))
    console.print(f"\n[cyan]Parameters:[/cyan] {param_count:,}")


if __name__ == "__main__":
    app()
