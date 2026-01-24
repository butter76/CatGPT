#!/usr/bin/env python3
"""Evaluate JAX models on chess benchmarks.

This script loads trained checkpoints, wraps them in engines, and evaluates
them on various benchmarks (primarily chess puzzles).

Usage:
    # Evaluate with defaults (prompts for run name)
    uv run python scripts/evaluate_jax.py

    # Override checkpoint path
    uv run python scripts/evaluate_jax.py checkpoint=checkpoints_jax/epoch_50

    # Use MCTS engine with custom simulations
    uv run python scripts/evaluate_jax.py engine.type=mcts engine.mcts.num_simulations=1600

    # Run on all benchmarks
    uv run python scripts/evaluate_jax.py benchmark.names=[puzzles,high_rated_puzzles]

    # Quick test with fewer puzzles
    uv run python scripts/evaluate_jax.py benchmark.max_puzzles=100

    # Use highest matmul precision (for ONNX/TensorRT comparison)
    uv run python scripts/evaluate_jax.py compute.matmul_precision=highest

    # Disable W&B logging
    uv run python scripts/evaluate_jax.py wandb.enabled=false
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from catgpt.jax.configs import JaxEvalConfig, jax_eval_config_from_dict

console = Console()

# Default benchmark paths
PUZZLE_BENCHMARKS = {
    "puzzles": "puzzles_path",
    "high_rated_puzzles": "high_rated_puzzles_path",
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


def prompt_run_name() -> str:
    """Prompt user for a run name to organize W&B runs.

    Returns:
        The run name entered by the user.
    """
    print("\n" + "=" * 50)
    print("CatGPT JAX Evaluation")
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


def _get_benchmark_path(cfg: JaxEvalConfig, name: str) -> Path:
    """Get the file path for a benchmark by name."""
    if name == "puzzles":
        return Path(cfg.benchmark.puzzles_path)
    elif name == "high_rated_puzzles":
        return Path(cfg.benchmark.high_rated_puzzles_path)
    else:
        raise ValueError(f"Unknown benchmark: {name}")


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


@hydra.main(version_base=None, config_path="../configs", config_name="jax_eval")
def main(cfg: DictConfig) -> None:
    """Main evaluation entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Prompt for run name first (before any logging setup)
    run_name = prompt_run_name()

    # Convert OmegaConf to typed config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    eval_cfg = jax_eval_config_from_dict(config_dict)  # type: ignore[arg-type]

    # Setup logging
    setup_logging(eval_cfg.verbose)
    logger.info("Starting CatGPT JAX evaluation")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Validate matmul precision
    valid_matmul_precisions = ("high", "highest")
    if eval_cfg.compute.matmul_precision not in valid_matmul_precisions:
        logger.error(f"Invalid matmul precision: {eval_cfg.compute.matmul_precision}")
        logger.info(f"Valid options: {valid_matmul_precisions}")
        raise SystemExit(1)

    # Validate compute dtype
    valid_dtypes = ("float32", "float16", "bfloat16")
    if eval_cfg.compute.compute_dtype not in valid_dtypes:
        logger.error(f"Invalid compute dtype: {eval_cfg.compute.compute_dtype}")
        logger.info(f"Valid options: {valid_dtypes}")
        raise SystemExit(1)

    # Validate engine type
    valid_engines = ("value", "policy", "mcts")
    if eval_cfg.engine.type not in valid_engines:
        logger.error(f"Invalid engine type: {eval_cfg.engine.type}")
        logger.info(f"Valid options: {valid_engines}")
        raise SystemExit(1)

    # Set JAX matmul precision BEFORE any JAX computation
    jax.config.update("jax_default_matmul_precision", eval_cfg.compute.matmul_precision)
    logger.info(f"JAX matmul precision: {eval_cfg.compute.matmul_precision}")

    # Map dtype string to jax dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    compute_dtype_jax = dtype_map[eval_cfg.compute.compute_dtype]
    logger.info(f"Compute dtype: {eval_cfg.compute.compute_dtype}")

    # Get checkpoint path
    checkpoint_path = Path(eval_cfg.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        raise SystemExit(1)

    # Validate benchmark names
    benchmark_names = eval_cfg.benchmark.names
    for name in benchmark_names:
        if name not in PUZZLE_BENCHMARKS:
            logger.error(f"Unknown benchmark: {name}")
            logger.info(f"Available benchmarks: {list(PUZZLE_BENCHMARKS.keys())}")
            raise SystemExit(1)
        bench_path = _get_benchmark_path(eval_cfg, name)
        if not bench_path.exists():
            logger.error(f"Benchmark file not found: {bench_path}")
            raise SystemExit(1)

    # Import here to avoid slow startup for --help
    from catgpt.jax.evaluation.benchmarks.puzzles import PuzzleBenchmark
    from catgpt.jax.evaluation.engines.mcts import MCTSConfig, MCTSEngine
    from catgpt.jax.evaluation.engines.policy_engine import PolicyEngine
    from catgpt.jax.evaluation.engines.value_engine import ValueEngine

    logger.info(f"Engine type: {eval_cfg.engine.type}")
    if eval_cfg.engine.type == "mcts":
        logger.info(
            f"MCTS simulations: {eval_cfg.engine.mcts.num_simulations}, "
            f"c_puct: {eval_cfg.engine.mcts.c_puct}"
        )

    # Initialize W&B if requested
    wandb_run = None
    if eval_cfg.wandb.enabled:
        try:
            import wandb as wb

            wandb_run = wb.init(
                project=eval_cfg.wandb.project,
                entity=eval_cfg.wandb.entity,
                name=run_name,
                tags=eval_cfg.wandb.tags,
                config={
                    "checkpoint": str(checkpoint_path),
                    "benchmarks": benchmark_names,
                    "max_puzzles": eval_cfg.benchmark.max_puzzles,
                    "matmul_precision": eval_cfg.compute.matmul_precision,
                    "compute_dtype": eval_cfg.compute.compute_dtype,
                    "engine_type": eval_cfg.engine.type,
                    "engine_batch_size": eval_cfg.engine.batch_size,
                    "mcts_simulations": eval_cfg.engine.mcts.num_simulations,
                    "mcts_c_puct": eval_cfg.engine.mcts.c_puct,
                },
            )
            logger.info(f"W&B initialized: {wb.run.url}")
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
            eval_cfg.wandb.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            eval_cfg.wandb.enabled = False

    # Load benchmarks
    benchmarks = {}
    for name in benchmark_names:
        csv_path = _get_benchmark_path(eval_cfg, name)
        benchmarks[name] = PuzzleBenchmark(
            csv_path,
            name=name,
            max_puzzles=eval_cfg.benchmark.max_puzzles,
        )

    # Load engine
    checkpoint_name = checkpoint_path.name
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    logger.info(f"{'=' * 60}")

    try:
        if eval_cfg.engine.type == "value":
            engine = ValueEngine.from_checkpoint(
                checkpoint_path,
                batch_size=eval_cfg.engine.batch_size,
                compute_dtype=compute_dtype_jax,
            )
        elif eval_cfg.engine.type == "policy":
            engine = PolicyEngine.from_checkpoint(
                checkpoint_path,
                batch_size=eval_cfg.engine.batch_size,
                compute_dtype=compute_dtype_jax,
            )
        else:  # engine_type == "mcts"
            mcts_config = MCTSConfig(
                num_simulations=eval_cfg.engine.mcts.num_simulations,
                c_puct=eval_cfg.engine.mcts.c_puct,
                fpu_value=eval_cfg.engine.mcts.fpu_value,
            )
            engine = MCTSEngine.from_checkpoint(
                checkpoint_path,
                config=mcts_config,
                batch_size=1,  # MCTS evaluates one position at a time
                compute_dtype=compute_dtype_jax,
            )
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise SystemExit(1)

    # Run evaluation
    all_results: dict[str, dict[str, any]] = {}
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
        if eval_cfg.wandb.enabled and wandb_run:
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

    all_results[checkpoint_name] = checkpoint_results

    # Print summary table
    _print_summary(all_results, benchmark_names)

    # Finish W&B run
    if eval_cfg.wandb.enabled and wandb_run:
        import wandb as wb

        wb.finish()

    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
