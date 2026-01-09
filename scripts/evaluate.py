#!/usr/bin/env python3
"""Evaluation script for CatGPT models.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/model.pt
"""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from catgpt.utils.logging import setup_logging

app = typer.Typer()
console = Console()


@app.command()
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Path to model checkpoint"),
    config: str = typer.Option("base", "--config", "-c", help="Config name"),
    split: str = typer.Option("test", "--split", "-s", help="Data split to evaluate"),
) -> None:
    """Evaluate a trained model on a dataset split.

    Args:
        checkpoint: Path to the model checkpoint.
        config: Configuration name to use.
        split: Data split to evaluate on.
    """
    setup_logging()

    if not checkpoint.exists():
        logger.error(f"Checkpoint not found: {checkpoint}")
        raise typer.Exit(1)

    logger.info(f"Evaluating checkpoint: {checkpoint}")
    logger.info(f"Using config: {config}, split: {split}")

    # TODO: Implement actual evaluation
    # 1. Load config
    # 2. Load model from checkpoint
    # 3. Load test data
    # 4. Run evaluation
    # 5. Print metrics

    console.print("[yellow]Evaluation not yet implemented[/yellow]")


if __name__ == "__main__":
    app()
