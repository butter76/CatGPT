"""Command-line interface for CatGPT."""

import typer
from rich.console import Console

from catgpt import __version__

app = typer.Typer(
    name="catgpt",
    help="CatGPT: ML research toolkit",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print version information."""
    console.print(f"[bold blue]CatGPT[/bold blue] v{__version__}")


@app.command()
def train(
    config: str = typer.Option("base", "--config", "-c", help="Config name to use"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run without training"),
) -> None:
    """Train a model with the specified configuration."""
    console.print(f"[bold green]Training[/bold green] with config: {config}")
    if dry_run:
        console.print("[yellow]Dry run mode - no training will occur[/yellow]")
        return

    # TODO: Implement actual training logic
    console.print("[dim]Training not yet implemented[/dim]")


@app.command()
def evaluate(
    checkpoint: str = typer.Argument(..., help="Path to model checkpoint"),
) -> None:
    """Evaluate a trained model."""
    console.print(f"[bold cyan]Evaluating[/bold cyan] checkpoint: {checkpoint}")
    # TODO: Implement evaluation logic
    console.print("[dim]Evaluation not yet implemented[/dim]")


if __name__ == "__main__":
    app()
