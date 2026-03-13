"""CLI interface for the Claude Computer Use harness.

Provides `catgpt computer-use run` and `catgpt computer-use check` commands.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="computer-use",
    help="Claude Computer Use harness for local macOS control.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description for Claude to accomplish"),
    model: str = typer.Option(
        "claude-sonnet-4-6",
        "--model",
        "-m",
        help="Claude model to use",
    ),
    width: int = typer.Option(
        1280,
        "--width",
        "-w",
        help="Target display width for Claude (screenshots scaled to this)",
    ),
    height: int = typer.Option(
        800,
        "--height",
        "-h",
        help="Target display height for Claude (screenshots scaled to this)",
    ),
    max_iterations: int = typer.Option(
        50,
        "--max-iterations",
        help="Maximum number of agentic loop iterations",
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip human confirmation prompts for all actions",
    ),
    zoom: bool = typer.Option(
        True,
        "--zoom/--no-zoom",
        help="Enable zoom capability for detailed screen inspection",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar="ANTHROPIC_API_KEY",
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    ),
) -> None:
    """Run Claude Computer Use to accomplish a task on your macOS desktop.

    Claude will take screenshots, analyze the screen, and execute mouse/keyboard
    actions to complete the given task. Each action is shown for confirmation
    unless --auto-approve is set.

    Examples:
        catgpt computer-use run "Open Safari and search for weather"
        catgpt computer-use run --auto-approve "Open Calculator and compute 42 * 17"
        catgpt computer-use run --model claude-opus-4-6 "Write a Python script in VS Code"
    """
    from catgpt.computer_use.harness import ComputerUseHarness

    if auto_approve:
        console.print(
            Panel(
                "[bold yellow]Auto-approve mode enabled.[/bold yellow]\n"
                "Claude will execute actions without confirmation.\n"
                "Press Ctrl+C at any time to stop.",
                border_style="yellow",
            )
        )

    try:
        harness = ComputerUseHarness(
            model=model,
            display_width=width,
            display_height=height,
            enable_zoom=zoom,
            max_iterations=max_iterations,
            auto_approve=auto_approve,
            api_key=api_key,
        )

        result = asyncio.run(harness.run(task))

        console.print("\n")
        console.print(
            Panel(
                result,
                title="[bold green]Final Result[/bold green]",
                border_style="green",
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(code=1) from None
    except ValueError as e:
        console.print(f"\n[red]Configuration error:[/red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command()
def check() -> None:
    """Check that all prerequisites for Claude Computer Use are satisfied.

    Verifies:
    - Running on macOS
    - cliclick is installed
    - ANTHROPIC_API_KEY is set
    - Display can be detected
    - Accessibility permissions guidance
    """
    table = Table(title="Computer Use Prerequisites", show_lines=True)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    all_ok = True

    # 1. macOS check
    is_macos = platform.system() == "Darwin"
    table.add_row(
        "Operating System",
        "[green]PASS[/green]" if is_macos else "[red]FAIL[/red]",
        f"{platform.system()} {platform.release()}"
        if is_macos
        else f"macOS required, got {platform.system()}",
    )
    if not is_macos:
        all_ok = False

    # 2. cliclick check
    cliclick_path = shutil.which("cliclick")
    table.add_row(
        "cliclick",
        "[green]PASS[/green]" if cliclick_path else "[red]FAIL[/red]",
        f"Found at {cliclick_path}"
        if cliclick_path
        else "Not found. Install with: [bold]brew install cliclick[/bold]",
    )
    if not cliclick_path:
        all_ok = False

    # 3. API key check
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    table.add_row(
        "ANTHROPIC_API_KEY",
        "[green]PASS[/green]" if has_api_key else "[red]FAIL[/red]",
        "Environment variable is set"
        if has_api_key
        else "Not set. Export with: [bold]export ANTHROPIC_API_KEY=sk-...[/bold]",
    )
    if not has_api_key:
        all_ok = False

    # 4. Python dependencies check
    deps_ok = True
    missing_deps = []
    for pkg in ("anthropic", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            deps_ok = False
            missing_deps.append(pkg)

    table.add_row(
        "Python Dependencies",
        "[green]PASS[/green]" if deps_ok else "[red]FAIL[/red]",
        "All required packages installed"
        if deps_ok
        else f"Missing: {', '.join(missing_deps)}. Run: [bold]uv sync --extra computer-use[/bold]",
    )
    if not deps_ok:
        all_ok = False

    # 5. Display detection
    display_ok = False
    display_msg = ""
    if is_macos:
        try:
            from catgpt.computer_use.display import get_display_info

            info = get_display_info()
            display_ok = True
            display_msg = (
                f"{info.width}x{info.height} "
                f"(scale={info.scale_factor}x, "
                f"physical={info.physical_width}x{info.physical_height})"
            )
        except Exception as e:
            display_msg = f"Detection failed: {e}"
    else:
        display_msg = "Skipped (not macOS)"

    table.add_row(
        "Display Detection",
        "[green]PASS[/green]" if display_ok else "[yellow]WARN[/yellow]",
        display_msg,
    )

    # 6. Accessibility permissions guidance
    table.add_row(
        "Accessibility Permissions",
        "[yellow]INFO[/yellow]",
        "Grant your terminal app permissions in:\n"
        "System Settings → Privacy & Security → Accessibility\n"
        "(Cannot be checked programmatically)",
    )

    # 7. Screen Recording permissions guidance
    table.add_row(
        "Screen Recording Permissions",
        "[yellow]INFO[/yellow]",
        "Grant your terminal app permissions in:\n"
        "System Settings → Privacy & Security → Screen Recording\n"
        "(Required for screencapture on macOS Sequoia+)",
    )

    console.print(table)

    if all_ok:
        console.print(
            "\n[bold green]All checks passed![/bold green] You're ready to use Claude Computer Use."
        )
    else:
        console.print(
            "\n[bold red]Some checks failed.[/bold red] "
            "Please resolve the issues above before proceeding."
        )
        raise typer.Exit(code=1)
