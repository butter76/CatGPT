"""Safety layer for Claude computer use.

Provides human-in-the-loop confirmation prompts and action logging
to ensure safe operation when Claude controls a real macOS desktop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from catgpt.computer_use.computer import ToolResult

console = Console()


@dataclass
class ActionLog:
    """Structured logging of all computer use actions."""

    log_path: Path | None = None
    _entries: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.log_path is None:
            self.log_path = Path.home() / ".catgpt" / "computer_use_actions.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: str,
        params: dict[str, Any],
        result: ToolResult,
        approved: bool = True,
    ) -> None:
        """Log an action execution."""
        entry = {
            "timestamp": time.time(),
            "action": action,
            "params": _sanitize_params(params),
            "approved": approved,
            "output": result.output,
            "error": result.error,
            "has_screenshot": result.base64_image is not None,
        }
        self._entries.append(entry)

        # Append to log file
        assert self.log_path is not None
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            logger.warning(f"Failed to write action log: {e}")

    @property
    def entries(self) -> list[dict[str, Any]]:
        """Get all logged entries."""
        return list(self._entries)


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serializable params (like base64 image data) for logging."""
    sanitized = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, (list, tuple)):
            sanitized[k] = list(v)
        else:
            sanitized[k] = str(v)
    return sanitized


def confirm_action(
    action: str,
    params: dict[str, Any],
    auto_approve: bool = False,
) -> bool:
    """Request human confirmation before executing an action.

    Args:
        action: The action type being executed.
        params: The action parameters.
        auto_approve: If True, always approve without prompting.

    Returns:
        True if the action is approved, False otherwise.
    """
    if auto_approve:
        return True

    # Build a human-readable description of the action
    description = _describe_action(action, params)

    panel = Panel(
        Text.from_markup(description),
        title="[bold yellow]Action Confirmation[/bold yellow]",
        border_style="yellow",
    )
    console.print(panel)

    try:
        response = console.input("[bold]Approve? [y/N/a(lways)]: [/bold]").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[red]Action cancelled.[/red]")
        return False

    if response in ("a", "always"):
        # Signal to caller to enable auto_approve going forward
        # We return True and the caller checks for this special case
        console.print("[green]Auto-approving all subsequent actions.[/green]")
        return True  # Caller should check for 'always' mode separately

    return response in ("y", "yes")


def should_enable_auto_approve(response_text: str) -> bool:
    """Check if the user's confirmation response indicates 'always approve'."""
    return response_text.strip().lower() in ("a", "always")


def _describe_action(action: str, params: dict[str, Any]) -> str:
    """Create a human-readable description of an action."""
    coord = params.get("coordinate")
    text = params.get("text")

    descriptions: dict[str, str] = {
        "left_click": f"[cyan]Left click[/cyan] at ({coord})",
        "right_click": f"[cyan]Right click[/cyan] at ({coord})",
        "double_click": f"[cyan]Double click[/cyan] at ({coord})",
        "triple_click": f"[cyan]Triple click[/cyan] at ({coord})",
        "middle_click": f"[cyan]Middle click[/cyan] at ({coord})",
        "mouse_move": f"[cyan]Move mouse[/cyan] to ({coord})",
        "left_click_drag": f"[cyan]Drag[/cyan] to ({coord})",
        "type": f"[cyan]Type text:[/cyan] '{text}'",
        "key": f"[cyan]Press key(s):[/cyan] {text}",
        "scroll": (
            f"[cyan]Scroll[/cyan] {params.get('scroll_direction', 'down')} "
            f"by {params.get('scroll_amount', 3)} at ({coord})"
        ),
        "screenshot": "[cyan]Take screenshot[/cyan]",
        "cursor_position": "[cyan]Get cursor position[/cyan]",
    }

    return descriptions.get(action, f"[cyan]{action}[/cyan] with params: {params}")
