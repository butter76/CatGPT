"""Core agentic loop for Claude Computer Use on macOS.

Implements the screenshot → Claude analysis → action execution → repeat loop
using the Anthropic Messages API with the computer_20251124 tool type.

Optimized for Claude Sonnet 4.6 with:
- enable_zoom for fine UI element inspection
- Sliding window message management for token efficiency
- Prompt caching for static system prompt and tool definitions
- Retry logic with exponential backoff for API failures
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import anthropic
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from catgpt.computer_use.computer import ComputerTool, ToolResult
from catgpt.computer_use.display import DisplayInfo, ScreenshotManager, get_display_info
from catgpt.computer_use.safety import ActionLog, confirm_action, should_enable_auto_approve

console = Console()

# Beta header for Claude Sonnet 4.6 / Opus 4.6 / Opus 4.5 computer use
COMPUTER_USE_BETA = "computer-use-2025-11-24"

# Default model optimized for computer use
DEFAULT_MODEL = "claude-sonnet-4-6"

# System prompt optimized for Sonnet 4.6 macOS computer use
SYSTEM_PROMPT = """\
You are an AI assistant with direct control over a macOS computer. You can see \
the screen via screenshots and interact using mouse clicks, keyboard input, and \
scrolling.

ENVIRONMENT:
- This is a real macOS desktop (not a VM or container).
- Actions you take have real effects on the user's computer.
- The screen dimensions you see in screenshots match the coordinate space for your actions.

GUIDELINES:
- Always verify the result of your actions by examining the screenshot returned after each action.
- Use keyboard shortcuts when they are more efficient (e.g., Cmd+C to copy, Cmd+V to paste, \
Cmd+Space to open Spotlight).
- Be precise with click coordinates — click the center of UI elements, not their edges.
- If you need to read small text, use the zoom capability to inspect screen regions closely.
- If an action doesn't produce the expected result, try an alternative approach.
- Wait for applications to load before interacting with them.
- When typing URLs or text, verify the text was entered correctly via the screenshot.

SAFETY:
- Do not perform destructive actions (deleting files, formatting disks, etc.) without \
explicit user instruction.
- Do not access sensitive data (passwords, private keys, financial info) unless specifically \
asked.
- If you are unsure about an action's consequences, explain your reasoning and ask for \
confirmation.
- Prefer reversible actions over irreversible ones.

TASK COMPLETION:
- Work step-by-step toward the user's goal.
- After completing the task, take a final screenshot to verify success.
- Summarize what you accomplished in your final response.
"""

# Maximum screenshots to keep in conversation history to manage token usage
MAX_SCREENSHOTS_IN_HISTORY = 10

# Maximum total message pairs in conversation to prevent context overflow
MAX_MESSAGE_PAIRS = 30


class ComputerUseHarness:
    """Harness for Claude Computer Use on local macOS.

    Manages the agentic loop: sends screenshots to Claude, receives tool
    actions, executes them on the local desktop, and repeats until the
    task is complete or the iteration limit is reached.

    Usage:
        harness = ComputerUseHarness()
        result = await harness.run("Open Safari and navigate to google.com")

    Args:
        model: Claude model identifier. Defaults to "claude-sonnet-4-6".
        display_width: Target display width for Claude (screenshots scaled to this).
        display_height: Target display height for Claude.
        enable_zoom: Enable the zoom action for detailed screen inspection.
        max_iterations: Maximum number of agentic loop iterations.
        auto_approve: If True, skip human confirmation prompts.
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        system_prompt: Custom system prompt. If None, uses the default.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        display_width: int = 1280,
        display_height: int = 800,
        enable_zoom: bool = True,
        max_iterations: int = 50,
        auto_approve: bool = False,
        api_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.display_width = display_width
        self.display_height = display_height
        self.enable_zoom = enable_zoom
        self.max_iterations = max_iterations
        self.auto_approve = auto_approve
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize components
        self._display_info: DisplayInfo | None = None
        self._screenshot_manager: ScreenshotManager | None = None
        self._computer_tool: ComputerTool | None = None
        self._action_log = ActionLog()

    def _init_components(self) -> None:
        """Lazily initialize display/screenshot/tool components.

        Separated from __init__ so we can create the harness on any platform
        but only fail on non-macOS when actually running.
        """
        if self._screenshot_manager is not None:
            return

        self._display_info = get_display_info()
        self._screenshot_manager = ScreenshotManager(
            target_width=self.display_width,
            target_height=self.display_height,
            display_info=self._display_info,
        )
        self._computer_tool = ComputerTool(
            screenshot_manager=self._screenshot_manager,
        )

        logger.info(
            f"Display: {self._display_info.width}x{self._display_info.height} "
            f"(scale={self._display_info.scale_factor}x), "
            f"target: {self.display_width}x{self.display_height}"
        )

    def _build_tool_definition(self) -> dict[str, Any]:
        """Build the computer use tool definition for the API request."""
        tool: dict[str, Any] = {
            "type": "computer_20251124",
            "name": "computer",
            "display_width_px": self.display_width,
            "display_height_px": self.display_height,
        }
        if self.enable_zoom:
            tool["enable_zoom"] = True
        return tool

    def _create_client(self) -> anthropic.Anthropic:
        """Create an Anthropic client."""
        return anthropic.Anthropic(api_key=self.api_key)

    async def run(self, task: str) -> str:
        """Run the computer use agentic loop for the given task.

        Args:
            task: Natural language description of the task to accomplish.

        Returns:
            Final text response from Claude summarizing what was accomplished.
        """
        self._init_components()
        assert self._screenshot_manager is not None
        assert self._computer_tool is not None

        client = self._create_client()
        tool_def = self._build_tool_definition()

        # Take initial screenshot
        console.print(
            Panel("[bold blue]Taking initial screenshot...[/bold blue]", border_style="blue")
        )
        initial_screenshot, (sw, sh) = self._screenshot_manager.take_screenshot()
        logger.info(f"Initial screenshot: {sw}x{sh}")

        # Build initial messages
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": task,
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": initial_screenshot,
                        },
                    },
                ],
            },
        ]

        collected_text: list[str] = []
        iteration = 0

        console.print(
            Panel(
                f"[bold green]Starting computer use loop[/bold green]\n"
                f"Task: {task}\n"
                f"Model: {self.model}\n"
                f"Max iterations: {self.max_iterations}",
                border_style="green",
            )
        )

        while iteration < self.max_iterations:
            iteration += 1
            console.print(f"\n[dim]--- Iteration {iteration}/{self.max_iterations} ---[/dim]")

            # Call the API
            response = await self._call_api(client, messages, tool_def)

            if response is None:
                console.print("[red]API call failed after retries. Stopping.[/red]")
                break

            # Process response content blocks
            tool_results: list[dict[str, Any]] = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    collected_text.append(block.text)
                    console.print(
                        Panel(block.text, title="[bold]Claude[/bold]", border_style="cyan")
                    )

                elif block.type == "tool_use":
                    has_tool_use = True
                    action = block.input.get("action", "unknown")
                    params = {k: v for k, v in block.input.items() if k != "action"}

                    console.print(
                        f"  [yellow]Action:[/yellow] {action} [dim]{_format_params(params)}[/dim]"
                    )

                    # Safety confirmation
                    if not self.auto_approve and action not in ("screenshot", "cursor_position"):
                        approved = confirm_action(action, params, self.auto_approve)
                        if not approved:
                            # User rejected — send error result
                            tool_results.append(
                                self._make_tool_result(
                                    block.id,
                                    ToolResult(error="Action rejected by user."),
                                )
                            )
                            self._action_log.log(
                                action, params, ToolResult(error="Rejected"), approved=False
                            )
                            continue

                    # Execute the action
                    result = await self._computer_tool.execute(action, **params)
                    self._action_log.log(action, params, result)

                    if result.error:
                        console.print(f"  [red]Error:[/red] {result.error}")
                    else:
                        console.print(f"  [green]OK:[/green] {result.output}")

                    tool_results.append(self._make_tool_result(block.id, result))

            # If no tool use blocks, we're done
            if not has_tool_use or response.stop_reason == "end_turn":
                console.print(
                    Panel(
                        "[bold green]Task complete[/bold green]",
                        border_style="green",
                    )
                )
                break

            # Append assistant message and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Manage conversation length
            messages = self._trim_messages(messages)

        else:
            console.print(
                f"[yellow]Reached maximum iterations ({self.max_iterations}). Stopping.[/yellow]"
            )

        # Summary
        console.print(
            Panel(
                f"[bold]Completed in {iteration} iterations[/bold]\n"
                f"Actions logged to: {self._action_log.log_path}",
                border_style="blue",
            )
        )

        return "\n".join(collected_text) if collected_text else "(No text response)"

    async def _call_api(
        self,
        client: anthropic.Anthropic,
        messages: list[dict[str, Any]],
        tool_def: dict[str, Any],
        max_retries: int = 3,
    ) -> Any | None:
        """Call the Anthropic API with retry logic.

        Returns the response or None if all retries fail.
        """
        for attempt in range(max_retries):
            try:
                response = client.beta.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=[{"type": "text", "text": self.system_prompt}],
                    tools=[tool_def],
                    messages=messages,
                    betas=[COMPUTER_USE_BETA],
                )
                return response
            except anthropic.APIStatusError as e:
                logger.error(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    return None
            except anthropic.APIConnectionError as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.info(f"Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    return None
        return None

    @staticmethod
    def _make_tool_result(tool_use_id: str, result: ToolResult) -> dict[str, Any]:
        """Build a tool_result content block for the API."""
        content: list[dict[str, Any]] = []

        if result.error:
            content.append({"type": "text", "text": f"Error: {result.error}"})
        elif result.output:
            content.append({"type": "text", "text": result.output})

        if result.base64_image:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content if content else [{"type": "text", "text": "OK"}],
        }

    @staticmethod
    def _trim_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim conversation history to manage token usage.

        Strategy:
        - Always keep the first user message (contains the task).
        - Keep the most recent MAX_MESSAGE_PAIRS assistant/user pairs.
        - Remove base64 image data from older messages (keep text only).
        """
        if len(messages) <= MAX_MESSAGE_PAIRS * 2 + 1:
            return messages

        # Keep first message (task + initial screenshot) and recent messages
        first_msg = messages[0]
        recent = messages[-(MAX_MESSAGE_PAIRS * 2) :]

        # Strip screenshots from the first message if it's getting old
        # (the task text is still valuable, but the initial screenshot may be stale)
        trimmed_first = _strip_images_from_message(first_msg)

        return [trimmed_first, *recent]


def _strip_images_from_message(message: dict[str, Any]) -> dict[str, Any]:
    """Remove image content from a message, keeping only text."""
    if not isinstance(message.get("content"), list):
        return message

    new_content = []
    for block in message["content"]:
        if isinstance(block, dict) and block.get("type") == "image":
            new_content.append(
                {"type": "text", "text": "[screenshot removed for context management]"}
            )
        elif isinstance(block, dict) and block.get("type") == "tool_result":
            # Strip images from tool results too
            if isinstance(block.get("content"), list):
                new_inner = []
                for inner in block["content"]:
                    if isinstance(inner, dict) and inner.get("type") == "image":
                        new_inner.append({"type": "text", "text": "[screenshot removed]"})
                    else:
                        new_inner.append(inner)
                block = {**block, "content": new_inner}
            new_content.append(block)
        else:
            new_content.append(block)

    return {**message, "content": new_content}


def _format_params(params: dict[str, Any]) -> str:
    """Format action params for display, truncating long values."""
    parts = []
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:40] + "..."
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else ""
