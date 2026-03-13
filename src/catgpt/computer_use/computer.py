"""Computer tool implementation for macOS.

Executes mouse, keyboard, and scroll actions on the local macOS desktop
using `cliclick` for mouse/keyboard and `osascript` for scrolling.

Requires:
    - cliclick: `brew install cliclick`
    - macOS Accessibility permissions for the terminal application
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from catgpt.computer_use.display import ScreenshotManager


@dataclass
class ToolResult:
    """Result from executing a computer tool action."""

    output: str = ""
    error: str | None = None
    base64_image: str | None = None


class ToolError(Exception):
    """Error raised when a tool action fails."""


# Mapping from Claude key names to cliclick key names.
# cliclick uses specific key identifiers for kp: (key press) commands.
CLICLICK_KEY_MAP: dict[str, str] = {
    # Special keys
    "return": "return",
    "enter": "return",
    "tab": "tab",
    "escape": "escape",
    "esc": "escape",
    "delete": "delete",
    "backspace": "delete",
    "forwarddelete": "fwd-delete",
    "space": "space",
    "home": "home",
    "end": "end",
    "pageup": "page-up",
    "page_up": "page-up",
    "pagedown": "page-down",
    "page_down": "page-down",
    # Arrow keys
    "up": "arrow-up",
    "down": "arrow-down",
    "left": "arrow-left",
    "right": "arrow-right",
    "arrowup": "arrow-up",
    "arrowdown": "arrow-down",
    "arrowleft": "arrow-left",
    "arrowright": "arrow-right",
    "arrow_up": "arrow-up",
    "arrow_down": "arrow-down",
    "arrow_left": "arrow-left",
    "arrow_right": "arrow-right",
    # Function keys
    **{f"f{i}": f"f{i}" for i in range(1, 21)},
}

# Modifier keys that cliclick uses with kd:/ku: (key down/up) commands
CLICLICK_MODIFIER_MAP: dict[str, str] = {
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "opt": "alt",
    "shift": "shift",
    "fn": "fn",
}

# Actions that take a coordinate parameter
COORDINATE_ACTIONS = {
    "left_click",
    "right_click",
    "double_click",
    "triple_click",
    "middle_click",
    "mouse_move",
    "left_click_drag",
    "scroll",
}

# Actions that take a text parameter
TEXT_ACTIONS = {"type", "key"}

# Actions that take no parameters
NO_PARAM_ACTIONS = {"screenshot", "cursor_position"}


@dataclass
class ComputerTool:
    """Executes computer use actions on macOS.

    Uses cliclick for mouse/keyboard operations and osascript for scrolling.
    Handles coordinate scaling between Claude's coordinate space and the
    actual screen coordinates.
    """

    screenshot_manager: ScreenshotManager
    action_delay: float = 0.5
    """Delay in seconds after an action before taking a screenshot."""
    typing_delay_ms: int = 12
    """Delay between keystrokes in milliseconds for typing."""
    _cliclick_path: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cliclick_path = shutil.which("cliclick")

    def _require_cliclick(self) -> str:
        """Return the cliclick path, raising an error if not installed."""
        if self._cliclick_path is None:
            raise ToolError(
                "cliclick is not installed. Install it with: brew install cliclick\n"
                "Then grant your terminal Accessibility permissions in:\n"
                "  System Settings → Privacy & Security → Accessibility"
            )
        return self._cliclick_path

    async def execute(self, action: str, **params: Any) -> ToolResult:
        """Execute a computer use action and return the result.

        This is the main entry point for all actions. It dispatches to the
        appropriate handler, waits briefly, then captures a screenshot for
        visual feedback (for most actions).

        Args:
            action: The action type (e.g., "left_click", "type", "screenshot").
            **params: Action-specific parameters (coordinate, text, etc.).

        Returns:
            ToolResult with output, optional error, and optional base64 screenshot.
        """
        try:
            result = await self._dispatch(action, **params)
        except ToolError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            logger.exception(f"Unexpected error executing action '{action}'")
            return ToolResult(error=f"Unexpected error: {e}")

        # Take a screenshot after most actions for visual feedback
        if action not in ("screenshot", "cursor_position"):
            await asyncio.sleep(self.action_delay)
            try:
                base64_img, _ = self.screenshot_manager.take_screenshot()
                result.base64_image = base64_img
            except Exception as e:
                logger.warning(f"Failed to capture post-action screenshot: {e}")

        return result

    async def _dispatch(self, action: str, **params: Any) -> ToolResult:
        """Dispatch an action to the appropriate handler."""
        match action:
            case "screenshot":
                return await self._screenshot()
            case "cursor_position":
                return await self._cursor_position()
            case "left_click":
                return await self._click(params, click_type="c")
            case "right_click":
                return await self._click(params, click_type="rc")
            case "double_click":
                return await self._click(params, click_type="dc")
            case "triple_click":
                return await self._click(params, click_type="tc")
            case "middle_click":
                return await self._middle_click(params)
            case "mouse_move":
                return await self._mouse_move(params)
            case "left_click_drag":
                return await self._left_click_drag(params)
            case "type":
                return await self._type_text(params)
            case "key":
                return await self._key_press(params)
            case "scroll":
                return await self._scroll(params)
            case _:
                raise ToolError(f"Unknown action: {action}")

    async def _screenshot(self) -> ToolResult:
        """Take a screenshot and return it."""
        base64_img, (w, h) = self.screenshot_manager.take_screenshot()
        return ToolResult(
            output=f"Screenshot captured ({w}x{h})",
            base64_image=base64_img,
        )

    async def _cursor_position(self) -> ToolResult:
        """Get the current cursor position."""
        cliclick = self._require_cliclick()
        result = subprocess.run(
            [cliclick, "p:"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise ToolError(f"Failed to get cursor position: {result.stderr}")
        return ToolResult(output=result.stdout.strip())

    def _resolve_coordinate(self, params: dict[str, Any]) -> tuple[int, int]:
        """Extract and scale a coordinate from params."""
        coord = params.get("coordinate")
        if coord is None or not isinstance(coord, (list, tuple)) or len(coord) != 2:
            raise ToolError(
                f"Invalid coordinate: {coord}. Expected [x, y] with two non-negative integers."
            )
        x, y = int(coord[0]), int(coord[1])
        if x < 0 or y < 0:
            raise ToolError(f"Coordinates must be non-negative, got ({x}, {y})")
        return self.screenshot_manager.scale_coordinates(x, y)

    async def _click(self, params: dict[str, Any], click_type: str) -> ToolResult:
        """Execute a click action (left, right, double, triple)."""
        cliclick = self._require_cliclick()
        sx, sy = self._resolve_coordinate(params)
        self._run_cliclick(cliclick, f"{click_type}:{sx},{sy}")
        return ToolResult(output=f"Clicked at ({sx}, {sy}) with type '{click_type}'")

    async def _middle_click(self, params: dict[str, Any]) -> ToolResult:
        """Execute a middle click.

        cliclick doesn't natively support middle click, so we use osascript
        with CGEvent as a workaround.
        """
        sx, sy = self._resolve_coordinate(params)

        # Move mouse first, then use osascript for middle click event
        cliclick = self._require_cliclick()
        self._run_cliclick(cliclick, f"m:{sx},{sy}")

        # Use osascript to generate a middle click via CGEvent
        script = f"""
        use framework "Cocoa"
        set pt to current application's CGPointMake({sx}, {sy})
        set evt to current application's CGEventCreateMouseEvent(missing value, 25, pt, 2)
        current application's CGEventPost(0, evt)
        delay 0.05
        set evt to current application's CGEventCreateMouseEvent(missing value, 26, pt, 2)
        current application's CGEventPost(0, evt)
        """
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            raise ToolError("Middle click timed out")

        return ToolResult(output=f"Middle clicked at ({sx}, {sy})")

    async def _mouse_move(self, params: dict[str, Any]) -> ToolResult:
        """Move the mouse cursor."""
        cliclick = self._require_cliclick()
        sx, sy = self._resolve_coordinate(params)
        self._run_cliclick(cliclick, f"m:{sx},{sy}")
        return ToolResult(output=f"Moved mouse to ({sx}, {sy})")

    async def _left_click_drag(self, params: dict[str, Any]) -> ToolResult:
        """Click and drag from current/start position to the target coordinate."""
        cliclick = self._require_cliclick()

        # The start coordinate is where we start dragging from
        start_coord = params.get("start_coordinate")
        end_coord = params.get("coordinate")

        if end_coord is None or not isinstance(end_coord, (list, tuple)) or len(end_coord) != 2:
            raise ToolError(f"Invalid target coordinate for drag: {end_coord}")

        end_x, end_y = int(end_coord[0]), int(end_coord[1])
        ex, ey = self.screenshot_manager.scale_coordinates(end_x, end_y)

        if start_coord and isinstance(start_coord, (list, tuple)) and len(start_coord) == 2:
            start_x, start_y = int(start_coord[0]), int(start_coord[1])
            sx, sy = self.screenshot_manager.scale_coordinates(start_x, start_y)
        else:
            # Start from current cursor position
            sx, sy = ex, ey  # Will use dd: from current pos

        # cliclick drag: dd = drag down (start), du = drag up (end)
        self._run_cliclick(cliclick, f"dd:{sx},{sy}", f"du:{ex},{ey}")
        return ToolResult(output=f"Dragged from ({sx}, {sy}) to ({ex}, {ey})")

    async def _type_text(self, params: dict[str, Any]) -> ToolResult:
        """Type text using cliclick."""
        cliclick = self._require_cliclick()
        text = params.get("text")
        if text is None or not isinstance(text, str):
            raise ToolError(f"Invalid text parameter: {text}")

        # cliclick t: types text. Use w: for wait between keystrokes
        self._run_cliclick(cliclick, f"t:{text}")
        # Truncate display for long text
        display_text = text if len(text) <= 50 else text[:50] + "..."
        return ToolResult(output=f"Typed: '{display_text}'")

    async def _key_press(self, params: dict[str, Any]) -> ToolResult:
        """Press a key or key combination.

        Handles:
        - Single keys: "Return", "Tab", "Escape", etc.
        - Key combinations: "cmd+a", "ctrl+shift+t", "alt+F4"
        - Multiple keys separated by space: "cmd+a cmd+c"
        """
        cliclick = self._require_cliclick()
        text = params.get("text")
        if text is None or not isinstance(text, str):
            raise ToolError(f"Invalid key parameter: {text}")

        # Parse key combinations (e.g., "cmd+a", "ctrl+shift+t")
        commands = self._parse_key_combo(text)
        self._run_cliclick(cliclick, *commands)
        return ToolResult(output=f"Pressed key(s): '{text}'")

    def _parse_key_combo(self, key_string: str) -> list[str]:
        """Parse a key combination string into cliclick commands.

        Examples:
            "Return" → ["kp:return"]
            "cmd+a" → ["kd:cmd", "kp:a", "ku:cmd"]
            "ctrl+shift+t" → ["kd:ctrl", "kd:shift", "kp:t", "ku:shift", "ku:ctrl"]
            "cmd+a cmd+c" → [..., ...]
        """
        all_commands: list[str] = []

        # Handle space-separated key combos (multiple combos in sequence)
        # But be careful: "space" is a single key, not a separator
        # So we split on spaces only if the combo contains + signs or is a known key
        combos = self._split_key_string(key_string)

        for combo in combos:
            parts = [p.strip().lower() for p in combo.split("+")]
            modifiers: list[str] = []
            keys: list[str] = []

            for part in parts:
                if part in CLICLICK_MODIFIER_MAP:
                    modifiers.append(CLICLICK_MODIFIER_MAP[part])
                elif part in CLICLICK_KEY_MAP:
                    keys.append(CLICLICK_KEY_MAP[part])
                elif len(part) == 1:
                    # Single character — type it as a key press
                    keys.append(part)
                else:
                    # Try case-insensitive lookup
                    lower = part.lower().replace("_", "").replace("-", "")
                    if lower in CLICLICK_KEY_MAP:
                        keys.append(CLICLICK_KEY_MAP[lower])
                    else:
                        # Pass through as-is (cliclick may accept it)
                        keys.append(part)

            # Build cliclick commands: hold modifiers, press keys, release modifiers
            for mod in modifiers:
                all_commands.append(f"kd:{mod}")
            for key in keys:
                all_commands.append(f"kp:{key}")
            for mod in reversed(modifiers):
                all_commands.append(f"ku:{mod}")

        return all_commands

    @staticmethod
    def _split_key_string(key_string: str) -> list[str]:
        """Split a key string into individual combos.

        Handles cases like "cmd+a" (single combo), "Return" (single key),
        "cmd+a cmd+c" (two combos separated by space).
        """
        # If the string contains "+", treat spaces as combo separators
        if "+" in key_string:
            return key_string.split(" ")

        # Single key name — return as-is
        return [key_string]

    async def _scroll(self, params: dict[str, Any]) -> ToolResult:
        """Scroll at a given coordinate.

        Uses osascript with CGEvent for reliable scroll generation on macOS.
        """
        sx, sy = self._resolve_coordinate(params)
        direction = params.get("scroll_direction", "down")
        amount = params.get("scroll_amount", 3)

        if not isinstance(amount, int) or amount <= 0:
            amount = 3

        # Move mouse to the target position first
        cliclick = self._require_cliclick()
        self._run_cliclick(cliclick, f"m:{sx},{sy}")

        # Determine scroll delta (negative = scroll down, positive = scroll up)
        delta = amount if direction == "up" else -amount

        # Try using osascript with CGEvent for scrolling
        self._scroll_via_osascript(delta)

        return ToolResult(output=f"Scrolled {direction} by {amount} at ({sx}, {sy})")

    @staticmethod
    def _scroll_via_osascript(delta: int) -> None:
        """Generate scroll events using osascript and CGEvent."""
        script = f"""
        use framework "Cocoa"
        set evt to current application's CGEventCreateScrollWheelEvent(missing value, 0, 1, {delta})
        current application's CGEventPost(0, evt)
        """
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Scroll via osascript timed out")

    @staticmethod
    def _run_cliclick(cliclick_path: str, *commands: str) -> str:
        """Run a cliclick command and return stdout.

        Args:
            cliclick_path: Path to the cliclick binary.
            *commands: cliclick action commands (e.g., "c:100,200", "t:hello").

        Returns:
            stdout from cliclick.

        Raises:
            ToolError: If the command fails.
        """
        cmd = [cliclick_path, *commands]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise ToolError(f"cliclick failed: {result.stderr.strip()}")
            return result.stdout
        except subprocess.TimeoutExpired:
            raise ToolError(f"cliclick command timed out: {' '.join(commands)}")
        except FileNotFoundError:
            raise ToolError(
                "cliclick not found. Install with: brew install cliclick"
            )
