"""Tests for the Claude Computer Use harness.

Covers:
- Coordinate scaling math
- Key mapping (Claude key names → cliclick syntax)
- Action dispatching logic
- Display info parsing
- Safety confirmation logic (mocked)
- Message construction for the API
- Agentic loop with mocked API responses
"""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from catgpt.computer_use.computer import (
    CLICLICK_KEY_MAP,
    CLICLICK_MODIFIER_MAP,
    ComputerTool,
    ToolError,
    ToolResult,
)
from catgpt.computer_use.display import DisplayInfo, ScreenshotManager
from catgpt.computer_use.harness import (
    COMPUTER_USE_BETA,
    DEFAULT_MODEL,
    ComputerUseHarness,
    _format_params,
    _strip_images_from_message,
)
from catgpt.computer_use.safety import ActionLog, _describe_action, _sanitize_params

# =============================================================================
# DisplayInfo tests
# =============================================================================


class TestDisplayInfo:
    def test_basic_properties(self) -> None:
        info = DisplayInfo(width=1440, height=900, scale_factor=2.0)
        assert info.width == 1440
        assert info.height == 900
        assert info.scale_factor == 2.0
        assert info.physical_width == 2880
        assert info.physical_height == 1800

    def test_non_retina(self) -> None:
        info = DisplayInfo(width=1920, height=1080, scale_factor=1.0)
        assert info.physical_width == 1920
        assert info.physical_height == 1080

    def test_frozen(self) -> None:
        info = DisplayInfo(width=1440, height=900, scale_factor=2.0)
        with pytest.raises(AttributeError):
            info.width = 1920  # type: ignore[misc]


# =============================================================================
# ScreenshotManager coordinate scaling tests
# =============================================================================


class TestCoordinateScaling:
    def test_identity_no_scaling(self) -> None:
        """When target matches physical resolution, coordinates map directly."""
        info = DisplayInfo(width=1280, height=800, scale_factor=1.0)
        mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        x, y = mgr.scale_coordinates(640, 400)
        assert x == 640
        assert y == 400

    def test_retina_2x_no_downscale(self) -> None:
        """Retina 2x display, target larger than physical: no scaling applied.

        Physical = 2880x1800. Target = 3000x3000 (larger → no downscale).
        Claude coords = physical coords. Screen logical = physical / 2.
        """
        info = DisplayInfo(width=1440, height=900, scale_factor=2.0)
        mgr = ScreenshotManager(target_width=3000, target_height=3000, display_info=info)
        x, y = mgr.scale_coordinates(1440, 900)
        # Physical coords = Claude coords = (1440, 900)
        # Logical coords = (1440/2, 900/2) = (720, 450)
        assert x == 720
        assert y == 450

    def test_downscale_with_retina(self) -> None:
        """Retina 2x display with downscaling to 1280x800.

        Physical = 2880x1800. Scale to 1280x800:
          scale_w = 1280/2880 ≈ 0.444
          scale_h = 800/1800 ≈ 0.444
          img_scale = min(0.444, 0.444) = 0.444

        Claude coordinate (640, 400):
          phys = (640/0.444, 400/0.444) = (1440, 900)
          logical = (1440/2, 900/2) = (720, 450)
        """
        info = DisplayInfo(width=1440, height=900, scale_factor=2.0)
        mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        x, y = mgr.scale_coordinates(640, 400)
        assert x == 720
        assert y == 450

    def test_corner_coordinates(self) -> None:
        """Origin (0, 0) should always map to (0, 0)."""
        info = DisplayInfo(width=1440, height=900, scale_factor=2.0)
        mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        x, y = mgr.scale_coordinates(0, 0)
        assert x == 0
        assert y == 0

    def test_non_retina_downscale(self) -> None:
        """Non-Retina display with downscaling."""
        info = DisplayInfo(width=1920, height=1080, scale_factor=1.0)
        mgr = ScreenshotManager(target_width=1280, target_height=720, display_info=info)
        # Physical = 1920x1080. Scale to 1280x720:
        # scale_w = 1280/1920 ≈ 0.667, scale_h = 720/1080 ≈ 0.667
        # Claude coord (640, 360) → phys (960, 540) → logical (960, 540)
        x, y = mgr.scale_coordinates(640, 360)
        assert x == 960
        assert y == 540


# =============================================================================
# Screenshot capture and image scaling tests
# =============================================================================


class TestScreenshotManager:
    def test_scale_image_no_upscale(self) -> None:
        """Image smaller than target should not be upscaled."""
        info = DisplayInfo(width=800, height=600, scale_factor=1.0)
        mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        img = Image.new("RGB", (800, 600), "red")
        result = mgr._scale_image(img)
        assert result.size == (800, 600)

    def test_scale_image_downscale(self) -> None:
        """Image larger than target should be downscaled maintaining aspect ratio."""
        info = DisplayInfo(width=1920, height=1080, scale_factor=1.0)
        mgr = ScreenshotManager(target_width=1280, target_height=720, display_info=info)
        img = Image.new("RGB", (1920, 1080), "blue")
        result = mgr._scale_image(img)
        # Scale = min(1280/1920, 720/1080) = min(0.667, 0.667) = 0.667
        # New size = (1280, 720)
        assert result.size == (1280, 720)

    def test_image_to_base64(self) -> None:
        """Base64 encoding should produce valid PNG data."""
        img = Image.new("RGB", (100, 100), "green")
        b64 = ScreenshotManager._image_to_base64(img)
        # Decode and verify it's valid PNG
        data = base64.b64decode(b64)
        result = Image.open(io.BytesIO(data))
        assert result.size == (100, 100)


# =============================================================================
# Key mapping tests
# =============================================================================


class TestKeyMapping:
    def test_common_keys_mapped(self) -> None:
        """All common special keys should have mappings."""
        for key in ["return", "enter", "tab", "escape", "delete", "space"]:
            assert key in CLICLICK_KEY_MAP, f"Missing key mapping: {key}"

    def test_arrow_keys_mapped(self) -> None:
        for key in ["up", "down", "left", "right", "arrowup", "arrowdown"]:
            assert key in CLICLICK_KEY_MAP

    def test_function_keys_mapped(self) -> None:
        for i in range(1, 13):
            assert f"f{i}" in CLICLICK_KEY_MAP

    def test_modifiers_mapped(self) -> None:
        for mod in ["cmd", "command", "ctrl", "control", "alt", "option", "shift"]:
            assert mod in CLICLICK_MODIFIER_MAP


class TestKeyComboParser:
    def setup_method(self) -> None:
        info = DisplayInfo(width=1280, height=800, scale_factor=1.0)
        mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        self.tool = ComputerTool(screenshot_manager=mgr)

    def test_single_key(self) -> None:
        cmds = self.tool._parse_key_combo("Return")
        assert cmds == ["kp:return"]

    def test_single_modifier_key_combo(self) -> None:
        cmds = self.tool._parse_key_combo("cmd+a")
        assert cmds == ["kd:cmd", "kp:a", "ku:cmd"]

    def test_double_modifier_combo(self) -> None:
        cmds = self.tool._parse_key_combo("ctrl+shift+t")
        assert cmds == ["kd:ctrl", "kd:shift", "kp:t", "ku:shift", "ku:ctrl"]

    def test_multiple_combos(self) -> None:
        cmds = self.tool._parse_key_combo("cmd+a cmd+c")
        assert cmds == ["kd:cmd", "kp:a", "ku:cmd", "kd:cmd", "kp:c", "ku:cmd"]

    def test_escape_key(self) -> None:
        cmds = self.tool._parse_key_combo("Escape")
        assert cmds == ["kp:escape"]

    def test_space_key(self) -> None:
        # "space" as a single key name (not a separator)
        cmds = self.tool._parse_key_combo("space")
        assert cmds == ["kp:space"]


# =============================================================================
# ComputerTool action dispatching tests
# =============================================================================


class TestComputerToolActions:
    def setup_method(self) -> None:
        info = DisplayInfo(width=1280, height=800, scale_factor=1.0)
        self.mgr = ScreenshotManager(target_width=1280, target_height=800, display_info=info)
        self.tool = ComputerTool(screenshot_manager=self.mgr)

    def test_resolve_coordinate_valid(self) -> None:
        x, y = self.tool._resolve_coordinate({"coordinate": [100, 200]})
        assert x == 100
        assert y == 200

    def test_resolve_coordinate_missing(self) -> None:
        with pytest.raises(ToolError, match="Invalid coordinate"):
            self.tool._resolve_coordinate({})

    def test_resolve_coordinate_wrong_length(self) -> None:
        with pytest.raises(ToolError, match="Invalid coordinate"):
            self.tool._resolve_coordinate({"coordinate": [100]})

    def test_resolve_coordinate_negative(self) -> None:
        with pytest.raises(ToolError, match="non-negative"):
            self.tool._resolve_coordinate({"coordinate": [-1, 200]})

    @pytest.mark.asyncio
    async def test_type_text_invalid(self) -> None:
        # Mock cliclick as available so we reach parameter validation
        self.tool._cliclick_path = "/usr/local/bin/cliclick"
        result = await self.tool.execute("type")
        assert result.error is not None
        assert "Invalid text" in result.error

    @pytest.mark.asyncio
    async def test_key_press_invalid(self) -> None:
        # Mock cliclick as available so we reach parameter validation
        self.tool._cliclick_path = "/usr/local/bin/cliclick"
        result = await self.tool.execute("key")
        assert result.error is not None
        assert "Invalid key" in result.error

    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        result = await self.tool.execute("nonexistent_action")
        assert result.error is not None
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_screenshot_action(self) -> None:
        """Screenshot action should capture and return base64 image."""
        # Create a fake screenshot file
        fake_img = Image.new("RGB", (1280, 800), "blue")
        fake_b64 = ScreenshotManager._image_to_base64(fake_img)

        with patch.object(self.mgr, "take_screenshot", return_value=(fake_b64, (1280, 800))):
            result = await self.tool.execute("screenshot")
            assert result.error is None
            assert result.base64_image is not None
            assert "1280x800" in result.output


# =============================================================================
# Safety layer tests
# =============================================================================


class TestSafetyLayer:
    def test_sanitize_params(self) -> None:
        params = {"coordinate": [100, 200], "text": "hello", "nested": {"a": 1}}
        sanitized = _sanitize_params(params)
        assert sanitized["coordinate"] == [100, 200]
        assert sanitized["text"] == "hello"
        assert isinstance(sanitized["nested"], str)

    def test_describe_action_click(self) -> None:
        desc = _describe_action("left_click", {"coordinate": [100, 200]})
        assert "Left click" in desc
        assert "[100, 200]" in desc

    def test_describe_action_type(self) -> None:
        desc = _describe_action("type", {"text": "hello world"})
        assert "Type text" in desc
        assert "hello world" in desc

    def test_describe_action_unknown(self) -> None:
        desc = _describe_action("new_action", {"foo": "bar"})
        assert "new_action" in desc

    def test_action_log(self, tmp_path) -> None:
        log = ActionLog(log_path=tmp_path / "test_log.jsonl")
        result = ToolResult(output="clicked", error=None, base64_image="abc123")
        log.log("left_click", {"coordinate": [100, 200]}, result)

        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry["action"] == "left_click"
        assert entry["approved"] is True
        assert entry["has_screenshot"] is True

        # Verify file was written
        assert log.log_path is not None
        lines = log.log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["action"] == "left_click"


# =============================================================================
# Harness tests (mocked API)
# =============================================================================


class TestHarness:
    def test_tool_definition(self) -> None:
        harness = ComputerUseHarness(
            display_width=1280,
            display_height=800,
            enable_zoom=True,
            api_key="test-key",
        )
        tool_def = harness._build_tool_definition()
        assert tool_def["type"] == "computer_20251124"
        assert tool_def["name"] == "computer"
        assert tool_def["display_width_px"] == 1280
        assert tool_def["display_height_px"] == 800
        assert tool_def["enable_zoom"] is True

    def test_tool_definition_no_zoom(self) -> None:
        harness = ComputerUseHarness(
            enable_zoom=False,
            api_key="test-key",
        )
        tool_def = harness._build_tool_definition()
        assert "enable_zoom" not in tool_def

    def test_make_tool_result_with_image(self) -> None:
        result = ToolResult(
            output="Clicked at (100, 200)",
            base64_image="base64data",
        )
        tr = ComputerUseHarness._make_tool_result("tool123", result)
        assert tr["type"] == "tool_result"
        assert tr["tool_use_id"] == "tool123"
        assert len(tr["content"]) == 2
        assert tr["content"][0]["type"] == "text"
        assert tr["content"][1]["type"] == "image"
        assert tr["content"][1]["source"]["data"] == "base64data"

    def test_make_tool_result_error(self) -> None:
        result = ToolResult(error="Something went wrong")
        tr = ComputerUseHarness._make_tool_result("tool456", result)
        assert tr["content"][0]["text"].startswith("Error:")

    def test_make_tool_result_empty(self) -> None:
        result = ToolResult()
        tr = ComputerUseHarness._make_tool_result("tool789", result)
        assert tr["content"][0]["text"] == "OK"

    def test_strip_images_from_message(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Do something"},
                {
                    "type": "image",
                    "source": {"type": "base64", "data": "big_image_data"},
                },
            ],
        }
        stripped = _strip_images_from_message(msg)
        assert len(stripped["content"]) == 2
        assert stripped["content"][0]["type"] == "text"
        assert stripped["content"][1]["type"] == "text"
        assert "removed" in stripped["content"][1]["text"]

    def test_trim_messages_short(self) -> None:
        """Short message lists should not be trimmed."""
        messages = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "response"},
            {"role": "user", "content": "followup"},
        ]
        result = ComputerUseHarness._trim_messages(messages)
        assert len(result) == 3

    def test_model_defaults(self) -> None:
        harness = ComputerUseHarness(api_key="test-key")
        assert harness.model == DEFAULT_MODEL
        assert harness.display_width == 1280
        assert harness.display_height == 800
        assert harness.enable_zoom is True
        assert harness.max_iterations == 50

    def test_missing_api_key(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="API key required"),
        ):
            ComputerUseHarness(api_key=None)


class TestHarnessAgenticLoop:
    """Integration tests for the agentic loop with mocked API."""

    @pytest.mark.asyncio
    async def test_single_text_response(self) -> None:
        """API returns text only (no tool use) — loop should complete immediately."""
        harness = ComputerUseHarness(api_key="test-key")

        # Mock display/screenshot components
        fake_img = Image.new("RGB", (1280, 800), "blue")
        fake_b64 = ScreenshotManager._image_to_base64(fake_img)

        mock_display_info = DisplayInfo(width=1280, height=800, scale_factor=1.0)
        mock_screenshot_mgr = MagicMock(spec=ScreenshotManager)
        mock_screenshot_mgr.take_screenshot.return_value = (fake_b64, (1280, 800))
        mock_screenshot_mgr.scale_coordinates.side_effect = lambda x, y: (x, y)

        harness._display_info = mock_display_info
        harness._screenshot_manager = mock_screenshot_mgr
        harness._computer_tool = ComputerTool(screenshot_manager=mock_screenshot_mgr)

        # Mock API response — text only, no tool use
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "I have completed the task."

        mock_response = MagicMock()
        mock_response.content = [mock_text_block]
        mock_response.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.beta.messages.create.return_value = mock_response

        with patch.object(harness, "_create_client", return_value=mock_client):
            result = await harness.run("Test task")

        assert "completed the task" in result

    @pytest.mark.asyncio
    async def test_tool_use_then_end(self) -> None:
        """API returns tool_use, then end_turn — loop should complete in 2 iterations."""
        harness = ComputerUseHarness(api_key="test-key", auto_approve=True)

        fake_img = Image.new("RGB", (1280, 800), "blue")
        fake_b64 = ScreenshotManager._image_to_base64(fake_img)

        mock_display_info = DisplayInfo(width=1280, height=800, scale_factor=1.0)
        mock_screenshot_mgr = MagicMock(spec=ScreenshotManager)
        mock_screenshot_mgr.take_screenshot.return_value = (fake_b64, (1280, 800))
        mock_screenshot_mgr.scale_coordinates.side_effect = lambda x, y: (x, y)

        harness._display_info = mock_display_info
        harness._screenshot_manager = mock_screenshot_mgr
        harness._computer_tool = ComputerTool(screenshot_manager=mock_screenshot_mgr)

        # First response: tool_use (screenshot)
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_1"
        mock_tool_block.input = {"action": "screenshot"}

        mock_response_1 = MagicMock()
        mock_response_1.content = [mock_tool_block]
        mock_response_1.stop_reason = "tool_use"

        # Second response: text (done)
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = "Done! I took a screenshot."

        mock_response_2 = MagicMock()
        mock_response_2.content = [mock_text_block]
        mock_response_2.stop_reason = "end_turn"

        mock_client = MagicMock()
        mock_client.beta.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
        ]

        with patch.object(harness, "_create_client", return_value=mock_client):
            result = await harness.run("Take a screenshot")

        assert "screenshot" in result.lower()
        assert mock_client.beta.messages.create.call_count == 2


# =============================================================================
# Utility function tests
# =============================================================================


class TestUtilities:
    def test_format_params_empty(self) -> None:
        assert _format_params({}) == ""

    def test_format_params_simple(self) -> None:
        result = _format_params({"coordinate": [100, 200]})
        assert "coordinate=[100, 200]" in result

    def test_format_params_truncation(self) -> None:
        long_text = "a" * 100
        result = _format_params({"text": long_text})
        assert "..." in result

    def test_beta_header_value(self) -> None:
        assert COMPUTER_USE_BETA == "computer-use-2025-11-24"

    def test_default_model(self) -> None:
        assert DEFAULT_MODEL == "claude-sonnet-4-6"
