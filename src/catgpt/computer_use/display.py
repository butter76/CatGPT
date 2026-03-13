"""Display detection, screenshot capture, and coordinate scaling for macOS.

Handles Retina displays, screenshot downscaling for optimal Claude vision
performance, and coordinate mapping between Claude's coordinate space and
the actual screen coordinates.
"""

from __future__ import annotations

import base64
import io
import json
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DisplayInfo:
    """Information about the macOS display."""

    width: int
    """Logical pixel width (what the OS reports, e.g. 1440 for a Retina MacBook)."""
    height: int
    """Logical pixel height."""
    scale_factor: float
    """Retina scale factor (2.0 for Retina, 1.0 for non-Retina)."""

    @property
    def physical_width(self) -> int:
        """Physical pixel width (actual screenshot resolution)."""
        return int(self.width * self.scale_factor)

    @property
    def physical_height(self) -> int:
        """Physical pixel height."""
        return int(self.height * self.scale_factor)


def get_display_info() -> DisplayInfo:
    """Detect the primary macOS display resolution and scale factor.

    Uses system_profiler to query display hardware information.
    Falls back to sensible defaults if detection fails.

    Returns:
        DisplayInfo with logical dimensions and scale factor.

    Raises:
        RuntimeError: If not running on macOS.
    """
    if platform.system() != "Darwin":
        raise RuntimeError(
            f"Display detection is only supported on macOS. Current platform: {platform.system()}"
        )

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(result.stdout)

        displays = data.get("SPDisplaysDataType", [])
        for gpu in displays:
            for display in gpu.get("spdisplays_ndrvs", []):
                # Parse resolution string like "1440 x 900 @ 60.00Hz"
                # or get from _spdisplays_pixels / _spdisplays_resolution
                resolution = display.get("_spdisplays_resolution", "")
                retina = display.get("spdisplays_retina", "")

                # Try to parse pixel dimensions
                pixels = display.get("_spdisplays_pixels", "")
                if pixels:
                    # Format: "2560 x 1600" (physical) or similar
                    parts = pixels.replace(" ", "").split("x")
                    if len(parts) == 2:
                        phys_w, phys_h = int(parts[0]), int(parts[1])
                        # Determine scale factor from Retina flag
                        scale = 2.0 if "Yes" in retina else 1.0
                        logical_w = int(phys_w / scale)
                        logical_h = int(phys_h / scale)
                        return DisplayInfo(
                            width=logical_w,
                            height=logical_h,
                            scale_factor=scale,
                        )

                # Fallback: parse resolution string
                if resolution:
                    parts = resolution.split("@")[0].replace(" ", "").split("x")
                    if len(parts) == 2:
                        w, h = int(parts[0]), int(parts[1])
                        scale = 2.0 if "Yes" in retina else 1.0
                        return DisplayInfo(width=w, height=h, scale_factor=scale)

    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError):
        pass

    # Fallback: try using screenresolution or defaults
    return _fallback_display_info()


def _fallback_display_info() -> DisplayInfo:
    """Fallback display detection using Python/AppKit or sensible defaults."""
    try:
        # Try using AppKit (available on macOS with Python framework)
        result = subprocess.run(
            [
                "python3",
                "-c",
                "from AppKit import NSScreen; s = NSScreen.mainScreen(); "
                "f = s.frame(); b = s.backingScaleFactor(); "
                "print(f'{int(f.size.width)},{int(f.size.height)},{b}')",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) == 3:
                return DisplayInfo(
                    width=int(parts[0]),
                    height=int(parts[1]),
                    scale_factor=float(parts[2]),
                )
    except (subprocess.TimeoutExpired, ValueError):
        pass

    # Final fallback: common MacBook Pro resolution
    return DisplayInfo(width=1440, height=900, scale_factor=2.0)


class ScreenshotManager:
    """Manages screenshot capture and scaling for Claude computer use.

    Captures screenshots using macOS native `screencapture`, downscales them
    to the target resolution for Claude, and handles coordinate mapping.
    """

    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 800,
        display_info: DisplayInfo | None = None,
    ) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.display_info = display_info or get_display_info()
        self._screenshot_path = Path(tempfile.gettempdir()) / "claude_cu_screenshot.png"

    def take_screenshot(self) -> tuple[str, tuple[int, int]]:
        """Capture a screenshot and return it as base64 PNG.

        The screenshot is captured at native resolution using macOS screencapture,
        then downscaled to fit within the target dimensions while maintaining
        aspect ratio.

        Returns:
            Tuple of (base64_encoded_png, (width, height)) where dimensions
            are of the scaled image (matching Claude's coordinate space).
        """
        # Capture screenshot silently (-x = no sound)
        subprocess.run(
            ["screencapture", "-x", "-t", "png", str(self._screenshot_path)],
            check=True,
            timeout=10,
        )

        # Open and downscale
        with Image.open(self._screenshot_path) as img:
            scaled = self._scale_image(img)
            return self._image_to_base64(scaled), scaled.size

    def _scale_image(self, img: Image.Image) -> Image.Image:
        """Scale image to fit within target dimensions, maintaining aspect ratio."""
        orig_w, orig_h = img.size

        # Calculate scale factor to fit within target
        scale_w = self.target_width / orig_w
        scale_h = self.target_height / orig_h
        scale = min(scale_w, scale_h, 1.0)  # Never upscale

        if scale >= 1.0:
            return img.copy()

        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        return img.resize((new_w, new_h), Image.LANCZOS)

    @staticmethod
    def _image_to_base64(img: Image.Image) -> str:
        """Convert a PIL Image to base64-encoded PNG string."""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    def scale_coordinates(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int]:
        """Map coordinates from Claude's coordinate space to screen logical coordinates.

        Claude operates in the coordinate space of the downscaled screenshot
        (target_width x target_height). We need to map these back to the
        actual screen's logical pixel coordinates.

        Args:
            x: X coordinate in Claude's space.
            y: Y coordinate in Claude's space.

        Returns:
            Tuple of (screen_x, screen_y) in logical screen pixels.
        """
        # We need to know the actual scaled size to compute the mapping.
        # The screenshot was downscaled from physical pixels to target.
        # Screen logical coords = physical coords / scale_factor
        # Claude coords map to physical coords, then to logical.

        phys_w = self.display_info.physical_width
        phys_h = self.display_info.physical_height

        # Compute the same scale factor used in _scale_image
        scale_w = self.target_width / phys_w
        scale_h = self.target_height / phys_h
        img_scale = min(scale_w, scale_h, 1.0)

        # Actual scaled image size
        if img_scale >= 1.0:
            # No scaling was applied; Claude coords = physical pixel coords
            screen_x = x / self.display_info.scale_factor
            screen_y = y / self.display_info.scale_factor
        else:
            # Map from Claude's scaled space back to physical pixels
            phys_x = x / img_scale
            phys_y = y / img_scale
            # Map from physical to logical
            screen_x = phys_x / self.display_info.scale_factor
            screen_y = phys_y / self.display_info.scale_factor

        return round(screen_x), round(screen_y)
