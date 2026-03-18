"""Claude Computer Use harness for local macOS control.

This module provides a harness that enables Claude Sonnet 4.6 to control
a local macOS computer directly (no VM) via the Anthropic Computer Use API.

The harness implements the agentic loop:
  screenshot → Claude analysis → action execution → repeat

Usage:
    from catgpt.computer_use import ComputerUseHarness

    harness = ComputerUseHarness()
    result = await harness.run("Open Safari and navigate to google.com")
"""

from catgpt.computer_use.harness import ComputerUseHarness

__all__ = ["ComputerUseHarness"]
