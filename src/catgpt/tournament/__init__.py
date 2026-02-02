"""Tournament module for engine-vs-engine matches with SPRT testing."""

from catgpt.tournament.game_runner import GameResult, GameRunner
from catgpt.tournament.openings import load_openings
from catgpt.tournament.sprt import SPRTCalculator, SPRTResult, SPRTStatus

__all__ = [
    "SPRTCalculator",
    "SPRTResult",
    "SPRTStatus",
    "GameRunner",
    "GameResult",
    "load_openings",
]
