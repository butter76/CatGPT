"""Chess engines for evaluation."""

from catgpt.jax.evaluation.engines.base import Engine
from catgpt.jax.evaluation.engines.value_engine import ValueEngine

__all__ = ["Engine", "ValueEngine"]
