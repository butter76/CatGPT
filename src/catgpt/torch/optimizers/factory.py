"""Factory function for creating optimizers from config."""

from typing import TYPE_CHECKING

import torch
from torch.optim import Optimizer

from catgpt.torch.optimizers.splus import SPlus

if TYPE_CHECKING:
    from catgpt.core.configs.schema import OptimizerConfig


def create_optimizer(
    params,
    config: "OptimizerConfig",
) -> Optimizer:
    """Create an optimizer from configuration.

    Args:
        params: Model parameters (from model.parameters()).
        config: Optimizer configuration.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    name = config.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )

    if name == "splus":
        return SPlus(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            b1=config.splus_b1,
            b2=config.splus_b2,
            ema_rate=config.splus_ema_rate,
            inverse_every=config.splus_inverse_every,
            max_dim=config.splus_max_dim,
        )

    msg = f"Unknown optimizer: {name}. Choose from: adamw, splus"
    raise ValueError(msg)


def create_scheduler(
    optimizer: Optimizer,
    config: "SchedulerConfig",  # noqa: F821
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule.
        config: Scheduler configuration.
        total_steps: Total number of training steps.

    Returns:
        Configured scheduler, or None if name is "constant".

    Raises:
        ValueError: If scheduler name is not recognized.
    """
    from catgpt.core.configs.schema import SchedulerConfig

    if not isinstance(config, SchedulerConfig):
        # Handle dict-like config
        config = SchedulerConfig(**config) if isinstance(config, dict) else config

    name = config.name.lower()

    if name == "constant":
        return None

    # Calculate warmup and decay steps
    warmup_steps = config.warmup_steps
    decay_steps = total_steps - warmup_steps

    if name == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                # Linear warmup
                return step / max(warmup_steps, 1)
            # Cosine decay
            progress = (step - warmup_steps) / max(decay_steps, 1)
            return config.min_lr_ratio + (1 - config.min_lr_ratio) * (
                1 + __import__("math").cos(__import__("math").pi * progress)
            ) / 2

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if name == "linear":
        # Linear decay with warmup
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(decay_steps, 1)
            return config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    msg = f"Unknown scheduler: {name}. Choose from: constant, cosine, linear"
    raise ValueError(msg)
