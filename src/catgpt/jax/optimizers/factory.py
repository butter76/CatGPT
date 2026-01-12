"""Factory functions for creating JAX optimizers from config."""

from typing import TYPE_CHECKING

import optax

from catgpt.jax.optimizers.splus import splus

if TYPE_CHECKING:
    from catgpt.jax.configs import JaxOptimizerConfig, JaxSchedulerConfig


def create_optimizer(
    config: "JaxOptimizerConfig",
    scheduler_config: "JaxSchedulerConfig | None" = None,
    total_steps: int | None = None,
) -> optax.GradientTransformation:
    """Create an optax optimizer from configuration.

    Args:
        config: Optimizer configuration.
        scheduler_config: Optional scheduler configuration.
        total_steps: Total training steps (needed for some schedulers).

    Returns:
        Configured optax GradientTransformation.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    name = config.name.lower()

    # Get learning rate (possibly with schedule)
    learning_rate = create_lr_schedule(
        config.learning_rate, scheduler_config, total_steps
    )

    if name == "adamw":
        return optax.adamw(
            learning_rate=learning_rate,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    if name == "splus":
        return splus(
            learning_rate=learning_rate,
            b1=config.splus_b1,
            b2=config.splus_b2,
            ema_rate=config.splus_ema_rate,
            inverse_every=config.splus_inverse_every,
            max_dim=config.splus_max_dim,
            weight_decay=config.weight_decay,
        )

    if name == "sgd":
        return optax.chain(
            optax.add_decayed_weights(config.weight_decay),
            optax.sgd(learning_rate=learning_rate, momentum=config.b1),
        )

    msg = f"Unknown optimizer: {name}. Choose from: adamw, splus, sgd"
    raise ValueError(msg)


def create_lr_schedule(
    base_lr: float,
    scheduler_config: "JaxSchedulerConfig | None",
    total_steps: int | None,
) -> optax.Schedule | float:
    """Create a learning rate schedule.

    Args:
        base_lr: Base learning rate.
        scheduler_config: Scheduler configuration. If None, returns constant lr.
        total_steps: Total training steps.

    Returns:
        Learning rate schedule or constant value.
    """
    if scheduler_config is None:
        return base_lr

    name = scheduler_config.name.lower()

    if name == "constant":
        return base_lr

    warmup_steps = scheduler_config.warmup_steps
    min_lr = base_lr * scheduler_config.min_lr_ratio

    if total_steps is None:
        # Can't create decay schedule without total steps
        return optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
        )

    # Ensure warmup doesn't exceed total steps
    warmup_steps = min(warmup_steps, total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)

    if name == "cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=min_lr,
        )

    if name == "linear":
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(
                    init_value=0.0,
                    end_value=base_lr,
                    transition_steps=warmup_steps,
                ),
                optax.linear_schedule(
                    init_value=base_lr,
                    end_value=min_lr,
                    transition_steps=decay_steps,
                ),
            ],
            boundaries=[warmup_steps],
        )

    msg = f"Unknown scheduler: {name}. Choose from: constant, cosine, linear"
    raise ValueError(msg)


def create_optimizer_with_gradient_clipping(
    config: "JaxOptimizerConfig",
    scheduler_config: "JaxSchedulerConfig | None" = None,
    total_steps: int | None = None,
    gradient_clip: float | None = None,
) -> optax.GradientTransformation:
    """Create optimizer with optional gradient clipping.

    Args:
        config: Optimizer configuration.
        scheduler_config: Optional scheduler configuration.
        total_steps: Total training steps.
        gradient_clip: Maximum gradient norm. If None, no clipping.

    Returns:
        Configured optax GradientTransformation with clipping.
    """
    optimizer = create_optimizer(config, scheduler_config, total_steps)

    if gradient_clip is not None:
        return optax.chain(
            optax.clip_by_global_norm(gradient_clip),
            optimizer,
        )

    return optimizer
