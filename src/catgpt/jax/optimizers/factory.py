"""Factory functions for creating JAX optimizers from config."""

from typing import TYPE_CHECKING

import jax.numpy as jnp
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
            eps=config.splus_eps,
            inverse_every=config.splus_inverse_every,
            max_dim=config.splus_max_dim,
            weight_decay=config.weight_decay,
            nonstandard_strings=config.splus_nonstandard_strings,
            nonstandard_constant=config.splus_nonstandard_constant,
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

    if name == "deepseek":
        return _deepseek_schedule(
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            stable_fraction=scheduler_config.stable_fraction,
            cooldown_fraction=scheduler_config.cooldown_fraction,
        )

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

    msg = f"Unknown scheduler: {name}. Choose from: deepseek, cosine, linear, constant"
    raise ValueError(msg)


def _deepseek_schedule(
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    stable_fraction: float,
    cooldown_fraction: float,
) -> optax.Schedule:
    """DeepSeek-V3 style multi-phase learning rate schedule.

    Phases:
    1. Linear warmup: 0 → base_lr over warmup_steps
    2. Stable: constant at base_lr for stable_fraction of post-warmup steps
    3. Cosine decay: base_lr → min_lr following cosine curve
    4. Linear cooldown: min_lr → 0 over cooldown_fraction of post-warmup steps
       (second-order optimizers like SPlus/Muon handle near-zero LRs gracefully)

    The cosine decay phase occupies the remaining fraction:
        1.0 - stable_fraction - cooldown_fraction

    See: https://arxiv.org/abs/2412.19437 Section 4.2

    Args:
        base_lr: Peak learning rate.
        min_lr: Minimum learning rate (for cosine decay end and cooldown).
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        stable_fraction: Fraction of post-warmup steps at constant peak LR.
        cooldown_fraction: Fraction of post-warmup steps at constant min LR.

    Returns:
        An optax Schedule (callable: step → lr).
    """
    post_warmup = max(total_steps - warmup_steps, 1)
    stable_steps = int(post_warmup * stable_fraction)
    cooldown_steps = int(post_warmup * cooldown_fraction)
    cosine_steps = max(post_warmup - stable_steps - cooldown_steps, 1)

    # Phase boundaries (in global step space)
    stable_end = warmup_steps + stable_steps
    cosine_end = stable_end + cosine_steps

    def schedule_fn(step):
        step = jnp.asarray(step, dtype=jnp.float32)

        # Phase 1: Linear warmup 0 → base_lr
        warmup_lr = base_lr * step / jnp.maximum(warmup_steps, 1)

        # Phase 2: Constant at base_lr
        stable_lr = jnp.float32(base_lr)

        # Phase 3: Cosine decay base_lr → min_lr
        cosine_progress = (step - stable_end) / jnp.maximum(cosine_steps, 1)
        cosine_progress = jnp.clip(cosine_progress, 0.0, 1.0)
        cosine_lr = min_lr + (base_lr - min_lr) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_progress))

        # Phase 4: Linear cooldown min_lr → 0
        # Second-order optimizers (SPlus, Muon) handle near-zero LRs gracefully
        cooldown_progress = (step - cosine_end) / jnp.maximum(cooldown_steps, 1)
        cooldown_progress = jnp.clip(cooldown_progress, 0.0, 1.0)
        cooldown_lr = min_lr * (1.0 - cooldown_progress)

        # Select phase based on step
        lr = jnp.where(
            step < warmup_steps,
            warmup_lr,
            jnp.where(
                step < stable_end,
                stable_lr,
                jnp.where(step < cosine_end, cosine_lr, cooldown_lr),
            ),
        )
        return lr

    return schedule_fn


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
