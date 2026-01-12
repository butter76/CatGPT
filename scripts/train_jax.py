#!/usr/bin/env python3
"""JAX training script for CatGPT models.

This script handles the full JAX training pipeline with:
- Hydra configuration management
- W&B experiment tracking
- Multi-device support (via pmap or sharded arrays)
- Checkpoint saving with model/tokenizer configs (Orbax)

Usage:
    # Single device training
    uv run python scripts/train_jax.py

    # Override config values
    uv run python scripts/train_jax.py model.hidden_size=512 training.batch_size=128

    # Use a different config
    uv run python scripts/train_jax.py --config-name=jax_base

    # Resume from checkpoint
    uv run python scripts/train_jax.py +resume_from=checkpoints_jax/epoch_10

    # Quick test run
    uv run python scripts/train_jax.py training.max_steps=100 training.steps_per_epoch=50
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from catgpt.jax.configs import (
    JaxCheckpointConfig,
    JaxDataConfig,
    JaxDistributedConfig,
    JaxExperimentConfig,
    JaxModelConfig,
    JaxOptimizerConfig,
    JaxSchedulerConfig,
    JaxTokenizerConfig,
    JaxTrainingConfig,
    JaxWandbConfig,
    jax_config_from_dict,
)
from catgpt.jax.data.dataloader import create_dataloader
from catgpt.jax.models.transformer import BidirectionalTransformer, TransformerConfig
from catgpt.jax.optimizers.factory import create_lr_schedule, create_optimizer_with_gradient_clipping
from catgpt.jax.training.trainer import Trainer


def set_seed(seed: int) -> jax.Array:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed.

    Returns:
        JAX PRNG key.
    """
    np.random.seed(seed)
    return jax.random.key(seed)


def setup_logging_jax(level: str = "INFO") -> None:
    """Configure logging for JAX training."""
    import sys

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )


def prompt_run_name() -> str:
    """Prompt user for a run name to organize checkpoints.

    Returns:
        The run name entered by the user.
    """
    print("\n" + "=" * 50)
    print("CatGPT JAX Training")
    print("=" * 50)
    run_name = input("Enter run name: ").strip()
    if not run_name:
        raise ValueError("Run name cannot be empty")
    # Sanitize: replace spaces with underscores, remove problematic chars
    run_name = run_name.replace(" ", "_")
    run_name = "".join(c for c in run_name if c.isalnum() or c in "_-")
    print(f"Checkpoints will be saved to: checkpoints_jax/{run_name}/")
    print("=" * 50 + "\n")
    return run_name


@hydra.main(version_base=None, config_path="../configs", config_name="jax_base")
def main(cfg: DictConfig) -> None:
    """Main JAX training entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Prompt for run name first (before any logging setup)
    run_name = prompt_run_name()

    # Setup logging
    setup_logging_jax(level=cfg.get("logging", {}).get("level", "INFO"))
    logger.info("Starting CatGPT JAX training")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Log JAX device info
    devices = jax.devices()
    logger.info(f"JAX devices: {len(devices)} - {devices}")
    logger.info(f"JAX backend: {jax.default_backend()}")

    # Set seed and get RNG
    seed = cfg.get("seed", 42)
    rng = set_seed(seed)

    # Convert OmegaConf to typed config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_config = jax_config_from_dict(config_dict)  # type: ignore[arg-type]

    # Update checkpoint directory with run name
    base_checkpoint_dir = experiment_config.checkpoint.dir
    experiment_config.checkpoint.dir = Path(base_checkpoint_dir) / run_name
    logger.info(f"Checkpoint directory: {experiment_config.checkpoint.dir}")

    # Use run name for WandB if not already set
    if experiment_config.wandb.run_name is None:
        experiment_config.wandb.run_name = run_name

    # Create model config
    model_cfg = experiment_config.model
    tokenizer_cfg = experiment_config.tokenizer

    # Update model config with tokenizer info and output head settings
    transformer_config = TransformerConfig(
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ff_dim=model_cfg.ff_dim,
        vocab_size=model_cfg.vocab_size,
        seq_length=tokenizer_cfg.sequence_length,
        activation=model_cfg.activation,
        dropout_rate=model_cfg.dropout_rate,
        output_heads=model_cfg.output_heads,  # Include HL-Gauss config
    )

    # Create model
    model = BidirectionalTransformer(config=transformer_config)

    # Initialize model parameters
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, tokenizer_cfg.sequence_length), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input, train=False)

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info(f"Model parameters: {param_count:,}")

    # Create learning rate schedule (for WandB logging)
    lr_schedule = create_lr_schedule(
        experiment_config.optimizer.learning_rate,
        experiment_config.scheduler,
        total_steps=experiment_config.training.max_steps,
    )

    # Create optimizer with gradient clipping
    optimizer = create_optimizer_with_gradient_clipping(
        experiment_config.optimizer,
        experiment_config.scheduler,
        total_steps=experiment_config.training.max_steps,
        gradient_clip=experiment_config.training.gradient_clip,
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    num_devices = len(devices)

    train_dataloader = create_dataloader(
        experiment_config.data,
        split="train",
        batch_size=experiment_config.training.batch_size,
        num_devices=num_devices,
        device_index=0,
        tokenizer_config=tokenizer_cfg,
        output_heads_config=model_cfg.output_heads,  # For HL-Gauss transform
    )

    val_dataloader = None
    if experiment_config.data.val_path:
        val_dataloader = create_dataloader(
            experiment_config.data,
            split="val",
            batch_size=experiment_config.training.batch_size,
            num_devices=num_devices,
            device_index=0,
            tokenizer_config=tokenizer_cfg,
            output_heads_config=model_cfg.output_heads,  # For HL-Gauss transform
        )

    # Debug: Check first batch
    logger.info("Testing data loader with first batch...")
    try:
        sample_batch = next(iter(train_dataloader))
        logger.info(
            f"Sample batch - input shape: {sample_batch['input'].shape}, "
            f"target shape: {sample_batch['target'].shape}, "
            f"input dtype: {sample_batch['input'].dtype}, "
            f"target dtype: {sample_batch['target'].dtype}"
        )
        logger.info(f"Sample input (first 10 tokens): {sample_batch['input'][0, :10].tolist()}")
        # Target is now HL-Gauss distribution - show expected value instead
        num_bins = sample_batch['target'].shape[-1]
        bin_centers = (np.arange(num_bins) + 0.5) / num_bins
        expected_values = np.sum(sample_batch['target'][:3] * bin_centers, axis=-1)
        logger.info(f"Sample target expected values (first 3): {expected_values.tolist()}")
    except Exception as e:
        logger.error(f"Error loading first batch: {e}")
        raise

    # Create trainer
    trainer = Trainer(
        model=model,
        params=params,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=experiment_config.training,
        checkpoint_config=experiment_config.checkpoint,
        wandb_config=experiment_config.wandb,
        model_config=model_cfg,
        tokenizer_config=tokenizer_cfg,
        full_config=experiment_config,
        lr_schedule=lr_schedule,
        rng=rng,
    )

    # Resume from checkpoint if specified
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Run training
    results = trainer.fit()

    # Log final results
    logger.info(f"Training complete! Results: {results}")


if __name__ == "__main__":
    main()
