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
from catgpt.jax.data.dataloader import (
    PlaceholderDataLoader,
    PlaceholderDataset,
    create_dataloader,
)
from catgpt.jax.models.transformer import BidirectionalTransformer, TransformerConfig
from catgpt.jax.optimizers.factory import create_optimizer_with_gradient_clipping
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


@hydra.main(version_base=None, config_path="../configs", config_name="jax_base")
def main(cfg: DictConfig) -> None:
    """Main JAX training entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Setup logging
    setup_logging_jax(level=cfg.get("logging", {}).get("level", "INFO"))
    logger.info("Starting CatGPT JAX training")
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

    # Create model config
    model_cfg = experiment_config.model
    tokenizer_cfg = experiment_config.tokenizer

    # Update model config with tokenizer info
    transformer_config = TransformerConfig(
        hidden_size=model_cfg.hidden_size,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        ff_dim=model_cfg.ff_dim,
        vocab_size=model_cfg.vocab_size,
        seq_length=tokenizer_cfg.sequence_length,
        activation=model_cfg.activation,
        dropout_rate=model_cfg.dropout_rate,
    )

    # Create model
    model = BidirectionalTransformer(config=transformer_config)

    # Initialize model parameters
    rng, init_rng = jax.random.split(rng)
    dummy_input = jnp.zeros((1, tokenizer_cfg.sequence_length), dtype=jnp.int32)
    params = model.init(init_rng, dummy_input, train=False)

    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    logger.info(f"Model parameters: {param_count:,}")

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

    try:
        train_dataloader = create_dataloader(
            experiment_config.data,
            split="train",
            batch_size=experiment_config.training.batch_size,
            num_devices=num_devices,
            device_index=0,
            tokenizer_config=tokenizer_cfg,
        )
    except Exception as e:
        logger.warning(f"Failed to create grain dataloader: {e}")
        logger.info("Falling back to placeholder dataset...")
        train_dataset = PlaceholderDataset(
            num_samples=10000,
            seq_length=tokenizer_cfg.sequence_length,
            vocab_size=model_cfg.vocab_size,
            seed=experiment_config.data.seed,
        )
        train_dataloader = PlaceholderDataLoader(
            train_dataset,
            batch_size=experiment_config.training.batch_size,
            shuffle=True,
        )

    val_dataloader = None
    if experiment_config.data.val_path:
        try:
            val_dataloader = create_dataloader(
                experiment_config.data,
                split="val",
                batch_size=experiment_config.training.batch_size,
                num_devices=num_devices,
                device_index=0,
                tokenizer_config=tokenizer_cfg,
            )
        except Exception as e:
            logger.warning(f"Failed to create validation dataloader: {e}")
            val_dataset = PlaceholderDataset(
                num_samples=1000,
                seq_length=tokenizer_cfg.sequence_length,
                vocab_size=model_cfg.vocab_size,
                seed=experiment_config.data.seed + 1,
            )
            val_dataloader = PlaceholderDataLoader(
                val_dataset,
                batch_size=experiment_config.training.batch_size,
                shuffle=False,
                drop_last=False,
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
        logger.info(f"Sample target (first 3): {sample_batch['target'][:3].tolist()}")
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
