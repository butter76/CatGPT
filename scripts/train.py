

#!/usr/bin/env python3
"""Training script for CatGPT models.

This script handles the full training pipeline with:
- Hydra configuration management
- W&B experiment tracking
- Multi-GPU support via DDP
- Checkpoint saving with model/tokenizer configs

Usage:
    # Single GPU training
    uv run python scripts/train.py

    # Override config values
    uv run python scripts/train.py model.hidden_size=512 training.batch_size=128

    # Multi-GPU training (8 GPUs)
    torchrun --nproc_per_node=8 scripts/train.py

    # Resume from checkpoint
    uv run python scripts/train.py +resume_from=checkpoints/epoch_10
"""

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from catgpt.core.configs.schema import (
    CheckpointConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    WandbConfig,
    config_from_dict,
)
from catgpt.core.utils.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
)
from catgpt.core.utils.logging import setup_logging
from catgpt.torch.data.dataloader import create_dataloader
from catgpt.torch.models.transformer import BidirectionalTransformer, TransformerConfig
from catgpt.torch.optimizers.factory import create_optimizer, create_scheduler
from catgpt.torch.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration.
    """
    # Setup logging (only on main process for cleaner output)
    if is_main_process():
        setup_logging(level=cfg.get("logging", {}).get("level", "INFO"))
        logger.info("Starting CatGPT training")
        logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Setup distributed training
    distributed_enabled = cfg.get("distributed", {}).get("enabled", False)
    if distributed_enabled or get_world_size() > 1:
        backend = cfg.get("distributed", {}).get("backend", "nccl")
        setup_distributed(backend=backend)

    rank = get_rank()
    world_size = get_world_size()

    # Set seed (different per rank for data diversity, same for model init)
    seed = cfg.get("seed", 42)
    set_seed(seed + rank)

    # Convert OmegaConf to typed config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    experiment_config = config_from_dict(config_dict)  # type: ignore[arg-type]

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
    )

    # Create model
    model = BidirectionalTransformer(transformer_config)
    if is_main_process():
        logger.info(f"Model parameters: {model.num_parameters():,}")

    # Create optimizer
    optimizer = create_optimizer(model.parameters(), experiment_config.optimizer)

    # Create data loaders
    if is_main_process():
        logger.info("Creating data loaders...")

    train_dataloader = create_dataloader(
        experiment_config.data,
        split="train",
        batch_size=experiment_config.training.batch_size,
        world_size=world_size,
        rank=rank,
        tokenizer_config=tokenizer_cfg,
    )

    val_dataloader = None
    if experiment_config.data.val_path:
        val_dataloader = create_dataloader(
            experiment_config.data,
            split="val",
            batch_size=experiment_config.training.batch_size,
            world_size=world_size,
            rank=rank,
            tokenizer_config=tokenizer_cfg,
        )

    # Debug: Check first batch
    if is_main_process():
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

    # Create scheduler (use max_steps from config)
    scheduler = create_scheduler(
        optimizer,
        experiment_config.scheduler,
        total_steps=experiment_config.training.max_steps,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        scheduler=scheduler,
        training_config=experiment_config.training,
        checkpoint_config=experiment_config.checkpoint,
        wandb_config=experiment_config.wandb,
        model_config=model_cfg,
        tokenizer_config=tokenizer_cfg,
        full_config=experiment_config,
    )

    # Resume from checkpoint if specified
    resume_from = cfg.get("resume_from", None)
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Run training
    results = trainer.fit()

    # Log final results
    if is_main_process():
        logger.info(f"Training complete! Results: {results}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()
