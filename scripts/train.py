#!/usr/bin/env python3
"""Training script for CatGPT models.

This script demonstrates how to set up training using Hydra for configuration.

Usage:
    python scripts/train.py
    python scripts/train.py model.hidden_size=512
    python scripts/train.py --config-name=large
"""

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from catgpt.core.utils import setup_logging


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Main training entry point.

    Args:
        cfg: Hydra configuration.
    """
    setup_logging(level=cfg.logging.level)

    logger.info("Starting training with configuration:")
    logger.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # TODO: Implement actual training pipeline
    # 1. Set up data loaders
    # 2. Initialize model (from catgpt.torch or catgpt.jax)
    # 3. Create optimizer and scheduler
    # 4. Run training loop

    logger.info("Training script placeholder - implement your training logic here!")


if __name__ == "__main__":
    main()
