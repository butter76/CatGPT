"""Training loop implementation."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    """Configuration for the Trainer."""

    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float | None = 1.0
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every: int = 10
    eval_every: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Generic trainer for PyTorch models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            config: Trainer configuration.
        """
        self.config = config or TrainerConfig()
        self.model = model.to(self.config.device)
        self.optimizer = optimizer
        self.current_epoch = 0

        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trainer initialized on device: {self.config.device}")

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        for batch in progress:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1
            progress.set_postfix(loss=f"{loss:.4f}")

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: dict) -> float:
        """Perform a single training step.

        Args:
            batch: A batch of training data.

        Returns:
            Loss value for this step.
        """
        self.optimizer.zero_grad()

        # Move batch to device
        batch = {k: v.to(self.config.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Forward pass - subclasses should override this
        # This is a placeholder that assumes batch has 'input' and 'target' keys
        output = self.model(batch.get("input"))
        loss = nn.functional.mse_loss(output, batch.get("target"))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )

        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, path: Path | None = None) -> Path:
        """Save a checkpoint.

        Args:
            path: Optional path to save to. Defaults to checkpoint_dir.

        Returns:
            Path where checkpoint was saved.
        """
        if path is None:
            path = self.config.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load a checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
