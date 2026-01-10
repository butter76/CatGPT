"""PyTorch training loop with W&B logging and DDP support."""

import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from catgpt.core.utils.distributed import (
    all_reduce_mean,
    barrier,
    get_device,
    get_rank,
    get_world_size,
    is_distributed,
    is_main_process,
)

if TYPE_CHECKING:
    from catgpt.core.configs.schema import (
        CheckpointConfig,
        ExperimentConfig,
        ModelConfig,
        TokenizerConfig,
        TrainingConfig,
        WandbConfig,
    )


class Trainer:
    """PyTorch trainer with W&B logging and multi-GPU (DDP) support.

    Training is step-based with "pseudo-epochs" - fixed step intervals
    at which validation and checkpointing occur.

    Features:
    - Distributed Data Parallel (DDP) for multi-GPU training
    - Weights & Biases integration for experiment tracking
    - Checkpoint saving with model/tokenizer configs
    - Gradient clipping and accumulation
    - torch.compile support for efficiency
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        scheduler: LRScheduler | None = None,
        *,
        training_config: "TrainingConfig",
        checkpoint_config: "CheckpointConfig",
        wandb_config: "WandbConfig",
        model_config: "ModelConfig | None" = None,
        tokenizer_config: "TokenizerConfig | None" = None,
        full_config: "ExperimentConfig | None" = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train (will be wrapped with DDP if distributed).
            optimizer: The optimizer.
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            scheduler: Optional learning rate scheduler.
            training_config: Training hyperparameters.
            checkpoint_config: Checkpointing configuration.
            wandb_config: W&B logging configuration.
            model_config: Model config to save with checkpoints.
            tokenizer_config: Tokenizer config to save with checkpoints.
            full_config: Full experiment config for W&B logging.
        """
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.wandb_config = wandb_config
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.full_config = full_config

        # Device setup
        self.device = get_device()
        self.rank = get_rank()
        self.world_size = get_world_size()

        # Move model to device
        self.model = model.to(self.device)

        # Wrap with DDP if distributed
        if is_distributed():
            self.model = DDP(self.model, device_ids=[self.rank])
            logger.info(f"Wrapped model with DDP (rank {self.rank}/{self.world_size})")

        # Compile model if requested
        if training_config.compile_model:
            logger.info("Compiling model with torch.compile...")
            self._compiled_model = torch.compile(self._get_raw_model())
        else:
            self._compiled_model = None

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Loss function for binary classification (win probability)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize W&B
        self._wandb_run = None
        if wandb_config.enabled and is_main_process():
            self._init_wandb()

        # Create checkpoint directory
        if is_main_process():
            checkpoint_config.dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized on {self.device} "
            f"(rank {self.rank}, world_size {self.world_size})"
        )

    @property
    def current_epoch(self) -> int:
        """Current pseudo-epoch (based on steps_per_epoch)."""
        return self.global_step // self.training_config.steps_per_epoch

    def _get_raw_model(self) -> nn.Module:
        """Get the underlying model (unwrap DDP if needed)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            config_dict = {}
            if self.full_config is not None:
                from catgpt.core.configs.schema import config_to_dict

                config_dict = config_to_dict(self.full_config)

            self._wandb_run = wandb.init(
                project=self.wandb_config.project,
                entity=self.wandb_config.entity,
                name=self.wandb_config.run_name,
                tags=self.wandb_config.tags,
                config=config_dict,
            )
            logger.info(f"W&B initialized: {wandb.run.url}")  # type: ignore[union-attr]
        except ImportError:
            logger.warning("wandb not installed, skipping W&B initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def _infinite_dataloader(self) -> Iterator[dict[str, Tensor]]:
        """Create an infinite iterator over the training dataloader."""
        while True:
            yield from self.train_dataloader

    def fit(self) -> dict[str, Any]:
        """Run the full training loop.

        Training is step-based. Validation runs at the end of each
        pseudo-epoch (every `steps_per_epoch` steps).

        Returns:
            Dictionary with final metrics.
        """
        max_steps = self.training_config.max_steps
        steps_per_epoch = self.training_config.steps_per_epoch
        total_epochs = max_steps // steps_per_epoch

        logger.info(
            f"Starting training for {max_steps} steps "
            f"({total_epochs} pseudo-epochs of {steps_per_epoch} steps each)"
        )

        model = self._compiled_model if self._compiled_model is not None else self.model
        model.train()

        accumulation_steps = self.training_config.gradient_accumulation_steps
        data_iter = self._infinite_dataloader()

        # Progress bar for entire training
        pbar = None
        if is_main_process():
            pbar = tqdm(
                total=max_steps,
                initial=self.global_step,
                desc="Training",
                unit="step",
            )

        epoch_loss = 0.0
        epoch_batches = 0
        last_epoch = self.current_epoch

        while self.global_step < max_steps:
            # Get next batch
            batch = next(data_iter)
            loss = self._train_step(batch, model)

            # Gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            epoch_loss += loss.item()
            epoch_batches += 1

            # Optimizer step after accumulation
            if (epoch_batches) % accumulation_steps == 0:
                # Gradient clipping
                if self.training_config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.training_config.gradient_clip,
                    )

                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        epoch=self.current_epoch,
                    )

                # Log step metrics
                if (
                    is_main_process()
                    and self.global_step % self.wandb_config.log_every_steps == 0
                ):
                    self._log_step(loss.item())

                # Check for pseudo-epoch boundary
                if self.current_epoch > last_epoch:
                    self._on_epoch_end(epoch_loss / max(epoch_batches, 1))
                    epoch_loss = 0.0
                    epoch_batches = 0
                    last_epoch = self.current_epoch

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Final epoch end (if we didn't just finish one)
        if epoch_batches > 0:
            self._on_epoch_end(epoch_loss / max(epoch_batches, 1))

        # Final checkpoint
        if is_main_process():
            self.save_checkpoint(self.checkpoint_config.dir / "final")

        # Clean up W&B
        if self._wandb_run is not None:
            import wandb

            wandb.finish()

        return {
            "final_step": self.global_step,
            "final_epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }

    def _on_epoch_end(self, train_loss: float) -> None:
        """Handle end of pseudo-epoch: validation, logging, checkpointing."""
        # Reduce train loss across processes
        if is_distributed():
            train_loss_tensor = torch.tensor(train_loss, device=self.device)
            train_loss = all_reduce_mean(train_loss_tensor).item()

        # Validation
        val_metrics = {}
        if self.val_dataloader is not None:
            val_metrics = self._eval()
            val_loss = val_metrics.get("val_loss", float("inf"))

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if is_main_process():
                    self.save_checkpoint(self.checkpoint_config.dir / "best")

        # Log epoch summary
        if is_main_process():
            self._log_epoch(train_loss, val_metrics)

        # Save checkpoint
        if self._should_save_checkpoint() and is_main_process():
            self.save_checkpoint(
                self.checkpoint_config.dir / f"epoch_{self.current_epoch}"
            )
            self._cleanup_old_checkpoints()

        # Synchronize before continuing
        barrier()

    def _train_step(self, batch: dict[str, Tensor], model: nn.Module) -> Tensor:
        """Perform a single training step.

        Args:
            batch: Batch with 'input' (tokens) and 'target' (win probability).
            model: The model (potentially compiled).

        Returns:
            Loss tensor.
        """
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device).float()

        # Ensure targets have correct shape
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # Forward pass (use logits for numerical stability with BCE)
        raw_model = self._get_raw_model()
        if hasattr(raw_model, "forward_logits"):
            logits = raw_model.forward_logits(inputs)
        else:
            logits = model(inputs)

        loss = self.loss_fn(logits, targets)
        return loss

    @torch.inference_mode()
    def _eval(self) -> dict[str, float]:
        """Run evaluation on validation set.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_dataloader:
            inputs = batch["input"].to(self.device)
            targets = batch["target"].to(self.device).float()

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

            raw_model = self._get_raw_model()
            if hasattr(raw_model, "forward_logits"):
                logits = raw_model.forward_logits(inputs)
            else:
                logits = self.model(inputs)

            loss = self.loss_fn(logits, targets)
            total_loss += loss.item() * inputs.size(0)

            # Accuracy (threshold at 0.5)
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

        # Reduce across processes
        if is_distributed():
            metrics_tensor = torch.tensor(
                [total_loss, total_correct, total_samples],
                device=self.device,
            )
            torch.distributed.all_reduce(metrics_tensor)
            total_loss, total_correct, total_samples = metrics_tensor.tolist()

        val_loss = total_loss / max(total_samples, 1)
        val_accuracy = total_correct / max(total_samples, 1)

        self.model.train()

        return {
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

    def _should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint this pseudo-epoch."""
        return (self.current_epoch + 1) % self.checkpoint_config.save_every_epochs == 0

    def _log_step(self, loss: float) -> None:
        """Log metrics for a training step."""
        if self._wandb_run is not None:
            import wandb

            lr = self.optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "train/loss": loss,
                    "train/learning_rate": lr,
                    "train/global_step": self.global_step,
                    "train/epoch": self.current_epoch,
                },
                step=self.global_step,
            )

    def _log_epoch(self, train_loss: float, val_metrics: dict[str, float]) -> None:
        """Log metrics for a pseudo-epoch."""
        msg = f"Epoch {self.current_epoch} (step {self.global_step}): train_loss={train_loss:.4f}"
        if val_metrics:
            msg += f", val_loss={val_metrics.get('val_loss', 0):.4f}"
            msg += f", val_acc={val_metrics.get('val_accuracy', 0):.4f}"
        logger.info(msg)

        if self._wandb_run is not None:
            import wandb

            metrics = {
                "epoch": self.current_epoch,
                "train/epoch_loss": train_loss,
            }
            for k, v in val_metrics.items():
                metrics[f"val/{k.replace('val_', '')}"] = v

            wandb.log(metrics, step=self.global_step)

    def save_checkpoint(self, path: Path | str) -> None:
        """Save a training checkpoint.

        Saves:
        - model.pt: Model state dict
        - optimizer.pt: Optimizer state dict
        - trainer_state.json: Training state (step, epoch, etc.)
        - model_config.yaml: Model configuration
        - tokenizer_config.yaml: Tokenizer configuration
        - config.yaml: Full experiment configuration

        Args:
            path: Directory to save checkpoint to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        raw_model = self._get_raw_model()
        torch.save(raw_model.state_dict(), path / "model.pt")

        # Save optimizer
        if self.checkpoint_config.save_optimizer:
            torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
            if self.scheduler is not None:
                torch.save(self.scheduler.state_dict(), path / "scheduler.pt")

        # Save trainer state
        trainer_state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }
        with (path / "trainer_state.json").open("w") as f:
            json.dump(trainer_state, f, indent=2)

        # Save configs (essential for model reconstruction)
        if self.model_config is not None:
            model_dict = (
                asdict(self.model_config)
                if hasattr(self.model_config, "__dataclass_fields__")
                else dict(self.model_config)
            )
            OmegaConf.save(OmegaConf.create(model_dict), path / "model_config.yaml")

        if self.tokenizer_config is not None:
            tok_dict = (
                asdict(self.tokenizer_config)
                if hasattr(self.tokenizer_config, "__dataclass_fields__")
                else dict(self.tokenizer_config)
            )
            OmegaConf.save(OmegaConf.create(tok_dict), path / "tokenizer_config.yaml")

        if self.full_config is not None:
            from catgpt.core.configs.schema import config_to_dict

            OmegaConf.save(
                OmegaConf.create(config_to_dict(self.full_config)),
                path / "config.yaml",
            )

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path | str) -> None:
        """Load a training checkpoint to resume training.

        Args:
            path: Directory containing the checkpoint.
        """
        path = Path(path)

        # Load model
        model_path = path / "model.pt"
        if model_path.exists():
            state_dict = torch.load(
                model_path, map_location=self.device, weights_only=True
            )
            self._get_raw_model().load_state_dict(state_dict)

        # Load optimizer
        optimizer_path = path / "optimizer.pt"
        if optimizer_path.exists():
            state_dict = torch.load(
                optimizer_path, map_location=self.device, weights_only=True
            )
            self.optimizer.load_state_dict(state_dict)

        # Load scheduler
        scheduler_path = path / "scheduler.pt"
        if scheduler_path.exists() and self.scheduler is not None:
            state_dict = torch.load(
                scheduler_path, map_location=self.device, weights_only=True
            )
            self.scheduler.load_state_dict(state_dict)

        # Load trainer state
        state_path = path / "trainer_state.json"
        if state_path.exists():
            with state_path.open() as f:
                state = json.load(f)
            self.global_step = state.get("global_step", 0)
            self.best_val_loss = state.get("best_val_loss", float("inf"))

        logger.info(
            f"Loaded checkpoint from {path} (step {self.global_step}, epoch {self.current_epoch})"
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_dir = self.checkpoint_config.dir
        keep_last = self.checkpoint_config.keep_last

        # Find all epoch checkpoints
        epoch_dirs = sorted(
            checkpoint_dir.glob("epoch_*"),
            key=lambda p: int(p.name.split("_")[1]),
        )

        # Remove old ones
        for old_dir in epoch_dirs[:-keep_last]:
            shutil.rmtree(old_dir)
            logger.debug(f"Removed old checkpoint: {old_dir}")
