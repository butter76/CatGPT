"""JAX training loop with W&B logging and multi-device support."""

import json
import shutil
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

# dtype mapping for mixed precision
DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

if TYPE_CHECKING:
    from catgpt.jax.configs import (
        JaxCheckpointConfig,
        JaxExperimentConfig,
        JaxModelConfig,
        JaxTokenizerConfig,
        JaxTrainingConfig,
        JaxWandbConfig,
    )


class TrainState(train_state.TrainState):
    """Extended train state with additional tracking."""

    # Use default None for optional fields
    rng: jax.Array | None = None


class Trainer:
    """JAX trainer with W&B logging and multi-device support.

    Training is step-based with "pseudo-epochs" - fixed step intervals
    at which validation and checkpointing occur.

    Features:
    - Multi-device training with pmap or single-device with jit
    - Weights & Biases integration for experiment tracking
    - Checkpoint saving with model/tokenizer configs (Orbax or simple)
    - Gradient clipping and accumulation
    - JIT compilation for efficiency
    """

    def __init__(
        self,
        model: nn.Module,
        params: dict,
        optimizer: optax.GradientTransformation,
        train_dataloader: Any,
        val_dataloader: Any | None = None,
        *,
        training_config: "JaxTrainingConfig",
        checkpoint_config: "JaxCheckpointConfig",
        wandb_config: "JaxWandbConfig",
        model_config: "JaxModelConfig | None" = None,
        tokenizer_config: "JaxTokenizerConfig | None" = None,
        full_config: "JaxExperimentConfig | None" = None,
        rng: jax.Array | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The Flax model.
            params: Model parameters.
            optimizer: Optax optimizer.
            train_dataloader: Training data loader.
            val_dataloader: Optional validation data loader.
            training_config: Training hyperparameters.
            checkpoint_config: Checkpointing configuration.
            wandb_config: W&B logging configuration.
            model_config: Model config to save with checkpoints.
            tokenizer_config: Tokenizer config to save with checkpoints.
            full_config: Full experiment config for W&B logging.
            rng: PRNG key for randomness. If None, uses jax.random.key(0).
        """
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.wandb_config = wandb_config
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.full_config = full_config

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Device setup
        self.num_devices = jax.device_count()
        self.local_devices = jax.local_devices()
        logger.info(f"JAX devices: {self.num_devices} ({self.local_devices})")

        # Mixed precision setup
        self.mixed_precision = training_config.mixed_precision
        self.compute_dtype = DTYPE_MAP.get(
            training_config.precision_dtype, jnp.bfloat16
        ) if self.mixed_precision else jnp.float32

        # Set matmul precision for tensor cores
        # "high" enables TF32 on Ampere+, "highest" uses full precision
        matmul_precision = training_config.matmul_precision
        if matmul_precision in ("default", "high", "highest"):
            jax.config.update("jax_default_matmul_precision", matmul_precision)
            logger.info(f"Set JAX matmul precision to '{matmul_precision}'")

        if self.mixed_precision:
            logger.info(f"Mixed precision enabled: compute_dtype={training_config.precision_dtype}")

        # Initialize RNG
        if rng is None:
            rng = jax.random.key(0)
        self.rng = rng

        # Create train state
        self.state = TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
            rng=rng,
        )

        # Training state tracking
        self.global_step = 0
        self.best_val_loss = float("inf")

        # JIT compile training and evaluation steps
        if training_config.jit_compile:
            self._train_step = jax.jit(self._train_step_impl)
            self._eval_step = jax.jit(self._eval_step_impl)
        else:
            self._train_step = self._train_step_impl
            self._eval_step = self._eval_step_impl

        # Initialize W&B
        self._wandb_run = None
        if wandb_config.enabled:
            self._init_wandb()

        # Create checkpoint directory
        checkpoint_config.dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized with {self.num_devices} device(s), "
            f"JIT={training_config.jit_compile}"
        )

    @property
    def current_epoch(self) -> int:
        """Current pseudo-epoch (based on steps_per_epoch)."""
        return self.global_step // self.training_config.steps_per_epoch

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb

            config_dict = {}
            if self.full_config is not None:
                from catgpt.jax.configs import jax_config_to_dict

                config_dict = jax_config_to_dict(self.full_config)

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

    def _infinite_dataloader(self) -> Iterator[dict[str, np.ndarray]]:
        """Create an infinite iterator over the training dataloader."""
        while True:
            yield from self.train_dataloader

    def _train_step_impl(
        self,
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """Perform a single training step (implementation).

        Args:
            state: Current train state.
            batch: Batch with 'input' (tokens) and 'target' (win probability).

        Returns:
            Tuple of (new state, metrics dict).
        """
        compute_dtype = self.compute_dtype

        def loss_fn(params):
            logits = state.apply_fn(
                params,
                batch["input"],
                train=True,
                return_logits=True,
                compute_dtype=compute_dtype,
            )

            targets = batch["target"]
            if targets.ndim == 1:
                targets = targets[:, None]

            # Cast logits to float32 for loss computation (numerical stability)
            logits_f32 = logits.astype(jnp.float32)
            targets_f32 = targets.astype(jnp.float32)

            # Binary cross-entropy with logits
            loss = optax.sigmoid_binary_cross_entropy(logits_f32, targets_f32).mean()
            return loss, logits_f32

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute accuracy
        preds = (jax.nn.sigmoid(logits) > 0.5).astype(jnp.float32)
        targets = batch["target"]
        if targets.ndim == 1:
            targets = targets[:, None]
        accuracy = (preds == targets.astype(jnp.float32)).mean()

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }

        return state, metrics

    def _eval_step_impl(
        self,
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Perform a single evaluation step (implementation).

        Args:
            state: Current train state.
            batch: Batch with 'input' (tokens) and 'target' (win probability).

        Returns:
            Metrics dict.
        """
        logits = state.apply_fn(
            state.params,
            batch["input"],
            train=False,
            return_logits=True,
            compute_dtype=self.compute_dtype,
        )

        targets = batch["target"]
        if targets.ndim == 1:
            targets = targets[:, None]

        # Cast to float32 for loss computation (numerical stability)
        logits_f32 = logits.astype(jnp.float32)
        targets_f32 = targets.astype(jnp.float32)

        # Binary cross-entropy with logits
        loss = optax.sigmoid_binary_cross_entropy(logits_f32, targets_f32).mean()

        # Compute accuracy
        preds = (jax.nn.sigmoid(logits_f32) > 0.5).astype(jnp.float32)
        accuracy = (preds == targets_f32).mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
        }

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
        accumulation_steps = self.training_config.gradient_accumulation_steps

        logger.info(
            f"Starting training for {max_steps} steps "
            f"({total_epochs} pseudo-epochs of {steps_per_epoch} steps each)"
        )

        data_iter = self._infinite_dataloader()

        epoch_loss = 0.0
        epoch_batches = 0
        last_epoch = self.current_epoch

        # Progress bar for current pseudo-epoch
        steps_in_current_epoch = self.global_step % steps_per_epoch
        epoch_pbar = tqdm(
            total=steps_per_epoch,
            initial=steps_in_current_epoch,
            desc=f"Epoch {self.current_epoch}/{total_epochs}",
            unit="step",
        )

        accumulated_grads = None
        accumulation_count = 0

        while self.global_step < max_steps:
            # Get next batch
            batch = next(data_iter)

            # Convert to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}

            # Training step
            self.state, metrics = self._train_step(self.state, batch)

            loss = float(metrics["loss"])
            epoch_loss += loss
            epoch_batches += 1
            accumulation_count += 1

            # Step after accumulation
            if accumulation_count >= accumulation_steps:
                self.global_step += 1
                accumulation_count = 0

                # Update progress bar
                epoch_pbar.update(1)
                epoch_pbar.set_postfix(loss=f"{loss:.4f}")

                # Log step metrics
                if self.global_step % self.wandb_config.log_every_steps == 0:
                    self._log_step(loss)

                # Check for pseudo-epoch boundary
                if self.current_epoch > last_epoch:
                    epoch_pbar.close()

                    self._on_epoch_end(epoch_loss / max(epoch_batches, 1))
                    epoch_loss = 0.0
                    epoch_batches = 0
                    last_epoch = self.current_epoch

                    # Start new epoch progress bar
                    if self.global_step < max_steps:
                        epoch_pbar = tqdm(
                            total=steps_per_epoch,
                            desc=f"Epoch {self.current_epoch}/{total_epochs}",
                            unit="step",
                        )

        # Close progress bar
        epoch_pbar.close()

        # Final epoch end (if we didn't just finish one)
        if epoch_batches > 0:
            self._on_epoch_end(epoch_loss / max(epoch_batches, 1))

        # Final checkpoint
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
        # Validation
        val_metrics = {}
        if self.val_dataloader is not None:
            val_metrics = self._eval()
            val_loss = val_metrics.get("val_loss", float("inf"))

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.checkpoint_config.dir / "best")

        # Log epoch summary
        self._log_epoch(train_loss, val_metrics)

        # Save checkpoint
        if self._should_save_checkpoint():
            self.save_checkpoint(
                self.checkpoint_config.dir / f"epoch_{self.current_epoch}"
            )
            self._cleanup_old_checkpoints()

    def _eval(self) -> dict[str, float]:
        """Run evaluation on validation set.

        Returns:
            Dictionary of validation metrics.
        """
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        max_eval_steps = self.training_config.max_eval_steps

        val_iterator = tqdm(
            self.val_dataloader,
            desc="Validation",
            unit="batch",
            leave=False,
            total=max_eval_steps,
        )

        for step_idx, batch in enumerate(val_iterator):
            if max_eval_steps is not None and step_idx >= max_eval_steps:
                break

            # Convert to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}

            metrics = self._eval_step(self.state, batch)

            batch_size = batch["input"].shape[0]
            total_loss += float(metrics["loss"]) * batch_size
            total_correct += float(metrics["accuracy"]) * batch_size
            total_samples += batch_size

        val_loss = total_loss / max(total_samples, 1)
        val_accuracy = total_correct / max(total_samples, 1)

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

            # Get current learning rate from optimizer state
            # This is a simplified version - full implementation would
            # extract from the optax state
            lr = self._get_current_lr()

            wandb.log(
                {
                    "train/loss": loss,
                    "train/learning_rate": lr,
                    "train/global_step": self.global_step,
                    "train/epoch": self.current_epoch,
                },
                step=self.global_step,
            )

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer state."""
        # Try to extract from hyperparams if available
        opt_state = self.state.opt_state
        if hasattr(opt_state, "hyperparams") and "learning_rate" in opt_state.hyperparams:
            return float(opt_state.hyperparams["learning_rate"])
        # Fallback to config
        if self.full_config is not None:
            return self.full_config.optimizer.learning_rate
        return 0.0

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
        - params.msgpack or orbax checkpoint: Model parameters
        - opt_state.msgpack: Optimizer state (if configured)
        - trainer_state.json: Training state (step, epoch, etc.)
        - model_config.yaml: Model configuration
        - tokenizer_config.yaml: Tokenizer configuration
        - config.yaml: Full experiment configuration

        Args:
            path: Directory to save checkpoint to.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.checkpoint_config.use_orbax:
            self._save_checkpoint_orbax(path)
        else:
            self._save_checkpoint_simple(path)

        # Save trainer state
        trainer_state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }
        with (path / "trainer_state.json").open("w") as f:
            json.dump(trainer_state, f, indent=2)

        # Save configs
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
            from catgpt.jax.configs import jax_config_to_dict

            OmegaConf.save(
                OmegaConf.create(jax_config_to_dict(self.full_config)),
                path / "config.yaml",
            )

        logger.info(f"Saved checkpoint to {path}")

    def _save_checkpoint_orbax(self, path: Path) -> None:
        """Save checkpoint using Orbax."""
        try:
            import orbax.checkpoint as ocp

            # Orbax requires absolute paths
            path = path.resolve()

            checkpointer = ocp.PyTreeCheckpointer()
            checkpointer.save(
                path / "params",
                self.state.params,
                force=True,  # Allow overwriting existing checkpoints
            )

            if self.checkpoint_config.save_optimizer:
                checkpointer.save(
                    path / "opt_state",
                    self.state.opt_state,
                    force=True,  # Allow overwriting existing checkpoints
                )
        except ImportError:
            logger.warning("Orbax not installed, falling back to simple checkpointing")
            self._save_checkpoint_simple(path)

    def _save_checkpoint_simple(self, path: Path) -> None:
        """Save checkpoint using simple serialization."""
        from flax.serialization import to_bytes

        with (path / "params.msgpack").open("wb") as f:
            f.write(to_bytes(self.state.params))

        if self.checkpoint_config.save_optimizer:
            with (path / "opt_state.msgpack").open("wb") as f:
                f.write(to_bytes(self.state.opt_state))

    def load_checkpoint(self, path: Path | str) -> None:
        """Load a training checkpoint to resume training.

        Args:
            path: Directory containing the checkpoint.
        """
        path = Path(path)

        if self.checkpoint_config.use_orbax:
            self._load_checkpoint_orbax(path)
        else:
            self._load_checkpoint_simple(path)

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

    def _load_checkpoint_orbax(self, path: Path) -> None:
        """Load checkpoint using Orbax."""
        try:
            import orbax.checkpoint as ocp

            # Orbax requires absolute paths
            path = path.resolve()

            checkpointer = ocp.PyTreeCheckpointer()

            params_path = path / "params"
            if params_path.exists():
                params = checkpointer.restore(params_path)
                self.state = self.state.replace(params=params)

            opt_state_path = path / "opt_state"
            if opt_state_path.exists():
                opt_state = checkpointer.restore(opt_state_path)
                self.state = self.state.replace(opt_state=opt_state)
        except ImportError:
            logger.warning("Orbax not installed, falling back to simple loading")
            self._load_checkpoint_simple(path)

    def _load_checkpoint_simple(self, path: Path) -> None:
        """Load checkpoint using simple serialization."""
        from flax.serialization import from_bytes

        params_path = path / "params.msgpack"
        if params_path.exists():
            with params_path.open("rb") as f:
                params = from_bytes(self.state.params, f.read())
            self.state = self.state.replace(params=params)

        opt_state_path = path / "opt_state.msgpack"
        if opt_state_path.exists():
            with opt_state_path.open("rb") as f:
                opt_state = from_bytes(self.state.opt_state, f.read())
            self.state = self.state.replace(opt_state=opt_state)

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
