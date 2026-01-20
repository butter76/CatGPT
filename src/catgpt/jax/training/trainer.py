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

from catgpt.jax.optimizers.splus import splus_get_eval_params

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
        lr_schedule: optax.Schedule | float | None = None,
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
            lr_schedule: Learning rate schedule for accurate WandB logging.
            rng: PRNG key for randomness. If None, uses jax.random.key(0).
        """
        self.training_config = training_config
        self.checkpoint_config = checkpoint_config
        self.wandb_config = wandb_config
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.full_config = full_config
        self.lr_schedule = lr_schedule

        # Check if using SPlus optimizer (requires different eval params)
        self.use_splus = (
            full_config is not None
            and full_config.optimizer.name.lower() == "splus"
        )

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
            batch: Batch with 'input' (tokens) and 'target' (HL-Gauss distribution).

        Returns:
            Tuple of (new state, metrics dict).
        """
        compute_dtype = self.compute_dtype
        head_config = self.model_config.output_heads if self.model_config else None

        def loss_fn(params):
            outputs = state.apply_fn(
                params,
                batch["input"],
                train=True,
                compute_dtype=compute_dtype,
            )  # dict[str, jax.Array]

            losses = {}

            # Self head: cross-entropy token reconstruction
            if "self" in outputs:
                # outputs["self"]: (batch, seq, vocab_size)
                # batch["input"]: (batch, seq) integer tokens
                self_loss = optax.softmax_cross_entropy_with_integer_labels(
                    outputs["self"],
                    batch["input"],
                ).mean()
                weight = head_config.self_weight if head_config else 0.1
                losses["self"] = self_loss * weight

            # Value head: cross-entropy with HL-Gauss target distribution
            if "value_logit" in outputs:
                # outputs["value_logit"]: (batch, num_bins) logits
                # batch["target"]: (batch, num_bins) HL-Gauss probability distribution
                value_loss = optax.softmax_cross_entropy(
                    outputs["value_logit"].astype(jnp.float32),
                    batch["target"].astype(jnp.float32),
                ).mean()
                weight = head_config.value_weight if head_config else 1.0
                losses["value"] = value_loss * weight

            # Policy head: cross-entropy with soft policy targets
            if "policy_logit" in outputs and "policy_target" in batch:
                # outputs["policy_logit"]: (batch, 64*73) logits
                # batch["policy_target"]: (batch, 64*73) policy distribution
                policy_loss = optax.softmax_cross_entropy(
                    outputs["policy_logit"].astype(jnp.float32),
                    batch["policy_target"].astype(jnp.float32),
                ).mean()
                weight = head_config.policy_weight if head_config else 1.0
                losses["policy"] = policy_loss * weight

            # Soft policy head: auxiliary head for softened policy target (KataGo method)
            # Applies temperature to soften the policy target, forcing the model to learn
            # relative rankings of lower-probability moves, not just the top 1-2.
            if "soft_policy_logit" in outputs and "policy_target" in batch:
                # Compute soft target: p^(1/T) then renormalize
                temp = head_config.soft_policy_temperature if head_config else 4.0
                # Add small epsilon for numerical stability before taking power
                soft_target = jnp.pow(batch["policy_target"] + 1e-10, 1.0 / temp)
                soft_target = soft_target / jnp.sum(soft_target, axis=-1, keepdims=True)

                soft_policy_loss = optax.softmax_cross_entropy(
                    outputs["soft_policy_logit"].astype(jnp.float32),
                    soft_target.astype(jnp.float32),
                ).mean()
                weight = head_config.soft_policy_weight if head_config else 8.0
                losses["soft_policy"] = soft_policy_loss * weight

            # Next capture head: cross-entropy with masking for None values
            # Target is -1 for positions without future captures
            if "next_capture_logit" in outputs and "next_capture_target" in batch:
                target = batch["next_capture_target"]  # (batch,) int32, -1 = invalid
                mask = (target >= 0).astype(jnp.float32)  # (batch,)

                # Replace -1 with 0 for safe indexing (masked out anyway)
                safe_target = jnp.maximum(target, 0)
                per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(
                    outputs["next_capture_logit"].astype(jnp.float32),
                    safe_target,
                )
                # Average over valid samples only
                masked_loss = jnp.sum(per_sample_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
                weight = head_config.next_capture_weight if head_config else 0.1
                losses["next_capture"] = masked_loss * weight

            # Next pawn move head: cross-entropy with masking for None values
            # Target is -1 for positions without future pawn moves
            if "next_pawn_move_logit" in outputs and "next_pawn_move_target" in batch:
                target = batch["next_pawn_move_target"]  # (batch,) int32, -1 = invalid
                mask = (target >= 0).astype(jnp.float32)  # (batch,)

                # Replace -1 with 0 for safe indexing (masked out anyway)
                safe_target = jnp.maximum(target, 0)
                per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(
                    outputs["next_pawn_move_logit"].astype(jnp.float32),
                    safe_target,
                )
                # Average over valid samples only
                masked_loss = jnp.sum(per_sample_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
                weight = head_config.next_pawn_move_weight if head_config else 0.1
                losses["next_pawn_move"] = masked_loss * weight

            total_loss = sum(losses.values())
            return total_loss, (outputs, losses)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (total_loss, (outputs, losses)), grads = grad_fn(state.params)

        # Update state
        state = state.apply_gradients(grads=grads)

        # Compute metrics
        metrics = {"loss": total_loss}

        # Per-head losses for logging
        for head_name, head_loss in losses.items():
            metrics[f"{head_name}_loss"] = head_loss

        # Value metrics (using expected value from HL-Gauss distribution)
        if "value" in outputs:
            # outputs["value"] is the expected value (batch,)
            # For target expected value, compute from target distribution
            num_bins = batch["target"].shape[-1]
            bin_centers = (jnp.arange(num_bins) + 0.5) / num_bins
            target_expected = jnp.sum(batch["target"] * bin_centers, axis=-1)  # (batch,)

            # MSE (on probability scale, comparing expected values)
            value_mse = jnp.mean((outputs["value"] - target_expected.astype(jnp.float32)) ** 2)
            metrics["value_mse"] = value_mse

            # Binary accuracy (binarize both predictions AND targets)
            preds = (outputs["value"] > 0.5).astype(jnp.float32)
            targets_binary = (target_expected > 0.5).astype(jnp.float32)
            accuracy = (preds == targets_binary).mean()
            metrics["accuracy"] = accuracy

        # Self head metrics
        if "self" in outputs:
            # Token prediction accuracy
            pred_tokens = jnp.argmax(outputs["self"], axis=-1)  # (batch, seq)
            self_accuracy = (pred_tokens == batch["input"]).mean()
            metrics["self_accuracy"] = self_accuracy

        # Policy head metrics
        if "policy_logit" in outputs and "policy_target" in batch:
            # Top-1 accuracy: compare argmax of prediction vs argmax of target
            pred_moves = jnp.argmax(outputs["policy_logit"], axis=-1)  # (batch,)
            target_moves = jnp.argmax(batch["policy_target"], axis=-1)  # (batch,)
            policy_accuracy = (pred_moves == target_moves).mean()
            metrics["policy_accuracy"] = policy_accuracy

        # Next capture head metrics (accuracy on valid samples only)
        if "next_capture_logit" in outputs and "next_capture_target" in batch:
            target = batch["next_capture_target"]
            mask = target >= 0
            pred = jnp.argmax(outputs["next_capture_logit"], axis=-1)
            correct = (pred == target) & mask
            accuracy = jnp.sum(correct.astype(jnp.float32)) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
            metrics["next_capture_accuracy"] = accuracy
            # Also track fraction of valid samples
            metrics["next_capture_valid_frac"] = jnp.mean(mask.astype(jnp.float32))

        # Next pawn move head metrics (accuracy on valid samples only)
        if "next_pawn_move_logit" in outputs and "next_pawn_move_target" in batch:
            target = batch["next_pawn_move_target"]
            mask = target >= 0
            pred = jnp.argmax(outputs["next_pawn_move_logit"], axis=-1)
            correct = (pred == target) & mask
            accuracy = jnp.sum(correct.astype(jnp.float32)) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
            metrics["next_pawn_move_accuracy"] = accuracy
            # Also track fraction of valid samples
            metrics["next_pawn_move_valid_frac"] = jnp.mean(mask.astype(jnp.float32))

        return state, metrics

    def _eval_step_impl(
        self,
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Perform a single evaluation step (implementation).

        Args:
            state: Train state (params may be EMA params for SPlus).
            batch: Batch with 'input' (tokens) and 'target' (HL-Gauss distribution).

        Returns:
            Metrics dict.
        """
        head_config = self.model_config.output_heads if self.model_config else None

        outputs = state.apply_fn(
            state.params,
            batch["input"],
            train=False,
            compute_dtype=self.compute_dtype,
        )  # dict[str, jax.Array]

        losses = {}

        # Self head: cross-entropy token reconstruction
        if "self" in outputs:
            self_loss = optax.softmax_cross_entropy_with_integer_labels(
                outputs["self"],
                batch["input"],
            ).mean()
            weight = head_config.self_weight if head_config else 0.1
            losses["self"] = self_loss * weight

        # Value head: cross-entropy with HL-Gauss target distribution
        if "value_logit" in outputs:
            # outputs["value_logit"]: (batch, num_bins) logits
            # batch["target"]: (batch, num_bins) HL-Gauss probability distribution
            value_loss = optax.softmax_cross_entropy(
                outputs["value_logit"].astype(jnp.float32),
                batch["target"].astype(jnp.float32),
            ).mean()
            weight = head_config.value_weight if head_config else 1.0
            losses["value"] = value_loss * weight

        # Policy head: cross-entropy with soft policy targets
        if "policy_logit" in outputs and "policy_target" in batch:
            # outputs["policy_logit"]: (batch, 64*73) logits
            # batch["policy_target"]: (batch, 64*73) policy distribution
            policy_loss = optax.softmax_cross_entropy(
                outputs["policy_logit"].astype(jnp.float32),
                batch["policy_target"].astype(jnp.float32),
            ).mean()
            weight = head_config.policy_weight if head_config else 1.0
            losses["policy"] = policy_loss * weight

        # Soft policy head: auxiliary head for softened policy target (KataGo method)
        if "soft_policy_logit" in outputs and "policy_target" in batch:
            # Compute soft target: p^(1/T) then renormalize
            temp = head_config.soft_policy_temperature if head_config else 4.0
            soft_target = jnp.pow(batch["policy_target"] + 1e-10, 1.0 / temp)
            soft_target = soft_target / jnp.sum(soft_target, axis=-1, keepdims=True)

            soft_policy_loss = optax.softmax_cross_entropy(
                outputs["soft_policy_logit"].astype(jnp.float32),
                soft_target.astype(jnp.float32),
            ).mean()
            weight = head_config.soft_policy_weight if head_config else 8.0
            losses["soft_policy"] = soft_policy_loss * weight

        # Next capture head: cross-entropy with masking for None values
        if "next_capture_logit" in outputs and "next_capture_target" in batch:
            target = batch["next_capture_target"]
            mask = (target >= 0).astype(jnp.float32)

            safe_target = jnp.maximum(target, 0)
            per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(
                outputs["next_capture_logit"].astype(jnp.float32),
                safe_target,
            )
            masked_loss = jnp.sum(per_sample_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            weight = head_config.next_capture_weight if head_config else 0.1
            losses["next_capture"] = masked_loss * weight

        # Next pawn move head: cross-entropy with masking for None values
        if "next_pawn_move_logit" in outputs and "next_pawn_move_target" in batch:
            target = batch["next_pawn_move_target"]
            mask = (target >= 0).astype(jnp.float32)

            safe_target = jnp.maximum(target, 0)
            per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(
                outputs["next_pawn_move_logit"].astype(jnp.float32),
                safe_target,
            )
            masked_loss = jnp.sum(per_sample_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
            weight = head_config.next_pawn_move_weight if head_config else 0.1
            losses["next_pawn_move"] = masked_loss * weight

        total_loss = sum(losses.values())

        # Compute metrics
        metrics = {"loss": total_loss}

        # Per-head losses
        for head_name, head_loss in losses.items():
            metrics[f"{head_name}_loss"] = head_loss

        # Value metrics (using expected value from HL-Gauss distribution)
        if "value" in outputs:
            # outputs["value"] is the expected value (batch,)
            # For target expected value, compute from target distribution
            num_bins = batch["target"].shape[-1]
            bin_centers = (jnp.arange(num_bins) + 0.5) / num_bins
            target_expected = jnp.sum(batch["target"] * bin_centers, axis=-1)  # (batch,)

            # MSE (on probability scale, comparing expected values)
            value_mse = jnp.mean((outputs["value"] - target_expected.astype(jnp.float32)) ** 2)
            metrics["value_mse"] = value_mse

            # Binary accuracy (binarize both predictions AND targets)
            preds = (outputs["value"] > 0.5).astype(jnp.float32)
            targets_binary = (target_expected > 0.5).astype(jnp.float32)
            accuracy = (preds == targets_binary).mean()
            metrics["accuracy"] = accuracy

        # Self head metrics
        if "self" in outputs:
            pred_tokens = jnp.argmax(outputs["self"], axis=-1)
            self_accuracy = (pred_tokens == batch["input"]).mean()
            metrics["self_accuracy"] = self_accuracy

        # Policy head metrics
        if "policy_logit" in outputs and "policy_target" in batch:
            # Top-1 accuracy: compare argmax of prediction vs argmax of target
            pred_moves = jnp.argmax(outputs["policy_logit"], axis=-1)  # (batch,)
            target_moves = jnp.argmax(batch["policy_target"], axis=-1)  # (batch,)
            policy_accuracy = (pred_moves == target_moves).mean()
            metrics["policy_accuracy"] = policy_accuracy

        # Next capture head metrics (accuracy on valid samples only)
        if "next_capture_logit" in outputs and "next_capture_target" in batch:
            target = batch["next_capture_target"]
            mask = target >= 0
            pred = jnp.argmax(outputs["next_capture_logit"], axis=-1)
            correct = (pred == target) & mask
            accuracy = jnp.sum(correct.astype(jnp.float32)) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
            metrics["next_capture_accuracy"] = accuracy
            metrics["next_capture_valid_frac"] = jnp.mean(mask.astype(jnp.float32))

        # Next pawn move head metrics (accuracy on valid samples only)
        if "next_pawn_move_logit" in outputs and "next_pawn_move_target" in batch:
            target = batch["next_pawn_move_target"]
            mask = target >= 0
            pred = jnp.argmax(outputs["next_pawn_move_logit"], axis=-1)
            correct = (pred == target) & mask
            accuracy = jnp.sum(correct.astype(jnp.float32)) / jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
            metrics["next_pawn_move_accuracy"] = accuracy
            metrics["next_pawn_move_valid_frac"] = jnp.mean(mask.astype(jnp.float32))

        return metrics

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

                # Update progress bar with key metrics
                epoch_pbar.update(1)
                postfix = {"loss": f"{loss:.4f}"}
                if "value_mse" in metrics:
                    postfix["mse"] = f"{float(metrics['value_mse']):.6f}"
                if "accuracy" in metrics:
                    postfix["acc"] = f"{float(metrics['accuracy']):.2%}"
                if "self_accuracy" in metrics:
                    postfix["self"] = f"{float(metrics['self_accuracy']):.2%}"
                if "policy_accuracy" in metrics:
                    postfix["pol"] = f"{float(metrics['policy_accuracy']):.2%}"
                epoch_pbar.set_postfix(postfix)

                # Log step metrics
                if self.global_step % self.wandb_config.log_every_steps == 0:
                    self._log_step(metrics)

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

    def get_eval_params(self) -> dict:
        """Get parameters for evaluation.

        For SPlus optimizer, returns EMA-corrected parameters.
        For other optimizers, returns the regular params.

        Returns:
            Parameters to use for evaluation.
        """
        if self.use_splus:
            splus_state = self._find_splus_state(self.state.opt_state)
            if splus_state is None:
                logger.warning("Could not find SPlusState in opt_state, using regular params")
                return self.state.params
            return splus_get_eval_params(splus_state)
        return self.state.params

    def _find_splus_state(self, opt_state):
        """Recursively find SPlusState in the optimizer state tree.

        The optimizer may be wrapped with gradient clipping, so the structure
        varies. This function searches for the SPlusState which has the 'ema'
        attribute needed for evaluation.

        Args:
            opt_state: Optimizer state (potentially nested tuple).

        Returns:
            SPlusState if found, None otherwise.
        """
        from catgpt.jax.optimizers.splus import SPlusState

        # Check if this is the SPlusState
        if isinstance(opt_state, SPlusState):
            return opt_state

        # If it's a tuple/list, search recursively
        if isinstance(opt_state, (tuple, list)):
            for item in opt_state:
                result = self._find_splus_state(item)
                if result is not None:
                    return result

        return None

    def _eval(self) -> dict[str, float]:
        """Run evaluation on validation set.

        Returns:
            Dictionary of validation metrics.
        """
        # Get eval params (EMA for SPlus, regular params otherwise)
        eval_params = self.get_eval_params()

        # Accumulators for weighted averaging
        metric_sums: dict[str, float] = {}
        total_samples = 0

        max_eval_steps = self.training_config.max_eval_steps

        val_iterator = tqdm(
            self.val_dataloader,
            desc="Validation",
            unit="batch",
            leave=False,
            total=max_eval_steps,
        )

        # Create eval state with appropriate params (EMA for SPlus)
        eval_state = self.state.replace(params=eval_params)

        for step_idx, batch in enumerate(val_iterator):
            if max_eval_steps is not None and step_idx >= max_eval_steps:
                break

            # Convert to JAX arrays
            batch = {k: jnp.array(v) for k, v in batch.items()}

            metrics = self._eval_step(eval_state, batch)

            batch_size = batch["input"].shape[0]
            total_samples += batch_size

            # Accumulate all metrics
            for key, value in metrics.items():
                if key not in metric_sums:
                    metric_sums[key] = 0.0
                metric_sums[key] += float(value) * batch_size

            # Update progress bar with running averages
            postfix = {}
            if "loss" in metric_sums:
                postfix["loss"] = f"{metric_sums['loss'] / total_samples:.4f}"
            if "value_mse" in metric_sums:
                postfix["mse"] = f"{metric_sums['value_mse'] / total_samples:.6f}"
            if "accuracy" in metric_sums:
                postfix["acc"] = f"{metric_sums['accuracy'] / total_samples:.2%}"
            if "self_accuracy" in metric_sums:
                postfix["self"] = f"{metric_sums['self_accuracy'] / total_samples:.2%}"
            if "policy_accuracy" in metric_sums:
                postfix["pol"] = f"{metric_sums['policy_accuracy'] / total_samples:.2%}"
            val_iterator.set_postfix(postfix)

        # Average all metrics
        result = {}
        for key, total in metric_sums.items():
            result[f"val_{key}"] = total / max(total_samples, 1)

        return result

    def _should_save_checkpoint(self) -> bool:
        """Check if we should save a checkpoint this pseudo-epoch."""
        return (self.current_epoch + 1) % self.checkpoint_config.save_every_epochs == 0

    def _log_step(self, metrics: dict[str, jax.Array]) -> None:
        """Log metrics for a training step.

        Args:
            metrics: Dictionary of metrics from training step.
        """
        if self._wandb_run is not None:
            import wandb

            # Get current learning rate from optimizer state
            lr = self._get_current_lr()

            log_dict = {
                "train/loss": float(metrics["loss"]),
                "train/learning_rate": lr,
                "train/global_step": self.global_step,
                "train/epoch": self.current_epoch,
            }

            # Add per-head losses
            if "value_loss" in metrics:
                log_dict["train/value_loss"] = float(metrics["value_loss"])
            if "self_loss" in metrics:
                log_dict["train/self_loss"] = float(metrics["self_loss"])
            if "policy_loss" in metrics:
                log_dict["train/policy_loss"] = float(metrics["policy_loss"])
            if "soft_policy_loss" in metrics:
                log_dict["train/soft_policy_loss"] = float(metrics["soft_policy_loss"])

            # Add value metrics
            if "value_mse" in metrics:
                log_dict["train/value_mse"] = float(metrics["value_mse"])
            if "accuracy" in metrics:
                log_dict["train/accuracy"] = float(metrics["accuracy"])

            # Add self head metrics
            if "self_accuracy" in metrics:
                log_dict["train/self_accuracy"] = float(metrics["self_accuracy"])

            # Add policy head metrics
            if "policy_accuracy" in metrics:
                log_dict["train/policy_accuracy"] = float(metrics["policy_accuracy"])

            # Add next capture head metrics
            if "next_capture_loss" in metrics:
                log_dict["train/next_capture_loss"] = float(metrics["next_capture_loss"])
            if "next_capture_accuracy" in metrics:
                log_dict["train/next_capture_accuracy"] = float(metrics["next_capture_accuracy"])
            if "next_capture_valid_frac" in metrics:
                log_dict["train/next_capture_valid_frac"] = float(metrics["next_capture_valid_frac"])

            # Add next pawn move head metrics
            if "next_pawn_move_loss" in metrics:
                log_dict["train/next_pawn_move_loss"] = float(metrics["next_pawn_move_loss"])
            if "next_pawn_move_accuracy" in metrics:
                log_dict["train/next_pawn_move_accuracy"] = float(metrics["next_pawn_move_accuracy"])
            if "next_pawn_move_valid_frac" in metrics:
                log_dict["train/next_pawn_move_valid_frac"] = float(metrics["next_pawn_move_valid_frac"])

            wandb.log(log_dict, step=self.global_step)

    def _get_current_lr(self) -> float:
        """Get current learning rate from schedule or optimizer state."""
        # Use the schedule if provided (most accurate)
        if self.lr_schedule is not None:
            if callable(self.lr_schedule):
                return float(self.lr_schedule(self.global_step))
            return float(self.lr_schedule)
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

            # For SPlus, also save the EMA params for inference use
            if self.use_splus:
                ema_params = self.get_eval_params()
                checkpointer.save(
                    path / "ema_params",
                    ema_params,
                    force=True,
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

        # For SPlus, also save the EMA params for inference use
        if self.use_splus:
            ema_params = self.get_eval_params()
            with (path / "ema_params.msgpack").open("wb") as f:
                f.write(to_bytes(ema_params))

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
