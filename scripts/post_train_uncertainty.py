#!/usr/bin/env python3
"""Post-train a new uncertainty head on a pre-trained transformer.

The uncertainty head predicts per-move value variance from the backbone's
representations. It uses Q·K^T attention (same architecture as the policy head)
with softplus activation for non-negative variance output.

Training phases:
  Phase 1 (0 to freeze_steps): Backbone frozen, only uncertainty head trains
  Phase 2 (freeze_steps to end): Backbone unfreezes with 1/100th head LR

Data: Enriched .bag files from generate_move_values.py containing per-child
bestQ distributions and WDL values from a teacher model. Variance targets are
computed in [-1, 1] Q-space from the stored bestQ probability distributions.

Usage:
    uv run python scripts/post_train_uncertainty.py \\
        --checkpoint checkpoints_jax/WDL_main/final \\
        --train-data "~/qu_bag/*.bag" \\
        --val-data "~/qu_bag_val/*.bag" \\
        --output-dir checkpoints_jax/uncertainty_head \\
        --batch-size 512 \\
        --max-steps 30000 \\
        --freeze-steps 6000
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import grain.python as pygrain
import jax
import jax.numpy as jnp
import msgpack
import numpy as np
import optax
from flax import traverse_util
from flax.training import train_state
from loguru import logger
from omegaconf import OmegaConf

from catgpt.core.data.grain.bagz import BagDataSource
from catgpt.core.utils.policy import parse_uci_move
from catgpt.core.utils.squares import flip_square, parse_square
from catgpt.core.utils.tokenizer import TokenizerConfig, tokenize
from catgpt.jax.evaluation.checkpoint import load_checkpoint
from catgpt.jax.models.transformer import BidirectionalTransformer

# dtype mapping
DTYPE_MAP = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

# Underpromotion piece type to index offset (must match policy.py)
_UNDERPROMO_PIECE_OFFSET = {"n": 0, "b": 1, "r": 2}


# =============================================================================
# Data pipeline
# =============================================================================


class ConvertEnrichedBagToUncertainty(pygrain.MapTransform):
    """Convert enriched .bag data to (tokens, variance_target, variance_mask).

    Reads the 'child_evals' field from enriched .bag files and computes
    per-move variance targets in [-1, 1] Q-space.

    Variance computation per child move:
        bin_centers_q = 2*(i+0.5)/num_bins - 1   for i in [0, num_bins)
        E[Q]   = sum(probs * bin_centers_q)
        E[Q^2] = sum(probs * bin_centers_q^2)
        Var(Q) = E[Q^2] - E[Q]^2
    """

    def __init__(
        self,
        tokenizer_config: TokenizerConfig | None = None,
        num_bins: int = 81,
    ) -> None:
        super().__init__()
        self._tok_config = tokenizer_config or TokenizerConfig()
        self._num_bins = num_bins

        # Precompute bin centers in [-1, 1] space
        self._bin_centers = (2 * (np.arange(num_bins) + 0.5) / num_bins - 1).astype(
            np.float32
        )
        self._bin_centers_sq = (self._bin_centers**2).astype(np.float32)

    def map(self, element: bytes):
        """Map enriched .bag record to (tokens, variance_target, variance_mask).

        Args:
            element: Raw msgpack bytes from .bag file.

        Returns:
            Tuple of (tokens, variance_target_flat, variance_mask_flat).
        """
        data = msgpack.unpackb(element, raw=False)

        fen = data["fen"]
        flip = fen.split()[1] == "b"

        # Tokenize FEN
        tokens = tokenize(fen, self._tok_config)

        # Initialize targets: (64, 73) = (from_square, to_plane)
        variance_target = np.zeros((64, 73), dtype=np.float32)
        variance_mask = np.zeros((64, 73), dtype=np.float32)

        child_evals = data.get("child_evals", [])
        for move_uci, bestq_bytes, _wdl_value in child_evals:
            # Decode bestQ distribution
            probs = np.frombuffer(bestq_bytes, dtype=np.float16).astype(np.float32)

            # Compute variance in [-1, 1] space
            e_q = np.dot(probs, self._bin_centers)
            e_q2 = np.dot(probs, self._bin_centers_sq)
            var = max(e_q2 - e_q**2, 0.0)  # Clamp for numerical safety

            # Encode move to (from_idx, to_idx) matching policy tensor layout
            from_sq, to_sq, promo = parse_uci_move(move_uci)
            if flip:
                from_sq = flip_square(from_sq)
                to_sq = flip_square(to_sq)

            from_idx = parse_square(from_sq)
            if promo and promo != "q":
                file_diff = ord(to_sq[0]) - ord(from_sq[0])
                to_idx = 64 + _UNDERPROMO_PIECE_OFFSET[promo] * 3 + (file_diff + 1)
            else:
                to_idx = parse_square(to_sq)

            variance_target[from_idx, to_idx] = var
            variance_mask[from_idx, to_idx] = 1.0

        return tokens, variance_target.reshape(-1), variance_mask.reshape(-1)


class BatchToDict(pygrain.MapTransform):
    """Convert batched tuples to training dict."""

    def map(self, element):
        tokens, variance_target, variance_mask = element
        return {
            "input": np.asarray(tokens, dtype=np.int32),
            "uncertainty_target": np.asarray(variance_target, dtype=np.float32),
            "uncertainty_mask": np.asarray(variance_mask, dtype=np.float32),
        }


def create_dataloader(
    data_path: str,
    *,
    batch_size: int,
    tokenizer_config: TokenizerConfig,
    num_bins: int = 81,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 4,
) -> pygrain.DataLoader:
    """Create a PyGrain DataLoader for enriched .bag files.

    Args:
        data_path: Path or glob pattern for .bag files.
        batch_size: Batch size.
        tokenizer_config: Tokenizer configuration.
        num_bins: Number of HL-Gauss bins.
        shuffle: Whether to shuffle the data.
        seed: Random seed.
        num_workers: Number of data loading workers.

    Returns:
        PyGrain DataLoader yielding dicts with 'input', 'uncertainty_target', 'uncertainty_mask'.
    """
    bag_source = BagDataSource(data_path)
    logger.info(f"Data source: {len(bag_source):,} records from {data_path}")

    sampler = pygrain.IndexSampler(
        num_records=len(bag_source),
        shard_options=pygrain.NoSharding(),
        shuffle=shuffle,
        seed=seed,
        num_epochs=None,  # Infinite
    )

    transformations = (
        ConvertEnrichedBagToUncertainty(tokenizer_config, num_bins=num_bins),
        pygrain.Batch(batch_size=batch_size, drop_remainder=True),
        BatchToDict(),
    )

    return pygrain.DataLoader(
        data_source=bag_source,
        sampler=sampler,
        operations=transformations,
        worker_count=num_workers,
        read_options=pygrain.ReadOptions(prefetch_buffer_size=2),
    )


# =============================================================================
# Model setup
# =============================================================================


def create_model_with_uncertainty(
    checkpoint_path: str | Path,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[BidirectionalTransformer, dict, "JaxModelConfig", "JaxTokenizerConfig"]:
    """Load pretrained model and add uncertainty head.

    Loads the checkpoint, enables the uncertainty head in the config,
    creates a new model, initializes the new head parameters, and merges
    them with the pretrained backbone parameters.

    Args:
        checkpoint_path: Path to pretrained checkpoint.
        compute_dtype: Compute dtype for initialization.

    Returns:
        Tuple of (model, merged_params, model_config, tokenizer_config).
    """
    # Load pretrained checkpoint
    loaded = load_checkpoint(checkpoint_path)
    pretrained_params = loaded.params

    # Enable uncertainty head
    model_config = copy.deepcopy(loaded.model_config)
    model_config.output_heads.uncertainty_head = True

    # Create new model with uncertainty head
    model = BidirectionalTransformer.from_model_config(model_config)

    # Initialize full model to get new head parameter shapes
    rng = jax.random.key(42)
    dummy_input = jnp.zeros(
        (1, loaded.tokenizer_config.sequence_length), dtype=jnp.int32
    )
    full_init_params = model.init(
        rng, dummy_input, train=False, compute_dtype=compute_dtype
    )

    # Merge: pretrained backbone + newly initialized uncertainty head
    pretrained_flat = traverse_util.flatten_dict(pretrained_params)
    init_flat = traverse_util.flatten_dict(full_init_params)

    merged_flat = {}
    new_param_names = []
    for key in init_flat:
        if key in pretrained_flat:
            merged_flat[key] = pretrained_flat[key]
        else:
            merged_flat[key] = init_flat[key]
            new_param_names.append("/".join(str(k) for k in key))

    merged_params = traverse_util.unflatten_dict(merged_flat)

    # Log what's new
    new_param_count = sum(
        init_flat[k].size
        for k in init_flat
        if k not in pretrained_flat
    )
    pretrained_count = sum(v.size for v in pretrained_flat.values())
    logger.info(
        f"Model params: {pretrained_count:,} pretrained + {new_param_count:,} new "
        f"({len(new_param_names)} tensors)"
    )
    for name in new_param_names:
        logger.info(f"  New: {name}")

    return model, merged_params, model_config, loaded.tokenizer_config


# =============================================================================
# Optimizer
# =============================================================================


def create_post_train_optimizer(
    params: dict,
    head_lr: float,
    freeze_steps: int,
    backbone_lr_ratio: float = 0.01,
    weight_decay: float = 0.01,
    warmup_steps: int = 500,
    gradient_clip: float = 1.0,
) -> tuple[optax.GradientTransformation, callable]:
    """Create AdamW with differential LR and freeze/unfreeze schedule.

    Phase 1 (step < freeze_steps): backbone LR = 0, head LR = head_lr (with warmup)
    Phase 2 (step >= freeze_steps): backbone LR = head_lr * backbone_lr_ratio

    Args:
        params: Model parameters (for structure-based labeling).
        head_lr: Learning rate for the uncertainty head.
        freeze_steps: Number of steps to keep backbone frozen.
        backbone_lr_ratio: Backbone LR as fraction of head_lr after unfreezing.
        weight_decay: Weight decay for AdamW.
        warmup_steps: Linear warmup steps for head LR.
        gradient_clip: Maximum gradient norm.

    Returns:
        Tuple of (optimizer, lr_schedule_fn) where lr_schedule_fn maps step to head LR.
    """

    # Head schedule: linear warmup then constant
    def head_schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        warmup_factor = jnp.minimum(step / jnp.maximum(warmup_steps, 1), 1.0)
        return head_lr * warmup_factor

    # Backbone schedule: zero during freeze, then head_lr * ratio
    def backbone_schedule(step):
        step = jnp.asarray(step, dtype=jnp.float32)
        unfrozen = jnp.where(step >= freeze_steps, 1.0, 0.0)
        return head_lr * backbone_lr_ratio * unfrozen

    # Label function: uncertainty_* → "head", everything else → "backbone"
    def label_fn(params):
        def _label_leaf(path, _leaf):
            for part in path:
                key = part.key if hasattr(part, "key") else str(part)
                if "uncertainty" in key:
                    return "head"
            return "backbone"

        return jax.tree_util.tree_map_with_path(_label_leaf, params)

    optimizer = optax.multi_transform(
        transforms={
            "head": optax.chain(
                optax.clip_by_global_norm(gradient_clip),
                optax.adamw(head_schedule, weight_decay=weight_decay),
            ),
            "backbone": optax.chain(
                optax.clip_by_global_norm(gradient_clip),
                optax.adamw(backbone_schedule, weight_decay=weight_decay),
            ),
        },
        param_labels=label_fn,
    )

    return optimizer, head_schedule


# =============================================================================
# Training
# =============================================================================


class TrainState(train_state.TrainState):
    """Extended train state."""

    pass


def make_train_step(model, compute_dtype):
    """Create JIT-compiled training step function.

    Args:
        model: The BidirectionalTransformer model.
        compute_dtype: Compute dtype for mixed precision.

    Returns:
        JIT-compiled train_step function.
    """

    @jax.jit
    def train_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """Single training step: forward + uncertainty MSE loss + backward.

        Loss: mean squared error between predicted and target variance,
        averaged over legal moves only.
        """

        def loss_fn(params):
            outputs = state.apply_fn(
                params,
                batch["input"],
                train=True,
                compute_dtype=compute_dtype,
            )

            pred = outputs["uncertainty"].astype(jnp.float32)  # (batch, 4672)
            target = batch["uncertainty_target"].astype(jnp.float32)
            mask = batch["uncertainty_mask"].astype(jnp.float32)

            sq_err = (pred - target) ** 2
            masked_sq_err = sq_err * mask

            # Average over legal moves only
            loss = masked_sq_err.sum() / jnp.maximum(mask.sum(), 1.0)

            # Auxiliary metrics
            # Mean absolute error on legal moves
            mae = (jnp.abs(pred - target) * mask).sum() / jnp.maximum(mask.sum(), 1.0)

            # Mean predicted variance (on legal moves)
            mean_pred = (pred * mask).sum() / jnp.maximum(mask.sum(), 1.0)
            mean_target = (target * mask).sum() / jnp.maximum(mask.sum(), 1.0)

            return loss, {"mae": mae, "mean_pred": mean_pred, "mean_target": mean_target}

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux_metrics), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)

        metrics = {"loss": loss, **aux_metrics}
        metrics["grad_norm"] = optax.global_norm(grads)

        return state, metrics

    return train_step


def make_eval_step(model, compute_dtype):
    """Create JIT-compiled evaluation step function."""

    @jax.jit
    def eval_step(
        state: TrainState,
        batch: dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        outputs = state.apply_fn(
            state.params,
            batch["input"],
            train=False,
            compute_dtype=compute_dtype,
        )

        pred = outputs["uncertainty"].astype(jnp.float32)
        target = batch["uncertainty_target"].astype(jnp.float32)
        mask = batch["uncertainty_mask"].astype(jnp.float32)

        sq_err = (pred - target) ** 2
        masked_sq_err = sq_err * mask
        loss = masked_sq_err.sum() / jnp.maximum(mask.sum(), 1.0)
        mae = (jnp.abs(pred - target) * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        mean_pred = (pred * mask).sum() / jnp.maximum(mask.sum(), 1.0)
        mean_target = (target * mask).sum() / jnp.maximum(mask.sum(), 1.0)

        return {
            "loss": loss,
            "mae": mae,
            "mean_pred": mean_pred,
            "mean_target": mean_target,
        }

    return eval_step


def evaluate(
    state: TrainState,
    val_loader,
    eval_step_fn,
    max_eval_steps: int = 200,
) -> dict[str, float]:
    """Run evaluation and return averaged metrics."""
    metric_sums: dict[str, float] = {}
    total_batches = 0

    for step_idx, batch in enumerate(val_loader):
        if step_idx >= max_eval_steps:
            break

        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        metrics = eval_step_fn(state, batch_jax)

        for k, v in metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        total_batches += 1

    return {k: v / max(total_batches, 1) for k, v in metric_sums.items()}


def save_checkpoint(
    state: TrainState,
    path: Path,
    global_step: int,
    model_config,
    tokenizer_config,
    best_val_loss: float,
) -> None:
    """Save checkpoint with params, state, and configs."""
    path.mkdir(parents=True, exist_ok=True)

    # Save params with Orbax
    try:
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path.resolve() / "params", state.params, force=True)
    except ImportError:
        from flax.serialization import to_bytes

        with (path / "params.msgpack").open("wb") as f:
            f.write(to_bytes(state.params))

    # Trainer state
    trainer_state = {
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }
    with (path / "trainer_state.json").open("w") as f:
        json.dump(trainer_state, f, indent=2)

    # Model config
    model_dict = asdict(model_config)
    OmegaConf.save(OmegaConf.create(model_dict), path / "model_config.yaml")

    # Tokenizer config
    tok_dict = asdict(tokenizer_config)
    OmegaConf.save(OmegaConf.create(tok_dict), path / "tokenizer_config.yaml")

    logger.info(f"Saved checkpoint to {path} (step={global_step})")


# =============================================================================
# Main
# =============================================================================


def resolve_data_path(pattern: str) -> str:
    """Resolve data path: if directory, glob for *.bag; otherwise return as-is."""
    expanded = str(Path(pattern).expanduser())
    if Path(expanded).is_dir():
        bags = sorted(glob.glob(os.path.join(expanded, "*.bag")))
        if not bags:
            raise FileNotFoundError(f"No .bag files in {expanded}")
        # Return glob pattern for BagDataSource
        return os.path.join(expanded, "*.bag")
    return expanded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-train uncertainty head on pre-trained transformer."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to pretrained checkpoint."
    )
    parser.add_argument(
        "--train-data", required=True, help="Training data path/glob/dir."
    )
    parser.add_argument(
        "--val-data", default=None, help="Validation data path/glob/dir."
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints_jax/uncertainty",
        help="Output directory for checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--freeze-steps", type=int, default=6000)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument(
        "--backbone-lr-ratio",
        type=float,
        default=0.01,
        help="Backbone LR = head_lr * ratio after unfreezing (default: 0.01 = 1/100).",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--compute-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--matmul-precision",
        default="high",
        choices=["default", "high", "highest"],
    )
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between evals.")
    parser.add_argument("--save-every", type=int, default=2000, help="Steps between saves.")
    parser.add_argument("--max-eval-steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", default="catgpt-uncertainty")
    parser.add_argument("--wandb-name", default=None)

    args = parser.parse_args()

    # JAX setup
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    compute_dtype = DTYPE_MAP.get(args.compute_dtype, jnp.bfloat16)

    logger.info(f"JAX devices: {jax.device_count()} - {jax.devices()}")
    logger.info(f"JAX backend: {jax.default_backend()}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model, params, model_config, tokenizer_config = create_model_with_uncertainty(
        args.checkpoint, compute_dtype=compute_dtype
    )

    tok_config = TokenizerConfig(
        sequence_length=tokenizer_config.sequence_length,
        include_halfmove=tokenizer_config.include_halfmove,
    )
    num_bins = model_config.output_heads.value_num_bins

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    train_path = resolve_data_path(args.train_data)
    train_loader = create_dataloader(
        train_path,
        batch_size=args.batch_size,
        tokenizer_config=tok_config,
        num_bins=num_bins,
        shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    val_loader = None
    if args.val_data:
        val_path = resolve_data_path(args.val_data)
        val_loader = create_dataloader(
            val_path,
            batch_size=args.batch_size,
            tokenizer_config=tok_config,
            num_bins=num_bins,
            shuffle=False,
            seed=args.seed,
            num_workers=max(1, args.num_workers // 2),
        )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    optimizer, lr_schedule = create_post_train_optimizer(
        params,
        head_lr=args.head_lr,
        freeze_steps=args.freeze_steps,
        backbone_lr_ratio=args.backbone_lr_ratio,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )

    # -------------------------------------------------------------------------
    # JIT compile train/eval steps
    # -------------------------------------------------------------------------
    train_step_fn = make_train_step(model, compute_dtype)
    eval_step_fn = make_eval_step(model, compute_dtype)

    # -------------------------------------------------------------------------
    # W&B
    # -------------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args),
            )
            logger.info(f"W&B: {wandb.run.url}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}")

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0

    logger.info(
        f"Starting post-training: {args.max_steps} steps, "
        f"freeze_steps={args.freeze_steps}, "
        f"head_lr={args.head_lr}, backbone_lr_ratio={args.backbone_lr_ratio}, "
        f"batch_size={args.batch_size}"
    )

    data_iter = iter(train_loader)
    start_time = time.time()
    log_loss_sum = 0.0
    log_steps = 0

    while global_step < args.max_steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        # Train step
        state, metrics = train_step_fn(state, batch_jax)
        global_step += 1

        loss = float(metrics["loss"])
        log_loss_sum += loss
        log_steps += 1

        # Phase transition logging
        if global_step == args.freeze_steps:
            logger.info(
                f"Step {global_step}: Unfreezing backbone "
                f"(backbone_lr={args.head_lr * args.backbone_lr_ratio:.2e})"
            )

        # Periodic logging
        if global_step % args.log_every == 0:
            avg_loss = log_loss_sum / log_steps
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            phase = "frozen" if global_step < args.freeze_steps else "unfrozen"
            current_lr = float(lr_schedule(global_step))

            logger.info(
                f"[{phase}] step={global_step}/{args.max_steps} | "
                f"loss={avg_loss:.6f} | mae={float(metrics['mae']):.6f} | "
                f"pred={float(metrics['mean_pred']):.6f} tgt={float(metrics['mean_target']):.6f} | "
                f"lr={current_lr:.2e} | grad={float(metrics['grad_norm']):.4f} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            if wandb_run is not None:
                import wandb

                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/mae": float(metrics["mae"]),
                        "train/mean_pred": float(metrics["mean_pred"]),
                        "train/mean_target": float(metrics["mean_target"]),
                        "train/grad_norm": float(metrics["grad_norm"]),
                        "train/head_lr": current_lr,
                        "train/phase": 0 if global_step < args.freeze_steps else 1,
                    },
                    step=global_step,
                )

            log_loss_sum = 0.0
            log_steps = 0

        # Evaluation
        if val_loader is not None and global_step % args.eval_every == 0:
            val_metrics = evaluate(
                state, val_loader, eval_step_fn, max_eval_steps=args.max_eval_steps
            )
            logger.info(
                f"  VAL step={global_step}: loss={val_metrics['loss']:.6f} | "
                f"mae={val_metrics['mae']:.6f}"
            )

            if wandb_run is not None:
                import wandb

                wandb.log(
                    {f"val/{k}": v for k, v in val_metrics.items()},
                    step=global_step,
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    state,
                    output_dir / "best",
                    global_step,
                    model_config,
                    tokenizer_config,
                    best_val_loss,
                )

        # Periodic save
        if global_step % args.save_every == 0:
            save_checkpoint(
                state,
                output_dir / f"step_{global_step}",
                global_step,
                model_config,
                tokenizer_config,
                best_val_loss,
            )

    # Final save
    save_checkpoint(
        state,
        output_dir / "final",
        global_step,
        model_config,
        tokenizer_config,
        best_val_loss,
    )

    elapsed = time.time() - start_time
    logger.info(
        f"Post-training complete: {global_step} steps in {elapsed:.0f}s "
        f"({global_step / elapsed:.1f} steps/s), best_val_loss={best_val_loss:.6f}"
    )

    if wandb_run is not None:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
