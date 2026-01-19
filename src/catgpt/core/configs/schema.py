"""Strongly-typed configuration schemas for CatGPT training.

These dataclasses provide validation, IDE support, and serve as the
single source of truth for all configuration options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    name: str = "transformer"
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int | None = None  # Defaults to 4 * hidden_size if None
    vocab_size: int = 28  # From tokenizer.VOCAB_SIZE
    seq_length: int = 64
    activation: str = "gelu"

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_size

        if self.hidden_size % self.num_heads != 0:
            msg = f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)


@dataclass
class TokenizerConfig:
    """Configuration for FEN tokenization."""

    sequence_length: int = 64
    include_halfmove: bool = False


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""

    name: str = "adamw"  # "adamw" | "splus"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # AdamW-specific
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # SPlus-specific
    splus_b1: float = 0.9
    splus_b2: float = 0.999
    splus_ema_rate: float = 0.999
    splus_inverse_every: int = 100
    splus_max_dim: int = 10000


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio


@dataclass
class TrainingConfig:
    """Configuration for training loop.

    Training is step-based with "pseudo-epochs" - fixed step intervals
    at which validation runs and checkpoints are saved.
    """

    max_steps: int = 300_000  # Total training steps
    steps_per_epoch: int = 3000  # Steps per pseudo-epoch (for eval/checkpoint)
    batch_size: int = 64
    gradient_clip: float | None = 1.0
    gradient_accumulation_steps: int = 1
    max_eval_steps: int | None = None  # Max validation steps (None = full dataset)

    # Compilation
    compile_model: bool = True  # JIT compile the model


@dataclass
class DataConfig:
    """Configuration for data loading."""

    train_path: str = "data/train/*.bag"
    val_path: str | None = "data/val/*.bag"
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    seed: int = 42  # Random seed for data shuffling


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""

    dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every_epochs: int = 1  # Save every N pseudo-epochs
    keep_last: int = 5
    save_optimizer: bool = True

    def __post_init__(self) -> None:
        """Convert string to Path if needed."""
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    project: str = "catgpt"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    log_every_steps: int = 10


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enabled: bool = False
    backend: str = "jax"  # JAX distributed training


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    seed: int = 42


def config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Create ExperimentConfig from a dictionary (e.g., from OmegaConf).

    Args:
        data: Dictionary with configuration values.

    Returns:
        ExperimentConfig instance.
    """
    return ExperimentConfig(
        model=ModelConfig(**data.get("model", {})),
        tokenizer=TokenizerConfig(**data.get("tokenizer", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        scheduler=SchedulerConfig(**data.get("scheduler", {})),
        training=TrainingConfig(**data.get("training", {})),
        data=DataConfig(**data.get("data", {})),
        checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
        wandb=WandbConfig(**data.get("wandb", {})),
        distributed=DistributedConfig(**data.get("distributed", {})),
        seed=data.get("seed", 42),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert ExperimentConfig to a dictionary for serialization.

    Args:
        config: ExperimentConfig instance.

    Returns:
        Dictionary representation.
    """
    from dataclasses import asdict

    result = asdict(config)
    # Convert Path objects to strings for YAML serialization
    result["checkpoint"]["dir"] = str(result["checkpoint"]["dir"])
    return result
