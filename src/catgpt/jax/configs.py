"""JAX-specific configuration schemas for CatGPT training.

Extends the core configuration with JAX-specific options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JaxOutputHeadConfig:
    """Configuration for output heads.

    The model can have multiple output heads for different tasks:
    - self_head: Token reconstruction (auxiliary task for stability)
    - value_head: Win probability prediction (primary task)
    - policy_head: Move distribution prediction

    Loss weights control the contribution of each head to the total loss.

    The value head uses HL-Gauss (Histogram Loss with Gaussian targets) which
    converts scalar win probability into a soft categorical distribution,
    enabling training with cross-entropy loss instead of MSE. This approach
    scales better with larger networks and reduces overfitting.
    See: https://arxiv.org/abs/2403.03950

    The policy head outputs a (64, 73) tensor representing move probabilities
    over (from_square, to_square) pairs. The 73 "to" indices are:
    - 0-63: Normal destination squares
    - 64-72: Underpromotion targets (3 pieces × 3 directions)
    """

    # Head enable flags
    self_head: bool = True  # Token reconstruction (auxiliary task)
    value_head: bool = True  # Win probability
    policy_head: bool = False  # Move distribution

    # Loss weights
    self_weight: float = 0.1  # Usually lower than primary task
    value_weight: float = 1.0
    policy_weight: float = 1.0

    # Soft policy auxiliary target (KataGo method)
    # Applies temperature to soften the policy target, forcing the model to learn
    # relative rankings of lower-probability moves, not just the top 1-2.
    # See: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#auxiliary-soft-policy-target
    soft_policy_head: bool = False  # Enable auxiliary soft policy head
    soft_policy_temperature: float = 4.0  # Temperature for softening (higher = more uniform)
    soft_policy_weight: float = 8.0  # Loss weight (compensates for smaller gradients)

    # Square prediction heads (noisy auxiliary targets)
    # Predict which square has the piece that will be captured next / pawn that will move next
    # These are 64-way classification tasks with masking for positions without valid targets
    next_capture_head: bool = False
    next_capture_weight: float = 0.1  # Small weight for noisy auxiliary target

    next_pawn_move_head: bool = False
    next_pawn_move_weight: float = 0.1  # Small weight for noisy auxiliary target

    # HL-Gauss configuration for value head
    # Converts scalar win probability (0-1) to categorical distribution
    value_num_bins: int = 81  # Number of bins for value distribution
    value_sigma_ratio: float = 0.75  # sigma = ratio * bin_width (0.75 spreads to ~5 bins)


@dataclass
class SmolgenConfig:
    """Configuration for Smolgen attention bias generation.

    Smolgen dynamically generates position-dependent attention biases conditioned
    on the input, rather than using fixed positional encodings. This allows the
    model to learn that certain square pairs should attend differently based on
    the actual board state (e.g., open vs closed positions).

    Architecture:
    1. Compress: (seq_len, hidden) → (seq_len, hidden_channels) - no bias
    2. Flatten: → (seq_len * hidden_channels,) - global representation
    3. Dense1 → GELU → LayerNorm: → (hidden_size,)
    4. Dense2 → GELU → LayerNorm: → (num_heads * gen_size,)
    5. Shared weight_gen: → (num_heads, seq_len, seq_len) - no activation

    The final weight_gen layer is shared across all transformer blocks.

    See: https://lczero.org/blog/2024/02/transformer-progress/
    """

    enabled: bool = False
    hidden_channels: int = 32  # Compression dimension
    hidden_size: int = 256  # Hidden layer size
    gen_size: int = 256  # Per-head generation dimension


@dataclass
class JaxModelConfig:
    """Configuration for JAX model architecture."""

    name: str = "transformer"
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int | None = None  # Defaults to 4 * hidden_size if None
    vocab_size: int = 26  # From tokenizer.VOCAB_SIZE
    seq_length: int = 64
    activation: str = "gelu"

    # Output head configuration
    output_heads: JaxOutputHeadConfig = field(default_factory=JaxOutputHeadConfig)

    # Smolgen: dynamic attention bias generation
    smolgen: SmolgenConfig = field(default_factory=SmolgenConfig)

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_size

        if self.hidden_size % self.num_heads != 0:
            msg = f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)

        # Convert dict to dataclass if needed (for YAML loading)
        if isinstance(self.output_heads, dict):
            self.output_heads = JaxOutputHeadConfig(**self.output_heads)
        if isinstance(self.smolgen, dict):
            self.smolgen = SmolgenConfig(**self.smolgen)


@dataclass
class JaxTokenizerConfig:
    """Configuration for FEN tokenization."""

    sequence_length: int = 64
    include_halfmove: bool = False


@dataclass
class JaxOptimizerConfig:
    """Configuration for JAX optimizer."""

    name: str = "adamw"  # "adamw" | "splus"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # AdamW-specific
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    # SPlus-specific
    splus_b1: float = 0.9
    splus_b2: float = 0.999
    splus_ema_rate: float = 0.999
    splus_inverse_every: int = 100
    splus_max_dim: int = 10000


@dataclass
class JaxSchedulerConfig:
    """Configuration for learning rate scheduler."""

    name: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio


@dataclass
class JaxTrainingConfig:
    """Configuration for JAX training loop.

    Training is step-based with "pseudo-epochs" - fixed step intervals
    at which validation runs and checkpoints are saved.
    """

    max_steps: int = 300_000  # Total training steps
    steps_per_epoch: int = 3000  # Steps per pseudo-epoch (for eval/checkpoint)
    batch_size: int = 64
    gradient_clip: float | None = 1.0
    gradient_accumulation_steps: int = 1
    max_eval_steps: int | None = None  # Max validation steps (None = full dataset)

    # JAX-specific
    jit_compile: bool = True  # JIT compile the training step
    use_pmap: bool = False  # Use pmap for multi-device training
    donate_argnums: bool = True  # Donate buffers for memory efficiency

    # Mixed precision settings
    mixed_precision: bool = True  # Enable bfloat16 mixed precision
    precision_dtype: str = "bfloat16"  # "bfloat16" or "float16"
    # Tensor core settings: "default", "high", "highest" (highest uses TF32 on Ampere+)
    matmul_precision: str = "high"


@dataclass
class JaxDataConfig:
    """Configuration for data loading."""

    train_path: str = "data/train/*.bag"
    val_path: str | None = "data/val/*.bag"
    num_workers: int = 4
    prefetch_factor: int = 2
    seed: int = 42  # Random seed for data shuffling


@dataclass
class JaxCheckpointConfig:
    """Configuration for checkpointing."""

    dir: Path = field(default_factory=lambda: Path("checkpoints"))
    save_every_epochs: int = 1  # Save every N pseudo-epochs
    keep_last: int = 5
    save_optimizer: bool = True
    use_orbax: bool = True  # Use Orbax for checkpointing

    def __post_init__(self) -> None:
        """Convert string to Path if needed."""
        if isinstance(self.dir, str):
            self.dir = Path(self.dir)


@dataclass
class JaxWandbConfig:
    """Configuration for Weights & Biases logging."""

    enabled: bool = True
    project: str = "catgpt-jax"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    log_every_steps: int = 10


@dataclass
class JaxDistributedConfig:
    """Configuration for JAX distributed training."""

    enabled: bool = False
    mesh_shape: tuple[int, ...] | None = None  # For sharded training
    data_axis: str = "data"  # Name of the data parallelism axis


@dataclass
class JaxExperimentConfig:
    """Top-level JAX configuration combining all sub-configs."""

    model: JaxModelConfig = field(default_factory=JaxModelConfig)
    tokenizer: JaxTokenizerConfig = field(default_factory=JaxTokenizerConfig)
    optimizer: JaxOptimizerConfig = field(default_factory=JaxOptimizerConfig)
    scheduler: JaxSchedulerConfig = field(default_factory=JaxSchedulerConfig)
    training: JaxTrainingConfig = field(default_factory=JaxTrainingConfig)
    data: JaxDataConfig = field(default_factory=JaxDataConfig)
    checkpoint: JaxCheckpointConfig = field(default_factory=JaxCheckpointConfig)
    wandb: JaxWandbConfig = field(default_factory=JaxWandbConfig)
    distributed: JaxDistributedConfig = field(default_factory=JaxDistributedConfig)

    seed: int = 42


def jax_config_from_dict(data: dict[str, Any]) -> JaxExperimentConfig:
    """Create JaxExperimentConfig from a dictionary.

    Args:
        data: Dictionary with configuration values.

    Returns:
        JaxExperimentConfig instance.
    """
    return JaxExperimentConfig(
        model=JaxModelConfig(**data.get("model", {})),
        tokenizer=JaxTokenizerConfig(**data.get("tokenizer", {})),
        optimizer=JaxOptimizerConfig(**data.get("optimizer", {})),
        scheduler=JaxSchedulerConfig(**data.get("scheduler", {})),
        training=JaxTrainingConfig(**data.get("training", {})),
        data=JaxDataConfig(**data.get("data", {})),
        checkpoint=JaxCheckpointConfig(**data.get("checkpoint", {})),
        wandb=JaxWandbConfig(**data.get("wandb", {})),
        distributed=JaxDistributedConfig(**data.get("distributed", {})),
        seed=data.get("seed", 42),
    )


def jax_config_to_dict(config: JaxExperimentConfig) -> dict[str, Any]:
    """Convert JaxExperimentConfig to a dictionary for serialization.

    Args:
        config: JaxExperimentConfig instance.

    Returns:
        Dictionary representation.
    """
    from dataclasses import asdict

    result = asdict(config)
    # Convert Path objects to strings for YAML serialization
    result["checkpoint"]["dir"] = str(result["checkpoint"]["dir"])
    return result


# =============================================================================
# Evaluation Configuration
# =============================================================================


@dataclass
class JaxMCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 800
    c_puct: float = 1.75
    fpu_value: float = -1.0


@dataclass
class JaxEvalEngineConfig:
    """Configuration for evaluation engine."""

    type: str = "value"  # "value", "policy", or "mcts"
    batch_size: int = 256
    mcts: JaxMCTSConfig = field(default_factory=JaxMCTSConfig)

    def __post_init__(self) -> None:
        """Convert dict to dataclass if needed."""
        if isinstance(self.mcts, dict):
            self.mcts = JaxMCTSConfig(**self.mcts)


@dataclass
class JaxEvalBenchmarkConfig:
    """Configuration for evaluation benchmarks."""

    names: list[str] = field(default_factory=lambda: ["puzzles"])
    max_puzzles: int | None = 10000
    puzzles_path: str = "puzzles/puzzles.csv"
    high_rated_puzzles_path: str = "puzzles/high_rated_puzzles.csv"


@dataclass
class JaxEvalComputeConfig:
    """Configuration for evaluation compute settings."""

    matmul_precision: str = "high"  # "high" or "highest"
    compute_dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"


@dataclass
class JaxEvalWandbConfig:
    """Configuration for evaluation W&B logging."""

    enabled: bool = True
    project: str = "catgpt-puzzles"
    entity: str | None = None
    tags: list[str] = field(default_factory=lambda: ["eval"])


@dataclass
class JaxEvalConfig:
    """Top-level configuration for JAX model evaluation."""

    # Checkpoint path(s) - can be a single path or list
    checkpoint: str = "checkpoints_jax/best"

    engine: JaxEvalEngineConfig = field(default_factory=JaxEvalEngineConfig)
    benchmark: JaxEvalBenchmarkConfig = field(default_factory=JaxEvalBenchmarkConfig)
    compute: JaxEvalComputeConfig = field(default_factory=JaxEvalComputeConfig)
    wandb: JaxEvalWandbConfig = field(default_factory=JaxEvalWandbConfig)

    verbose: bool = False

    def __post_init__(self) -> None:
        """Convert dicts to dataclasses if needed."""
        if isinstance(self.engine, dict):
            self.engine = JaxEvalEngineConfig(**self.engine)
        if isinstance(self.benchmark, dict):
            self.benchmark = JaxEvalBenchmarkConfig(**self.benchmark)
        if isinstance(self.compute, dict):
            self.compute = JaxEvalComputeConfig(**self.compute)
        if isinstance(self.wandb, dict):
            self.wandb = JaxEvalWandbConfig(**self.wandb)


def jax_eval_config_from_dict(data: dict[str, Any]) -> JaxEvalConfig:
    """Create JaxEvalConfig from a dictionary.

    Args:
        data: Dictionary with configuration values.

    Returns:
        JaxEvalConfig instance.
    """
    return JaxEvalConfig(
        checkpoint=data.get("checkpoint", "checkpoints_jax/best"),
        engine=JaxEvalEngineConfig(**data.get("engine", {})),
        benchmark=JaxEvalBenchmarkConfig(**data.get("benchmark", {})),
        compute=JaxEvalComputeConfig(**data.get("compute", {})),
        wandb=JaxEvalWandbConfig(**data.get("wandb", {})),
        verbose=data.get("verbose", False),
    )
