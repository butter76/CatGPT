"""JAX-specific configuration schemas for CatGPT training.

Extends the core configuration with JAX-specific options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Re-export configs for convenient import
__all__ = ["KeelConfig", "ResidualGateConfig", "SmolgenConfig", "JaxModelConfig", "JaxExperimentConfig"]


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
    - 64-72: Underpromotion targets (3 pieces x 3 directions)
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

    # Optimistic policy auxiliary target (LC0 BT3/BT4 method)
    # A separate policy head trained on the same MCTS targets but with per-sample
    # weights that emphasize positions where the value target (st_q) significantly
    # exceeded the model's value prediction. This creates a policy biased toward
    # finding tactics and surprising wins.
    #
    # Weighting: w = sigmoid((z - strength) * 3) where z = (st_q - value_pred) / value_std
    # value_std is computed directly from the bestQ HL-Gauss distribution (no extra head).
    # Only positions where z > strength (~2 std devs better than expected) contribute.
    # The per-sample gating already heavily downweights most samples, so the loss
    # weight should be ~1.0 (same as vanilla policy).
    optimistic_policy_head: bool = False  # Enable optimistic policy head
    optimistic_policy_weight: float = 1.0  # Loss weight (per-sample gating handles regularization)
    optimistic_strength: float = 2.0  # Z-score threshold (higher = more selective)

    # Attention-based policy head (LC0-style)
    # Instead of a simple Dense(73) projection, uses Q·K^T attention for 64x64 main
    # move logits, with a separate projection scaled up for underpromotions.
    # See: https://lczero.org/blog/2024/02/transformer-progress/
    policy_attention_head: bool = True  # Use Q·K^T attention for 64x64 logits
    policy_qk_dim: int = 32  # Dimension for Q/K projections

    # WDL (Win/Draw/Loss) classification head: 3-class auxiliary target from game result
    # Shares the value MLP with 3 extra output logits (W, D, L).
    wdl_weight: float = 0.05  # Loss weight (noisy target, keep low)

    # Uncertainty head: per-move value variance prediction
    # Uses Q·K^T attention (same architecture as policy head) with softplus activation
    # to predict non-negative variance for each legal move. Trained on teacher-generated
    # per-child bestQ distribution variances from generate_move_values.py.
    uncertainty_head: bool = False  # Enable per-move variance prediction
    uncertainty_weight: float = 1.0  # Loss weight for MSE variance loss

    # HL-Gauss configuration for value head
    # Converts scalar win probability (0-1) to categorical distribution
    value_num_bins: int = 81  # Number of bins for value distribution
    value_sigma_ratio: float = 0.75  # sigma = ratio * bin_width (0.75 spreads to ~5 bins)


@dataclass
class ResidualGateConfig:
    """Configuration for learnable per-layer residual gates.

    Each sublayer (attention and FFN) in each transformer block gets a learnable
    scalar gate that scales its output before the residual connection:
        x = x + gate * sublayer(x)

    This provides several benefits:
    - ReZero-style (init=0): Easier gradient flow at init, each layer starts as identity
    - Identity-style (init=1): Standard residual behavior but with learnable scaling
    - Per-sublayer gates allow attention vs FFN to scale differently

    The model can learn to dynamically adjust the contribution of each sublayer,
    potentially improving training stability and convergence.

    See: https://arxiv.org/abs/2003.04887 (ReZero)
    """

    enabled: bool = False
    init_value: float = 1.0  # Initial gate value (0 for ReZero, 1 for identity start)
    per_dim: bool = False  # If True, use per-dimension gates instead of scalar


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
class KeelConfig:
    """Configuration for Keel normalization (Post-LN with Highway-style connection).

    Keel replaces the standard residual pathway with a Highway-style connection
    and uses Post-LN to normalize after the residual addition. This enables
    stable training at extreme depths (1000+ layers) while maintaining the
    expressivity advantages of Post-LN.

    Forward pass per sub-layer (l-th layer):
        x_{l+1} = LN(alpha * x_l + F_l(LN(x_l)))

    Where alpha = total_sub_layers = 2 * num_layers (attention + FFN per block).

    Implementation details from the paper:
    - First block: degrades to Pre-LN (no outer LN, no alpha) for stable initialization
    - When enabled, supersedes ResidualGateConfig (gates are not used)

    See: https://arxiv.org/abs/2601.19895
    """

    enabled: bool = False
    alpha: float | None = None  # Highway scaling factor. None = auto (2 * num_layers)


@dataclass
class JaxModelConfig:
    """Configuration for JAX model architecture."""

    name: str = "transformer"
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int | None = None  # Defaults to 4 * hidden_size if None
    attention_dim: int | None = None  # Expanded attention dimension (None = hidden_size)
    vocab_size: int = 26  # From tokenizer.VOCAB_SIZE
    seq_length: int = 64
    activation: str = "gelu"

    # Position embedding type: "absolute" (learnable per-position) or "magating" (mult/add gates)
    # MaGating uses learnable multiplicative and additive gates instead of additive positional embedding.
    position_embedding: str = "magating"

    # Output head configuration
    output_heads: JaxOutputHeadConfig = field(default_factory=JaxOutputHeadConfig)

    # Smolgen: dynamic attention bias generation
    smolgen: SmolgenConfig = field(default_factory=SmolgenConfig)

    # Learnable per-layer residual gates
    residual_gates: ResidualGateConfig = field(default_factory=ResidualGateConfig)

    # Keel: Post-LN with Highway-style connection
    keel: KeelConfig = field(default_factory=KeelConfig)

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.hidden_size

        # Expanded attention: QKV projects to attention_dim, allowing more heads
        # with the same head_dim. Output projects back to hidden_size.
        if self.attention_dim is None:
            self.attention_dim = self.hidden_size

        if self.attention_dim % self.num_heads != 0:
            msg = f"attention_dim ({self.attention_dim}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)

        # Convert dict to dataclass if needed (for YAML loading)
        if isinstance(self.output_heads, dict):
            self.output_heads = JaxOutputHeadConfig(**self.output_heads)
        if isinstance(self.smolgen, dict):
            self.smolgen = SmolgenConfig(**self.smolgen)
        if isinstance(self.residual_gates, dict):
            self.residual_gates = ResidualGateConfig(**self.residual_gates)
        if isinstance(self.keel, dict):
            self.keel = KeelConfig(**self.keel)


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
    # Nonstandard scaling: params matching these strings get constant scaling instead of shape-based
    # Default: embeddings and layernorm use constant scaling; dense matrices use 2/(dim0+dim1)
    splus_nonstandard_strings: list[str] = field(default_factory=lambda: ["embed", "layernorm"])
    splus_nonstandard_constant: float = 0.001


@dataclass
class JaxSchedulerConfig:
    """Configuration for learning rate scheduler.

    Supported schedules:
    - "deepseek": DeepSeek-V3 style multi-phase schedule
        warmup → stable (constant peak) → cosine decay → linear cooldown (min → 0)
        See: https://arxiv.org/abs/2412.19437 Section 4.2
    - "cosine": Standard warmup + cosine decay
    - "linear": Warmup + linear decay
    - "constant": No decay (constant learning rate)
    """

    name: str = "deepseek"  # "deepseek" | "cosine" | "linear" | "constant"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1  # min_lr = learning_rate * min_lr_ratio

    # DeepSeek schedule: fraction of post-warmup steps for each phase
    # The cosine decay phase gets the remainder: 1 - stable_fraction - cooldown_fraction
    stable_fraction: float = 0.68  # Fraction at constant peak LR after warmup
    cooldown_fraction: float = 0.07  # Fraction at constant min LR at end


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
class JaxMetricsConfig:
    """Configuration for training metrics collection.

    Controls which additional metrics are computed and logged during training.
    Some metrics (like per-head gradient norms) require extra backward passes
    and can be expensive, so they're computed at a lower frequency.
    """

    # Layer gradient norms: L2 norm of gradients per transformer block/embedding/head
    log_layer_grad_norms: bool = True

    # Head gradient norms: L2 norm of gradients contributed by each loss head
    # Requires separate backward pass per head - expensive but informative
    log_head_grad_norms: bool = True
    head_grad_norm_every_steps: int = 100  # Only compute every N steps to reduce overhead


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
    metrics: JaxMetricsConfig = field(default_factory=JaxMetricsConfig)
    distributed: JaxDistributedConfig = field(default_factory=JaxDistributedConfig)

    seed: int = 42

    def __post_init__(self) -> None:
        """Convert dicts to dataclasses if needed."""
        if isinstance(self.metrics, dict):
            self.metrics = JaxMetricsConfig(**self.metrics)


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
        metrics=JaxMetricsConfig(**data.get("metrics", {})),
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
class JaxFractionalMCTSConfig:
    """Configuration for Fractional MCTS search with iterative deepening."""

    c_puct: float = 1.75
    policy_coverage_threshold: float = 0.80
    min_total_evals: int = 400
    initial_budget: float = 1.0
    budget_multiplier: float = 1.2


@dataclass
class JaxEvalEngineConfig:
    """Configuration for evaluation engine."""

    type: str = "value"  # "value", "policy", "mcts", or "fractional_mcts"
    batch_size: int = 256
    num_workers: int = 1  # Number of parallel workers (each loads own engine)
    mcts: JaxMCTSConfig = field(default_factory=JaxMCTSConfig)
    fractional_mcts: JaxFractionalMCTSConfig = field(default_factory=JaxFractionalMCTSConfig)

    def __post_init__(self) -> None:
        """Convert dict to dataclass if needed."""
        if isinstance(self.mcts, dict):
            self.mcts = JaxMCTSConfig(**self.mcts)
        if isinstance(self.fractional_mcts, dict):
            self.fractional_mcts = JaxFractionalMCTSConfig(**self.fractional_mcts)


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
