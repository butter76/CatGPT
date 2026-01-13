"""Bidirectional Transformer model for chess position evaluation (JAX/Flax)."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from catgpt.jax.configs import JaxModelConfig, JaxOutputHeadConfig

# Type alias for dtype
Dtype = Any


@dataclass
class TransformerConfig:
    """Configuration for the bidirectional transformer."""

    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int | None = None  # Defaults to 4 * hidden_size
    vocab_size: int = 28  # From tokenizer.VOCAB_SIZE
    seq_length: int = 64
    activation: str = "gelu"
    dropout_rate: float = 0.0

    # Output head configuration
    output_heads: JaxOutputHeadConfig = field(default_factory=JaxOutputHeadConfig)

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            object.__setattr__(self, "ff_dim", 4 * self.hidden_size)

        if self.hidden_size % self.num_heads != 0:
            msg = f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)

        # Convert dict to JaxOutputHeadConfig if needed
        if isinstance(self.output_heads, dict):
            object.__setattr__(self, "output_heads", JaxOutputHeadConfig(**self.output_heads))


class TransformerBlock(nn.Module):
    """Single transformer encoder block with bidirectional attention.

    Uses Mix-LN (https://arxiv.org/abs/2412.13795) which combines Post-LN for
    early layers and Pre-LN for later layers. This ensures effective gradient
    flow throughout the network:
    - Post-LN in early layers: maintains larger gradients in deeper layers
    - Pre-LN in later layers: prevents gradient vanishing in early layers
    """

    hidden_size: int
    num_heads: int
    ff_dim: int
    layer_idx: int  # Current layer index (0-indexed)
    num_layers: int  # Total number of layers
    activation: str = "gelu"
    dropout_rate: float = 0.0
    dtype: Dtype = jnp.float32  # Compute dtype for mixed precision

    def _get_activation(self) -> callable:
        """Get activation function by name."""
        activations = {
            "gelu": nn.gelu,
            "relu": nn.relu,
            "silu": nn.silu,
            "tanh": jnp.tanh,
        }
        if self.activation not in activations:
            msg = f"Unknown activation: {self.activation}. Choose from {list(activations.keys())}"
            raise ValueError(msg)
        return activations[self.activation]

    def _use_post_ln(self) -> bool:
        """Determine if this layer should use Post-LN (early layers) or Pre-LN (later layers)."""
        # Mix-LN: Post-LN for first half, Pre-LN for second half
        return self.layer_idx < self.num_layers // 4

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Forward pass with Mix-LN architecture.

        Mix-LN applies Post-LN to early layers and Pre-LN to later layers,
        combining the benefits of both normalization strategies.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).
            train: Whether in training mode.

        Returns:
            Output tensor, same shape as input.
        """
        use_post_ln = self._use_post_ln()

        if use_post_ln:
            # Post-LN: normalize AFTER adding residual
            # Self-attention with residual
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                deterministic=not train,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, x)
            x = nn.LayerNorm(dtype=jnp.float32)((x + attn_out).astype(jnp.float32)).astype(self.dtype)

            # Feed-forward with residual
            ff_out = nn.Dense(self.ff_dim, dtype=self.dtype)(x)
            ff_out = self._get_activation()(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            ff_out = nn.Dense(self.hidden_size, dtype=self.dtype)(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            x = nn.LayerNorm(dtype=jnp.float32)((x + ff_out).astype(jnp.float32)).astype(self.dtype)
        else:
            # Pre-LN: normalize BEFORE sublayer
            # Self-attention with residual (NO causal mask = bidirectional)
            # LayerNorm in float32 for numerical stability, then cast back
            normed = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_size,
                deterministic=not train,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(normed, normed)
            x = x + attn_out

            # Feed-forward with residual
            normed = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
            ff_out = nn.Dense(self.ff_dim, dtype=self.dtype)(normed)
            ff_out = self._get_activation()(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            ff_out = nn.Dense(self.hidden_size, dtype=self.dtype)(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            x = x + ff_out

        return x


class BidirectionalTransformer(nn.Module):
    """Bidirectional (non-causal) transformer for chess position evaluation.

    This model processes chess positions represented as token sequences
    and outputs a win probability distribution. Uses full attention (no causal mask)
    since all position information is available simultaneously.

    Architecture:
    - Token embedding
    - Absolute (learnable) positional embedding
    - N transformer encoder blocks
    - Mean pooling over sequence
    - Linear projection to HL-Gauss categorical distribution

    Value Head (HL-Gauss):
    Instead of outputting a single scalar, the value head outputs logits over
    bins representing the win probability range [0, 1]. This enables training
    with cross-entropy loss which scales better than MSE regression.
    See: https://arxiv.org/abs/2403.03950

    Supports mixed precision training via compute_dtype parameter.
    Embeddings and final output are kept in float32 for stability.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        train: bool = False,
        compute_dtype: Dtype = jnp.float32,
    ) -> dict[str, jax.Array]:
        """Forward pass returning dict of output heads.

        Args:
            x: Input token indices, shape (batch, seq_len).
            train: Whether in training mode.
            compute_dtype: Dtype for intermediate computations (bfloat16 for mixed precision).

        Returns:
            Dictionary with output heads:
            - "self": Token reconstruction logits (batch, seq, vocab_size) if self_head enabled
            - "value_logit": HL-Gauss logits (batch, num_bins) for cross-entropy loss
            - "value_probs": Softmax probabilities (batch, num_bins) for visualization
            - "value": Expected win probability scalar (batch,) for metrics/inference
        """
        batch_size, seq_len = x.shape
        head_config = self.config.output_heads

        # Token embedding (keep in float32 for stability, then cast)
        token_emb = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=jnp.float32,  # Embeddings always in float32
        )(x)  # (batch, seq, hidden)

        # Learnable absolute positional embedding
        positions = jnp.arange(seq_len)
        pos_emb = nn.Embed(
            num_embeddings=self.config.seq_length,
            features=self.config.hidden_size,
            dtype=jnp.float32,  # Embeddings always in float32
        )(positions)  # (seq, hidden)

        # Combine embeddings and cast to compute dtype
        hidden = (token_emb + pos_emb).astype(compute_dtype)

        # Pass through transformer blocks (in compute_dtype)
        # Uses Mix-LN: Post-LN for first half of layers, Pre-LN for second half
        num_layers = self.config.num_layers
        for i in range(num_layers):
            hidden = TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim or (4 * self.config.hidden_size),
                layer_idx=i,
                num_layers=num_layers,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                dtype=compute_dtype,
                name=f"block_{i}",
            )(hidden, train=train)

        # Final norm (float32 for stability)
        hidden = nn.LayerNorm(dtype=jnp.float32)(hidden.astype(jnp.float32))
        # hidden: (batch, seq, hidden) - per-token representations

        # === Output heads ===
        outputs: dict[str, jax.Array] = {}

        # Self head: per-token reconstruction (BEFORE pooling)
        if head_config.self_head:
            self_logits = nn.Dense(
                self.config.vocab_size,
                dtype=jnp.float32,
                name="self_head",
            )(hidden)  # (batch, seq, vocab_size)
            outputs["self"] = self_logits

        # Value head: HL-Gauss categorical distribution from pooled representation
        if head_config.value_head:
            # Mean pooling over sequence dimension
            pooled = hidden.mean(axis=1)  # (batch, hidden)

            # Output logits for each bin in the HL-Gauss distribution
            num_bins = head_config.value_num_bins
            value_logits = nn.Dense(
                num_bins,
                dtype=jnp.float32,
                name="value_head",
            )(pooled)  # (batch, num_bins)

            # Softmax to get probability distribution
            value_probs = jax.nn.softmax(value_logits, axis=-1)  # (batch, num_bins)

            # Expected value: weighted sum of bin centers
            # Bins span [0, 1], so centers are at (i + 0.5) / num_bins for i in [0, num_bins)
            bin_centers = (jnp.arange(num_bins) + 0.5) / num_bins  # (num_bins,)
            expected_value = jnp.sum(value_probs * bin_centers, axis=-1)  # (batch,)

            outputs["value_logit"] = value_logits  # For cross-entropy loss
            outputs["value_probs"] = value_probs  # For visualization
            outputs["value"] = expected_value  # Scalar for metrics/inference

        return outputs

    @classmethod
    def from_model_config(cls, config: JaxModelConfig) -> "BidirectionalTransformer":
        """Create model from a JaxModelConfig instance.

        Args:
            config: JaxModelConfig from schema.

        Returns:
            BidirectionalTransformer instance.
        """
        transformer_config = TransformerConfig(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            vocab_size=config.vocab_size,
            seq_length=config.seq_length,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            output_heads=config.output_heads,
        )
        return cls(config=transformer_config)

    @classmethod
    def create_and_init(
        cls,
        config: JaxModelConfig | TransformerConfig,
        rng: jax.Array,
        compute_dtype: Dtype = jnp.float32,
    ) -> tuple["BidirectionalTransformer", dict]:
        """Create model and initialize parameters.

        Args:
            config: Model configuration.
            rng: PRNG key for initialization.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Tuple of (model, initialized parameters).
        """
        if isinstance(config, JaxModelConfig):
            model = cls.from_model_config(config)
            seq_length = config.seq_length
        else:
            model = cls(config=config)
            seq_length = config.seq_length

        # Initialize with dummy input (always use float32 for init)
        dummy_input = jnp.zeros((1, seq_length), dtype=jnp.int32)
        params = model.init(rng, dummy_input, train=False, compute_dtype=compute_dtype)

        return model, params
