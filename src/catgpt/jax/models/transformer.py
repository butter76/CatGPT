"""Bidirectional Transformer model for chess position evaluation (JAX/Flax)."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from catgpt.jax.configs import JaxModelConfig

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

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            object.__setattr__(self, "ff_dim", 4 * self.hidden_size)

        if self.hidden_size % self.num_heads != 0:
            msg = f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with bidirectional attention."""

    hidden_size: int
    num_heads: int
    ff_dim: int
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

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).
            train: Whether in training mode.

        Returns:
            Output tensor, same shape as input.
        """
        # Self-attention with residual (NO causal mask = bidirectional)
        normed = nn.LayerNorm(dtype=self.dtype)(x)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            deterministic=not train,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
        )(normed, normed)
        x = x + attn_out

        # Feed-forward with residual
        normed = nn.LayerNorm(dtype=self.dtype)(x)
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
    and outputs a win probability. Uses full attention (no causal mask)
    since all position information is available simultaneously.

    Architecture:
    - Token embedding
    - Absolute (learnable) positional embedding
    - N transformer encoder blocks
    - Mean pooling over sequence
    - Linear projection to win probability (sigmoid output)

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
        return_logits: bool = False,
        compute_dtype: Dtype = jnp.float32,
    ) -> jax.Array:
        """Forward pass computing win probability.

        Args:
            x: Input token indices, shape (batch, seq_len).
            train: Whether in training mode.
            return_logits: If True, return raw logits instead of probabilities.
            compute_dtype: Dtype for intermediate computations (bfloat16 for mixed precision).

        Returns:
            Win probability, shape (batch, 1), values in [0, 1].
            If return_logits=True, returns raw logits instead.
        """
        batch_size, seq_len = x.shape

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
        for i in range(self.config.num_layers):
            hidden = TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim or (4 * self.config.hidden_size),
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                dtype=compute_dtype,
                name=f"block_{i}",
            )(hidden, train=train)

        # Final norm (in compute_dtype)
        hidden = nn.LayerNorm(dtype=compute_dtype)(hidden)

        # Mean pooling over sequence dimension
        pooled = hidden.mean(axis=1)  # (batch, hidden)

        # Output head (cast back to float32 for output stability)
        pooled_f32 = pooled.astype(jnp.float32)
        logits = nn.Dense(1, dtype=jnp.float32)(pooled_f32)  # (batch, 1)

        if return_logits:
            return logits
        return jax.nn.sigmoid(logits)

    def forward_logits(
        self,
        x: jax.Array,
        *,
        train: bool = False,
        compute_dtype: Dtype = jnp.float32,
    ) -> jax.Array:
        """Forward pass returning raw logits (before sigmoid).

        Useful for training with binary cross-entropy for numerical stability.

        Args:
            x: Input token indices, shape (batch, seq_len).
            train: Whether in training mode.
            compute_dtype: Dtype for intermediate computations.

        Returns:
            Raw logits, shape (batch, 1).
        """
        return self.__call__(x, train=train, return_logits=True, compute_dtype=compute_dtype)

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
