"""Bidirectional Transformer model for chess position evaluation (JAX/Flax).

Uses HybridNorm architecture from "HybridNorm: Towards Stable and Efficient
Transformer Training via Hybrid Normalization" (https://arxiv.org/abs/2503.04598).

HybridNorm combines:
- QKV Normalization in attention: per-head normalization of Q, K, V vectors for stable information flow
- Post-Norm in FFN: FFN(Norm(x)) + Norm(x) where residual is the normalized input

HybridNorm* variant (enabled via hybridnorm_star=True):
The first transformer block receives special treatment to stabilize early training:
- First block attention: Pre-Norm + QKV-Norm (MHA_QKV(Norm(x)) + x)
- First block FFN: Pre-Norm (FFN(Norm(y)) + y, residual is non-normalized)
All subsequent blocks use standard HybridNorm.
"""

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

    # HybridNorm* variant: special treatment of first transformer block
    # When True, first block uses Pre-Norm + QKV-Norm in attention and Pre-Norm in FFN
    # See: https://arxiv.org/abs/2503.04598 Section 3 "Special Treatment of First Block"
    hybridnorm_star: bool = False

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


class QKVNormAttention(nn.Module):
    """Multi-head attention with QKV normalization (HybridNorm).

    Applies LayerNorm to Q, K, V vectors individually before computing attention.
    This stabilizes information flow between layers without dampening the residual path.
    """

    num_heads: int
    hidden_size: int
    dropout_rate: float = 0.0
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool = False) -> jax.Array:
        """Forward pass with QKV normalization.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).
            train: Whether in training mode.

        Returns:
            Output tensor, same shape as input.
        """
        batch_size, seq_len, _ = x.shape
        head_dim = self.hidden_size // self.num_heads

        # Project to Q, K, V
        q = nn.Dense(self.hidden_size, dtype=self.dtype, name="query")(x)
        k = nn.Dense(self.hidden_size, dtype=self.dtype, name="key")(x)
        v = nn.Dense(self.hidden_size, dtype=self.dtype, name="value")(x)

        # Reshape to (batch, seq_len, num_heads, head_dim) BEFORE normalization
        # This is critical for HybridNorm: normalize per-head, not across all heads
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, head_dim)

        # QKV Normalization: normalize each head independently (per-head norm)
        # LayerNorm in float32 for numerical stability, operates on last dim (head_dim)
        q = nn.LayerNorm(dtype=jnp.float32, name="q_norm")(q.astype(jnp.float32)).astype(self.dtype)
        k = nn.LayerNorm(dtype=jnp.float32, name="k_norm")(k.astype(jnp.float32)).astype(self.dtype)
        v = nn.LayerNorm(dtype=jnp.float32, name="v_norm")(v.astype(jnp.float32)).astype(self.dtype)

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = head_dim**-0.5
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(attn_weights)

        # Apply attention to values
        attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

        # Reshape back to (batch, seq_len, hidden_size)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # Output projection
        attn_out = nn.Dense(self.hidden_size, dtype=self.dtype, name="out_proj")(attn_out)
        attn_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(attn_out)

        return attn_out


class TransformerBlock(nn.Module):
    """Single transformer encoder block with HybridNorm architecture.

    HybridNorm combines:
    - QKV Normalization in attention: per-head LayerNorm on Q/K/V after reshape
    - Post-Norm in FFN: FFN(Norm(x)) + Norm(x), residual is the normalized input

    HybridNorm* (first block only, when is_first_block=True):
    - Pre-Norm + QKV Normalization in attention: Norm before MHA input, plus QKV-Norm
    - Pre-Norm in FFN: FFN(Norm(x)) + x, residual is the original (non-normalized) input

    This provides stable training (from QKV norm) while maintaining effective
    depth and strong performance (from the FFN normalization structure).
    """

    hidden_size: int
    num_heads: int
    ff_dim: int
    activation: str = "gelu"
    dropout_rate: float = 0.0
    dtype: Dtype = jnp.float32  # Compute dtype for mixed precision
    is_first_block: bool = False  # Enable HybridNorm* treatment for first block

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
        """Forward pass with HybridNorm architecture.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).
            train: Whether in training mode.

        Returns:
            Output tensor, same shape as input.
        """
        if self.is_first_block:
            # === HybridNorm* First Block ===
            # Attention: Pre-Norm + QKV-Norm
            # Y^0 = MHA_QKV(Norm(X^0)) + X^0
            attn_input = nn.LayerNorm(dtype=jnp.float32, name="attn_norm")(
                x.astype(jnp.float32)
            ).astype(self.dtype)
            attn_out = QKVNormAttention(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(attn_input, train=train)
            y = x + attn_out  # Residual is original x

            # FFN: Pre-Norm (residual is y, not normalized y)
            # X^1 = FFN(Norm(Y^0)) + Y^0
            ffn_input = nn.LayerNorm(dtype=jnp.float32, name="ffn_norm")(
                y.astype(jnp.float32)
            ).astype(self.dtype)
            ff_out = nn.Dense(self.ff_dim, dtype=self.dtype)(ffn_input)
            ff_out = self._get_activation()(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            ff_out = nn.Dense(self.hidden_size, dtype=self.dtype)(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            x = ff_out + y  # Residual is y (non-normalized)
        else:
            # === Standard HybridNorm (blocks l > 0) ===
            # Attention with QKV Normalization (no pre-norm)
            # Y^l = MHA_QKV(X^l) + X^l
            attn_out = QKVNormAttention(
                num_heads=self.num_heads,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
            )(x, train=train)
            x = x + attn_out

            # FFN with Post-Norm: FFN(Norm(x)) + Norm(x)
            # X^{l+1} = FFN(Norm(Y^l)) + Norm(Y^l)
            x = nn.LayerNorm(dtype=jnp.float32, name="ffn_norm")(
                x.astype(jnp.float32)
            ).astype(self.dtype)
            ff_out = nn.Dense(self.ff_dim, dtype=self.dtype)(x)
            ff_out = self._get_activation()(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            ff_out = nn.Dense(self.hidden_size, dtype=self.dtype)(ff_out)
            ff_out = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(ff_out)
            x = ff_out + x  # Residual is the normalized x

        return x


class BidirectionalTransformer(nn.Module):
    """Bidirectional (non-causal) transformer for chess position evaluation.

    This model processes chess positions represented as token sequences
    and outputs a win probability distribution. Uses full attention (no causal mask)
    since all position information is available simultaneously.

    Architecture (HybridNorm):
    - Token embedding
    - Absolute (learnable) positional embedding
    - N transformer encoder blocks with HybridNorm:
        - QKV Normalization in attention
        - Post-Norm in FFN
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
        # Each block uses HybridNorm: QKV-norm in attention + FFN(Norm(x))+Norm(x) structure
        # If hybridnorm_star is enabled, first block uses special treatment (Pre-Norm + QKV-Norm)
        for i in range(self.config.num_layers):
            is_first_block = (i == 0) and self.config.hybridnorm_star
            hidden = TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim or (4 * self.config.hidden_size),
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                dtype=compute_dtype,
                is_first_block=is_first_block,
                name=f"block_{i}",
            )(hidden, train=train)

        # Cast back to float32 for output heads (blocks end with LayerNorm in float32)
        hidden = hidden.astype(jnp.float32)
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
            hybridnorm_star=config.hybridnorm_star,
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
