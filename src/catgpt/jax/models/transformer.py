"""Bidirectional Transformer model for chess position evaluation (JAX/Flax)."""

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from catgpt.core.data.grain.coders import POLICY_SHAPE, POLICY_TO_DIM
from catgpt.jax.configs import JaxModelConfig, JaxOutputHeadConfig

# Type alias for dtype
Dtype = Any

# Policy output dimensions: (64 from_squares, 73 to_squares)
_POLICY_FROM_DIM, _POLICY_TO_DIM = POLICY_SHAPE


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


class QKVNormMultiHeadAttention(nn.Module):
    """Multi-head attention with per-head QKV normalization.

    Implements QKV-Norm as described in HybridNorm (https://arxiv.org/abs/2503.04598):
        attn_QKV(Q, K, V) = softmax(Norm(Q) * Norm(K)^T / sqrt(d_k)) * Norm(V)

    Normalization is applied per-head (not along the full hidden dimension).
    """

    num_heads: int
    qkv_features: int
    deterministic: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs_q: jax.Array, inputs_kv: jax.Array) -> jax.Array:
        """Apply multi-head attention with QKV normalization.

        Args:
            inputs_q: Query input, shape (batch, seq_len, features).
            inputs_kv: Key/Value input, shape (batch, seq_len, features).

        Returns:
            Attention output, shape (batch, seq_len, features).
        """
        batch_size, seq_len, _ = inputs_q.shape
        head_dim = self.qkv_features // self.num_heads

        if self.qkv_features % self.num_heads != 0:
            msg = f"qkv_features ({self.qkv_features}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)

        # Project to Q, K, V
        query = nn.Dense(self.qkv_features, dtype=self.dtype, name="query")(inputs_q)
        key = nn.Dense(self.qkv_features, dtype=self.dtype, name="key")(inputs_kv)
        value = nn.Dense(self.qkv_features, dtype=self.dtype, name="value")(inputs_kv)

        # Reshape to (batch, seq_len, num_heads, head_dim)
        query = query.reshape(batch_size, seq_len, self.num_heads, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_heads, head_dim)
        value = value.reshape(batch_size, seq_len, self.num_heads, head_dim)

        # Apply per-head LayerNorm along head_dim (last axis)
        # Normalize in float32 for stability, then cast back
        query = nn.LayerNorm(dtype=jnp.float32, name="query_norm")(
            query.astype(jnp.float32)
        ).astype(self.dtype)
        key = nn.LayerNorm(dtype=jnp.float32, name="key_norm")(
            key.astype(jnp.float32)
        ).astype(self.dtype)
        value = nn.LayerNorm(dtype=jnp.float32, name="value_norm")(
            value.astype(jnp.float32)
        ).astype(self.dtype)

        # Transpose to (batch, num_heads, seq_len, head_dim) for attention computation
        query = jnp.transpose(query, (0, 2, 1, 3))
        key = jnp.transpose(key, (0, 2, 1, 3))
        value = jnp.transpose(value, (0, 2, 1, 3))

        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_weights = jnp.matmul(query, jnp.transpose(key, (0, 1, 3, 2))) / jnp.sqrt(head_dim)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = jnp.matmul(attn_weights, value)

        # Transpose back to (batch, seq_len, num_heads, head_dim)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        # Reshape to (batch, seq_len, qkv_features)
        attn_output = attn_output.reshape(batch_size, seq_len, self.qkv_features)

        # Final output projection
        output = nn.Dense(self.qkv_features, dtype=self.dtype, name="out")(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Single transformer encoder block with bidirectional attention."""

    hidden_size: int
    num_heads: int
    ff_dim: int
    activation: str = "gelu"
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
        """Forward pass with Peri-LN architecture and QKV-Norm.

        Peri-LN applies layer normalization peripherally around each sublayer,
        i.e., both before (input) AND after (output) each module:
            y = x + Norm(Module(Norm(x)))

        This strikes an ideal balance in variance growth, avoiding both vanishing
        gradients (Post-LN) and massive activations (Pre-LN).
        See: https://arxiv.org/abs/2502.02732

        QKV-Norm is applied within the attention mechanism per-head for improved
        training stability. See: https://arxiv.org/abs/2503.04598

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).
            train: Whether in training mode.

        Returns:
            Output tensor, same shape as input.
        """
        # Self-attention with Peri-LN and QKV-Norm (NO causal mask = bidirectional)
        # Input norm: LayerNorm in float32 for numerical stability, then cast back
        normed = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
        attn_out = QKVNormMultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_size,
            deterministic=not train,
            dtype=self.dtype,
        )(normed, normed)
        # Output norm: normalize attention output before residual addition
        attn_out = nn.LayerNorm(dtype=jnp.float32)(attn_out.astype(jnp.float32)).astype(self.dtype)
        x = x + attn_out

        # Feed-forward with Peri-LN
        # Input norm
        normed = nn.LayerNorm(dtype=jnp.float32)(x.astype(jnp.float32)).astype(self.dtype)
        ff_out = nn.Dense(self.ff_dim, dtype=self.dtype)(normed)
        ff_out = self._get_activation()(ff_out)
        ff_out = nn.Dense(self.hidden_size, dtype=self.dtype)(ff_out)
        # Output norm: normalize MLP output before residual addition
        ff_out = nn.LayerNorm(dtype=jnp.float32)(ff_out.astype(jnp.float32)).astype(self.dtype)
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
    - Initial embedding LayerNorm (Peri-LN)
    - N transformer encoder blocks with Peri-LN
    - Final LayerNorm
    - Mean pooling over sequence
    - Linear projection to HL-Gauss categorical distribution

    Normalization (Peri-LN):
    Uses Peri-LN architecture which applies layer normalization peripherally
    around each sublayer (both input AND output): y = x + Norm(Module(Norm(x)))
    This provides more stable training than Pre-LN (prone to massive activations)
    or Post-LN (prone to vanishing gradients).
    See: https://arxiv.org/abs/2502.02732

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
            - "policy_logit": Move distribution logits (batch, 64*73) if policy_head enabled
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
        for i in range(self.config.num_layers):
            hidden = TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim or (4 * self.config.hidden_size),
                activation=self.config.activation,
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

        # Policy head: per-token projection to (64, 73) move distribution
        if head_config.policy_head:
            # Project each of the 64 token representations to 73 dimensions
            # hidden: (batch, 64, hidden_size) -> (batch, 64, 73)
            policy_logits = nn.Dense(
                _POLICY_TO_DIM,
                dtype=jnp.float32,
                name="policy_head",
            )(hidden)  # (batch, 64, 73)

            # Flatten for cross-entropy loss: (batch, 64*73) = (batch, 4672)
            policy_logits_flat = policy_logits.reshape(batch_size, -1)

            outputs["policy_logit"] = policy_logits_flat

        # Soft policy head: auxiliary head for softened policy target (KataGo method)
        # Uses a separate head to predict policy^(1/T) which forces the model to learn
        # relative rankings of lower-probability moves, not just the top 1-2.
        # See: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#auxiliary-soft-policy-target
        if head_config.soft_policy_head:
            # Same architecture as policy head but separate parameters
            soft_policy_logits = nn.Dense(
                _POLICY_TO_DIM,
                dtype=jnp.float32,
                name="soft_policy_head",
            )(hidden)  # (batch, 64, 73)

            # Flatten for cross-entropy loss: (batch, 64*73) = (batch, 4672)
            soft_policy_logits_flat = soft_policy_logits.reshape(batch_size, -1)

            outputs["soft_policy_logit"] = soft_policy_logits_flat

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
