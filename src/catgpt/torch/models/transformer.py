"""Bidirectional Transformer model for chess position evaluation."""

import math
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from catgpt.torch.models.base import BaseModel


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

    def __post_init__(self) -> None:
        """Set defaults and validate."""
        if self.ff_dim is None:
            object.__setattr__(self, "ff_dim", 4 * self.hidden_size)

        if self.hidden_size % self.num_heads != 0:
            msg = f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with bidirectional attention."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        # Feed-forward network
        ff_dim = config.ff_dim or (4 * config.hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden_size, ff_dim),
            self._get_activation(config.activation),
            nn.Linear(ff_dim, config.hidden_size),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        if name not in activations:
            msg = f"Unknown activation: {name}. Choose from {list(activations.keys())}"
            raise ValueError(msg)
        return activations[name]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with pre-norm architecture.

        Args:
            x: Input tensor, shape (batch, seq_len, hidden_size).

        Returns:
            Output tensor, same shape as input.
        """
        # Self-attention with residual (NO causal mask = bidirectional)
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x


class BidirectionalTransformer(BaseModel):
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
    """

    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # Learnable absolute positional embedding
        self.position_embedding = nn.Embedding(config.seq_length, config.hidden_size)

        # Register position indices as buffer (not a parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(config.seq_length).unsqueeze(0),
            persistent=False,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_size)

        # Output head: mean pooling -> linear -> sigmoid for win probability
        self.output_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        # Standard initialization for transformer
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass computing win probability.

        Args:
            x: Input token indices, shape (batch, seq_len).

        Returns:
            Win probability, shape (batch, 1), values in [0, 1].
        """
        batch_size, seq_len = x.shape

        # Get embeddings
        token_emb = self.token_embedding(x)  # (batch, seq, hidden)
        pos_emb = self.position_embedding(self.position_ids[:, :seq_len])  # (1, seq, hidden)

        # Combine embeddings
        hidden = token_emb + pos_emb

        # Pass through transformer blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Final norm
        hidden = self.final_norm(hidden)

        # Mean pooling over sequence dimension
        pooled = hidden.mean(dim=1)  # (batch, hidden)

        # Output head with sigmoid for probability
        logits = self.output_head(pooled)  # (batch, 1)
        return torch.sigmoid(logits)

    def forward_logits(self, x: Tensor) -> Tensor:
        """Forward pass returning raw logits (before sigmoid).

        Useful for training with BCEWithLogitsLoss for numerical stability.

        Args:
            x: Input token indices, shape (batch, seq_len).

        Returns:
            Raw logits, shape (batch, 1).
        """
        batch_size, seq_len = x.shape

        # Get embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(self.position_ids[:, :seq_len])
        hidden = token_emb + pos_emb

        # Pass through transformer blocks
        for block in self.blocks:
            hidden = block(hidden)

        # Final norm and pooling
        hidden = self.final_norm(hidden)
        pooled = hidden.mean(dim=1)

        # Return raw logits
        return self.output_head(pooled)

    @classmethod
    def from_model_config(cls, config: "ModelConfig") -> Self:  # noqa: F821
        """Create model from a ModelConfig instance.

        Args:
            config: ModelConfig from schema.

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
        )
        return cls(transformer_config)
