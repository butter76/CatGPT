"""Pytest configuration and shared fixtures."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def sample_batch() -> dict[str, torch.Tensor]:
    """Create a sample batch for testing."""
    batch_size = 4
    seq_len = 16
    hidden_size = 32

    return {
        "input": torch.randn(batch_size, seq_len, hidden_size),
        "target": torch.randn(batch_size, seq_len, hidden_size),
        "mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
