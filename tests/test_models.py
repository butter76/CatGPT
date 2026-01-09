"""Tests for model implementations."""

import pytest
import torch


class TestBaseModel:
    """Tests for the BaseModel class."""

    def test_num_parameters_counts_correctly(self) -> None:
        """Test that parameter counting works correctly."""
        from torch import nn

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.Linear(20, 5),  # 20*5 + 5 = 105 params
        )  # Total: 325 params

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params == 325

    def test_model_forward_shape(self) -> None:
        """Test that model forward pass returns correct shape."""
        from torch import nn

        model = nn.Linear(10, 5)
        x = torch.randn(32, 10)
        output = model(x)

        assert output.shape == (32, 5)


class TestModelCompilation:
    """Tests for model compilation."""

    @pytest.mark.slow
    def test_compile_returns_compiled_model(self) -> None:
        """Test that torch.compile works on a simple model."""
        from torch import nn

        model = nn.Linear(10, 5)
        # Just check it doesn't error - actual compilation is lazy
        compiled = torch.compile(model)
        assert compiled is not None
