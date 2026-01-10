"""Tests for PyTorch evaluation metrics."""

import torch

from catgpt.torch.evaluation import accuracy, top_k_accuracy


class TestAccuracy:
    """Tests for the accuracy metric."""

    def test_perfect_accuracy(self) -> None:
        """Test accuracy with all correct predictions."""
        predictions = torch.tensor([0, 1, 2, 3, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        assert accuracy(predictions, targets) == 1.0

    def test_zero_accuracy(self) -> None:
        """Test accuracy with all wrong predictions."""
        predictions = torch.tensor([1, 2, 3, 4, 0])
        targets = torch.tensor([0, 1, 2, 3, 4])
        assert accuracy(predictions, targets) == 0.0

    def test_partial_accuracy(self) -> None:
        """Test accuracy with some correct predictions."""
        predictions = torch.tensor([0, 1, 0, 0, 4])
        targets = torch.tensor([0, 1, 2, 3, 4])
        assert accuracy(predictions, targets) == 0.6

    def test_accuracy_with_logits(self) -> None:
        """Test accuracy from logits (multi-class predictions)."""
        # Logits where argmax gives [0, 1, 2]
        predictions = torch.tensor([
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ])
        targets = torch.tensor([0, 1, 2])
        assert accuracy(predictions, targets) == 1.0


class TestTopKAccuracy:
    """Tests for the top-k accuracy metric."""

    def test_top_1_same_as_accuracy(self) -> None:
        """Test that top-1 accuracy equals regular accuracy."""
        predictions = torch.tensor([
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ])
        targets = torch.tensor([0, 1, 2])
        assert top_k_accuracy(predictions, targets, k=1) == 1.0

    def test_top_k_with_second_best(self) -> None:
        """Test top-k includes second-best predictions."""
        predictions = torch.tensor([
            [10.0, 9.0, 1.0],  # Top-2: [0, 1]
            [1.0, 10.0, 9.0],  # Top-2: [1, 2]
        ])
        # Target 1 is in top-2 for first sample
        # Target 2 is in top-2 for second sample
        targets = torch.tensor([1, 2])
        assert top_k_accuracy(predictions, targets, k=2) == 1.0

    def test_top_k_misses_when_not_in_k(self) -> None:
        """Test top-k correctly misses when target not in top-k."""
        predictions = torch.tensor([
            [10.0, 9.0, 1.0],  # Top-2: [0, 1], target 2 not in top-2
        ])
        targets = torch.tensor([2])
        assert top_k_accuracy(predictions, targets, k=2) == 0.0
