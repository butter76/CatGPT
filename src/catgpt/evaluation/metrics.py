"""Evaluation metrics for model performance."""

import torch


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or probabilities).
        targets: Ground truth labels.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    predicted_classes = predictions.argmax(dim=-1) if predictions.dim() > 1 else predictions

    correct = (predicted_classes == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5,
) -> float:
    """Compute top-k classification accuracy.

    Args:
        predictions: Model predictions (logits or probabilities).
        targets: Ground truth labels.
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float between 0 and 1.
    """
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 0:
        targets = targets.unsqueeze(0)

    _, top_k_preds = predictions.topk(k, dim=-1)
    targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
    correct = (top_k_preds == targets_expanded).any(dim=-1).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0
