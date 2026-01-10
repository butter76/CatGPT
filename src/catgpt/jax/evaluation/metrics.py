"""JAX evaluation metrics for model performance."""

import jax.numpy as jnp
from jax import Array


def accuracy(predictions: Array, targets: Array) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or probabilities).
        targets: Ground truth labels.

    Returns:
        Accuracy as a float between 0 and 1.
    """
    predicted_classes = jnp.argmax(predictions, axis=-1) if predictions.ndim > 1 else predictions

    correct = jnp.sum(predicted_classes == targets)
    total = targets.size
    return float(correct / total) if total > 0 else 0.0


def top_k_accuracy(
    predictions: Array,
    targets: Array,
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
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    if targets.ndim == 0:
        targets = targets[None]

    top_k_preds = jnp.argsort(predictions, axis=-1)[..., -k:]
    targets_expanded = jnp.expand_dims(targets, axis=-1)
    correct = jnp.sum(jnp.any(top_k_preds == targets_expanded, axis=-1))
    total = targets.size
    return float(correct / total) if total > 0 else 0.0
