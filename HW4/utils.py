"""Shared utilities: reproducibility, feature scaling and evaluation metrics.

Kept dependency-light on purpose (only NumPy / PyTorch) so the repository can be
graded without installing scikit-learn.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Args:
        seed: Integer seed shared across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Return a valid ``torch.device``, falling back to CPU when needed.

    Args:
        requested: One of ``"cpu"``, ``"cuda"``, ``"mps"`` or ``"auto"``.

    Returns:
        A concrete :class:`torch.device`.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@dataclass
class StandardScaler:
    """Per-feature z-score scaler fitted on the *training* features only.

    Storing the statistics explicitly (rather than re-fitting on each split)
    prevents information from the validation/test periods leaking into the
    training normalisation, which is essential for an honest time-series
    evaluation.
    """

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Compute the per-column mean and standard deviation.

        Args:
            x: Array of shape ``(num_rows, num_features)``.

        Returns:
            ``self`` so the call can be chained.
        """
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0.0] = 1.0  # guard against constant columns
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the stored standardisation to ``x``."""
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler.transform called before fit().")
        return (x - self.mean) / self.std


def regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute MSE/MAE overall and per forecast horizon.

    Args:
        preds: Predicted returns, shape ``(num_samples, D)``.
        targets: Ground-truth returns, shape ``(num_samples, D)``.

    Returns:
        Dictionary with the overall ``mse``/``mae`` plus ``mse_h{d}`` entries.
    """
    err = preds - targets
    metrics: Dict[str, float] = {
        "mse": float(np.mean(err ** 2)),
        "mae": float(np.mean(np.abs(err))),
    }
    for d in range(targets.shape[1]):
        metrics[f"mse_h{d + 1}"] = float(np.mean(err[:, d] ** 2))
    return metrics


def classification_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 for the binary buy-vs-pass task.

    Args:
        logits: Raw model outputs (pre-sigmoid), shape ``(num_samples,)``.
        labels: Ground-truth 0/1 labels, shape ``(num_samples,)``.

    Returns:
        Dictionary of scalar metrics.
    """
    preds = (logits > 0.0).astype(np.int64)  # sigmoid(x) > 0.5  <=>  x > 0
    labels = labels.astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": (tp + tn) / max(len(labels), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_rate": float(np.mean(labels)),
    }


def enforce_power_constraint(x: torch.Tensor) -> torch.Tensor:
    """Scale a coded signal so that ``E[||x||^2] <= 1`` (Part 2 channel rule).

    The expectation is approximated over the batch dimension, matching the
    average-power constraint stated in the assignment.

    Args:
        x: Coded symbols of shape ``(batch, num_symbols)``.

    Returns:
        The power-normalised tensor with the same shape.
    """
    power = x.pow(2).mean()
    return x / torch.sqrt(power + 1e-9)
