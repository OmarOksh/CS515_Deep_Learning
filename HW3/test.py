"""Model evaluation with per-class accuracy reporting for HW3."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def run_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> float:
    """Evaluate model and print per-class accuracy.

    Args:
        model:       Network (already on *device*).
        loader:      Test data loader.
        device:      Compute device.
        num_classes: Number of classes.

    Returns:
        Overall accuracy.
    """
    model.eval()
    correct, n = 0, 0
    class_correct: List[int] = [0] * num_classes
    class_total: List[int] = [0] * num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t] += 1

    overall = correct / n
    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {overall:.4f}  ({correct}/{n})\n")
    for i in range(num_classes):
        acc = class_correct[i] / class_total[i] if class_total[i] else 0
        name = CIFAR10_CLASSES[i] if i < len(CIFAR10_CLASSES) else str(i)
        print(f"  {name:<12}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return overall
