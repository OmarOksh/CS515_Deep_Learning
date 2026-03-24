"""Model evaluation on a held-out test set with per-class reporting."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import Params
from train import get_transforms


@torch.no_grad()
def run_test(
    model: nn.Module,
    params: Params,
    device: torch.device,
) -> None:
    """Load the best checkpoint and evaluate on the test set.

    Prints overall accuracy as well as per-class accuracy.

    Args:
        model:  The neural network (architecture must match the checkpoint).
        params: Global configuration (supplies dataset info, paths, etc.).
        device: Compute device.
    """
    tf = get_transforms(params, train=False)

    if params.data.dataset == "mnist":
        test_ds = datasets.MNIST(params.data.data_dir, train=False,
                                 download=True, transform=tf)
    else:
        test_ds = datasets.CIFAR10(params.data.data_dir, train=False,
                                   download=True, transform=tf)

    loader = DataLoader(test_ds, batch_size=params.train.batch_size,
                        shuffle=False, num_workers=params.data.num_workers)

    model.load_state_dict(
        torch.load(params.misc.save_path, map_location=device)
    )
    model.eval()

    correct, n = 0, 0
    num_classes = params.model.num_classes
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += (p == t).item()
            class_total[t] += 1

    print(f"\n=== Test Results ===")
    print(f"Overall accuracy: {correct / n:.4f}  ({correct}/{n})\n")
    for i in range(num_classes):
        acc = class_correct[i] / class_total[i] if class_total[i] else 0.0
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")
