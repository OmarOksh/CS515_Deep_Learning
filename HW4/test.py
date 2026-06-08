"""Evaluation entry point for the Part-1 tasks.

Loads the best checkpoint and reports task-appropriate metrics on the test
split: per-horizon MSE/MAE for the regression tasks, or accuracy / precision /
recall / F1 for the turning-point detector.
"""
from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parameters import Config
from utils import classification_metrics, regression_metrics


@torch.no_grad()
def _collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader and return stacked (preds, targets)."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x).cpu().numpy()
        preds.append(out)
        targets.append(y.numpy())
    if not preds:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(preds), np.concatenate(targets)


def run_test(
    model: nn.Module,
    cfg: Config,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ``model`` on the test loader and print a report.

    Args:
        model: Trained network.
        cfg: Full run configuration.
        loader: Test data loader.
        device: Compute device.

    Returns:
        Dictionary of computed metrics (also printed).
    """
    if os.path.exists(cfg.train.save_path):
        model.load_state_dict(torch.load(cfg.train.save_path, map_location=device))
    model.to(device)

    preds, targets = _collect_predictions(model, loader, device)
    if len(preds) == 0:
        print("[test] No test windows available.")
        return {}

    print("\n=== Test Results ===")
    if cfg.data.task == "turning":
        metrics = classification_metrics(preds, targets)
        print(f"  task           : turning-point (gamma={cfg.data.gamma})")
        print(f"  positive rate  : {metrics['positive_rate']:.4f}")
        print(f"  accuracy       : {metrics['accuracy']:.4f}")
        print(f"  precision      : {metrics['precision']:.4f}")
        print(f"  recall         : {metrics['recall']:.4f}")
        print(f"  f1             : {metrics['f1']:.4f}")
    else:
        metrics = regression_metrics(preds, targets)
        print(f"  task           : {cfg.data.task}")
        print(f"  overall MSE    : {metrics['mse']:.6e}")
        print(f"  overall MAE    : {metrics['mae']:.6e}")
        for d in range(targets.shape[1]):
            print(f"  MSE horizon d={d + 1}: {metrics[f'mse_h{d + 1}']:.6e}")
    return metrics
