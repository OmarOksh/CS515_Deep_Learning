"""Training loop shared by all Part-1 tasks.

Regression tasks (``returns`` / ``rolling``) minimise mean-squared error on the
predicted return vector; the turning-point task minimises binary cross-entropy
on a single logit. The optimiser is Adam or AdamW per the assignment, and the
checkpoint with the lowest validation loss is saved.
"""
from __future__ import annotations

import copy
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from parameters import Config


def _make_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Instantiate Adam or AdamW from the training config."""
    params = dict(
        lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    if cfg.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **params)
    return torch.optim.Adam(model.parameters(), **params)


def _positive_weight(loader: DataLoader) -> float:
    """Estimate ``#negatives / #positives`` over a loader's labels.

    Buy events (Part d) are rare, so an unweighted BCE loss makes the model
    collapse to the majority "pass" class. Passing this ratio as ``pos_weight``
    to :class:`torch.nn.BCEWithLogitsLoss` rebalances the two classes.
    """
    pos, total = 0.0, 0.0
    for _, y in loader:
        pos += float(y.sum())
        total += y.numel()
    neg = total - pos
    return (neg / pos) if pos > 0 else 1.0


def _make_criterion(task: str, train_loader: DataLoader) -> nn.Module:
    """Return the loss appropriate to the task.

    Args:
        task: ``"returns"`` / ``"rolling"`` / ``"turning"``.
        train_loader: Used to estimate class balance for the turning task.
    """
    if task == "turning":
        pos_weight = torch.tensor(_positive_weight(train_loader))
        print(f"[train] BCE pos_weight = {pos_weight.item():.2f}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.MSELoss()


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    log_interval: int = 0,
) -> float:
    """Run one pass over ``loader``; train when ``optimizer`` is given.

    Args:
        model: The network.
        loader: Data loader for this split.
        criterion: Loss function.
        device: Compute device.
        optimizer: If provided, parameters are updated (training mode).
        log_interval: Print running loss every ``log_interval`` batches.

    Returns:
        The mean per-sample loss over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            if is_train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs = x.size(0)
            total_loss += loss.detach().item() * bs
            n += bs
            if is_train and log_interval and (batch_idx + 1) % log_interval == 0:
                print(f"    [{batch_idx + 1}/{len(loader)}] loss: {total_loss / n:.6f}")

    return total_loss / max(n, 1)


def run_training(
    model: nn.Module,
    cfg: Config,
    loaders: Tuple[DataLoader, DataLoader],
    device: torch.device,
) -> nn.Module:
    """Train ``model`` and keep the best-validation checkpoint.

    Args:
        model: Network to train (already moved to ``device``).
        cfg: Full run configuration.
        loaders: ``(train_loader, val_loader)``.
        device: Compute device.

    Returns:
        The model loaded with its best-validation weights.
    """
    train_loader, val_loader = loaders
    criterion = _make_criterion(cfg.data.task, train_loader)
    optimizer = _make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    os.makedirs(os.path.dirname(cfg.train.save_path) or ".", exist_ok=True)

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss = _run_epoch(model, train_loader, criterion, device,
                             optimizer, cfg.train.log_interval)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, cfg.train.save_path)
            marker = "  <- saved"
        print(f"Epoch {epoch:3d}/{cfg.train.epochs} | "
              f"train {tr_loss:.6f} | val {val_loss:.6f}{marker}")

    model.load_state_dict(best_weights)
    print(f"\nTraining complete. Best validation loss: {best_val:.6f}")
    return model
