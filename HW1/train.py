"""
Training module for MLP experiments.

Provides data loading, single-epoch training with optional L1 regularization,
validation, LR scheduler selection, and early stopping logic.
"""

import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(params, train: bool = True) -> transforms.Compose:
    """
    Build the image transform pipeline for a given dataset and split.

    Args:
        params: ExperimentParams instance.
        train: If True, include data augmentation (CIFAR-10 only).

    Returns:
        A torchvision Compose transform.
    """
    mean, std = params.mean, params.std

    if params.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(params) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        params: ExperimentParams instance.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_tf = get_transforms(params, train=True)
    val_tf = get_transforms(params, train=False)

    if params.dataset == "mnist":
        train_ds = datasets.MNIST(
            params.data_dir, train=True, download=True, transform=train_tf
        )
        val_ds = datasets.MNIST(
            params.data_dir, train=False, download=True, transform=val_tf
        )
    else:
        train_ds = datasets.CIFAR10(
            params.data_dir, train=True, download=True, transform=train_tf
        )
        val_ds = datasets.CIFAR10(
            params.data_dir, train=False, download=True, transform=val_tf
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
    )
    return train_loader, val_loader


def l1_regularization(model: nn.Module) -> torch.Tensor:
    """
    Compute the L1 norm of all trainable weight parameters.

    Args:
        model: The neural network model.

    Returns:
        Scalar tensor with the summed L1 norm.
    """
    l1_norm = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if "weight" in name:
            l1_norm = l1_norm + torch.norm(param, 1)
    return l1_norm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    l1_lambda: float = 0.0,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    Args:
        model: The neural network model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Compute device.
        log_interval: Print progress every N batches.
        l1_lambda: L1 regularization strength (0.0 = disabled).

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)

        # Add L1 penalty if enabled
        if l1_lambda > 0.0:
            loss = loss + l1_lambda * l1_regularization(model)

        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  [{batch_idx+1}/{len(loader)}] "
                f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}"
            )

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation set.

    Args:
        model: The neural network model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)

    return total_loss / n, correct / n


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create an LR scheduler based on the specified type.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: One of 'step', 'cosine', or 'plateau'.
        epochs: Total training epochs (used by CosineAnnealing).

    Returns:
        An LRScheduler instance.
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )
    else:  # "step"
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.5
        )


def run_training(model: nn.Module, params, device: torch.device) -> None:
    """
    Full training loop with LR scheduling and optional early stopping.

    Trains the model, tracks the best validation accuracy, saves the best
    checkpoint, and optionally applies early stopping if patience is set.

    Args:
        model: The neural network model.
        params: ExperimentParams instance.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.learning_rate,
        weight_decay=params.weight_decay,
    )
    scheduler = build_scheduler(optimizer, params.scheduler, params.epochs)

    best_acc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, params.epochs + 1):
        print(f"\nEpoch {epoch}/{params.epochs}")

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            params.log_interval,
            l1_lambda=params.l1_lambda,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler (ReduceLROnPlateau uses val_loss)
        if params.scheduler == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if params.early_stop_patience > 0 and patience_counter >= params.early_stop_patience:
            print(
                f"\n  Early stopping triggered at epoch {epoch} "
                f"(no improvement for {params.early_stop_patience} epochs)"
            )
            break

    model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
