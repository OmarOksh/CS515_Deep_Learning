"""Training loop with validation and best-model checkpointing."""

from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import Params


def get_transforms(params: Params, train: bool = True) -> transforms.Compose:
    """Build a ``torchvision.transforms.Compose`` pipeline.

    For CIFAR-10 training, random cropping and horizontal flipping are applied.
    MNIST uses only normalisation (no augmentation).

    Args:
        params: Global configuration.
        train:  If ``True``, include data-augmentation transforms.

    Returns:
        Composed transform pipeline.
    """
    mean, std = params.data.mean, params.data.std

    if params.data.dataset == "mnist":
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
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_loaders(params: Params) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        params: Global configuration.

    Returns:
        A ``(train_loader, val_loader)`` tuple.
    """
    train_tf = get_transforms(params, train=True)
    val_tf = get_transforms(params, train=False)

    if params.data.dataset == "mnist":
        train_ds = datasets.MNIST(params.data.data_dir, train=True,
                                  download=True, transform=train_tf)
        val_ds = datasets.MNIST(params.data.data_dir, train=False,
                                download=True, transform=val_tf)
    else:
        train_ds = datasets.CIFAR10(params.data.data_dir, train=True,
                                    download=True, transform=train_tf)
        val_ds = datasets.CIFAR10(params.data.data_dir, train=False,
                                  download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=params.train.batch_size,
                              shuffle=True, num_workers=params.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=params.train.batch_size,
                            shuffle=False, num_workers=params.data.num_workers)
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """Run a single training epoch.

    Args:
        model:        The neural network.
        loader:       Training data loader.
        optimizer:    Parameter optimiser.
        criterion:    Loss function.
        device:       Compute device.
        log_interval: Print progress every *log_interval* batches.

    Returns:
        ``(average_loss, accuracy)`` over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Args:
        model:     The neural network.
        loader:    Validation data loader.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        ``(average_loss, accuracy)`` over the validation set.
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


def run_training(
    model: nn.Module,
    params: Params,
    device: torch.device,
) -> None:
    """Full training loop with validation and best-model saving.

    Args:
        model:  The neural network (already on *device*).
        params: Global configuration.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.train.learning_rate,
        weight_decay=params.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5,
    )

    best_acc = 0.0
    best_weights = None

    for epoch in range(1, params.train.epochs + 1):
        print(f"\nEpoch {epoch}/{params.train.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, params.misc.log_interval,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params.misc.save_path)
            print(f"  Saved best model (val_acc={best_acc:.4f})")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
