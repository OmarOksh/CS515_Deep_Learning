"""
Transfer Learning on CIFAR-10 (Part A)
=======================================

Compares two strategies for adapting ImageNet-pretrained models to CIFAR-10:

**Option 1 — Resize inputs to 224×224 (ImageNet size)**
    CIFAR-10 images (32×32) are upscaled to 224×224 so the pretrained
    convolutional filters operate at the resolution they were trained on.
    Early layers are *frozen*; only the final fully-connected head is trained.

**Option 2 — Replace early conv, keep 32×32 inputs**
    The first convolution is replaced with a smaller kernel suited to 32×32
    images.  Because early features are now randomly initialised, the
    *entire* network is fine-tuned end-to-end.

Both options are evaluated for ResNet-18 and VGG-16.

Usage::

    python pretrained.py [--model resnet|vgg] [--epochs 10] [--lr 1e-4]
                         [--batch_size 64] [--device cuda]
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class TransferConfig:
    """Configuration for a transfer-learning experiment.

    Attributes:
        model_name:  Backbone architecture ('resnet' or 'vgg').
        epochs:      Number of fine-tuning epochs.
        lr:          Initial learning rate.
        batch_size:  Mini-batch size.
        device:      Compute device string ('cpu', 'cuda', 'mps').
        num_classes: Number of CIFAR-10 classes.
        data_dir:    Root directory for dataset caching.
        num_workers: Data-loader worker processes.
    """
    model_name: str = "resnet"
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 64
    device: str = "cpu"
    num_classes: int = 10
    data_dir: str = "./data"
    num_workers: int = 2


def parse_args() -> TransferConfig:
    """Parse command-line arguments into a :class:`TransferConfig`.

    Returns:
        Populated configuration dataclass.
    """
    parser = argparse.ArgumentParser(
        description="Transfer Learning on CIFAR-10 (Part A)"
    )
    parser.add_argument("--model", choices=["resnet", "vgg"], default="resnet",
                        help="Backbone: resnet (ResNet-18) or vgg (VGG-16).")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    return TransferConfig(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )


# ── Data helpers ───────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_loaders(
    cfg: TransferConfig,
    resize: int | None = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train / test loaders, optionally resizing to *resize*.

    Args:
        cfg:    Experiment configuration.
        resize: If given, resize images to ``(resize, resize)`` before
                augmentation.  Used by Option 1 (224×224).

    Returns:
        ``(train_loader, test_loader)`` tuple.
    """
    train_tfms: List[transforms.transforms.Transform] = []
    test_tfms: List[transforms.transforms.Transform] = []

    if resize is not None:
        train_tfms.append(transforms.Resize(resize))
        test_tfms.append(transforms.Resize(resize))

    train_tfms += [
        transforms.RandomCrop(resize or 32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    test_tfms += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True,  download=True,
                                transform=transforms.Compose(train_tfms))
    test_ds  = datasets.CIFAR10(cfg.data_dir, train=False, download=True,
                                transform=transforms.Compose(test_tfms))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


# ── Model builders ─────────────────────────────────────────────────────────────

def build_resnet18_option1(num_classes: int = 10) -> nn.Module:
    """ResNet-18 for Option 1: freeze backbone, train only the FC head.

    Images will be resized to 224×224 externally so the pretrained conv
    filters see the resolution they were trained on.

    Args:
        num_classes: Number of output logits.

    Returns:
        ResNet-18 with frozen backbone and a fresh classifier head.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze every parameter
    for param in model.parameters():
        param.requires_grad = False

    # Replace the FC head (its params are unfrozen by default)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet18_option2(num_classes: int = 10) -> nn.Module:
    """ResNet-18 for Option 2: replace first conv for 32×32, fine-tune all.

    The original ``conv1`` (7×7, stride 2) plus max-pool aggressively
    down-sample 224×224 inputs.  For 32×32 CIFAR images we swap it with a
    3×3, stride-1 convolution and remove the max-pool, then fine-tune the
    entire network.

    Args:
        num_classes: Number of output logits.

    Returns:
        ResNet-18 with modified stem and all parameters unfrozen.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace the 7×7 stem with a CIFAR-friendly 3×3 conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                            bias=False)
    model.maxpool = nn.Identity()  # remove aggressive 2× down-sampling

    # Fresh classification head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vgg16_option1(num_classes: int = 10) -> nn.Module:
    """VGG-16 for Option 1: freeze features, train only the classifier.

    Args:
        num_classes: Number of output logits.

    Returns:
        VGG-16 with frozen feature extractor and a fresh classifier.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the last FC layer in the classifier
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def build_vgg16_option2(num_classes: int = 10) -> nn.Module:
    """VGG-16 for Option 2: replace first conv for 32×32, fine-tune all.

    The first conv is swapped to a 3×3 kernel with stride 1 (no change in
    resolution), and the entire network is fine-tuned.  The classifier's
    input dimension changes because 32×32 images produce 1×1 feature maps
    after VGG's five max-pool stages, so the first Linear must expect 512.

    Args:
        num_classes: Number of output logits.

    Returns:
        VGG-16 with modified first conv, adjusted classifier, all unfrozen.
    """
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Replace first conv (originally 3→64, k=3, pad=1 — same, but we
    # reinitialise so early features adapt to low-res input)
    model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    # 32×32 → five max-pools → 1×1 → flatten = 512
    model.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


# ── Training / evaluation ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model:     Network (already on *device*).
        loader:    Training data loader.
        optimizer: Optimiser.
        criterion: Loss function.
        device:    Compute device.

    Returns:
        ``(avg_loss, accuracy)`` for the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a dataset.

    Args:
        model:     Network (already on *device*).
        loader:    Data loader (typically test set).
        criterion: Loss function.
        device:    Compute device.

    Returns:
        ``(avg_loss, accuracy)``.
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        n += imgs.size(0)

    return total_loss / n, correct / n


def run_experiment(
    tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TransferConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Fine-tune *model* and return its best test accuracy.

    Only parameters with ``requires_grad=True`` are passed to the optimiser,
    so frozen backbones remain untouched.

    Args:
        tag:          Human-readable experiment label (for printing).
        model:        Network to train (already on *device*).
        train_loader: Training data loader.
        test_loader:  Test data loader.
        cfg:          Experiment configuration.
        device:       Compute device.

    Returns:
        Dictionary with ``'best_val_acc'``, ``'final_train_acc'``, and
        ``'final_train_loss'`` keys.
    """
    criterion = nn.CrossEntropyLoss()

    # Only optimise unfrozen parameters
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                gamma=0.5)

    best_acc = 0.0
    best_weights = None
    save_path = f"best_transfer_{tag}.pth"

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  Trainable params: {num_trainable:,} / {num_total:,} "
          f"({100*num_trainable/num_total:.1f}%)")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:>2}/{cfg.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"  >> Best test accuracy: {best_acc:.4f}\n")

    return {
        "best_val_acc": best_acc,
        "final_train_acc": tr_acc,
        "final_train_loss": tr_loss,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run both transfer-learning options for the selected backbone."""
    cfg = parse_args()

    device = torch.device(
        cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    results: Dict[str, Dict[str, float]] = {}

    if cfg.model_name == "resnet":
        backbone_label = "ResNet-18"
        build_opt1 = build_resnet18_option1
        build_opt2 = build_resnet18_option2
    else:
        backbone_label = "VGG-16"
        build_opt1 = build_vgg16_option1
        build_opt2 = build_vgg16_option2

    # ── Option 1: resize to 224×224, freeze backbone ───────────────────
    print(f"\n{'#'*60}")
    print(f"  OPTION 1  –  Resize to 224×224, freeze early layers")
    print(f"  Backbone : {backbone_label}")
    print(f"{'#'*60}")

    train_loader_224, test_loader_224 = get_loaders(cfg, resize=224)
    model_opt1 = build_opt1(num_classes=cfg.num_classes).to(device)
    tag_opt1 = f"{cfg.model_name}_option1_resize224"
    results[tag_opt1] = run_experiment(
        tag_opt1, model_opt1, train_loader_224, test_loader_224, cfg, device,
    )

    # ── Option 2: keep 32×32, modify early conv, fine-tune all ─────────
    print(f"\n{'#'*60}")
    print(f"  OPTION 2  –  Modify early conv, fine-tune entire network")
    print(f"  Backbone : {backbone_label}")
    print(f"{'#'*60}")

    train_loader_32, test_loader_32 = get_loaders(cfg, resize=None)
    model_opt2 = build_opt2(num_classes=cfg.num_classes).to(device)
    tag_opt2 = f"{cfg.model_name}_option2_finetune32"
    results[tag_opt2] = run_experiment(
        tag_opt2, model_opt2, train_loader_32, test_loader_32, cfg, device,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY  –  {backbone_label}")
    print(f"{'='*60}")
    print(f"  {'Experiment':<40} {'Best Test Acc':>12}")
    print(f"  {'-'*52}")
    for tag, res in results.items():
        print(f"  {tag:<40} {res['best_val_acc']:>11.4f}")
    print()


if __name__ == "__main__":
    main()
