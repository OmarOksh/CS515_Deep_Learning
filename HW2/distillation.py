"""
Knowledge Distillation on CIFAR-10 (Part B)
=============================================

Implements the four steps required by the homework:

1. Train a **SimpleCNN** (student) from scratch on CIFAR-10.
2. Train a **ResNet-18** (teacher) from scratch **with** and **without** label
   smoothing; compare results.
3. Use the best ResNet as a teacher to train SimpleCNN via standard
   **knowledge distillation** (Hinton et al., 2015).  Compare FLOPs of
   teacher vs. student.
4. Train **MobileNetV2** using ResNet as teacher with a *modified*
   distillation scheme: the teacher's output is converted so that only
   the **true class** retains the teacher's predicted probability while all
   other classes share the remaining probability equally.  This encodes
   per-example difficulty.  Compare FLOPs and accuracy of ResNet vs.
   MobileNet.

Usage::

    python distillation.py [--epochs 20] [--lr 1e-3] [--batch_size 128]
                           [--temperature 4.0] [--alpha 0.7]
                           [--label_smoothing 0.1] [--device cuda]

References:
    - Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a NN.
    - Müller, Kornblith & Hinton (2019). When Does Label Smoothing Help?
    - Yuan et al. (2020). Revisiting KD via Label Smoothing Regularization.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Re-use the model definitions from the models/ package
from models.CNN import SimpleCNN
from models.ResNet import BasicBlock, ResNet
from models.mobilenet import MobileNetV2


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class DistillConfig:
    """Configuration for knowledge-distillation experiments.

    Attributes:
        epochs:          Training epochs per experiment.
        lr:              Initial learning rate.
        batch_size:      Mini-batch size.
        device:          Compute device string.
        temperature:     Softmax temperature for distillation (T).
        alpha:           Weight for the distillation loss (1-alpha = CE weight).
        label_smoothing: Smoothing factor for label-smoothing experiments.
        num_classes:     Number of CIFAR-10 classes.
        data_dir:        Root directory for dataset caching.
        num_workers:     Data-loader worker processes.
    """
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 128
    device: str = "cpu"
    temperature: float = 4.0
    alpha: float = 0.7
    label_smoothing: float = 0.1
    num_classes: int = 10
    data_dir: str = "./data"
    num_workers: int = 2


def parse_args() -> DistillConfig:
    """Parse CLI arguments into a :class:`DistillConfig`.

    Returns:
        Populated configuration dataclass.
    """
    p = argparse.ArgumentParser(
        description="Knowledge Distillation on CIFAR-10 (Part B)"
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--temperature", type=float, default=4.0,
                   help="Softmax temperature T for distillation.")
    p.add_argument("--alpha", type=float, default=0.7,
                   help="Weight for distillation loss (1-alpha = hard CE).")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing factor (0 = no smoothing).")
    args = p.parse_args()

    return DistillConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        alpha=args.alpha,
        label_smoothing=args.label_smoothing,
    )


# ── Data ───────────────────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def get_loaders(cfg: DistillConfig) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train / test loaders with standard augmentation.

    Args:
        cfg: Experiment configuration.

    Returns:
        ``(train_loader, test_loader)`` tuple.
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True,
                                transform=train_tf)
    test_ds = datasets.CIFAR10(cfg.data_dir, train=False, download=True,
                               transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


# ── FLOPs counting ─────────────────────────────────────────────────────────────

def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 32, 32)) -> int:
    """Estimate multiply-accumulate operations (MACs) using ptflops.

    Falls back to a simple parameter-based estimate if ``ptflops`` is not
    installed.

    Args:
        model:      Network to profile.
        input_size: Input tensor shape (batch dim included).

    Returns:
        Estimated number of MACs (often reported as "FLOPs" in papers).
    """
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, input_size[1:], as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        return int(macs)
    except ImportError:
        # Rough estimate: 2 * parameters (each param used in one multiply
        # and one add per forward pass — very approximate).
        total_params = sum(p.numel() for p in model.parameters())
        return 2 * total_params


def format_flops(flops: int) -> str:
    """Format a FLOPs count into a human-readable string.

    Args:
        flops: Number of FLOPs (or MACs).

    Returns:
        Formatted string, e.g. ``'45.2 MFLOPs'``.
    """
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    return f"{flops / 1e3:.2f} KFLOPs"


# ── Training helpers ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute top-1 accuracy on a dataset.

    Args:
        model:  Network (already on *device*).
        loader: Data loader.
        device: Compute device.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n += imgs.size(0)
    return correct / n


def train_from_scratch(
    tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DistillConfig,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Train a model from scratch with optional label smoothing.

    Args:
        tag:              Experiment label for printing / saving.
        model:            Network (already on *device*).
        train_loader:     Training data loader.
        test_loader:      Test data loader.
        cfg:              Experiment configuration.
        device:           Compute device.
        label_smoothing:  Smoothing factor (0 = standard CE).

    Returns:
        The model loaded with the best checkpoint weights.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )

    best_acc = 0.0
    best_weights = None
    save_path = f"best_{tag}.pth"

    smoothing_str = f" (label_smoothing={label_smoothing})" if label_smoothing else ""
    print(f"\n{'='*60}")
    print(f"  Training from scratch: {tag}{smoothing_str}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)

        scheduler.step()
        val_acc = evaluate(model, test_loader, device)

        print(f"  Epoch {epoch:>2}/{cfg.epochs}  "
              f"train_loss={total_loss/n:.4f}  train_acc={correct/n:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"  >> Best test accuracy: {best_acc:.4f}\n")
    return model


# ── Distillation losses ───────────────────────────────────────────────────────

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Standard knowledge-distillation loss (Hinton et al., 2015).

    Combines a soft KL-divergence term (student vs. teacher softmax at
    temperature *T*) with a hard cross-entropy term against the ground-truth
    labels.

    .. math::

        L = \\alpha \\cdot T^2 \\cdot \\text{KL}(\\sigma(z_t/T) \\| \\sigma(z_s/T))
            + (1 - \\alpha) \\cdot \\text{CE}(z_s, y)

    Args:
        student_logits: Raw logits from the student, shape ``(B, C)``.
        teacher_logits: Raw logits from the teacher, shape ``(B, C)``.
        labels:         Ground-truth class indices, shape ``(B,)``.
        temperature:    Softmax temperature *T*.
        alpha:          Interpolation weight for the soft loss.

    Returns:
        Scalar loss tensor.
    """
    soft_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)

    # KL(teacher || student) — using log-probs for numerical stability
    kl = F.kl_div(
        soft_student, soft_teacher, log_target=True, reduction="batchmean"
    )
    soft_loss = kl * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def difficulty_based_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    num_classes: int = 10,
) -> torch.Tensor:
    """Modified distillation loss using teacher confidence as difficulty.

    Instead of using the full teacher distribution, we construct a *modified*
    soft target:

    - The **true class** receives the probability the teacher assigned to it.
    - All **other classes** share the remaining probability equally.

    This encodes per-example difficulty: when the teacher is confident on the
    true class the soft target is nearly one-hot; when the teacher is
    uncertain the soft target is more uniform.

    Args:
        student_logits: Raw logits from the student, shape ``(B, C)``.
        teacher_logits: Raw logits from the teacher, shape ``(B, C)``.
        labels:         Ground-truth class indices, shape ``(B,)``.
        temperature:    Softmax temperature *T*.
        alpha:          Interpolation weight for the soft loss.
        num_classes:    Number of classes *C*.

    Returns:
        Scalar loss tensor.
    """
    # Teacher probabilities at temperature T
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)  # (B, C)

    # Probability the teacher assigns to the true class
    batch_size = labels.size(0)
    true_class_prob = teacher_probs[
        torch.arange(batch_size, device=labels.device), labels
    ]  # (B,)

    # Build modified soft target: true class keeps its probability,
    # remaining probability is split equally among the other classes
    remaining = (1.0 - true_class_prob) / (num_classes - 1)  # (B,)

    soft_target = remaining.unsqueeze(1).expand(-1, num_classes).clone()  # (B, C)
    soft_target[torch.arange(batch_size, device=labels.device), labels] = true_class_prob

    # KL divergence between student (at temperature T) and modified target
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(
        log_student, soft_target, reduction="batchmean"
    ) * (temperature ** 2)

    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ── Distillation training loop ─────────────────────────────────────────────────

def train_with_distillation(
    tag: str,
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DistillConfig,
    device: torch.device,
    use_difficulty_based: bool = False,
) -> nn.Module:
    """Train a student model using a frozen teacher for knowledge distillation.

    Args:
        tag:                   Experiment label for printing / saving.
        student:               Student network (already on *device*).
        teacher:               Teacher network (already on *device*, frozen).
        train_loader:          Training data loader.
        test_loader:           Test data loader.
        cfg:                   Experiment configuration.
        device:                Compute device.
        use_difficulty_based:  If ``True``, use the modified difficulty-based
                               distillation loss instead of standard KD.

    Returns:
        Student model loaded with the best checkpoint weights.
    """
    teacher.eval()

    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )

    best_acc = 0.0
    best_weights = None
    save_path = f"best_{tag}.pth"

    loss_name = "difficulty-based KD" if use_difficulty_based else "standard KD"
    print(f"\n{'='*60}")
    print(f"  Distillation: {tag}  ({loss_name})")
    print(f"  T={cfg.temperature}  alpha={cfg.alpha}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            student_logits = student(imgs)

            if use_difficulty_based:
                loss = difficulty_based_distillation_loss(
                    student_logits, teacher_logits, labels,
                    cfg.temperature, cfg.alpha, cfg.num_classes,
                )
            else:
                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    cfg.temperature, cfg.alpha,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct += student_logits.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)

        scheduler.step()
        val_acc = evaluate(student, test_loader, device)

        print(f"  Epoch {epoch:>2}/{cfg.epochs}  "
              f"train_loss={total_loss/n:.4f}  train_acc={correct/n:.4f}  "
              f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, save_path)

    if best_weights is not None:
        student.load_state_dict(best_weights)
    print(f"  >> Best test accuracy: {best_acc:.4f}\n")
    return student


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all four knowledge-distillation experiments."""
    cfg = parse_args()

    device = torch.device(
        cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(cfg)

    # Keep track of results for the final summary table
    results: Dict[str, Dict[str, str]] = {}

    # ── Step 1: Train SimpleCNN from scratch ───────────────────────────
    print(f"\n{'#'*60}")
    print("  STEP 1  –  Train SimpleCNN from scratch")
    print(f"{'#'*60}")

    cnn_baseline = SimpleCNN(num_classes=cfg.num_classes).to(device)
    cnn_baseline = train_from_scratch(
        "simplecnn_baseline", cnn_baseline,
        train_loader, test_loader, cfg, device,
    )
    cnn_flops = count_flops(cnn_baseline)
    cnn_acc = evaluate(cnn_baseline, test_loader, device)
    results["SimpleCNN (baseline)"] = {
        "acc": f"{cnn_acc:.4f}",
        "flops": format_flops(cnn_flops),
    }

    # ── Step 2: Train ResNet from scratch ± label smoothing ────────────
    print(f"\n{'#'*60}")
    print("  STEP 2  –  Train ResNet-18 from scratch (± label smoothing)")
    print(f"{'#'*60}")

    # 2a: Without label smoothing
    resnet_no_ls = ResNet(BasicBlock, [2, 2, 2, 2],
                          num_classes=cfg.num_classes).to(device)
    resnet_no_ls = train_from_scratch(
        "resnet18_no_ls", resnet_no_ls,
        train_loader, test_loader, cfg, device,
        label_smoothing=0.0,
    )
    resnet_flops = count_flops(resnet_no_ls)
    resnet_no_ls_acc = evaluate(resnet_no_ls, test_loader, device)
    results["ResNet-18 (no LS)"] = {
        "acc": f"{resnet_no_ls_acc:.4f}",
        "flops": format_flops(resnet_flops),
    }

    # 2b: With label smoothing
    resnet_ls = ResNet(BasicBlock, [2, 2, 2, 2],
                       num_classes=cfg.num_classes).to(device)
    resnet_ls = train_from_scratch(
        "resnet18_ls", resnet_ls,
        train_loader, test_loader, cfg, device,
        label_smoothing=cfg.label_smoothing,
    )
    resnet_ls_acc = evaluate(resnet_ls, test_loader, device)
    results["ResNet-18 (LS=0.1)"] = {
        "acc": f"{resnet_ls_acc:.4f}",
        "flops": format_flops(resnet_flops),
    }

    # Pick the best teacher
    if resnet_ls_acc >= resnet_no_ls_acc:
        teacher = resnet_ls
        teacher_tag = "ResNet-18 (LS)"
    else:
        teacher = resnet_no_ls
        teacher_tag = "ResNet-18 (no LS)"
    teacher.eval()
    print(f"  Selected teacher: {teacher_tag}")

    # ── Step 3: Standard KD  →  SimpleCNN ─────────────────────────────
    print(f"\n{'#'*60}")
    print("  STEP 3  –  Knowledge Distillation: ResNet → SimpleCNN")
    print(f"{'#'*60}")

    cnn_kd = SimpleCNN(num_classes=cfg.num_classes).to(device)
    cnn_kd = train_with_distillation(
        "simplecnn_kd", cnn_kd, teacher,
        train_loader, test_loader, cfg, device,
        use_difficulty_based=False,
    )
    cnn_kd_acc = evaluate(cnn_kd, test_loader, device)
    results["SimpleCNN (KD)"] = {
        "acc": f"{cnn_kd_acc:.4f}",
        "flops": format_flops(cnn_flops),
    }

    # ── Step 4: Difficulty-based KD  →  MobileNetV2 ───────────────────
    print(f"\n{'#'*60}")
    print("  STEP 4  –  Difficulty-based KD: ResNet → MobileNetV2")
    print(f"{'#'*60}")

    mobilenet_kd = MobileNetV2(num_classes=cfg.num_classes).to(device)
    mobilenet_kd = train_with_distillation(
        "mobilenet_difficulty_kd", mobilenet_kd, teacher,
        train_loader, test_loader, cfg, device,
        use_difficulty_based=True,
    )
    mobilenet_flops = count_flops(mobilenet_kd)
    mobilenet_kd_acc = evaluate(mobilenet_kd, test_loader, device)
    results["MobileNetV2 (diff-KD)"] = {
        "acc": f"{mobilenet_kd_acc:.4f}",
        "flops": format_flops(mobilenet_flops),
    }

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<28} {'Test Acc':>10} {'FLOPs':>16}")
    print(f"  {'-'*54}")
    for name, info in results.items():
        print(f"  {name:<28} {info['acc']:>10} {info['flops']:>16}")
    print()


if __name__ == "__main__":
    main()
