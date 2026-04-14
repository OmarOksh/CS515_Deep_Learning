"""
Data Augmentation and Adversarial Robustness on CIFAR-10 (HW3)
===============================================================

Implements all five requirements:

1. Evaluate a fine-tuned ResNet-18 (from HW2) on **CIFAR-10-C** corruptions.
2. Re-fine-tune with **AugMix** and evaluate on clean + corrupted data.
3. **PGD attacks** (L∞ ε=4/255, L2 ε=0.25) on both models; **GradCAM**
   visualisation on clean vs adversarial; **t-SNE** of adversarial samples.
4. Use AugMix teacher for **knowledge distillation** → MobileNetV2 student.
5. **Adversarial transferability**: generate PGD samples from teacher,
   test on student.

Usage::

    python robustness.py --device cuda --epochs 10 --batch_size 128

References:
    [1] Hendrycks & Dietterich (2019). Benchmarking NN Robustness.
    [2] Hendrycks et al. (2020). AugMix.
    [3] Madry et al. (2018). Towards Deep Learning Models Resistant to
        Adversarial Attacks.
"""

from __future__ import annotations

import argparse
import copy
import os
import ssl
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from models.CNN import SimpleCNN
from models.ResNet import BasicBlock, ResNet
from models.mobilenet import MobileNetV2

ssl._create_default_https_context = ssl._create_unverified_context

# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Global configuration for all HW3 experiments.

    Attributes:
        epochs:          Fine-tuning epochs.
        lr:              Learning rate.
        batch_size:      Mini-batch size.
        device:          Compute device string.
        num_classes:     Number of CIFAR-10 classes.
        data_dir:        Dataset cache directory.
        num_workers:     DataLoader workers.
        temperature:     KD softmax temperature.
        alpha:           KD loss interpolation weight.
        label_smoothing: Label smoothing for teacher.
        kd_epochs:       Distillation training epochs.
    """
    epochs: int = 10
    lr: float = 1e-4
    batch_size: int = 128
    device: str = "cpu"
    num_classes: int = 10
    data_dir: str = "./data"
    num_workers: int = 2
    temperature: float = 4.0
    alpha: float = 0.7
    label_smoothing: float = 0.1
    kd_epochs: int = 20


def parse_args() -> Config:
    """Parse CLI arguments into a :class:`Config`.

    Returns:
        Populated configuration dataclass.
    """
    p = argparse.ArgumentParser(description="HW3: Robustness & Adversarial")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--kd_epochs", type=int, default=20)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    args = p.parse_args()
    return Config(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        kd_epochs=args.kd_epochs,
        temperature=args.temperature,
        alpha=args.alpha,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════════════

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_test_transform() -> transforms.Compose:
    """Standard CIFAR-10 test transform (no augmentation).

    Returns:
        Composed transform pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_train_transform(use_augmix: bool = False) -> transforms.Compose:
    """CIFAR-10 training transform with optional AugMix.

    Args:
        use_augmix: If ``True``, prepend AugMix to the pipeline.

    Returns:
        Composed transform pipeline.
    """
    tfms: List = []
    if use_augmix:
        tfms.append(transforms.AugMix())
    tfms += [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    return transforms.Compose(tfms)


def get_loaders(
    cfg: Config,
    use_augmix: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Build CIFAR-10 train / test loaders.

    Args:
        cfg:        Configuration.
        use_augmix: Whether to use AugMix in training transforms.

    Returns:
        ``(train_loader, test_loader)`` tuple.
    """
    train_tf = get_train_transform(use_augmix)
    test_tf = get_test_transform()

    train_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True,
                                transform=train_tf)
    test_ds = datasets.CIFAR10(cfg.data_dir, train=False, download=True,
                               transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
#  CIFAR-10-C loader
# ══════════════════════════════════════════════════════════════════════════════

CORRUPTION_TYPES = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def download_cifar10c(data_dir: str = "./data") -> str:
    """Download CIFAR-10-C dataset if not present.

    Args:
        data_dir: Root data directory.

    Returns:
        Path to the CIFAR-10-C directory.
    """
    c10c_dir = os.path.join(data_dir, "CIFAR-10-C")
    if os.path.isdir(c10c_dir) and os.path.exists(os.path.join(c10c_dir, "labels.npy")):
        return c10c_dir

    os.makedirs(c10c_dir, exist_ok=True)
    url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
    tar_path = os.path.join(data_dir, "CIFAR-10-C.tar")

    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10-C...")
        import urllib.request
        urllib.request.urlretrieve(url, tar_path)

    print("Extracting CIFAR-10-C...")
    import tarfile
    with tarfile.open(tar_path) as tf:
        tf.extractall(data_dir)

    return c10c_dir


def evaluate_cifar10c(
    model: nn.Module,
    device: torch.device,
    data_dir: str = "./data",
    severity: int = 5,
    batch_size: int = 128,
) -> Dict[str, float]:
    """Evaluate model on all CIFAR-10-C corruption types at a given severity.

    Args:
        model:      Network (already on *device*).
        device:     Compute device.
        data_dir:   Root data directory.
        severity:   Corruption severity level (1–5).
        batch_size: Evaluation batch size.

    Returns:
        Dictionary mapping corruption name to accuracy.
    """
    c10c_dir = download_cifar10c(data_dir)
    labels = np.load(os.path.join(c10c_dir, "labels.npy"))
    test_tf = get_test_transform()

    results: Dict[str, float] = {}
    model.eval()

    for ctype in CORRUPTION_TYPES:
        fpath = os.path.join(c10c_dir, f"{ctype}.npy")
        if not os.path.exists(fpath):
            print(f"  [skip] {ctype}.npy not found")
            continue

        images = np.load(fpath)
        # Each corruption has 5 severities × 10000 images
        start = (severity - 1) * 10000
        end = severity * 10000
        imgs_sev = images[start:end]
        labs_sev = labels[start:end]

        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(imgs_sev), batch_size):
                batch_imgs = imgs_sev[i:i + batch_size]
                batch_labs = labs_sev[i:i + batch_size]

                tensors = torch.stack([
                    test_tf(Image.fromarray(img)) for img in batch_imgs
                ]).to(device)
                targets = torch.tensor(batch_labs, dtype=torch.long, device=device)

                preds = model(tensors).argmax(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

        acc = correct / total
        results[ctype] = acc
        print(f"  {ctype:<25} {acc:.4f}")

    avg = np.mean(list(results.values()))
    results["AVERAGE"] = avg
    print(f"  {'AVERAGE':<25} {avg:.4f}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder (Option 2 style from HW2)
# ══════════════════════════════════════════════════════════════════════════════

def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """Build a pretrained ResNet-18 adapted for CIFAR-10 (Option 2).

    Replaces the 7×7 stem with a 3×3 conv and removes the max-pool, then
    replaces the final FC layer.

    Args:
        num_classes: Number of output logits.

    Returns:
        ResNet-18 with modified stem, all parameters unfrozen.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(
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
        Accuracy in [0, 1].
    """
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n += imgs.size(0)
    return correct / n


def fine_tune(
    tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Fine-tune a model on CIFAR-10.

    Args:
        tag:              Experiment name for printing/saving.
        model:            Network (already on *device*).
        train_loader:     Training data loader.
        test_loader:      Test data loader.
        cfg:              Configuration.
        device:           Compute device.
        label_smoothing:  Label smoothing factor.

    Returns:
        Model loaded with best checkpoint weights.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_acc = 0.0
    best_weights = None
    save_path = f"best_{tag}.pth"

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {tag}")
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
        val_acc = evaluate_model(model, test_loader, device)
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


# ══════════════════════════════════════════════════════════════════════════════
#  PGD Attack
# ══════════════════════════════════════════════════════════════════════════════

def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int = 20,
    norm: str = "Linf",
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) adversarial attack.

    Args:
        model:  Network (in eval mode).
        images: Clean input batch, shape ``(B, C, H, W)``.
        labels: Ground-truth labels, shape ``(B,)``.
        eps:    Perturbation budget.
        alpha:  Step size per iteration.
        steps:  Number of PGD iterations.
        norm:   ``'Linf'`` or ``'L2'``.

    Returns:
        Adversarial images (same shape as *images*).
    """
    adv = images.clone().detach()
    adv += torch.empty_like(adv).uniform_(-eps, eps)
    adv = torch.clamp(adv, 0, 1)  # note: we work in [0,1] space before normalisation

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]

        with torch.no_grad():
            if norm == "Linf":
                adv = adv + alpha * grad.sign()
                delta = torch.clamp(adv - images, min=-eps, max=eps)
                adv = torch.clamp(images + delta, 0, 1)
            elif norm == "L2":
                grad_norm = grad.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                grad_norm = torch.clamp(grad_norm, min=1e-8)
                adv = adv + alpha * grad / grad_norm
                delta = adv - images
                delta_norm = delta.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
                factor = torch.min(torch.ones_like(delta_norm), eps / (delta_norm + 1e-8))
                adv = torch.clamp(images + delta * factor, 0, 1)

    return adv.detach()


def evaluate_pgd(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    eps: float,
    norm: str = "Linf",
    steps: int = 20,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Evaluate clean and PGD-adversarial accuracy.

    The attack is applied in **normalised** space so the model receives
    normalised inputs directly.  Epsilon budgets are specified in [0, 1]
    pixel space and converted internally.

    Args:
        model:       Network (already on *device*).
        test_loader: Test data loader (images already normalised).
        device:      Compute device.
        eps:         Perturbation budget in [0,1] pixel space.
        norm:        ``'Linf'`` or ``'L2'``.
        steps:       PGD iterations.
        max_batches: If > 0, evaluate only this many batches.

    Returns:
        ``(clean_accuracy, adversarial_accuracy)``.
    """
    model.eval()

    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)

    # Convert eps from pixel [0,1] to per-channel normalised space
    if norm == "Linf":
        eps_norm = (eps / std).min().item()  # conservative
        alpha_norm = eps_norm / 10
    else:
        eps_norm = eps / std.mean().item()
        alpha_norm = eps_norm / 10

    clean_correct, adv_correct, total = 0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(test_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        # Clean accuracy
        with torch.no_grad():
            clean_correct += model(imgs).argmax(1).eq(labels).sum().item()

        # PGD attack in normalised space
        adv_imgs = pgd_attack(model, imgs, labels, eps_norm, alpha_norm, steps, norm)

        with torch.no_grad():
            adv_correct += model(adv_imgs).argmax(1).eq(labels).sum().item()

        total += labels.size(0)

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    return clean_acc, adv_acc


# ══════════════════════════════════════════════════════════════════════════════
#  GradCAM
# ══════════════════════════════════════════════════════════════════════════════

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Computes a heatmap highlighting regions most responsible for a given
    class prediction.

    Args:
        model:        Neural network.
        target_layer: Convolutional layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd = target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o.detach()))
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def __call__(
        self, x: torch.Tensor, class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute GradCAM heatmap.

        Args:
            x:         Input tensor, shape ``(1, C, H, W)``.
            class_idx: Target class; if ``None``, uses the predicted class.

        Returns:
            ``(heatmap, predicted_class)`` where heatmap is in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        h, w = x.shape[2], x.shape[3]
        heatmap = np.array(
            Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
        ) / 255.0
        return heatmap, class_idx

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks."""
        self._fwd.remove()
        self._bwd.remove()


def overlay_heatmap(
    img_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a GradCAM heatmap on an image.

    Args:
        img_np:  RGB image in [0, 1], shape ``(H, W, 3)``.
        heatmap: Heatmap in [0, 1], shape ``(H, W)``.
        alpha:   Blending factor.

    Returns:
        Blended image as uint8 array.
    """
    rgb = cm.get_cmap("jet")(heatmap)[:, :, :3]
    return (np.clip((1 - alpha) * img_np + alpha * rgb, 0, 1) * 255).astype(np.uint8)


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CIFAR-10 tensor to a numpy image in [0, 1].

    Args:
        tensor: Shape ``(1, 3, H, W)`` or ``(3, H, W)``.

    Returns:
        Numpy array of shape ``(H, W, 3)`` in [0, 1].
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    t = tensor.clone().cpu()
    t = t * torch.tensor(CIFAR10_STD).view(3, 1, 1) + torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)


def visualise_gradcam_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_tag: str,
    eps: float = 4 / 255,
    target_layer_name: str = "layer4",
) -> None:
    """Find misclassified adversarial samples and show GradCAM comparison.

    Args:
        model:             Network.
        test_loader:       Test data loader.
        device:            Compute device.
        model_tag:         Label for filenames.
        eps:               L∞ perturbation budget.
        target_layer_name: Name of the layer to hook for GradCAM.
    """
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    eps_norm = (eps / std).min().item()
    alpha_norm = eps_norm / 10

    # Find the target layer
    if hasattr(model, "layer4"):
        target_layer = model.layer4[-1]
    else:
        # For torchvision ResNet
        target_layer = getattr(model, target_layer_name)[-1]

    found = 0
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    titles_row = ["Clean Image", "GradCAM (clean)", "Overlay (clean)",
                   "Adversarial Image", "GradCAM (adv)", "Overlay (adv)"]

    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        for i in range(imgs.size(0)):
            x_clean = imgs[i:i + 1]
            y_true = labels[i].item()

            # Check clean is correct
            with torch.no_grad():
                clean_pred = model(x_clean).argmax(1).item()
            if clean_pred != y_true:
                continue

            # Generate adversarial
            x_adv = pgd_attack(model, x_clean, labels[i:i + 1], eps_norm, alpha_norm, 20, "Linf")

            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1).item()
            if adv_pred == y_true:
                continue  # attack failed, try next

            # GradCAM on clean
            gc = GradCAM(model, target_layer)
            hm_clean, _ = gc(x_clean, class_idx=y_true)
            gc.remove_hooks()

            # GradCAM on adversarial
            gc2 = GradCAM(model, target_layer)
            hm_adv, _ = gc2(x_adv, class_idx=adv_pred)
            gc2.remove_hooks()

            img_clean_np = denormalise(x_clean)
            img_adv_np = denormalise(x_adv)

            row = found
            axes[row, 0].imshow(img_clean_np)
            axes[row, 0].set_title(f"Clean: {CIFAR10_CLASSES[y_true]}\nPred: {CIFAR10_CLASSES[clean_pred]}", fontsize=9)
            axes[row, 1].imshow(hm_clean, cmap="jet")
            axes[row, 1].set_title("GradCAM (clean)", fontsize=9)
            axes[row, 2].imshow(overlay_heatmap(img_clean_np, hm_clean))
            axes[row, 2].set_title("Overlay (clean)", fontsize=9)

            if row == 0:
                # Show adversarial in a second figure row — we reuse axes
                pass

            found += 1
            if found >= 2:
                break

        if found >= 2:
            break

    # Now make a proper 2-row × 6-col figure for both samples
    # Redo with proper layout
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 6, figsize=(20, 7))
    fig2.suptitle(f"GradCAM: Clean vs Adversarial ({model_tag}, PGD20 L∞ ε=4/255)",
                  fontsize=13, fontweight="bold")

    found = 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        for i in range(imgs.size(0)):
            x_clean = imgs[i:i + 1]
            y_true = labels[i].item()

            with torch.no_grad():
                clean_pred = model(x_clean).argmax(1).item()
            if clean_pred != y_true:
                continue

            x_adv = pgd_attack(model, x_clean, labels[i:i + 1], eps_norm, alpha_norm, 20, "Linf")
            with torch.no_grad():
                adv_pred = model(x_adv).argmax(1).item()
            if adv_pred == y_true:
                continue

            gc = GradCAM(model, target_layer)
            hm_clean, _ = gc(x_clean, class_idx=y_true)
            gc.remove_hooks()

            gc2 = GradCAM(model, target_layer)
            hm_adv, _ = gc2(x_adv, class_idx=adv_pred)
            gc2.remove_hooks()

            img_c = denormalise(x_clean)
            img_a = denormalise(x_adv)

            r = found
            axes2[r, 0].imshow(img_c); axes2[r, 0].set_title(f"Clean\nTrue: {CIFAR10_CLASSES[y_true]}", fontsize=8)
            axes2[r, 1].imshow(hm_clean, cmap="jet"); axes2[r, 1].set_title("GradCAM\n(clean)", fontsize=8)
            axes2[r, 2].imshow(overlay_heatmap(img_c, hm_clean)); axes2[r, 2].set_title("Overlay\n(clean)", fontsize=8)
            axes2[r, 3].imshow(img_a); axes2[r, 3].set_title(f"Adversarial\nPred: {CIFAR10_CLASSES[adv_pred]}", fontsize=8)
            axes2[r, 4].imshow(hm_adv, cmap="jet"); axes2[r, 4].set_title("GradCAM\n(adversarial)", fontsize=8)
            axes2[r, 5].imshow(overlay_heatmap(img_a, hm_adv)); axes2[r, 5].set_title("Overlay\n(adversarial)", fontsize=8)

            for ax in axes2[r]:
                ax.axis("off")

            found += 1
            if found >= 2:
                break
        if found >= 2:
            break

    plt.tight_layout()
    plt.savefig(f"gradcam_{model_tag}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved gradcam_{model_tag}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  t-SNE visualisation (clean + adversarial)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Extract penultimate-layer features from a torchvision ResNet-18.

    Hooks into the avgpool layer and returns the flattened output.

    Args:
        model:  ResNet-18 (torchvision-style).
        images: Batch of images, shape ``(N, 3, 32, 32)``.
        device: Compute device.

    Returns:
        Feature array of shape ``(N, 512)``.
    """
    features_list: List[torch.Tensor] = []

    def hook_fn(module, inp, out):
        features_list.append(out.flatten(1).detach().cpu())

    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        model(images.to(device))
    handle.remove()
    return features_list[0].numpy()


def visualise_tsne_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_tag: str,
    n_samples: int = 1000,
    eps: float = 4 / 255,
) -> None:
    """t-SNE plot showing clean and PGD-adversarial samples.

    Args:
        model:      Network.
        test_loader: Test data loader.
        device:     Compute device.
        model_tag:  Label for filenames.
        n_samples:  Number of clean samples to use.
        eps:        L∞ perturbation budget.
    """
    model.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    eps_norm = (eps / std).min().item()
    alpha_norm = eps_norm / 10

    all_imgs, all_labels = [], []
    for imgs, labels in test_loader:
        all_imgs.append(imgs)
        all_labels.append(labels)
        if sum(x.size(0) for x in all_imgs) >= n_samples:
            break

    clean = torch.cat(all_imgs)[:n_samples].to(device)
    labels = torch.cat(all_labels)[:n_samples].to(device)

    # Generate adversarial
    adv_list = []
    bs = 128
    for i in range(0, n_samples, bs):
        batch = clean[i:i + bs]
        batch_labels = labels[i:i + bs]
        adv_list.append(pgd_attack(model, batch, batch_labels, eps_norm, alpha_norm, 20, "Linf"))
    adversarial = torch.cat(adv_list)

    # Extract features
    feat_clean = extract_features(model, clean, device)
    feat_adv = extract_features(model, adversarial, device)

    # t-SNE
    combined = np.concatenate([feat_clean, feat_adv], axis=0)
    labels_np = labels.cpu().numpy()
    labels_combined = np.concatenate([labels_np, labels_np])
    is_adv = np.array([0] * n_samples + [1] * n_samples)

    print(f"  Running t-SNE on {combined.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(combined)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left: coloured by class, shape by clean/adv
    for c in range(10):
        mask_clean = (labels_combined == c) & (is_adv == 0)
        mask_adv = (labels_combined == c) & (is_adv == 1)
        ax1.scatter(embedded[mask_clean, 0], embedded[mask_clean, 1],
                    s=10, alpha=0.6, label=f"{CIFAR10_CLASSES[c]}")
        ax1.scatter(embedded[mask_adv, 0], embedded[mask_adv, 1],
                    s=10, alpha=0.3, marker="x")
    ax1.set_title("t-SNE by Class (• clean, × adversarial)", fontsize=11)
    ax1.legend(fontsize=7, markerscale=2, loc="best")

    # Right: coloured by clean vs adversarial
    ax2.scatter(embedded[:n_samples, 0], embedded[:n_samples, 1],
                s=10, alpha=0.5, c="steelblue", label="Clean")
    ax2.scatter(embedded[n_samples:, 0], embedded[n_samples:, 1],
                s=10, alpha=0.5, c="tomato", label="Adversarial")
    ax2.set_title("t-SNE: Clean vs Adversarial", fontsize=11)
    ax2.legend(fontsize=9)

    fig.suptitle(f"t-SNE of Features ({model_tag}, PGD20 L∞ ε=4/255)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"tsne_adv_{model_tag}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved tsne_adv_{model_tag}.png")


# ══════════════════════════════════════════════════════════════════════════════
#  Knowledge Distillation (difficulty-based, reused from HW2)
# ══════════════════════════════════════════════════════════════════════════════

def difficulty_based_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    num_classes: int = 10,
) -> torch.Tensor:
    """Modified distillation loss using teacher confidence as difficulty.

    The true class retains the teacher's probability; remaining probability
    is split equally among other classes.

    Args:
        student_logits: Student logits ``(B, C)``.
        teacher_logits: Teacher logits ``(B, C)``.
        labels:         Ground-truth labels ``(B,)``.
        temperature:    Softmax temperature.
        alpha:          Soft loss weight.
        num_classes:    Number of classes.

    Returns:
        Scalar loss.
    """
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    batch_size = labels.size(0)
    true_prob = teacher_probs[torch.arange(batch_size, device=labels.device), labels]

    remaining = (1.0 - true_prob) / (num_classes - 1)
    soft_target = remaining.unsqueeze(1).expand(-1, num_classes).clone()
    soft_target[torch.arange(batch_size, device=labels.device), labels] = true_prob

    log_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(log_student, soft_target, reduction="batchmean") * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


def train_with_distillation(
    tag: str,
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    """Train student via difficulty-based knowledge distillation.

    Args:
        tag:          Experiment name.
        student:      Student network.
        teacher:      Teacher network (frozen).
        train_loader: Training data loader.
        test_loader:  Test data loader.
        cfg:          Configuration.
        device:       Compute device.

    Returns:
        Student with best checkpoint loaded.
    """
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.kd_epochs)

    best_acc = 0.0
    best_weights = None
    save_path = f"best_{tag}.pth"

    print(f"\n{'='*60}")
    print(f"  Distillation: {tag}")
    print(f"  T={cfg.temperature}  alpha={cfg.alpha}")
    print(f"{'='*60}")

    for epoch in range(1, cfg.kd_epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            student_logits = student(imgs)

            loss = difficulty_based_kd_loss(
                student_logits, teacher_logits, labels,
                cfg.temperature, cfg.alpha, cfg.num_classes,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct += student_logits.argmax(1).eq(labels).sum().item()
            n += imgs.size(0)

        scheduler.step()
        val_acc = evaluate_model(student, test_loader, device)
        print(f"  Epoch {epoch:>2}/{cfg.kd_epochs}  "
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


# ══════════════════════════════════════════════════════════════════════════════
#  FLOPs counting
# ══════════════════════════════════════════════════════════════════════════════

def count_flops(model: nn.Module) -> int:
    """Estimate MACs for a CIFAR-10 model using ptflops.

    Args:
        model: Network to profile.

    Returns:
        Estimated MACs.
    """
    try:
        from ptflops import get_model_complexity_info
        macs, _ = get_model_complexity_info(
            model, (3, 32, 32), as_strings=False,
            print_per_layer_stat=False, verbose=False,
        )
        return int(macs)
    except ImportError:
        return sum(p.numel() for p in model.parameters()) * 2


def format_flops(flops: int) -> str:
    """Format FLOPs count.

    Args:
        flops: Number of FLOPs/MACs.

    Returns:
        Human-readable string.
    """
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    return f"{flops / 1e6:.2f} MFLOPs"


# ══════════════════════════════════════════════════════════════════════════════
#  Adversarial Transferability
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_transferability(
    teacher: nn.Module,
    student: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    eps: float = 4 / 255,
    max_batches: int = 20,
) -> Tuple[float, float, float, float]:
    """Generate adversarial samples from teacher, evaluate on both models.

    Args:
        teacher:     Teacher network.
        student:     Student network.
        test_loader: Test data loader.
        device:      Compute device.
        eps:         L∞ perturbation budget.
        max_batches: Number of batches to evaluate.

    Returns:
        ``(teacher_clean, teacher_adv, student_clean, student_adv)`` accuracies.
    """
    teacher.eval()
    student.eval()
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    eps_norm = (eps / std).min().item()
    alpha_norm = eps_norm / 10

    t_clean, t_adv, s_clean, s_adv, total = 0, 0, 0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(test_loader):
        if batch_idx >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            t_clean += teacher(imgs).argmax(1).eq(labels).sum().item()
            s_clean += student(imgs).argmax(1).eq(labels).sum().item()

        # Generate adversarial from TEACHER
        adv = pgd_attack(teacher, imgs, labels, eps_norm, alpha_norm, 20, "Linf")

        with torch.no_grad():
            t_adv += teacher(adv).argmax(1).eq(labels).sum().item()
            s_adv += student(adv).argmax(1).eq(labels).sum().item()

        total += labels.size(0)

    return t_clean / total, t_adv / total, s_clean / total, s_adv / total


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all five HW3 experiments."""
    cfg = parse_args()

    device = torch.device(
        cfg.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Standard loaders (no AugMix)
    train_loader, test_loader = get_loaders(cfg, use_augmix=False)

    # ═══════════════════════════════════════════════════════════════════
    #  PART 1: Fine-tune ResNet-18 (standard) + evaluate on CIFAR-10-C
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  PART 1: Fine-tune ResNet-18 (standard) + CIFAR-10-C")
    print(f"{'#'*60}")

    model_std = build_resnet18_cifar(cfg.num_classes).to(device)
    model_std = fine_tune("resnet18_standard", model_std,
                          train_loader, test_loader, cfg, device,
                          label_smoothing=cfg.label_smoothing)

    clean_acc_std = evaluate_model(model_std, test_loader, device)
    print(f"\n  Clean test accuracy (standard): {clean_acc_std:.4f}")

    print("\n  CIFAR-10-C evaluation (standard model):")
    c10c_std = evaluate_cifar10c(model_std, device, cfg.data_dir,
                                  severity=5, batch_size=cfg.batch_size)

    # ═══════════════════════════════════════════════════════════════════
    #  PART 2: Fine-tune ResNet-18 with AugMix + evaluate on CIFAR-10-C
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  PART 2: Fine-tune ResNet-18 (AugMix) + CIFAR-10-C")
    print(f"{'#'*60}")

    train_loader_aug, _ = get_loaders(cfg, use_augmix=True)

    model_aug = build_resnet18_cifar(cfg.num_classes).to(device)
    model_aug = fine_tune("resnet18_augmix", model_aug,
                          train_loader_aug, test_loader, cfg, device,
                          label_smoothing=cfg.label_smoothing)

    clean_acc_aug = evaluate_model(model_aug, test_loader, device)
    print(f"\n  Clean test accuracy (AugMix): {clean_acc_aug:.4f}")

    print("\n  CIFAR-10-C evaluation (AugMix model):")
    c10c_aug = evaluate_cifar10c(model_aug, device, cfg.data_dir,
                                  severity=5, batch_size=cfg.batch_size)

    # Plot corruption comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    ctypes = [c for c in CORRUPTION_TYPES if c in c10c_std and c in c10c_aug]
    x = np.arange(len(ctypes))
    w = 0.35
    ax.bar(x - w/2, [c10c_std[c] for c in ctypes], w, label="Standard", color="steelblue")
    ax.bar(x + w/2, [c10c_aug[c] for c in ctypes], w, label="AugMix", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(ctypes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("CIFAR-10-C Robustness: Standard vs AugMix (severity 5)")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("cifar10c_comparison.png", dpi=150, bbox_inches="tight")
    print("  Saved cifar10c_comparison.png")

    # ═══════════════════════════════════════════════════════════════════
    #  PART 3: PGD attacks + GradCAM + t-SNE
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  PART 3: PGD Attacks + GradCAM + t-SNE")
    print(f"{'#'*60}")

    for mtag, mdl in [("standard", model_std), ("augmix", model_aug)]:
        print(f"\n  --- {mtag.upper()} model ---")

        # PGD L∞
        print(f"  PGD20 L∞ ε=4/255:")
        c_acc, a_acc = evaluate_pgd(mdl, test_loader, device,
                                     eps=4/255, norm="Linf", steps=20)
        print(f"    Clean: {c_acc:.4f}  Adversarial: {a_acc:.4f}")

        # PGD L2
        print(f"  PGD20 L2 ε=0.25:")
        c_acc2, a_acc2 = evaluate_pgd(mdl, test_loader, device,
                                       eps=0.25, norm="L2", steps=20)
        print(f"    Clean: {c_acc2:.4f}  Adversarial: {a_acc2:.4f}")

        # GradCAM
        print(f"  Generating GradCAM visualisation...")
        visualise_gradcam_adversarial(mdl, test_loader, device, mtag)

        # t-SNE
        print(f"  Generating t-SNE...")
        visualise_tsne_adversarial(mdl, test_loader, device, mtag,
                                    n_samples=1000)

    # ═══════════════════════════════════════════════════════════════════
    #  PART 4: AugMix teacher → Knowledge Distillation → MobileNetV2
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  PART 4: Knowledge Distillation (AugMix teacher → MobileNetV2)")
    print(f"{'#'*60}")

    student_augmix = MobileNetV2(num_classes=cfg.num_classes).to(device)
    student_augmix = train_with_distillation(
        "mobilenet_augmix_teacher", student_augmix, model_aug,
        train_loader, test_loader, cfg, device,
    )
    student_aug_acc = evaluate_model(student_augmix, test_loader, device)

    # Also do KD with standard teacher for comparison
    student_std = MobileNetV2(num_classes=cfg.num_classes).to(device)
    student_std = train_with_distillation(
        "mobilenet_std_teacher", student_std, model_std,
        train_loader, test_loader, cfg, device,
    )
    student_std_acc = evaluate_model(student_std, test_loader, device)

    print(f"\n  KD Results:")
    print(f"    Standard teacher  → MobileNetV2: {student_std_acc:.4f}")
    print(f"    AugMix teacher    → MobileNetV2: {student_aug_acc:.4f}")

    teacher_flops = count_flops(model_aug)
    student_flops = count_flops(student_augmix)
    print(f"    Teacher FLOPs: {format_flops(teacher_flops)}")
    print(f"    Student FLOPs: {format_flops(student_flops)}")
    print(f"    Compression:   {teacher_flops / student_flops:.1f}×")

    # ═══════════════════════════════════════════════════════════════════
    #  PART 5: Adversarial Transferability
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  PART 5: Adversarial Transferability")
    print(f"{'#'*60}")

    # Use AugMix teacher to generate adversarial, test on its student
    print("\n  AugMix teacher → student (MobileNetV2):")
    t_c, t_a, s_c, s_a = evaluate_transferability(
        model_aug, student_augmix, test_loader, device, eps=4/255, max_batches=20,
    )
    print(f"    Teacher clean: {t_c:.4f}  Teacher adv: {t_a:.4f}")
    print(f"    Student clean: {s_c:.4f}  Student adv (transferred): {s_a:.4f}")

    # Also test standard teacher → its student
    print("\n  Standard teacher → student (MobileNetV2):")
    t_c2, t_a2, s_c2, s_a2 = evaluate_transferability(
        model_std, student_std, test_loader, device, eps=4/255, max_batches=20,
    )
    print(f"    Teacher clean: {t_c2:.4f}  Teacher adv: {t_a2:.4f}")
    print(f"    Student clean: {s_c2:.4f}  Student adv (transferred): {s_a2:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<50} {'Value':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Standard ResNet-18 clean accuracy':<50} {clean_acc_std:>10.4f}")
    print(f"  {'AugMix   ResNet-18 clean accuracy':<50} {clean_acc_aug:>10.4f}")
    print(f"  {'Standard CIFAR-10-C avg accuracy':<50} {c10c_std.get('AVERAGE', 0):>10.4f}")
    print(f"  {'AugMix   CIFAR-10-C avg accuracy':<50} {c10c_aug.get('AVERAGE', 0):>10.4f}")
    print(f"  {'KD: Standard teacher → MobileNetV2':<50} {student_std_acc:>10.4f}")
    print(f"  {'KD: AugMix teacher   → MobileNetV2':<50} {student_aug_acc:>10.4f}")
    print()


if __name__ == "__main__":
    main()
