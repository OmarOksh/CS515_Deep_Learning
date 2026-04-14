"""
Parameter configuration for HW3: Robustness and Adversarial Samples.

Uses dataclasses for structured configuration and argparse for CLI parsing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataParams:
    """Dataset-related configuration.

    Attributes:
        dataset:     Dataset name (always 'cifar10' for this HW).
        data_dir:    Root directory for downloading / caching.
        num_workers: DataLoader worker processes.
        mean:        Per-channel normalisation mean.
        std:         Per-channel normalisation std.
    """
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 2
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.2023, 0.1994, 0.2010)


@dataclass
class ModelParams:
    """Model configuration.

    Attributes:
        model:         Architecture name.
        num_classes:   Output classes.
        resnet_layers: Blocks per ResNet stage.
    """
    model: str = "resnet"
    num_classes: int = 10
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])


@dataclass
class TrainParams:
    """Training hyper-parameters.

    Attributes:
        epochs:          Fine-tuning epochs.
        batch_size:      Mini-batch size.
        learning_rate:   Initial learning rate.
        weight_decay:    L2 regularisation.
        label_smoothing: Smoothing factor.
    """
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


@dataclass
class AttackParams:
    """Adversarial attack configuration.

    Attributes:
        pgd_steps:   Number of PGD iterations.
        eps_linf:    L∞ perturbation budget (pixel space).
        eps_l2:      L2 perturbation budget (pixel space).
    """
    pgd_steps: int = 20
    eps_linf: float = 4 / 255
    eps_l2: float = 0.25


@dataclass
class KDParams:
    """Knowledge distillation configuration.

    Attributes:
        temperature: Softmax temperature.
        alpha:       Soft loss weight.
        kd_epochs:   Training epochs for distillation.
    """
    temperature: float = 4.0
    alpha: float = 0.7
    kd_epochs: int = 20


@dataclass
class MiscParams:
    """Runtime settings.

    Attributes:
        seed:   Random seed.
        device: Compute device.
        mode:   Experiment mode.
    """
    seed: int = 42
    device: str = "cpu"
    mode: str = "all"


@dataclass
class Params:
    """Top-level configuration aggregating all parameter groups.

    Attributes:
        data:    Dataset parameters.
        model:   Model parameters.
        train:   Training hyper-parameters.
        attack:  Adversarial attack settings.
        kd:      Knowledge distillation settings.
        misc:    Runtime settings.
    """
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    train: TrainParams = field(default_factory=TrainParams)
    attack: AttackParams = field(default_factory=AttackParams)
    kd: KDParams = field(default_factory=KDParams)
    misc: MiscParams = field(default_factory=MiscParams)


def get_params() -> Params:
    """Parse command-line arguments and return a :class:`Params` instance.

    Returns:
        Populated configuration.
    """
    parser = argparse.ArgumentParser(
        description="HW3: Robustness & Adversarial Samples on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--kd_epochs", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    args = parser.parse_args()

    return Params(
        train=TrainParams(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            label_smoothing=args.label_smoothing,
        ),
        attack=AttackParams(pgd_steps=args.pgd_steps),
        kd=KDParams(
            temperature=args.temperature,
            alpha=args.alpha,
            kd_epochs=args.kd_epochs,
        ),
        misc=MiscParams(device=args.device),
    )
