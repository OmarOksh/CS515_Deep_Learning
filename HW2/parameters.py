"""
Parameter configuration for training and evaluation.

Uses dataclasses for structured configuration and argparse for CLI parsing,
as required by the homework specification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataParams:
    """Dataset-related configuration.

    Attributes:
        dataset:     Name of the dataset ('mnist' or 'cifar10').
        data_dir:    Root directory for downloading / caching datasets.
        num_workers: Number of data-loader worker processes.
        mean:        Per-channel mean for normalisation.
        std:         Per-channel standard deviation for normalisation.
    """
    dataset: str = "mnist"
    data_dir: str = "./data"
    num_workers: int = 2
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)


@dataclass
class ModelParams:
    """Model architecture configuration.

    Attributes:
        model:         Architecture name (mlp | cnn | vgg | resnet | mobilenet).
        input_size:    Flattened input dimensionality (used by MLP).
        hidden_sizes:  Hidden-layer widths for MLP.
        num_classes:   Number of output classes.
        dropout:       Dropout probability (MLP).
        vgg_depth:     VGG variant depth ('11', '13', '16', or '19').
        resnet_layers: Number of residual blocks per stage (length-4 list).
    """
    model: str = "mlp"
    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    num_classes: int = 10
    dropout: float = 0.3
    vgg_depth: str = "16"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])


@dataclass
class TrainParams:
    """Training hyper-parameters.

    Attributes:
        epochs:        Total number of training epochs.
        batch_size:    Mini-batch size.
        learning_rate: Initial learning rate for the optimiser.
        weight_decay:  L2 regularisation coefficient.
    """
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class MiscParams:
    """Miscellaneous / runtime settings.

    Attributes:
        seed:         Random seed for reproducibility.
        device:       Preferred compute device ('cpu', 'cuda', 'mps').
        save_path:    File path for saving the best model checkpoint.
        log_interval: Print training stats every *log_interval* batches.
        mode:         Run mode ('train', 'test', or 'both').
    """
    seed: int = 42
    device: str = "cpu"
    save_path: str = "best_model.pth"
    log_interval: int = 100
    mode: str = "both"


@dataclass
class Params:
    """Top-level configuration that aggregates all parameter groups.

    Attributes:
        data:  Dataset parameters.
        model: Model architecture parameters.
        train: Training hyper-parameters.
        misc:  Runtime / miscellaneous settings.
    """
    data: DataParams = field(default_factory=DataParams)
    model: ModelParams = field(default_factory=ModelParams)
    train: TrainParams = field(default_factory=TrainParams)
    misc: MiscParams = field(default_factory=MiscParams)


def get_params() -> Params:
    """Parse command-line arguments and return a structured :class:`Params` object.

    Returns:
        Fully populated :class:`Params` dataclass instance.
    """
    parser = argparse.ArgumentParser(
        description="Deep Learning on MNIST / CIFAR-10"
    )

    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="both",
        help="Run mode: train, test, or both (default: both)."
    )
    parser.add_argument(
        "--dataset", choices=["mnist", "cifar10"], default="mnist",
        help="Dataset to use (default: mnist)."
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "vgg", "resnet", "mobilenet"],
        default="mlp",
        help="Model architecture (default: mlp)."
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Compute device: cpu, cuda, or mps (default: cpu)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Mini-batch size (default: 64)."
    )
    parser.add_argument(
        "--vgg_depth", choices=["11", "13", "16", "19"], default="16",
        help="VGG variant depth (default: 16)."
    )
    parser.add_argument(
        "--resnet_layers", type=int, nargs=4, default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Blocks per ResNet stage (default: 2 2 2 2 = ResNet-18)."
    )

    args = parser.parse_args()

    # ── Dataset-dependent defaults ──────────────────────────────────────
    if args.dataset == "mnist":
        input_size = 784                       # 1 × 28 × 28
        mean: Tuple[float, ...] = (0.1307,)
        std: Tuple[float, ...]  = (0.3081,)
    else:                                      # cifar10
        input_size = 3072                      # 3 × 32 × 32
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

    return Params(
        data=DataParams(
            dataset=args.dataset,
            mean=mean,
            std=std,
        ),
        model=ModelParams(
            model=args.model,
            input_size=input_size,
            vgg_depth=args.vgg_depth,
            resnet_layers=args.resnet_layers,
        ),
        train=TrainParams(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        ),
        misc=MiscParams(
            device=args.device,
            mode=args.mode,
        ),
    )
