"""
Experiment configuration module.

Provides the ExperimentParams dataclass and CLI argument parser for
configuring MLP training, including architecture, regularization,
scheduler, early stopping, and batch-normalization ablation options.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ExperimentParams:
    """
    Stores all hyperparameters and configuration for a single experiment run.

    Attributes:
        mode: One of 'train', 'test', or 'both'.
        dataset: Dataset name — 'mnist' or 'cifar10'.
        model: Model architecture — 'mlp', 'cnn', 'vgg', or 'resnet'.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimizer.
        device: Compute device string (e.g. 'cpu', 'cuda', 'mps').
        batch_size: Mini-batch size for data loaders.
        vgg_depth: VGG variant depth.
        resnet_layers: Block counts per ResNet stage.
        activation: Activation function — 'relu' or 'gelu'.
        scheduler: LR scheduler type — 'step', 'cosine', or 'plateau'.
        l1_lambda: L1 regularization coefficient (0.0 to disable).
        use_bn: Whether to use batch normalization.
        bn_position: BN placement — 'before' or 'after' activation.
        early_stop_patience: Epochs to wait for val improvement before stopping (0 = disabled).
    """

    # --- CLI arguments ---
    mode: str
    dataset: str
    model: str
    epochs: int
    learning_rate: float
    device: str
    batch_size: int
    vgg_depth: str
    resnet_layers: List[int]
    activation: str
    scheduler: str
    l1_lambda: float
    use_bn: bool
    bn_position: str
    early_stop_patience: int

    # --- Defaults ---
    data_dir: str = "./data"
    num_workers: int = 2
    num_classes: int = 10
    dropout: float = 0.3
    weight_decay: float = 1e-4
    seed: int = 42
    save_path: str = "best_model.pth"
    log_interval: int = 100
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])

    # --- Computed after init ---
    input_size: int = field(init=False)
    mean: Tuple[float, ...] = field(init=False)
    std: Tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize dataset-dependent parameters."""
        if self.dataset == "mnist":
            self.input_size = 784
            self.mean = (0.1307,)
            self.std = (0.3081,)
        else:
            self.input_size = 3072
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)


def get_params() -> ExperimentParams:
    """
    Parse command-line arguments and return an ExperimentParams instance.

    Returns:
        ExperimentParams: Fully configured experiment parameters.
    """
    parser = argparse.ArgumentParser(
        description="Deep Learning on MNIST / CIFAR-10"
    )

    parser.add_argument(
        "--mode", choices=["train", "test", "both"], default="both"
    )
    parser.add_argument(
        "--dataset", choices=["mnist", "cifar10"], default="mnist"
    )
    parser.add_argument(
        "--model", choices=["mlp", "cnn", "vgg", "resnet"], default="mlp"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Width of hidden layers",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout probability"
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="Activation function",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="L2 regularization / weight decay",
    )

    # ---- NEW arguments for missing experiments ----
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=0.0,
        help="L1 regularization coefficient (0.0 = disabled)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["step", "cosine", "plateau"],
        default="step",
        help="LR scheduler: step, cosine, or plateau",
    )
    parser.add_argument(
        "--no_bn",
        action="store_true",
        help="Disable batch normalization entirely",
    )
    parser.add_argument(
        "--bn_position",
        type=str,
        choices=["before", "after"],
        default="before",
        help="Place BN before or after activation",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs (0 = disabled)",
    )

    # VGG / ResNet specific
    parser.add_argument(
        "--vgg_depth", choices=["11", "13", "16", "19"], default="16"
    )
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Blocks per ResNet layer (default: 2 2 2 2 = ResNet-18)",
    )

    args = parser.parse_args()

    return ExperimentParams(
        mode=args.mode,
        dataset=args.dataset,
        model=args.model,
        epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        batch_size=args.batch_size,
        vgg_depth=args.vgg_depth,
        resnet_layers=args.resnet_layers,
        activation=args.activation,
        hidden_sizes=args.hidden_sizes,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        # New fields
        l1_lambda=args.l1_lambda,
        scheduler=args.scheduler,
        use_bn=not args.no_bn,
        bn_position=args.bn_position,
        early_stop_patience=args.early_stop_patience,
    )
