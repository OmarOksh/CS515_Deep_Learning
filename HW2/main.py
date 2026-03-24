"""Entry point: parse CLI arguments, build a model, train and/or test it."""

from __future__ import annotations

import random
import ssl

import numpy as np
import torch
import torch.nn as nn

from parameters import Params, get_params
from models.MLP import MLP
from models.CNN import MNIST_CNN, SimpleCNN
from models.VGG import VGG
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from train import run_training
from test import run_test

# Fix for macOS SSL certificate verification error when downloading datasets
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(params: Params) -> nn.Module:
    """Instantiate the neural-network architecture specified in *params*.

    Args:
        params: Global configuration (model name, dataset, hyper-parameters).

    Returns:
        An uninitialised (randomly initialised) ``nn.Module``.

    Raises:
        ValueError: If the model/dataset combination is invalid or unknown.
    """
    model_name = params.model.model
    dataset = params.data.dataset
    nc = params.model.num_classes

    if model_name == "mlp":
        return MLP(
            input_size=params.model.input_size,
            hidden_sizes=params.model.hidden_sizes,
            num_classes=nc,
            dropout=params.model.dropout,
        )

    if model_name == "cnn":
        if dataset == "mnist":
            return MNIST_CNN(num_classes=nc)
        return SimpleCNN(num_classes=nc)

    if model_name == "vgg":
        if dataset == "mnist":
            raise ValueError(
                "VGG requires 3-channel images; use --dataset cifar10."
            )
        return VGG(depth=params.model.vgg_depth, num_classes=nc)

    if model_name == "resnet":
        if dataset == "mnist":
            raise ValueError(
                "ResNet requires 3-channel images; use --dataset cifar10."
            )
        return ResNet(BasicBlock, params.model.resnet_layers, num_classes=nc)

    if model_name == "mobilenet":
        if dataset == "mnist":
            raise ValueError(
                "MobileNetV2 requires 3-channel images; use --dataset cifar10."
            )
        return MobileNetV2(num_classes=nc)

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    """Parse arguments, build the model, and run training / testing."""
    params = get_params()

    set_seed(params.misc.seed)
    print(f"Seed set to: {params.misc.seed}")
    print(f"Dataset: {params.data.dataset}  |  Model: {params.model.model}")

    device = torch.device(
        params.misc.device if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    model = build_model(params).to(device)
    print(model)

    if params.misc.mode in ("train", "both"):
        run_training(model, params, device)

    if params.misc.mode in ("test", "both"):
        run_test(model, params, device)


if __name__ == "__main__":
    main()
