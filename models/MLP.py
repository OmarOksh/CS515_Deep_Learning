"""
Multi-Layer Perceptron (MLP) architectures for MNIST classification.

Provides two implementations:
    - MLP: Built with nn.Sequential.
    - MLP2: Built with nn.ModuleList for explicit layer control.
"""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Multi-Layer Perceptron using nn.Sequential.

    Args:
        input_size (int): Flattened input dimension (e.g. 784 for MNIST).
        hidden_sizes (List[int]): Width of each hidden layer.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
        activation (str): 'relu' or 'gelu'.
        use_bn (bool): Whether to include BatchNorm1d layers.
        bn_position (str): 'before' or 'after' activation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_bn: bool = True,
        bn_position: str = "before",
    ):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(nn.Flatten())

        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))

            if use_bn and bn_position == "before":
                layers.append(nn.BatchNorm1d(h))

            if activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())

            if use_bn and bn_position == "after":
                layers.append(nn.BatchNorm1d(h))

            layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sequential network."""
        return self.net(x)


class MLP2(nn.Module):
    """
    Multi-Layer Perceptron using nn.ModuleList.

    Args:
        input_size (int): Flattened input dimension.
        hidden_sizes (List[int]): Width of each hidden layer.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability.
        activation (str): 'relu' or 'gelu'.
        use_bn (bool): Whether to include BatchNorm1d layers.
        bn_position (str): 'before' or 'after' activation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.3,
        activation: str = "relu",
        use_bn: bool = True,
        bn_position: str = "before",
    ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.use_bn = use_bn
        self.bn_position = bn_position

        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.drops = nn.ModuleList()

        prev_dim = input_size
        for h_dim in hidden_sizes:
            self.linears.append(nn.Linear(prev_dim, h_dim))
            self.bns.append(nn.BatchNorm1d(h_dim) if use_bn else nn.Identity())
            if activation.lower() == "gelu":
                self.acts.append(nn.GELU())
            else:
                self.acts.append(nn.ReLU())
            self.drops.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass with optional feature extraction for t-SNE.

        Args:
            x: Input tensor.
            return_features: If True, returns (logits, penultimate_features).
        """
        x = self.flatten(x)

        for linear, bn, act, drop in zip(self.linears, self.bns, self.acts, self.drops):
            x = linear(x)
            if self.use_bn and self.bn_position == "before":
                x = bn(x)
            x = act(x)
            if self.use_bn and self.bn_position == "after":
                x = bn(x)
            x = drop(x)

        features = x
        logits = self.output_layer(x)

        if return_features:
            return logits, features
        return logits
