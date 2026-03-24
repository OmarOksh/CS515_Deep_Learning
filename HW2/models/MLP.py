"""Multi-Layer Perceptron architectures."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Fully-connected feed-forward network with BatchNorm and Dropout.

    The input tensor is flattened to ``(B, input_size)`` before being passed
    through a stack of ``Linear → BatchNorm1d → ReLU → Dropout`` blocks,
    followed by a final linear classification head.

    Args:
        input_size:   Flattened input dimensionality (e.g. 784 for MNIST).
        hidden_sizes: List of hidden-layer widths.
        num_classes:  Number of output logits.
        dropout:      Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, input_size)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = x.view(x.size(0), -1)  # flatten (B, 1, 28, 28) → (B, 784)
        return self.net(x)


class MLP2(nn.Module):
    """Minimal MLP without BatchNorm or Dropout (for reference / comparison).

    Args:
        input_dim:   Flattened input dimensionality.
        hidden_dims: List of hidden-layer widths.
        num_classes: Number of output logits.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] | None = None,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            prev_dim = h_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C, H, W)`` or ``(B, input_dim)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = x.view(x.size(0), -1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
