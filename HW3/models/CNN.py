"""Convolutional Neural Network architectures for MNIST and CIFAR-10."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """Simple two-layer CNN designed for 1-channel 28×28 images (MNIST).

    Architecture::

        Conv2d(1→20, k=5) → ReLU → MaxPool(2)
        Conv2d(20→50, k=5) → ReLU → MaxPool(2)
        Linear(800→500) → ReLU
        Linear(500→num_classes)

    Args:
        num_classes: Number of output logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # (in_channels, out_channels, kernel_size, stride)
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        # After two conv+pool layers on 28×28 input we get 4×4 feature maps
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 1, 28, 28)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        # Output size formula: (W - F + 2P) / S + 1
        x = F.relu(self.conv1(x))   # (28-5)/1+1 = 24
        x = F.max_pool2d(x, 2, 2)   # 24/2 = 12
        x = F.relu(self.conv2(x))   # (12-5)/1+1 = 8
        x = F.max_pool2d(x, 2, 2)   # 8/2 = 4  →  50 × 4 × 4
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(nn.Module):
    """Lightweight CNN for 3-channel 32×32 images with Kaiming initialisation.

    Architecture::

        Conv2d(3→32, k=3, pad=1) → ReLU → MaxPool(2)   # 32→16
        Conv2d(32→64, k=3, pad=1) → ReLU → MaxPool(2)   # 16→8
        Linear(64·8·8 → 128) → ReLU
        Linear(128 → num_classes)

    Kaiming (He) initialisation is applied to all Conv2d and Linear layers.
    This keeps activation variance stable through ReLU networks by sampling
    weights from ``N(0, sqrt(2 / fan_in))``.

    Args:
        num_classes: Number of output logits.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)   # 32×32 → 16×16 → 8×8
        self.fc2 = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming normal initialisation to Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, 32, 32)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = F.relu(self.conv1(x))    # padding=1 keeps spatial size
        x = F.max_pool2d(x, 2)       # 32→16
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)       # 16→8
        x = x.view(x.size(0), -1)    # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
