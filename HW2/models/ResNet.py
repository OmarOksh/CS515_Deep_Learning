"""ResNet implementation for CIFAR-10 classification.

Reference:
    He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep Residual Learning
    for Image Recognition. CVPR, pp. 770-778.
"""

from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaLayer(nn.Module):
    """Apply an arbitrary callable as an ``nn.Module``.

    This is used by *option A* shortcuts in the CIFAR-10 ResNet, which
    down-sample via slicing and zero-pad channels instead of using a
    learned 1×1 convolution.

    Args:
        lambd: Callable applied to the input tensor.
    """

    def __init__(self, lambd) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    """Basic residual block (two 3×3 convolutions) for ResNet-18 / 34.

    Structure::

        x ──→ Conv3×3 → BN → ReLU → Conv3×3 → BN ──→ (+) → ReLU
        │                                               ↑
        └───────────── shortcut ────────────────────────┘

    When spatial resolution or channel count changes, the shortcut is
    either a 1×1 convolution (option B) or a slice-and-pad (option A).

    Args:
        in_channels: Number of input channels.
        channels:    Number of output channels.
        stride:      Stride for the first convolution.
        norm:        Normalisation layer constructor.
        option:      Shortcut strategy: ``'A'`` (pad) or ``'B'`` (1×1 conv).
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        norm: Type[nn.Module] = nn.BatchNorm2d,
        option: str = "B",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm(channels)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != channels:
            if option == "A":
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, channels // 4, channels // 4),
                        "constant", 0,
                    )
                )
            else:  # option B
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, self.expansion * channels,
                              kernel_size=1, stride=stride, bias=False),
                    norm(self.expansion * channels),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(B, channels, H/stride, W/stride)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """ResNet for CIFAR-10 (3×3 initial conv, no initial max-pool).

    The network consists of an initial 3×3 convolution followed by four
    residual stages, global average pooling, and a linear classifier.

    Args:
        block:       Residual block class (e.g. :class:`BasicBlock`).
        num_blocks:  Number of blocks in each of the four stages.
        norm:        Normalisation layer constructor.
        num_classes: Number of output logits.

    Example::

        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> logits = model(torch.randn(8, 3, 32, 32))
        >>> logits.shape
        torch.Size([8, 10])
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        norm: Type[nn.Module] = nn.BatchNorm2d,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels: int = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], norm, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        channels: int,
        num_blocks: int,
        norm: Type[nn.Module],
        stride: int,
    ) -> nn.Sequential:
        """Construct one residual stage.

        Args:
            block:      Block class.
            channels:   Output channel width for this stage.
            num_blocks: Number of blocks to stack.
            norm:       Normalisation layer constructor.
            stride:     Stride for the first block (subsequent blocks use 1).

        Returns:
            ``nn.Sequential`` containing all blocks for this stage.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []
        for s in strides:
            layers.append(block(self.in_channels, channels, s, norm))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)
