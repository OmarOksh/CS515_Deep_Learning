"""VGG network family for CIFAR-10 classification.

Reference:
    Simonyan, K. and Zisserman, A. (2014). Very Deep Convolutional Networks
    for Large-Scale Image Recognition. arXiv:1409.1556.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn


# Channel configurations for each VGG variant; 'M' = MaxPool2d.
_CFG: Dict[str, List[Union[int, str]]] = {
    "11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512, "M"],
    "19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M",
           512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    """VGG network with batch normalisation, adapted for 32×32 CIFAR images.

    After five max-pool stages the spatial resolution is 1×1, producing a
    512-d feature vector that feeds into a three-layer classifier head.

    Args:
        depth:       VGG variant ('11', '13', '16', or '19').
        norm:        Normalisation layer constructor (default: ``nn.BatchNorm2d``).
        num_classes: Number of output logits.
    """

    def __init__(
        self,
        depth: str = "16",
        norm: type = nn.BatchNorm2d,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.features = self._make_layers(depth, norm)
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _make_layers(
        depth: str,
        norm: type = nn.BatchNorm2d,
    ) -> nn.Sequential:
        """Build the convolutional feature extractor from a config string.

        Args:
            depth: Key into ``_CFG`` selecting the VGG variant.
            norm:  Normalisation layer constructor.

        Returns:
            ``nn.Sequential`` containing all conv/BN/ReLU/pool layers.
        """
        layers: List[nn.Module] = []
        in_channels = 3
        for v in _CFG[depth]:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                assert isinstance(v, int)
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                    norm(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, 3, 32, 32)``.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
