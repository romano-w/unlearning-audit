"""ResNet-18 adapted for CIFAR-10 (32x32 input).

Standard torchvision ResNet-18 uses a 7x7 conv + maxpool front end
designed for 224x224 ImageNet images.  For 32x32 CIFAR, the common
practice (He et al. 2016, BackdoorBench, etc.) replaces the front end
with a single 3x3 conv and drops the max-pool so the spatial resolution
isn't destroyed in the first layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

if TYPE_CHECKING:
    from unlearning_audit.config import ModelConfig


class CifarResNet(ResNet):
    """ResNet variant with a CIFAR-friendly stem."""

    def __init__(self, block: type, layers: list[int], num_classes: int = 10) -> None:
        super().__init__(block, layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Identity()  # type: ignore[assignment]


def build_resnet18(cfg: ModelConfig) -> CifarResNet:
    return CifarResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.num_classes)


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.arch == "resnet18":
        return build_resnet18(cfg)
    raise ValueError(f"Unknown architecture: {cfg.arch}")
