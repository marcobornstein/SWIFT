"""CIFAR-style ResNet.

Narrower than the ImageNet ResNet: 16/32/64 channels over three stages with no
initial downsampling, which is the standard variant for 32x32 inputs. Kept
architecturally identical to the model used for the paper's results, including
the (unusual) biased convolutions ahead of each batch-norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONFIGS: dict[int, tuple[str, list[int]]] = {
    18: ("basic", [2, 2, 2, 2]),
    34: ("basic", [3, 4, 6, 3]),
    50: ("bottleneck", [3, 4, 6, 3]),
    101: ("bottleneck", [3, 4, 23, 3]),
    152: ("bottleneck", [3, 8, 36, 3]),
}


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = _shortcut(in_planes, planes, stride, self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = _shortcut(in_planes, planes, stride, self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


def _shortcut(
    in_planes: int, planes: int, stride: int, expansion: int
) -> nn.Module:
    if stride == 1 and in_planes == expansion * planes:
        return nn.Sequential()
    return nn.Sequential(
        nn.Conv2d(in_planes, expansion * planes, kernel_size=1, stride=stride, bias=True),
        nn.BatchNorm2d(expansion * planes),
    )


class ResNet(nn.Module):
    def __init__(self, depth: int, num_classes: int = 10) -> None:
        super().__init__()
        if depth not in _CONFIGS:
            raise ValueError(
                f"ResNet depth must be one of {sorted(_CONFIGS)}, got {depth}"
            )
        kind, num_blocks = _CONFIGS[depth]
        block = BasicBlock if kind == "basic" else Bottleneck

        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(
        self, block: type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = []
        for block_stride in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, block_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def build_model(depth: int, num_classes: int = 10) -> ResNet:
    return ResNet(depth, num_classes)
