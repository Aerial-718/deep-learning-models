"""经典 LeNet 风格卷积神经网络。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    """用于 28×28 灰度图像分类的 LeNet 风格网络。

    输入会先补零到 32×32，随后依次通过：
    C1(1→6) - S2 - C3(6→16) - S4 - C5(16→120) - F6(120→84) - 输出层。
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes 必须为正整数")

        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != 1:
            raise ValueError(
                f"期望输入形状为 [N, 1, H, W]，实际为 {tuple(inputs.shape)}"
            )
        if inputs.shape[-2:] != (28, 28):
            raise ValueError(
                f"期望图像尺寸为 28×28，实际为 {tuple(inputs.shape[-2:])}"
            )

        inputs = F.pad(inputs, (2, 2, 2, 2))
        features = self.features(inputs)
        features = torch.flatten(features, 1)
        return self.classifier(features)

