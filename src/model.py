"""
Custom RestNet CNN for CIFAR-10 classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnGelu(nn.Module):
    """Conv2d → BatchNorm2d → GELU activation block."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module): #learns to re-weight feature channels. Forces to focus on the discriminative channels of each class. Used to fix wrong prediction for visually similar classes like cat and dog, automobile and truck, etc.

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 se_reduction: int = 16):
        super().__init__()
        self.conv1 = ConvBnGelu(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se = SEBlock(out_ch, reduction=se_reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)              
        return F.gelu(out + self.shortcut(x))

class CIFAR10Net(nn.Module):
    """
    ResNet-style CNN for CIFAR-10.

    Architecture:
        Stem        : 3  -> 64  ch
        Stage 1     : 64  -> 64  ch, 2 blocks (32x32)
        Stage 2     : 64  -> 128 ch, 2 blocks (16x16)
        Stage 3     : 128 -> 256 ch, 2 blocks (8x8)
        Stage 4     : 256 -> 512 ch, 2 blocks (4x4)
        Head        : GlobalAvgPool -> Dropout -> Linear(10)
    """

    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.stem   = ConvBnGelu(3, 64)
        self.stage1 = self._make_stage(64,  64,  n_blocks=2, stride=1)
        self.stage2 = self._make_stage(64,  128, n_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, n_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, n_blocks=2, stride=2)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(num_classes=10, dropout=0.3):
    model = CIFAR10Net(num_classes=num_classes, dropout=dropout)
    print(f"[model] CIFAR10Net | params={model.count_parameters():,}")
    return model


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 32, 32)
    print(f"Output shape: {model(dummy).shape}") 