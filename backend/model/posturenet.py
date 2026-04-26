"""
PostureNet: Custom CNN backbone for spine deformity detection.
ResNet-34 style architecture with channel-wise attention gates (SE blocks).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention gate."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """Residual block with SE attention."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


class PostureNet(nn.Module):
    """
    Dual-head backbone:
      - feature_map: fed to SEA Generalizer Layer
      - Returns feature maps at multiple scales for heatmap head
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        # Stages
        self.stage1 = self._make_stage(64, 64, blocks=3)
        self.stage2 = self._make_stage(64, 128, blocks=4, stride=2)
        self.stage3 = self._make_stage(128, 256, blocks=6, stride=2)
        self.stage4 = self._make_stage(256, 512, blocks=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def _make_stage(self, in_ch: int, out_ch: int, blocks: int, stride: int = 1):
        layers = [ResBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        s1 = self.stage1(x)   # /4
        s2 = self.stage2(s1)  # /8
        s3 = self.stage3(s2)  # /16
        s4 = self.stage4(s3)  # /32
        global_feat = self.gap(s4).flatten(1)  # (B, 512)
        return s4, global_feat, (s1, s2, s3)
