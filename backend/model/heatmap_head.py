"""
Heatmap Regression Head: 3 deconvolution layers producing 17 Gaussian keypoint heatmaps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    """
    Takes encoder feature maps and produces per-keypoint heatmaps.
    Uses transposed convolutions (deconv) to upsample back to 64x64.
    """
    def __init__(self, in_channels: int = 512, num_keypoints: int = 17):
        super().__init__()
        self.deconv1 = self._deconv_block(in_channels, 256)
        self.deconv2 = self._deconv_block(256, 256)
        self.deconv3 = self._deconv_block(256, 256)
        self.final = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def _deconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = self.deconv1(feat_map)
        x = self.deconv2(x)
        x = self.deconv3(x)
        heatmaps = self.final(x)  # (B, 17, 64, 64)
        return heatmaps


def heatmaps_to_keypoints(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Convert heatmaps (B, K, H, W) to keypoint coordinates (B, K, 2).
    Uses soft-argmax for differentiability.
    """
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    flat = F.softmax(flat, dim=-1)
    xs = torch.arange(W, device=heatmaps.device, dtype=torch.float32)
    ys = torch.arange(H, device=heatmaps.device, dtype=torch.float32)
    grid_x = xs.repeat(H, 1).reshape(-1)
    grid_y = ys.unsqueeze(1).repeat(1, W).reshape(-1)
    pred_x = (flat * grid_x).sum(dim=-1) / W
    pred_y = (flat * grid_y).sum(dim=-1) / H
    return torch.stack([pred_x, pred_y], dim=-1)  # (B, K, 2) normalized [0,1]


def gaussian_heatmap(size: int, center_x: float, center_y: float, sigma: float = 2.0):
    """Generate a 2D Gaussian heatmap for a single keypoint."""
    import numpy as np
    x = np.arange(size)
    y = np.arange(size)
    xv, yv = np.meshgrid(x, y)
    cx = center_x * size
    cy = center_y * size
    heatmap = np.exp(-((xv - cx) ** 2 + (yv - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)
