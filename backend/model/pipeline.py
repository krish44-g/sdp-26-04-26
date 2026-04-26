"""
Multi-label Deformity Classifier and end-to-end PostureNet pipeline.
"""
import torch
import torch.nn as nn
from .posturenet import PostureNet
from .heatmap_head import HeatmapHead, heatmaps_to_keypoints
from .sea_generalizer import SEAGeneralizer, compute_ratios
import numpy as np


class DeformityClassifier(nn.Module):
    """FC classifier on SEA-corrected feature vector."""
    def __init__(self, sea_dim: int = 64, global_dim: int = 512, num_classes: int = 7):
        super().__init__()
        in_dim = sea_dim + global_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, sea_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sea_feat, global_feat], dim=-1)
        return self.head(x)  # raw logits


class SpineAIModel(nn.Module):
    """
    Full end-to-end SpineAI model.
    Input: image (B, 3, 256, 256) + ethnicity_idx (B,)
    Output: heatmaps, keypoints, corrected_ratios, logits
    """
    def __init__(self, num_keypoints: int = 17, num_classes: int = 7, num_ethnicities: int = 6):
        super().__init__()
        self.backbone = PostureNet()
        self.heatmap_head = HeatmapHead(in_channels=512, num_keypoints=num_keypoints)
        self.sea = SEAGeneralizer(num_ethnicities=num_ethnicities, output_dim=64)
        self.classifier = DeformityClassifier(sea_dim=64, global_dim=512, num_classes=num_classes)

    def forward(
        self,
        images: torch.Tensor,
        ethnicity_idx: torch.Tensor,
        raw_ratios: torch.Tensor = None,
    ):
        feat_map, global_feat, _ = self.backbone(images)
        heatmaps = self.heatmap_head(feat_map)
        keypoints = heatmaps_to_keypoints(heatmaps)  # (B, K, 2) normalized

        if raw_ratios is None:
            # Compute ratios from predicted keypoints (inference mode)
            kp_np = keypoints.detach().cpu().numpy()
            ratios_list = [compute_ratios(kp_np[i]) for i in range(len(kp_np))]
            raw_ratios = torch.tensor(
                [[r["THR"], r["SHR"], r["LBP"], r["CLB"]] for r in ratios_list],
                dtype=torch.float32, device=images.device
            )

        sea_feat, corrected_ratios = self.sea(raw_ratios, ethnicity_idx, global_feat)
        logits = self.classifier(sea_feat, global_feat)

        return {
            "heatmaps": heatmaps,
            "keypoints": keypoints,
            "raw_ratios": raw_ratios,
            "corrected_ratios": corrected_ratios,
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
        }
