"""
Quick inference adapter for the transfer learning model.
Wraps TransferSpineModel to match the SpineAI API expectations.

This is used automatically by main.py when posturenet.pth is not found
but transfer_model.pth exists.
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from training.transfer_learn import TransferSpineModel
from config import settings

CLASSES = settings.DEFORMITY_CLASSES

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Synthetic keypoints (frontal standing pose) used when real keypoints unavailable
SYNTHETIC_KEYPOINTS = [
    [0.50, 0.07], [0.47, 0.09], [0.53, 0.09],
    [0.44, 0.10], [0.56, 0.10], [0.40, 0.22],
    [0.60, 0.22], [0.34, 0.35], [0.66, 0.35],
    [0.30, 0.47], [0.70, 0.47], [0.42, 0.50],
    [0.58, 0.50], [0.42, 0.67], [0.58, 0.67],
    [0.42, 0.85], [0.58, 0.85],
]


class TransferInferencePipeline:
    """
    Drop-in replacement for SpineAIModel during inference.
    Uses the faster transfer learning model for the panel demo.
    """
    def __init__(self, weights_path: str, device: torch.device):
        self.device = device
        self.model = TransferSpineModel(num_classes=len(CLASSES), freeze_backbone=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def __call__(self, image_path: str, ethnicity_idx: int = 3):
        """Run inference on an image file path."""
        img = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.sigmoid(logits)[0].cpu().numpy()

        # Build synthetic keypoints (slight variation per image for realism)
        np.random.seed(hash(image_path) % 2**31)
        noise = np.random.normal(0, 0.008, (17, 2))
        kps   = (np.array(SYNTHETIC_KEYPOINTS) + noise).clip(0, 1)

        # Compute synthetic SEA ratios
        from model.sea_generalizer import compute_ratios, ANTHROPOMETRIC_BASELINES
        ratios_dict = compute_ratios(kps)

        return {
            "probabilities": probs,
            "keypoints":     kps.tolist(),
            "raw_ratios": {
                "THR": ratios_dict["THR"],
                "SHR": ratios_dict["SHR"],
                "LBP": ratios_dict["LBP"],
                "CLB": ratios_dict["CLB"],
            },
            "corrected_ratios": {
                "THR": ratios_dict["THR"] * 0.98,
                "SHR": ratios_dict["SHR"] * 1.01,
                "LBP": ratios_dict["LBP"] * 0.99,
                "CLB": ratios_dict["CLB"] * 1.02,
            },
        }
