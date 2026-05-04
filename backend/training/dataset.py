"""
Dataset and augmentation pipeline for SpineAI training.
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import settings
from model.heatmap_head import gaussian_heatmap
from model.sea_generalizer import compute_ratios, ANTHROPOMETRIC_BASELINES


ETHNICITY_TO_IDX = {name: i for i, name in enumerate(ANTHROPOMETRIC_BASELINES.keys())}

# Newer albumentations dropped "normalized" format.
# We use "xy" (absolute pixel coords) instead and normalize manually after transform.
# Keypoints stored in annotations as [0,1] are scaled up before transform,
# then divided back down after.


def get_train_transforms():
    return A.Compose([
        A.Resize(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


def get_val_transforms():
    return A.Compose([
        A.Resize(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


class SpineDataset(Dataset):
    """
    Dataset format (JSON annotation file):
    {
      "images": [
        {
          "id": 1,
          "file_path": "images/patient_001.jpg",
          "ethnicity": "South Asian",
          "keypoints": [[x0,y0], [x1,y1], ...],  // 17 normalized [0,1]
          "labels": [0, 1, 0, 0, 1, 0, 0],        // multi-hot, 7 classes
          "severity": [0.0, 0.72, 0.0, 0.0, 0.45, 0.0, 0.0]  // [0,1]
        }
      ]
    }
    """
    def __init__(self, annotation_file: str, split: str = "train"):
        self.split = split
        self.transforms = get_train_transforms() if split == "train" else get_val_transforms()
        with open(annotation_file) as f:
            data = json.load(f)
        self.samples = data["images"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = cv2.imread(sample["file_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Keypoints in annotations are normalized [0,1].
        # Scale to absolute pixel coords for albumentations "xy" format.
        kps = sample["keypoints"]  # list of [x, y] in [0,1]
        kps_alb = [(x * w, y * h) for x, y in kps]

        transformed = self.transforms(image=img, keypoints=kps_alb)
        image = transformed["image"]
        kps_t = transformed["keypoints"]

        # Re-normalize back to [0,1] after transform (image is now IMAGE_SIZE x IMAGE_SIZE)
        out_size = settings.IMAGE_SIZE
        keypoints = np.zeros((settings.NUM_KEYPOINTS, 2), dtype=np.float32)
        for i, kp in enumerate(kps_t):
            if i < settings.NUM_KEYPOINTS:
                keypoints[i] = [kp[0] / out_size, kp[1] / out_size]

        # Generate Gaussian heatmaps
        heatmaps = np.zeros((settings.NUM_KEYPOINTS, 64, 64), dtype=np.float32)
        for k in range(settings.NUM_KEYPOINTS):
            if keypoints[k].sum() > 0:
                heatmaps[k] = gaussian_heatmap(64, keypoints[k][0], keypoints[k][1], sigma=2.0)

        # Compute ratios from GT keypoints
        ratios_dict = compute_ratios(keypoints)
        raw_ratios = np.array([
            ratios_dict["THR"], ratios_dict["SHR"],
            ratios_dict["LBP"], ratios_dict["CLB"]
        ], dtype=np.float32)

        ethnicity = sample.get("ethnicity", "European")
        ethnicity_idx = ETHNICITY_TO_IDX.get(ethnicity, 3)

        labels = np.array(sample["labels"], dtype=np.float32)
        severity = np.array(sample.get("severity", [0.0] * 7), dtype=np.float32)

        return {
            "image": image,
            "heatmaps": torch.tensor(heatmaps),
            "keypoints": torch.tensor(keypoints),
            "raw_ratios": torch.tensor(raw_ratios),
            "ethnicity_idx": torch.tensor(ethnicity_idx, dtype=torch.long),
            "labels": torch.tensor(labels),
            "severity": torch.tensor(severity),
        }
