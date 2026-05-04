"""
Inference endpoint: accepts posture image + ethnicity, returns predictions.
"""
import torch
import cv2
import numpy as np
import uuid
import time
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from api.schemas import AnalysisResponse, EthnicityEnum, KeypointResult, DeformityResult, RatioResult
from config import settings, UPLOAD_DIR
from model.sea_generalizer import ETHNICITY_TO_IDX, compute_ratios

router = APIRouter()

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]


def get_model():
    """Dependency: load model singleton."""
    from main import model, device
    return model, device


def preprocess_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image for inference."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return img


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    ethnicity: EthnicityEnum = Form(EthnicityEnum.european),
    age: int = Form(None),
    sex: str = Form(None),
    model_tuple=Depends(get_model),
):
    model, device = model_tuple

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG/PNG images are supported")

    if file.size and file.size > settings.MAX_IMAGE_SIZE:
        raise HTTPException(400, "Image too large (max 10MB)")

    # Save uploaded file
    analysis_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".jpg"
    save_path = UPLOAD_DIR / f"{analysis_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    start_time = time.time()

    try:
        # Preprocess
        image_tensor = preprocess_image(str(save_path)).to(device)
        ethnicity_idx = torch.tensor(
            [ETHNICITY_TO_IDX.get(ethnicity.value, 3)], dtype=torch.long
        ).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor, ethnicity_idx)

        elapsed_ms = (time.time() - start_time) * 1000

        # Parse keypoints
        kps = outputs["keypoints"][0].cpu().numpy()  # (17, 2)
        keypoint_results = [
            KeypointResult(index=i, name=KEYPOINT_NAMES[i], x=float(kps[i, 0]), y=float(kps[i, 1]))
            for i in range(settings.NUM_KEYPOINTS)
        ]

        # Parse probabilities
        probs = outputs["probabilities"][0].cpu().numpy()
        deformity_results = [
            DeformityResult(
                name=cls,
                probability=float(probs[i]),
                detected=bool(probs[i] >= 0.5)
            )
            for i, cls in enumerate(settings.DEFORMITY_CLASSES)
        ]

        # Parse ratios
        raw = outputs["raw_ratios"][0].cpu().numpy()
        corr = outputs["corrected_ratios"][0].cpu().numpy()
        raw_ratios = RatioResult(THR=float(raw[0]), SHR=float(raw[1]), LBP=float(raw[2]), CLB=float(raw[3]))
        corr_ratios = RatioResult(THR=float(corr[0]), SHR=float(corr[1]), LBP=float(corr[2]), CLB=float(corr[3]))

        # Store analysis for report generation
        # Store analysis for report generation
        from cache import store_analysis
        store_analysis(analysis_id, {
            "deformities": [{"name": d.name, "probability": d.probability} for d in deformity_results],
            "corrected_ratios": {"THR": float(corr[0]), "SHR": float(corr[1]), "LBP": float(corr[2]), "CLB": float(corr[3])},
            "ethnicity": ethnicity.value,
            "patient_info": {"age": age, "sex": sex},
        })
        print(f"DEBUG - Stored analysis_id: {analysis_id}")

        return AnalysisResponse(
            analysis_id=analysis_id,
            image_url=f"/uploads/{analysis_id}{ext}",
            keypoints=keypoint_results,
            deformities=deformity_results,
            raw_ratios=raw_ratios,
            corrected_ratios=corr_ratios,
            ethnicity=ethnicity.value,
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
