import os
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    APP_NAME: str = "SpineAI"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = "sqlite+aiosqlite:///./spineai.db"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_WEIGHTS_PATH: str = "model/weights/posturenet.pth"
    UPLOAD_DIR: str = "uploads"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    IMAGE_SIZE: int = 256
    NUM_KEYPOINTS: int = 17
    NUM_CLASSES: int = 7
    DEFORMITY_CLASSES: list[str] = [
        "Normal", "Scoliosis", "FHP",
        "Kyphosis", "Lordosis", "Pelvic Tilt", "Genu Valgum"
    ]
    ETHNICITIES: list[str] = [
        "East Asian", "South Asian", "Sub-Saharan African",
        "European", "Latin American", "Middle Eastern"
    ]
    SECRET_KEY: str = "SDP_project"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / settings.UPLOAD_DIR
UPLOAD_DIR.mkdir(exist_ok=True)
