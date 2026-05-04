"""
SpineAI FastAPI backend entrypoint.
Run: uvicorn main:app --reload --port 8000
"""
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from config import settings, UPLOAD_DIR
from model.pipeline import SpineAIModel
from api.routes import inference, report

# Global model state
model = None
device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SpineAI] Loading model on {device}...")
    model = SpineAIModel().to(device)
    
    # Load weights if available
    import os
    weights_path = settings.MODEL_WEIGHTS_PATH
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[SpineAI] Weights loaded from {weights_path}")
    else:
        print(f"[SpineAI] No weights found at {weights_path} — running with random init (for dev)")
    
    model.eval()
    print("[SpineAI] Model ready.")
    yield
    print("[SpineAI] Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-powered spine deformity and posture detection API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Register routes
app.include_router(inference.router, prefix="/api/v1", tags=["Inference"])
app.include_router(report.router, prefix="/api/v1", tags=["Report"])


@app.get("/api/v1/health")
async def health():
    return {
        "status": "ok",
        "model": "SpineAI PostureNet",
        "version": settings.VERSION,
        "device": str(device),
        "classes": settings.DEFORMITY_CLASSES,
    }


@app.get("/")
async def root():
    return {"message": "SpineAI API — visit /docs for interactive API documentation"}

@app.get("/api/v1/debug/cache")
async def debug_cache():
    from cache import _analysis_cache
    return {"cache_keys": list(_analysis_cache.keys())}
