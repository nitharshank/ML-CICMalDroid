"""
FastAPI server for Android malware detection.

Endpoints:
  GET  /health             liveness check
  GET  /features           returns the list of training features (debug)
  POST /predict            multipart upload of an .apk -> verdict JSON

Run locally:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from extractors.drebin_extractor import DrebinExtractor
from utils.model_loader import EnsembleModel

# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("malware-api")

# IMPORTANT: BASE_DIR points to the project ROOT (one level up from backend/),
# so it can reach the existing processed/ and models/ folders.
# If you copied processed/ and models/ INTO backend/, change this back to:
#     BASE_DIR = Path(__file__).parent.resolve()
BASE_DIR = Path(__file__).parent.parent.resolve()
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB ceiling

app = FastAPI(
    title="Android Malware Detection API",
    version="1.0.0",
    description="Static-analysis APK classifier using a Drebin-style "
                "ensemble (RF + XGBoost + MLP).",
)

# CORS open for development. Tighten for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Load models + extractor at startup
# --------------------------------------------------------------------------- #
ensemble = EnsembleModel(BASE_DIR)
extractor: Optional[DrebinExtractor] = None


@app.on_event("startup")
def _load_artifacts() -> None:
    global extractor
    log.info("Loading models from %s", BASE_DIR)
    ensemble.load()
    extractor = DrebinExtractor(ensemble.features)
    log.info("Ready. %d features.", len(ensemble.features))


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    feature_count: int


class ModelProbs(BaseModel):
    rf: float
    xgb: float
    mlp: float


class Probabilities(BaseModel):
    benign: float
    malware: float


class ExtractedInfo(BaseModel):
    package_name: Optional[str] = None
    app_name: Optional[str] = None
    version_name: Optional[str] = None
    min_sdk: Optional[str] = None
    target_sdk: Optional[str] = None
    permissions_declared: int = 0
    activities: int = 0
    services: int = 0
    receivers: int = 0
    providers: int = 0
    suspicious_api_hits: List[str] = []
    urls_found: int = 0
    ips_found: int = 0
    matched_features_count: int = 0
    matched_features: List[str] = []


class PredictResponse(BaseModel):
    verdict: str
    is_malware: bool
    confidence: float
    probabilities: Probabilities
    models: ModelProbs
    active_features: int
    extracted: ExtractedInfo
    elapsed_ms: int


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if ensemble.scaler is not None else "loading",
        feature_count=len(ensemble.features),
    )


@app.get("/features")
def features():
    return {"count": len(ensemble.features), "features": ensemble.features}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if extractor is None:
        raise HTTPException(503, "Models not yet loaded.")

    name = (file.filename or "").lower()
    if not name.endswith(".apk"):
        raise HTTPException(400, "Only .apk files are accepted.")

    # Stream the upload to a temp file with a size cap.
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".apk")
    written = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(413, "APK exceeds 500 MB limit.")
            tmp.write(chunk)
        tmp.close()

        log.info("Received %s (%.2f MB)", file.filename, written / 1e6)

        t0 = time.time()
        try:
            vec, meta = extractor.extract(tmp.name)
        except Exception as e:
            log.exception("Feature extraction failed")
            raise HTTPException(422, f"Could not parse APK: {e}")

        probs = ensemble.predict(vec)
        elapsed = int((time.time() - t0) * 1000)

        p_mal = probs["ensemble"]
        p_ben = 1.0 - p_mal
        is_mal = p_mal >= 0.5

        response = PredictResponse(
            verdict="MALWARE" if is_mal else "BENIGN",
            is_malware=is_mal,
            confidence=p_mal if is_mal else p_ben,
            probabilities=Probabilities(benign=p_ben, malware=p_mal),
            models=ModelProbs(rf=probs["rf"], xgb=probs["xgb"], mlp=probs["mlp"]),
            active_features=int(vec.sum()),
            extracted=ExtractedInfo(**meta),
            elapsed_ms=elapsed,
        )
        log.info("Prediction response: %s", response.model_dump())
        return response

    finally:
        try:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=300,  # 5 min keep-alive for large uploads on WiFi
    )
