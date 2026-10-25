"""
app.py
------
FastAPI application that serves the fruit disease EfficientNet-B3 model
and the improved severity & lesion segmentation pipeline.

Endpoints:
    GET  /                  → Health check & status
    GET  /health            → Health check (Docker / Cloud / Spring Boot)
    GET  /version           → API & service version info
    GET  /classes           → Returns all 21 class names recognised by the model
    GET  /fruits            → Returns list of 5 supported fruit crop categories
    GET  /info              → Model metadata, image limits, and pipeline thresholds
    GET  /treatments        → Full 3-tier treatment guidance database
    GET  /treatments/{name} → Treatment advice for a specific disease class
    POST /predict           → Upload an image (multipart/form-data), receive prediction
    POST /predict/json      → Upload base64 image (JSON payload), receive prediction
    POST /explain           → Disease prediction with LayerCAM / Grad-CAM visual heatmap
    GET  /docs              → Auto-generated Swagger UI
"""

import io
import sys
import base64
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure backend directory is in sys.path regardless of where uvicorn is invoked from
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Import improved inference function and treatment data with dual import resilience
try:
    from model import predict, CLASS_NAMES, format_class_name
    from treatments import TREATMENTS
except ImportError:
    from backend.model import predict, CLASS_NAMES, format_class_name
    from backend.treatments import TREATMENTS

# ---------------------------------------------------------------------------
# App Definition
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fruit Disease Detection & Severity API",
    description=(
        "Upload a fruit image and receive disease classification, highly accurate "
        "severity estimation, and precise lesion segmentation overlays. "
        "Specifically optimized for Guava (Anthracnose, Phytophthora, Scab, Stylar rot) "
        "and Pomegranate Alternaria, while maintaining full accuracy across all 21 classes."
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration
# Supports standard origins and credentialed requests from Next.js (port 3000)
# and Spring Boot (port 8080)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Allowed MIME types and size constraints
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/pjpeg",
    "image/x-png",
    "image/tiff",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


# ---------------------------------------------------------------------------
# Pydantic Schemas for Base64 and JSON payloads
# ---------------------------------------------------------------------------
class Base64ImageRequest(BaseModel):
    image: Optional[str] = Field(
        None,
        description="Base64-encoded image string or data URL (e.g., 'data:image/jpeg;base64,...')",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
    )
    image_base64: Optional[str] = Field(
        None,
        description="Alternative key for base64-encoded image string",
    )


# ---------------------------------------------------------------------------
# Helper: Decode PIL Image from uploaded bytes or base64
# ---------------------------------------------------------------------------
def _load_image_from_bytes(data: bytes, filename: Optional[str] = None, content_type: Optional[str] = None) -> Image.Image:
    if not data or len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB.",
        )

    is_valid_type = (content_type in ALLOWED_TYPES) if content_type else False
    has_valid_ext = any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS) if filename else False

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        if not is_valid_type and not has_valid_ext:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported or unrecognized file type: '{content_type or 'unknown'}'. "
                    f"Please upload an image file: {', '.join(sorted(ALLOWED_TYPES))}"
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image file: {exc}",
        )

    return image


def _load_image_from_base64(b64_string: str) -> Image.Image:
    if not b64_string or not isinstance(b64_string, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'image' field must be a non-empty base64 string.",
        )

    clean_b64 = b64_string.strip()
    if "," in clean_b64 and clean_b64.startswith("data:image"):
        clean_b64 = clean_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(clean_b64)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 encoding: {exc}",
        )

    return _load_image_from_bytes(raw_bytes)


# ---------------------------------------------------------------------------
# Health & Status Endpoints
# ---------------------------------------------------------------------------
@app.get("/", summary="Root Health Check", tags=["Health & Status"])
@app.get("/health", summary="Service Health Check", tags=["Health & Status"])
@app.get("/healthz", summary="Container Liveness Check", tags=["Health & Status"])
@app.get("/api/health", summary="API Health Check", tags=["Health & Status"])
def health_check():
    """Confirms the API is operational, returning model name, classes, and version."""
    return {
        "status":  "ok",
        "message": "Fruit Disease Detection API is running.",
        "model":   "EfficientNet-B3",
        "classes": len(CLASS_NAMES),
        "version": "2.0.0",
    }


@app.get("/version", summary="Version Info", tags=["Health & Status"])
@app.get("/api/version", summary="API Version Info", tags=["Health & Status"])
def get_version():
    """Returns detailed semantic versioning and capability information."""
    return {
        "service": "FruitGuard ML Service",
        "version": "2.0.0",
        "features": [
            "EfficientNet-B3 image classification",
            "Multi-scale LayerCAM attention mapping",
            "Botanical color space lesion segmentation",
            "3-tier severity estimation (Mild, Moderate, Severe)",
            "Dynamic treatment recommendation",
            "Lesion boundary contours & base64 visualization",
        ],
    }


# ---------------------------------------------------------------------------
# Metadata & Knowledge Base Endpoints
# ---------------------------------------------------------------------------
@app.get("/classes", summary="List All 21 Recognisable Classes", tags=["Metadata"])
@app.get("/api/classes", summary="List Classes (API Route)", tags=["Metadata"])
def get_classes():
    """
    Returns the 21 fruit-disease classes that the model recognises,
    both grouped by fruit category and formatted for UI presentation.
    """
    formatted = [format_class_name(c) for c in CLASS_NAMES]
    fruits: Dict[str, list] = {}
    for raw, label in zip(CLASS_NAMES, formatted):
        fruit = raw.split("_")[0]
        fruits.setdefault(fruit, []).append(label)

    return {
        "classes":       formatted,
        "total_classes": len(CLASS_NAMES),
        "class_names":   formatted,
        "raw_classes":   CLASS_NAMES,
        "by_fruit":      fruits,
    }


@app.get("/fruits", summary="List Supported Fruit Types", tags=["Metadata"])
@app.get("/api/fruits", summary="List Fruit Types (API Route)", tags=["Metadata"])
def get_fruits():
    """Returns all 5 supported fruit crop categories."""
    unique_fruits = sorted(list(set(c.split("_")[0] for c in CLASS_NAMES)))
    return {
        "total_fruits": len(unique_fruits),
        "fruits": unique_fruits,
    }


@app.get("/info", summary="System Information & Constraints", tags=["Metadata"])
@app.get("/api/info", summary="System Information (API Route)", tags=["Metadata"])
def get_info():
    """Returns configuration details, supported image formats, and pipeline thresholds."""
    return {
        "model": "EfficientNet-B3",
        "input_resolution": "300x300",
        "severity_thresholds": {
            "mild_max_pct": 15.0,
            "moderate_max_pct": 40.0,
            "severe_min_pct": 40.0,
        },
        "supported_mime_types": sorted(list(ALLOWED_TYPES)),
        "max_upload_size_mb": MAX_FILE_SIZE // (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Treatment Database Endpoints
# ---------------------------------------------------------------------------
@app.get("/treatments", summary="Get Full Treatments Database", tags=["Treatments"])
@app.get("/api/treatments", summary="Get Full Treatments (API Route)", tags=["Treatments"])
def get_all_treatments():
    """Returns the entire 3-tier treatment guidance database for all 21 classes."""
    return {
        "total_classes": len(TREATMENTS),
        "treatments": TREATMENTS,
    }


@app.get("/treatments/{class_name}", summary="Get Treatment Advice for a Class", tags=["Treatments"])
@app.get("/api/treatments/{class_name}", summary="Get Treatment Advice (API Route)", tags=["Treatments"])
def get_treatment_for_class(class_name: str):
    """
    Retrieve treatment advice for a specific disease class.
    Accepts raw label (e.g., 'Mango_Anthracnose') or formatted label (e.g., 'Mango Anthracnose').
    """
    normalized_key = class_name.replace(" ", "_").strip()
    advice = TREATMENTS.get(normalized_key)

    if not advice:
        for k, v in TREATMENTS.items():
            if k.lower() == normalized_key.lower():
                advice = v
                normalized_key = k
                break

    if not advice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No treatment advice found for class '{class_name}'. Check /classes for valid names.",
        )

    return {
        "class_name": normalized_key,
        "formatted_name": format_class_name(normalized_key),
        "treatments": advice,
    }


# ---------------------------------------------------------------------------
# Core Prediction Endpoints
# ---------------------------------------------------------------------------
@app.post("/predict", summary="Predict Fruit Disease (Multipart Form)", tags=["Prediction"])
@app.post("/api/predict", summary="Predict Fruit Disease (API Route)", tags=["Prediction"])
async def predict_disease(file: UploadFile = File(..., description="Fruit image file (JPEG, PNG, WEBP, BMP)")):
    """
    Primary image analysis endpoint accepting multipart/form-data.

    Returns:
      - **result**: Human-readable predicted class (e.g., `"Mango Anthracnose"`)
      - **severity**: Infection level (`"Healthy"` | `"Mild"` | `"Moderate"` | `"Severe"`)
      - **diseased_area_pct**: Percentage (0.0 to 100.0) of fruit area infected
      - **treatment**: Severity-specific treatment recommendations
      - **highlighted_image**: Base64-encoded PNG with transparent red lesion overlay and green contours
      - **confidence**: Prediction confidence percentage (0.0 to 100.0)
      - **fruit**: Fruit category (`"Avocado"`, `"Guava"`, `"Mango"`, `"Orange"`, `"Pomegranate"`)
      - **raw_class**: Internal identifier for Spring Boot / DB mapping (`"Mango_Anthracnose"`)
      - **class_name**: Alias for raw_class
      - **heatmap_image**: Base64-encoded Grad-CAM attention heatmap (None if Healthy)
    """
    contents = await file.read()
    image = _load_image_from_bytes(contents, filename=file.filename, content_type=file.content_type)

    try:
        result = predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return JSONResponse(content=result)


@app.post("/predict/json", summary="Predict Fruit Disease (Base64 JSON)", tags=["Prediction"])
@app.post("/predict-base64", summary="Predict Fruit Disease (Base64 Alias)", tags=["Prediction"])
@app.post("/api/predict/json", summary="Predict Fruit Disease (JSON API Route)", tags=["Prediction"])
def predict_disease_base64(payload: Base64ImageRequest):
    """
    Alternative prediction endpoint accepting a JSON payload with a base64-encoded image string.
    Convenient for frontend webcam streams, HTML canvas exports, or mobile applications.
    """
    raw_b64 = payload.image or payload.image_base64
    if not raw_b64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'image' or 'image_base64' in JSON payload.",
        )
    image = _load_image_from_base64(raw_b64)

    try:
        result = predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Explainability / Grad-CAM Dedicated Endpoint
# ---------------------------------------------------------------------------
@app.post("/explain", summary="Generate Disease Prediction & Grad-CAM Heatmap", tags=["Explainability"])
@app.post("/api/explain", summary="Explain Prediction (API Route)", tags=["Explainability"])
async def explain_prediction(file: UploadFile = File(...)):
    """
    Runs model inference and returns the full diagnosis with explicit emphasis on the
    LayerCAM / Grad-CAM visual attention heatmap (`heatmap_image`) alongside the lesion mask.
    """
    contents = await file.read()
    image = _load_image_from_bytes(contents, filename=file.filename, content_type=file.content_type)

    try:
        result = predict(image)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )

    return JSONResponse(content=result)
