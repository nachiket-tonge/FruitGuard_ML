"""
app.py
------
FastAPI application that serves the fruit disease EfficientNet-B3 model
as a REST API.

Endpoints:
    GET  /           → Health check
    GET  /classes    → Returns all 21 class names the model recognises
    POST /predict    → Upload an image, receive a JSON prediction
    GET  /docs       → Auto-generated Swagger UI (for testing without a frontend)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Import inference function (model loads once here at startup)
from model import predict, CLASS_NAMES, format_class_name

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fruit Disease Detection API",
    description=(
        "Upload a fruit image and receive a disease classification "
        "powered by a fine-tuned EfficientNet-B3 model trained on 21 classes "
        "across 5 fruit types (Avocado, Guava, Mango, Orange, Pomegranate)."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow requests from ANY origin so any website can call this API.
# For production, replace allow_origins=["*"] with your website's domain,
# e.g. allow_origins=["https://yourwebsite.com"]
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # <-- change to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Allowed image MIME types
# ---------------------------------------------------------------------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Health check")
def root():
    """Quick health check — confirms the API is running."""
    return {
        "status":  "ok",
        "message": "Fruit Disease Detection API is running.",
        "model":   "EfficientNet-B3",
        "classes": len(CLASS_NAMES),
    }


@app.get("/classes", summary="List all recognisable classes")
def get_classes():
    """
    Returns the 21 fruit-disease class names that the model can predict,
    both in raw internal format and human-readable format.
    """
    formatted = [format_class_name(c) for c in CLASS_NAMES]

    fruits = {}
    for raw, label in zip(CLASS_NAMES, formatted):
        fruit = raw.split("_")[0]
        fruits.setdefault(fruit, []).append(label)

    return {
        "total_classes": len(CLASS_NAMES),
        "class_names":   formatted,
        "by_fruit":      fruits,
    }


@app.post("/predict", summary="Predict fruit disease from an image")
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a fruit image (JPEG / PNG / WEBP / BMP) and receive:

    - **result**: Human-readable predicted class name, e.g. `"Mango Anthracnose"`
    - **severity**: Infection level — `"Healthy"` | `"Mild"` | `"Moderate"` | `"Severe"`
    - **diseased_area_pct**: Estimated % of image pixels identified as diseased tissue
    - **treatment**: Severity-specific treatment or care advice
    - **highlighted_image**: Base64-encoded PNG with diseased regions highlighted in red.
      Embed directly in HTML: `<img src="data:image/png;base64,{value}">`
      Or in JS: `img.src = "data:image/png;base64," + result.highlighted_image`

    Preprocessing applied (identical to the val/test transform used during training):
    `Resize(300×300) → ToTensor → Normalize(ImageNet mean/std)`
    """
    # ── Validate content type ────────────────────────────────────────────
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Please upload one of: {', '.join(ALLOWED_TYPES)}"
            ),
        )

    # ── Read and decode image ────────────────────────────────────────────
    try:
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image file: {exc}",
        )

    # ── Run inference ─────────────────────────────────────────────────────
    try:
        result = predict(image)   # returns a dict with result, severity, treatment, highlighted_image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    # ── Return result ─────────────────────────────────────────────────────
    # predict() already returns a fully-structured dict — return it directly.
    return JSONResponse(content=result)
