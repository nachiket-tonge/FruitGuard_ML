"""
model.py
--------
Loads the trained EfficientNet-B3 fruit disease model once at startup and
exposes a single `predict()` function for inference.

The model is loaded from:
    model_output/fruit_disease_efficientb3.pth

Class order must match the alphabetically-sorted class names produced by
torchvision.datasets.ImageFolder (same order as Unified_Dataset/class_weights.json).

Formatting:
    Internal class labels use underscores (e.g. "Mango_Stem_end_Rot").
    The public API returns human-readable labels (e.g. "Mango Stem End Rot").

Severity pipeline:
    After classification, the original RGB image is analysed using HSV colour
    thresholding (severity.py) to estimate diseased pixel %, mapped to
    Mild / Moderate / Severe.  A highlighted PNG is returned as base64.
"""

import os
from pathlib import Path
from typing import List, Dict, Union

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Treatment advice lookup (3-tier: Mild / Moderate / Severe / Healthy)
from treatments import TREATMENTS

# Severity analysis and image highlighting
from severity import analyze_severity, highlight_disease

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent          # .../backend/
MODEL_PATH = BASE_DIR / "model_output" / "fruit_disease_efficientb3.pth"

# ---------------------------------------------------------------------------
# Class names — must be alphabetically sorted (mirrors ImageFolder ordering)
# ---------------------------------------------------------------------------
CLASS_NAMES: List[str] = [
    "Avocado_Anthracnose",
    "Avocado_Healthy",
    "Avocado_Scab",
    "Guava_Anthracnose",
    "Guava_Healthy",
    "Guava_Phytophthora",
    "Guava_Scab",
    "Guava_Styler_and_Root",
    "Mango_Alternaria",
    "Mango_Anthracnose",
    "Mango_Healthy",
    "Mango_Scab",
    "Mango_Stem_end_Rot",
    "Orange_Citrus_Canker",
    "Orange_Citrus_Melanose",
    "Orange_Healthy",
    "Pomegranate_Alternaria",
    "Pomegranate_Anthracnose",
    "Pomegranate_Bacterial_Blight",
    "Pomegranate_Cercospora",
    "Pomegranate_Healthy",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Image pre-processing (must match validation transforms used during training)
# ---------------------------------------------------------------------------
IMG_SIZE = 300  # EfficientNet-B3 native size

MEAN = [0.485, 0.456, 0.406]  # ImageNet statistics
STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Label formatting
# ---------------------------------------------------------------------------
def format_class_name(raw: str) -> str:
    """
    Convert internal underscore label to a human-readable string.

    Examples
    --------
    "Mango_Anthracnose"    → "Mango Anthracnose"
    "Mango_Stem_end_Rot"   → "Mango Stem End Rot"
    "Guava_Styler_and_Root"→ "Guava Styler And Root"
    """
    return " ".join(word.capitalize() for word in raw.split("_"))

import timm

# ---------------------------------------------------------------------------
# Model — built once at module import time (i.e., once at server startup)
# ---------------------------------------------------------------------------
def _build_model() -> torch.nn.Module:
    """Recreate the EfficientNet-B3 head using timm and load saved weights."""
    model = timm.create_model("efficientnet_b3", pretrained=False)

    # Replace classifier head to match the custom trained head
    import torch.nn as nn
    in_features = model.classifier.in_features  # 1536 for B3 in timm
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}\n"
            "Make sure fruit_disease_efficientb3.pth is inside backend/model_output/"
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"[model.py] Model loaded on {DEVICE}  ({MODEL_PATH.name})")
    return model


# Load model at import time so the first request is fast
_model: torch.nn.Module = _build_model()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict(image: Image.Image) -> Dict:
    """
    Run full inference pipeline on a PIL image:
      1. EfficientNet-B3 classification → disease class
      2. HSV colour analysis            → severity (Mild/Moderate/Severe)
      3. Tiered treatment lookup        → specific advice
      4. Region highlighting            → base64 PNG with red overlay

    Preprocessing (matches val_transform used during training):
        Resize(300×300) → ToTensor → Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded image (any size/mode — converted to RGB internally).

    Returns
    -------
    dict with keys:
        result             (str)   Human-readable class label
        severity           (str)   "Healthy" | "Mild" | "Moderate" | "Severe"
        diseased_area_pct  (float) % of image pixels identified as diseased
        treatment          (str)   Severity-specific treatment advice
        highlighted_image  (str)   base64-encoded PNG with diseased regions
                                   highlighted in red; embed as:
                                   <img src="data:image/png;base64,{value}">
    """
    # ── Step 0: ensure RGB ────────────────────────────────────────────────
    image = image.convert("RGB")

    # ── Step 1: EfficientNet classification ───────────────────────────────
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)               # (1, NUM_CLASSES)
        probs  = F.softmax(logits, dim=1)[0]  # (NUM_CLASSES,)

    best_idx   = int(probs.argmax())
    best_label = CLASS_NAMES[best_idx]        # e.g. "Mango_Anthracnose"

    # ── Step 2: severity analysis (HSV thresholding on original image) ────
    severity, diseased_pct, mask = analyze_severity(image, best_label)

    # ── Step 3: tiered treatment advice ───────────────────────────────────
    tier_advice = TREATMENTS.get(best_label, {})
    treatment   = tier_advice.get(
        severity,
        tier_advice.get("Healthy", "No specific treatment advice available.")
    )

    # ── Step 4: generate highlighted image ────────────────────────────────
    highlighted_b64 = highlight_disease(image, mask)

    return {
        "result":            format_class_name(best_label),
        "severity":          severity,
        "diseased_area_pct": diseased_pct,
        "treatment":         treatment,
        "highlighted_image": highlighted_b64,
    }
