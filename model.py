"""
model.py
--------
Loads the trained EfficientNet-B3 fruit disease model and serves inference
fused with improved severity analysis and lesion segmentation.

The model is loaded from:
    model_output/fruit_disease_efficientb3.pth

Class order matches the alphabetically-sorted class names produced by
torchvision.datasets.ImageFolder.

Severity pipeline:
    After classification, the original RGB image is analysed using botanical
    color spaces (HSV, LAB) and multi-scale LayerCAM spatial activation cues
    (severity.py) to estimate diseased pixel %, mapped to Mild / Moderate / Severe.
    Both a highlighted lesion mask and Grad-CAM attention heatmap are generated.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, cast

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

try:
    from treatments import TREATMENTS
    from severity import (
        analyze_severity,
        highlight_disease,
        generate_cam_heatmap,
        guava_scab_detector,
        guava_scab_overlay,
        is_scab_candidate_disease,
        guava_phytophthora_detector,
        phytophthora_overlay,
        is_phytophthora_disease,
        encode_mask_base64,
    )
except ImportError:
    from backend.treatments import TREATMENTS
    from backend.severity import (
        analyze_severity,
        highlight_disease,
        generate_cam_heatmap,
        guava_scab_detector,
        guava_scab_overlay,
        is_scab_candidate_disease,
        guava_phytophthora_detector,
        phytophthora_overlay,
        is_phytophthora_disease,
        encode_mask_base64,
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_output" / "fruit_disease_efficientb3.pth"

# ---------------------------------------------------------------------------
# Class names (21 classes) — must be alphabetically sorted
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
IMG_SIZE = 300
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_class_name(raw: str) -> str:
    """Convert internal underscore label to human-readable string."""
    return " ".join(word.capitalize() for word in raw.split("_"))


def _build_model() -> torch.nn.Module:
    """Recreate the EfficientNet-B3 architecture and load weights."""
    model = cast(Any, timm.create_model("efficientnet_b3", pretrained=False))
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "in_features"):
        raise RuntimeError("Expected EfficientNet-B3 classifier with in_features")
    in_features = int(getattr(classifier, "in_features"))
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
        if os.environ.get("FRUITGUARD_MOCK_MODEL", "").lower() in ("1", "true"):
            print(f"[model.py] WARNING: Weights file not found at {MODEL_PATH}. Initialized in mock mode (FRUITGUARD_MOCK_MODEL=1).")
            model.to(DEVICE)
            model.eval()
            return model
        raise FileNotFoundError(
            f"Model weights not found at: {MODEL_PATH}\n"
            "Make sure fruit_disease_efficientb3.pth is inside model_output/ (or backend/model_output/).\n"
            "To run the API for testing without weights, set environment variable FRUITGUARD_MOCK_MODEL=1."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"[model.py] Model loaded on {DEVICE} ({MODEL_PATH.name})")
    return model


_model: torch.nn.Module = _build_model()


def _extract_Class_Activation_Map(tensor: torch.Tensor, target_class: int) -> np.ndarray:
    """
    Compute a high-resolution LayerCAM spatial importance map from EfficientNet-B3's
    intermediate and final convolutional blocks to guide severity analysis.
    """
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    blocks = cast(List[nn.Module], getattr(_model, "blocks"))
    conv_head = cast(nn.Module, getattr(_model, "conv_head"))

    h1 = blocks[4].register_forward_hook(forward_hook)
    b1 = blocks[4].register_full_backward_hook(backward_hook)
    h2 = conv_head.register_forward_hook(forward_hook)
    b2 = conv_head.register_full_backward_hook(backward_hook)

    try:
        _model.zero_grad()
        logits = _model(tensor)
        score = logits[0, target_class]
        score.backward()

        grad_head, grad_mid = gradients[0], gradients[1]
        act_mid, act_head   = activations[0], activations[1]

        cam_mid = F.relu(grad_mid) * F.relu(act_mid)
        cam_mid = cam_mid.sum(dim=1, keepdim=True)
        cam_mid = F.interpolate(cam_mid, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        cam_head = F.relu(grad_head) * F.relu(act_head)
        cam_head = cam_head.sum(dim=1, keepdim=True)
        cam_head = F.interpolate(cam_head, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        def _norm_tensor(c: torch.Tensor) -> np.ndarray:
            c_np = c.squeeze().detach().cpu().numpy()
            c_min, c_max = c_np.min(), c_np.max()
            return (c_np - c_min) / (c_max - c_min + 1e-8)

        def _norm_np(c_np: np.ndarray) -> np.ndarray:
            c_min, c_max = c_np.min(), c_np.max()
            return (c_np - c_min) / (c_max - c_min + 1e-8)

        fused_cam = 0.55 * _norm_tensor(cam_mid) + 0.45 * _norm_tensor(cam_head)
        return _norm_np(fused_cam)
    finally:
        h1.remove()
        b1.remove()
        h2.remove()
        b2.remove()


def predict(image: Image.Image) -> Dict[str, Any]:
    """
    Run full inference pipeline on a PIL image:
      1. EfficientNet-B3 classification → disease class & confidence
      2. LayerCAM activation mapping    → spatial feature attention
      3. Hybrid severity analysis       → lesion segmentation & percentage
      4. Tiered treatment lookup        → specific remedy guidance
      5. Diagnostic visual overlays     → lesion contour & Grad-CAM heatmap

    Parameters
    ----------
    image : PIL.Image.Image
        The uploaded image (any size/mode — converted to RGB internally).

    Returns
    -------
    dict with keys:
        result             (str)   Human-readable class label (e.g. "Mango Anthracnose")
        severity           (str)   "Healthy" | "Mild" | "Moderate" | "Severe"
        diseased_area_pct  (float) % of fruit pixels identified as diseased
        treatment          (str)   Severity-specific treatment advice
        highlighted_image  (str)   base64-encoded PNG with lesion boundaries & fill
        confidence         (float) Prediction confidence percentage (0.0 - 100.0)
        fruit              (str)   Fruit category ("Avocado", "Guava", "Mango", "Orange", "Pomegranate")
        raw_class          (str)   Internal class identifier (e.g. "Mango_Anthracnose")
        class_name         (str)   Alias for raw_class
        heatmap_image      (str|None) base64-encoded PNG Grad-CAM attention heatmap (None if Healthy)
    """
    image = image.convert("RGB")
    transformed = cast(torch.Tensor, _transform(image))
    tensor = transformed.unsqueeze(0).to(DEVICE)

    # 1. Classification (under torch.no_grad for speed and zero autograd memory retention)
    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    best_idx   = int(probs.argmax())
    best_label = CLASS_NAMES[best_idx]
    confidence = round(float(probs[best_idx].item()) * 100.0, 2)

    # 2. Extract multi-scale LayerCAM if diseased
    cam_map = None
    if "Healthy" not in best_label:
        try:
            cam_tensor = tensor.clone().detach().requires_grad_(True)
            cam_map = _extract_Class_Activation_Map(cam_tensor, best_idx)
        except Exception as exc:
            print(f"[model.py] Warning: CAM extraction failed: {exc}")
            cam_map = None

    # 3. Hybrid Severity Analysis (Fruit Mask + Color Space + LayerCAM Fusion)
    severity, diseased_pct, mask, debug_masks = analyze_severity(
        image, best_label, cam_map=cam_map, return_debug_masks=True
    )

    # 4. Tiered treatment advice
    tier_advice = TREATMENTS.get(best_label, {})
    treatment   = tier_advice.get(
        severity,
        tier_advice.get("Healthy", "No specific treatment advice available.")
    )

    # 5. Generate clean highlighted image & Grad-CAM heatmap
    highlighted_b64 = highlight_disease(image, mask, class_name=best_label)
    heatmap_b64     = generate_cam_heatmap(image, cam_map)

    return {
        # Core fields matching previous API & frontend contract:
        "result":            format_class_name(best_label),
        "severity":          severity,
        "diseased_area_pct": diseased_pct,
        "treatment":         treatment,
        "highlighted_image": highlighted_b64,
        # Enhanced fields for comprehensive frontend & backend integration:
        "confidence":        confidence,
        "fruit":             best_label.split("_")[0],
        "raw_class":         best_label,
        "class_name":        best_label,
        "heatmap_image":     heatmap_b64,
        # Tri-mask validation outputs (Fruit Mask, Candidate Mask, Final Lesion Mask):
        "debug_masks": {
            "fruit_mask":     encode_mask_base64(debug_masks.get("fruit_mask")),
            "candidate_mask": encode_mask_base64(debug_masks.get("candidate_mask")),
            "final_mask":     encode_mask_base64(debug_masks.get("final_mask")),
        },
    }
