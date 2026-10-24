"""
severity.py
-----------
Image-processing based severity analysis using HSV colour thresholding.

Workflow:
  1. Convert uploaded PIL image to HSV colour space (OpenCV).
  2. Apply disease-specific HSV colour masks to identify diseased pixels
     (brown/dark lesions, yellow/water-soaked areas, etc.).
  3. Apply morphological operations to clean up noise.
  4. Calculate diseased pixel % → map to Mild / Moderate / Severe.
  5. Generate a highlighted image (red overlay + contours) as a
     base64-encoded PNG string for direct embedding in the frontend.

Severity thresholds (configurable):
  < 15%  →  Mild
  15-40% →  Moderate
  > 40%  →  Severe
"""

import cv2
import numpy as np
import base64
from PIL import Image
from typing import Tuple

# ---------------------------------------------------------------------------
# HSV colour profiles per disease group
# Each profile = list of {lower, upper} ranges — masks are OR-combined.
# OpenCV HSV scale: H 0–180, S 0–255, V 0–255
# ---------------------------------------------------------------------------
_PROFILES: dict = {

    # ── Dark brown / black lesions ────────────────────────────────────────
    # Covers: Anthracnose (all fruits), Scab, Alternaria, Cercospora,
    #         Melanose, Stem-end Rot, Bacterial Blight, Styler-and-Root
    "brown_dark": [
        # warm brown tones (low value = dark lesion centres)
        {"lower": np.array([0,  40,  15]), "upper": np.array([20, 255, 160])},
        # red-brown wrap-around in OpenCV hue wheel
        {"lower": np.array([158, 30, 15]), "upper": np.array([180, 255, 160])},
    ],

    # ── Yellow / orange / water-soaked lesions ────────────────────────────
    # Covers: Citrus Canker (yellow halo + corky lesion)
    "yellow_orange": [
        {"lower": np.array([10,  70, 120]), "upper": np.array([38, 255, 255])},
    ],

    # ── Healthy — no diseased pixel profile ───────────────────────────────
    "healthy": [],
}

# Map raw class names → colour profile key
DISEASE_PROFILE: dict[str, str] = {
    "Avocado_Anthracnose":          "brown_dark",
    "Avocado_Scab":                 "brown_dark",
    "Guava_Anthracnose":            "brown_dark",
    "Guava_Phytophthora":           "brown_dark",
    "Guava_Scab":                   "brown_dark",
    "Guava_Styler_and_Root":        "brown_dark",
    "Mango_Alternaria":             "brown_dark",
    "Mango_Anthracnose":            "brown_dark",
    "Mango_Scab":                   "brown_dark",
    "Mango_Stem_end_Rot":           "brown_dark",
    "Orange_Citrus_Canker":         "yellow_orange",
    "Orange_Citrus_Melanose":       "brown_dark",
    "Pomegranate_Alternaria":       "brown_dark",
    "Pomegranate_Anthracnose":      "brown_dark",
    "Pomegranate_Bacterial_Blight": "brown_dark",
    "Pomegranate_Cercospora":       "brown_dark",
}

# ---------------------------------------------------------------------------
# Severity thresholds — tweak here if results feel off after testing
# ---------------------------------------------------------------------------
_MILD_THRESHOLD     = 15.0   # below this % → Mild
_MODERATE_THRESHOLD = 40.0   # below this % → Moderate; above → Severe


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_mask(hsv: np.ndarray, profile_key: str) -> np.ndarray:
    """
    Build a combined binary mask for all colour ranges in the given profile.
    Applies morphological open+close to remove noise and fill gaps.
    """
    ranges = _PROFILES.get(profile_key, [])
    if not ranges:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask |= cv2.inRange(hsv, r["lower"], r["upper"])

    # Remove specks (open) then fill small holes in lesions (close)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_severity(
    pil_image: Image.Image,
    class_name: str,
) -> Tuple[str, float, np.ndarray]:
    """
    Analyse a PIL image for diseased pixel percentage.

    Parameters
    ----------
    pil_image  : PIL.Image  (RGB)
    class_name : raw internal class label, e.g. "Mango_Anthracnose"

    Returns
    -------
    severity : str   — "Healthy" | "Mild" | "Moderate" | "Severe"
    pct      : float — diseased area as % of total image pixels
    mask     : np.ndarray uint8 — binary mask (255 = diseased pixel)
    """
    # Healthy fruits skip colour analysis
    if "Healthy" in class_name:
        h, w = np.array(pil_image).shape[:2]
        return "Healthy", 0.0, np.zeros((h, w), dtype=np.uint8)

    profile_key = DISEASE_PROFILE.get(class_name, "brown_dark")

    # PIL RGB → OpenCV BGR → HSV
    bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask = _build_mask(hsv, profile_key)

    total_px    = mask.size
    diseased_px = int(np.sum(mask > 0))
    pct         = round((diseased_px / total_px) * 100, 1)

    if pct < _MILD_THRESHOLD:
        severity = "Mild"
    elif pct < _MODERATE_THRESHOLD:
        severity = "Moderate"
    else:
        severity = "Severe"

    return severity, pct, mask


def highlight_disease(pil_image: Image.Image, mask: np.ndarray) -> str:
    """
    Overlay diseased regions on the original image with a red highlight
    and draw contour outlines around each lesion cluster.

    Parameters
    ----------
    pil_image : PIL.Image (any mode — converted to RGB internally)
    mask      : np.ndarray uint8 — binary mask from analyze_severity()

    Returns
    -------
    str : base64-encoded PNG string.
          To use in HTML: <img src="data:image/png;base64,{returned_string}">
          To use in JS:   img.src = `data:image/png;base64,${result.highlighted_image}`
    """
    img_rgb = np.array(pil_image.convert("RGB")).copy()

    # Semi-transparent red fill over diseased pixels
    overlay = img_rgb.copy()
    overlay[mask > 0] = [220, 50, 50]                          # vivid red
    blended = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0) # 45% red tint

    # Crisp contour lines around each diseased region boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 80, 80), thickness=2)

    # Encode RGB result → PNG bytes → base64 string
    _, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
