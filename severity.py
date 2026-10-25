"""
severity.py
-----------
Enhanced severity estimation and disease region segmentation for fruit images.
Specifically calibrated for Guava (Anthracnose, Phytophthora, Scab, Stylar and Root Rot)
and Pomegranate Alternaria, while strictly preserving reliable performance on Avocado,
Mango, Orange, and other Pomegranate diseases.

Key Calibrations:
1. Full-Fruit & Close-Up Fruit Segmentation:
   - Eliminates over-erosion on close-up and macro fruit photographs.
   - Accurately captures fruit silhouettes against natural orchard foliage and dark backgrounds.
2. Complete-Lesion Guava Segmentation:
   - Uses botanical color spaces (HSV, LAB) and Laplacian surface roughness to capture:
     * Full-surface scabby reticulation and pustules (Guava Scab, Stylar/Root rot).
     * Complete necrotic brown/orange rot and water-soaked patches (Guava Phytophthora, Anthracnose).
     * Distinguishes healthy smooth green/yellow skin from corky and necrotic tissue.
3. True Pomegranate Alternaria Detection:
   - Accurately captures black charcoal rot and sunken dull brown necrotic blotches.
   - Protects healthy red/pink/orange rind without false suppression.
4. Non-Destructive CAM Fusion:
   - Uses LayerCAM as an additive confidence enhancer for subtle lesion boundaries.
   - Eliminates destructive peripheral suppression that previously clamped lesions to a central blob.
"""

import cv2
import numpy as np
import base64
from PIL import Image
from typing import Tuple, Optional
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

_MILD_THRESHOLD = 15.0
_MODERATE_THRESHOLD = 40.0

# Backward compatibility with severity.py profile mapping
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


def _strip_augmentation_borders(gray: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Strip black rotation corners and stretched augmentation streaks."""
    h, w = gray.shape
    valid = np.ones((h, w), dtype=np.uint8) * 255

    # 1. Dark corners
    dark_corners = (gray < 22).astype(np.uint8) * 255
    ff_dark = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for cx, cy in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if dark_corners[cy, cx] == 255:
            cv2.floodFill(dark_corners, ff_dark, (cx, cy), 128)
    valid[dark_corners == 128] = 0

    # 2. Structure Tensor Coherence for 1D stretched augmentation streaks
    g_float = gray.astype(np.float32)
    gx = cv2.Sobel(g_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_float, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 4.0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 4.0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 4.0)

    trace = jxx + jyy + 1e-5
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2))
    coherence = (diff / trace) ** 2
    grad_energy = jxx + jyy

    is_streak = (coherence > 0.72) & (grad_energy > 15.0)

    abs_gx = cv2.GaussianBlur(np.abs(gx), (9, 9), 0)
    abs_gy = cv2.GaussianBlur(np.abs(gy), (9, 9), 0)
    ortho_streak = ((abs_gx < 2.8) & (abs_gy > 6.5)) | ((abs_gy < 2.8) & (abs_gx > 6.5))

    streak_raw = ((is_streak | ortho_streak).astype(np.uint8)) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(streak_raw, connectivity=8)
    border_streaks = np.zeros((h, w), dtype=np.uint8)
    for lbl in range(1, num_labels):
        x, y, bw, bh, area = stats[lbl]
        if area < 80:
            continue
        touches_edge = (x <= 6) or (y <= 6) or (x + bw >= w - 6) or (y + bh >= h - 6)
        in_center = (x < w * 0.35 < x + bw) and (y < h * 0.35 < y + bh) and (area > 0.35 * h * w)
        if touches_edge and not in_center:
            border_streaks[labels == lbl] = 255

    border_streaks = cv2.dilate(
        border_streaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1
    )
    valid[border_streaks > 0] = 0
    return valid


def _extract_fruit_mask(pil_image: Image.Image, bgr: np.ndarray, class_name: str) -> np.ndarray:
    """
    Extracts an accurate binary fruit silhouette mask (255 = fruit, 0 = background).
    Correctly segments both isolated fruits and close-up/macro images filling the frame.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    valid_canvas = _strip_augmentation_borders(gray, bgr)
    fruit_type = class_name.split("_")[0]

    # Sample corners to determine background profile
    c_sz = max(6, int(min(h, w) * 0.05))
    corners_bgr = np.concatenate([
        bgr[:c_sz, :c_sz].reshape(-1, 3),
        bgr[:c_sz, -c_sz:].reshape(-1, 3),
        bgr[-c_sz:, :c_sz].reshape(-1, 3),
        bgr[-c_sz:, -c_sz:].reshape(-1, 3)
    ])
    corners_hsv = cv2.cvtColor(corners_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    c_med_val = float(np.median(corners_hsv[:, 2]))
    c_med_sat = float(np.median(corners_hsv[:, 1]))

    # Studio white background detection
    studio_white = (hsv[:, :, 2] > 235) & (hsv[:, :, 1] < 28)
    valid_canvas[studio_white] = 0

    border_sat = np.concatenate([
        hsv[0:6, :, 1].ravel(), hsv[h - 6:h, :, 1].ravel(),
        hsv[:, 0:6, 1].ravel(), hsv[:, w - 6:w, 1].ravel()
    ])
    border_val = np.concatenate([
        hsv[0:6, :, 2].ravel(), hsv[h - 6:h, :, 2].ravel(),
        hsv[:, 0:6, 2].ravel(), hsv[:, w - 6:w, 2].ravel()
    ])

    # If borders are predominantly bright white paper/studio backdrop
    if np.mean((border_val > 185) & (border_sat < 55)) > 0.40:
        bg_light = ((hsv[:, :, 2] > 180) & (hsv[:, :, 1] < 55)).astype(np.uint8) * 255
        ff_light = np.zeros((h + 2, w + 2), dtype=np.uint8)
        step = max(1, w // 16)
        for x in range(0, w, step):
            if bg_light[0, x] == 255:
                cv2.floodFill(bg_light, ff_light, (x, 0), 128)
            if bg_light[h - 1, x] == 255:
                cv2.floodFill(bg_light, ff_light, (x, h - 1), 128)
        step_y = max(1, h // 16)
        for y in range(0, h, step_y):
            if bg_light[y, 0] == 255:
                cv2.floodFill(bg_light, ff_light, (0, y), 128)
            if bg_light[y, w - 1] == 255:
                cv2.floodFill(bg_light, ff_light, (w - 1, y), 128)
        fruit_mask = (bg_light != 128).astype(np.uint8) * 255
        fruit_mask = cv2.bitwise_and(fruit_mask, valid_canvas)
    else:
        # Orchard / field GrabCut segmentation:
        # Isolate the fruit from foliage, branches, soil, and orchard background.
        # Note: We intentionally do NOT globally zero out reddish-brown tones
        # so that disease lesions remain intact within the fruit silhouette.
        if fruit_type == "Pomegranate":
            pale_corner = ((hsv[:, :, 1] < 58) & (hsv[:, :, 2] > 150) & (lab[:, :, 1] < 138)).astype(np.uint8) * 255
            ff_pale = np.zeros((h + 2, w + 2), dtype=np.uint8)
            for cx_c, cy_c in [(4, 4), (w - 5, 4), (4, h - 5), (w - 5, h - 5)]:
                if pale_corner[cy_c, cx_c] == 255:
                    cv2.floodFill(pale_corner, ff_pale, (cx_c, cy_c), 128)
            valid_canvas[pale_corner == 128] = 0

        # Downscale for GrabCut
        scale = 160.0 / float(max(h, w))
        sh, sw = max(32, int(h * scale)), max(32, int(w * scale))
        small_bgr = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        small_valid = cv2.resize(valid_canvas, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        small_lab = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2LAB)

        gc_mask = np.full((sh, sw), cv2.GC_PR_BGD, dtype=np.uint8)
        rim_y = max(2, int(sh * 0.04))
        rim_x = max(2, int(sw * 0.04))
        gc_mask[:rim_y, :] = cv2.GC_BGD
        gc_mask[sh - rim_y:, :] = cv2.GC_BGD
        gc_mask[:, :rim_x] = cv2.GC_BGD
        gc_mask[:, sw - rim_x:] = cv2.GC_BGD
        gc_mask[small_valid == 0] = cv2.GC_BGD

        # Dark or studio background prior on borders
        if c_med_val < 60:
            gc_mask[small_hsv[:, :, 2] < 45] = cv2.GC_BGD
        elif c_med_val > 185 and c_med_sat < 55:
            gc_mask[(small_hsv[:, :, 2] > 185) & (small_hsv[:, :, 1] < 55)] = cv2.GC_BGD

        if fruit_type == "Pomegranate":
            foliage = (small_hsv[:, :, 0] >= 28) & (small_hsv[:, :, 0] <= 95) & (small_lab[:, :, 1] < 125)
            gc_mask[foliage] = cv2.GC_BGD

        # Foreground initialization in center (covers the central fruit area)
        cx, cy = sw // 2, sh // 2
        rx, ry = int(sw * 0.44), int(sh * 0.44)
        cv2.ellipse(gc_mask, (cx, cy), (rx, ry), 0, 0, 360, cv2.GC_PR_FGD, -1)
        cv2.ellipse(gc_mask, (cx, cy), (int(rx * 0.50), int(ry * 0.50)), 0, 0, 360, cv2.GC_FGD, -1)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small_bgr, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
            small_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            fruit_mask = cv2.resize(small_fg, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            fruit_mask = valid_canvas.copy()

        fruit_mask = cv2.bitwise_and(fruit_mask, valid_canvas)

    # Extract dominant single connected fruit contour and FILL interior completely
    # Filling ensures internal necrotic lesions and spots are not hollowed out
    contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.04 * (h * w):
            clean_fruit = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(clean_fruit, [largest], -1, 255, thickness=cv2.FILLED)
            fruit_mask = clean_fruit
        else:
            fruit_mask = valid_canvas.copy()

    fruit_area = int(np.count_nonzero(fruit_mask))
    equiv_radius = max(10.0, np.sqrt(fruit_area / np.pi))

    open_mult = 0.08 if fruit_type == "Pomegranate" else 0.04
    open_k = max(3, int(equiv_radius * open_mult) | 1)
    opened = cv2.morphologyEx(
        fruit_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
    )
    if np.count_nonzero(opened) > 0.65 * fruit_area:
        fruit_mask = opened

    return fruit_mask


# ---------------------------------------------------------------------------
# Scab Candidate Detector & Overlay Logic
# Applicable ONLY to:
# 1. Guava Scab
# 2. Guava Anthracnose
# 3. Pomegranate Cercospora
# Note: Guava Stylar and Root Rot moved to ILLUMINATION_AWARE_DISEASES
# ---------------------------------------------------------------------------
SCAB_CANDIDATE_DISEASES = {
    "guava_scab",
    "guava_anthracnose",
    "pomegranate_cercospora",
    "cercospora",
}


def is_scab_candidate_disease(class_name: Optional[str]) -> bool:
    """Check if class_name matches one of the scab candidate disease classes."""
    if not class_name:
        return False
    norm = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    return norm in SCAB_CANDIDATE_DISEASES


# ---------------------------------------------------------------------------
# Illumination-Aware Candidate Detector
# Applicable ONLY to:
# 1. Pomegranate Anthracnose
# 2. Guava Phytophthora
# 3. Guava Stylar and Root Rot  (all spelling variants)
# ---------------------------------------------------------------------------
ILLUMINATION_AWARE_DISEASES = {
    "pomegranate_anthracnose",
    "guava_phytophthora",
    "phytophthora",
    "guava_styler_and_root",
    "guava_stylar_and_root",
    "guava_styler_and_rot",
    "guava_stylar_and_rot",
    "guava_styler_end_rot",
    "guava_stylar_end_rot",
    "guava_stylar_end_root",
    "guava_styler_end_root",
}


def is_illumination_aware_disease(class_name: Optional[str]) -> bool:
    """
    Check if class_name should be processed by the illumination-aware
    candidate detector instead of the standard pipeline.
    Covers: Pomegranate Anthracnose, Guava Phytophthora, Guava Stylar and Root.
    """
    if not class_name:
        return False
    norm = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    return norm in ILLUMINATION_AWARE_DISEASES


def guava_scab_detector(rgb, fruit_mask, return_components: bool = False):
    """
    Selective guava-scab & necrotic spot candidate detector.
    Applicable to: Guava Scab, Guava Anthracnose, Guava Stylar and Root, Pomegranate Cercospora.

    Combines:
    - Exclusion of healthy green fruit skin
    - Brown/reddish necrotic pigmentation
    - Strong darkness contrast relative to surrounding fruit skin
    - Local surface texture and boundary edge gradient
    - Connected-component filtering preserving tiny real scab spots

    Returns:
        final_mask (bool ndarray), or if return_components=True: (final_mask, candidate_mask, interior_mask)
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    roi = fruit_mask.astype(bool)

    if roi.sum() < 100:
        if return_components:
            return np.zeros_like(roi), np.zeros_like(roi), np.zeros_like(roi)
        return np.zeros_like(roi)

    # 1. Exclude a narrow boundary (0.8%) to reduce edge background artifacts.
    h, w = roi.shape
    radius = max(1, int(min(h, w) * 0.008))
    ksize = 2 * radius + 1

    interior = cv2.erode(
        roi.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    ).astype(bool)

    if interior.sum() < 0.5 * roi.sum():
        interior = roi.copy()

    # 2. Color spaces: HSV, LAB, Gray
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)

    # Healthy green skin rejection:
    # Green guava skin has Hue 33..85, negative a* in LAB (A < 127), and moderate/high L
    is_healthy_green = (H >= 33) & (H <= 85) & (A < 127) & (L > 60)

    # 3. Genuine necrotic / scab / anthracnose / cercospora pigmentation
    # Brown/reddish in HSV: H <= 30 or H >= 168 with moderate saturation
    brown_hsv = ((H <= 30) | (H >= 168)) & (S >= 28) & (V <= 220)
    # Brown/rust in LAB: red shift (A >= 130) and yellow (B >= 122) with L <= 210
    brown_lab = (A >= 130) & (B >= 122) & (L <= 210)
    # Deep black/charcoal necrosis (anthracnose rot center / dark cercospora spot)
    deep_dark_necrosis = (L < 80) & (V < 85) & (S < 65)

    lesion_color = (brown_hsv | brown_lab | deep_dark_necrosis) & (~is_healthy_green)

    # 4. Compare brightness with surrounding fruit skin (local contrast)
    local_brightness = cv2.GaussianBlur(L, (0, 0), sigmaX=9)
    local_darkness = local_brightness - L

    # Adaptive contrast floor: must be darker than local skin
    dark_threshold = max(7.0, float(np.percentile(local_darkness[interior], 78)))
    dark_contrast = local_darkness > dark_threshold

    # 5. Measure local texture and edge gradient
    local_gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    residual = gray - local_gray
    local_texture = cv2.GaussianBlur(residual ** 2, (0, 0), sigmaX=3) ** 0.5
    texture_threshold = max(3.0, float(np.percentile(local_texture[interior], 70)))
    textured = local_texture > texture_threshold

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    has_edge = grad > 5.0

    # 6. Dual confirmation:
    # A pixel must have genuine lesion color AND local darkness AND (texture or edge)
    # This prevents natural skin pores, mottling, and smooth shadow gradients from triggering
    brown_candidate = (brown_hsv | brown_lab) & dark_contrast & (textured | has_edge)
    dark_candidate = deep_dark_necrosis & dark_contrast & (textured | has_edge)

    candidate = interior & (brown_candidate | dark_candidate)

    # 7. Clean noise while preserving small scab spots
    mask = candidate.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Retain small components because scab can appear as tiny speckles
    min_area = max(4, int(interior.sum() * 0.00003))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    final_mask = cleaned.astype(bool) & interior

    if return_components:
        return final_mask, candidate, interior
    return final_mask


def guava_scab_overlay(rgb, mask, alpha=0.48):
    """Overlay detected scab candidates in red."""
    out = rgb.astype(np.float32).copy()
    selected = mask.astype(bool)

    red = np.array([255, 0, 0], dtype=np.float32)

    out[selected] = (
        (1 - alpha) * out[selected] + alpha * red
    )

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Guava Phytophthora Candidate Detector & Overlay Logic
# Applicable ONLY to:
# - Guava Phytophthora (Guava_Phytophthora)
# ---------------------------------------------------------------------------
def is_phytophthora_disease(class_name: Optional[str]) -> bool:
    """Check if class_name matches Guava Phytophthora."""
    if not class_name:
        return False
    norm = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    return norm in ("guava_phytophthora", "phytophthora")


def guava_phytophthora_detector(rgb, fruit_mask, return_components: bool = False):
    """
    Selective guava Phytophthora candidate detector.
    Anchored to genuine necrotic water-soaked rot and pale mycelium contrast.

    Returns:
        final_mask, dark_rot_mask, pale_region_mask
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    roi = fruit_mask.astype(bool)

    if roi.sum() < 100:
        if return_components:
            return np.zeros_like(roi), np.zeros_like(roi), np.zeros_like(roi)
        return np.zeros_like(roi), np.zeros_like(roi), np.zeros_like(roi)

    # 1. Fruit interior: suppress boundary artifacts
    h, w = roi.shape
    radius = max(1, int(min(h, w) * 0.008))
    ksize = 2 * radius + 1

    interior = cv2.erode(
        roi.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    ).astype(bool)

    if interior.sum() < 0.5 * roi.sum():
        interior = roi.copy()

    # 2. Color and texture features
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    # Healthy green exclusion
    is_healthy_green = (H >= 33) & (H <= 85) & (A < 127) & (L > 60)

    # Distance from neutral grey in LAB chromatic channels
    neutral_distance = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    # Local color and lightness contrast
    local_A = cv2.GaussianBlur(A, (0, 0), 9)
    local_B = cv2.GaussianBlur(B, (0, 0), 9)
    local_L = cv2.GaussianBlur(L, (0, 0), 9)

    color_difference = np.sqrt((A - local_A) ** 2 + (B - local_B) ** 2)
    local_darkness = local_L - L

    # Surface roughness
    fine_background = cv2.GaussianBlur(gray, (0, 0), 1.2)
    fine_texture = cv2.GaussianBlur((gray - fine_background) ** 2, (0, 0), 2.0) ** 0.5

    coarse_background = cv2.GaussianBlur(gray, (0, 0), 4.0)
    coarse_texture = cv2.GaussianBlur((gray - coarse_background) ** 2, (0, 0), 3.0) ** 0.5

    # 3. DARK ROT CANDIDATES (Anchored to true water-soaked necrosis)
    # Must have absolute contrast floor (> 8.0 units darker than local skin)
    darkness_threshold = max(8.0, float(np.percentile(local_darkness[interior], 75)))
    dark_contrast = local_darkness > darkness_threshold

    brown_rot = (A > 130) & (B > 120) & (L < 130)
    deep_rot = (L < 85) & (V < 90)

    texture_threshold = max(2.8, float(np.percentile(coarse_texture[interior], 65)))
    textured = coarse_texture > texture_threshold

    dark_rot = interior & (
        (brown_rot | deep_rot) &
        dark_contrast &
        textured &
        (~is_healthy_green)
    )

    # 4. PALE/GREY WATER-SOAKED OR MYCELIUM REGIONS
    # True pale rot has low chromatic saturation (neutral) AND differs from local skin
    pale_neutral = (neutral_distance < 20.0) & (S < 45)
    pale_brightness = (L > 95.0) & (L < 225.0)  # Avoid pure specular glare
    color_diff_threshold = max(7.0, float(np.percentile(color_difference[interior], 65)))
    pale_color_diff = color_difference > color_diff_threshold
    pale_textured = (fine_texture > 2.5) | (coarse_texture > 3.0)

    # Reject smooth, extremely bright reflections
    reflection_like = (V > 245) & (fine_texture < 2.0)

    pale_region = (
        interior &
        pale_neutral &
        pale_brightness &
        pale_color_diff &
        pale_textured &
        (~is_healthy_green) &
        (~reflection_like)
    )

    # 5. Clean masks without filling large healthy areas
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dark_u8 = cv2.morphologyEx(dark_rot.astype(np.uint8), cv2.MORPH_CLOSE, kernel_medium)
    pale_u8 = cv2.morphologyEx(pale_region.astype(np.uint8), cv2.MORPH_CLOSE, kernel_small)

    min_area = max(4, int(interior.sum() * 0.00003))

    def remove_small_components(mask):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        return cleaned.astype(bool)

    dark_rot_mask = (
        remove_small_components(dark_u8) & interior
    )

    pale_region_mask = (
        remove_small_components(pale_u8) & interior
    )

    # Combined candidate mask
    final_mask = dark_rot_mask | pale_region_mask

    return final_mask, dark_rot_mask, pale_region_mask


def phytophthora_overlay(rgb, mask, color=(255, 0, 0), alpha=0.45):
    """Overlay detected Phytophthora candidates in red/custom color."""
    out = rgb.astype(np.float32).copy()
    selected = mask.astype(bool)
    color = np.array(color, dtype=np.float32)

    out[selected] = (
        (1 - alpha) * out[selected] +
        alpha * color
    )

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Illumination-Aware Candidate Detector Helpers
# Shared utilities: clean_mask, make_overlay, illumination_aware_candidates
# ---------------------------------------------------------------------------

def clean_mask(mask, min_area: int = 20) -> np.ndarray:
    """
    Remove tiny connected components and retain larger regions.

    Args:
        mask: Boolean or binary image.
        min_area: Minimum connected-component area in pixels.

    Returns:
        Boolean mask.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    return cleaned.astype(bool)


def make_overlay(img, disease_mask, alpha: float = 0.48) -> np.ndarray:
    """
    Overlay candidate disease regions in red.

    Args:
        img: RGB image, uint8.
        disease_mask: Boolean mask.
        alpha: Overlay opacity.

    Returns:
        RGB overlay image, uint8.
    """
    img = np.asarray(img, dtype=np.uint8)
    mask = np.asarray(disease_mask).astype(bool)

    if mask.shape != img.shape[:2]:
        raise ValueError("Disease mask dimensions must match image.")

    overlay = img.copy().astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * red
    return np.clip(overlay, 0, 255).astype(np.uint8)


def illumination_aware_candidates(img, f_mask) -> np.ndarray:
    """
    Detect suspicious fruit-surface regions using illumination-normalized
    brightness, local color contrast, and multi-scale texture.

    Applicable to:
        - Pomegranate Anthracnose
        - Guava Phytophthora
        - Guava Stylar and Root Rot

    Args:
        img: RGB image, uint8, shape (H, W, 3).
        f_mask: Boolean fruit mask, shape (H, W).

    Returns:
        Boolean candidate disease mask, shape (H, W).

    Notes:
        This produces suspicious-region candidates, not confirmed lesions.
        Thresholds are adaptive to the pixels inside the fruit mask.
    """
    img = np.asarray(img, dtype=np.uint8)
    fruit = np.asarray(f_mask).astype(bool)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be an RGB image with 3 channels.")

    h, w = img.shape[:2]

    if fruit.shape != (h, w):
        raise ValueError("f_mask must match image dimensions.")

    if np.count_nonzero(fruit) < 100:
        raise ValueError("Fruit mask is empty or too small.")

    # 1. Reduce boundary leakage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(fruit.astype(np.uint8), kernel, iterations=1).astype(bool)
    if interior.sum() < 0.7 * fruit.sum():
        interior = fruit.copy()

    # 2. Color spaces
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)

    L, A, B = [c.astype(np.float32) for c in cv2.split(lab)]
    H, S, V = [c.astype(np.float32) for c in cv2.split(hsv)]

    # 3. Illumination normalization:
    # Estimate slowly-varying illumination from the fruit interior only,
    # dividing by a blurred mask so background pixels don't bleed in.
    sigma = max(5.0, min(h, w) * 0.035)
    fruit_float = interior.astype(np.float32)
    local_weight = cv2.GaussianBlur(fruit_float, (0, 0), sigmaX=sigma)
    local_light = cv2.GaussianBlur(L * fruit_float, (0, 0), sigmaX=sigma) / (local_weight + 1e-5)

    normalized_L = np.clip(L - local_light + np.median(L[interior]), 0, 255)
    local_darkness = local_light - L

    # 4. Local color anomaly
    sigma_color = max(2.0, min(h, w) * 0.012)
    local_A = cv2.GaussianBlur(A, (0, 0), sigmaX=sigma_color)
    local_B = cv2.GaussianBlur(B, (0, 0), sigmaX=sigma_color)
    color_anomaly = np.sqrt((A - local_A) ** 2 + (B - local_B) ** 2)

    # Brown / reddish pigmentation
    brown_color = (
        (A > 130) &
        (B > 120) &
        (normalized_L < np.percentile(normalized_L[interior], 85))
    )

    # 5. Multi-scale texture
    fine_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    fine_texture = cv2.GaussianBlur((gray - fine_background) ** 2, (0, 0), sigmaX=2.0) ** 0.5

    coarse_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=4.0)
    coarse_texture = cv2.GaussianBlur((gray - coarse_background) ** 2, (0, 0), sigmaX=3.0) ** 0.5

    # Adaptive thresholds from the fruit interior
    dark_threshold = max(5.0, float(np.percentile(local_darkness[interior], 70)))
    color_threshold = max(5.0, float(np.percentile(color_anomaly[interior], 65)))
    texture_threshold = max(3.0, float(np.percentile(coarse_texture[interior], 55)))

    # 6. Dark lesion candidates
    dark_candidates = (
        (local_darkness > dark_threshold) &
        ((coarse_texture > texture_threshold) | (color_anomaly > color_threshold))
    )

    brown_candidates = (
        brown_color &
        ((coarse_texture > texture_threshold) | (color_anomaly > color_threshold))
    )

    # 7. Pale / grey patch candidates
    # Pale patches have low chroma relative to surrounding skin,
    # but do not require them to be globally bright.
    neutral_distance = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
    neutral_threshold = min(40.0, float(np.percentile(neutral_distance[interior], 45)))
    saturation_threshold = min(110.0, float(np.percentile(S[interior], 60)))

    pale_or_grey = (
        (neutral_distance < neutral_threshold) &
        (S < saturation_threshold) &
        (
            (normalized_L > np.percentile(normalized_L[interior], 35)) |
            (V > np.percentile(V[interior], 60))
        )
    )

    pale_texture_threshold = max(2.0, float(np.percentile(fine_texture[interior], 45)))
    pale_candidates = (
        pale_or_grey &
        (
            (fine_texture > pale_texture_threshold) |
            (color_anomaly > color_threshold) |
            (coarse_texture > texture_threshold)
        )
    )

    # Reject smooth, extremely bright specular reflections
    reflection = (
        (V > 248) &
        (S < 35) &
        (fine_texture < np.percentile(fine_texture[interior], 60))
    )
    pale_candidates &= ~reflection

    # 8. Combine candidate signals (must remain inside fruit interior)
    candidates = (dark_candidates | brown_candidates | pale_candidates) & interior

    # 9. Clean small isolated noise without large-scale filling
    candidates_u8 = candidates.astype(np.uint8)
    candidates_u8 = cv2.morphologyEx(
        candidates_u8, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    candidates_u8 = cv2.morphologyEx(
        candidates_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    min_area = max(4, int(interior.sum() * 0.00003))
    cleaned = clean_mask(candidates_u8.astype(bool), min_area=min_area)

    # Final hard constraint: never mark pixels outside the fruit
    return cleaned & interior


def _segment_lesions(
    bgr: np.ndarray,
    fruit_mask: np.ndarray,
    class_name: str,
    cam_map: Optional[np.ndarray] = None,
    return_components: bool = False,
):
    """
    Identifies diseased tissue within the fruit silhouette mask.
    Uses botanical color spaces (HSV, LAB), Laplacian texture roughness,
    and additive LayerCAM activation cues.
    - For Guava Scab, Guava Anthracnose, Guava Stylar and Root, and Pomegranate Cercospora:
      applies specialized guava_scab_detector candidate logic.
    - For Guava Phytophthora:
      applies specialized guava_phytophthora_detector candidate logic.
    """
    h, w = bgr.shape[:2]
    fruit_bool = fruit_mask > 0
    total_fruit_px = int(np.count_nonzero(fruit_bool))
    if total_fruit_px < 50:
        if return_components:
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
        return np.zeros((h, w), dtype=np.uint8)

    # Priority 1: Illumination-Aware detector
    # Covers: Pomegranate Anthracnose, Guava Phytophthora, Guava Stylar and Root Rot
    if is_illumination_aware_disease(class_name):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            final_mask = illumination_aware_candidates(rgb, fruit_bool)
            final_u8 = (final_mask.astype(np.uint8)) * 255
            if return_components:
                return final_u8, final_u8  # candidate == final for this detector
            return final_u8
        except (ValueError, Exception) as exc:
            print(f"[severity.py] illumination_aware_candidates failed for {class_name}: {exc}")
            if return_components:
                return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.uint8)

    # Priority 2: Scab / Anthracnose / Cercospora detector
    if is_scab_candidate_disease(class_name):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if return_components:
                final_mask, cand_mask, _ = guava_scab_detector(rgb, fruit_bool, return_components=True)
                return (final_mask.astype(np.uint8)) * 255, (cand_mask.astype(np.uint8)) * 255
            scab_mask_bool = guava_scab_detector(rgb, fruit_bool)
            return (scab_mask_bool.astype(np.uint8)) * 255
        except ValueError:
            if return_components:
                return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.uint8)

    # Priority 3: Legacy Phytophthora detector (fallback if not in illumination-aware set)
    if is_phytophthora_disease(class_name):
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            final_mask, dark_rot, pale_reg = guava_phytophthora_detector(rgb, fruit_bool)
            cand_mask = dark_rot | pale_reg
            if return_components:
                return (final_mask.astype(np.uint8)) * 255, (cand_mask.astype(np.uint8)) * 255
            return (final_mask.astype(np.uint8)) * 255
        except ValueError:
            if return_components:
                return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            return np.zeros((h, w), dtype=np.uint8)

    fruit_type = class_name.split("_")[0]
    equiv_radius = max(15.0, np.sqrt(total_fruit_px / np.pi))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)
    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    # Local texture roughness (Laplacian gradient energy)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    lap_blur = cv2.GaussianBlur(lap, (5, 5), 0)

    # Normalized L to cancel global illumination gradients
    sigma = max(15, int(equiv_radius * 0.38) | 1)
    m_float = fruit_bool.astype(np.float32)
    smooth_L = cv2.GaussianBlur(L * m_float, (0, 0), sigmaX=sigma) / (
        cv2.GaussianBlur(m_float, (0, 0), sigmaX=sigma) + 1e-5
    )
    L_norm = np.clip((L / (smooth_L + 8.0)) * 175.0, 0, 255)

    block_sz = max(15, int(equiv_radius * 0.22) | 1)
    local_spots = cv2.adaptiveThreshold(
        cv2.GaussianBlur(L_norm.astype(np.uint8), (5, 5), 0),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block_sz,
        C=12,
    )

    # LayerCAM high-confidence cue (used ONLY additively, NEVER to suppress)
    cam_high_cue = np.zeros((h, w), dtype=bool)
    if cam_map is not None and isinstance(cam_map, np.ndarray) and cam_map.size > 0:
        cam_resized = cv2.resize(cam_map.astype(np.float32), (w, h))
        cam_fruit = cam_resized[fruit_bool]
        if len(cam_fruit) > 0 and cam_fruit.max() > cam_fruit.min():
            c_p70 = float(np.percentile(cam_fruit, 70))
            cam_high_cue = (cam_resized >= c_p70) & fruit_bool

    # ──────────────────────────────────────────────────────────────────────────
    # 1. AVOCADO
    # ──────────────────────────────────────────────────────────────────────────
    if fruit_type == "Avocado":
        v_med = float(np.percentile(V[fruit_bool], 65))
        l_med = float(np.percentile(L_norm[fruit_bool], 60))
        is_green_skin = (H >= 28) & (H <= 88) & (A < 123)

        lesion_bool = (
            (L_norm < l_med - 22.0)
            | (V < min(90.0, v_med * 0.65))
            | ((A >= 125) & (H < 28))
            | ((local_spots > 0) & (~is_green_skin))
        ) & fruit_bool

    # ──────────────────────────────────────────────────────────────────────────
    # 2. GUAVA (Anthracnose, Phytophthora, Scab, Stylar and Root Rot)
    # ──────────────────────────────────────────────────────────────────────────
    elif fruit_type == "Guava":
        # Healthy skin: Clean vibrant green or smooth yellow blush
        clean_green = (A <= 122.0) & (H >= 28.0) & (H <= 88.0) & (V >= 60.0)
        clean_yellow = (A <= 123.0) & (H >= 20.0) & (H <= 35.0) & (V >= 150.0)
        glare = (V > 225.0) & (S < 35.0)

        # 1. Necrotic brown/orange rot (Phytophthora, Anthracnose, Stylar End Rot)
        brown_rot = (A >= 126.5) & (H <= 36.0) & (V <= 210.0)
        dark_rot = (V <= 75.0) | (L <= 95.0)

        # 2. Scab / pustules / corky reticulation (Guava Scab, Stylar reticulation)
        scab_texture = (lap_blur > 4.0) & ((A >= 123.5) | (H < 35.0) | (V < 115.0))
        corky_lesion = (A >= 125.5) & (V < 190.0)

        # 3. White fungal mycelium
        chroma_mag = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
        white_mycelium = (S < 45.0) & (V > 95.0) & (chroma_mag < 18.0)

        raw_lesions = (brown_rot | dark_rot | scab_texture | corky_lesion | white_mycelium)

        # Additive LayerCAM reinforcement for subtle boundaries
        if np.any(cam_high_cue):
            cam_reinforced = cam_high_cue & ((A >= 123.0) | (V < 120.0))
            raw_lesions = raw_lesions | cam_reinforced

        lesion_bool = raw_lesions & (~clean_green) & (~clean_yellow) & (~glare) & fruit_bool

    # ──────────────────────────────────────────────────────────────────────────
    # 3. POMEGRANATE
    # ──────────────────────────────────────────────────────────────────────────
    elif fruit_type == "Pomegranate":
        if "Alternaria" in class_name:
            # Healthy pomegranate rind: warm red, pink, or orange rind
            healthy_rind = (A >= 135.0) & (V >= 90.0) & (H <= 25.0) & (H >= 5.0)
            glare = (V > 225.0) & (S < 30.0)

            # 1. Dark charcoal / black necrotic rot
            dark_rot = (V <= 80.0) | (L <= 90.0)

            # 2. Dull sunken brown necrotic rot
            brown_necrotic = (A < 136.0) & (B < 138.0) & (V < 115.0)

            # 3. Local spot infections
            spot_rot = (local_spots > 0) & (V < 120.0)

            raw_lesions = (dark_rot | brown_necrotic | spot_rot)

            # Additive LayerCAM reinforcement
            if np.any(cam_high_cue):
                cam_reinforced = cam_high_cue & ((V < 110.0) | (A < 135.0))
                raw_lesions = raw_lesions | cam_reinforced

            lesion_bool = raw_lesions & (~healthy_rind) & (~glare) & fruit_bool

        else:
            # Other pomegranate diseases (Anthracnose, Bacterial Blight, Cercospora)
            healthy_pom = ((A >= 144) & (V >= 115) & (L_norm >= 135)) | ((V >= 165) & (A >= 132))
            lesion_bool = (
                (L_norm < 134)
                | (V < 100)
                | ((H >= 7) & (H <= 26) & (A < 145) & (V < 155))
            ) & (~healthy_pom) & fruit_bool

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MANGO
    # ──────────────────────────────────────────────────────────────────────────
    elif fruit_type == "Mango":
        v_med = float(np.percentile(V[fruit_bool], 65))
        a_med = float(np.percentile(A[fruit_bool], 50))
        l_med = float(np.percentile(L_norm[fruit_bool], 60))

        lesion_bool = (
            (L_norm < l_med - 26.0)
            | (V < v_med * 0.58)
            | ((A > a_med + 9.0) & (L_norm < l_med - 12.0))
            | (local_spots > 0)
        ) & fruit_bool

        chroma_mag = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
        white_mould = (S < 50) & (V > 115) & (chroma_mag < 15.0) & fruit_bool
        if np.count_nonzero(lesion_bool) > 0.04 * total_fruit_px:
            lesion_bool = lesion_bool | white_mould

    # ──────────────────────────────────────────────────────────────────────────
    # 5. ORANGE
    # ──────────────────────────────────────────────────────────────────────────
    else:
        green_ratio = np.count_nonzero((H >= 33) & (H <= 88) & fruit_bool) / float(total_fruit_px)
        if green_ratio > 0.25:
            healthy_green = (H >= 34) & (H <= 88) & (A < 126) & (L_norm > 122)
            lesion_bool = (
                (A >= 127)
                | (L_norm < 128)
                | ((H >= 12) & (H < 33) & (S > 65))
            ) & (~healthy_green) & fruit_bool
        else:
            v_med = float(np.percentile(V[fruit_bool], 65))
            s_med = float(np.percentile(S[fruit_bool], 65))
            lesion_bool = (
                (L_norm < 138)
                | (V < v_med * 0.68)
                | (S < s_med * 0.55)
            ) & fruit_bool

    # Suppress studio white
    studio_white = (V > 235) & (S < 25)
    lesion_bool = lesion_bool & (~studio_white) & fruit_bool

    mask = (lesion_bool.astype(np.uint8)) * 255

    # Scale-aware morphological operations
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    min_area = max(8, int(total_fruit_px * 0.0004))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.bitwise_and(clean_mask, fruit_mask)
    if return_components:
        return clean_mask, mask
    return clean_mask


def encode_mask_base64(mask: Optional[np.ndarray]) -> Optional[str]:
    """Encodes a binary or grayscale mask (uint8 or bool) into a base64 PNG string."""
    if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    if mask.dtype == bool:
        u8_mask = mask.astype(np.uint8) * 255
    elif mask.max() == 1:
        u8_mask = (mask * 255).astype(np.uint8)
    else:
        u8_mask = mask.astype(np.uint8)
    _, buffer = cv2.imencode(".png", u8_mask)
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def analyze_severity(
    pil_image: Image.Image,
    class_name: str,
    cam_map: Optional[np.ndarray] = None,
    return_debug_masks: bool = False,
):
    """
    Analyzes disease severity on the input image.

    Returns:
        severity (str): 'Healthy' | 'Mild' | 'Moderate' | 'Severe'
        diseased_area_pct (float): Percentage (0.0 to 100.0) of fruit area infected
        lesion_mask (np.ndarray): Binary uint8 mask (255 = diseased, 0 = healthy)
        debug_info (dict, optional): Dict with 'fruit_mask', 'candidate_mask', 'final_mask'
    """
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]

    if "Healthy" in class_name:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        if return_debug_masks:
            return "Healthy", 0.0, empty_mask, {
                "fruit_mask": empty_mask,
                "candidate_mask": empty_mask,
                "final_mask": empty_mask,
            }
        return "Healthy", 0.0, empty_mask

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    fruit_mask = _extract_fruit_mask(pil_image, bgr, class_name)
    total_fruit_px = int(np.count_nonzero(fruit_mask))

    if total_fruit_px == 0:
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        if return_debug_masks:
            return "Mild", 0.0, empty_mask, {
                "fruit_mask": fruit_mask,
                "candidate_mask": empty_mask,
                "final_mask": empty_mask,
            }
        return "Mild", 0.0, empty_mask

    if return_debug_masks:
        lesion_mask, candidate_mask = _segment_lesions(bgr, fruit_mask, class_name, cam_map=cam_map, return_components=True)
    else:
        lesion_mask = _segment_lesions(bgr, fruit_mask, class_name, cam_map=cam_map, return_components=False)
        candidate_mask = lesion_mask

    diseased_px = int(np.count_nonzero(lesion_mask))

    pct = round((diseased_px / total_fruit_px) * 100.0, 1)
    pct = min(pct, 100.0)

    if pct < _MILD_THRESHOLD:
        severity = "Mild"
    elif pct < _MODERATE_THRESHOLD:
        severity = "Moderate"
    else:
        severity = "Severe"

    if return_debug_masks:
        debug_dict = {
            "fruit_mask": fruit_mask,
            "candidate_mask": candidate_mask,
            "final_mask": lesion_mask,
        }
        return severity, pct, lesion_mask, debug_dict

    return severity, pct, lesion_mask


def highlight_disease(
    pil_image: Image.Image,
    mask: Optional[np.ndarray],
    class_name: Optional[str] = None,
) -> str:
    """
    Generates a base64-encoded PNG image with diseased areas highlighted.
    For Guava Scab, Guava Anthracnose, Guava Stylar and Root, and Pomegranate Cercospora,
    uses guava_scab_overlay with clean neon-green boundary contours.
    """
    img_rgb = np.array(pil_image.convert("RGB")).copy()
    h, w = img_rgb.shape[:2]

    if mask is not None and isinstance(mask, np.ndarray) and np.count_nonzero(mask) > 0:
        # Ensure mask is 2D and matches image dimensions
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        selected = mask > 0
        if is_illumination_aware_disease(class_name):
            # Use make_overlay for all illumination-aware classes (red tint, 0.48 opacity)
            img_rgb = make_overlay(img_rgb, selected, alpha=0.48)
        elif is_phytophthora_disease(class_name):
            img_rgb = phytophthora_overlay(img_rgb, selected, color=(255, 0, 0), alpha=0.45)
        elif is_scab_candidate_disease(class_name):
            img_rgb = guava_scab_overlay(img_rgb, selected, alpha=0.48)
        else:
            overlay = img_rgb.copy()
            overlay[selected] = [230, 40, 40]
            img_rgb[selected] = cv2.addWeighted(img_rgb, 0.52, overlay, 0.48, 0)[selected]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_draw_area = max(8, int((h * w) * 0.00015))
            clean_contours = [c for c in contours if cv2.contourArea(c) >= min_draw_area]
            cv2.drawContours(img_rgb, clean_contours, -1, (0, 255, 90), thickness=2, lineType=cv2.LINE_AA)

    _, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def generate_cam_heatmap(pil_image: Image.Image, cam_map: Optional[np.ndarray]) -> Optional[str]:
    """
    Generate a base64-encoded PNG image with a Grad-CAM / LayerCAM jet heatmap overlay.
    Returns None if cam_map is None or empty.
    """
    if cam_map is None or not isinstance(cam_map, np.ndarray) or cam_map.size == 0:
        return None

    img_rgb = np.array(pil_image.convert("RGB")).copy()
    h, w = img_rgb.shape[:2]

    try:
        cam_resized = cv2.resize(cam_map.astype(np.float32), (w, h))
        c_min, c_max = float(cam_resized.min()), float(cam_resized.max())
        if c_max > c_min:
            cam_norm = (cam_resized - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_resized)

        heatmap = (cam_norm * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_rgb, 0.60, heatmap_rgb, 0.40, 0)
        _, buffer = cv2.imencode(".png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer.tobytes()).decode("utf-8")
    except Exception as exc:
        print(f"[severity.py] Failed to generate heatmap: {exc}")
        return None
