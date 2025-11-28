import cv2
import numpy as np
import os
import glob
from pathlib import Path

# ================================================================
# CONFIG
# ================================================================
INPUT_DIR = "data/face"               # root of all face datasets
OUTPUT_DIR = "results_day3/face"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# UI cropping ratios (tuned for your thermal camera screenshots)
TOP_UI_FRAC   = 0.15   # remove top 15% (date / time / °C text)
RIGHT_UI_FRAC = 0.18   # remove right 18% (temperature scale bar)

FACE_PERCENTILE = 70   # brightness percentile for initial mask
KERNEL_SIZE     = 7    # morphology kernel size

HOTSPOT_PERCENTILE = 95
HEATMAP_COLORMAP  = cv2.COLORMAP_JET


# ================================================================
# 1) Convert to grayscale (for any palette)
# ================================================================
def to_gray(img):
    if img is None:
        return None

    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


# ================================================================
# 2) Face ROI detection with UI cropping
# ================================================================
def detect_face_roi(gray):
    """
    Returns:
      roi      : cropped face region (grayscale)
      mask     : full-size binary mask (1 on face box)
      bbox     : (x1, y1, x2, y2) in original image coords
    """
    H, W = gray.shape

    # 1) crop out UI regions (top bar + right scale bar)
    top_crop   = int(H * TOP_UI_FRAC)
    right_crop = int(W * RIGHT_UI_FRAC)

    y0, y1 = top_crop, H
    x0, x1 = 0, W - right_crop

    work = gray[y0:y1, x0:x1]

    # 2) threshold on brightness inside cropped area
    thr = np.percentile(work, FACE_PERCENTILE)
    _, mask_small = cv2.threshold(work, int(thr), 255, cv2.THRESH_BINARY)

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3) find candidate face blobs
    cnts, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None

    best_cnt = None
    best_area = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = h / float(w + 1e-6)

        # reject tiny blobs and very flat text-like shapes
        if area < 0.02 * work.size:
            continue
        if h < 0.25 * H:
            continue
        if aspect < 0.8 or aspect > 3.5:
            continue

        if area > best_area:
            best_area = area
            best_cnt = c

    if best_cnt is None:
        return None, None, None

    # 4) convert bbox back to full-image coords
    x, y, w, h = cv2.boundingRect(best_cnt)
    x1_full = x0 + x
    y1_full = y0 + y
    x2_full = x1_full + w
    y2_full = y1_full + h

    # add small margin
    margin_x = int(0.08 * w)
    margin_y = int(0.10 * h)

    x1_full = max(0, x1_full - margin_x)
    y1_full = max(0, y1_full - margin_y)
    x2_full = min(W, x2_full + margin_x)
    y2_full = min(H, y2_full + margin_y)

    # build full-size mask
    face_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.rectangle(face_mask, (x1_full, y1_full), (x2_full, y2_full), 255, -1)

    # extract ROI
    roi = gray[y1_full:y2_full, x1_full:x2_full]
    bbox = (x1_full, y1_full, x2_full, y2_full)

    return roi, face_mask, bbox


# ================================================================
# 3) Face region definitions (forehead, cheeks, nose, mouth)
# ================================================================
def get_face_regions(roi):
    """
    Split face ROI into semantic subregions using simple ratios.
    Returns dict with region name -> (x1, y1, x2, y2)
    """
    h, w = roi.shape
    regions = {}

    # rows (top to bottom)
    y_forehead_top = int(0.05 * h)
    y_forehead_bot = int(0.30 * h)

    y_eye_mid      = int(0.45 * h)
    y_nose_bot     = int(0.65 * h)

    y_mouth_top    = int(0.65 * h)
    y_mouth_bot    = int(0.85 * h)

    # columns (left to right)
    x_left_inner   = int(0.05 * w)
    x_left_outer   = int(0.45 * w)

    x_right_inner  = int(0.55 * w)
    x_right_outer  = int(0.95 * w)

    x_center_left  = int(0.35 * w)
    x_center_right = int(0.65 * w)

    # forehead: wide top band
    regions["forehead"] = (
        int(0.20 * w), y_forehead_top,
        int(0.80 * w), y_forehead_bot
    )

    # left cheek
    regions["left_cheek"] = (
        x_left_inner, y_forehead_bot,
        x_left_outer, y_nose_bot
    )

    # right cheek
    regions["right_cheek"] = (
        x_right_inner, y_forehead_bot,
        x_right_outer, y_nose_bot
    )

    # nose (center column)
    regions["nose"] = (
        x_center_left, y_forehead_bot,
        x_center_right, y_nose_bot
    )

    # mouth / perioral region
    regions["mouth"] = (
        x_center_left, y_mouth_top,
        x_center_right, y_mouth_bot
    )

    return regions


def extract_region_stats(roi, box):
    x1, y1, x2, y2 = box
    patch = roi[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    patch_f = patch.astype(np.float32)
    return {
        "mean": float(np.mean(patch_f)),
        "max":  float(np.max(patch_f)),
        "min":  float(np.min(patch_f)),
        "var":  float(np.var(patch_f))
    }


# ================================================================
# 4) Global face temperature + asymmetry
# ================================================================
def extract_global_face_features(roi):
    roi_f = roi.astype(np.float32)
    mean_temp = float(np.mean(roi_f))
    max_temp  = float(np.max(roi_f))
    min_temp  = float(np.min(roi_f))
    median    = float(np.median(roi_f))
    variance  = float(np.var(roi_f))

    # gradient strength (edges / vascular pattern)
    gx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    gradient_strength = float(np.mean(grad_mag))

    # left-right asymmetry (simple, like breast)
    h, w = roi.shape
    mid = w // 2
    left  = roi_f[:, :mid]
    right = roi_f[:, mid:]

    min_w = min(left.shape[1], right.shape[1])
    left_eq  = left[:, :min_w]
    right_eq = right[:, :min_w]

    diff = np.abs(left_eq - right_eq)
    asym_score = float(np.mean(diff))

    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_color = cv2.applyColorMap(diff_norm.astype(np.uint8), HEATMAP_COLORMAP)
    # tile to full width for visualization
    diff_full = cv2.resize(diff_color, (w, h))

    # cheek-level asymmetry (optional: mean of each half)
    left_mean  = float(np.mean(left_eq))
    right_mean = float(np.mean(right_eq))
    cheek_diff = abs(left_mean - right_mean)

    features = {
        "mean_temp": mean_temp,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "median_temp": median,
        "variance": variance,
        "gradient_strength": gradient_strength,
        "face_asymmetry_score": asym_score,
        "cheek_side_diff": cheek_diff
    }

    return features, diff_full


# ================================================================
# 5) Hotspot extraction on face
# ================================================================
def extract_face_hotspots(roi, percentile=HOTSPOT_PERCENTILE):
    roi_f = roi.astype(np.float32)

    thr = np.percentile(roi_f, percentile)
    _, mask = cv2.threshold(roi, int(thr), 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = roi.shape[0] * roi.shape[1]
    total_hot_area = 0
    hotspot_count = 0
    largest_area = 0
    largest_centroid = (0, 0)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue

        hotspot_count += 1
        total_hot_area += area

        if area > largest_area:
            largest_area = area
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                largest_centroid = (cx, cy)

    area_ratio = total_hot_area / roi_area if roi_area > 0 else 0.0

    # visualization overlay
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    hot_vis = cv2.addWeighted(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR),
                              0.6, mask_color, 0.4, 0)

    metrics = {
        "hotspot_count": int(hotspot_count),
        "hotspot_area_ratio": float(area_ratio),
        "largest_hotspot_area": float(largest_area),
        "largest_hotspot_centroid": largest_centroid
    }

    return mask, hot_vis, metrics


# ================================================================
# 6) Draw bbox overlay
# ================================================================
def draw_box(gray, bbox):
    x1, y1, x2, y2 = bbox
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# ================================================================
# MAIN PIPELINE
# ================================================================
all_faces = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    all_faces.extend(glob.glob(os.path.join(INPUT_DIR, "**", ext), recursive=True))

print("Total face images found:", len(all_faces))

for path in all_faces:
    fname = Path(path).stem
    print("Processing:", fname)

    img = cv2.imread(path)
    gray = to_gray(img)
    if gray is None:
        print(" -> could not read")
        continue

    roi, face_mask, bbox = detect_face_roi(gray)
    if roi is None:
        print(" -> no face detected")
        continue

    # 1) global features & asymmetry map
    global_feats, asym_heat = extract_global_face_features(roi)

    # 2) regional stats
    regions = get_face_regions(roi)
    region_stats = {}
    for name, box in regions.items():
        stats = extract_region_stats(roi, box)
        region_stats[name] = stats

    # 3) hotspots
    hot_mask, hot_vis, hot_metrics = extract_face_hotspots(roi)

    # 4) save images
    base = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(base + "_gray.png", gray)
    cv2.imwrite(base + "_mask.png", face_mask)
    cv2.imwrite(base + "_roi.png", roi)
    cv2.imwrite(base + "_overlay.png", draw_box(gray, bbox))
    cv2.imwrite(base + "_asymmetry_heatmap.png", asym_heat)
    cv2.imwrite(base + "_hotspots.png", hot_vis)

    # 5) write summary
    with open(base + "_summary.txt", "w") as f:
        f.write(f"File: {os.path.basename(path)}\n\n")

        f.write("GLOBAL FACE FEATURES:\n")
        f.write(f"  Mean Temp: {global_feats['mean_temp']:.2f}\n")
        f.write(f"  Max Temp : {global_feats['max_temp']:.2f}\n")
        f.write(f"  Min Temp : {global_feats['min_temp']:.2f}\n")
        f.write(f"  Median   : {global_feats['median_temp']:.2f}\n")
        f.write(f"  Variance : {global_feats['variance']:.2f}\n")
        f.write(f"  Gradient Strength: {global_feats['gradient_strength']:.3f}\n")
        f.write(f"  Face Asymmetry Score: {global_feats['face_asymmetry_score']:.3f}\n")
        f.write(f"  Cheek Side Diff: {global_feats['cheek_side_diff']:.3f}\n")

        f.write("\nREGION FEATURES:\n")
        for name, stats in region_stats.items():
            if stats is None:
                f.write(f"  {name}: None\n")
            else:
                f.write(f"  {name}:\n")
                f.write(f"    Mean: {stats['mean']:.2f}\n")
                f.write(f"    Max : {stats['max']:.2f}\n")
                f.write(f"    Min : {stats['min']:.2f}\n")
                f.write(f"    Var : {stats['var']:.2f}\n")

        f.write("\nHOTSPOTS:\n")
        f.write(f"  Count      : {hot_metrics['hotspot_count']}\n")
        f.write(f"  Area Ratio : {hot_metrics['hotspot_area_ratio']:.4f}\n")
        f.write(f"  Largest Area     : {hot_metrics['largest_hotspot_area']:.2f}\n")
        f.write(f"  Largest Centroid : {hot_metrics['largest_hotspot_centroid']}\n")

print("\nDAY-3 Face module completed.")
print("Outputs saved to:", OUTPUT_DIR)
