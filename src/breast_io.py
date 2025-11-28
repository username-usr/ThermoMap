import cv2
import numpy as np
import os
import glob
from pathlib import Path

# ================================================================
# CONFIG
# ================================================================
INPUT_DIRS = [
    "data/breast/benign",
    "data/breast/malignant"
]

OUTPUT_DIR = "results_day2/breast"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- parameters for ROI ---
VERT_TOP = 0.20        # remove upper region (neck/shoulders)
VERT_BOTTOM = 0.12     # remove lower region (abdomen)
SIDE_DILATION = 15     # horizontal fill for oblique images

# --- asymmetry ---
HEATMAP_COLORMAP = cv2.COLORMAP_JET


# ================================================================
# (1) USE YOUR CURRENT TORSO MASK
# ================================================================
def make_torso_mask(gray, percentile=56, ks=6, extend=0.15):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t = np.percentile(blur, percentile)
    _, raw = cv2.threshold(blur, int(t), 255, cv2.THRESH_BINARY)

    # morphology
    kernel = np.ones((ks, ks), np.uint8)
    mask = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    torso = np.zeros_like(mask)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.02 * mask.size:
            cv2.drawContours(torso, [largest], -1, 255, thickness=cv2.FILLED)

    # extend downward
    ys, xs = np.where(torso > 0)
    if len(xs) > 0:
        y_min, y_max = ys.min(), ys.max()
        H = mask.shape[0]
        ext_pix = int(H * extend)
        torso[y_max:min(H, y_max + ext_pix), :] = 255

    # remove top + bottom
    H, W = torso.shape
    torso[:int(H*VERT_TOP), :] = 0
    torso[int(H*(1-VERT_BOTTOM)):, :] = 0

    # horizontal dilation
    h_kernel = np.ones((1, SIDE_DILATION), np.uint8)
    torso = cv2.dilate(torso, h_kernel, iterations=1)

    return torso


# ================================================================
# (2) POSE DETECTION
# ================================================================
def detect_pose(gray, mask, filename):
    name = filename.lower()

    if "oblleft" in name:
        return "left_oblique"
    if "oblright" in name:
        return "right_oblique"
    if "anterior" in name or "front" in name:
        return "frontal"

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return "frontal"

    w = gray.shape[1]
    cx = xs.mean()
    cx_ratio = cx / w

    if cx_ratio < 0.40:
        return "right_oblique"
    elif cx_ratio > 0.60:
        return "left_oblique"
    else:
        return "frontal"


# ================================================================
# (3) BREAST ROI EXTRACTION
# ================================================================
def extract_breast_roi(gray, mask, pose):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    torso = gray[y_min:y_max, x_min:x_max]
    H, W = torso.shape
    if H < 40 or W < 40:
        return None, None

    y_top = int(H * 0.05)
    y_bottom = int(H * 0.95)

    if pose == "frontal":
        x_left = int(W * 0.12)
        x_right = int(W * 0.88)

    elif pose == "left_oblique":
        x_left = int(W * 0.25)
        x_right = int(W * 0.98)

    elif pose == "right_oblique":
        x_left = int(W * 0.02)
        x_right = int(W * 0.75)

    else:
        x_left = int(W * 0.12)
        x_right = int(W * 0.88)

    roi = torso[y_top:y_bottom, x_left:x_right]
    if roi.size == 0:
        return None, None

    coords = {
        "x1": x_min + x_left,
        "x2": x_min + x_right,
        "y1": y_min + y_top,
        "y2": y_min + y_bottom
    }

    return roi, coords


# ================================================================
# (4) **FIXED** FRONTAL ASYMMETRY (Equal-width halves)
# ================================================================
def lr_asymmetry(roi):
    h, w = roi.shape

    mid = w // 2
    left = roi[:, :mid]
    right = roi[:, mid:]

    # enforce EXACT same width
    Lw = left.shape[1]
    Rw = right.shape[1]
    min_w = min(Lw, Rw)

    left_eq = left[:, :min_w]
    right_eq = right[:, :min_w]

    # diff calculation
    diff = cv2.absdiff(left_eq, right_eq)

    # heatmap
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_color = cv2.applyColorMap(diff_norm.astype(np.uint8), HEATMAP_COLORMAP)

    score = float(np.mean(diff))

    return left_eq, right_eq, diff_color, score


# ================================================================
# (5) OBLIQUE ASYMMETRY
# ================================================================
def oblique_asymmetry(roi):
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    heat = cv2.applyColorMap(
        cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        HEATMAP_COLORMAP
    )

    score = float(np.mean(mag))
    return heat, score


# ================================================================
# (6) DRAW ROI
# ================================================================
def draw_roi_box(gray, coords):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out,
                  (coords["x1"], coords["y1"]),
                  (coords["x2"], coords["y2"]),
                  (0, 255, 0), 2)
    return out

# ================================================================
# (7) HOTSPOT EXTRACTION (simple + effective)
# ================================================================
def extract_hotspots(roi, percentile=90):
    """
    Extract hotspots from breast ROI using percentiles.
    Returns:
      - hotspot_mask (binary)
      - hotspot_overlay (colored visualization)
      - hotspot_metrics dict
    """

    # 1) Compute threshold
    thr = np.percentile(roi, percentile)
    _, hot_mask = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)

    # 2) Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)
    hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # 3) Contours
    contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hotspot_count = len(contours)
    largest_area = 0
    largest_centroid = (0, 0)

    roi_area = roi.shape[0] * roi.shape[1]
    hotspot_area_total = 0

    for c in contours:
        area = cv2.contourArea(c)
        hotspot_area_total += area
        if area > largest_area:
            largest_area = area
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                largest_centroid = (cx, cy)

    area_ratio = hotspot_area_total / roi_area if roi_area > 0 else 0

    # 4) create colored overlay
    heat_mask = cv2.applyColorMap(hot_mask, cv2.COLORMAP_JET)
    heat_mask = cv2.addWeighted(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), 0.6,
                                heat_mask, 0.4, 0)

    metrics = {
        "hotspot_count": hotspot_count,
        "hotspot_area_ratio": float(area_ratio),
        "largest_hotspot_area": float(largest_area),
        "largest_hotspot_centroid": largest_centroid
    }

    return hot_mask, heat_mask, metrics

# ================================================================
# (8) TEMPERATURE FEATURES (simple + effective)
# ================================================================
def extract_temperature_features(roi):
    roi_float = roi.astype(np.float32)

    mean_temp = float(np.mean(roi_float))
    max_temp = float(np.max(roi_float))
    min_temp = float(np.min(roi_float))
    median_temp = float(np.median(roi_float))
    variance_temp = float(np.var(roi_float))

    # temperature gradient magnitude
    gx = cv2.Sobel(roi_float, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    gradient_strength = float(np.mean(grad_mag))

    # simple nipple-line difference (frontal only)
    h, w = roi.shape
    mid_y = int(h * 0.45)      # approximate nipple-line height
    left_half = roi_float[mid_y, :w//2]
    right_half = roi_float[mid_y, w//2:]

    nipple_diff = float(abs(np.mean(left_half) - np.mean(right_half))) if w > 2 else 0

    features = {
        "mean_temp": mean_temp,
        "max_temp": max_temp,
        "min_temp": min_temp,
        "median_temp": median_temp,
        "variance": variance_temp,
        "gradient_strength": gradient_strength,
        "nipple_line_diff": nipple_diff
    }

    return features


# ================================================================
# MAIN PIPELINE
# ================================================================
all_images = []
for folder in INPUT_DIRS:
    all_images.extend(
        glob.glob(os.path.join(folder, "*.jpg")) +
        glob.glob(os.path.join(folder, "*.png")) +
        glob.glob(os.path.join(folder, "*.jpeg")) +
        glob.glob(os.path.join(folder, "*.bmp"))
    )

print("Total images:", len(all_images))

for path in all_images:
    fname = os.path.basename(path)
    stem = Path(fname).stem

    print("Processing:", fname)

    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(" -> skip, cannot read")
        continue

    # 1) Torso mask
    mask = make_torso_mask(gray)

    # 2) Pose
    pose = detect_pose(gray, mask, fname)

    # 3) Breast ROI
    roi, coords = extract_breast_roi(gray, mask, pose)
    if roi is None:
        print(" -> skip, no ROI")
        continue

    # 4) Asymmetry
    if pose == "frontal":
        left, right, heat, score = lr_asymmetry(roi)
    else:
        left, right = None, None
        heat, score = oblique_asymmetry(roi)

    # 5) Hotspot extraction
    hot_mask, hot_vis, hot_metrics = extract_hotspots(roi)

    # 6) Temperature features
    temp_features = extract_temperature_features(roi)


    # 7) Save outputs
    base = os.path.join(OUTPUT_DIR, stem)

    cv2.imwrite(base + "_gray.png", gray)
    cv2.imwrite(base + "_mask.png", mask)
    cv2.imwrite(base + "_roi.png", roi)
    cv2.imwrite(base + "_heatmap.png", heat)
    cv2.imwrite(base + "_hotspot_mask.png", hot_mask)
    cv2.imwrite(base + "_hotspots.png", hot_vis)


    if left is not None:
        cv2.imwrite(base + "_left.png", left)
        cv2.imwrite(base + "_right.png", right)

    overlay = draw_roi_box(gray, coords)
    cv2.imwrite(base + "_overlay.png", overlay)

    # summary
    with open(base + "_summary.txt", "w") as f:
        f.write(f"File: {fname}\n")
        f.write(f"Pose: {pose}\n")
        f.write(f"Asymmetry Score: {score:.4f}\n")
        
        f.write("\nTEMPERATURE FEATURES:\n")
        f.write(f"  Mean Temperature: {temp_features['mean_temp']:.2f}\n")
        f.write(f"  Max Temperature: {temp_features['max_temp']:.2f}\n")
        f.write(f"  Min Temperature: {temp_features['min_temp']:.2f}\n")
        f.write(f"  Median Temperature: {temp_features['median_temp']:.2f}\n")
        f.write(f"  Variance: {temp_features['variance']:.2f}\n")
        f.write(f"  Gradient Strength: {temp_features['gradient_strength']:.3f}\n")
        f.write(f"  Nipple-Line Temperature Difference: {temp_features['nipple_line_diff']:.2f}\n")

        f.write("\nHOTSPOTS:\n")
        f.write(f"  Count: {hot_metrics['hotspot_count']}\n")
        f.write(f"  Area Ratio: {hot_metrics['hotspot_area_ratio']:.4f}\n")
        f.write(f"  Largest Area: {hot_metrics['largest_hotspot_area']:.2f}\n")
        f.write(f"  Largest Centroid: {hot_metrics['largest_hotspot_centroid']}\n")


print("\nDAY-2 Breast module completed.")
print("Outputs saved to:", OUTPUT_DIR)
