import cv2
import numpy as np
import os
from pathlib import Path
import torch

def predict_feet_mask(gray_img, model_path="feet_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = gray_img.shape

    # Resize → normalize
    resized = cv2.resize(gray_img, (256, 256))
    inp = resized.astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)

    # Load model once
    global FEET_MODEL
    if 'FEET_MODEL' not in globals():
        FEET_MODEL = UNetSmall().to(device)
        FEET_MODEL.load_state_dict(torch.load(model_path, map_location=device))
        FEET_MODEL.eval()

    with torch.no_grad():
        out = FEET_MODEL(inp)
        out = torch.sigmoid(out)[0,0].cpu().numpy()

    mask_small = (out > 0.5).astype(np.uint8)*255
    mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)

    return mask


# ---------------------------
# SMALL UNET MODEL (same as training)
# ---------------------------
class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNetSmall(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(1, 32); self.pool1 = torch.nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64); self.pool2 = torch.nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128); self.pool3 = torch.nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256); self.pool4 = torch.nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        self.up4 = torch.nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv4 = DoubleConv(512, 256)
        self.up3 = torch.nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = torch.nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = torch.nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv1 = DoubleConv(64, 32)
        self.out_conv = torch.nn.Conv2d(32, 1, 1)

    def forward(self, x):
        c1 = self.down1(x); p1 = self.pool1(c1)
        c2 = self.down2(p1); p2 = self.pool2(c2)
        c3 = self.down3(p2); p3 = self.pool3(c3)
        c4 = self.down4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        u4 = self.up4(bn); u4 = torch.cat([u4, c4], dim=1); c4 = self.conv4(u4)
        u3 = self.up3(c4); u3 = torch.cat([u3, c3], dim=1); c3 = self.conv3(u3)
        u2 = self.up2(c3); u2 = torch.cat([u2, c2], dim=1); c2 = self.conv2(u2)
        u1 = self.up1(c2); u1 = torch.cat([u1, c1], dim=1); c1 = self.conv1(u1)
        return self.out_conv(c1)


# ================================================================
# CONFIG
# ================================================================
INPUT_ROOT = "data/feet"

OUTPUT_DIR = "results_day3/feet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# bottom crop start (keep lower ~55–60% where feet live)
BOTTOM_START_FRAC = 0.45

# gradient threshold for edges (can tweak 20–40)
EDGE_THRESH = 25

# morphology kernel for connecting foot edges
KERNEL_SIZE = 7

# plantar region (bottom part of each foot ROI)
PLANTAR_FRAC_FROM_BOTTOM = 0.6   # keep bottom 60%

HOTSPOT_PERCENTILE = 95
HEATMAP_COLORMAP = cv2.COLORMAP_JET


# ================================================================
# UTILS
# ================================================================
def to_gray(img):
    """Convert to blurred grayscale."""
    if img is None:
        return None
    if len(img.shape) == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


# ================================================================
# 1) FEET SEGMENTATION USING GRADIENT + CLUSTERED MIDLINE
# ================================================================
def segment_feet(gray):
    """
    Returns:
      feet_mask_full, left_box, right_box, global_box, left_mask_full, right_mask_full
    """

    # 1) Predict mask with DL
    mask = predict_feet_mask(gray)

    # 2) find feet bounding box
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None, None, None, None

    global_box = (xs.min(), ys.min(), xs.max(), ys.max())

    # 3) split mask into left/right
    mid_x = gray.shape[1] // 2

    left_mask = np.zeros_like(mask)
    right_mask = np.zeros_like(mask)

    left_mask[:, :mid_x] = mask[:, :mid_x]
    right_mask[:, mid_x:] = mask[:, mid_x:]

    # 4) get each bounding box
    def box_from(m):
        y, x = np.where(m > 0)
        if len(x)==0: return None
        return (x.min(), y.min(), x.max(), y.max())

    left_box = box_from(left_mask)
    right_box = box_from(right_mask)

    return mask, left_box, right_box, global_box, left_mask, right_mask



# ================================================================
# 2) ROI UTILS
# ================================================================
def crop_roi(gray, box):
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return gray[y1:y2, x1:x2]


def equalize_rois(left_roi, right_roi):
    if left_roi is None or right_roi is None:
        return left_roi, right_roi
    hL, wL = left_roi.shape
    hR, wR = right_roi.shape
    h = min(hL, hR)
    w = min(wL, wR)
    return left_roi[:h, :w], right_roi[:h, :w]


def get_plantar_roi(foot_roi):
    """Bottom PLANTAR_FRAC_FROM_BOTTOM of a single-foot ROI."""
    if foot_roi is None:
        return None
    h, w = foot_roi.shape
    start_y = int(h * (1.0 - PLANTAR_FRAC_FROM_BOTTOM))
    return foot_roi[start_y:, :]


# ================================================================
# 3) TEMPERATURE / GRADIENT / ASYMMETRY
# ================================================================
def temp_stats(roi):
    if roi is None:
        return None
    roi_f = roi.astype(np.float32)
    return {
        "mean": float(np.mean(roi_f)),
        "max": float(np.max(roi_f)),
        "min": float(np.min(roi_f)),
        "var": float(np.var(roi_f)),
        "median": float(np.median(roi_f)),
    }


def gradient_strength(roi):
    if roi is None:
        return None
    roi_f = roi.astype(np.float32)
    gx = cv2.Sobel(roi_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(mag))


def feet_asymmetry(left_roi, right_roi):
    """
    Whole-foot asymmetry using equalized ROIs.
    Returns:
      asym_score, asym_heatmap (or None), (left_eq, right_eq)
    """
    if left_roi is None or right_roi is None:
        # graceful fallback: no asymmetry when one foot missing
        dummy_heat = np.zeros((50, 100, 3), dtype=np.uint8)
        return 0.0, dummy_heat, (left_roi, right_roi)

    L, R = equalize_rois(left_roi, right_roi)
    if L is None or R is None or L.size == 0 or R.size == 0:
        dummy_heat = np.zeros((50, 100, 3), dtype=np.uint8)
        return 0.0, dummy_heat, (L, R)

    Lf = L.astype(np.float32)
    Rf = R.astype(np.float32)

    diff = np.abs(Lf - Rf)
    asym_score = float(np.mean(diff))

    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_color = cv2.applyColorMap(diff_norm.astype(np.uint8),
                                   HEATMAP_COLORMAP)

    # for nicer visualization, put both feet width-wise
    h, w, _ = diff_color.shape
    diff_full = cv2.resize(diff_color, (2 * w, h))

    return asym_score, diff_full, (L, R)


# ================================================================
# 4) HOTSPOTS (on plantar region only)
# ================================================================
def extract_hotspots(plantar_roi, percentile=HOTSPOT_PERCENTILE):
    if plantar_roi is None:
        return (np.zeros((1, 1), dtype=np.uint8),
                np.zeros((1, 1, 3), dtype=np.uint8),
                {"hotspot_count": 0,
                 "hotspot_area_ratio": 0.0,
                 "largest_hotspot_area": 0.0,
                 "largest_hotspot_centroid": (0, 0)})

    roi = plantar_roi
    roi_f = roi.astype(np.float32)
    thr = np.percentile(roi_f, percentile)

    _, mask = cv2.threshold(roi, int(thr), 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = roi.shape[0] * roi.shape[1]
    total_hot_area = 0
    hotspot_count = 0
    largest_area = 0
    largest_centroid = (0, 0)

    for c in contours:
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

    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    hot_vis = cv2.addWeighted(
        cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), 0.6, mask_color, 0.4, 0
    )

    metrics = {
        "hotspot_count": int(hotspot_count),
        "hotspot_area_ratio": float(area_ratio),
        "largest_hotspot_area": float(largest_area),
        "largest_hotspot_centroid": largest_centroid,
    }

    return mask, hot_vis, metrics


# ================================================================
# 5) OVERLAY
# ================================================================
def draw_overlay(gray, global_box):
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if global_box is not None:
        x1, y1, x2, y2 = global_box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


# ================================================================
# MAIN PIPELINE
# ================================================================
# Collect only THERMAL images
thermal_paths = []
for root, dirs, files in os.walk(INPUT_ROOT):
    if "thermal" not in root.lower():
        continue
    for fname in files:
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            thermal_paths.append(os.path.join(root, fname))

print("Total thermal feet images found:", len(thermal_paths))

for path in thermal_paths:
    img = cv2.imread(path)
    gray = to_gray(img)
    if gray is None:
        print("Could not read:", path)
        continue

    rel = Path(path).relative_to(INPUT_ROOT)
    stem = rel.stem.replace(os.sep, "_")

    print("Processing:", rel)

    # 1) segmentation
    (
        feet_mask,
        left_box,
        right_box,
        global_box,
        left_mask_full,
        right_mask_full,
    ) = segment_feet(gray)

    if feet_mask is None or global_box is None:
        print(" -> no feet detected, skipping")
        continue

    # 2) whole-foot ROIs
    gx1, gy1, gx2, gy2 = global_box
    feet_roi = gray[gy1:gy2, gx1:gx2]

    left_roi = crop_roi(gray, left_box) if left_box is not None else None
    right_roi = crop_roi(gray, right_box) if right_box is not None else None

    # 3) plantar ROIs (bottom part of each foot)
    plantar_left = get_plantar_roi(left_roi)
    plantar_right = get_plantar_roi(right_roi)

    # 4) temperature + gradient
    left_stats = temp_stats(left_roi)
    right_stats = temp_stats(right_roi)
    left_grad = gradient_strength(left_roi)
    right_grad = gradient_strength(right_roi)

    # 5) asymmetry (whole-foot)
    asym_score, asym_heat, (left_eq, right_eq) = feet_asymmetry(
        left_roi, right_roi
    )

    # 6) hotspots (plantar only, combined)
    #    We can merge plantar ROIs side-by-side just for hotspot viz
    if plantar_left is not None and plantar_right is not None:
        plantar_both = np.hstack(equalize_rois(plantar_left, plantar_right))
    elif plantar_left is not None:
        plantar_both = plantar_left
    elif plantar_right is not None:
        plantar_both = plantar_right
    else:
        plantar_both = None

    hot_mask, hot_vis, hot_metrics = extract_hotspots(plantar_both)

    # 7) save images
    base = os.path.join(OUTPUT_DIR, stem)

    cv2.imwrite(base + "_gray.png", gray)
    cv2.imwrite(base + "_feet_mask.png", feet_mask)
    cv2.imwrite(base + "_overlay.png", draw_overlay(gray, global_box))
    cv2.imwrite(base + "_feet_roi.png", feet_roi)
    if left_roi is not None:
        cv2.imwrite(base + "_left.png", left_roi)
    if right_roi is not None:
        cv2.imwrite(base + "_right.png", right_roi)
    cv2.imwrite(base + "_asymmetry_heatmap.png", asym_heat)
    cv2.imwrite(base + "_plantar_hotspots.png", hot_vis)

    # 8) summary
    group = "unknown"
    parts = rel.parts
    if len(parts) >= 2:
        group = "/".join(parts[:2])  # e.g. diabetic/R0 or healthy/thermal

    with open(base + "_summary.txt", "w") as f:
        f.write(f"File: {rel}\n")
        f.write(f"Group: {group}\n\n")

        f.write(f"GLOBAL FEET ROI BOX: {global_box}\n\n")

        f.write("LEFT FOOT (whole):\n")
        if left_stats is None:
            f.write("  Not detected\n")
        else:
            f.write(f"  Mean Temp : {left_stats['mean']:.2f}\n")
            f.write(f"  Max Temp  : {left_stats['max']:.2f}\n")
            f.write(f"  Min Temp  : {left_stats['min']:.2f}\n")
            f.write(f"  Median    : {left_stats['median']:.2f}\n")
            f.write(f"  Variance  : {left_stats['var']:.2f}\n")
            f.write(f"  Gradient Strength: {left_grad:.3f}\n")

        f.write("\nRIGHT FOOT (whole):\n")
        if right_stats is None:
            f.write("  Not detected\n")
        else:
            f.write(f"  Mean Temp : {right_stats['mean']:.2f}\n")
            f.write(f"  Max Temp  : {right_stats['max']:.2f}\n")
            f.write(f"  Min Temp  : {right_stats['min']:.2f}\n")
            f.write(f"  Median    : {right_stats['median']:.2f}\n")
            f.write(f"  Variance  : {right_stats['var']:.2f}\n")
            f.write(f"  Gradient Strength: {right_grad:.3f}\n")

        f.write("\nASYMMETRY (whole feet):\n")
        f.write(f"  Mean L-R difference score: {asym_score:.3f}\n")

        f.write("\nPLANTAR HOTSPOTS (bottom of feet):\n")
        f.write(f"  Count          : {hot_metrics['hotspot_count']}\n")
        f.write(
            f"  Area Ratio     : {hot_metrics['hotspot_area_ratio']:.4f}\n"
        )
        f.write(
            f"  Largest Area   : {hot_metrics['largest_hotspot_area']:.2f}\n"
        )
        f.write(
            f"  Largest Center : {hot_metrics['largest_hotspot_centroid']}\n"
        )

print("\nFeet module finished.")
print("Outputs in:", OUTPUT_DIR)
