import cv2
import numpy as np
import glob
import os

# ---- UPDATED INPUT DATASET FOLDERS (data/ instead of datasets/) ----
DATASETS = {
    "breast_benign":     "data/breast/benign",
    "breast_malignant":  "data/breast/malignant",
    "face_detection":    "data/face/detection",
    "face_recognition":  "data/face/recognition",
    "face_emotion":      "data/face/emotion",
    "body_lwir":         "data/body/lwir",
    "feet_diabetic_th":  "data/feet/diabetic/thermal",
    "feet_healthy_th":   "data/feet/healthy/thermal"
}

# ---- OUTPUT DIRECTORY ----
OUTPUT_DIR = "results_day1"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------- IMAGE PREPROCESSING FUNCTION --------
def preprocess_image(path):
    """Load → grayscale → denoise → normalize → hotspot"""

    # Load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # Gaussian denoising
    denoised = cv2.GaussianBlur(img, (5, 5), 0)

    # Percentile-based normalization
    p2, p98 = np.percentile(denoised, (2, 98))
    norm = (denoised - p2) / (p98 - p2)
    norm = np.clip(norm, 0, 1)
    norm = (norm * 255).astype(np.uint8)

    # Hotspot extraction (Otsu)
    _, hotspot = cv2.threshold(norm, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return {
        "gray": img,
        "denoised": denoised,
        "normalized": norm,
        "hotspot": hotspot
    }


# ------------ MAIN PROCESSING LOOP ------------
for name, folder in DATASETS.items():
    print(f"\nProcessing dataset: {name}")
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    # Get image paths
    images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        images.extend(glob.glob(os.path.join(folder, ext)))

    # Take first 10 images
    images = images[:10]

    if len(images) == 0:
        print(f"⚠ No images found in {folder}")
        continue

    for idx, img_path in enumerate(images):
        processed = preprocess_image(img_path)
        if processed is None:
            continue

        cv2.imwrite(os.path.join(out_dir, f"{idx}_gray.png"), processed["gray"])
        cv2.imwrite(os.path.join(out_dir, f"{idx}_denoise.png"), processed["denoised"])
        cv2.imwrite(os.path.join(out_dir, f"{idx}_norm.png"), processed["normalized"])
        cv2.imwrite(os.path.join(out_dir, f"{idx}_hot.png"), processed["hotspot"])

    print(f"✔ Saved 10 processed samples to {out_dir}")

print("\n🎉 DAY 1 PREPROCESSING COMPLETE!")
print("All outputs saved in:", OUTPUT_DIR)
