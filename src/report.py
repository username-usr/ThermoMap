import os
import json
from pathlib import Path

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors


# ===============================================================
# CONFIG
# ===============================================================
RESULTS_ROOT = "results_day3"
REPORTS_DIR = "reports"
JSON_FILE = "dummy_patients.json"

os.makedirs(REPORTS_DIR, exist_ok=True)

styles = getSampleStyleSheet()
title_style = styles["Title"]
h1_style = styles["Heading1"]
h2_style = styles["Heading2"]
body_style = styles["BodyText"]


# ===============================================================
# HELPERS
# ===============================================================

def add_image(flow, path, width=400):
    """Insert image if file exists."""
    if os.path.exists(path):
        flow.append(Image(path, width=width, height=width * 0.75))
    else:
        flow.append(Paragraph(f"[Missing file: {path}]", body_style))
    flow.append(Spacer(1, 12))


def read_summary(stem, modality):
    """Reads *_summary.txt for a given stem & modality."""
    folder = Path(RESULTS_ROOT) / modality
    summary_path = folder / f"{stem}_summary.txt"

    if not summary_path.exists():
        return []

    items = []
    with open(summary_path, "r") as f:
        for line in f:
            if ":" in line:
                k, v = line.split(":", 1)
                items.append((k.strip(), v.strip()))
    return items


def summary_table(flow, items, title="Summary"):
    """Create a neat table for summary metrics."""
    if not items:
        flow.append(Paragraph(f"{title}: No summary available.", body_style))
        flow.append(Spacer(1, 20))
        return

    flow.append(Paragraph(title, h2_style))

    table_data = [(k, v) for k, v in items]
    tbl = Table(table_data, colWidths=[180, 280])

    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
    ]))

    flow.append(tbl)
    flow.append(Spacer(1, 20))


# ===============================================================
# SMART BREAST FILE FINDER
# ===============================================================
def find_breast_images(stem, bdir):
    """
    Detect correct filenames automatically.
    Returns:
       gray_path, roi_path, heatmap_path, hotspots_path
    """

    # gray
    gray_path = bdir / f"{stem}_gray.png"

    # ROI candidates
    roi_candidates = [
        f"{stem}_breast_roi.png",
        f"{stem}_roi.png",
        f"{stem}_oblleft_roi.png",
        f"{stem}_oblright_roi.png",
        f"{stem}_anterior_roi.png",
    ]

    # asymmetry heatmap candidates
    heatmap_candidates = [
        f"{stem}_breast_asymmetry_heatmap.png",
        f"{stem}_asymmetry_heatmap.png",
        f"{stem}_oblleft_heatmap.png",
        f"{stem}_oblright_heatmap.png",
        f"{stem}_anterior_heatmap.png",
    ]

    # hotspots
    hotspots_candidates = [
        f"{stem}_hotspots.png",
        f"{stem}_hotspot.png",
        f"{stem}_oblleft_hotspots.png",
        f"{stem}_oblright_hotspots.png",
        f"{stem}_anterior_hotspots.png",
    ]

    roi_path = next((bdir / f for f in roi_candidates if (bdir / f).exists()), None)
    heatmap_path = next((bdir / f for f in heatmap_candidates if (bdir / f).exists()), None)
    hot_path = next((bdir / f for f in hotspots_candidates if (bdir / f).exists()), None)

    return gray_path, roi_path, heatmap_path, hot_path


# ===============================================================
# REPORT GENERATOR
# ===============================================================
def generate_report_for_patient(patient):
    """
    patient structure:
    {
      "patient_id": "CASE_01",
      "persona": {...},
      "modalities": {"breast": "...", "face": "...", "feet": "..."},
      "diseased": True/False
    }
    """

    patient_id = patient["patient_id"]
    persona = patient["persona"]
    mods = patient["modalities"]

    pdf_path = os.path.join(REPORTS_DIR, f"{patient_id}.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    flow = []

    # =====================================================
    # PAGE 1: HEADER + PERSONA
    # =====================================================
    flow.append(Paragraph("ThermoMap Thermal Imaging Report", title_style))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style))
    flow.append(Paragraph(f"<b>Name:</b> {persona['name']}", body_style))
    flow.append(Paragraph(f"<b>Age:</b> {persona['age']}", body_style))
    flow.append(Paragraph(f"<b>Gender:</b> {persona['gender']}", body_style))
    flow.append(Paragraph(f"<b>Risk Level:</b> {persona['risk_level']}", body_style))
    flow.append(Paragraph(f"<b>Notes:</b> {persona['notes']}", body_style))
    flow.append(Spacer(1, 20))

    flow.append(PageBreak())

    # =====================================================
    # BREAST SECTION (Females only)
    # =====================================================
    if persona["gender"] == "F" and mods.get("breast"):
        stem = mods["breast"]
        flow.append(Paragraph("Breast Thermal Analysis", h1_style))

        bdir = Path(RESULTS_ROOT) / "breast"
        gray_path, roi_path, heatmap_path, hot_path = find_breast_images(stem, bdir)

        add_image(flow, str(gray_path))
        add_image(flow, str(roi_path))
        add_image(flow, str(heatmap_path))
        add_image(flow, str(hot_path))

        items = read_summary(stem, "breast")
        summary_table(flow, items, title="Breast Summary")

        flow.append(PageBreak())

    # =====================================================
    # FACE SECTION
    # =====================================================
    if mods.get("face"):
        stem = mods["face"]
        flow.append(Paragraph("Face Thermal Analysis", h1_style))

        fdir = Path(RESULTS_ROOT) / "face"

        add_image(flow, str(fdir / f"{stem}_gray.png"))
        add_image(flow, str(fdir / f"{stem}_roi.png"))
        add_image(flow, str(fdir / f"{stem}_asymmetry_heatmap.png"))
        add_image(flow, str(fdir / f"{stem}_hotspots.png"))

        items = read_summary(stem, "face")
        summary_table(flow, items, title="Face Summary")

        flow.append(PageBreak())

    # =====================================================
    # FEET SECTION
    # =====================================================
    if mods.get("feet"):
        stem = mods["feet"]
        flow.append(Paragraph("Feet Thermal Analysis", h1_style))

        feet_dir = Path(RESULTS_ROOT) / "feet"

        add_image(flow, str(feet_dir / f"{stem}_gray.png"))
        add_image(flow, str(feet_dir / f"{stem}_feet_mask.png"))
        add_image(flow, str(feet_dir / f"{stem}_feet_roi.png"))
        add_image(flow, str(feet_dir / f"{stem}_left.png"))
        add_image(flow, str(feet_dir / f"{stem}_right.png"))
        add_image(flow, str(feet_dir / f"{stem}_asymmetry_heatmap.png"))
        add_image(flow, str(feet_dir / f"{stem}_plantar_hotspots.png"))

        items = read_summary(stem, "feet")
        summary_table(flow, items, title="Feet Summary")

        flow.append(PageBreak())

    # =====================================================
    # FINAL PAGE
    # =====================================================
    flow.append(Paragraph("System Overview", h1_style))
    flow.append(Paragraph(
        """
        ThermoMap is a multi-region thermal imaging analysis system that extracts 
        non-diagnostic thermal insights from breast, facial, and plantar thermograms. 
        The system combines classical image processing, thermal feature engineering, 
        and deep learning-based segmentation for accurate region-of-interest detection 
        and consistent thermal analysis.
        """,
        body_style
    ))

    doc.build(flow)
    print(f"[OK] Generated report: {pdf_path}")


# ===============================================================
# MAIN ENTRY
# ===============================================================
def main():
    if not os.path.exists(JSON_FILE):
        print(f"ERROR: Missing {JSON_FILE}. Run build_dummy_patients.py first.")
        return

    with open(JSON_FILE, "r") as f:
        patients = json.load(f)

    print(f"Generating reports for {len(patients)} dummy cases...")

    for p in patients:
        generate_report_for_patient(p)


if __name__ == "__main__":
    main()
