# ThermoMap – Multi-Region Thermal Imaging Analysis System

ThermoMap is an end-to-end **thermal imaging analysis pipeline** designed to extract **non-diagnostic thermal insights** from:

- **Breast thermograms**  
- **Facial thermal images**  
- **Feet plantar thermograms**

The system combines **classical computer vision**, **deep learning segmentation**, and **automated PDF report generation** to create a complete, clinic-style, multi-region thermal analytics tool.

> **ThermoMap is NOT a medical diagnostic tool.**  
> It is a research, imaging-analysis, and engineering demonstration project.

---

## Key Features

### **Breast Thermal Module**
- Pose-aware breast ROI extraction (frontal, left oblique, right oblique)
- Torso segmentation and cleaning
- Hotspot extraction and temperature statistics
- Asymmetry heatmaps
- Detailed breast thermal summaries

### **Face Thermal Module**
- Robust face isolation (removes UI overlays, icons, temperature scales)
- Facial ROI extraction
- Left/right thermal asymmetry analysis
- Hotspot detection & temperature mapping
- Cleaning of noisy thermal frames

### **Feet Thermal Module (Hybrid CV + Deep Learning)**
- **UNet-based thermal feet segmentation** (trained on custom masks)
- Automatic left/right foot detection with k-means splitting
- Toe-safe ROI extraction
- Plantar hotspot extraction
- L–R asymmetry scoring
- Temperature features (mean, max, variance, gradient strength)

### **Reporting Engine**
- Fully automated **multi-page PDF report generation**
- Includes:
  - Personas for each dummy patient
  - ROIs, heatmaps, segmentation masks
  - Summary tables with metrics
- Automatically generates **10 dummy patients** for demonstration

---

## Project Structure

The project is organized into several main directories and key files:

| Directory/File | Description |
| :--- | :--- |
| **ThermoMap/** | **Root directory** of the project. |
| **data/** | Contains raw or pre-processed input data. |
| **sample_inputs/** | Directory for holding example or test thermal image inputs. |
| **src/** | **Source code** for various operations, including preprocessing, model training, I/O, and report generation. |
| `preprocess_day1.py` | Script likely used for initial data preparation or cleaning for the first day's run. |
| `breast_io.py` | Code dedicated to processing or analyzing breast thermal images. |
| `face_io.py` | Input/Output handling for face thermal images. |
| `feet_io.py` | Input/Output handling for feet thermal images. |
| `train_feet_unet.py` | Script for **training a UNet model** specifically for feet thermal image segmentation/analysis. |
| `build_dummy_patients.py` | Script to generate synthetic/placeholder patient data. |
| `generate_reports_from_json.py` | Script to create final reports from processed data stored in a JSON format. |
| **models/** | Stores trained machine learning models. |
| `feet_unet.pth` | **Pre-trained UNet model weights** for feet analysis. |
| **results_day3/** | Stores the output results from processing runs, specifically for "Day 3". |
| **breast/** | Processed thermal analysis results for the breast. |
| **face/** | Processed thermal analysis results for the face. |
| **feet/** | Processed thermal analysis results for the feet. |
| **reports/** | Stores generated patient reports (e.g., PDF format). |
| `CASE_01.pdf` | An example of a generated final report. |
| `dummy_patients.json` | JSON file containing the structure or data for dummy patients. |
| `requirements.txt` | Lists the necessary Python packages and their versions for the project. |
| `README.md` | This file, providing an overview of the project structure. |

---

## Setup Instructions

### Clone the repository:
```bash
git clone https://github.com/username-usr/ThermoMap.git
cd ThermoMap
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the pipeline:
```bash
python src/preprocess_day1.py
python src/breast_io.py
python src/face_io.py
python src/feet_io.py
```

### Generate a dummy patient detail:
```bash
pyhton src/dummy_patient.py
```
#### Output:
- `dummy_patients.json`
- 10 simulated patients with persona + modality assignments

## Generate PDFs report
```bash
python src/report.py
```
This create the reports at `report/` directory.

**Each PDF includes:**

- Patient details
- Breast/face/feet analysis
- ROIs, heatmaps, asymmetry maps
- Summary tables
- Final interpretation page

## License

This project is licensed under the MIT License.

## Author
- Ashwin, 3rd year UG @ Amrita University, Banglore.
- GitHub : https://github.com/username-usr
- Gmail  : ashwinvenkat2408@gmail.com

## Purpose of Project

**ThermoMap demonstrates engineering skills in:**

- Medical imaging pipelines
- Thermal image processing
- Deep learning segmentation
- Feature engineering
- Report generation
- Modular Python application development

## Contributions

Contributions, suggestions, and improvements are always welcome.
Feel free to open an issue or pull request.
Sign Off!!!
