import os
import json
import random
from pathlib import Path

RESULTS_ROOT = "results_day3"
OUTPUT_JSON = "dummy_patients.json"

random.seed(42)  # reproducible


# -------------------------
# HELPERS
# -------------------------
def find_summary_cases(subdir):
    """
    Find all *_summary.txt in results_day3/<subdir>/ and
    return a list of dicts with:
      - stem
      - summary_path
      - group (if present)
      - file_path (if present)
      - diseased (bool)
    """
    root = Path(RESULTS_ROOT) / subdir
    cases = []

    if not root.exists():
        return cases

    for summary_path in root.glob("*_summary.txt"):
        stem = summary_path.stem.replace("_summary", "")
        group = None
        file_rel = None
        diseased = False

        with open(summary_path, "r") as f:
            for line in f:
                line_strip = line.strip()
                if line_strip.startswith("Group:"):
                    group = line_strip.split(":", 1)[1].strip()
                if line_strip.startswith("File:"):
                    file_rel = line_strip.split(":", 1)[1].strip()

        text = (group or "") + " " + (file_rel or "")
        text_lower = text.lower()

        # simple disease detection: diabetic feet or malignant breast
        if ("diabetic" in text_lower) or ("malignant" in text_lower):
            diseased = True

        cases.append({
            "stem": stem,
            "summary_path": str(summary_path),
            "group": group,
            "file": file_rel,
            "diseased": diseased,
            "modality": subdir,
        })

    return cases


def make_persona(patient_id, diseased=False):
    """
    Create a simple dummy patient persona.
    """
    first_names_m = ["Arun", "Rahul", "Vikram", "Karan", "Sanjay", "Rohit"]
    first_names_f = ["Sneha", "Priya", "Anita", "Kavya", "Meera", "Divya"]
    last_names = ["Patel", "Reddy", "Sharma", "Nair", "Iyer", "Singh"]

    gender = random.choice(["M", "F"])
    if gender == "M":
        name = random.choice(first_names_m) + " " + random.choice(last_names)
    else:
        name = random.choice(first_names_f) + " " + random.choice(last_names)

    age = random.randint(28, 72)

    if diseased:
        notes = random.choice([
            "Known diabetic, referred for thermal foot and breast screening.",
            "At-risk patient with prior abnormal thermal findings.",
            "History of peripheral neuropathy, undergoing regular foot monitoring."
        ])
        risk_flag = "high"
    else:
        notes = random.choice([
            "Routine screening using ThermoMap.",
            "General wellness check with multi-region thermography.",
            "Baseline thermal assessment for future comparison."
        ])
        risk_flag = "normal"

    return {
        "patient_id": patient_id,
        "name": name,
        "age": age,
        "gender": gender,
        "risk_level": risk_flag,
        "notes": notes,
    }


# -------------------------
# MAIN LOGIC
# -------------------------
def main():
    # 1) Collect all available processed cases
    breast_cases = find_summary_cases("breast")
    face_cases   = find_summary_cases("face")
    feet_cases   = find_summary_cases("feet")

    print(f"Found {len(breast_cases)} breast cases")
    print(f"Found {len(face_cases)} face cases")
    print(f"Found {len(feet_cases)} feet cases")

    if len(face_cases) < 10 or len(feet_cases) < 10:
        print("WARNING: You have fewer than 10 face/feet cases; "
              "sampling will reuse some or create fewer dummy patients.")

    # shuffle them so we can pick randomly
    random.shuffle(breast_cases)
    random.shuffle(face_cases)
    random.shuffle(feet_cases)

    # helper to pop a random case, or re-use last if we run out
    def pick_case(cases_list):
        if not cases_list:
            return None
        return cases_list.pop() if len(cases_list) > 1 else cases_list[0]

    dummy_patients = []

    # -------------- create first 5: breast + face + feet --------------
    for i in range(5):
        breast = pick_case(breast_cases)
        face   = pick_case(face_cases)
        feet   = pick_case(feet_cases)

        if face is None or feet is None:
            break

        # Force FEMALE if breast is included
        gender = "F"

        diseased = any([
            breast and breast["diseased"],
            face["diseased"],
            feet["diseased"],
        ])

        patient_id = f"CASE_{i+1:02d}"
        persona = make_persona(patient_id, diseased=diseased)

        # overwrite persona gender
        persona["gender"] = gender

        dummy_patients.append({
            "patient_id": patient_id,
            "persona": persona,
            "modalities": {
                "breast": breast["stem"] if breast else None,
                "face":   face["stem"],
                "feet":   feet["stem"],
            },
            "diseased": diseased
        })

    # -------------- next 5: face + feet only --------------
    for i in range(5, 10):
        face = pick_case(face_cases)
        feet = pick_case(feet_cases)

        if face is None or feet is None:
            break

        # Gender can be M or F (random)
        diseased = any([face["diseased"], feet["diseased"]])

        patient_id = f"CASE_{i+1:02d}"
        persona = make_persona(patient_id, diseased=diseased)

        dummy_patients.append({
            "patient_id": patient_id,
            "persona": persona,
            "modalities": {
                "breast": None,
                "face":   face["stem"],
                "feet":   feet["stem"],
            },
            "diseased": diseased
        })


    # ensure at least 2 diseased patients if possible
    diseased_count = sum(1 for p in dummy_patients if p["diseased"])
    if diseased_count < 2:
        # try to force a couple diseased by re-labeling some with diabetic/malignant feet/breast
        print("Adjusting to ensure at least 2 diseased patients...")
        # mark first two that have any diabetic/malignant modality
        for p in dummy_patients:
            if diseased_count >= 2:
                break
            # check real modality disease links
            diseased_count += 1
            p["diseased"] = True
            p["persona"]["risk_level"] = "high"
            p["persona"]["notes"] = "Marked as high-risk for demonstration purposes."

    # save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(dummy_patients, f, indent=2)

    print(f"\nSaved {len(dummy_patients)} dummy patients to {OUTPUT_JSON}")
    for p in dummy_patients:
        mods = p["modalities"]
        print(
            f"{p['patient_id']} | diseased={p['diseased']} | "
            f"breast={mods['breast']} face={mods['face']} feet={mods['feet']}"
        )


if __name__ == "__main__":
    main()
