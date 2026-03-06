import os
import csv
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Point this to wherever you unzipped EuroSAT
EUROSAT_DIR = Path("../EuroSAT")   # adjust if EuroSAT is in a different location

OUTPUT_TRAIN_CSV = Path("data/train.csv")
OUTPUT_VAL_CSV   = Path("data/val.csv")

TRAIN_SAMPLES_PER_CLASS = 40   # 40 x 10 classes = 400 training samples
VAL_SAMPLES_PER_CLASS   = 10   # 10 x 10 classes = 100 validation samples
# ──────────────────────────────────────────────────────────────────────────────

# Create the output data folder if it doesn't exist
OUTPUT_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)

# Map each class folder name to a numeric class ID
class_folders = sorted([f for f in EUROSAT_DIR.iterdir() if f.is_dir()])
class_to_id   = {folder.name: idx for idx, folder in enumerate(class_folders)}

print("Classes found:")
for name, idx in class_to_id.items():
    print(f"  {idx}: {name}")

train_rows = []
val_rows   = []

for folder in class_folders:
    images = sorted(folder.glob("*.jpg"))

    if len(images) == 0:
        print(f"WARNING: No .jpg images found in {folder} — skipping")
        continue

    # Take first N for train, next N for val
    train_images = images[:TRAIN_SAMPLES_PER_CLASS]
    val_images   = images[TRAIN_SAMPLES_PER_CLASS:TRAIN_SAMPLES_PER_CLASS + VAL_SAMPLES_PER_CLASS]

    for img in train_images:
        train_rows.append({
            "image":    str(img.resolve()),
            "label":    str(img.resolve()),   # same file used as proxy label
            "class_id": class_to_id[folder.name],
            "class_name": folder.name
        })

    for img in val_images:
        val_rows.append({
            "image":    str(img.resolve()),
            "label":    str(img.resolve()),
            "class_id": class_to_id[folder.name],
            "class_name": folder.name
        })

# Write CSVs
fieldnames = ["image", "label", "class_id", "class_name"]

with open(OUTPUT_TRAIN_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_rows)

with open(OUTPUT_VAL_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(val_rows)

print(f"\nDone!")
print(f"  Training samples : {len(train_rows)} → saved to {OUTPUT_TRAIN_CSV}")
print(f"  Validation samples: {len(val_rows)}  → saved to {OUTPUT_VAL_CSV}")
print(f"\nFirst 3 rows of train.csv:")
for row in train_rows[:3]:
    print(f"  {row}")