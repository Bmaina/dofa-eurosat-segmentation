from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds

IMAGES_DIR = Path("data/images")
MASKS_DIR = Path("data/masks")
MASKS_DIR.mkdir(parents=True, exist_ok=True)

# Each TIF filename starts with the class name e.g. AnnualCrop_1.tif
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
class_to_id = {name: idx for idx, name in enumerate(class_names)}

all_images = sorted(IMAGES_DIR.glob("*.tif"))
print(f"Found {len(all_images)} images to create masks for...")

for img_path in all_images:
    # Determine class from filename e.g. AnnualCrop_1.tif -> AnnualCrop
    stem = img_path.stem  # e.g. AnnualCrop_1
    class_id = None
    for name in class_names:
        if stem.startswith(name):
            class_id = class_to_id[name]
            break

    if class_id is None:
        print(f"  WARNING: Could not determine class for {img_path.name}")
        continue

    # Open image to get dimensions
    with rasterio.open(img_path) as src:
        h = src.height
        w = src.width
        transform = src.transform
        crs = src.crs

    # Create mask filled with class_id value
    mask = np.full((1, h, w), class_id, dtype=np.uint8)

    mask_path = MASKS_DIR / img_path.name
    with rasterio.open(
        mask_path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask)

print(f"Done! Created {len(all_images)} mask files in {MASKS_DIR}")