from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import from_bounds

EUROSAT_DIR = Path("EuroSAT/2750")
OUTPUT_DIR = Path("data/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class_folders = sorted([f for f in EUROSAT_DIR.iterdir() if f.is_dir()])
class_to_id = {folder.name: idx for idx, folder in enumerate(class_folders)}

converted = 0
for folder in class_folders:
    images = sorted(folder.glob("*.jpg"))
    samples = images[:50]  # 40 train + 10 val per class
    for img_path in samples:
        out_path = OUTPUT_DIR / f"{img_path.stem}.tif"
        if out_path.exists():
            continue
        img = np.array(Image.open(img_path)).astype(np.uint8)
        if img.ndim == 2:
            img = np.stack([img, img, img])
        else:
            img = img.transpose(2, 0, 1)
        h, w = img.shape[1], img.shape[2]
        transform = from_bounds(0, 0, 1, 1, w, h)
        with rasterio.open(
            out_path, "w",
            driver="GTiff",
            height=h, width=w,
            count=img.shape[0],
            dtype=np.uint8,
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(img)
        converted += 1
        if converted % 50 == 0:
            print(f"  Converted {converted} images...")

print(f"\nDone! Converted {converted} images to {OUTPUT_DIR}")