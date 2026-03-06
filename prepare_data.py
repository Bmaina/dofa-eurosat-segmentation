import csv
from pathlib import Path

IMAGES_DIR = Path("data/images")
MASKS_DIR  = Path("data/masks")
OUTPUT_TRAIN_CSV = Path("data/trn.csv")
OUTPUT_VAL_CSV   = Path("data/val.csv")
OUTPUT_TEST_CSV  = Path("data/tst.csv")
TRAIN_SPLIT = 400
VAL_SPLIT   = 100

all_images = sorted(IMAGES_DIR.glob("*.tif"))
train_images = all_images[:TRAIN_SPLIT]
val_images   = all_images[TRAIN_SPLIT:TRAIN_SPLIT + VAL_SPLIT]
test_images  = val_images  # reuse val as test

def write_csv(path, images):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        for img in images:
            mask = MASKS_DIR / img.name
            writer.writerow([str(img.resolve()), str(mask.resolve())])

write_csv(OUTPUT_TRAIN_CSV, train_images)
write_csv(OUTPUT_VAL_CSV,   val_images)
write_csv(OUTPUT_TEST_CSV,  test_images)

print(f"Done!")
print(f"  Train : {len(train_images)} -> {OUTPUT_TRAIN_CSV}")
print(f"  Val   : {len(val_images)}   -> {OUTPUT_VAL_CSV}")
print(f"  Test  : {len(test_images)}  -> {OUTPUT_TEST_CSV}")