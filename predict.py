"""
predict.py
==========
Command-line inference interface for the DOFA EuroSAT segmentation model.

Takes a GeoTIFF or image file as input, runs DOFA segmentation,
and saves a classified GeoTIFF + PNG visualisation as output.

Usage:
    # Single image
    python predict.py --input path/to/image.tif --output outputs/prediction.tif

    # Batch — all .tif files in a folder
    python predict.py --input path/to/folder/ --output outputs/

    # With custom checkpoint
    python predict.py --input image.tif --checkpoint logs/.../model.ckpt

    # Save PNG visualisation alongside GeoTIFF
    python predict.py --input image.tif --output pred.tif --visualise

    # Specify wavelengths (default: Sentinel-2 RGB)
    python predict.py --input image.tif --wavelengths 0.665 0.549 0.481

Examples:
    python predict.py --input data/test_patch.tif --output outputs/pred.tif --visualise
    python predict.py --input data/patches/ --output outputs/preds/ --visualise
"""

import sys
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = Path(
    r"C:\geo-deep-learning\logs\gdl_experiment\version_11\checkpoints"
    r"\model-epoch=00-val_loss=0.141.ckpt"
)
DEFAULT_WAVELENGTHS = [0.665, 0.549, 0.481]   # Sentinel-2 RGB
PATCH_SIZE          = 64
MEAN                = [0.485, 0.456, 0.406]
STD                 = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Veg", "Highway",
    "Industrial", "Pasture", "Permanent Crop", "Residential",
    "River", "Sea / Lake",
]
CLASS_COLORS = [
    (0.78, 0.52, 0.25), (0.13, 0.37, 0.13), (0.42, 0.68, 0.35),
    (0.65, 0.60, 0.45), (0.45, 0.45, 0.55), (0.55, 0.75, 0.40),
    (0.25, 0.55, 0.15), (0.75, 0.60, 0.50), (0.20, 0.50, 0.80),
    (0.10, 0.30, 0.70),
]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> torch.nn.Module:
    """Load DOFA model from checkpoint. Returns model in eval mode."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Update DEFAULT_CHECKPOINT at the top of predict.py"
        )

    sys.path.insert(0, r"C:\geo-deep-learning")
    import segmentation_models_pytorch as smp
    from geo_deep_learning.tasks_with_models.segmentation_dofa import SegmentationDOFA

    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hp    = ckpt["hyper_parameters"]

    model = SegmentationDOFA(
        encoder       = hp["encoder"],
        pretrained    = False,
        image_size    = tuple(hp["image_size"]),
        num_classes   = hp["num_classes"],
        max_samples   = hp.get("max_samples", 2),
        loss          = smp.losses.DiceLoss(mode="multiclass", from_logits=True),
        class_labels  = hp.get("class_labels"),
        class_colors  = hp.get("class_colors"),
        freeze_layers = hp.get("freeze_layers"),
    )
    model.configure_model()
    state = {k.removeprefix("model."): v for k, v in ckpt["state_dict"].items()}
    model.model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_image(image_path: Path) -> tuple:
    """
    Load an image file (GeoTIFF or PNG/JPG) and return
    (array float32 CHW 0-1, rasterio profile or None).
    """
    image_path = Path(image_path)
    profile    = None

    if image_path.suffix.lower() in (".tif", ".tiff"):
        with rasterio.open(image_path) as src:
            data    = src.read().astype(np.float32)
            profile = src.profile
        # Normalise to 0-1
        if data.max() > 1.0:
            p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
            data    = np.clip((data - p2) / (p98 - p2 + 1e-6), 0, 1)
        # Use first 3 bands as RGB
        if data.shape[0] > 3:
            data = data[:3]
        elif data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
    else:
        pil   = Image.open(image_path).convert("RGB")
        data  = np.array(pil).transpose(2, 0, 1).astype(np.float32) / 255.0

    return data, profile


def preprocess(array: np.ndarray) -> torch.Tensor:
    """Normalise CHW float array to ImageNet stats tensor."""
    tf = T.Compose([
        T.Resize((PATCH_SIZE, PATCH_SIZE)),
        T.Normalize(mean=MEAN, std=STD),
    ])
    pil    = Image.fromarray((array.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8))
    tensor = tf(T.ToTensor()(pil))
    return tensor


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_patch(model, array: np.ndarray, wavelengths: list) -> np.ndarray:
    """
    Run DOFA on a single CHW patch. Returns class map (H, W) int.
    Uses sliding window if image is larger than PATCH_SIZE.
    """
    _, H, W = array.shape
    wv      = torch.tensor([wavelengths])

    if H <= PATCH_SIZE and W <= PATCH_SIZE:
        # Single forward pass
        tensor = preprocess(array).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor, wv)
            pred   = output.out.argmax(dim=1).squeeze(0).numpy()
        # Resize back to original
        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred     = np.array(pred_pil.resize((W, H), Image.NEAREST))
        return pred.astype(np.int32)

    # Sliding window for larger images
    stride    = PATCH_SIZE // 2
    conf_map  = np.zeros((H, W, len(CLASS_NAMES)), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, stride):
            for x in range(0, W - PATCH_SIZE + 1, stride):
                chip   = array[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                tensor = preprocess(chip).unsqueeze(0)
                out    = model(tensor, wv)
                probs  = out.out.softmax(dim=1).squeeze(0).permute(1, 2, 0).numpy()
                conf_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE]  += probs
                count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1.0

    conf_map /= np.maximum(count_map, 1.0)[:, :, np.newaxis]
    return conf_map.argmax(axis=2).astype(np.int32)


# ── Output saving ─────────────────────────────────────────────────────────────

def save_prediction(pred: np.ndarray, out_path: Path, profile=None):
    """Save prediction as GeoTIFF, preserving CRS/transform if available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if profile is not None:
        p = profile.copy()
        p.update(count=1, dtype="int32", driver="GTiff", compress="lzw",
                 height=pred.shape[0], width=pred.shape[1])
        with rasterio.open(out_path, "w", **p) as dst:
            dst.write(pred[np.newaxis, :].astype(np.int32))
    else:
        H, W = pred.shape
        transform = from_bounds(0, 0, W, H, W, H)
        with rasterio.open(out_path, "w", driver="GTiff", count=1, dtype="int32",
                           width=W, height=H, transform=transform) as dst:
            dst.write(pred[np.newaxis, :].astype(np.int32))


def save_visualisation(image: np.ndarray, pred: np.ndarray, out_path: Path,
                       input_name: str = ""):
    """Save side-by-side RGB + segmentation PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rgb  = image.transpose(1, 2, 0).clip(0, 1)
    H, W = pred.shape
    seg_rgb = np.zeros((H, W, 3))
    for i, c in enumerate(CLASS_COLORS):
        seg_rgb[pred == i] = c

    legend = [mpatches.Patch(color=c, label=CLASS_NAMES[i])
              for i, c in enumerate(CLASS_COLORS)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("DOFA Segmentation — " + input_name, fontsize=12, fontweight="bold")
    axes[0].imshow(rgb);                           axes[0].set_title("Input");       axes[0].axis("off")
    axes[1].imshow(seg_rgb, interpolation="nearest"); axes[1].set_title("Prediction"); axes[1].axis("off")
    fig.legend(handles=legend, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), fontsize=8, framealpha=0.9)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_metadata(pred: np.ndarray, out_path: Path, input_path: Path,
                  elapsed: float, wavelengths: list):
    """Save prediction metadata as JSON."""
    unique, counts = np.unique(pred, return_counts=True)
    class_dist     = {CLASS_NAMES[int(u)]: int(c) for u, c in zip(unique, counts)
                      if int(u) < len(CLASS_NAMES)}
    dominant       = CLASS_NAMES[int(unique[counts.argmax()])]

    meta = {
        "input":          str(input_path),
        "shape":          list(pred.shape),
        "wavelengths_um": wavelengths,
        "dominant_class": dominant,
        "class_distribution": class_dist,
        "inference_time_s":   round(elapsed, 3),
        "model":          "DOFA-EuroSAT (val_loss=0.141)",
        "patch_size":     PATCH_SIZE,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DOFA EuroSAT segmentation inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",       required=True,
                        help="Input image (.tif/.png/.jpg) or folder of images")
    parser.add_argument("--output",      required=True,
                        help="Output .tif file or folder (for batch mode)")
    parser.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT),
                        help="Path to DOFA checkpoint (.ckpt)")
    parser.add_argument("--wavelengths", nargs=3, type=float,
                        default=DEFAULT_WAVELENGTHS,
                        metavar=("R", "G", "B"),
                        help="Sensor wavelengths in micrometres (default: Sentinel-2 RGB)")
    parser.add_argument("--visualise",   action="store_true",
                        help="Save PNG visualisation alongside GeoTIFF")
    parser.add_argument("--no-meta",     action="store_true",
                        help="Skip saving JSON metadata")
    return parser.parse_args()


def run_single(model, input_path: Path, output_path: Path,
               wavelengths: list, visualise: bool, save_meta: bool):
    """Run inference on a single image file."""
    print(f"  Input  : {input_path.name}")

    image, profile = load_image(input_path)
    t0             = time.time()
    pred           = predict_patch(model, image, wavelengths)
    elapsed        = time.time() - t0

    # Dominant class
    unique, counts = np.unique(pred, return_counts=True)
    dominant       = CLASS_NAMES[int(unique[counts.argmax()])]

    save_prediction(pred, output_path, profile)
    print(f"  Output : {output_path}")
    print(f"  Class  : {dominant}  ({elapsed:.2f}s)")

    if visualise:
        vis_path = output_path.with_suffix(".png")
        save_visualisation(image, pred, vis_path, input_path.stem)
        print(f"  Visual : {vis_path.name}")

    if save_meta:
        meta_path = output_path.with_suffix(".json")
        save_metadata(pred, meta_path, input_path, elapsed, wavelengths)
        print(f"  Meta   : {meta_path.name}")

    return pred, elapsed


def main():
    args        = parse_args()
    input_path  = Path(args.input)
    output_path = Path(args.output)
    checkpoint  = Path(args.checkpoint)
    wavelengths = args.wavelengths

    print("DOFA EuroSAT Segmentation — predict.py")
    print("-" * 40)
    print(f"Checkpoint : {checkpoint.name}")
    print(f"Wavelengths: {wavelengths} um")
    print()

    # Load model once
    print("Loading model...")
    model = load_model(checkpoint)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model ready — {total/1e6:.1f}M params")
    print()

    # Single file or batch
    if input_path.is_dir():
        exts   = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
        inputs = sorted([f for f in input_path.iterdir() if f.suffix.lower() in exts])
        if not inputs:
            print(f"No image files found in {input_path}")
            sys.exit(1)

        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Batch mode — {len(inputs)} files")
        print()

        total_time = 0
        for i, inp in enumerate(inputs, 1):
            out = output_path / (inp.stem + "_pred.tif")
            print(f"[{i}/{len(inputs)}]")
            _, elapsed = run_single(model, inp, out, wavelengths,
                                    args.visualise, not args.no_meta)
            total_time += elapsed
            print()

        print(f"Batch complete — {len(inputs)} files in {total_time:.1f}s")

    else:
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            sys.exit(1)
        run_single(model, input_path, output_path, wavelengths,
                   args.visualise, not args.no_meta)

    print("\nDone.")


if __name__ == "__main__":
    main()
