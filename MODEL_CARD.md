# Model Card — DOFA EuroSAT Land Cover Segmenter

## Model Summary

| Field | Value |
|---|---|
| **Model name** | DOFA EuroSAT Land Cover Segmenter |
| **Version** | v1.0 — epoch 00, val_loss 0.141 |
| **Task** | Semantic segmentation — 10-class land cover |
| **Architecture** | DOFA (Dynamic One-For-All) — ViT-Base/16 encoder + UPerNet decoder |
| **Parameters** | 140M total (105M encoder frozen, 35M decoder trainable) |
| **Input** | RGB satellite imagery, 64×64 pixels |
| **Output** | Per-pixel land cover class (0–9) |
| **Framework** | PyTorch Lightning |
| **Checkpoint** | `logs/gdl_experiment/version_11/checkpoints/model-epoch=00-val_loss=0.141.ckpt` |

---

## Intended Use

This model classifies land cover in satellite imagery into 10 EuroSAT categories. It is designed for:

- **Land cover mapping** from Sentinel-2 RGB imagery
- **Change detection** by comparing segmentation maps across time periods
- **Deforestation monitoring** — detecting Forest → other class transitions
- **Building damage assessment** — detecting Residential/Industrial → bare land transitions
- **Portfolio demonstration** of geospatial foundation model fine-tuning

### Primary users
Geospatial analysts, remote sensing researchers, GeoAI practitioners.

### Out-of-scope uses
- Very high resolution imagery below 5m (model trained on 64m EuroSAT patches)
- Hyperspectral data beyond 3 RGB bands without wavelength adaptation
- Real-time inference at scale without batching optimisation
- Medical, legal, or safety-critical decisions without human review

---

## Training Data

| Field | Value |
|---|---|
| **Dataset** | EuroSAT RGB |
| **Source** | Sentinel-2 multispectral imagery (ESA Copernicus) |
| **Coverage** | 34 European countries |
| **Patches** | 27,000 labelled 64×64 patches |
| **Classes** | 10 land cover categories (2,000–3,000 samples each) |
| **Split** | 80% train / 20% validation |
| **Resolution** | ~10m per pixel (Sentinel-2 RGB bands) |

---

## Classes

| ID | Class | Colour |
|---|---|---|
| 0 | Annual Crop | 🟫 Brown |
| 1 | Forest | 🟢 Dark Green |
| 2 | Herbaceous Vegetation | 🟩 Light Green |
| 3 | Highway | 🟨 Tan |
| 4 | Industrial | 🩶 Grey-Blue |
| 5 | Pasture | 💚 Mid Green |
| 6 | Permanent Crop | 🌿 Olive Green |
| 7 | Residential | 🟤 Dusty Rose |
| 8 | River | 🔵 Blue |
| 9 | Sea / Lake | 🔷 Dark Blue |

---

## Performance

| Metric | Value |
|---|---|
| **val_loss** (DiceLoss) | 0.141 |
| **Best epoch** | 0 |
| **Training epochs** | 1 (early stopping) |
| **Optimiser** | Adam (lr=1e-4) |
| **Loss function** | Dice Loss (multiclass) |

> **Note:** Evaluated on EuroSAT validation set (European land cover). Performance on out-of-distribution geographies (tropics, arid regions) may vary and should be validated before operational use.

---

## Architecture

```
Input: RGB patch (3, 64, 64) + wavelengths tensor [0.665, 0.549, 0.481] μm
  ↓
DOFA Encoder (ViT-Base/16, 105M params — FROZEN)
  Pretrained on multi-sensor satellite imagery
  Wavelength-conditioned patch embedding
  ↓
UPerNet Decoder (35M params — TRAINABLE)
  Multi-scale feature fusion
  ↓
Output: Class logits (10, 64, 64)
  ↓
argmax → Class map (64, 64) int
```

**Key design principle:** DOFA uses wavelength-conditioned embeddings, meaning the same encoder can process optical, SAR, and hyperspectral data by passing the appropriate wavelength values — enabling true multi-sensor transfer learning.

---

## Usage

### Quick start
```bash
python predict.py --input path/to/image.tif --output outputs/pred.tif --visualise
```

### Python API
```python
from predict import load_model, load_image, predict_patch

model  = load_model("logs/.../model-epoch=00-val_loss=0.141.ckpt")
image, profile = load_image("path/to/image.tif")
pred   = predict_patch(model, image, wavelengths=[0.665, 0.549, 0.481])
# pred: numpy array (H, W) with class IDs 0-9
```

### Batch inference
```bash
python predict.py --input data/patches/ --output outputs/preds/ --visualise
```

### With different sensor wavelengths
```bash
# PlanetScope
python predict.py --input image.tif --wavelengths 0.630 0.540 0.485

# Maxar VHR
python predict.py --input image.tif --wavelengths 0.660 0.546 0.478
```

---

## Limitations & Biases

- **Geographic bias:** EuroSAT covers only European land cover — tropical forests, savanna, and arid environments are underrepresented in training data
- **Resolution:** Optimised for ~10m Sentinel-2 resolution; very high resolution (<1m) imagery may not segment correctly without retraining
- **Seasonal variation:** No explicit seasonal conditioning — results may vary between summer and winter acquisitions
- **Cloud cover:** No cloud masking applied — cloudy pixels will be misclassified
- **Urban complexity:** Dense urban environments with mixed land use may show lower accuracy than rural areas

---

## Ethical Considerations

- This model was developed for research and portfolio demonstration purposes
- Damage assessment outputs (e.g. Gaza project) are derived from a model trained on European land cover and should **not** be used for operational humanitarian decisions without validation against ground truth data
- Deforestation estimates are indicative and should be validated against official sources before reporting

---

## Related Projects

| Project | How this model is used |
|---|---|
| [dofa-eurosat-segmentation](https://github.com/Bmaina/dofa-eurosat-segmentation) | Primary training and evaluation |
| [amazon-change-detection](https://github.com/Bmaina/amazon-change-detection) | Zero-shot transfer — Sentinel-2 Rondônia |
| [gaza-damage-assessment](https://github.com/Bmaina/gaza-damage-assessment) | Zero-shot transfer — Maxar VHR Gaza |

---

## Citation

```bibtex
@misc{dofa2024,
  title  = {DOFA: Dynamic One-For-All Foundation Model for Earth Observation},
  author = {Xiong, Zhitong and others},
  year   = {2024},
  url    = {https://huggingface.co/earthflow/DOFA}
}

@misc{eurosat2019,
  title  = {EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author = {Helber, Patrick and others},
  year   = {2019},
  url    = {https://github.com/phelber/EuroSAT}
}
```

---

## Model Details

| Field | Value |
|---|---|
| **Developed by** | Benson M. Gachaga |
| **Model type** | Fine-tuned geospatial foundation model |
| **Language** | Python 3.12 |
| **License** | MIT |
| **Contact** | linkedin.com/in/bensonmgachaga |
