# 🛰️ DOFA-EuroSAT Segmentation

> **End-to-End GeoAI Pipeline for Land Cover Classification using Sentinel-2 Satellite Imagery**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-purple?logo=pytorch) ![HuggingFace](https://img.shields.io/badge/🤗-DOFA_Foundation_Model-yellow) ![Sentinel](https://img.shields.io/badge/🌍-Sentinel--2_EO_Data-green) ![Classes](https://img.shields.io/badge/🗺️-10_Land_Cover_Classes-orange) ![Params](https://img.shields.io/badge/140M-Parameters-lightgrey)

---

## 🌍 What Is This Project?

This project trains an AI model to look at satellite images and automatically identify what type of land is in each pixel, whether it is a forest, a river, a highway, farmland, or a city. This is called **semantic segmentation**, giving every pixel in an image a meaningful label.

Imagine looking at a satellite photo of the Earth from space. To a human, it is obvious which parts are forests, which are rivers, and which are cities. But teaching a computer to do this automatically at scale, across entire countries, updated every few days, is one of the most powerful capabilities in modern GeoAI.

This pipeline uses **Sentinel-2** satellite imagery from the European Space Agency, freely available, globally covering, and updated every 5 days, to train a deep learning model that can classify land cover with no human labelling required at inference time.

---

## 🎯 Why This Matters for GeoAI

Land cover classification from satellite imagery is a foundational capability that powers dozens of real-world applications:

| Use Case | Description |
|---|---|
| 🌊 **Flood Mapping** | Detect flooded areas in near real-time after disasters by comparing pre and post imagery — critical for emergency response. |
| 🔥 **Wildfire Damage** | Map burned areas after wildfires to quantify destruction, guide reforestation, and assess ecological impact. |
| 🏗️ **Urban Expansion** | Track how cities grow over time, monitor illegal construction, and support urban planning decisions. |
| 🌾 **Food Security** | Monitor crop types and agricultural land use to forecast yields and detect drought stress early. |
| ☮️ **Conflict & Peacekeeping** | Detect infrastructure damage, population displacement, and deforestation in conflict zones for UN operational intelligence. |
| 🌡️ **Climate Monitoring** | Track deforestation, glacier retreat, and land degradation as indicators of climate change over decades. |

---

## 📡 The Data — EuroSAT Sentinel-2

**What is Sentinel-2?** Sentinel-2 is a pair of Earth observation satellites operated by the European Space Agency (ESA) as part of the Copernicus programme. They orbit at 786km altitude, cover the entire Earth every 5 days, and capture images at 10-metre resolution, meaning each pixel represents a 10m × 10m square on the ground.

**What is EuroSAT?** EuroSAT is a benchmark dataset of 27,000 labelled Sentinel-2 image patches, each 64×64 pixels, covering 10 different land cover classes across Europe. It is widely used in the research community to train and evaluate geospatial machine learning models.

> Each patch is **64×64 pixels** at 10m resolution = **640m × 640m of real Earth surface** per image.

### 🗺️ 10 Land Cover Classes

| Class ID | Icon | Class Name |
|---|---|---|
| Class 0 | 🌾 | Annual Crop |
| Class 1 | 🌲 | Forest |
| Class 2 | 🌿 | Herbaceous Vegetation |
| Class 3 | 🛣️ | Highway |
| Class 4 | 🏭 | Industrial |
| Class 5 | 🐄 | Pasture |
| Class 6 | 🍇 | Permanent Crop |
| Class 7 | 🏘️ | Residential |
| Class 8 | 🏞️ | River |
| Class 9 | 🌊 | Sea / Lake |

---

## 🧠 The Model — DOFA Foundation Model

> **What is a Foundation Model?** A foundation model is a large AI model trained on massive amounts of data that can be reused and fine-tuned for many different tasks. Instead of training from scratch every time, you start with a model that already understands the world — similar to how a human expert brings prior knowledge to a new problem.

**DOFA** (Dynamic One-For-All) is a Vision Transformer foundation model specifically designed for Earth Observation data. It was pretrained on millions of satellite images across multiple sensors and wavelengths, meaning it already "knows" what forests, water bodies, and urban areas look like from space before training begins.

### Architecture — Encoder → Neck → Decoder

```
📸 Input Image          🧠 DOFA Encoder         🔀 UperNet Neck         🎯 FCN Head             🗺️ Output
Sentinel-2 Image    →   Vision Transformer   →   Multi-scale feature  →   Per-pixel class    →   Segmentation Map
64×64 × 3 bands         768-dim embeddings       fusion & alignment       prediction              64×64 pixels
(R, G, B)               105M params (frozen)     Learnable                35M params              10 classes
```

> The encoder is **frozen** (pretrained weights preserved) — only the neck and head are trained. This is called **transfer learning**.

### Why Freeze the Encoder?

The DOFA encoder has already learned powerful representations of satellite imagery from millions of images. Freezing it means we keep all that knowledge intact and only train the smaller decoder layers to map those representations to our specific 10 classes. This makes training faster, requires less data, and prevents overfitting.

### What Are Wavelengths?

DOFA is a dynamic model — it adapts its processing based on which spectral bands (wavelengths of light) are provided. We use three bands: Red (0.665μm), Green (0.549μm), and Blue (0.481μm) — the same bands the human eye sees.

---

## ⚙️ How Training Works — Step by Step

| Step | Name | Description |
|---|---|---|
| 1 | **Data Loading** | CSV files point to image and mask pairs. DataLoader batches 4 images at a time. |
| 2 | **Augmentation** | Random flips, rotations, and crops applied during training to increase diversity. |
| 3 | **Normalisation** | Pixel values scaled to a standard range using dataset mean and std statistics. |
| 4 | **Forward Pass** | Image passed through Encoder → Neck → Head to produce class probability maps. |
| 5 | **Loss Calculation** | DiceLoss compares predictions to ground truth masks. Lower = better predictions. |
| 6 | **Backpropagation** | Gradients flow back through the network. Only unfrozen layers are updated. |
| 7 | **Optimisation** | Adam optimiser updates weights. Learning rate reduced automatically when loss plateaus. |
| 8 | **Validation** | After each epoch, model evaluated on held-out val set. Best checkpoint saved. |

> **Why DiceLoss?** DiceLoss measures the overlap between predicted and actual class regions. It handles class imbalance better than accuracy — highways occupy far fewer pixels than forests, so standard accuracy would just ignore them.

### Training Configuration

```
Model         : DOFA-Base (Vision Transformer)
Parameters    : 140M total | 35M trainable | 105M frozen
Dataset       : EuroSAT Sentinel-2
Train samples : 400 patches (40 per class)
Val samples   : 100 patches (10 per class)
Batch size    : 4 images per step
Epochs        : 5
Loss function : DiceLoss (multiclass)
Optimiser     : Adam (lr=6e-5)
Scheduler     : ReduceLROnPlateau (patience=3)
Hardware      : CPU (no GPU required)
```

---

## 📊 Results & Model Performance

| Metric | Value |
|---|---|
| Total Parameters | 140M |
| Trainable Parameters | 35M |
| Training Samples | 400 |
| Training Epochs | 5 |

> **Why These Numbers Matter:** Training a 140M parameter model from scratch would require millions of images and weeks of GPU time. By using DOFA's pretrained weights and only training 35M parameters (25% of the model), we achieved a functional segmentation pipeline in minutes on a standard laptop CPU — demonstrating the power of transfer learning with foundation models.

The model uses **Mean Intersection over Union (mIoU)** as its primary evaluation metric. IoU measures how much the predicted region for each class overlaps with the actual ground truth region. A perfect prediction scores 1.0, random guessing scores close to 0.

---

## 🚀 How to Reproduce This Pipeline

### 1. Clone and Set Up

```bash
git clone https://github.com/Bmaina/dofa-eurosat-segmentation.git
cd dofa-eurosat-segmentation
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download and Prepare Data

```bash
# Download EuroSAT from https://madm.dfki.de/files/sentinel/EuroSAT.zip
# Extract into the project folder, then run:
python prepare_data.py      # Creates train/val/test CSV files
python convert_to_tif.py   # Converts JPG patches to GeoTIFF format
python create_masks.py     # Creates single-band class ID mask TIFs
```

### 3. Train the Model

```bash
python geo_deep_learning/train.py fit \
  --config configs/dofa_config_RGB.yaml
```

### 4. Expected Training Output

```
Seed set to 42
GPU available: False, used: False        # Runs on CPU
Created dataset for trn split with 400 patches
Created dataset for val split with 100 patches
Downloading DOFA weights (402MB)...      # Only on first run

  | Name   | Type                  | Params
  | model  | DOFASegmentationModel | 140 M
  35.0 M   Trainable params
  105 M    Non-trainable params (frozen encoder)

Epoch 1/5: train_loss=0.94 val_loss=0.88
Epoch 2/5: train_loss=0.89 val_loss=0.84
...
```

---

## 🛠️ Technology Stack

`Python 3.12` · `PyTorch 2.x` · `PyTorch Lightning` · `HuggingFace Hub` · `Rasterio` · `TorchMetrics` · `segmentation-models-pytorch` · `TorchGeo` · `MLflow` · `NumPy` · `Pandas` · `GeoPandas`

---

## 🎯 Relevance to UN GeoAI Operations

This pipeline directly demonstrates capabilities required in operational GeoAI roles within UN peacekeeping and humanitarian contexts:

| Capability | Description |
|---|---|
| 🏗️ **Damage Assessment** | The segmentation approach used here is directly applicable to building footprint extraction and infrastructure damage detection from VHR imagery. |
| 🔄 **Change Detection** | By running the model on pre and post-event imagery, changed pixels can be identified — the foundation of conflict damage and disaster impact mapping. |
| 📦 **Scalable Pipelines** | The modular encoder-neck-decoder architecture and CSV-based data pipeline are designed for operational deployment across large EO datasets. |

---

## 📚 References

- **geo-deep-learning** — NRCan framework: [github.com/NRCan/geo-deep-learning](https://github.com/NRCan/geo-deep-learning)
- **DOFA Foundation Model** — [huggingface.co/earthflow/DOFA](https://huggingface.co/earthflow/DOFA)
- **EuroSAT Dataset** — Helber et al. (2019), IEEE JSTARS
- **Sentinel-2** — ESA Copernicus Programme: [sentinel.esa.int](https://sentinel.esa.int)
- **UperNet** — Xiao et al. (2018), Unified Perceptual Parsing

---

<div align="center">
  <strong>Built by Benson M. Gachaga</strong> — Data Scientist | GeoAI Practitioner | Remote Sensing Specialist<br/>
  MBA · M.S. Geoinformation & Earth Observation · PMP · Microsoft Certified Power BI<br/><br/>
  <a href="https://linkedin.com/in/bensonmgachaga">LinkedIn</a> &nbsp;·&nbsp; <a href="https://github.com/Bmaina">GitHub</a>
</div>
