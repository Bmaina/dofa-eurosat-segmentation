\# DOFA Segmentation Pipeline — EuroSAT Sentinel-2



End-to-end geospatial deep learning pipeline for land cover classification

using the DOFA (Dynamic One-For-All) foundation model on EuroSAT Sentinel-2 imagery.



\## What This Does



\- Classifies Sentinel-2 satellite image patches into 10 land cover classes

\- Uses a pretrained 140M parameter Vision Transformer (DOFA base) encoder

\- Encoder-Neck-Decoder architecture: DOFA encoder → UperNet neck → FCN head

\- Trained with DiceLoss, Adam optimizer, ReduceLROnPlateau scheduling

\- Built on the NRCan geo-deep-learning framework



\## Land Cover Classes



| ID | Class |

|----|-------|

| 0  | AnnualCrop |

| 1  | Forest |

| 2  | HerbaceousVegetation |

| 3  | Highway |

| 4  | Industrial |

| 5  | Pasture |

| 6  | PermanentCrop |

| 7  | Residential |

| 8  | River |

| 9  | SeaLake |



\## Architecture

```

Sentinel-2 Image (3 bands: R, G, B)

&nbsp;       ↓

DOFA Encoder (ViT-Base, 768-dim embeddings, pretrained, frozen)

&nbsp;       ↓

UperNet Neck (multi-scale feature fusion)

&nbsp;       ↓

FCN Head (per-pixel class prediction)

&nbsp;       ↓

Segmentation Map (10 classes)

```



\## Setup

```bash

git clone https://github.com/yourusername/dofa-eurosat-segmentation

cd dofa-eurosat-segmentation

python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt

pip install -e .

```



\## Data Preparation



Download EuroSAT dataset and run:

```bash

python prepare\_data.py

python convert\_to\_tif.py

python create\_masks.py

```



\## Training

```bash

python geo\_deep\_learning/train.py fit --config configs/dofa\_config\_RGB.yaml

```



\## Key Results



\- Dataset: EuroSAT Sentinel-2 (400 train / 100 val / 100 test patches)

\- Model: DOFA-Base (140M params, 35M trainable)

\- Epochs: 5

\- Loss: DiceLoss (multiclass)



\## References



\- \[geo-deep-learning](https://github.com/NRCan/geo-deep-learning) — NRCan framework

\- \[DOFA](https://huggingface.co/earthflow/DOFA) — Foundation model

\- \[EuroSAT](https://github.com/phelber/EuroSAT) — Dataset

