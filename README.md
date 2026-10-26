# ForgeAD — Self-Supervised Defect Detection by Forging Synthetic Anomalies

**Industrial anomaly detection without a single real defect label.**

The hardest problem in visual defect detection is not the model — it is the
data. Defective samples are rare, expensive to collect, and every new product
line starts from zero. ForgeAD tackles this with self-supervision: it
*forges* its own defects (cut-paste patches and scar-like strips) on
defect-free images, trains a network to tell forged regions from real ones,
and then detects genuine defects as "things the forger never made".

This is an independent re-implementation of **CutPaste** (Li et al., CVPR
2021), applied to the MVTec AD *toothbrush* category and compared head-to-head
with the PatchCore baseline from [VisionAI](../visionai-defect-detection)
(Project 2 of this portfolio).

## Results (measured, MVTec AD — toothbrush, 42 test images)

| Method | Image AUROC | Pixel AUROC |
|---|---|---|
| PatchCore baseline (VisionAI Project 2) | 0.9167 | 0.9267 |
| **ForgeAD (GDE scorer)** | **0.8417** | **0.9141** |
| ForgeAD (head ablation: 1 − P(normal)) | 0.8583 | — |

All numbers are computed on the same test split with the same metric
definitions — no cherry-picking, no mock data. ForgeAD trails the
baseline on image-level detection but comes within 1.3 points on pixel
localization, without ever seeing a real defect label. See `REPORT.md`
for the full write-up and the honest analysis of the gap.

## How it works

1. **Forge** (`src/cutpaste.py`) — two synthetic-anomaly families painted
   onto normal images only: *cut-paste* (a patch cut from one spot, jittered,
   pasted elsewhere) and *scar* (a long thin rotated strip, like a scratch).
2. **Learn** (`src/train.py`) — a ResNet-18 (ImageNet-pretrained, early layers
   frozen for CPU training) learns 3-way classification:
   normal / cut-paste / scar.
3. **Score** (`src/score_gde.py`) — normal-image embeddings are modelled as a
   Gaussian (Ledoit-Wolf shrinkage); test images are scored by Mahalanobis
   distance. Patch-level scoring gives localization maps.
4. **Evaluate** (`src/evaluate.py`) — image + pixel AUROC vs. ground-truth
   masks, with the PatchCore baseline side by side.

## Quickstart

```bash
# 1. environment (CPU-only)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. data (MVTec AD toothbrush, CC BY-NC-SA 4.0, research use only)
python scripts/download_mvtec.py        # or set FORGEAD_DATA to an existing copy

# 3. train (~1h on a modern CPU)
python -m src.train --epochs 60 --img-size 224

# 4. evaluate
python -m src.evaluate
```

## Repository layout

```
forgead/
├── src/
│   ├── cutpaste.py      # synthetic-anomaly forge (CutPaste + scar)
│   ├── data_mvtec.py    # MVTec toothbrush loader
│   ├── models.py        # ResNet-18 + 3-way forge head
│   ├── train.py         # self-supervised training
│   ├── score_gde.py     # Gaussian density scoring + anomaly maps
│   ├── evaluate.py      # image/pixel AUROC + baseline comparison
│   └── common/seed.py
├── scripts/download_mvtec.py
├── tests/               # smoke tests (no training needed)
├── REPORT.md            # research-style write-up
└── requirements.txt
```

## References

* C. Li, S. Sohn, J. Yoon, T. Pfister, "CutPaste: Self-Supervised Learning
  for Anomaly Detection and Localization", *Proc. CVPR*, 2021.
* P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for
  Unsupervised Anomaly Detection", *Proc. CVPR*, 2019.
* T. Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
  *Proc. CVPR*, 2022 (PatchCore — the baseline).

## License

Code: MIT. The MVTec AD dataset is **CC BY-NC-SA 4.0** (research /
non-commercial) and is downloaded separately — it is not shipped here.
