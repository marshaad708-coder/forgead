# ForgeAD: Self-Supervised Defect Detection by Forging Synthetic Anomalies

**Arshaad — October 2026**
*Project 3 of my portfolio for the China masters application. It builds
directly on VisionAI (Project 2).*

## 1. The problem I wanted to solve

In my last project I built a defect detector for factory products, and it
worked well. But one thing kept bothering me: every method I tried needed
either labeled defective samples (which factories almost never have) or a
big pile of normal images to model "normality". For a new production line,
you start with a handful of normal images and zero defects. So I asked
myself a simple question — what if the model could manufacture its own
training signal? What if it learned what a defect looks like by forging
fake ones on normal images, without ever seeing a real defect?

That question became this project.

## 2. Background

Two pieces of prior work shaped what I built:

* **PatchCore** (Roth et al., CVPR 2022) was the baseline from my Project 2.
  It stores patch embeddings of normal images in a memory bank and flags
  anything far from the bank as anomalous. On the MVTec toothbrush set it
  gave me image AUROC **0.9167** and pixel AUROC **0.9267**. I kept the
  exact same data split and metric definitions so the comparison here is
  apples to apples.
* **CutPaste** (Li et al., CVPR 2021) is the idea I re-implemented from
  scratch: paint synthetic anomalies onto normal images, train a network to
  classify normal vs. forged, then score real images with a Gaussian fit
  over the learned embeddings. The paper reports around 0.95 AUROC averaged
  over MVTec. I wanted to see how far an independent implementation gets on
  a single category, on CPU, with my own code.

## 3. What I built

The pipeline has four parts, all in `src/`:

1. **The forge** (`cutpaste.py`). Two families of fake defects, applied
   only to defect-free training images. *Cut-paste*: cut a patch
   (2–15% of the image area), jitter its brightness and color, rotate it,
   and paste it somewhere else. *Scar*: cut a long thin strip, rotate it
   randomly, paste it back — it looks roughly like a scratch. No real
   defect label is used anywhere.
2. **Training** (`train.py`). A ResNet-18, pretrained on ImageNet, learns
   a 3-way classification: normal / cut-paste / scar. I froze the early
   layers (stem through layer2) to keep training feasible on CPU — the
   paper fine-tunes everything, so this is a documented shortcut on my
   side. Model selection used forged copies of 6 held-out normal images,
   which means no real defect ever influenced training. 60 epochs,
   224×224 input, Adam with a cosine schedule. Training accuracy reached
   1.000; I also made the script save a checkpoint every epoch after a
   machine restart once killed a 38-epoch run.
3. **Scoring** (`score_gde.py`). I fit a Gaussian (Ledoit-Wolf shrinkage)
   over the 512-d embeddings of the 60 normal training images and score
   test images by Mahalanobis distance — far from normal means anomalous.
   For localization I fit one shared Gaussian over all patch positions of
   the dense feature map, score each position, and upsample the map. I
   also tried a simpler ablation: just use `1 − P(normal)` from the
   classifier head, with no density fitting at all.
4. **Evaluation** (`evaluate.py`). Image-level and pixel-level AUROC on
   the 42 test images (12 good, 30 defective, 30 ground-truth masks),
   printed next to the PatchCore baseline.

## 4. Experiments

Dataset: MVTec AD, *toothbrush* — 60 normal images for training; 12 normal
+ 30 defective for testing. Everything ran on CPU. The only thing I
compared against is my own measured PatchCore baseline from Project 2.

## 5. Results

| Method | Image AUROC | Pixel AUROC |
|---|---|---|
| PatchCore baseline (Project 2, measured) | 0.9167 | 0.9267 |
| **ForgeAD (Gaussian scorer)** | **0.8417** | **0.9141** |
| ForgeAD (head ablation: 1 − P(normal)) | 0.8583 | — |

I am reporting this exactly as measured: ForgeAD does not beat the
baseline on image-level detection. I would rather show an honest negative
result than tune the numbers. A few things are still worth noting. Pixel
localization lands within 1.3 points of the baseline (0.9141 vs 0.9267),
which is decent for a model that never saw a real defect. And the simple
head ablation beats the Gaussian scorer (0.8583 vs 0.8417) — fitting a
Gaussian in 512 dimensions on 60 images is noisy, and the classifier head
turned out better calibrated than the density fit.

Why the gap to PatchCore? Four reasons, as far as I can tell:

1. **Too few normal images for the density fit.** Sixty samples in 512
   dimensions is thin, even with shrinkage.
2. **Frozen early layers.** My CPU shortcut; the paper's full fine-tuning
   adapts low-level texture filters to the product, which likely matters
   for toothbrush bristles.
3. **The forgeries are crude.** Hand-made cut-paste patches and scars do
   not really look like the thin bristle defects in this category. There
   is a synthetic-to-real gap and it shows.
4. **The pretext task was too easy.** Validation accuracy hit 1.0000 by
   epoch 7 and stayed there, so the network stopped learning richer
   features early.

## 6. What I would do next

* Train on a GPU with all layers unfrozen, following the paper exactly.
* Replace the hand-made forge with a learned one — a small diffusion
  model conditioned on normal images would make far more realistic
  synthetic defects.
* Test on more MVTec categories (textures like carpet and leather behave
  very differently from objects like toothbrush).
* Try harder pretext tasks so the representations keep improving past
  the first few epochs.

## 7. References

1. C. Li, S. Sohn, J. Yoon, T. Pfister, "CutPaste: Self-Supervised Learning
   for Anomaly Detection and Localization", *Proc. CVPR*, 2021.
2. T. Roth, L. Pemula, J. Zepeda, B. Schölkopf, T. Brox, P. Gehler,
   "Towards Total Recall in Industrial Anomaly Detection", *Proc. CVPR*, 2022.
3. P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, "MVTec AD — A
   Comprehensive Real-World Dataset for Unsupervised Anomaly Detection",
   *Proc. CVPR*, 2019.
4. K. Song, Y. Yan, "A noise robust method based on completed local binary
   patterns for hot-rolled steel strip surface defects", *Applied Surface
   Science*, vol. 285, 2013 (dataset used in Project 2, Track A).
