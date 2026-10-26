"""Evaluate ForgeAD on the MVTec toothbrush test set.

Metrics (same definitions as VisionAI Project 2's Track B, so the two
are directly comparable):
  * image AUROC -- over the 42 test images (12 good / 30 defective)
  * pixel AUROC -- over every test pixel vs the ground-truth masks

Two image-level scorers are reported: the Gaussian-density (GDE)
Mahalanobis score and the ablation head score (1 - P(normal)).
Anomaly maps come from the patch-level GDE.

Writes results/forgead_results.json and prints a comparison table
against the PatchCore baseline measured in VisionAI.

Usage:
    python -m src.evaluate --checkpoint checkpoints/forgead_best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.common.seed import set_seed
from src.data_mvtec import list_test, list_train_good, load_mask, resolve_root
from src.models import ForgeNet
from src.score_gde import GDEScorer

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path.home() / "workspace" / "visionai-defect-detection" / "results" / "mvtec_toothbrush_results.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default=str(REPO_ROOT / "checkpoints" / "forgead_best.pt"))
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "results" / "forgead_results.json"))
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    ckpt = torch_load(a.checkpoint)
    img_size = ckpt["img_size"]
    model = ForgeNet(num_classes=ckpt["n_classes"], pretrained=False,
                     freeze_until=ckpt.get("freeze_until", "layer2"))
    model.load_state_dict(ckpt["state_dict"])

    root = resolve_root(a.data_root)
    train_good = list_train_good(root)
    test_items = list_test(root)
    paths = [p for p, _, _ in test_items]
    labels = np.array([y for _, y, _ in test_items])
    print(f"test images: {len(paths)} ({int((labels == 0).sum())} good / "
          f"{int((labels == 1).sum())} defective), img_size={img_size}")

    scorer = GDEScorer(model, train_good, img_size)
    img_gde = scorer.image_scores(paths)
    img_head = scorer.head_scores(paths)
    maps = scorer.anomaly_maps(paths)

    image_auroc_gde = float(roc_auc_score(labels, img_gde))
    image_auroc_head = float(roc_auc_score(labels, img_head))

    # pixel AUROC over all test pixels
    px_scores, px_labels = [], []
    for (p, y, mpath), amap in zip(test_items, maps):
        if mpath is None:
            mask = np.zeros((img_size, img_size), dtype=np.uint8)
        else:
            mask = (np.array(load_mask(mpath, img_size)) > 127).astype(np.uint8)
        px_scores.append(amap.ravel())
        px_labels.append(mask.ravel())
    px_scores = np.concatenate(px_scores)
    px_labels = np.concatenate(px_labels)
    pixel_auroc = float(roc_auc_score(px_labels, px_scores))

    baseline = {}
    if BASELINE_PATH.exists():
        b = json.loads(BASELINE_PATH.read_text())
        baseline = {"image_auroc": b["image_auroc"], "pixel_auroc": b["pixel_auroc"],
                    "method": "PatchCore (VisionAI Project 2, measured)"}

    results = {
        "method": "ForgeAD: CutPaste-style self-supervised forge + Gaussian density scoring",
        "reference": "C. Li et al., CutPaste, CVPR 2021 (independent re-implementation)",
        "category": "toothbrush",
        "n_test": len(paths),
        "n_anomalous": int((labels == 1).sum()),
        "img_size": img_size,
        "image_auroc_gde": image_auroc_gde,
        "image_auroc_head_ablation": image_auroc_head,
        "pixel_auroc": pixel_auroc,
        "baseline": baseline,
        "note": "Pixel AUROC is computed over all test pixels, same as the baseline.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    print("\n=== comparison vs PatchCore baseline (VisionAI Project 2) ===")
    if baseline:
        print(f"  image AUROC : ForgeAD-GDE {image_auroc_gde:.4f}  vs  PatchCore {baseline['image_auroc']:.4f}")
        print(f"  image AUROC : ForgeAD-head {image_auroc_head:.4f}  (ablation, no density fit)")
        print(f"  pixel AUROC : ForgeAD     {pixel_auroc:.4f}  vs  PatchCore {baseline['pixel_auroc']:.4f}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
