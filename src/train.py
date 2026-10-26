"""Train ForgeNet on self-forged anomalies.

Only defect-free training images are used. Each epoch, every training
image is forged on the fly into a random class (normal / cutpaste /
scar); the network learns to tell forged regions from real ones.
Model selection uses a held-out slice of NORMAL images (their forged
copies), so no real defect label ever leaks into training.

Usage:
    python -m src.train --epochs 60 --img-size 224
"""
import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.common.seed import set_seed
from src.cutpaste import LABEL_CUTPASTE, LABEL_NORMAL, LABEL_SCAR, forge
from src.data_mvtec import IMAGENET_MEAN, IMAGENET_STD, list_train_good, load_rgb, resolve_root
from src.models import ForgeNet

REPO_ROOT = Path(__file__).resolve().parent.parent


class ForgeDataset(Dataset):
    def __init__(self, paths, img_size: int, aug_factor: int, seed: int, kinds=(0, 1, 2)):
        self.paths = paths
        self.img_size = img_size
        self.aug_factor = aug_factor
        self.rng = random.Random(seed)
        self.kinds = kinds
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths) * self.aug_factor

    def __getitem__(self, idx):
        img = load_rgb(self.paths[idx // self.aug_factor], self.img_size)
        kind = self.rng.choice(self.kinds)
        forged, _ = forge(img, kind, self.rng) if kind != LABEL_NORMAL else (img, None)
        return self.tf(forged), kind


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--aug-factor", type=int, default=6)
    p.add_argument("--val-images", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--freeze-until", type=str, default="layer2")
    p.add_argument("--no-scar", action="store_true",
                   help="ablation: train 2-way (normal vs cutpaste) without the scar family")
    p.add_argument("--resume", action="store_true",
                   help="resume from checkpoints/<name>_last.pt if it matches this config")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "checkpoints" / "forgead_best.pt"))
    p.add_argument("--log", type=str, default=str(REPO_ROOT / "results" / "forgead_training_log.csv"))
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(x)
        correct += (out.argmax(1) == y).sum().item()
        total += len(x)
    return loss_sum / total, correct / total


def main():
    a = parse_args()
    set_seed(a.seed)
    kinds = (LABEL_NORMAL, LABEL_CUTPASTE) if a.no_scar else (LABEL_NORMAL, LABEL_CUTPASTE, LABEL_SCAR)
    n_classes = len(kinds)

    root = resolve_root(a.data_root)
    all_paths = list_train_good(root)
    assert len(all_paths) == 60, f"expected 60 train/good images, found {len(all_paths)}"
    rng = random.Random(a.seed)
    val_paths = rng.sample(all_paths, a.val_images)
    train_paths = [p for p in all_paths if p not in val_paths]
    print(f"train images: {len(train_paths)}, val images: {len(val_paths)}, "
          f"classes: {n_classes}, data: {root}")

    train_ds = ForgeDataset(train_paths, a.img_size, a.aug_factor, seed=a.seed, kinds=kinds)
    val_ds = ForgeDataset(val_paths, a.img_size, max(2, a.aug_factor // 2), seed=a.seed + 1, kinds=kinds)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=a.batch, shuffle=False, num_workers=0)

    model = ForgeNet(num_classes=n_classes, pretrained=True, freeze_until=a.freeze_until)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    last_path = out_path.parent / (out_path.stem + "_last.pt")

    def snapshot(epoch_done: int, best_val_acc: float, tag: str):
        torch.save({"state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch_done,
                    "img_size": a.img_size,
                    "n_classes": n_classes,
                    "kinds": list(kinds),
                    "freeze_until": a.freeze_until,
                    "best_val_acc": best_val_acc,
                    "seed": a.seed}, last_path if tag == "last" else out_path)

    best_val = -1.0
    start_epoch = 1
    if last_path.exists() and a.resume:
        prev = torch.load(last_path, map_location="cpu", weights_only=False)
        if (prev["n_classes"] == n_classes and prev["img_size"] == a.img_size
                and prev["kinds"] == list(kinds)):
            model.load_state_dict(prev["state_dict"])
            optimizer.load_state_dict(prev["optimizer"])
            scheduler.load_state_dict(prev["scheduler"])
            best_val = prev["best_val_acc"]
            start_epoch = prev["epoch"] + 1
            print(f"resumed from epoch {prev['epoch']} (best val {best_val:.4f})")
    log_mode = "a" if start_epoch > 1 and Path(a.log).exists() else "w"
    with open(a.log, log_mode, newline="") as f:
        w = csv.writer(f)
        if log_mode == "w":
            w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for epoch in range(start_epoch, a.epochs + 1):
            tr_loss, tr_acc = run_epoch(model, train_dl, criterion, optimizer)
            va_loss, va_acc = run_epoch(model, val_dl, criterion)
            scheduler.step()
            w.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                        f"{va_loss:.4f}", f"{va_acc:.4f}", f"{optimizer.param_groups[0]['lr']:.2e}"])
            f.flush()
            improved = va_acc > best_val
            if improved:
                best_val = va_acc
            # persist every epoch: a killed run loses nothing but the current epoch
            snapshot(epoch, best_val, "last")
            if improved:
                snapshot(epoch, best_val, "best")
            if epoch == 1 or epoch % 10 == 0 or epoch == a.epochs:
                print(f"epoch {epoch:3d}/{a.epochs}  train acc {tr_acc:.3f}  "
                      f"val acc {va_acc:.3f} (best {best_val:.3f})", flush=True)
    print(f"done. best val acc {best_val:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
