"""MVTec AD 'toothbrush' data access for ForgeAD.

Layout expected (same as VisionAI Project 2):
    <root>/train/good/*.png
    <root>/test/good/*.png
    <root>/test/defective/*.png
    <root>/ground_truth/defective/<stem>_mask.png

Root resolution order:
    1. --data-root CLI arg / FORGEAD_DATA env var
    2. ./data_raw/mvtec/toothbrush  (repo-local symlink or download target)
"""
import os
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "data_raw" / "mvtec" / "toothbrush"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_root(explicit: str | None = None) -> Path:
    for cand in (explicit, os.environ.get("FORGEAD_DATA")):
        if cand:
            p = Path(cand)
            if (p / "train" / "good").is_dir():
                return p
    if (DEFAULT_ROOT / "train" / "good").is_dir():
        return DEFAULT_ROOT
    raise FileNotFoundError(
        "MVTec toothbrush data not found. Run scripts/download_mvtec.py "
        "or set FORGEAD_DATA / --data-root."
    )


def list_train_good(root: Path) -> list[Path]:
    return sorted((root / "train" / "good").glob("*.png"))


def list_test(root: Path) -> list[tuple[Path, int, Path | None]]:
    """Return [(image_path, label, mask_path|None)], label 0=good, 1=defective."""
    items: list[tuple[Path, int, Path | None]] = []
    for p in sorted((root / "test" / "good").glob("*.png")):
        items.append((p, 0, None))
    gt = root / "ground_truth" / "defective"
    for p in sorted((root / "test" / "defective").glob("*.png")):
        mask = gt / f"{p.stem}_mask.png"
        items.append((p, 1, mask if mask.exists() else None))
    return items


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)


def load_mask(path: Path, size: int) -> Image.Image:
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return m.point(lambda v: 255 if v > 127 else 0)
