"""Smoke tests for the synthetic-anomaly forge (no training needed)."""
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cutpaste import LABEL_CUTPASTE, LABEL_SCAR, forge, forge_cutpaste, forge_scar


def _dummy():
    return Image.new("RGB", (256, 256), (128, 128, 128))


def test_forge_changes_image():
    rng = random.Random(0)
    img = _dummy()
    for kind, fn in ((LABEL_CUTPASTE, forge_cutpaste), (LABEL_SCAR, forge_scar)):
        out, bbox = fn(img, rng)
        assert out.size == img.size
        assert list(out.getdata()) != list(img.getdata()), f"kind {kind} left image unchanged"
        x, y, w, h = bbox
        assert 0 <= x < 256 and 0 <= y < 256 and w > 0 and h > 0
    out, bbox = forge(img, 0, rng)
    assert bbox is None and list(out.getdata()) == list(img.getdata())


def test_forge_on_real_image():
    from src.data_mvtec import list_train_good, load_rgb, resolve_root
    root = resolve_root()
    img = load_rgb(list_train_good(root)[0], 224)
    rng = random.Random(1)
    out, _ = forge(img, LABEL_CUTPASTE, rng)
    assert out.size == (224, 224)
    print("forge smoke test on real toothbrush image: OK")
