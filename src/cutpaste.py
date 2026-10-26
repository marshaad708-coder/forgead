"""Synthetic-anomaly 'forge' for ForgeAD.

Implements the two augmentation families of CutPaste
(C. Li et al., "CutPaste: Self-Supervised Learning for Anomaly Detection
and Localization", CVPR 2021):

  * CUTPASTE (label 1): cut a rectangular patch from a random source
    location, optionally rotate / jitter it, and paste it at a random
    target location.
  * SCAR (label 2): cut a long, thin, randomly rotated strip and paste it
    back -- mimics scratches / cracks.

Label 0 = untouched normal image. Only NORMAL (defect-free) training
images are ever forged, so no real defect labels are needed: the
supervision is entirely self-generated.
"""
import math
import random

from PIL import Image, ImageEnhance

LABEL_NORMAL = 0
LABEL_CUTPASTE = 1
LABEL_SCAR = 2


def _rand_patch_box(rng: random.Random, w: int, h: int,
                    area_ratio: tuple[float, float] = (0.02, 0.15),
                    aspect: tuple[float, float] = (0.3, 3.33)) -> tuple[int, int, int, int]:
    area = rng.uniform(*area_ratio) * w * h
    ar = rng.uniform(*aspect)
    pw = int(round(math.sqrt(area * ar)))
    ph = int(round(math.sqrt(area / ar)))
    pw = max(2, min(pw, w - 1))
    ph = max(2, min(ph, h - 1))
    x = rng.randint(0, w - pw)
    y = rng.randint(0, h - ph)
    return x, y, pw, ph


def _jitter_patch(patch: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.75:
        patch = patch.rotate(rng.choice([0, 90, 180, 270]), expand=True)
    if rng.random() < 0.5:
        patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        patch = ImageEnhance.Color(patch).enhance(rng.uniform(0.85, 1.15))
    return patch


def forge_cutpaste(img: Image.Image, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return (forged image, pasted bbox (x, y, w, h))."""
    w, h = img.size
    sx, sy, pw, ph = _rand_patch_box(rng, w, h)
    patch = _jitter_patch(img.crop((sx, sy, sx + pw, sy + ph)), rng)
    pw2, ph2 = patch.size
    tx = rng.randint(0, w - pw2)
    ty = rng.randint(0, h - ph2)
    out = img.copy()
    out.paste(patch, (tx, ty))
    return out, (tx, ty, pw2, ph2)


def forge_scar(img: Image.Image, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Return (forged image, pasted bbox (x, y, w, h))."""
    w, h = img.size
    length = int(rng.uniform(0.15, 0.60) * max(w, h))
    thick = rng.randint(2, max(3, min(w, h) // 40))
    strip = img.crop((0, 0, min(length, w - 1), thick))
    strip = strip.resize((length, thick), Image.BILINEAR)
    strip = _jitter_patch(strip, rng)
    angle = rng.uniform(0, 180)
    strip = strip.rotate(angle, expand=True, resample=Image.BILINEAR)
    pw, ph = strip.size
    if pw >= w or ph >= h:  # degenerate after rotation; fall back to plain cutpaste
        return forge_cutpaste(img, rng)
    tx = rng.randint(0, w - pw)
    ty = rng.randint(0, h - ph)
    out = img.copy()
    out.paste(strip, (tx, ty))
    return out, (tx, ty, pw, ph)


def forge(img: Image.Image, kind: int, rng: random.Random) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    if kind == LABEL_CUTPASTE:
        return forge_cutpaste(img, rng)
    if kind == LABEL_SCAR:
        return forge_scar(img, rng)
    return img, None
