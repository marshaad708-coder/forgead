"""Gaussian density scoring on ForgeNet embeddings.

After self-supervised training, the pooled 512-d embeddings of NORMAL
images are modelled as a Gaussian (Ledoit-Wolf shrinkage covariance).
Test images are scored by Mahalanobis distance: far from the normal
manifold => anomalous. This mirrors the GDE scorer of the CutPaste
paper (Li et al., CVPR 2021).

Localization reuses the same idea per feature-map position: one shared
Gaussian is fit over all normal patch features; each test position is
scored, and the score map is upsampled to image size.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from src.data_mvtec import IMAGENET_MEAN, IMAGENET_STD, load_rgb


def _preprocess(paths, img_size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return torch.stack([tf(load_rgb(p, img_size)) for p in paths])


@torch.no_grad()
def _embed_batches(model, x: torch.Tensor, batch: int = 16) -> torch.Tensor:
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model.features(x[i:i + batch]))
    return torch.cat(outs)


@torch.no_grad()
def _feature_maps(model, x: torch.Tensor, batch: int = 16) -> torch.Tensor:
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        outs.append(model.feature_map(x[i:i + batch]))
    return torch.cat(outs)  # (N, 512, H', W')


class GDEScorer:
    def __init__(self, model, train_good_paths, img_size: int):
        self.model = model
        self.img_size = img_size
        x = _preprocess(train_good_paths, img_size)
        z = _embed_batches(model, x).numpy()
        self.lw = LedoitWolf().fit(z)
        self.mean_ = self.lw.location_
        self.prec_ = self.lw.precision_  # inverse covariance
        # shared patch-level Gaussian over all normal positions
        fmap = _feature_maps(model, x)                       # (N, C, H', W')
        n, c, hh, ww = fmap.shape
        patches = fmap.permute(0, 2, 3, 1).reshape(n * hh * ww, c).numpy()
        self.patch_lw = LedoitWolf().fit(patches)
        self.grid = (hh, ww)

    def _mahal(self, z: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
        d = z - mean
        return np.einsum("ij,jk,ik->i", d, prec, d)

    @torch.no_grad()
    def image_scores(self, paths) -> np.ndarray:
        x = _preprocess(paths, self.img_size)
        z = _embed_batches(self.model, x).numpy()
        return self._mahal(z, self.mean_, self.prec_)

    @torch.no_grad()
    def anomaly_maps(self, paths, smooth_sigma: float = 4.0) -> list[np.ndarray]:
        """Return per-image (H, W) anomaly maps in [0, 1]-ish raw scores."""
        from scipy.ndimage import gaussian_filter
        x = _preprocess(paths, self.img_size)
        fmap = _feature_maps(self.model, x)
        n, c, hh, ww = fmap.shape
        patches = fmap.permute(0, 2, 3, 1).reshape(n * hh * ww, c).numpy()
        s = self._mahal(patches, self.patch_lw.location_, self.patch_lw.precision_)
        s = s.reshape(n, hh, ww)
        maps = []
        for i in range(n):
            m = F.interpolate(torch.from_numpy(s[i])[None, None].float(),
                              size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)[0, 0].numpy()
            maps.append(gaussian_filter(m, smooth_sigma))
        return maps

    @torch.no_grad()
    def head_scores(self, paths, batch: int = 16) -> np.ndarray:
        """Ablation scorer: 1 - P(normal) straight from the forge head."""
        self.model.eval()
        x = _preprocess(paths, self.img_size)
        probs = []
        for i in range(0, len(x), batch):
            probs.append(torch.softmax(self.model(x[i:i + batch]), dim=1)[:, 0])
        return 1.0 - torch.cat(probs).numpy()
