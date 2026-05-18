#!/usr/bin/env python
"""Lipo (WORC) MR tumor segmentation -- MRI-EDAIN v2 trainer.

Differences from the BraTS T1n entry point:
    - Lipo MR is NOT skull-stripped; background is not exactly zero. We Otsu-
      crop the bounding box and then z-score over the cropped volume.
    - File pairing comes from `Lipo-XXX_MR_N_image.nii.gz` and
      `Lipo-XXX_MR_N_segmentation.nii.gz` glob; subjects with multiple
      `_MR_N_` files (multi-tumor; e.g. Lipo-073) are AUTO-EXCLUDED, matching
      the project memory's single-label model constraint.
    - lipo_split.json uses the format
          { "<fold_id>": { "n_val": ..., "val_subjects_sorted": [...] }, ... }
      with the train set derived as `all_subjects \\ val_subjects`.

The rest of the pipeline (per-volume z-score upstream, MRIEDAINLayer with
RQ-spline, 3-phase lambda schedule, EMA, function-space anchor loss + KL
loss, Phase-I diagnostic r_i logging) is identical to the BraTS trainer.

Smoke test:
    python lipo_mri_edain_v2.py --smoke --epochs 2
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import affine_transform, gaussian_filter, zoom
from skimage.filters import threshold_otsu
from torch import amp

from monai.data import CacheDataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponent,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandomizableTransform,
    SaveImage,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.utils import set_determinism

# --- v2 components ---------------------------------------------------------
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from mri_edain_v2.losses import (  # noqa: E402
    CombinedLoss,
    FunctionSpaceAnchorLoss,
    KLAnchorLoss,
)
from mri_edain_v2.modules import (  # noqa: E402
    PERCENTILES,
    CoordinateStandardizer,
    MRIEDAINLayer,
    compute_non_affineness,
    percentile_summary,
    rq_spline_apply,
    rq_spline_parameterize,
)
from mri_edain_v2.training import (  # noqa: E402
    EMAWrapper,
    LambdaScheduler,
    apply_artifacts_to_standardizer,
    load_precomputed_artifacts,
    precompute_fold_artifacts,
    save_precomputed_artifacts,
)
from mri_edain_v2.training.precompute import default_per_case_preprocessor  # noqa: E402


# =============================================================================
# Path configuration
# =============================================================================

_DEFAULT_DATA_DIR = (Path(__file__).resolve().parent.parent.parent
                     / "dataset" / "lipo")

DATA_DIR: Path = _DEFAULT_DATA_DIR
SPLIT_JSON: Path = _DEFAULT_DATA_DIR.parent / "lipo_split.json"  # may live alongside dataset
OUT_DIR: Path = Path("./outputs/lipo_mri_edain_v2")
LOG_DIR: Path = Path("./logs/lipo_mri_edain_v2")
ARTIFACT_DIR: Path = Path("./artifacts/lipo_mri_edain_v2")

# Hyperparameters (blueprint section 5.2).
MAX_EPOCHS = 1000
VAL_INTERVAL = 2
EARLY_STOP_PATIENCE = 75
BASE_LR = 1.0e-2
BATCH_SIZE = 2
NUM_PATCHES = 4
NUM_WORKERS = 4

# Spline / hypernet config (blueprint section 2.4-2.5).
K_KNOTS = 9
B_SUPP = 4.0
ALPHA_TAIL = 0.5
MIN_DERIVATIVE = 1e-3
HYPERNET_HIDDEN = 64

# Loss schedule defaults (blueprint section 5.1).
LAMBDA_ANC_INIT = 1.0e-2
LAMBDA_ANC_FINAL = 1.0e-4
LAMBDA_KL_FINAL = 1.0e-4

EMA_DECAY = 0.99
HYPERNET_GRAD_CLIP = 1.0


# =============================================================================
# Otsu helper for CropForegroundd (Lipo has air background, not skull-stripped)
# =============================================================================

def otsu_select_fn(x):
    x_np = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    x_np = x_np.astype(np.float32)
    try:
        thr = float(threshold_otsu(x_np))
    except Exception:
        thr = float(x_np.mean())
    return x_np > thr


# =============================================================================
# Lipo-specific per-case preprocessor for the precompute pipeline.
# (default_per_case_preprocessor assumes nonzero background.)
# =============================================================================

def lipo_per_case_preprocessor(
    image_path: str,
    target_pixdim,
    foreground_method: str = "nonzero",  # ignored; we always Otsu
):
    """Apply Lipo training pipeline (Otsu crop + per-volume z-score) and
    return (X_zscored, mask). Background is identified post-zscore as `X != 0`.
    """
    from monai.transforms import (
        Compose,
        CropForeground,
        EnsureChannelFirst,
        LoadImage,
        NormalizeIntensity,
        Orientation,
        Spacing,
    )

    steps = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
    ]
    if target_pixdim is not None:
        steps.append(Spacing(pixdim=target_pixdim, mode="bilinear"))
    steps.append(CropForeground(select_fn=otsu_select_fn, margin=10))
    steps.append(NormalizeIntensity(nonzero=True))

    pipeline = Compose(steps)
    arr = pipeline(str(image_path))
    if torch.is_tensor(arr):
        X = arr[0].to(torch.float32).cpu()
    else:
        X = torch.as_tensor(np.asarray(arr)[0], dtype=torch.float32)
    mask = X != 0
    return X, mask


# =============================================================================
# nnU-Net style augmentations (copied verbatim from the v1 BraTS pipeline)
# =============================================================================

class NnUNetRandRotateScaled(MapTransform, RandomizableTransform):
    def __init__(self, keys, rotate_range=(-30, 30), scale_range=(0.7, 1.4),
                 anisotropic=False, anisotropy_axis=2):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob=1.0)
        self.rotate_range = rotate_range; self.scale_range = scale_range
        self.anisotropic = anisotropic; self.anisotropy_axis = anisotropy_axis
        self._do_rotate = False; self._do_scale = False
        self._rotation_angles = [0.0, 0.0, 0.0]; self._scale_factor = 1.0

    def randomize(self, data=None):
        r = self.R.random()
        if r < 0.08:   self._do_rotate = True;  self._do_scale = True
        elif r < 0.24: self._do_rotate = False; self._do_scale = True
        elif r < 0.40: self._do_rotate = True;  self._do_scale = False
        else:          self._do_rotate = False; self._do_scale = False
        self._rotation_angles = [0.0, 0.0, 0.0]
        if self._do_rotate:
            if self.anisotropic:
                self._rotation_angles[self.anisotropy_axis] = np.deg2rad(
                    self.R.uniform(*self.rotate_range))
            else:
                self._rotation_angles = [
                    np.deg2rad(self.R.uniform(*self.rotate_range)) for _ in range(3)]
        self._scale_factor = self.R.uniform(*self.scale_range) if self._do_scale else 1.0

    def _build_affine_backward(self, shape):
        center = (np.array(shape, dtype=np.float64) - 1.0) / 2.0
        ax, ay, az = self._rotation_angles
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az),  np.cos(az), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                       [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)],
                       [0, np.sin(ax),  np.cos(ax)]])
        M = np.diag([self._scale_factor] * 3) @ Rz @ Ry @ Rx
        invM = np.linalg.inv(M)
        return invM, center - invM @ center

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._do_rotate and not self._do_scale: return d
        for key in self.keys:
            img = d[key]; is_label = "label" in key.lower()
            if torch.is_tensor(img): img_np = img.numpy(); was_tensor = True
            else: img_np = np.asarray(img); was_tensor = False
            result = np.zeros_like(img_np)
            invM, offset = self._build_affine_backward(img_np.shape[1:])
            for c in range(img_np.shape[0]):
                result[c] = affine_transform(
                    img_np[c], matrix=invM, offset=offset,
                    order=0 if is_label else 3, mode="constant", cval=0.0)
            d[key] = torch.from_numpy(result) if was_tensor else result
        return d


class CenterSpatialCropWithPadd(MapTransform):
    def __init__(self, keys, roi_size):
        MapTransform.__init__(self, keys); self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if torch.is_tensor(img): img_np = img.numpy(); was_tensor = True
            else: img_np = img; was_tensor = False
            spatial_shape = img_np.shape[1:]
            pad_need = []
            for i in range(3):
                diff = self.roi_size[i] - spatial_shape[i]
                pad_need.append((diff // 2, diff - diff // 2) if diff > 0 else (0, 0))
            if any(p[0] > 0 or p[1] > 0 for p in pad_need):
                img_np = np.pad(img_np, [(0, 0)] + pad_need,
                                mode="constant", constant_values=0)
                spatial_shape = img_np.shape[1:]
            starts = [max(0, (spatial_shape[i] - self.roi_size[i]) // 2) for i in range(3)]
            ends   = [starts[i] + self.roi_size[i] for i in range(3)]
            result = img_np[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
            d[key] = torch.from_numpy(result) if was_tensor else result
        return d


class NnUNetRandGaussianNoised(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.15, variance_range=(0.0, 0.1)):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.variance_range = variance_range; self._variance = 0.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform: self._variance = self.R.uniform(*self.variance_range)

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._do_transform: return d
        std = np.sqrt(self._variance)
        for key in self.keys:
            img = d[key]
            d[key] = (img + torch.randn_like(img) * std if torch.is_tensor(img)
                      else img + self.R.standard_normal(img.shape).astype(img.dtype) * std)
        return d


class NnUNetRandGaussianSmoothd(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.2, prob_per_channel=0.5, sigma_range=(0.5, 1.5)):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.prob_per_channel = prob_per_channel; self.sigma_range = sigma_range

    def randomize(self, data=None, num_channels=1):
        super().randomize(None); self._channel_flags = []; self._sigmas = []
        if self._do_transform:
            for _ in range(num_channels):
                do_ch = self.R.random() < self.prob_per_channel
                self._channel_flags.append(do_ch)
                self._sigmas.append(self.R.uniform(*self.sigma_range) if do_ch else None)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]; self.randomize(d, img.shape[0])
            if not self._do_transform: continue
            if torch.is_tensor(img): img_np = img.numpy(); was_tensor = True
            else: img_np = img; was_tensor = False
            result = img_np.copy()
            for c in range(img_np.shape[0]):
                if self._channel_flags[c]:
                    result[c] = gaussian_filter(img_np[c], sigma=self._sigmas[c], mode="nearest")
            d[key] = torch.from_numpy(result) if was_tensor else result
        return d


class NnUNetRandBrightnessd(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.15, factor_range=(0.7, 1.3)):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.factor_range = factor_range; self._factor = 1.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform: self._factor = self.R.uniform(*self.factor_range)

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._do_transform: return d
        for key in self.keys: d[key] = d[key] * self._factor
        return d


class NnUNetRandContrastd(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.15, factor_range=(0.65, 1.5)):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.factor_range = factor_range; self._factor = 1.0

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform: self._factor = self.R.uniform(*self.factor_range)

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._do_transform: return d
        for key in self.keys:
            img = d[key]
            if torch.is_tensor(img): img_np = img.detach().cpu().numpy(); was_tensor = True
            else: img_np = img; was_tensor = False
            result = img_np.copy()
            for c in range(img_np.shape[0]):
                cm = img_np[c].mean()
                result[c] = (img_np[c] - cm) * self._factor + cm
            d[key] = torch.from_numpy(result.astype(np.float32)) if was_tensor else result
        return d


class NnUNetRandSimulateLowResolutiond(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.25, prob_per_channel=0.5,
                 downsample_factor_range=(1.0, 2.0), anisotropic=False, anisotropy_axis=2):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.prob_per_channel = prob_per_channel
        self.downsample_factor_range = downsample_factor_range
        self.anisotropic = anisotropic; self.anisotropy_axis = anisotropy_axis

    def randomize(self, data=None, num_channels=1):
        super().randomize(None); self._channel_flags = []; self._factors = []
        if self._do_transform:
            for _ in range(num_channels):
                do_ch = self.R.random() < self.prob_per_channel
                self._channel_flags.append(do_ch)
                self._factors.append(
                    self.R.uniform(*self.downsample_factor_range) if do_ch else None)

    def _match_shape(self, arr, target_shape):
        result = np.zeros(target_shape, dtype=arr.dtype)
        slices_src, slices_dst = [], []
        for i in range(len(target_shape)):
            cs = min(arr.shape[i], target_shape[i])
            slices_src.append(slice((arr.shape[i] - cs) // 2, (arr.shape[i] - cs) // 2 + cs))
            slices_dst.append(slice((target_shape[i] - cs) // 2, (target_shape[i] - cs) // 2 + cs))
        result[tuple(slices_dst)] = arr[tuple(slices_src)]
        return result

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]; self.randomize(d, img.shape[0])
            if not self._do_transform: continue
            if torch.is_tensor(img): img_np = img.numpy(); was_tensor = True
            else: img_np = img; was_tensor = False
            result = img_np.copy(); original_shape = img_np.shape[1:]
            for c in range(img_np.shape[0]):
                if self._channel_flags[c]:
                    factor = self._factors[c]; vol = img_np[c]
                    zd = [1.0 / factor] * 3; zu = [factor] * 3
                    if self.anisotropic:
                        zd[self.anisotropy_axis] = 1.0; zu[self.anisotropy_axis] = 1.0
                    result[c] = self._match_shape(
                        zoom(zoom(vol, zd, order=0, mode="nearest"), zu, order=3, mode="nearest"),
                        original_shape)
            d[key] = torch.from_numpy(result) if was_tensor else result
        return d


class NnUNetRandGammaD(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.15, gamma_range=(0.7, 1.5),
                 invert_image=False, invert_prob=0.15):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=prob)
        self.gamma_range = gamma_range; self.invert_image = invert_image
        self.invert_prob = invert_prob; self._gamma = 1.0; self._invert = False

    def randomize(self, data=None):
        super().randomize(None)
        if self._do_transform:
            self._gamma = self.R.uniform(*self.gamma_range)
            self._invert = (self.R.random() < self.invert_prob
                            if self.invert_image is None else bool(self.invert_image))

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._do_transform: return d
        for key in self.keys:
            img = d[key]
            if torch.is_tensor(img): img_np = img.numpy(); was_tensor = True
            else: img_np = img; was_tensor = False
            result = img_np.copy()
            for c in range(img_np.shape[0]):
                vol = img_np[c]; orig_min, orig_max = vol.min(), vol.max()
                vol_n = ((vol - orig_min) / (orig_max - orig_min)
                         if orig_max - orig_min > 1e-8 else np.zeros_like(vol))
                vol_n = np.clip(vol_n, 0, 1)
                vol_g = (1.0 - np.power(np.clip(1.0 - vol_n, 1e-8, 1.0), self._gamma)
                         if self._invert else np.power(np.clip(vol_n, 1e-8, 1.0), self._gamma))
                result[c] = np.clip(vol_g, 0, 1) * (orig_max - orig_min) + orig_min
            d[key] = torch.from_numpy(result) if was_tensor else result
        return d


class NnUNetRandFlipd(MapTransform, RandomizableTransform):
    def __init__(self, keys, prob=0.5, spatial_axes=(0, 1, 2)):
        MapTransform.__init__(self, keys); RandomizableTransform.__init__(self, prob=1.0)
        self.flip_prob = prob; self.spatial_axes = spatial_axes; self._axes_to_flip = []

    def randomize(self, data=None):
        self._axes_to_flip = [ax for ax in self.spatial_axes if self.R.random() < self.flip_prob]

    def __call__(self, data):
        d = dict(data); self.randomize(d)
        if not self._axes_to_flip: return d
        flip_axes = [ax + 1 for ax in self._axes_to_flip]
        for key in self.keys:
            arr = d[key]
            d[key] = (torch.flip(arr, dims=flip_axes) if torch.is_tensor(arr)
                      else np.flip(arr, axis=flip_axes).copy())
        return d


class RemapLabelsd(MapTransform):
    """Any positive label -> 1 (tumor), 0 -> 0 (background)."""
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl = d[key]
            if torch.is_tensor(lbl): d[key] = (lbl > 0).long()
            else: d[key] = (np.asarray(lbl) > 0).astype(np.uint8)
        return d


# =============================================================================
# Model
# =============================================================================

class SegmentationModelWithMRIEDAINv2(nn.Module):
    def __init__(self, backbone: nn.Module, edain: MRIEDAINLayer):
        super().__init__()
        self.backbone = backbone
        self.edain = edain

    def forward(self, x, gamma_raw=None, return_diag=False):
        x_tilde, diag = self.edain(x, mask=(x != 0.0), gamma_raw=gamma_raw)
        logits = self.backbone(x_tilde)
        if return_diag:
            return logits, diag
        return logits


# =============================================================================
# Helpers
# =============================================================================

def _dice_from_labels(pred, target, eps=1e-6):
    if pred.dim() == 5 and pred.size(1) == 1: pred = pred[:, 0]
    if target.dim() == 5 and target.size(1) == 1: target = target[:, 0]
    pred_bin = (pred > 0.5); target_bin = (target > 0.5)
    intersect = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    union = pred_bin.sum(dim=(1, 2, 3)).float() + target_bin.sum(dim=(1, 2, 3)).float()
    return ((2.0 * intersect + eps) / (union + eps)).mean()


def poly_lr(step, max_steps, power=0.9):
    return (1.0 - step / max_steps) ** power


def get_deep_supervision_weights(num_stages):
    w = np.array([1.0 / (2 ** i) for i in range(num_stages)])
    return w / w.sum()


def normalize_logits_output(logits):
    if isinstance(logits, (list, tuple)): return list(logits)
    elif logits.dim() == 6: return list(logits.unbind(dim=1))
    else: return [logits]


def compute_oversized_patch(patch_size):
    return tuple(int(np.ceil(ps * 1.4)) for ps in patch_size)


def get_lr(optim):
    for pg in optim.param_groups: return pg.get("lr", None)


def extract_subject_id(image_path: str) -> str:
    """`Lipo-001_MR_1_image.nii.gz` -> `Lipo-001`."""
    base = Path(image_path).name
    return base.split("_")[0]


# =============================================================================
# Data loading + dataset fingerprint
# =============================================================================

def collect_lipo_files(data_dir: Path):
    """Glob image-segmentation pairs and exclude multi-tumor subjects."""
    image_paths = sorted(glob.glob(str(data_dir / "Lipo-*_MR_*_image.nii.gz")))
    pairs = []
    for img_path in image_paths:
        seg_path = img_path.replace("_image.nii.gz", "_segmentation.nii.gz")
        if os.path.exists(seg_path):
            pairs.append((img_path, seg_path))
        else:
            print(f"[WARN] segmentation missing for {Path(img_path).name}")
    if not pairs:
        raise FileNotFoundError(
            f"No Lipo image/segmentation pairs found in {data_dir}"
        )

    all_files = [
        {"image": img, "label": seg, "subject": extract_subject_id(img),
         "case_id": Path(img).name.replace("_image.nii.gz", "")}
        for img, seg in pairs
    ]

    # Exclude multi-tumor subjects (memory: Lipo-073 has 2 tumors, etc.)
    subj_counts = Counter(d["subject"] for d in all_files)
    multi = {s for s, c in subj_counts.items() if c > 1}
    if multi:
        for s in sorted(multi):
            print(f"[INFO] excluding multi-label subject {s} "
                  f"({subj_counts[s]} files)")
        all_files = [d for d in all_files if d["subject"] not in multi]
        print(f"[INFO] after exclusion: {len(all_files)} pairs")
    return all_files


def build_folds_from_split_json(all_files, split_json_path: Path):
    """lipo_split.json: {"<fold>": {"n_val": ..., "val_subjects_sorted": [...]}, ...}"""
    with open(split_json_path) as f:
        split = json.load(f)
    folds = []
    for k, v in split.items():
        fold_id = int(k)
        val_subjects = set(v["val_subjects_sorted"])
        train = [d for d in all_files if d["subject"] not in val_subjects]
        val = [d for d in all_files if d["subject"] in val_subjects]
        folds.append({"fold": fold_id, "train_files": train, "val_files": val})
    folds.sort(key=lambda x: x["fold"])
    return folds


def load_data_and_compute_fingerprint(data_dir: Path, split_json_path: Path,
                                      smoke: bool = False,
                                      patch_size_max: int = None,
                                      max_channels: int = 512,
                                      explicit_patch_size: tuple = None):
    all_files = collect_lipo_files(data_dir)
    folds = build_folds_from_split_json(all_files, split_json_path)

    if smoke:
        for fo in folds:
            fo["train_files"] = fo["train_files"][:5]
            fo["val_files"] = fo["val_files"][:2]

    sample = (folds[0]["train_files"][:5] if smoke
              else folds[0]["train_files"][:100])
    all_spacings, all_shapes = [], []
    for entry in sample:
        img = nib.load(entry["image"])
        all_spacings.append(img.header.get_zooms()[:3])
        all_shapes.append(img.header.get_data_shape()[:3])
    all_spacings = np.array(all_spacings)
    median_spacing = np.median(all_spacings, axis=0)

    anisotropic = False; anisotropy_axis = None
    target_spacing = median_spacing.copy()
    if median_spacing.max() / median_spacing.min() >= 3.0:
        anisotropic = True
        anisotropy_axis = int(np.argmax(median_spacing))
        target_spacing[anisotropy_axis] = np.percentile(
            all_spacings[:, anisotropy_axis], 10)

    resampled_shapes = [np.array(s) * (np.array(sp) / target_spacing)
                        for s, sp in zip(all_shapes, all_spacings)]
    median_shape = np.median(np.array(resampled_shapes), axis=0).astype(int)
    target_pixdim = tuple(target_spacing.tolist())

    if explicit_patch_size is not None:
        # User passed an exact (X, Y, Z) patch -- bypass the auto heuristics.
        # Useful for matching an nnU-Net plan exactly (e.g. for Lipo:
        # plan patch is [32, 224, 256] in (Z, Y, X) -> (256, 224, 32) here).
        patch_size = tuple(int(x) for x in explicit_patch_size)
    elif smoke:
        patch_size = (64, 64, 64)
    elif anisotropic:
        patch_size = list((0, 0, 0))
        patch_size[anisotropy_axis] = int(min(median_shape[anisotropy_axis], 128))
        for ax in range(3):
            if ax != anisotropy_axis:
                patch_size[ax] = int(min(median_shape[ax], 192))
        patch_size = tuple(patch_size)
    else:
        patch_size = tuple(int(min(s, 128)) for s in median_shape)

    # User-supplied global cap (only meaningful when patch was auto-derived).
    if patch_size_max is not None and explicit_patch_size is None:
        patch_size = tuple(int(min(ps, patch_size_max)) for ps in patch_size)

    channels = [32]; strides = []
    current_shape = np.array(patch_size, dtype=np.int32)
    while True:
        stride = [2, 2, 2]
        if anisotropic and len(strides) < 2: stride[anisotropy_axis] = 1
        if (np.ceil(current_shape / stride) < 4).any(): break
        strides.append(tuple(stride))
        current_shape = np.ceil(current_shape / stride)
        channels.append(min(channels[-1] * 2, max_channels))
    channels = tuple(channels)

    total_stride = np.array([1, 1, 1])
    for s in strides: total_stride *= np.array(s)
    patch_size_fixed = list(patch_size)
    for ax in range(3):
        if patch_size_fixed[ax] % total_stride[ax] != 0:
            # FLOOR (not ceil) so we never overshoot the median image size on
            # the anisotropic axis. nnU-Net plans confirm this: e.g. Lipo
            # median Z=47, total_stride_Z=8 -> nnU-Net patch Z=32 (4*8) far
            # below 47.  We previously used ceil here which gave Z=48 > 47,
            # forcing SpatialPad to zero-fill many short-Z cases and erasing
            # tumor context.  max(total_stride, floor) guards the degenerate
            # case where median < total_stride.
            patch_size_fixed[ax] = max(
                int(total_stride[ax]),
                int(np.floor(patch_size_fixed[ax] / total_stride[ax])
                    * total_stride[ax]),
            )
    patch_size = tuple(patch_size_fixed)
    oversized_patch = compute_oversized_patch(patch_size)

    dynunet_strides = tuple([(1, 1, 1)] + list(strides))
    kernel_sizes = [(3, 3, 1) if (anisotropic and i < 2) else (3, 3, 3)
                    for i in range(len(channels))]
    up_kernel_sizes = tuple(strides)

    print(f"Anisotropic={anisotropic} axis={anisotropy_axis}")
    print(f"Target spacing: {target_pixdim}")
    print(f"Patch size: {patch_size}, Oversized: {oversized_patch}")
    print(f"Channels: {channels}, Strides: {strides}\n")

    return {
        "folds": folds,
        "target_pixdim": target_pixdim,
        "patch_size": patch_size,
        "oversized_patch": oversized_patch,
        "channels": channels,
        "strides": strides,
        "dynunet_strides": dynunet_strides,
        "kernel_sizes": kernel_sizes,
        "up_kernel_sizes": up_kernel_sizes,
        "anisotropic": anisotropic,
        "anisotropy_axis": anisotropy_axis,
    }


# =============================================================================
# Training loop
# =============================================================================

def train_fold(fold_info, fp, device, *, smoke: bool = False, max_epochs: int = None,
               artifact_path: Path = None, sw_batch_size: int = 1,
               hypernet_lr_factor: float = 0.1, frozen_hypernet: bool = False):
    CURR_FOLD = fold_info["fold"]
    target_pixdim = fp["target_pixdim"]
    patch_size = fp["patch_size"]
    oversized_patch = fp["oversized_patch"]
    anisotropic = fp["anisotropic"]
    anisotropy_axis = fp["anisotropy_axis"]
    channels = fp["channels"]
    dynunet_strides = fp["dynunet_strides"]
    kernel_sizes = fp["kernel_sizes"]
    up_kernel_sizes = fp["up_kernel_sizes"]

    print(f"\n{'='*60}")
    print(f"Fold {CURR_FOLD} (MRI-EDAIN v2, Lipo) | "
          f"train={len(fold_info['train_files'])} val={len(fold_info['val_files'])}")
    print(f"Device: {device} | Smoke: {smoke}")
    print(f"{'='*60}")

    epochs_target = max_epochs if max_epochs is not None else (2 if smoke else MAX_EPOCHS)

    # -------------------------------------------------------------------------
    # Precompute
    # -------------------------------------------------------------------------
    if artifact_path is None:
        artifact_path = ARTIFACT_DIR / f"fold_{CURR_FOLD}{'_smoke' if smoke else ''}.pt"

    if artifact_path.exists():
        print(f"[precompute] loading existing artifact: {artifact_path}")
        artifact = load_precomputed_artifacts(artifact_path)
    else:
        print(f"[precompute] running precompute "
              f"on {len(fold_info['train_files'])} cases (Otsu foreground)")
        artifact = precompute_fold_artifacts(
            fold_info["train_files"],
            target_pixdim=target_pixdim,
            foreground_method="otsu",  # nominal; lipo_per_case_preprocessor ignores
            K=K_KNOTS,
            B_supp=B_SUPP,
            nyul_iters=200,
            per_case_preprocessor=lipo_per_case_preprocessor,
            verbose=True,
            max_cases=(5 if smoke else None),
        )
        save_precomputed_artifacts(artifact, artifact_path)
        print(f"[precompute] saved -> {artifact_path}")

    print(f"[precompute] population landmarks: {artifact.population_landmarks.tolist()}")

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------
    standardizer = CoordinateStandardizer(n_dim=len(artifact.percentiles))
    apply_artifacts_to_standardizer(artifact, standardizer)

    edain_layer = MRIEDAINLayer(
        standardizer=standardizer,
        theta_0=artifact.theta_0,
        K=K_KNOTS, B_supp=B_SUPP,
        alpha_tail=ALPHA_TAIL, min_derivative=MIN_DERIVATIVE,
        hypernet_hidden_dim=HYPERNET_HIDDEN, hypernet_zero_init=True,
        percentiles=artifact.percentiles,
    )

    backbone = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        filters=channels, strides=dynunet_strides,
        kernel_size=kernel_sizes, upsample_kernel_size=up_kernel_sizes,
        act_name=("leakyrelu", {"negative_slope": 0.01}),
        norm_name="instance",
        deep_supervision=True, deep_supr_num=3,
    )
    model = SegmentationModelWithMRIEDAINv2(backbone, edain_layer).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    edain_params = sum(p.numel() for p in model.edain.parameters() if p.requires_grad)
    print(f"Backbone params: {(total_params - edain_params) / 1e6:.2f} M | "
          f"EDAINv2 hypernet params: {edain_params}")

    # -------------------------------------------------------------------------
    # Transforms (Lipo: Otsu CropForeground, NOT >0)
    # -------------------------------------------------------------------------
    _rot_range = (-180, 180) if anisotropic else (-30, 30)
    _aniso_ax = anisotropy_axis if anisotropic else 2

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim,
                 mode=("bilinear", "nearest")),
        RemapLabelsd(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image",
                        select_fn=otsu_select_fn, margin=10),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=oversized_patch),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                               spatial_size=oversized_patch, pos=1, neg=1,
                               num_samples=NUM_PATCHES),
        NnUNetRandRotateScaled(keys=["image", "label"], rotate_range=_rot_range,
                               scale_range=(0.7, 1.4),
                               anisotropic=anisotropic, anisotropy_axis=_aniso_ax),
        CenterSpatialCropWithPadd(keys=["image", "label"], roi_size=patch_size),
        NnUNetRandGaussianNoised(keys=["image"], prob=0.15),
        NnUNetRandGaussianSmoothd(keys=["image"], prob=0.2),
        NnUNetRandBrightnessd(keys=["image"], prob=0.15),
        NnUNetRandContrastd(keys=["image"], prob=0.15),
        NnUNetRandSimulateLowResolutiond(keys=["image"], prob=0.25,
                                         anisotropic=anisotropic,
                                         anisotropy_axis=_aniso_ax),
        NnUNetRandGammaD(keys=["image"], prob=0.15, invert_image=None, invert_prob=0.15),
        NnUNetRandFlipd(keys=["image", "label"], prob=0.5),
        EnsureTyped(keys=["image"], dtype=np.float32),
        EnsureTyped(keys=["label"], dtype=np.uint8),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_pixdim,
                 mode=("bilinear", "nearest")),
        RemapLabelsd(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image",
                        select_fn=otsu_select_fn, margin=10),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        ToTensord(keys=["image", "label"]),
    ])

    train_ds = CacheDataset(data=fold_info["train_files"], transform=train_transforms,
                            cache_rate=0, num_workers=NUM_WORKERS if not smoke else 0)
    val_ds = CacheDataset(data=fold_info["val_files"], transform=val_transforms,
                          cache_rate=(0 if smoke else 1),
                          num_workers=NUM_WORKERS if not smoke else 0)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS if not smoke else 0,
                              pin_memory=torch.cuda.is_available(),
                              collate_fn=list_data_collate,
                              persistent_workers=(NUM_WORKERS > 0 and not smoke))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=NUM_WORKERS if not smoke else 0,
                            pin_memory=torch.cuda.is_available(),
                            persistent_workers=(NUM_WORKERS > 0 and not smoke))

    # -------------------------------------------------------------------------
    # Losses, optimizer, scheduler, EMA
    # -------------------------------------------------------------------------
    seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True,
                             include_background=False, lambda_dice=1.0, lambda_ce=1.0)
    anchor_loss_fn = FunctionSpaceAnchorLoss(grid_size=50, B_supp=B_SUPP).to(device)
    kl_loss_fn = KLAnchorLoss(n_bins=50, value_range=(-B_SUPP, B_SUPP),
                              n_subsample=20000).to(device)
    combined_loss = CombinedLoss(lambda_anc=LAMBDA_ANC_INIT, lambda_kl=0.0).to(device)

    # Two param groups: backbone keeps the nnU-Net SGD-Nesterov-0.99 default,
    # hypernet uses a smaller LR (blueprint section 5.2 fallback).  The
    # hypernet is small (~6.6K params) and feeds non-linear softmax / softplus,
    # so a momentum=0.99 amplification at lr=1e-2 blew up spline params at
    # the phase 0 -> 1 transition in the first real Lipo fold run.
    hypernet_params = list(model.edain.hypernet.parameters())
    hypernet_id_set = {id(p) for p in hypernet_params}
    backbone_params = [p for p in model.parameters() if id(p) not in hypernet_id_set]
    hypernet_lr = BASE_LR * float(hypernet_lr_factor)
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": BASE_LR, "name": "backbone"},
            {"params": hypernet_params, "lr": hypernet_lr, "name": "hypernet"},
        ],
        momentum=0.99, nesterov=True, weight_decay=3e-5,
    )
    print(f"Optimizer: backbone lr={BASE_LR}, hypernet lr={hypernet_lr} "
          f"(factor {hypernet_lr_factor})", flush=True)

    iter_per_epoch = max(1, len(fold_info["train_files"]) * NUM_PATCHES // BATCH_SIZE)
    total_steps = epochs_target * iter_per_epoch
    if frozen_hypernet:
        # Baseline #5 (RQSplineFixed) mode: hypernet stays frozen for the
        # entire training. Phase scheduler is degenerate (phase = 0 forever).
        # anc / KL stay at 0; only U-Net trains, on f_{theta_0}(X).
        phase_0_end = total_steps + 1
        phase_1_end = total_steps + 2
        kl_ramp = 1
    else:
        phase_0_end = max(1, int(total_steps * 0.01))
        phase_1_end = max(phase_0_end + 1, int(total_steps * 0.10))
        kl_ramp = max(1, int((total_steps - phase_1_end) * 0.05))
    lambda_sched = LambdaScheduler(
        total_steps=total_steps,
        phase_0_end=phase_0_end, phase_1_end=phase_1_end,
        lambda_anc_init=LAMBDA_ANC_INIT, lambda_anc_final=LAMBDA_ANC_FINAL,
        lambda_kl_final=LAMBDA_KL_FINAL, kl_ramp_steps=kl_ramp,
    )
    print(f"Schedule: {lambda_sched}"
          + (" [frozen_hypernet = baseline #5 RQSplineFixed]" if frozen_hypernet else ""))

    ema = EMAWrapper(model.edain.hypernet, decay=EMA_DECAY)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    fold_dir = OUT_DIR / f"fold_{CURR_FOLD}{'_smoke' if smoke else ''}"
    log_fold = LOG_DIR / f"fold_{CURR_FOLD}{'_smoke' if smoke else ''}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_fold.mkdir(parents=True, exist_ok=True)
    ckpt_path = fold_dir / "best_model.pt"
    log_path = log_fold / "training_log.csv"

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "global_step", "phase", "train_loss", "train_seg",
                "train_anc", "train_kl", "lambda_anc", "lambda_kl",
                "train_dice", "val_loss", "val_dice", "lr", "time_sec",
                "r_median", "r_p90",
            ])

    best_metric = -1.0; best_epoch = -1; epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, epochs_target + 1):
        t0 = time.time()
        model.train()
        train_loss_accum = 0.0
        train_seg_accum = 0.0; train_anc_accum = 0.0; train_kl_accum = 0.0
        train_dice_accum = 0.0; num_batches = 0
        print(f"[ep {epoch}] training (phase={lambda_sched.phase(global_step)}, "
              f"~{iter_per_epoch} steps) ...", flush=True)

        for batch in train_loader:
            images = batch["image"].to(device).float()
            labels = batch["label"].to(device).long()

            phase = lambda_sched.phase(global_step)
            for p in model.edain.hypernet.parameters():
                p.requires_grad = (phase != 0)
            anc_w, kl_w = lambda_sched.lambdas(global_step)
            combined_loss.set_lambdas(anc_w, kl_w)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                x_tilde, diag = model.edain(images, mask=(images != 0.0))
                logits = normalize_logits_output(model.backbone(x_tilde))
                ds_weights = get_deep_supervision_weights(len(logits))
                seg_loss = sum(
                    seg_loss_fn(h,
                                F.interpolate(labels.float(), size=h.shape[-3:],
                                              mode="nearest").long()
                                if h.shape[-3:] != labels.shape[-3:] else labels
                                ) * ds_weights[i]
                    for i, h in enumerate(logits)
                )

                anchor_params = model.edain.anchor_spline_params()
                current_params = rq_spline_parameterize(
                    diag["theta"],
                    K=K_KNOTS, B_supp=B_SUPP,
                    alpha_tail=ALPHA_TAIL, min_derivative=MIN_DERIVATIVE,
                )
                anc_loss = anchor_loss_fn(current_params, anchor_params)

                if kl_w > 0:
                    kl_val = kl_loss_fn(x_tilde, mask=(images != 0.0))
                else:
                    kl_val = x_tilde.new_zeros(())

                out = combined_loss(seg_loss, anchor_loss=anc_loss, kl_loss=kl_val)
                loss = out.total

            # NaN guard #1 (pre-backward): if loss is already non-finite, skip
            # everything. We must NOT call scaler.update() here because no
            # scaler.scale() was ever called for this step -- GradScaler's
            # state machine asserts that found_inf has been recorded before
            # update, and we have nothing to record. The scale factor simply
            # stays the same and the next step proceeds normally.
            if not torch.isfinite(loss):
                print(f"[warn] non-finite loss at step {global_step} "
                      f"(seg={float(seg_loss):.4f} anc={float(anc_loss):.4f}); "
                      f"skipping optimizer step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.edain.hypernet.parameters(), HYPERNET_GRAD_CLIP
            )

            # NaN guard #2 (post-backward): even with finite loss, AMP can
            # produce inf/NaN gradients (overflow during unscale). Here
            # scaler.unscale_ has been called, so the inf check IS recorded;
            # scaler.update() is therefore valid -- it will simply lower the
            # scale factor for the next step.
            grad_finite = True
            for p in hypernet_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                print(f"[warn] non-finite hypernet grad at step {global_step}; "
                      f"skipping optimizer step", flush=True)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                global_step += 1
                continue

            scaler.step(optimizer); scaler.update()

            if phase != 0:
                ema.update(model.edain.hypernet)

            with torch.no_grad():
                train_dice_accum += _dice_from_labels(
                    logits[0].argmax(dim=1, keepdim=True), labels).item()

            train_loss_accum += loss.item()
            train_seg_accum += out.seg.item()
            train_anc_accum += out.anchor.item()
            train_kl_accum += out.kl.item()
            num_batches += 1
            global_step += 1

        epoch_loss = train_loss_accum / num_batches
        epoch_train_dice = train_dice_accum / num_batches
        val_loss_epoch = None; val_dice_epoch = None
        r_median_val = None; r_p90_val = None

        decay = poly_lr(epoch, epochs_target)
        for pg in optimizer.param_groups:
            if pg.get("name") == "hypernet":
                pg["lr"] = hypernet_lr * decay
            else:
                pg["lr"] = BASE_LR * decay

        if epoch % VAL_INTERVAL == 0 or epoch == epochs_target:
            model.eval()
            dice_scores = []; val_loss_list = []; r_values = []
            n_val = len(val_loader)
            print(f"[val] starting epoch {epoch} validation ({n_val} cases, "
                  f"sw_batch_size={sw_batch_size}) ...", flush=True)
            t_val = time.time()
            with torch.inference_mode(), \
                 amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()), \
                 ema.swap_in(model.edain.hypernet):
                for v_i, val_data in enumerate(val_loader):
                    val_images = val_data["image"].to(device).float()
                    val_labels = val_data["label"].to(device).long()

                    vol_mask = (val_images[0, 0] != 0.0)
                    gamma_vol = percentile_summary(
                        val_images[0, 0], vol_mask, percentiles=artifact.percentiles
                    )

                    def predictor(x):
                        x_tilde, _ = model.edain(
                            x, mask=(x != 0.0),
                            gamma_raw=gamma_vol.unsqueeze(0).expand(x.shape[0], -1),
                            return_diagnostics=False,
                        )
                        o = model.backbone(x_tilde)
                        if isinstance(o, (list, tuple)): return o[0]
                        return o

                    val_logits = sliding_window_inference(
                        inputs=val_images, roi_size=patch_size,
                        sw_batch_size=sw_batch_size,
                        predictor=predictor, overlap=0.5, mode="gaussian")
                    if isinstance(val_logits, (list, tuple)): val_logits = val_logits[0]
                    val_loss_list.append(seg_loss_fn(val_logits, val_labels).item())
                    post = KeepLargestConnectedComponent(applied_labels=[1])
                    val_pred = post(val_logits.argmax(dim=1, keepdim=True))
                    case_dice = float(_dice_from_labels(val_pred, val_labels).item())
                    dice_scores.append(case_dice)

                    gamma_std_vol = standardizer(gamma_vol.unsqueeze(0))
                    delta_vol = model.edain.hypernet(gamma_std_vol)
                    theta_vol = model.edain.theta_0 + delta_vol
                    r = compute_non_affineness(theta_vol, K=K_KNOTS, B_supp=B_SUPP)
                    r_values.append(float(r.item()))

                    # Free per-case temporaries before the next case (otherwise
                    # the CUDA caching allocator holds onto sliding-window peaks).
                    del val_logits, val_pred, val_images, val_labels
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Periodic progress so a stuck validation is obvious in logs.
                    if (v_i + 1) % 5 == 0 or v_i == n_val - 1:
                        avg = float(np.mean(dice_scores))
                        elapsed_val = time.time() - t_val
                        print(f"[val]   {v_i+1}/{n_val} | last_dice={case_dice:.4f} "
                              f"| running_avg={avg:.4f} | {elapsed_val:.1f}s",
                              flush=True)

            val_dice_epoch = float(np.mean(dice_scores))
            val_loss_epoch = float(np.mean(val_loss_list))
            r_arr = np.array(r_values)
            r_median_val = float(np.median(r_arr))
            r_p90_val = float(np.percentile(r_arr, 90))

            if val_dice_epoch > best_metric:
                best_metric = val_dice_epoch; best_epoch = epoch; epochs_no_improve = 0
                torch.save({
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_dice": val_dice_epoch,
                    "r_median": r_median_val,
                    "r_p90": r_p90_val,
                }, ckpt_path)
                print(f"  >>> New Best: Val_Dice={best_metric:.4f}")
            else:
                epochs_no_improve += VAL_INTERVAL

        elapsed = time.time() - t0
        log_str = (
            f"Ep {epoch:04d} | step={global_step:6d} | phase={lambda_sched.phase(global_step)} | "
            f"Loss={epoch_loss:.4f} (seg={train_seg_accum/num_batches:.4f} "
            f"anc={train_anc_accum/num_batches:.4f} kl={train_kl_accum/num_batches:.4f}) "
            f"lambdas=({combined_loss.lambda_anc:.1e},{combined_loss.lambda_kl:.1e}) | "
            f"Train_Dice={epoch_train_dice:.4f}"
        )
        if val_dice_epoch is not None:
            log_str += (f" | Val={val_dice_epoch:.4f} | Best={best_metric:.4f}"
                        f" | r_med={r_median_val:.4f} r_p90={r_p90_val:.4f}")
        log_str += f" | LR={get_lr(optimizer):.6f} | {elapsed:.1f}s"
        print(log_str, flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, global_step, lambda_sched.phase(global_step),
                f"{epoch_loss:.6f}", f"{train_seg_accum/num_batches:.6f}",
                f"{train_anc_accum/num_batches:.6f}",
                f"{train_kl_accum/num_batches:.6f}",
                f"{combined_loss.lambda_anc:.6e}", f"{combined_loss.lambda_kl:.6e}",
                f"{epoch_train_dice:.6f}",
                "" if val_loss_epoch is None else f"{val_loss_epoch:.6f}",
                "" if val_dice_epoch is None else f"{val_dice_epoch:.6f}",
                f"{get_lr(optimizer):.6f}", f"{elapsed:.2f}",
                "" if r_median_val is None else f"{r_median_val:.6f}",
                "" if r_p90_val is None else f"{r_p90_val:.6f}",
            ])

        if epochs_no_improve >= EARLY_STOP_PATIENCE and not smoke:
            print(f"[Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs.")
            break

    print(f"\nFold {CURR_FOLD} done | Best={best_metric:.4f} @ epoch {best_epoch}")
    return best_metric


# =============================================================================
# Main
# =============================================================================

def main():
    global DATA_DIR, SPLIT_JSON, OUT_DIR, LOG_DIR, ARTIFACT_DIR
    global NUM_PATCHES, BATCH_SIZE
    parser = argparse.ArgumentParser(description="Lipo MR - MRI-EDAIN v2")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold id from lipo_split.json (e.g. 0..4)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 5 train / 2 val / 2 epochs / patch=64")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help=f"Path to Lipo data dir containing Lipo-*_MR_*_image.nii.gz. "
             f"Default: {_DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--split_json", type=str, default=None,
        help=f"Path to lipo_split.json. Default: <data_dir>/lipo_split.json "
             f"or <data_dir>/../lipo_split.json.",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Override output / log / artifact root.",
    )
    parser.add_argument(
        "--num_patches", type=int, default=None,
        help=f"Patches per training image (default {NUM_PATCHES}). "
             f"Drop to 2 for 2080Ti-class GPUs (11 GB) if you hit OOM.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help=f"Train batch size (default {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--patch_size_max", type=int, default=None,
        help="Cap every auto-detected patch spatial dim. Ignored if "
             "--patch_size is given.",
    )
    parser.add_argument(
        "--patch_size", type=str, default=None,
        help="Explicit patch as comma-separated 'X,Y,Z' in this code's axis "
             "order. Overrides auto-fingerprint and --patch_size_max. "
             "nnU-Net's plan for Lipo is [Z=32, Y=224, X=256] -> pass "
             "'256,224,32'.",
    )
    parser.add_argument(
        "--max_channels", type=int, default=512,
        help="Cap on DynUNet filter count per level (default 512; nnU-Net "
             "uses 320 for Lipo).",
    )
    parser.add_argument(
        "--sw_batch_size", type=int, default=1,
        help="Sliding-window batch size at validation/inference (default 1). "
             "Each window holds a full DynUNet activation graph, so on a "
             "2080Ti at patch (256, 224, 32) anything above 1 risks memory "
             "thrashing and an apparent hang.",
    )
    parser.add_argument(
        "--hypernet_lr_factor", type=float, default=0.1,
        help="Hypernet LR as fraction of backbone LR (default 0.1; blueprint "
             "section 5.2 fallback for instability observed at phase 0->1 "
             "transition with the default 1e-2 nnU-Net LR). With momentum "
             "0.99 the steady-state amplification 1/(1-0.99)=100 makes the "
             "small (~6.6K params) hypernet diverge on the first real epoch "
             "of phase 1; 10x lower LR brings it back to a stable regime.",
    )
    parser.add_argument(
        "--frozen_hypernet", action="store_true",
        help="BASELINE #5 mode (blueprint section 6, RQSplineFixed). Forces "
             "the hypernet to stay frozen for the entire training. The spline "
             "is fixed at population Nyul (f_{theta_0}). Use this to obtain "
             "the proper control for `v2 spline learning improves over a "
             "fixed Nyul-spline preprocessing`. anc and KL losses stay at 0.",
    )
    args = parser.parse_args()

    if args.num_patches is not None:
        NUM_PATCHES = int(args.num_patches)
    if args.batch_size is not None:
        BATCH_SIZE = int(args.batch_size)

    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir).resolve()
    if args.split_json is not None:
        SPLIT_JSON = Path(args.split_json).resolve()
    else:
        # Try DATA_DIR/lipo_split.json first, then DATA_DIR/../lipo_split.json.
        candidates = [DATA_DIR / "lipo_split.json", DATA_DIR.parent / "lipo_split.json"]
        for c in candidates:
            if c.exists():
                SPLIT_JSON = c
                break
        else:
            SPLIT_JSON = candidates[0]  # keep first; error later
    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir) / "outputs"
        LOG_DIR = Path(args.out_dir) / "logs"
        ARTIFACT_DIR = Path(args.out_dir) / "artifacts"

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Lipo data dir not found: {DATA_DIR}\n"
            f"Pass --data_dir <path-to-lipo-data>."
        )
    if not SPLIT_JSON.exists():
        raise FileNotFoundError(
            f"lipo_split.json not found at {SPLIT_JSON}. Pass --split_json."
        )

    set_determinism(seed=args.seed)
    torch.backends.cudnn.benchmark = True

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LIPO MR  -  MRI-EDAIN v2")
    print("=" * 70)
    print(f"Data dir   : {DATA_DIR}")
    print(f"Split JSON : {SPLIT_JSON}")
    print(f"Output     : {OUT_DIR}")
    print(f"Artifacts  : {ARTIFACT_DIR}")
    print(f"Device     : GPU {args.gpu if torch.cuda.is_available() else 'N/A (CPU)'}")
    explicit_patch = None
    if args.patch_size is not None:
        parts = [p.strip() for p in args.patch_size.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"--patch_size must be 3 comma-separated ints (X,Y,Z), got {args.patch_size!r}"
            )
        explicit_patch = tuple(int(p) for p in parts)

    print(f"Smoke      : {args.smoke}  |  Epochs override: {args.epochs}")
    print(f"Batch={BATCH_SIZE}, NumPatches={NUM_PATCHES}, "
          f"PatchSize={explicit_patch or 'auto'}, "
          f"PatchSizeMax={args.patch_size_max}, MaxChannels={args.max_channels}")
    print("=" * 70 + "\n")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    fp = load_data_and_compute_fingerprint(
        DATA_DIR, SPLIT_JSON, smoke=args.smoke,
        patch_size_max=args.patch_size_max,
        max_channels=args.max_channels,
        explicit_patch_size=explicit_patch,
    )
    folds = fp["folds"]

    print(f"Loaded {len(folds)} folds from {SPLIT_JSON.name}:")
    for fo in folds:
        print(f"  Fold {fo['fold']}: train={len(fo['train_files'])} val={len(fo['val_files'])}")

    if args.fold is not None:
        fold_info = next(fo for fo in folds if fo["fold"] == args.fold)
        train_fold(fold_info, fp, device, smoke=args.smoke, max_epochs=args.epochs,
                   sw_batch_size=args.sw_batch_size,
                   hypernet_lr_factor=args.hypernet_lr_factor,
                   frozen_hypernet=args.frozen_hypernet)
    elif args.smoke:
        train_fold(folds[0], fp, device, smoke=True, max_epochs=args.epochs or 2,
                   sw_batch_size=args.sw_batch_size,
                   hypernet_lr_factor=args.hypernet_lr_factor,
                   frozen_hypernet=args.frozen_hypernet)
    else:
        for fold_info in folds:
            train_fold(fold_info, fp, device, max_epochs=args.epochs,
                       sw_batch_size=args.sw_batch_size,
                       hypernet_lr_factor=args.hypernet_lr_factor,
                       frozen_hypernet=args.frozen_hypernet)

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
