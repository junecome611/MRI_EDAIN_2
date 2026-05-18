"""Per-fold offline precomputation (blueprint section 2.3, 2.8, 4.4).

For each fold:
    1. Iterate the training cases through the SAME preprocessing the trainer
       will use (orient, spacing, foreground crop, per-volume z-score).
    2. Compute gamma_raw on the foreground of each z-scored volume.
    3. Fit the CoordinateStandardizer (per-percentile mean and std).
    4. Fit theta_0 via L-BFGS so the RQ-spline f_{theta_0} approximates the
       piecewise-linear mapping from population landmarks to Phi^{-1}(p).
    5. Cache the artifact to disk for reuse across seeds.

Artifact format: a single .pt file with a `PrecomputeArtifacts` dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from mri_edain_v2.modules.foreground import ForegroundExtractor
from mri_edain_v2.modules.nyul_init import fit_population_nyul_theta_0
from mri_edain_v2.modules.percentile import PERCENTILES, percentile_summary
from mri_edain_v2.modules.standardizer import CoordinateStandardizer


# ---------------------------------------------------------------------------
# Artifact container
# ---------------------------------------------------------------------------

@dataclass
class PrecomputeArtifacts:
    """Container for per-fold precomputed values."""
    standardizer_mu: torch.Tensor       # (n_perc,)
    standardizer_sigma: torch.Tensor    # (n_perc,)
    theta_0: torch.Tensor               # (3K - 1,)
    raw_gammas: torch.Tensor            # (N_train, n_perc), z-scored intensity scale
    population_landmarks: torch.Tensor  # (n_perc,) = raw_gammas.mean(dim=0)
    percentiles: Tuple[float, ...]
    K: int
    B_supp: float
    n_train_cases: int
    case_ids: List[str]
    notes: Dict[str, str]

    def to_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "standardizer_mu": self.standardizer_mu,
            "standardizer_sigma": self.standardizer_sigma,
            "theta_0": self.theta_0,
            "raw_gammas": self.raw_gammas,
            "population_landmarks": self.population_landmarks,
            "_percentiles": torch.tensor(self.percentiles, dtype=torch.float64),
            "_K": torch.tensor(self.K),
            "_B_supp": torch.tensor(self.B_supp),
            "_n_train_cases": torch.tensor(self.n_train_cases),
            "_case_ids_json": self.case_ids,
            "_notes_json": self.notes,
        }

    @classmethod
    def from_state_dict(cls, sd: Dict[str, torch.Tensor]) -> "PrecomputeArtifacts":
        return cls(
            standardizer_mu=sd["standardizer_mu"],
            standardizer_sigma=sd["standardizer_sigma"],
            theta_0=sd["theta_0"],
            raw_gammas=sd["raw_gammas"],
            population_landmarks=sd["population_landmarks"],
            percentiles=tuple(sd["_percentiles"].tolist()),
            K=int(sd["_K"].item()),
            B_supp=float(sd["_B_supp"].item()),
            n_train_cases=int(sd["_n_train_cases"].item()),
            case_ids=list(sd["_case_ids_json"]),
            notes=dict(sd["_notes_json"]),
        )


def save_precomputed_artifacts(art: PrecomputeArtifacts, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art.to_state_dict(), str(path))


def load_precomputed_artifacts(path: Union[str, Path]) -> PrecomputeArtifacts:
    sd = torch.load(str(path), map_location="cpu", weights_only=False)
    return PrecomputeArtifacts.from_state_dict(sd)


# ---------------------------------------------------------------------------
# Optional outlier clip transform (inline definition so this module has no
# dependency on a training entry script). Mirrors the implementation in
# code/lipo_mri_edain_v2.py:ClipForegroundPercentile so semantics match.
# ---------------------------------------------------------------------------

class _DefaultClipForegroundPercentile:
    """Non-dict transform that clips foreground (nonzero) voxels of a single
    array to [percentile_lo, percentile_hi]. Used inside the default precompute
    Compose when outlier_clip=='percentile'.
    """

    def __init__(self, percentile_lo: float = 0.005, percentile_hi: float = 0.995):
        self.percentile_lo = float(percentile_lo)
        self.percentile_hi = float(percentile_hi)

    def __call__(self, img):
        was_tensor = torch.is_tensor(img)
        orig_dtype = img.dtype if was_tensor else None
        arr = (img.detach().cpu().numpy().astype(np.float32)
               if was_tensor else np.asarray(img, dtype=np.float32))
        out = arr.copy()
        if out.ndim == 4:
            for c in range(out.shape[0]):
                fg = out[c][out[c] != 0]
                if fg.size >= 100:
                    lo, hi = np.percentile(
                        fg, [self.percentile_lo * 100, self.percentile_hi * 100]
                    )
                    out[c] = np.clip(out[c], lo, hi)
        elif out.ndim == 3:
            fg = out[out != 0]
            if fg.size >= 100:
                lo, hi = np.percentile(
                    fg, [self.percentile_lo * 100, self.percentile_hi * 100]
                )
                out = np.clip(out, lo, hi)
        if was_tensor:
            return torch.from_numpy(out).to(orig_dtype)
        return out


# ---------------------------------------------------------------------------
# Default per-case preprocessor: matches what the trainer does upstream of
# the spline. Users can pass in a custom one if their pipeline differs.
# ---------------------------------------------------------------------------

def default_per_case_preprocessor(
    image_path: str,
    target_pixdim: Optional[Tuple[float, float, float]] = None,
    foreground_method: str = "nonzero",
    outlier_clip: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the trainer's upstream pipeline to a single case and return
    (X_zscored, mask) ready for percentile_summary.

    Mirrors the MONAI Compose used by the trainer:
        LoadImage -> EnsureChannelFirst -> Orientation(RAS) -> Spacing
        -> CropForeground(>0) -> [opt clip 0.5/99.5 percentile]
        -> NormalizeIntensity(nonzero=True)

    `outlier_clip="percentile"` applies a per-volume [0.5%, 99.5%] foreground
    clip before z-score (nnU-Net default; recommended on body MR datasets
    where bright outliers inflate std and distort the post-z-score
    distribution).
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
    steps.append(CropForeground(select_fn=lambda x: x > 0, margin=5))
    if outlier_clip == "percentile":
        steps.append(_DefaultClipForegroundPercentile(0.005, 0.995))
    steps.append(NormalizeIntensity(nonzero=True))

    pipeline = Compose(steps)
    arr = pipeline(str(image_path))  # MetaTensor (1, D, H, W)
    if torch.is_tensor(arr):
        X = arr[0].to(torch.float32).cpu()
    else:
        X = torch.as_tensor(np.asarray(arr)[0], dtype=torch.float32)

    if foreground_method == "nonzero":
        mask = X != 0
    else:
        fe = ForegroundExtractor(method=foreground_method)
        mask = fe(X)

    return X, mask


# ---------------------------------------------------------------------------
# Main precompute entry
# ---------------------------------------------------------------------------

def precompute_fold_artifacts(
    train_cases: List[Dict[str, str]],
    *,
    target_pixdim: Optional[Tuple[float, float, float]] = None,
    foreground_method: str = "nonzero",
    K: int = 9,
    B_supp: float = 4.0,
    percentiles: Tuple[float, ...] = PERCENTILES,
    nyul_iters: int = 200,
    nyul_lr: float = 0.5,
    per_case_preprocessor: Optional[Callable] = None,
    verbose: bool = True,
    max_cases: Optional[int] = None,
    anchor_type: str = "population_nyul",
    outlier_clip: str = "none",
) -> PrecomputeArtifacts:
    """Run the offline precompute pipeline for one fold.

    Args:
        train_cases: list of dicts with at least key "image" (path to NIfTI)
            and optionally "case_id". The trainer's training fold files.
        target_pixdim: optional (sx, sy, sz) for MONAI Spacing. None -> skip.
        foreground_method: passed to ForegroundExtractor.
        K, B_supp: spline support / knot count.
        percentiles: which percentiles to extract (default 11 Shah landmarks).
        nyul_iters, nyul_lr: L-BFGS hyperparameters for theta_0 fit.
        per_case_preprocessor: callable(image_path, target_pixdim,
            foreground_method) -> (X_zscored, mask). If None, uses default.
        verbose: print progress.
        max_cases: optional cap (smoke-test mode).

    Returns:
        PrecomputeArtifacts containing fit statistics.
    """
    if per_case_preprocessor is None:
        # Use default; pass outlier_clip via functools.partial so the loop
        # below can still call the preprocessor with 3 positional args only.
        from functools import partial
        preprocessor = partial(default_per_case_preprocessor,
                               outlier_clip=outlier_clip)
    else:
        # Custom preprocessor: caller is responsible for wiring outlier_clip
        # (e.g. Lipo trainer uses functools.partial on its own preprocessor).
        preprocessor = per_case_preprocessor

    cases_to_use = train_cases[:max_cases] if max_cases else train_cases
    N = len(cases_to_use)
    if N < 2:
        raise ValueError(f"Need >= 2 training cases, got {N}")

    if verbose:
        print(f"[precompute] computing gamma on {N} training cases ...")

    case_ids: List[str] = []
    gamma_rows: List[torch.Tensor] = []
    n_failed = 0

    for i, case in enumerate(cases_to_use):
        cid = case.get("case_id") or Path(case["image"]).stem
        try:
            X_zs, mask = preprocessor(case["image"], target_pixdim, foreground_method)
            g = percentile_summary(X_zs, mask, percentiles=percentiles)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  [{i+1}/{N}] {cid} FAILED: {e}")
            continue

        case_ids.append(cid)
        gamma_rows.append(g)

        if verbose and (i == 0 or (i + 1) % max(1, N // 10) == 0 or i == N - 1):
            print(f"  [{i+1}/{N}] {cid}: gamma_raw[0,5,10] = "
                  f"({g[0].item():.3f}, {g[5].item():.3f}, {g[10].item():.3f})")

    if len(gamma_rows) < 2:
        raise RuntimeError(
            f"Only {len(gamma_rows)} cases produced gammas (failed: {n_failed}). "
            f"Cannot fit standardizer."
        )

    raw_gammas = torch.stack(gamma_rows, dim=0)  # (N, 11)

    # Fit per-coordinate standardizer.
    if verbose:
        print(f"[precompute] fitting CoordinateStandardizer on "
              f"{raw_gammas.shape[0]} gammas ...")
    standardizer = CoordinateStandardizer(n_dim=len(percentiles))
    standardizer.fit(raw_gammas)

    # Fit theta_0.
    if verbose:
        print(f"[precompute] fitting theta_0 (anchor_type={anchor_type}) via L-BFGS ...")
    theta_0 = fit_population_nyul_theta_0(
        raw_gammas,
        K=K,
        B_supp=B_supp,
        n_iter=nyul_iters,
        lr=nyul_lr,
        percentiles=percentiles,
        warn_threshold_r0=-1.0,  # we report r_0 manually below
        verbose=verbose,
        anchor_type=anchor_type,
    )

    # Sanity: compute final r_0.
    from mri_edain_v2.modules.nyul_init import compute_non_affineness

    r_0 = compute_non_affineness(theta_0, K=K, B_supp=B_supp).item()
    if verbose:
        print(f"[precompute] theta_0 non-affineness r_0 = {r_0:.4f}")
        if anchor_type == "population_nyul" and r_0 < 0.10:
            # Only flag if population_nyul (where small r_0 means the data
            # itself is too Gaussian for Nyul to do meaningful work).
            # For anchor_type='identity', r_0 ~ 0 is the intended behaviour.
            print(f"  WARNING: r_0 < 0.10 with population_nyul anchor "
                  f"-> population Nyul is near-affine for this dataset "
                  f"(blueprint section 4.4 step 6: Plan B trigger).")

    population_landmarks = raw_gammas.mean(dim=0)

    notes = {
        "n_failed": str(n_failed),
        "foreground_method": foreground_method,
        "target_pixdim": "none" if target_pixdim is None else str(tuple(target_pixdim)),
        "anchor_type": anchor_type,
        "r_0_non_affineness": f"{r_0:.6f}",
        "population_landmark_min": f"{population_landmarks.min().item():.4f}",
        "population_landmark_max": f"{population_landmarks.max().item():.4f}",
    }

    return PrecomputeArtifacts(
        standardizer_mu=standardizer.mu.detach().clone(),
        standardizer_sigma=standardizer.sigma.detach().clone(),
        theta_0=theta_0.detach().clone(),
        raw_gammas=raw_gammas.detach().clone(),
        population_landmarks=population_landmarks.detach().clone(),
        percentiles=tuple(percentiles),
        K=int(K),
        B_supp=float(B_supp),
        n_train_cases=int(raw_gammas.shape[0]),
        case_ids=case_ids,
        notes=notes,
    )


def apply_artifacts_to_standardizer(
    artifact: PrecomputeArtifacts,
    standardizer: CoordinateStandardizer,
) -> None:
    """Load fit statistics from a precomputed artifact into a CoordinateStandardizer."""
    if standardizer.n_dim != artifact.standardizer_mu.shape[0]:
        raise ValueError(
            f"Standardizer n_dim ({standardizer.n_dim}) does not match "
            f"artifact ({artifact.standardizer_mu.shape[0]})"
        )
    standardizer.mu.copy_(artifact.standardizer_mu)
    standardizer.sigma.copy_(artifact.standardizer_sigma)
    standardizer.is_fit.fill_(1)
