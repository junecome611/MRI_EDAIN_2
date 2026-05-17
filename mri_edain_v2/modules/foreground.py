"""Foreground mask extraction (blueprint section 2.9, 3.2).

Computed once on the raw scan and constant throughout training.
- BraTS (skull-stripped): non-zero mask
- LLD-MMRI / WORC (air background): Otsu + morphological closing
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

try:
    from skimage.filters import threshold_otsu
    from scipy.ndimage import binary_closing
    _HAS_SCIPY_SKIMAGE = True
except ImportError:
    _HAS_SCIPY_SKIMAGE = False


class ForegroundExtractor:
    """Stateless foreground mask computation.

    Args:
        method: "nonzero" | "otsu" | "auto".
            "nonzero" : mask = X > 0 (BraTS-style, skull-stripped data).
            "otsu"    : Otsu threshold + 3x3x3 closing.
            "auto"    : "nonzero" if fraction of exact zeros > 0.05 else "otsu".
        closing_iterations: number of binary closing iterations for "otsu". Default 1.
    """

    VALID_METHODS = ("nonzero", "otsu", "auto")

    def __init__(self, method: str = "auto", closing_iterations: int = 1):
        if method not in self.VALID_METHODS:
            raise ValueError(
                f"method must be one of {self.VALID_METHODS}, got {method!r}"
            )
        self.method = method
        self.closing_iterations = closing_iterations

    def __call__(self, X) -> torch.Tensor:
        """Compute foreground mask.

        Args:
            X: 3D volume tensor or array, shape [D, H, W] (single channel).

        Returns:
            Boolean torch.Tensor of shape [D, H, W].
        """
        is_tensor = torch.is_tensor(X)
        if is_tensor:
            device = X.device
            X_np = X.detach().cpu().numpy()
        else:
            device = torch.device("cpu")
            X_np = np.asarray(X)

        method = self._resolve_method(X_np)

        if method == "nonzero":
            mask_np = X_np > 0
        else:
            if not _HAS_SCIPY_SKIMAGE:
                raise ImportError(
                    "Otsu/closing path requires scikit-image and scipy. "
                    "Install with: pip install scikit-image scipy"
                )
            try:
                thr = float(threshold_otsu(X_np))
            except Exception:
                thr = float(X_np.mean())
            mask_np = X_np > thr
            if self.closing_iterations > 0:
                structure = np.ones((3, 3, 3), dtype=bool)
                mask_np = binary_closing(
                    mask_np, structure=structure, iterations=self.closing_iterations
                )

        return torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    def _resolve_method(self, X_np: np.ndarray) -> str:
        if self.method != "auto":
            return self.method
        zero_frac = float((X_np == 0).mean())
        return "nonzero" if zero_frac > 0.05 else "otsu"
