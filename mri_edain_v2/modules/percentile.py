"""11-landmark percentile summary on foreground (blueprint section 2.3, 4.1).

CRITICAL: output is detached. We deliberately stop gradients here so that
gradients flow only through the hypernetwork and spline parameters, never
through data-derived summary statistics (avoids the double-moving-target
problem described in error 13.2 of the blueprint).
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Shah 2011 11-landmark percentile set (blueprint section 2.3, 3.2 PERCENTILES)
PERCENTILES = (0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99)

# torch.quantile has a hard 2**24 element limit; we subsample beyond this.
# At >1M foreground voxels percentiles are already Monte-Carlo-tight to <1e-3,
# so a sample down to 16M is statistically harmless for Nyul landmarks.
_MAX_QUANTILE_INPUT = (1 << 24) - 1  # 16,777,215


def percentile_summary(
    X: torch.Tensor,
    mask: torch.Tensor,
    percentiles: tuple = PERCENTILES,
    min_foreground_voxels: int = 100,
    max_quantile_input: int = _MAX_QUANTILE_INPUT,
) -> torch.Tensor:
    """Compute the 11-d percentile vector on foreground voxels.

    Args:
        X: 3D volume tensor [D, H, W] (single channel).
        mask: boolean tensor of identical spatial shape, foreground = True.
        percentiles: tuple of percentile fractions in (0, 1).
        min_foreground_voxels: if foreground has fewer voxels than this, fall
            back to whole-volume percentiles (degenerate case guard, section 4.1).
        max_quantile_input: cap for torch.quantile (default 2**24 - 1). Inputs
            larger than this get random-sampled with replacement (cheap via
            torch.randint).

    Returns:
        gamma_raw: tensor of length len(percentiles), detached.
    """
    if X.shape != mask.shape:
        raise ValueError(
            f"X shape {tuple(X.shape)} does not match mask shape {tuple(mask.shape)}"
        )

    foreground = X[mask.to(torch.bool)]
    if foreground.numel() < min_foreground_voxels:
        foreground = X.reshape(-1)

    # Guard against torch.quantile's 2**24 element limit (e.g. large body MR
    # after Otsu crop). Random sample with replacement keeps the call cheap.
    if foreground.numel() > max_quantile_input:
        N = foreground.numel()
        idx = torch.randint(0, N, (max_quantile_input,), device=foreground.device)
        foreground = foreground[idx]

    q = torch.as_tensor(percentiles, dtype=X.dtype, device=X.device)
    gamma_raw = torch.quantile(foreground.to(torch.float32), q.to(torch.float32))
    return gamma_raw.to(X.dtype).detach()


class PercentileSummary(nn.Module):
    """Module wrapper around `percentile_summary` so it composes in a pipeline.

    Stateless. Output is detached (no gradients propagate through quantile).
    """

    PERCENTILES = PERCENTILES

    def __init__(
        self,
        percentiles: tuple = PERCENTILES,
        min_foreground_voxels: int = 100,
    ):
        super().__init__()
        if not all(0.0 < p < 1.0 for p in percentiles):
            raise ValueError(f"percentiles must lie in (0, 1), got {percentiles}")
        self.percentiles = tuple(percentiles)
        self.min_foreground_voxels = int(min_foreground_voxels)

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return percentile_summary(
            X,
            mask,
            percentiles=self.percentiles,
            min_foreground_voxels=self.min_foreground_voxels,
        )

    def extra_repr(self) -> str:
        return f"n_percentiles={len(self.percentiles)}"
