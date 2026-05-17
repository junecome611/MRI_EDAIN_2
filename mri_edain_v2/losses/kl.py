"""KL-to-N(0, 1) anchor (blueprint section 2.7, principle P4).

L_KL = sum_k KL(p_hat(X_tilde_k) || N(0, 1))

p_hat is estimated from the current batch's foreground voxels via a soft
Gaussian histogram (differentiable). The blueprint flags this as a WEAK
regulariser that MAY compress biologically meaningful tumor heterogeneity --
lambda_KL = 0 is a required ablation cell (B4) and tumor contrast preservation
kappa_i (Phase-I Metric 5) must be monitored.

For memory safety on large volumes, we randomly subsample foreground voxels
per-batch (default 50,000 voxels) before building the soft histogram. The
random sampling does NOT use gradient flow; it just shrinks the workload.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class KLAnchorLoss(nn.Module):
    """KL(p_hat(X_tilde foreground) || N(0, 1)) over a soft histogram.

    Args:
        n_bins: number of histogram bins (default 50; blueprint section 2.7).
        value_range: (lo, hi) histogram support. Default (-4, 4) matches the
            spline support.
        bandwidth: Gaussian kernel bandwidth in the same units as values.
            Default = bin width (no oversmoothing).
        n_subsample: max number of foreground voxels per batch element used
            for the histogram. Default 50,000. Set to 0 to disable subsampling.
        eps: numerical floor for log.
    """

    def __init__(
        self,
        n_bins: int = 50,
        value_range: Tuple[float, float] = (-4.0, 4.0),
        bandwidth: Optional[float] = None,
        n_subsample: int = 50000,
        eps: float = 1e-8,
    ):
        super().__init__()
        if n_bins < 4:
            raise ValueError(f"n_bins must be >= 4, got {n_bins}")
        lo, hi = float(value_range[0]), float(value_range[1])
        if hi <= lo:
            raise ValueError(f"value_range must be increasing, got ({lo}, {hi})")

        self.n_bins = int(n_bins)
        self.lo = lo
        self.hi = hi
        self.n_subsample = int(n_subsample)
        self.eps = float(eps)

        bin_edges = torch.linspace(lo, hi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = float(bin_edges[1] - bin_edges[0])

        self.bin_width = bin_width
        self.bandwidth = float(bandwidth) if bandwidth is not None else bin_width

        # Target distribution: standard normal pdf evaluated at bin centres,
        # converted to a probability mass over bins (sums to 1).
        target_logp = -0.5 * bin_centers.pow(2) - 0.5 * math.log(2 * math.pi)
        target_p = torch.exp(target_logp) * bin_width
        target_p = target_p / (target_p.sum() + self.eps)

        self.register_buffer("bin_centers", bin_centers)
        self.register_buffer("target_p", target_p)

    def _soft_histogram(self, values: torch.Tensor) -> torch.Tensor:
        """Differentiable soft histogram of `values` (1-D) via Gaussian kernel.

        Returns a probability mass over bins (sums to 1), shape (n_bins,).
        """
        diffs = values.unsqueeze(-1) - self.bin_centers  # (N, M)
        weights = torch.exp(-0.5 * (diffs / self.bandwidth).pow(2))
        counts = weights.sum(dim=0)
        return counts / (counts.sum() + self.eps)

    def forward(
        self,
        X_tilde: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            X_tilde: normalized volume(s). Supported shapes:
                (D, H, W), (B, D, H, W), (B, 1, D, H, W).
            mask: foreground mask, same spatial layout.

        Returns:
            Scalar mean KL across the batch.
        """
        if X_tilde.ndim == 5:
            X_tilde = X_tilde[:, 0]
            mask = mask[:, 0] if mask.ndim == 5 else mask
        elif X_tilde.ndim == 3:
            X_tilde = X_tilde.unsqueeze(0)
            mask = mask.unsqueeze(0)

        B = X_tilde.shape[0]
        mask_bool = mask.to(torch.bool)
        total = X_tilde.new_zeros(())
        valid = 0

        for b in range(B):
            fg = X_tilde[b][mask_bool[b]]
            if fg.numel() == 0:
                continue
            if self.n_subsample > 0 and fg.numel() > self.n_subsample:
                # Random subsample without gradient tracking on the index op.
                idx = torch.randperm(fg.numel(), device=fg.device)[: self.n_subsample]
                fg = fg[idx]

            p_hat = self._soft_histogram(fg)
            kl = (
                p_hat
                * (torch.log(p_hat + self.eps) - torch.log(self.target_p + self.eps))
            ).sum()
            total = total + kl
            valid += 1

        if valid == 0:
            return total  # zero scalar
        return total / valid

    def extra_repr(self) -> str:
        return (
            f"n_bins={self.n_bins}, range=({self.lo}, {self.hi}), "
            f"bandwidth={self.bandwidth:.4f}, n_subsample={self.n_subsample}"
        )
