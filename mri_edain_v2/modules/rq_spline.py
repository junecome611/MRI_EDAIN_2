"""Rational-quadratic monotone spline (blueprint section 2.5, 4.2, 4.3).

Reference: Durkan, Bekasov, Murray, Papamakarios. "Neural Spline Flows".
NeurIPS 2019 (arXiv:1906.04032), Eq. 4.

Differences from Durkan 2019 normalising-flow usage:
- Linear tails (NOT identity) with fixed slope alpha_tail in (0, 1), so outliers
  outside the support are compressed rather than passed through (blueprint
  section 2.5, error 13.6).
- Boundary derivatives are pinned to alpha_tail for C1 continuity at +/- B.

Supports both single-spline (1-D params) and batched (B, K+1) application.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Spline parameters dataclass
# -----------------------------------------------------------------------------

@dataclass
class SplineParams:
    """Parameters of a (possibly batched) RQ-spline.

    Shapes:
        knot_x: (K+1,) or (B, K+1)   -- monotone, ranges over [-B_supp, +B_supp]
        knot_y: (K+1,) or (B, K+1)   -- monotone, ranges over [-B_supp, +B_supp]
        derivs: same shape as knot_x -- positive, boundary = alpha_tail
        B_supp: scalar (Python float) -- magnitude of support, e.g. 4.0
        alpha_tail: scalar (Python float) -- tail slope, e.g. 0.5
    """
    knot_x: torch.Tensor
    knot_y: torch.Tensor
    derivs: torch.Tensor
    B_supp: float
    alpha_tail: float

    @property
    def K(self) -> int:
        return int(self.knot_x.shape[-1] - 1)

    @property
    def is_batched(self) -> bool:
        return self.knot_x.ndim == 2

    @property
    def batch_size(self) -> int:
        return int(self.knot_x.shape[0]) if self.is_batched else 1


# -----------------------------------------------------------------------------
# Parameterizer: logits -> SplineParams
# -----------------------------------------------------------------------------

def rq_spline_parameterize(
    theta_logits: torch.Tensor,
    K: int = 9,
    B_supp: float = 4.0,
    alpha_tail: float = 0.5,
    min_derivative: float = 1e-3,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
) -> SplineParams:
    """Convert raw (3K-1)-dim logits into monotone-enforced spline parameters.

    Layout of theta_logits along the last dim:
        [theta_w (K), theta_h (K), theta_d (K-1)]
    where theta_d covers ONLY the internal knots 1..K-1; the boundary
    derivatives at knots 0 and K are pinned to alpha_tail for C1 continuity.

    Args:
        theta_logits: shape (..., 3K - 1).
        K, B_supp, alpha_tail, min_derivative: see blueprint section 2.5.
        min_bin_width, min_bin_height: minimum fraction of the support that
            each bin must occupy (Durkan 2019 default 1e-3, mirrors the
            reference neural-spline-flows code). Without this floor, softmax
            can concentrate mass on a single bin, leaving other bins with
            near-zero width; rq_spline_apply then divides by ~0 and produces
            NaN. This was observed at step 2088 of Lipo fold-0 phase 1.

    Returns:
        SplineParams with knot_x/knot_y/derivs same leading dims as theta_logits.
    """
    expected = 3 * K - 1
    if theta_logits.shape[-1] != expected:
        raise ValueError(
            f"theta_logits last dim must be 3K-1 = {expected}, "
            f"got {theta_logits.shape[-1]} (K={K})"
        )

    theta_w = theta_logits[..., :K]
    theta_h = theta_logits[..., K:2 * K]
    theta_d = theta_logits[..., 2 * K:]  # (..., K-1)

    # Widths and heights via softmax * 2B, but each bin reserved a minimum
    # fraction `min_bin_*` of the total support. The remaining (1 - K * min_*)
    # fraction is distributed by softmax. Both sums still equal 2B exactly.
    w_softmax = torch.softmax(theta_w, dim=-1)
    h_softmax = torch.softmax(theta_h, dim=-1)
    widths = (min_bin_width + (1.0 - min_bin_width * K) * w_softmax) * 2.0 * B_supp
    heights = (min_bin_height + (1.0 - min_bin_height * K) * h_softmax) * 2.0 * B_supp

    # Internal derivatives via softplus + min floor (positive).
    internal_d = F.softplus(theta_d) + min_derivative  # (..., K-1)

    # Boundary derivatives are pinned to alpha_tail for C1 continuity with
    # the linear tail. They are NOT learnable in this design (blueprint 2.5).
    boundary_shape = list(theta_logits.shape[:-1]) + [1]
    boundary_d = theta_logits.new_full(boundary_shape, fill_value=alpha_tail)
    all_derivs = torch.cat([boundary_d, internal_d, boundary_d], dim=-1)  # (..., K+1)

    # Knot positions: x_0 = -B_supp, x_i = x_{i-1} + w_i.
    zeros_lead = theta_logits.new_zeros(boundary_shape)
    cumw = torch.cumsum(widths, dim=-1)
    cumh = torch.cumsum(heights, dim=-1)
    knot_x = torch.cat([zeros_lead, cumw], dim=-1) - B_supp  # (..., K+1)
    knot_y = torch.cat([zeros_lead, cumh], dim=-1) - B_supp

    return SplineParams(
        knot_x=knot_x,
        knot_y=knot_y,
        derivs=all_derivs,
        B_supp=B_supp,
        alpha_tail=alpha_tail,
    )


class RQSplineParameterizer(nn.Module):
    """nn.Module wrapper around `rq_spline_parameterize`."""

    def __init__(
        self,
        K: int = 9,
        B_supp: float = 4.0,
        alpha_tail: float = 0.5,
        min_derivative: float = 1e-3,
    ):
        super().__init__()
        self.K = int(K)
        self.B_supp = float(B_supp)
        self.alpha_tail = float(alpha_tail)
        self.min_derivative = float(min_derivative)

    def forward(self, theta_logits: torch.Tensor) -> SplineParams:
        return rq_spline_parameterize(
            theta_logits,
            K=self.K,
            B_supp=self.B_supp,
            alpha_tail=self.alpha_tail,
            min_derivative=self.min_derivative,
        )

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, B_supp={self.B_supp}, "
            f"alpha_tail={self.alpha_tail}, min_derivative={self.min_derivative}"
        )


# -----------------------------------------------------------------------------
# Apply: voxel-wise evaluation of a monotone spline
# -----------------------------------------------------------------------------

def _gather_per_voxel(t1d_or_2d: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather knot/deriv values at per-voxel bin indices.

    Args:
        t1d_or_2d: (K+1,) or (B, K+1) tensor (knot_x, knot_y, or derivs).
        idx:       voxel indices, (N,) for unbatched or (B, N) for batched.

    Returns:
        Gathered values, same shape as idx.
    """
    if t1d_or_2d.ndim == 1:
        return t1d_or_2d[idx]
    # batched: (B, K+1).gather(1, (B, N)) -> (B, N)
    return t1d_or_2d.gather(1, idx)


def rq_spline_apply(X: torch.Tensor, params: SplineParams) -> torch.Tensor:
    """Differentiable voxel-wise application of an RQ monotone spline.

    Supports two modes inferred from `params`:
        - Unbatched: params.knot_x is 1-D, applied to any-shape X.
        - Batched:   params.knot_x is 2-D (B, K+1); X must lead with that B.

    Implements Durkan 2019 Eq. 4 on [-B_supp, +B_supp]; clipped linear tails
    outside the support (blueprint section 2.5).
    """
    knot_x = params.knot_x
    knot_y = params.knot_y
    derivs = params.derivs
    Bsup = float(params.B_supp)
    alpha = float(params.alpha_tail)
    K = params.K

    if knot_x.ndim not in (1, 2):
        raise ValueError(f"params.knot_x must be 1-D or 2-D, got {knot_x.ndim}-D")
    is_batched = params.is_batched

    # Flatten X to either (N,) or (B, N) according to params layout.
    orig_shape = X.shape
    if is_batched:
        if X.shape[0] != knot_x.shape[0]:
            raise ValueError(
                f"Leading batch dim of X ({X.shape[0]}) does not match "
                f"params batch ({knot_x.shape[0]})"
            )
        spatial_shape = X.shape[1:]
        X_flat = X.reshape(X.shape[0], -1)
    else:
        spatial_shape = X.shape
        X_flat = X.reshape(-1)

    # Numerically nudge boundary so searchsorted assigns the boundary
    # voxels to a valid interior bin (knot_x[K-1], knot_x[K]] etc.
    eps_edge = 1e-6
    X_clamped = X_flat.clamp(min=-Bsup + eps_edge, max=Bsup - eps_edge)

    # side="right" so that values exactly on knot_x[j] map to bin j (not j-1).
    idx = torch.searchsorted(knot_x.contiguous(), X_clamped.contiguous(), side="right") - 1
    idx = idx.clamp(min=0, max=K - 1)

    x_left = _gather_per_voxel(knot_x, idx)
    x_right = _gather_per_voxel(knot_x, idx + 1)
    y_left = _gather_per_voxel(knot_y, idx)
    y_right = _gather_per_voxel(knot_y, idx + 1)
    d_left = _gather_per_voxel(derivs, idx)
    d_right = _gather_per_voxel(derivs, idx + 1)

    bin_w = x_right - x_left
    bin_h = y_right - y_left
    s = bin_h / bin_w  # average slope over the bin

    xi = (X_clamped - x_left) / bin_w
    one_minus_xi = 1.0 - xi
    zeta = xi * one_minus_xi

    # Durkan 2019 Eq. 4
    numer = bin_h * (s * xi * xi + d_left * zeta)
    denom = s + (d_left + d_right - 2.0 * s) * zeta
    f_inner = y_left + numer / denom

    # Linear tails: connect at y_0 = knot_y[..., 0] (= -B_supp numerically)
    # and y_K = knot_y[..., K] (= +B_supp numerically; widths/heights are
    # softmax * 2B so cumulative sum is exactly 2B). Use the actual stored
    # values to avoid any numerical drift relative to f_inner at the boundary.
    if is_batched:
        y_at_left = knot_y[:, 0:1]    # (B, 1)
        y_at_right = knot_y[:, -1:]   # (B, 1)
    else:
        y_at_left = knot_y[0]
        y_at_right = knot_y[-1]

    f_left_tail = y_at_left + alpha * (X_flat + Bsup)
    f_right_tail = y_at_right + alpha * (X_flat - Bsup)

    below = X_flat < -Bsup
    above = X_flat > Bsup
    in_support = ~(below | above)

    f_out_flat = torch.where(
        in_support,
        f_inner,
        torch.where(below, f_left_tail, f_right_tail),
    )

    if is_batched:
        return f_out_flat.reshape(X.shape[0], *spatial_shape)
    return f_out_flat.reshape(orig_shape)


class RQSplineApply(nn.Module):
    """Stateless nn.Module wrapper around `rq_spline_apply` for composability."""

    def forward(self, X: torch.Tensor, params: SplineParams) -> torch.Tensor:
        return rq_spline_apply(X, params)


# -----------------------------------------------------------------------------
# Convenience: outlier ratio (blueprint section 2.5)
# -----------------------------------------------------------------------------

@torch.no_grad()
def outlier_ratio(
    X: torch.Tensor, mask: torch.Tensor, B_supp: float
) -> torch.Tensor:
    """Fraction of foreground voxels with |X| > B_supp (scalar tensor).

    Reported per blueprint 2.5: if > 5% across many scans, widen B_supp.
    """
    fg = X[mask.to(torch.bool)]
    if fg.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=X.device)
    out = ((fg < -B_supp) | (fg > B_supp)).to(torch.float32).mean()
    return out
