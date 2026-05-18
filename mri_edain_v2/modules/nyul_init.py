"""Population Nyul precomputation (blueprint section 2.8, 4.4).

Fit theta^(0) such that the RQ-spline f_{theta^(0)} approximates the
piecewise-linear Nyul mapping that takes mean training percentiles into the
standard-normal z-scores at the same CDF values. Used as a FROZEN buffer
inside MRIEDAINLayer; the hypernetwork outputs a residual on top.

Also exposes `compute_non_affineness(theta)` which is Phase-I diagnostic
Metric 1 (blueprint section 8.1) and the sanity check at the end of fitting.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

from mri_edain_v2.modules.percentile import PERCENTILES
from mri_edain_v2.modules.rq_spline import (
    rq_spline_parameterize,
    rq_spline_apply,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _inverse_normal_cdf(p: torch.Tensor) -> torch.Tensor:
    """Inverse standard-normal CDF (probit). Uses torch.special.ndtri if
    available, otherwise the Beasley-Springer/Moro approximation via
    erfinv (sqrt(2) * erfinv(2p - 1))."""
    if hasattr(torch.special, "ndtri"):
        return torch.special.ndtri(p)
    return math.sqrt(2.0) * torch.special.erfinv(2.0 * p - 1.0)


def piecewise_linear_interp(
    grid: torch.Tensor,
    x_landmarks: torch.Tensor,
    y_landmarks: torch.Tensor,
) -> torch.Tensor:
    """Evaluate a piecewise-linear function defined by `(x_landmarks, y_landmarks)`
    at the points in `grid`. Linear extrapolation is used outside the landmark
    range (slope of first/last segment).

    Args:
        grid:        1-D tensor of evaluation points, shape (M,).
        x_landmarks: 1-D tensor of strictly increasing landmark x-positions, (L,).
        y_landmarks: 1-D tensor of landmark y-positions, (L,).

    Returns:
        Interpolated values, shape (M,).
    """
    if x_landmarks.ndim != 1 or y_landmarks.ndim != 1 or grid.ndim != 1:
        raise ValueError("piecewise_linear_interp expects 1-D inputs")
    L = x_landmarks.shape[0]
    if y_landmarks.shape[0] != L:
        raise ValueError("x_landmarks and y_landmarks must have same length")
    if L < 2:
        raise ValueError("need at least 2 landmarks")

    idx = torch.searchsorted(x_landmarks.contiguous(), grid.contiguous()) - 1
    idx = idx.clamp(min=0, max=L - 2)
    x0 = x_landmarks[idx]
    x1 = x_landmarks[idx + 1]
    y0 = y_landmarks[idx]
    y1 = y_landmarks[idx + 1]
    denom = (x1 - x0).clamp(min=1e-12)
    slope = (y1 - y0) / denom
    return y0 + slope * (grid - x0)


# -----------------------------------------------------------------------------
# Non-affineness ratio r (blueprint section 8.1 Metric 1, end of section 4.4)
# -----------------------------------------------------------------------------

@torch.no_grad()
def compute_non_affineness(
    theta: torch.Tensor,
    K: int = 9,
    B_supp: float = 4.0,
    grid_size: int = 200,
) -> torch.Tensor:
    """r = sqrt(SS_residual_from_best_affine_fit / SS_total_centered) of
    f_theta evaluated on a dense grid in [-B_supp, +B_supp].

    Returns:
        scalar tensor for unbatched theta, or (...,) for batched theta with
        leading dims matching the leading dims of theta.
    """
    device = theta.device
    t = torch.linspace(-B_supp, B_supp, grid_size, device=device, dtype=theta.dtype)

    params = rq_spline_parameterize(theta, K=K, B_supp=B_supp)
    if theta.ndim == 1:
        f = rq_spline_apply(t, params)  # (M,)
    else:
        # Broadcast grid to match leading dims of theta.
        lead = theta.shape[:-1]
        t_b = t.expand(*lead, grid_size).contiguous()
        f = rq_spline_apply(t_b, params)  # (..., M)

    t_mean = t.mean()
    f_mean = f.mean(dim=-1, keepdim=True)
    cov = ((t - t_mean) * (f - f_mean)).sum(dim=-1)
    var_t = ((t - t_mean) ** 2).sum()  # scalar (same for all batch elements)
    a_star = cov / (var_t + 1e-12)
    c_star = f_mean.squeeze(-1) - a_star * t_mean

    f_fit = a_star.unsqueeze(-1) * t + c_star.unsqueeze(-1)
    ss_res = ((f - f_fit) ** 2).sum(dim=-1)
    ss_tot = ((f - f_mean) ** 2).sum(dim=-1)
    ss_tot = ss_tot.clamp(min=1e-12)
    return torch.sqrt(ss_res / ss_tot)


# -----------------------------------------------------------------------------
# Population Nyul fitting (blueprint section 4.4)
# -----------------------------------------------------------------------------

def fit_population_nyul_theta_0(
    training_gammas: torch.Tensor,
    K: int = 9,
    B_supp: float = 4.0,
    n_iter: int = 200,
    lr: float = 0.5,
    grid_size: int = 200,
    percentiles: tuple = PERCENTILES,
    warn_threshold_r0: float = 0.10,
    verbose: bool = False,
    anchor_type: str = "population_nyul",
) -> torch.Tensor:
    """Fit theta^(0) given a choice of anchor target.

    `anchor_type` options:
      - "population_nyul" (blueprint default): the spline approximates the
        piecewise-linear mapping that takes population landmarks to standard-
        normal z-scores at the corresponding percentile positions. This is the
        classical Nyul behaviour. WARNING on body MR datasets like Lipo:
        if the source distribution is heavily peaked at the low end the
        mapping develops huge local slopes (observed >30x for Lipo 1%->10%),
        amplifying noise and harming tumour segmentation. Confirmed on Lipo
        fold-0 (Val Dice plateaued at ~0.61 vs ~0.79 baseline).
      - "identity": target(x) = x. f_{theta_0} is fit to the identity
        function. The spline at init is essentially a no-op, the backbone
        sees the upstream-z-scored input directly, and the hypernet learns
        deviations driven by Dice/CE alone. r_0 is small by construction
        (the only deviation from identity is the alpha_tail=0.5 boundary
        slope vs identity slope 1). Safer default for new datasets.

    Args:
        training_gammas: (N, 11) tensor of per-case 11-percentile gamma in
            z-scored intensity scale.
        K, B_supp: spline grid.
        n_iter, lr: L-BFGS max_iter and learning rate.
        grid_size: number of grid points used for the fit.
        percentiles: percentile fractions matching the gamma columns.
        warn_threshold_r0: warn if final r_0 falls below this value.
        anchor_type: "population_nyul" | "identity".

    Returns:
        theta_0 of shape (3K - 1,), detached.
    """
    if training_gammas.ndim != 2 or training_gammas.shape[1] != len(percentiles):
        raise ValueError(
            f"training_gammas must be (N, {len(percentiles)}), "
            f"got {tuple(training_gammas.shape)}"
        )
    if anchor_type not in ("population_nyul", "identity"):
        raise ValueError(
            f"anchor_type must be 'population_nyul' or 'identity', got {anchor_type!r}"
        )

    device = training_gammas.device
    dtype = torch.float32

    # 1. Population landmarks: mean across training scans per percentile slot.
    L = training_gammas.to(dtype).mean(dim=0)  # (11,)

    # 2. Target positions (depend on anchor_type).
    if anchor_type == "population_nyul":
        # Map population landmarks -> standard normal z-scores at percentile.
        p = torch.as_tensor(percentiles, dtype=torch.float64, device=device).clamp(1e-6, 1 - 1e-6)
        target = _inverse_normal_cdf(p).to(dtype)  # (11,)
    else:  # identity
        # Target at each landmark x is x itself -> piecewise linear gives y=x.
        target = L.clone()

    # Landmarks must be strictly increasing for piecewise-linear interp; if a
    # tiny tie sneaks in (degenerate training data), break it with a small eps.
    eps = 1e-6
    for i in range(1, L.shape[0]):
        if L[i] <= L[i - 1]:
            L[i] = L[i - 1] + eps

    # 3-4. Dense grid + target piecewise-linear samples.
    grid = torch.linspace(-B_supp, B_supp, grid_size, device=device, dtype=dtype)
    target_on_grid = piecewise_linear_interp(grid, L, target)

    # Only weight fit error within the landmark range. Outside that range the
    # target is extrapolated by the first/last landmark segment slope, which
    # for population_nyul on Lipo can be ~30x and produces absurd target
    # values (e.g. -93 at grid=-4) that the bounded spline (linear tail with
    # alpha_tail<<30) cannot match. Weighting those points dominates the loss
    # and ruins the fit even in the interior.
    fit_mask = (grid >= L[0]) & (grid <= L[-1])
    # Always include at least the landmark range; if it's degenerate, fall
    # back to the full grid to keep L-BFGS well-posed.
    if fit_mask.sum() < 5:
        fit_mask = torch.ones_like(grid, dtype=torch.bool)

    # 5. Optimise theta_0 via L-BFGS (one outer step with `n_iter` inner iters).
    theta_0 = torch.zeros(3 * K - 1, device=device, dtype=dtype, requires_grad=True)
    optim = torch.optim.LBFGS(
        [theta_0],
        lr=lr,
        max_iter=n_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optim.zero_grad()
        params = rq_spline_parameterize(theta_0, K=K, B_supp=B_supp)
        f_out = rq_spline_apply(grid, params)
        loss = ((f_out - target_on_grid) ** 2)[fit_mask].mean()
        loss.backward()
        return loss

    final_loss = optim.step(closure)
    if verbose:
        print(f"[fit_population_nyul_theta_0] anchor_type={anchor_type}, "
              f"final loss (within landmark range) = {float(final_loss):.6e}")

    # 6. Verify non-affineness of theta_0 (blueprint section 4.4 step 6).
    r_0 = compute_non_affineness(theta_0.detach(), K=K, B_supp=B_supp).item()
    if r_0 < warn_threshold_r0:
        warnings.warn(
            f"Population Nyul is near-affine (r_0 = {r_0:.4f} < "
            f"{warn_threshold_r0}). Reconsider anchor design "
            f"(blueprint section 4.4 step 6: Plan B trigger condition).",
            RuntimeWarning,
            stacklevel=2,
        )

    return theta_0.detach()


# -----------------------------------------------------------------------------
# Convenience class wrapper
# -----------------------------------------------------------------------------

class PopulationNyulInitializer:
    """Convenience wrapper around `fit_population_nyul_theta_0`.

    Usage::

        init = PopulationNyulInitializer(K=9, B_supp=4.0)
        theta_0 = init.fit(training_gammas)   # (N, 11) -> (3K-1,)
    """

    def __init__(
        self,
        K: int = 9,
        B_supp: float = 4.0,
        n_iter: int = 200,
        lr: float = 0.5,
        grid_size: int = 200,
        percentiles: tuple = PERCENTILES,
        warn_threshold_r0: float = 0.10,
    ):
        self.K = int(K)
        self.B_supp = float(B_supp)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.grid_size = int(grid_size)
        self.percentiles = tuple(percentiles)
        self.warn_threshold_r0 = float(warn_threshold_r0)

    def fit(
        self,
        training_gammas: torch.Tensor,
        verbose: bool = False,
        anchor_type: str = "population_nyul",
    ) -> torch.Tensor:
        return fit_population_nyul_theta_0(
            training_gammas,
            K=self.K,
            B_supp=self.B_supp,
            n_iter=self.n_iter,
            lr=self.lr,
            grid_size=self.grid_size,
            percentiles=self.percentiles,
            warn_threshold_r0=self.warn_threshold_r0,
            verbose=verbose,
            anchor_type=anchor_type,
        )
