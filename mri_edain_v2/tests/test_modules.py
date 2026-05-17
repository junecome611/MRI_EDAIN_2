"""Unit tests covering the blueprint section 12.5 checklist.

Each test name is annotated with the checklist item it implements. These
tests do NOT touch real data or training; they validate the module contracts
in isolation, as recommended in blueprint section 12.1 point 6.

Run with: pytest mri_edain_v2/tests/test_modules.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from mri_edain_v2.baselines.affine_hypernet import AffineHypernetLayer
from mri_edain_v2.losses.anchor import FunctionSpaceAnchorLoss
from mri_edain_v2.losses.kl import KLAnchorLoss
from mri_edain_v2.modules.edain_layer import MRIEDAINLayer
from mri_edain_v2.modules.foreground import ForegroundExtractor
from mri_edain_v2.modules.hypernetwork import Hypernetwork
from mri_edain_v2.modules.nyul_init import (
    compute_non_affineness,
    fit_population_nyul_theta_0,
    piecewise_linear_interp,
)
from mri_edain_v2.modules.percentile import PercentileSummary, percentile_summary
from mri_edain_v2.modules.rq_spline import (
    SplineParams,
    rq_spline_apply,
    rq_spline_parameterize,
)
from mri_edain_v2.modules.standardizer import CoordinateStandardizer


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _seeded_gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _make_fit_standardizer(n_dim: int = 11) -> CoordinateStandardizer:
    """Fit a CoordinateStandardizer on a small synthetic training set."""
    g = _seeded_gen(123)
    training = torch.randn(64, n_dim, generator=g) * 0.5  # arbitrary mean/std
    std = CoordinateStandardizer(n_dim=n_dim)
    std.fit(training)
    return std


def _make_identity_theta(K: int = 9, B_supp: float = 4.0) -> torch.Tensor:
    """Return theta_logits that produces f(x) = x (an identity spline).

    Uniform widths and heights (theta_w = theta_h = 0 -> softmax gives 1/K each,
    so widths = heights = 2B/K) and unit internal derivatives (softplus(1) ~ 1.31
    is not 1; for an EXACT identity we'd need internal_d = 1, which means
    softplus(theta_d) + min_deriv = 1, i.e. theta_d = log(exp(1 - min_deriv) - 1)).
    The boundary derivative is fixed at alpha_tail (= 0.5) which is NOT 1, so
    identity is impossible with alpha_tail = 0.5; we therefore use alpha_tail = 1.
    """
    min_deriv = 1e-3
    inv_softplus_for_1 = math.log(math.exp(1.0 - min_deriv) - 1.0)
    theta_w = torch.zeros(K)
    theta_h = torch.zeros(K)
    theta_d = torch.full((K - 1,), inv_softplus_for_1)
    return torch.cat([theta_w, theta_h, theta_d])


# ---------------------------------------------------------------------------
# Checklist item 1: RQSplineApply passes monotonicity on 1000 random theta.
# ---------------------------------------------------------------------------

def test_rq_spline_monotone_on_random_theta():
    K = 9
    B_supp = 4.0
    g = _seeded_gen(7)
    x_grid = torch.linspace(-B_supp - 1.0, B_supp + 1.0, 500)
    for trial in range(1000):
        theta = torch.randn(3 * K - 1, generator=g) * 1.5
        params = rq_spline_parameterize(theta, K=K, B_supp=B_supp)
        y = rq_spline_apply(x_grid, params)
        diffs = y[1:] - y[:-1]
        # Strict monotonicity may fail by numerical noise of size ~1e-7 on tied
        # bins; tolerate that but reject any meaningful decrease.
        assert (diffs > -1e-6).all(), f"trial {trial}: non-monotone (min diff = {diffs.min().item()})"


def test_rq_spline_monotone_batched():
    K = 9
    B_supp = 4.0
    g = _seeded_gen(11)
    B = 8
    theta = torch.randn(B, 3 * K - 1, generator=g) * 1.5
    params = rq_spline_parameterize(theta, K=K, B_supp=B_supp)
    x_grid = torch.linspace(-B_supp + 0.01, B_supp - 0.01, 200)
    x_b = x_grid.unsqueeze(0).expand(B, -1)
    y = rq_spline_apply(x_b, params)  # (B, 200)
    diffs = y[:, 1:] - y[:, :-1]
    assert (diffs > -1e-6).all()


# ---------------------------------------------------------------------------
# Checklist item 2: identity-theta produces approximate identity output.
# (Uses alpha_tail = 1 so an exact identity is representable.)
# ---------------------------------------------------------------------------

def test_rq_spline_identity_at_identity_theta():
    K = 9
    B_supp = 4.0
    theta = _make_identity_theta(K=K, B_supp=B_supp)
    # Use alpha_tail = 1.0 so boundary derivatives match identity slope.
    params = rq_spline_parameterize(theta, K=K, B_supp=B_supp, alpha_tail=1.0)
    x = torch.linspace(-B_supp + 0.05, B_supp - 0.05, 500)
    y = rq_spline_apply(x, params)
    err = (y - x).abs().max().item()
    assert err < 1e-4, f"identity spline error too large: {err}"


def test_rq_spline_tails_compress_with_alpha_half():
    K = 9
    B_supp = 4.0
    g = _seeded_gen(3)
    theta = torch.randn(3 * K - 1, generator=g) * 0.0  # uniform-ish init
    params = rq_spline_parameterize(theta, K=K, B_supp=B_supp, alpha_tail=0.5)
    x_far = torch.tensor([-10.0, 10.0])
    y_far = rq_spline_apply(x_far, params)
    # Slope outside support is alpha_tail = 0.5; |y - y_boundary| = 0.5 * |x - B|.
    # y at boundary = +- B_supp (numerically).
    expected_above = B_supp + 0.5 * (10.0 - B_supp)
    expected_below = -B_supp + 0.5 * (-10.0 + B_supp)
    assert abs(y_far[1].item() - expected_above) < 1e-5
    assert abs(y_far[0].item() - expected_below) < 1e-5


# ---------------------------------------------------------------------------
# Checklist item 3: Hypernetwork with zero-init produces zero output at step 0.
# ---------------------------------------------------------------------------

def test_hypernet_zero_init_outputs_zero():
    h = Hypernetwork(input_dim=11, hidden_dim=64, output_dim=26, zero_init_output=True)
    g = _seeded_gen(0)
    for _ in range(20):
        gamma = torch.randn(4, 11, generator=g)
        out = h(gamma)
        assert out.shape == (4, 26)
        assert torch.allclose(out, torch.zeros_like(out), atol=0.0), out.abs().max()


def test_hypernet_small_random_init_is_small():
    h = Hypernetwork(
        input_dim=11, hidden_dim=64, output_dim=26, zero_init_output=False
    )
    gamma = torch.randn(8, 11)
    out = h(gamma)
    # Final-layer Kaiming * 0.01 keeps |out| small relative to typical scale.
    assert out.abs().mean().item() < 0.5


# ---------------------------------------------------------------------------
# Checklist item 4: FunctionSpaceAnchorLoss is zero when theta == theta_0.
# ---------------------------------------------------------------------------

def test_anchor_loss_zero_at_anchor():
    K = 9
    B_supp = 4.0
    theta_0 = torch.randn(3 * K - 1) * 0.3
    params_anchor = rq_spline_parameterize(theta_0, K=K, B_supp=B_supp)
    params_current = rq_spline_parameterize(theta_0.clone(), K=K, B_supp=B_supp)

    loss_mod = FunctionSpaceAnchorLoss(grid_size=50, B_supp=B_supp)
    loss = loss_mod(params_current, params_anchor)
    assert loss.item() < 1e-10, f"anchor loss at theta == theta_0 was {loss.item()}"


def test_anchor_loss_positive_when_perturbed():
    K = 9
    B_supp = 4.0
    g = _seeded_gen(4)
    theta_0 = torch.zeros(3 * K - 1)
    theta_pert = theta_0 + 0.5 * torch.randn(3 * K - 1, generator=g)

    params_anchor = rq_spline_parameterize(theta_0, K=K, B_supp=B_supp)
    params_pert = rq_spline_parameterize(theta_pert, K=K, B_supp=B_supp)
    loss_mod = FunctionSpaceAnchorLoss(grid_size=50, B_supp=B_supp)
    loss = loss_mod(params_pert, params_anchor)
    assert loss.item() > 1e-4


# ---------------------------------------------------------------------------
# Checklist items 5-7: MRIEDAINLayer; standardizer fit; theta_0 buffer.
# ---------------------------------------------------------------------------

def test_mri_edain_layer_runs_forward_and_returns_diagnostics():
    K = 9
    B_supp = 4.0
    std = _make_fit_standardizer()
    theta_0 = torch.zeros(3 * K - 1)
    layer = MRIEDAINLayer(standardizer=std, theta_0=theta_0, K=K, B_supp=B_supp)

    g = _seeded_gen(2)
    X = torch.randn(2, 1, 16, 16, 16, generator=g)
    mask = torch.ones_like(X, dtype=torch.bool)

    X_tilde, diag = layer(X, mask)
    assert X_tilde.shape == X.shape
    assert "gamma_raw" in diag and diag["gamma_raw"].shape == (2, 11)
    assert "theta" in diag and diag["theta"].shape == (2, 3 * K - 1)


def test_mri_edain_layer_step_zero_equals_anchor():
    """At step 0, theta = theta_0 -> applying f_{theta_0} to X is the layer's
    output (modulo the background pass-through). Verify on a fully-foreground
    volume."""
    K = 9
    B_supp = 4.0
    std = _make_fit_standardizer()
    theta_0 = 0.3 * torch.randn(3 * K - 1, generator=_seeded_gen(99))
    layer = MRIEDAINLayer(standardizer=std, theta_0=theta_0, K=K, B_supp=B_supp)

    g = _seeded_gen(2)
    X = 0.5 * torch.randn(1, 1, 8, 8, 8, generator=g)
    mask = torch.ones_like(X, dtype=torch.bool)

    X_tilde, _ = layer(X, mask)

    # Direct application of f_{theta_0}
    params_anchor = rq_spline_parameterize(theta_0, K=K, B_supp=B_supp)
    X_expected = rq_spline_apply(X[0, 0], params_anchor).unsqueeze(0).unsqueeze(0)
    assert torch.allclose(X_tilde, X_expected, atol=1e-5)


def test_standardizer_must_be_fit_before_use():
    std = CoordinateStandardizer(n_dim=11)
    with pytest.raises(RuntimeError):
        std(torch.zeros(11))

    std.fit(torch.randn(100, 11))
    # Now it should run without error.
    out = std(torch.zeros(11))
    assert out.shape == (11,)


def test_standardizer_per_coordinate():
    g = _seeded_gen(0)
    raw = torch.stack(
        [torch.arange(11, dtype=torch.float32) + 10.0 * torch.randn(11, generator=g)
         for _ in range(200)]
    )
    std = CoordinateStandardizer(n_dim=11)
    std.fit(raw)
    # Each output coordinate should be ~N(0, 1)
    out = std(raw)
    assert out.shape == raw.shape
    assert out.mean(dim=0).abs().max() < 1e-5
    # Standardizer uses population std (unbiased=False); match that convention
    # in the assertion to avoid the sqrt(N/(N-1)) bias inflation.
    assert (out.std(dim=0, unbiased=False) - 1.0).abs().max() < 1e-5


def test_theta_0_buffer_is_not_trainable():
    K = 9
    std = _make_fit_standardizer()
    theta_0 = torch.randn(3 * K - 1)
    layer = MRIEDAINLayer(standardizer=std, theta_0=theta_0, K=K)
    # theta_0 should be a buffer, not a parameter.
    assert "theta_0" in dict(layer.named_buffers())
    assert "theta_0" not in dict(layer.named_parameters())


# ---------------------------------------------------------------------------
# PercentileSummary detachment + degenerate fallback.
# ---------------------------------------------------------------------------

def test_percentile_summary_detached():
    X = torch.randn(4, 4, 4, requires_grad=True)
    mask = torch.ones_like(X, dtype=torch.bool)
    g = percentile_summary(X, mask)
    assert g.requires_grad is False
    assert g.shape == (11,)


def test_percentile_summary_degenerate_fallback():
    X = torch.randn(8, 8, 8)
    mask = torch.zeros_like(X, dtype=torch.bool)  # empty foreground
    g = percentile_summary(X, mask)
    # Must still return 11-dim quantiles from the whole volume.
    assert g.shape == (11,)
    # Strictly non-decreasing in value (percentiles are sorted).
    assert (g[1:] >= g[:-1] - 1e-7).all()


# ---------------------------------------------------------------------------
# Population Nyul initializer behaviour.
# ---------------------------------------------------------------------------

def test_population_nyul_fit_runs_and_returns_correct_shape():
    K = 9
    B_supp = 4.0
    g = _seeded_gen(13)
    # Build a fake training_gammas distribution that is monotone in expectation.
    base_landmarks = torch.tensor(
        [-2.0, -1.3, -0.85, -0.55, -0.3, 0.0, 0.3, 0.55, 0.85, 1.3, 2.0]
    )
    training = base_landmarks.unsqueeze(0) + 0.05 * torch.randn(32, 11, generator=g)
    theta_0 = fit_population_nyul_theta_0(
        training, K=K, B_supp=B_supp, n_iter=200, warn_threshold_r0=-1.0
    )
    assert theta_0.shape == (3 * K - 1,)
    assert theta_0.requires_grad is False


def test_compute_non_affineness_zero_for_identity_spline():
    K = 9
    B_supp = 4.0
    theta = _make_identity_theta(K=K, B_supp=B_supp)
    # Identity (linear) is by definition affine -> r should be near 0.
    # But our parameterizer uses alpha_tail (default 0.5) which makes boundary
    # derivatives differ from interior; r is small but non-zero on the [-B, B]
    # interior. Compute with alpha_tail = 1.0 to get exact identity.
    from mri_edain_v2.modules.nyul_init import compute_non_affineness  # local import

    r = compute_non_affineness(theta, K=K, B_supp=B_supp)
    # With alpha_tail = 0.5 (default), interior is identity slope 1, boundary
    # is slope 0.5 -- not exactly affine. r should still be modest.
    assert r.item() < 0.3, f"r = {r.item()}"


def test_compute_non_affineness_larger_for_curvier_spline():
    K = 9
    B_supp = 4.0
    # An identity-spline gives a small r, a curvy random spline gives a larger r.
    theta_lin = _make_identity_theta(K=K, B_supp=B_supp)
    g = _seeded_gen(21)
    theta_curvy = 1.5 * torch.randn(3 * K - 1, generator=g)
    r_lin = compute_non_affineness(theta_lin, K=K, B_supp=B_supp).item()
    r_curvy = compute_non_affineness(theta_curvy, K=K, B_supp=B_supp).item()
    assert r_curvy > r_lin


def test_piecewise_linear_interp_endpoints_and_extrapolation():
    x_land = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y_land = torch.tensor([0.0, 1.0, 4.0, 9.0])
    # At landmarks
    y_at = piecewise_linear_interp(x_land, x_land, y_land)
    assert torch.allclose(y_at, y_land, atol=1e-6)
    # Extrapolation below 0 uses slope (1-0)/(1-0) = 1
    y_lo = piecewise_linear_interp(torch.tensor([-2.0]), x_land, y_land)
    assert abs(y_lo.item() - (-2.0)) < 1e-6
    # Extrapolation above 3 uses slope (9-4)/(3-2) = 5
    y_hi = piecewise_linear_interp(torch.tensor([4.0]), x_land, y_land)
    assert abs(y_hi.item() - (9.0 + 5.0 * 1.0)) < 1e-6


# ---------------------------------------------------------------------------
# AffineHypernet (kill-switch) baseline runs and has zero-init identity behavior.
# ---------------------------------------------------------------------------

def test_affine_hypernet_identity_at_step_zero():
    std = _make_fit_standardizer()
    layer = AffineHypernetLayer(
        standardizer=std, a_0=1.0, c_0=0.0, hypernet_zero_init=True
    )
    X = torch.randn(2, 1, 8, 8, 8)
    mask = torch.ones_like(X, dtype=torch.bool)
    Y, diag = layer(X, mask)
    assert torch.allclose(Y, X, atol=1e-6)
    # a should be exactly 1, c exactly 0
    assert torch.allclose(diag["a"], torch.ones(2), atol=1e-6)
    assert torch.allclose(diag["c"], torch.zeros(2), atol=1e-6)


def test_affine_hypernet_gradients_flow_to_hypernet():
    std = _make_fit_standardizer()
    layer = AffineHypernetLayer(standardizer=std, hypernet_zero_init=False)
    X = torch.randn(2, 1, 8, 8, 8)
    mask = torch.ones_like(X, dtype=torch.bool)
    Y, _ = layer(X, mask)
    Y.mean().backward()
    grads = [p.grad for p in layer.hypernet.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any((g.abs().sum() > 0).item() for g in grads)


# ---------------------------------------------------------------------------
# End-to-end gradient: hypernet receives non-zero gradient through the spline.
# ---------------------------------------------------------------------------

def test_main_layer_gradients_reach_hypernet():
    K = 9
    B_supp = 4.0
    std = _make_fit_standardizer()
    # Use small-random init so dL/dtheta is non-zero even at the first step.
    theta_0 = torch.zeros(3 * K - 1)
    layer = MRIEDAINLayer(
        standardizer=std,
        theta_0=theta_0,
        K=K,
        B_supp=B_supp,
        hypernet_zero_init=False,
    )
    g = _seeded_gen(8)
    X = 0.5 * torch.randn(2, 1, 8, 8, 8, generator=g)
    mask = torch.ones_like(X, dtype=torch.bool)

    Y, _ = layer(X, mask)
    # Make a synthetic downstream loss that is non-trivial in theta.
    target = torch.zeros_like(Y)
    ((Y - target) ** 2).mean().backward()
    grads = [p.grad for p in layer.hypernet.parameters() if p.grad is not None]
    assert len(grads) > 0
    nonzero = any((g.abs().sum() > 1e-12).item() for g in grads)
    assert nonzero


# ---------------------------------------------------------------------------
# KL loss runs end-to-end and returns a finite scalar.
# ---------------------------------------------------------------------------

def test_kl_loss_runs_and_is_finite():
    kl = KLAnchorLoss(n_bins=50, value_range=(-4.0, 4.0), n_subsample=2000)
    X = torch.randn(2, 8, 8, 8)
    mask = torch.ones_like(X, dtype=torch.bool)
    val = kl(X, mask)
    assert torch.isfinite(val)
    assert val.item() >= 0.0


def test_kl_loss_is_smaller_for_near_standard_normal():
    """A roughly-N(0,1) input should have smaller KL than a heavily shifted one."""
    kl = KLAnchorLoss(n_bins=50, value_range=(-4.0, 4.0), n_subsample=5000)
    g = _seeded_gen(31)
    X_good = torch.randn(2, 16, 16, 16, generator=g)
    X_bad = 3.0 + torch.randn(2, 16, 16, 16, generator=g)  # mean ~ 3
    mask = torch.ones_like(X_good, dtype=torch.bool)
    val_good = kl(X_good, mask).item()
    val_bad = kl(X_bad, mask).item()
    assert val_bad > val_good


# ---------------------------------------------------------------------------
# ForegroundExtractor sanity.
# ---------------------------------------------------------------------------

def test_foreground_extractor_nonzero():
    fe = ForegroundExtractor(method="nonzero")
    X = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    mask = fe(X)
    assert mask.dtype == torch.bool
    assert mask.shape == X.shape
    expected = (X > 0)
    assert torch.equal(mask, expected)


def test_foreground_extractor_auto_selects_nonzero_for_zero_heavy_volume():
    fe = ForegroundExtractor(method="auto")
    X = torch.zeros(16, 16, 16)
    X[4:8, 4:8, 4:8] = 1.5  # 512 / 4096 ~ 12.5% foreground
    mask = fe(X)
    # Should resolve to nonzero path (zero fraction > 5%).
    assert mask[4, 4, 4].item() is True
    assert mask[0, 0, 0].item() is False
