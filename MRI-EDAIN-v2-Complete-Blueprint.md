# MRI-EDAIN v2: Complete Engineering Blueprint for Code Generation

> **Purpose** | **目的**: This document is the single source of truth for implementing MRI-EDAIN v2. It is designed to be read by a local Claude instance to generate code, training pipelines, diagnostic protocols, and experiment orchestration. Conceptual context in Chinese; mathematical definitions, code interfaces, and decision criteria in English to avoid translation ambiguity.
>
> **本文档定位** | **Scope**: 完整工程蓝图，包含 architecture spec, mathematical foundation, training protocol, diagnostic protocol, baseline ladder, ablation matrix, decision criteria。Local Claude 应当按本文档的 specification 实现代码而非自由发挥。
>
> **Status** | **状态**: v2 (post critical review, post collapse analysis, post threshold critique). Supersedes v1.

---

## Table of Contents

1. [Executive Summary | 执行摘要](#1-executive-summary)
2. [Core Mathematical Foundation](#2-core-mathematical-foundation)
3. [Architecture Specification](#3-architecture-specification)
4. [Module-Level Pseudocode](#4-module-level-pseudocode)
5. [Training Protocol](#5-training-protocol)
6. [Baseline Ladder](#6-baseline-ladder)
7. [Ablation Matrix](#7-ablation-matrix)
8. [Diagnostic Protocol (3 Phases)](#8-diagnostic-protocol)
9. [Statistical Thresholds & Stop Criteria](#9-statistical-thresholds-and-stop-criteria)
10. [Decision Tree & Execution Timeline](#10-decision-tree-and-execution-timeline)
11. [Interpretability Artifacts](#11-interpretability-artifacts)
12. [Code Generation Guidance for Local Claude](#12-code-generation-guidance)
13. [Appendix A: Error & Correction History](#13-appendix-a-error-and-correction-history)
14. [Appendix B: Citation Discipline](#14-appendix-b-citation-discipline)
15. [Appendix C: Glossary](#15-appendix-c-glossary)

---

## 1. Executive Summary

### 1.1 中文高层总结

**问题定位**：原版 MRI-EDAIN 在 nnU-Net 框架内（Instance Norm 紧跟 first conv）的设计存在数学硬伤——任何 per-image per-channel affine 变换都会被 IN 精确吸收，导致两个 learnable scalar (m, s) 在数学上是冗余的。这解释了 paper 被 MICCAI 拒稿的根本原因。

**V2 方法学定位**：我们用 input-conditional rational-quadratic monotonic spline 替换原版的 affine 变换。它**架构上具有非线性 capacity**，理论上可以改变 intensity-dependent local contrast（IN 不能完全恢复）。但**这不是 free lunch**——训练后 spline **是否真的**学到有用的非线性、**是否真的** survive IN，是必须经过实验验证的 open question。

**V2 核心 spirit**：不在 diagnostic 完成前 commit final framing。先按 v2 实现完整方法，跑三阶段 diagnostic 协议，根据 multi-evidence concurrence 决定走 Plan A（method works）还是 Plan B（pivot to mechanistic study + non-IN backbone + OOD framing）。

**关键不确定性**：
- Spline 是否会 collapse 为 affine（理论上可能，需 diagnose）
- KL anchor 是否反而压掉 tumor heterogeneity（需 ablate）
- Population Nyúl 本身的非线性程度（需 verify）

### 1.2 English Decision-Critical Statement

**The method has architectural capacity for nonlinearity but no theoretical guarantee that training will preserve it under Dice+CE supervision with InstanceNorm downstream.** The diagnostic protocol (§8) is the central scientific instrument; the abandonment criterion (§9) is the central decision criterion. Implementation must produce all diagnostic metrics as standard training outputs, not as afterthoughts.

### 1.3 Six Core Principles | 六个准则

These six principles govern every implementation decision below. When in doubt, return to these.

**P1. Non-absorption verification is mandatory, not assumed.**
The claim "spline survives IN absorption" must be empirically verified per-experiment via post-IN feature difference (`Δ_post-IN`) and best-affine replacement tests. Implementation must log these metrics during training, not only at evaluation.

**P2. Affine-hypernetwork baseline is the kill-switch.**
A baseline that uses identical conditioning input and hypernetwork architecture, but outputs only affine `(a_i, c_i)`, must exist and be trained with identical hyperparameters. If the nonlinear spline does not exceed this baseline by `> δ_min` (TOST equivalence test), the nonlinear capacity contribution is not supported.

**P3. Function-space anchor, not parameter-space.**
The anchor loss regularizes the function `f_θ` toward the population Nyúl function `f_θ^(0)`, evaluated on a fixed sampling grid. Parameter-space `||θ - θ^(0)||²` is rejected because RQ-spline parameter symmetries (softmax shift-invariance) decouple parameter distance from function distance.

**P4. KL-to-N(0,1) is a weak regularizer with mandatory zero ablation.**
The KL anchor may compress tumor intensity contrast that is biologically meaningful. `λ_KL = 0` is a required ablation cell. Tumor contrast preservation (`κ_i`, defined in §8) must be monitored.

**P5. Thresholds are dataset-specific, not universal.**
The 0.005 Dice threshold from prior versions is rejected. The smallest effect size of interest `δ_min` is computed per-dataset as `max(0.005, 0.25 × σ_seed)`, where `σ_seed` is estimated empirically from 3+ seeds of the baseline configuration. Equivalence testing uses TOST.

**P6. Stop only on multi-evidence concurrence.**
Abandonment of the "nonlinear preprocessing as contribution" framing requires Phase I + Phase II + Phase III diagnostics to **all** indicate collapse, across ≥3 seeds on ≥1 dataset. No single metric is sufficient to trigger pivot.

---

## 2. Core Mathematical Foundation

### 2.1 IN Absorption Theorem (Why the original method failed)

**Theorem 2.1** (Exact invariance of Conv ∘ IN under per-image positive-affine input transformation)

Let `X_c` be a single-channel input volume and let `X'_c = m_c · X_c + s_c` for any `m_c > 0, s_c ∈ ℝ`. Let `IN` denote Instance Normalization computed per-instance per-channel. Then:

```
μ_c(X') = m_c · μ_c(X) + s_c
σ_c(X') = |m_c| · σ_c(X)

IN(X')_c = γ_c · [X'_c - μ_c(X')] / σ_c(X') + β_c
        = γ_c · [m_c(X_c - μ_c(X))] / [m_c · σ_c(X)] + β_c
        = γ_c · [X_c - μ_c(X)] / σ_c(X) + β_c
        = IN(X)_c
```

The same holds after Conv: `IN(Conv(X')) = IN(Conv(X))` modulo IN's numerical `ε`, because the per-channel mean and std of the post-conv feature map absorb the affine variation linearly. **Conclusion**: any per-image per-channel scalar affine transformation contributes exactly zero to the downstream loss gradient through Conv ∘ IN. This is algebra, not pathology.

### 2.2 Nonlinearity Survival via Mean Value Theorem

For a nonlinear monotonic `f`, the difference between two voxels `p, p'` satisfies:

```
f(X(p)) - f(X(p')) = f'(ξ_{p,p'}) · [X(p) - X(p')]    where ξ_{p,p'} ∈ (X(p'), X(p))
```

Crucially, `f'(ξ_{p,p'})` depends on the intensity range. This means nonlinear `f` differentially scales local contrast across different intensity regions. After Conv:

```
Y(p) - Y(p') = W ★ [f(X(p)) - f(X(p'))]
            = [f'(ξ) varying with intensity] · [W ★ (X(p) - X(p'))]
```

Spatial structure of `Y` changes in ways **not captured by per-channel mean/std alone**. IN normalizes only channel-wise mean/std; the residual structural difference survives.

**Critical caveat (P1 in action)**: This argument only proves that *if* `f` is genuinely nonlinear, it *may* survive IN. It does **not** prove that training will produce a genuinely nonlinear `f`. The latter is open and must be empirically verified.

### 2.3 Conditioning Input: Per-Coordinate Standardized Percentile Vector

For each modality `k`, on the foreground mask `Ω` (Otsu-thresholded or precomputed brain mask), compute:

```
γ_k^raw = [q_0.01, q_0.10, q_0.20, q_0.30, q_0.40, q_0.50, q_0.60, q_0.70, q_0.80, q_0.90, q_0.99]_k ∈ ℝ^11
```

These are the Shah-2011 11-landmark percentiles. **Per-coordinate standardization** (P5 mechanism + MIP-required):

```
γ_{k,i} = (γ_{k,i}^raw - μ_i^train) / σ_i^train,    i = 1, ..., 11
```

where `μ_i^train, σ_i^train` are the training-set statistics for each percentile slot independently (not a global mean/std over all 11 dimensions). These statistics are **detached** — gradients do not flow through them.

**Reference**: Gonzalez Ortiz, Guttag, Dalca, "Magnitude Invariant Parametrizations Improve Hypernetwork Learning" (ICLR 2024, arXiv:2304.07645).

### 2.4 Hypernetwork: 2-layer GELU MLP with Zero-Init Output

```
θ_k = θ_k^(0) + MLP_φ_k(γ_k)

MLP_φ_k(γ_k) = W_3 · GELU(W_2 · GELU(W_1 · γ_k + b_1) + b_2) + b_3

W_1 ∈ ℝ^{64 × 11},   Kaiming init
W_2 ∈ ℝ^{64 × 64},   Kaiming init
W_3 ∈ ℝ^{(3K-1) × 64},  zero init  (← critical for population-Nyúl-at-init)
b_1, b_2 ∈ ℝ^64,     zero init
b_3 ∈ ℝ^{3K-1},      zero init
```

At training step 0: `MLP_φ_k(γ_k) = 0`, so `θ_k = θ_k^(0)` → spline equals population Nyúl mapping. **Terminology**: this is the **population-Nyúl anchor**, not "identity anchor" (the spline at init is not `f(x) = x` unless population Nyúl happens to be identity).

### 2.5 Rational-Quadratic Monotonic Spline (Durkan et al., NeurIPS 2019)

Set `K = 9` knots. MLP output `θ_k = [θ^(w), θ^(h), θ^(d)] ∈ ℝ^{3K-1}` decomposes as:

```
θ^(w) ∈ ℝ^K       (width logits)
θ^(h) ∈ ℝ^K       (height logits)
θ^(d) ∈ ℝ^{K-1}   (internal derivative logits)
```

**Monotonicity enforcement** (Durkan 2019 recipe):

```
w = 2B · softmax(θ^(w))                    # bin widths, sum = 2B
h = 2B · softmax(θ^(h))                    # bin heights, sum = 2B
d = softplus(θ^(d)) + ε                    # internal derivatives, positive
ε = 1e-3                                   # min-derivative floor
B = 4.0                                    # spline support: [-B, B]
```

Knot positions:

```
x_0 = -B,  x_i = x_{i-1} + w_i,  i = 1, ..., K
y_0 = -B,  y_j = y_{j-1} + h_j,  j = 1, ..., K
```

Within bin `i`, the rational-quadratic interpolant is given by Durkan 2019 Eq. 4. For a voxel value `x ∈ [x_{i-1}, x_i]`, let `ξ = (x - x_{i-1}) / w_i`:

```
ζ = ξ(1-ξ),  s_i = h_i / w_i

f(x) = y_{i-1} + h_i · [s_i · ξ² + d_{i-1} · ζ] / [s_i + (d_{i-1} + d_i - 2s_i) · ζ]
```

**Tail behavior (v2 modification, NOT identity)**: identity tails preserve outliers, contradicting normalization intent. Use **clipped linear tail**:

```
For x > B:   f(x) = y_K + α_tail · (x - B),   α_tail = 0.5  (default, fixed)
For x < -B:  f(x) = y_0 + α_tail · (x + B)
```

This compresses outliers with a fixed slope < 1 while preserving differentiability.

**Outlier ratio monitoring** (must be logged per epoch):

```
ρ_outlier_i = |{v ∈ Ω_i : X_i(v) < -B  or  X_i(v) > B}| / |Ω_i|
```

If `ρ_outlier > 5%` across a substantial fraction of training scans, the support `B` is too narrow and must be widened.

### 2.6 Voxel-Wise Application

```
X̃_k(v) = f_{θ_k}(X_k(v)),   ∀v ∈ Ω_k
X̃_k(v) = X_k(v),             ∀v ∉ Ω_k    (background passed through)
```

### 2.7 Composite Loss

```
L = L_seg + λ_anc · L_anc + λ_KL · L_KL
```

**Segmentation loss** (nnU-Net standard, do not modify):

```
L_seg = L_Dice + L_CE
```

**Function-space anchor loss** (P3):

```
L_anc = (1/L) · Σ_k Σ_ℓ [f_{θ_k}(t_ℓ) - f_{θ_k^(0)}(t_ℓ)]²

where {t_ℓ}_{ℓ=1}^L is a fixed grid on [-B, B], L = 50
```

The grid `{t_ℓ}` is uniform by default; an alternative is training-set-percentile-weighted (ablate via B10).

**KL-to-N(0,1) anchor** (weak regularizer, P4):

```
L_KL = Σ_k KL(p̂(X̃_k) || N(0, 1))
```

where `p̂(X̃_k)` is estimated from the current batch's foreground voxels via either a Gaussian kernel density estimate or 50-bin histogram softmax. Computed on detached `X̃_k.detach()` to prevent KL gradient from dominating the spline (only the spline parameters affect this via the forward pass; gradients still flow to `θ_k`).

**Default weights** (will be ablated):

```
λ_anc:  init = 1e-2, cosine-decay to 1e-4 over Phase 2
λ_KL:   init = 0,    ramp to 1e-4 over the first 5% of Phase 2
```

### 2.8 Population Nyúl Precomputation

`θ_k^(0)` is computed once before training. Algorithm:

```
1. For each training scan i, compute γ_{k,i}^raw on foreground mask.
2. Compute population landmarks: L_k = mean over i of γ_{k,i}^raw  (per percentile slot)
3. Define standard scale: S = [0, 100]  (or [-3, 3] post-standardization domain)
4. Build piecewise-linear mapping: PL_k mapping L_k → standard percentile positions in S
5. Fit RQ-spline parameters θ_k^(0) such that f_{θ_k^(0)} approximates PL_k on grid {t_ℓ}:
   minimize ||f_{θ_k^(0)}(t_ℓ) - PL_k(t_ℓ)||² for ℓ = 1..L
   via L-BFGS on a 200-iteration budget.
6. Verify: median over training scans of r^(0)_k = non-affineness ratio of f_{θ_k^(0)}.
   If r^(0)_k < 0.10, population Nyúl is itself near-affine; flag this and reconsider
   anchor design (Plan B trigger condition).
```

### 2.9 Foreground Mask Computation

```
For BraTS (skull-stripped, background = 0):
    Ω_k = {v : X_k(v) > 0}     (non-zero mask)

For LLD-MMRI and WORC (raw scans with air background):
    Ω_k = Otsu_threshold(X_k)
    Then morphological closing with 3×3×3 kernel to fill holes.
```

Foreground mask is computed **before standardization**, on the raw scan, and is constant throughout training.

---

## 3. Architecture Specification

### 3.1 System Diagram

```
                    [Raw 3D MRI volume X]
                            │
                ┌───────────┴──────────────┐
                │                          │
        [Foreground mask Ω]      [Per-modality scans X_k]
                │                          │
                └────────┬─────────────────┘
                         ▼
              [Compute γ_k^raw on Ω]
                         │
                         ▼  (detached, no gradient)
            [Per-coordinate standardize γ_k]
                         │
                         ▼
            [Hypernetwork MLP_φ_k(γ_k)]
                         │
                         ▼
              [θ_k = θ_k^(0) + MLP output]
                         │
                         ▼
          [Construct RQ-spline f_{θ_k}]
                         │
                         ▼
      [Voxel-wise apply: X̃_k = f_{θ_k}(X_k)]
                         │
                         ▼   (Concatenate modalities)
                    [Backbone U-Net]
                         │
                         ▼
              [Segmentation logits]
                         │
                         ▼
               [Compute L_seg + L_anc + L_KL]
                         │
                         ▼
                 [Backprop to MLP_φ_k]
                 [Backprop to U-Net]
```

### 3.2 Module Interface Specification

```python
# === Core modules to implement ===

class ForegroundExtractor:
    """Computes foreground mask. Stateless. Used at preprocessing."""
    def __init__(self, method: str = "auto"):  # "nonzero" | "otsu" | "auto"
        ...
    def __call__(self, X: Tensor[D,H,W]) -> Tensor[D,H,W]:  # boolean mask
        ...


class PercentileSummary:
    """Computes 11-dim percentile vector on foreground. Detached output."""
    PERCENTILES = [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    def __init__(self):
        ...
    def __call__(self, X: Tensor[D,H,W], mask: Tensor[D,H,W]) -> Tensor[11]:
        # MUST return detached tensor (no grad)
        ...


class CoordinateStandardizer:
    """Per-coordinate (per-percentile-slot) standardization.
    Statistics are fit on training set ONCE before training and frozen."""
    def __init__(self):
        self.mu: Tensor[11] = None   # set by fit()
        self.sigma: Tensor[11] = None
    def fit(self, training_gammas: Tensor[N, 11]) -> None:
        # Compute per-slot mean and std across N training scans
        ...
    def __call__(self, gamma_raw: Tensor[11]) -> Tensor[11]:
        return (gamma_raw - self.mu) / (self.sigma + 1e-8)


class Hypernetwork(nn.Module):
    """2-layer GELU MLP. Output layer is zero-initialized."""
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64,
                 output_dim: int = 26,  # = 3K - 1 for K = 9
                 zero_init_output: bool = True):
        ...
    def forward(self, gamma: Tensor[B, 11]) -> Tensor[B, output_dim]:
        # Returns the RESIDUAL Δθ to be added to θ^(0)
        ...


class RQSplineParameterizer:
    """Converts raw logits (θ^(w), θ^(h), θ^(d)) into monotone-enforced
    (w, h, d, knot_x, knot_y)."""
    def __init__(self, K: int = 9, B: float = 4.0,
                 alpha_tail: float = 0.5, min_derivative: float = 1e-3):
        ...
    def __call__(self, theta: Tensor[B, 3K-1]) -> SplineParams:
        # SplineParams dataclass with: knot_x, knot_y, internal_derivs, tail_slope
        ...


class RQSplineApply:
    """Differentiable voxel-wise application of an RQ-spline.
    Implements Durkan 2019 Eq. 4 with linear tails."""
    def __call__(self, X: Tensor[D,H,W], params: SplineParams) -> Tensor[D,H,W]:
        ...


class PopulationNyulInitializer:
    """Precomputes θ^(0) from training set foreground percentiles
    via L-BFGS fit of RQ-spline to piecewise-linear Nyúl mapping."""
    def __init__(self, K: int = 9, B: float = 4.0):
        ...
    def fit(self, training_gammas: Tensor[N, 11]) -> Tensor[3K-1]:
        # Returns θ^(0)
        ...


class MRIEDAINLayer(nn.Module):
    """Top-level: one instance per modality. Holds θ^(0) buffer + Hypernet."""
    def __init__(self, K: int = 9, B: float = 4.0,
                 standardizer: CoordinateStandardizer,
                 theta_0: Tensor[3K-1]):
        self.register_buffer("theta_0", theta_0)  # FROZEN
        self.hypernet = Hypernetwork(...)
        ...
    def forward(self, X: Tensor[D,H,W], mask: Tensor[D,H,W]) -> Tuple[Tensor, Dict]:
        # Returns (X_tilde, diagnostics_dict_for_logging)
        ...


class FunctionSpaceAnchorLoss:
    """L_anc = (1/L) · Σ_k Σ_ℓ [f_θ(t_ℓ) - f_θ^(0)(t_ℓ)]²"""
    def __init__(self, grid_size: int = 50, B: float = 4.0,
                 grid_type: str = "uniform"):  # "uniform" | "percentile_weighted"
        self.t_grid = torch.linspace(-B, B, grid_size)
    def __call__(self, edain_layer: MRIEDAINLayer,
                 spline_params: SplineParams) -> Tensor[scalar]:
        ...


class KLAnchorLoss:
    """L_KL = KL(p̂(X̃) || N(0, 1)) via 50-bin histogram + softmax."""
    def __init__(self, n_bins: int = 50, range: Tuple[float, float] = (-4, 4)):
        ...
    def __call__(self, X_tilde: Tensor[D,H,W], mask: Tensor[D,H,W]) -> Tensor[scalar]:
        # On X_tilde.detach() to avoid this gradient dominating spline params
        ...
```

---

## 4. Module-Level Pseudocode

This section provides mid-level pseudocode for the most consequential modules. Adhere closely.

### 4.1 PercentileSummary (Module 1)

```python
def percentile_summary(X, mask):
    """
    Compute 11-dim percentile vector on foreground.
    
    CRITICAL: output must be detached. Quantile op is not meaningfully
    differentiable for our purpose, and we deliberately stop gradients
    here to avoid the double-moving-target problem.
    """
    foreground_voxels = X[mask]  # 1D tensor of foreground intensities
    
    if foreground_voxels.numel() < 100:
        # Degenerate case: too few foreground voxels
        # Fall back to whole-volume percentiles with warning
        foreground_voxels = X.flatten()
    
    percentiles = torch.tensor([0.01, 0.10, 0.20, 0.30, 0.40,
                                 0.50, 0.60, 0.70, 0.80, 0.90, 0.99])
    gamma_raw = torch.quantile(foreground_voxels, percentiles)
    return gamma_raw.detach()
```

### 4.2 RQSplineParameterizer (Module 2)

```python
def rq_spline_parameterize(theta_logits, K=9, B=4.0,
                            alpha_tail=0.5, min_derivative=1e-3):
    """
    Parse the (3K-1)-dim logit vector into monotone-enforced spline params.
    
    Critical invariants:
    - widths and heights sum to 2B exactly (softmax × 2B)
    - all internal derivatives > min_derivative (softplus + floor)
    - knots strictly monotone in both x and y
    """
    theta_w = theta_logits[:K]                # K width logits
    theta_h = theta_logits[K:2*K]             # K height logits
    theta_d = theta_logits[2*K:]              # (K-1) internal derivative logits
    
    # Widths and heights via softmax * 2B (ensures sum = 2B, monotone knots)
    widths = 2 * B * torch.softmax(theta_w, dim=-1)
    heights = 2 * B * torch.softmax(theta_h, dim=-1)
    
    # Internal derivatives via softplus + min floor
    internal_d = F.softplus(theta_d) + min_derivative  # length K-1
    
    # Boundary derivatives: linear tail slope
    # Total derivatives array: [α_tail, internal_d_1, ..., internal_d_{K-1}, α_tail]
    boundary_d = torch.full((1,), alpha_tail, device=theta_logits.device)
    all_derivs = torch.cat([boundary_d, internal_d, boundary_d])  # length K+1
    
    # Construct knot positions
    knot_x = torch.cat([torch.tensor([-B]), -B + torch.cumsum(widths, dim=-1)])
    knot_y = torch.cat([torch.tensor([-B]), -B + torch.cumsum(heights, dim=-1)])
    
    return SplineParams(knot_x=knot_x, knot_y=knot_y, derivs=all_derivs,
                        alpha_tail=alpha_tail, B=B)
```

### 4.3 RQSplineApply (Module 3)

```python
def rq_spline_apply(X, params):
    """
    Apply rational-quadratic monotone spline element-wise.
    
    Reference: Durkan et al. NeurIPS 2019, Eq. 4 (rational-quadratic)
              + linear tail extension outside [-B, B].
    
    Implementation MUST be differentiable end-to-end and support 3D volumes.
    """
    knot_x = params.knot_x         # (K+1,)
    knot_y = params.knot_y         # (K+1,)
    derivs = params.derivs         # (K+1,)
    B = params.B
    alpha = params.alpha_tail
    
    # Clamp to find bin index per voxel
    # Use torch.bucketize or equivalent
    X_clamped = torch.clamp(X, min=-B + 1e-6, max=B - 1e-6)
    bin_idx = torch.bucketize(X_clamped, knot_x) - 1  # [0, K-1]
    bin_idx = torch.clamp(bin_idx, 0, K - 1)
    
    # Gather per-voxel: x_left, x_right, y_left, y_right, d_left, d_right
    x_left = knot_x[bin_idx]
    x_right = knot_x[bin_idx + 1]
    y_left = knot_y[bin_idx]
    y_right = knot_y[bin_idx + 1]
    d_left = derivs[bin_idx]
    d_right = derivs[bin_idx + 1]
    
    # Local bin width, height
    bin_w = x_right - x_left
    bin_h = y_right - y_left
    s = bin_h / bin_w
    
    # ξ ∈ [0, 1] within bin
    xi = (X_clamped - x_left) / bin_w
    
    # Rational-quadratic interpolant (Durkan Eq. 4)
    numer = bin_h * (s * xi**2 + d_left * xi * (1 - xi))
    denom = s + (d_left + d_right - 2 * s) * xi * (1 - xi)
    f_inner = y_left + numer / denom
    
    # Linear tails (override for X outside [-B, B])
    f_left_tail = y_left[0] + alpha * (X - (-B))
    f_right_tail = y_right[-1] + alpha * (X - B)
    
    # Combine
    in_support = (X >= -B) & (X <= B)
    f_out = torch.where(in_support, f_inner,
            torch.where(X < -B, f_left_tail, f_right_tail))
    
    return f_out
```

### 4.4 PopulationNyulInitializer (Module 4)

```python
def fit_population_nyul_theta_0(training_gammas, K=9, B=4.0,
                                  n_iter=200, lr=0.01):
    """
    Compute θ^(0) ∈ ℝ^{3K-1} such that the RQ-spline approximates
    the piecewise-linear population Nyúl mapping.
    
    Steps:
    1. Compute population landmarks: L = mean over training scans
       of standardized γ_raw (after coordinate standardization)
    2. Define target standard scale positions: equally spaced in [-B, B]
    3. Construct piecewise-linear mapping (L_i → target_i)
    4. Sample target piecewise-linear mapping on dense grid
    5. Optimize θ_0 via L-BFGS to match RQ-spline output to target on grid
    """
    # 1. Population landmarks (standardized)
    population_landmarks = training_gammas.mean(dim=0)  # (11,)
    
    # 2. Target positions in standardized output space
    # Target maps each input percentile to its standardized output percentile
    target_landmarks = torch.tensor([-2.33, -1.28, -0.84, -0.52, -0.25,
                                      0.00, 0.25, 0.52, 0.84, 1.28, 2.33])
    # Above values are z-scores at standard normal percentiles 1%, 10%, ..., 99%
    
    # 3. Dense grid for fitting
    grid = torch.linspace(-B, B, 200)
    
    # 4. Construct target on grid via piecewise-linear interpolation
    target_on_grid = piecewise_linear_interp(grid,
                                              population_landmarks,
                                              target_landmarks,
                                              extrapolation="linear")
    
    # 5. Optimize θ_0 via L-BFGS
    theta_0 = torch.zeros(3 * K - 1, requires_grad=True)
    optim = torch.optim.LBFGS([theta_0], lr=lr, max_iter=n_iter)
    
    def closure():
        optim.zero_grad()
        params = rq_spline_parameterize(theta_0, K=K, B=B)
        f_out = rq_spline_apply(grid, params)
        loss = (f_out - target_on_grid).pow(2).mean()
        loss.backward()
        return loss
    
    for _ in range(n_iter):
        optim.step(closure)
    
    # 6. Verify non-affineness of θ_0 (critical sanity check, see §8 Metric 1)
    r_0 = compute_non_affineness(theta_0, K=K, B=B)
    if r_0 < 0.10:
        warnings.warn(f"Population Nyúl is near-affine (r_0 = {r_0:.4f}). "
                       "Consider Plan B trigger condition.")
    
    return theta_0.detach()
```

### 4.5 FunctionSpaceAnchorLoss (Module 5)

```python
def function_space_anchor_loss(spline_params_current, theta_0_buffer,
                                t_grid, K=9, B=4.0):
    """
    L_anc = mean over grid of [f_θ(t) - f_θ^(0)(t)]²
    
    Note: t_grid is fixed and pre-stored; it does NOT depend on the
    current batch. This avoids implicit batch-statistic coupling.
    """
    # Evaluate current spline on grid
    params_0 = rq_spline_parameterize(theta_0_buffer, K=K, B=B)
    f_at_t_current = rq_spline_apply(t_grid, spline_params_current)
    f_at_t_anchor = rq_spline_apply(t_grid, params_0)
    
    diff_squared = (f_at_t_current - f_at_t_anchor).pow(2)
    return diff_squared.mean()
```

### 4.6 Affine-Hypernet Baseline Layer (Module 6, the kill-switch)

```python
class AffineHypernetLayer(nn.Module):
    """
    The CRITICAL baseline (P2).
    Identical conditioning input, identical hypernet, but outputs only (a, c).
    Apply: X̃ = a · X + c
    
    Initialization: zero-init output → (a, c) start at (a_0, c_0) where
    (a_0, c_0) are the population-mean affine parameters from training set.
    """
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64,
                 standardizer: CoordinateStandardizer,
                 a_0: float, c_0: float):
        super().__init__()
        self.standardizer = standardizer
        self.hypernet = Hypernetwork(input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=2,
                                     zero_init_output=True)
        self.register_buffer("a_0", torch.tensor(a_0))
        self.register_buffer("c_0", torch.tensor(c_0))
    
    def forward(self, X, mask):
        gamma_raw = percentile_summary(X, mask)
        gamma = self.standardizer(gamma_raw)
        delta_a, delta_c = self.hypernet(gamma).split(1, dim=-1)
        # Parameterize a > 0 via exp; c unconstrained
        a = self.a_0 * torch.exp(0.5 * torch.tanh(delta_a))
        c = self.c_0 + delta_c
        X_tilde = a * X + c
        return X_tilde
```

This baseline must be trained with identical optimizer, schedule, augmentations, and λ_anc/λ_KL ablation as the main method.

---

## 5. Training Protocol

### 5.1 Three-Phase Schedule

```
Total training budget: 250,000 steps (1000 epochs × 250 it/epoch)

Phase 0  (steps 0 to 2,500;     ~1% of budget):
    - Hypernetwork FROZEN (gradients not flowing to MLP_φ_k)
    - U-Net trains on population-Nyúl-normalized inputs
    - Purpose: warm up U-Net before exposing it to learnable preprocessing
    - λ_anc = 1e-2, λ_KL = 0

Phase 1  (steps 2,500 to 25,000; ~9% of budget):
    - Hypernetwork unfrozen
    - λ_anc held at 1e-2 (strong anchor)
    - λ_KL = 0 (KL not yet active)
    - Purpose: let hypernetwork learn small perturbations near population Nyúl

Phase 2  (steps 25,000 to 250,000; remaining 90%):
    - λ_anc cosine-decay from 1e-2 to 1e-4
    - λ_KL ramps from 0 to 1e-4 over first 5% of Phase 2 (steps 25,000-36,250)
    - Purpose: spline shapes to task while staying near Nyúl population manifold
```

### 5.2 Optimizer Configuration

```
Backbone (U-Net):
    Optimizer: SGD with Nesterov momentum 0.99
    LR: 1e-2 with PolyLR schedule, exponent 0.9
    Weight decay: 3e-5

Hypernetwork (MLP_φ_k):
    Same LR as backbone (per MIP recommendation; do not reduce unless instability observed)
    Same weight decay
    Gradient clipping: global norm 1.0 (CRITICAL, separate from backbone)
```

If hypernetwork training is unstable (gradient norms exploding, oscillating spline outputs):

```
Fallback: reduce hypernet LR to 1e-3 (10× lower than backbone)
          Document this deviation in the methods section.
```

### 5.3 EMA at Inference

```
Maintain exponential moving average of hypernetwork weights:
    φ_k^EMA ← 0.99 · φ_k^EMA + 0.01 · φ_k  (every step after Phase 0)

At inference, use φ_k^EMA, not φ_k.
At validation during training, use φ_k^EMA for stability.
```

### 5.4 Augmentation

Inherit nnU-Net default augmentation pipeline:
- Spatial: rotation, scaling, mirroring
- Intensity: gamma transform, brightness, contrast, gaussian noise, gaussian blur

**Crucial note**: gamma transform changes input intensity distribution. The spline will be forced to handle augmented inputs that may have different intensity statistics than training distribution. This is **a feature, not a bug** — it forces the spline to learn task-adaptive behavior rather than memorize training statistics. Do not disable gamma augmentation.

---

## 6. Baseline Ladder

### 6.1 Required Baselines (8 total)

| # | Method | Purpose | Datasets | Tag |
|---|--------|---------|----------|-----|
| 1 | **No normalization** | Floor control | All | [STD] |
| 2 | **Per-volume z-score on foreground** | nnU-Net default; the bar | All | [LIT] Isensee 2021 |
| 3 | **Percentile clip [0.5, 99.5] + min-max** | Classical | All | [STD] |
| 4 | **Fixed Nyúl-11 (Shah 2011, piecewise-linear)** | Classical Nyúl | All | [LIT] Shah 2011 |
| 5 | **Fixed RQ-spline at θ^(0)** | Controls for RQ-spline parameterization itself | All | [SPEC] v2 |
| 6 | **Affine-hypernet baseline** (P2 kill-switch) | Tests if nonlinear capacity matters | All | v2 critical |
| 7 | **WhiteStripe** | Brain-only comparator | BraTS only | [LIT] Shinohara 2014 |
| 8 | **MRI-EDAIN v2 (ours)** | Main method | All | [SPEC] |

### 6.2 Baseline Implementation Notes

- **#1 No norm**: Pass raw intensity (after foreground masking). Used only as a sanity floor.
- **#2 Z-score**: `(X - μ_Ω) / σ_Ω` per volume, where `μ_Ω, σ_Ω` are foreground statistics.
- **#3 Percentile clip**: Clip to `[q_0.005, q_0.995]`, then min-max scale to `[0, 1]`.
- **#4 Nyúl-11**: Use `intensity_normalization.normalize.nyul.NyulNormalize` from Reinhold's package. Default percentile set: `{1, 10, 20, ..., 90, 99}`. Train reference landmarks on training fold only.
- **#5 Fixed RQ-spline**: Implement as our main method but freeze hypernetwork output to zero (so θ_k = θ_k^(0) always). This isolates the RQ-spline parameterization effect from the input-conditional effect.
- **#6 Affine-hypernet**: See module spec in §4.6. Must use identical conditioning input, identical hypernet architecture, identical training schedule.
- **#7 WhiteStripe**: Use `WhiteStripeNormalize(width=0.05)` on T1n. Applicable to BraTS only (requires NAWM).
- **#8 Ours**: Main method as specified §2–§5.

### 6.3 Pairwise Comparisons That Matter

The baseline ladder is designed to enable these contrasts:

| Contrast | Question Answered |
|----------|------------------|
| #2 vs #4 | Does Nyúl piecewise-linear beat z-score? (Established literature claim) |
| #4 vs #5 | Does RQ-spline parameterization itself help beyond piecewise-linear? |
| #5 vs #6 | Does input-conditional learning help beyond a fixed spline? |
| **#6 vs #8** | **Does the nonlinear capacity of the spline matter beyond affine?** (P2) |
| #2 vs #8 | End-to-end gain over the nnU-Net SOTA baseline |

---

## 7. Ablation Matrix

All ablations run on BraTS (largest dataset), single backbone (nnU-Net-style), 3 seeds per cell.

| Cell | Question | Levels | Tag |
|------|----------|--------|-----|
| **B1** | Input-conditional vs global learnable spline | (a) global (b) input-conditional | [LIT] |
| **B2** | Spline knot count K | {5, 9, 13, 17} | [LIT] Durkan |
| **B3** | λ_anc (function-space) | {0, 1e-4, 1e-3, 1e-2} | [ENG] |
| **B4** | λ_KL | {0, 1e-5, 1e-4, 1e-3} | v2 critical |
| **B5** | Summary vector dim | (a) 11D (b) 13D add {5,95} (c) 15D + WhiteStripe stats | [SPEC] |
| **B6** | Per-modality vs shared MLP | (a) per (b) shared+modality one-hot | [ENG] |
| **B7** | Output init: zero vs small-random | (a) zero (b) Kaiming × 0.01 | v2 |
| **B8** | Spline parameterization | (a) RQ (b) piecewise-linear (c) cubic Hermite | [LIT] |
| **B9** | Per-coordinate standardization of γ | (a) on (b) off | [LIT] MIP |
| **B10** | Anchor type | (a) function-space (b) parameter-space | v2 |
| **B11** | Tail behavior | (a) identity (b) clipped linear (c) clipped flat | v2 |
| **B12** | **Shuffled-γ test** | θ_i = h_φ(γ_j), j ≠ i | v2 brutal test |

### B12 (Shuffled-γ) Critical Note

This is a **post-hoc inference-time** test, not a training-time ablation. After training the main method:

```
For each test volume i:
    Replace γ_i with a randomly chosen γ_j from another test volume (j ≠ i)
    Compute spline params via the trained hypernet on γ_j
    Apply spline to X_i
    Evaluate Dice

If Dice on shuffled-γ is statistically indistinguishable from Dice on correct γ:
    → Hypernet is NOT using per-image conditioning
    → "Input-conditional" claim has failed
    → This is a strong negative signal even if other metrics look fine.
```

---

## 8. Diagnostic Protocol (3 Phases)

The diagnostic protocol is the central scientific instrument of this paper. It must be built into the training pipeline as standard logging, not as post-hoc analysis.

### 8.1 Phase I: Online Monitoring (every 5–10 epochs)

Five metrics, all computed on the validation set:

#### Metric 1: Non-affineness ratio `r_i`

For each validation volume `i`, fit the best linear approximation to the trained spline on a dense grid `{t_ℓ}`:

```
(a_i*, c_i*) = argmin_{a, c} Σ_ℓ [f_{θ_i}(t_ℓ) - (a · t_ℓ + c)]²

residual_i = Σ_ℓ [f_{θ_i}(t_ℓ) - (a_i* · t_ℓ + c_i*)]²
total_i = Σ_ℓ [f_{θ_i}(t_ℓ) - mean(f_{θ_i}(t_ℓ))]²

r_i = sqrt(residual_i / total_i)    # equivalent to sqrt(1 - R²_affine)
```

Report: median `r`, 90th percentile `r`, fraction of volumes with `r > 0.10`.

**Threshold (dataset-specific)**: `r < 0.05` consistently → spline is effectively affine. Not a universal threshold; calibrate against the value at training step 0 (which should be `r^(0)_k`, the population Nyúl non-affineness).

#### Metric 2: Derivative coefficient of variation `CV(f')`

```
CV(f'_i) = std_{t ∈ [-B, B]}(f'_{θ_i}(t)) / mean_{t ∈ [-B, B]}(f'_{θ_i}(t))
```

If `CV(f') ≈ 0`, the derivative is approximately constant — the spline is approximately affine.

**Threshold**: `CV(f') < 0.10` is a fast canary for affine collapse.

#### Metric 3: Post-IN feature difference `η_i` (P1 mechanism)

This is the **single most decisive metric**. Compute on a fixed comparison subset of validation volumes:

```
For each volume X_i:
    Y_base = IN(W ★ X_i + b)              # feature map without spline
    Y_spline = IN(W ★ f_{θ_i}(X_i) + b)   # feature map with trained spline
    
    Δ_pre_i = ||W ★ f_{θ_i}(X_i) - W ★ X_i||₂ / ||W ★ X_i||₂
    Δ_post_i = ||Y_spline - Y_base||₂ / ||Y_base||₂
    
    η_i = Δ_post_i / (Δ_pre_i + 1e-8)    # survival ratio
```

Interpretation:
- `η_i ≈ 1`: spline change fully survives IN (best case, validates non-absorption claim)
- `η_i ≈ 0`: spline change is fully absorbed by IN (worst case, confirms collapse)
- `η_i ∈ (0, 1)`: partial survival

**Threshold**: median `η < 0.20` indicates near-complete IN absorption.

#### Metric 4: Effective rank of spline output

Construct a matrix `V ∈ ℝ^{N × L}` where `V_{i, ℓ} = f_{θ_i}(t_ℓ)` for validation volumes `i = 1, ..., N` and grid points `t_ℓ`.

Project out the affine basis (spanned by `(1, t_ℓ)`):

```
V_residual = V - V_affine_proj
```

Compute effective rank (Roy & Vetterli 2007):

```
singular_values = SVD(V_residual)
normalized = singular_values / sum(singular_values)
H = -Σ normalized · log(normalized)
erank = exp(H)
```

**Threshold**: `erank < 2.3` after projecting out affine → spline outputs collapse into ≤ 2D non-affine subspace, suggesting near-collapse.

#### Metric 5: Tumor intensity contrast preservation `κ_i`

For volumes with ground-truth segmentation available:

```
For each volume i:
    μ_tumor_raw = mean(X_i over tumor mask)
    μ_normal_raw = mean(X_i over normal-appearing foreground)
    σ_normal_raw = std(X_i over normal-appearing foreground)
    C_i_raw = |μ_tumor_raw - μ_normal_raw| / σ_normal_raw
    
    # Same on transformed X̃
    C_i_tilde = same computation on X̃_i = f_{θ_i}(X_i)
    
    κ_i = C_i_tilde / C_i_raw
```

**Threshold**: `κ_i < 0.5` for a substantial fraction of volumes → tumor contrast is being compressed, likely by overly strong KL anchor. Trigger λ_KL reduction.

### 8.2 Phase II: One-Shot Confirmation (if Phase I triggers)

**Test A: Affine-perturbation Dice stability**

Replace the trained spline at inference time:

```
For random (α, β, γ, δ) drawn from small Gaussians N(0, 0.05):
    f_tilde(x) = α · f(β · x + γ) + δ
    
Compute Dice on validation set with this perturbed spline.
Repeat for ~20 random perturbations.
Compute: std(Dice) / mean(Dice)
```

**Interpretation**: if this ratio is below `δ_min / Dice_mean`, the affine degrees of freedom around the spline are invisible to the network → most of the spline's effective output lives in the affine subspace.

**Test B: Inference-time best-affine replacement**

```
For each validation volume i:
    Fit best affine (a_i*, c_i*) to the trained spline:
        f_θ_i(x) ≈ a_i* · x + c_i*
    
    Construct f̃_i(x) = a_i* · x + c_i*
    Run inference with f̃_i instead of f_θ_i
    Compute Dice_replaced_i

Compute paired difference: ΔDice_i = Dice_original_i - Dice_replaced_i
TOST equivalence test at margin δ_min:
    H_0: |E[ΔDice]| ≥ δ_min
    H_1: |E[ΔDice]| < δ_min  (equivalence)

If H_0 is rejected → spline is functionally equivalent to its affine fit.
```

### 8.3 Phase III: Final Decision (if Phase II confirms collapse)

**Test C: Retrain affine-hypernet baseline from scratch**

This is the most expensive but most decisive test:

```
Train baseline #6 (AffineHypernetLayer) from scratch with:
    - Identical optimizer, schedule, augmentation
    - Identical training duration
    - Same 3+ seeds as main method
    - Same train/val split

Compute Dice_affine_hypernet for each seed.
Compare against Dice_main_method on TOST at margin δ_min.

If equivalent → the nonlinear capacity of the spline brings no trainable advantage.
              → "Nonlinear spline" framing is dead.
If main method significantly higher → nonlinear capacity provides genuine training advantage.
              → Framing survives despite Phase I/II.
```

### 8.4 Phase II/III Conflict Resolution

It is possible for Phase II to indicate collapse but Phase III to show the main method outperforms the affine-hypernet baseline. Interpretation:

```
Phase II shows: at inference, the trained spline is functionally affine
                (the nonlinear part doesn't change downstream output much)

Phase III shows: the *training trajectory* benefited from nonlinear capacity
                 (even if the endpoint is affine-like)

This is a real phenomenon (cf. neural network "lottery ticket" literature).
Reporting strategy in this case:
    → Frame the contribution as "nonlinear capacity benefits optimization"
      rather than "nonlinear spline contains useful nonlinearity"
    → This is a weaker but still valid framing
```

### 8.5 Guardrails

- **Warmup**: do not check abandonment criteria before step 7,500 (Phase 1 end + 50% margin)
- **Persistence**: require 3 consecutive monitoring checkpoints (~15 epochs) to agree before triggering Phase II
- **Seeds**: require ≥3 seeds for any abandonment decision
- **Datasets**: prefer concurrence across 2+ datasets, but BraTS alone is sufficient given its size

---

## 9. Statistical Thresholds and Stop Criteria

### 9.1 Dataset-Specific Minimum Effect Size

Replace the rejected universal threshold of 0.005 Dice with a calibrated value:

```
σ_seed = std(Dice across ≥3 seeds for baseline #2 (z-score, nnU-Net default))

δ_min = max(0.005, 0.25 × σ_seed)
```

Conservative alternative for small datasets (e.g., WORC):

```
δ_min = max(0.01, 0.50 × σ_seed)
```

### 9.2 TOST Equivalence Testing

Replace the naive `|ΔDice| < threshold` with two one-sided tests:

```
H_0_upper: E[ΔDice] ≥ +δ_min     (method genuinely better)
H_0_lower: E[ΔDice] ≤ -δ_min     (method genuinely worse)
H_1: -δ_min < E[ΔDice] < +δ_min  (equivalence)

Reject H_0_upper at α=0.05 (one-sided t-test)
Reject H_0_lower at α=0.05 (one-sided t-test)

If both rejected → conclude equivalence (TOST positive)
```

### 9.3 Abandonment Criterion (P6)

**Trigger pivot away from "nonlinear preprocessing as central contribution" if ALL of the following hold:**

| Phase | Condition | Threshold |
|-------|-----------|-----------|
| I | median r_i and 90% percentile r_i | < 0.05 and < 0.10 |
| I | median η_i (post-IN survival) | < 0.20 |
| I | erank after affine projection | < 2.3 |
| I | Persistence | ≥ 3 consecutive checkpoints |
| II | Affine-perturbation Dice stability | std/mean < δ_min / Dice |
| II | Best-affine replacement TOST | equivalence at δ_min |
| III | Retrain affine-hypernet TOST | equivalence at δ_min |

And: across ≥3 seeds on ≥1 dataset.

### 9.4 Framing-Survival Criterion (anti-pivot)

The original framing survives if:

```
Either of:
    median r_i > 0.10  AND  median η_i > 0.50
    
    OR
    
    Retrain affine-hypernet TOST rejects equivalence (main method > affine by ≥ δ_min)
```

### 9.5 Ambiguous Zone

If Phase I/II/III give mixed signals:

```
Default action: continue training; expand to additional datasets;
                widen anchor schedule; ablate λ_anc and λ_KL more aggressively.

Do NOT pivot on partial evidence.
Do NOT claim main framing victory on partial evidence.
```

---

## 10. Decision Tree and Execution Timeline

### 10.1 Execution Plan (Route B per user direction)

```
Week 1-2: Infrastructure
    - Implement all 8 baseline pipelines (§6)
    - Implement core modules (§4)
    - Implement diagnostic logging (§8 Phase I metrics) as standard training output
    - Set up MONAI U-Net and nnU-Net-style backbones

Week 3-4: Baseline noise floor estimation
    - Run baseline #2 (z-score nnU-Net) with 5 seeds on BraTS
    - Compute σ_seed → δ_min
    - Run baseline #2 on LLD-MMRI and WORC for their respective δ_min

Week 5-6: Main method training on BraTS
    - 3 seeds of MRI-EDAIN v2
    - Phase I metrics logged every 10 epochs
    - Check thresholds continuously

Week 7: Phase II tests (if triggered) or expansion to other datasets (if not)
    - If Phase I indicates potential collapse: run Phase II Test A and Test B
    - If Phase I indicates survival: begin LLD-MMRI and WORC runs

Week 8: Phase III decisive test (if Phase II confirms collapse)
    - Retrain baseline #6 (affine-hypernet) on BraTS with 3 seeds
    - TOST equivalence test against main method

Week 9: Decision point
    - Plan A: main method works → continue with full experiment matrix
    - Plan B: pivot to mechanistic study + non-IN backbone + OOD framing

Week 10-16: Plan A or Plan B execution
```

### 10.2 Plan A: Method Works (Phase III rejects equivalence)

```
- Complete ablation matrix (B1-B12, §7)
- Run on LLD-MMRI and WORC with 5-fold cross-validation
- Run shuffled-γ test (B12) as final sanity check
- Generate interpretability artifacts (§11)
- Write paper as "input-conditional learnable Nyúl spline normalization"
```

### 10.3 Plan B: Method Collapses (Phase III confirms equivalence)

Pivot the framing:

```
New contribution: Mechanistic study of preprocessing absorption under InstanceNorm.

Three new experiments required:
    1. Replicate collapse on 2+ datasets with full diagnostic evidence
    2. Replace InstanceNorm with GroupNorm in first block; rerun
       → if spline now retains nonlinearity, confirms IN as the cause
    3. Add OOD evaluation: train on one site, evaluate on others
       → if affine-collapsed method still beats fixed Nyúl on OOD, frame
         as "end-to-end coupling beats fixed standardization for OOD"

Target venues for Plan B: MIDL, MELBA, IEEE TMI (mechanistic study + OOD)
Time required: ~6-8 additional weeks
```

### 10.4 Critical Decision Points

```
After Week 6 (main method training complete):
    - Look at Phase I metrics
    - If r_i median > 0.15 AND η_i median > 0.5 → strong signal of survival
                                                  → skip Phase II/III, proceed Plan A
    - Else → proceed to Phase II

After Week 7 (Phase II):
    - If Test A and Test B BOTH indicate collapse → proceed Phase III
    - Else → proceed Plan A with caveats documented

After Week 8 (Phase III):
    - TOST equivalence accepted → Plan B
    - TOST equivalence rejected → Plan A
    - TOST inconclusive → Plan A with explicit acknowledgment of ambiguity
```

---

## 11. Interpretability Artifacts

These are required deliverables in the final paper, not optional supplementary material.

### F1: Pre/Post Intensity Histograms

Per modality × dataset, overlay 4 histograms across all test volumes (low-alpha per volume):

1. Raw `X_k`
2. `X_k` after baseline z-score
3. `X_k` after fixed Nyúl-11
4. `X̃_k` after main method (ours)

Color overlay by tumor grade (BraTS post-treatment) or histology label (WORC).

**Quantitative claim**: report Jensen-Shannon divergence across volumes at each stage; tighter inter-volume distribution → better normalization.

### F2: Learned Curves Across Test Set

Single figure: all `f_{θ_i}` curves on common `(input, output)` axes, low-alpha per volume.

Color by tumor grade or anatomical region.

**Quantitative claim**: report distribution of `R²(f_{θ_i}, population_Nyúl)` across volumes. Tightly clustered around population Nyúl → minimal per-image adaptation. Wide spread → strong per-image adaptation.

### F3: Regression Onto Fixed Maps

For each test volume:

```
R²_Nyúl_i = R² of f_{θ_i} as function of population Nyúl mapping
R²_oracle_i = R² of f_{θ_i} as function of per-image oracle z-score
R²_affine_i = R² of f_{θ_i} as function of best-fit affine
```

Three scatter plots; story is what the joint distribution looks like:

- High `R²_affine`, low others → collapse to affine (Plan B trigger)
- High `R²_Nyúl`, low `R²_oracle` → just learned population Nyúl
- Intermediate all three → learned task-adaptive mixture (best case)

### F4: Spearman Correlations Heatmap

Rows: learned spline parameters (knot x positions, knot y positions, internal derivatives)
Columns: image properties (mean foreground intensity, p99, SNR estimate, tumor volume, modality)

**Expected**: at least one strong correlation (|ρ| > 0.5). If all ρ ≈ 0, the hypernetwork is not using the conditioning input meaningfully.

### F5: Counterfactual Response

For each perturbation, plot the parameter shift vs. perturbation magnitude with theoretical expectation overlaid:

| Perturbation | Theoretical Expectation |
|--------------|------------------------|
| Gamma (`X → X^γ`) | Spline should curve away from identity in opposite direction |
| Bias field (multiplicative) | Spline should NOT respond (it's voxel-wise; cannot fix spatial bias) — honest failure mode |
| Contrast inversion (`1 - X`) | Spline should refuse (monotonicity-enforced) — sanity check |
| Additive noise | Spline should be approximately unchanged at the global percentile structure |

Report Spearman correlation between perturbation magnitude and parameter shift for gamma and noise. For gamma, ρ > 0.8 is strong evidence the hypernet is doing physically meaningful adaptation.

---

## 12. Code Generation Guidance

This section is **specifically for the local Claude instance** that will generate the implementation code from this blueprint.

### 12.1 Top-Level Guidance

1. **Implement modules in the order listed in §4** (Module 1 → Module 6). Each later module depends on earlier ones.

2. **Match the interfaces in §3.2 exactly.** Do not invent new arguments, do not rename methods. The module interfaces are the contract; if they don't fit your implementation, refactor your implementation, not the interface.

3. **All percentile/statistics computations must be `.detach()`**. Gradients should flow only through the hypernetwork and spline parameters, never through the data-derived summary statistics.

4. **All diagnostic metrics must be computed and logged as part of the standard training loop**, every 5–10 epochs. Do not bolt them on as a separate analysis script. Use a `DiagnosticLogger` class wrapping the validation loop.

5. **Use `torch.bucketize` for the spline bin lookup**, not manual indexing. It is differentiable through `gather` operations and well-vectorized.

6. **Test every module in isolation before integrating.** Unit tests:
   - `RQSplineApply`: verify monotonicity (output is non-decreasing in input) on random `θ`
   - `RQSplineApply`: verify identity at `θ` corresponding to identity spline
   - `Hypernetwork` with zero output init: verify `MLP(γ) = 0` at step 0
   - `FunctionSpaceAnchorLoss`: verify loss is zero when `θ = θ^(0)`

### 12.2 What NOT to Do

- **Do NOT use cubic Hermite splines.** Durkan 2019 explicitly rejects them due to numerical instability. Use rational-quadratic only.
- **Do NOT use parameter-space anchor.** Always function-space anchor on a fixed grid (§4.5).
- **Do NOT use identity tails.** Use clipped linear tails (§2.5).
- **Do NOT skip the affine-hypernet baseline.** It is the kill-switch (P2).
- **Do NOT use universal thresholds like 0.005 Dice.** Always compute dataset-specific `δ_min` from `σ_seed`.
- **Do NOT change segmentation loss.** Stay with Dice+CE.
- **Do NOT add auxiliary losses to the U-Net.** All auxiliary losses (anchor, KL) are on the spline only.

### 12.3 Recommended Repository Structure

```
mri_edain_v2/
├── modules/
│   ├── foreground.py             # ForegroundExtractor
│   ├── percentile.py              # PercentileSummary
│   ├── standardizer.py            # CoordinateStandardizer
│   ├── hypernetwork.py            # Hypernetwork
│   ├── rq_spline.py               # RQSplineParameterizer + RQSplineApply
│   ├── nyul_init.py               # PopulationNyulInitializer
│   └── edain_layer.py             # MRIEDAINLayer (top-level)
├── losses/
│   ├── anchor.py                  # FunctionSpaceAnchorLoss
│   ├── kl.py                      # KLAnchorLoss
│   └── combined.py                # CombinedLoss = L_seg + λ_anc·L_anc + λ_KL·L_KL
├── baselines/
│   ├── no_norm.py
│   ├── zscore.py
│   ├── percentile_clip.py
│   ├── nyul_fixed.py
│   ├── rq_spline_fixed.py         # Baseline #5
│   ├── affine_hypernet.py         # Baseline #6 (kill-switch)
│   └── whitestripe.py
├── diagnostics/
│   ├── metrics.py                 # r_i, CV(f'), η_i, erank, κ_i
│   ├── phase_one.py               # online monitoring
│   ├── phase_two.py               # affine-perturb + best-affine replacement
│   ├── phase_three.py             # retrain affine-hypernet
│   └── decision_logic.py          # multi-evidence concurrence
├── training/
│   ├── trainer.py                 # standard training loop
│   ├── scheduler.py               # 3-phase λ_anc, λ_KL schedule
│   └── ema.py                     # EMA of hypernet weights
├── evaluation/
│   ├── tost.py                    # equivalence testing
│   ├── bootstrap.py               # BCa CIs
│   ├── statistical.py             # paired Wilcoxon + BH-FDR
│   └── interpretability.py        # F1-F5 artifacts
├── configs/
│   ├── brats.yaml
│   ├── lld_mmri.yaml
│   ├── worc.yaml
│   └── ablations/                 # B1-B12 configs
└── tests/
    └── test_modules.py
```

### 12.4 Configuration File Template (YAML)

```yaml
# configs/brats.yaml

dataset:
  name: brats_adult_2024
  modalities: ["t1n"]  # for v2; extend to all 4 for full BraTS later
  fold: 0
  total_folds: 5

method:
  type: "mri_edain_v2"  # or one of: "z_score", "nyul_fixed", "affine_hypernet", etc.
  
  spline:
    K: 9
    B: 4.0
    alpha_tail: 0.5
    min_derivative: 1.0e-3
  
  hypernet:
    hidden_dim: 64
    n_layers: 2
    activation: "gelu"
    output_init: "zero"  # or "small_random"

  conditioning:
    percentile_set: [0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
    coordinate_standardize: true
  
  losses:
    anchor:
      enabled: true
      type: "function_space"  # always; do not change
      grid_size: 50
      lambda_init: 1.0e-2
      lambda_final: 1.0e-4
      schedule: "cosine"
    kl:
      enabled: true
      lambda_init: 0.0
      lambda_final: 1.0e-4
      ramp_steps: 11250  # 5% of Phase 2

training:
  total_steps: 250000
  phase_0_steps: 2500
  phase_1_steps: 25000
  optimizer:
    backbone:
      type: "sgd_nesterov"
      lr: 1.0e-2
      momentum: 0.99
      weight_decay: 3.0e-5
      schedule: "polylr"
      poly_exponent: 0.9
    hypernet:
      lr: 1.0e-2  # same as backbone; reduce if unstable
      grad_clip: 1.0
  ema:
    enabled: true
    momentum: 0.99

diagnostic:
  phase_one:
    enabled: true
    check_every_epochs: 10
    warmup_until_step: 7500
    metrics: ["r_i", "cv_fprime", "eta_i", "erank", "kappa_i"]
    persistence_required: 3  # consecutive checkpoints

  phase_two:
    enabled: true
    affine_perturbation:
      n_perturbations: 20
      perturbation_std: 0.05
    best_affine_replacement:
      tost_alpha: 0.05

  phase_three:
    enabled: true
    affine_hypernet_seeds: 3

  decision:
    delta_min_strategy: "calibrated"  # uses σ_seed
    delta_min_minimum: 0.005
    delta_min_scale: 0.25
    require_multi_evidence: true

seeds: [42, 43, 44]

logging:
  every_n_steps: 100
  validate_every_epochs: 5
  artifacts_path: "./outputs/brats_fold0"
```

### 12.5 Critical Implementation Checks

Before declaring the implementation complete:

```
[ ] RQSplineApply passes monotonicity test on 1000 random θ samples
[ ] MRIEDAINLayer at step 0 produces output equal to applying f_{θ^(0)}
[ ] FunctionSpaceAnchorLoss is zero at step 0 (because θ = θ^(0))
[ ] Phase I metrics computed and logged in standard training loop
[ ] Affine-hypernet baseline (#6) has identical schedule to main method
[ ] CoordinateStandardizer fit BEFORE training, frozen during training
[ ] Population Nyúl θ^(0) computed once, registered as buffer, never updated
[ ] EMA wrapper around hypernet works correctly
[ ] All baselines #1-#7 implemented and runnable
[ ] Unit tests pass for all 6 core modules in §4
[ ] Training runs end-to-end on BraTS for at least 1000 steps without crashes
[ ] Diagnostic metrics produce sensible numbers at step 0 (η ≈ 1 mostly,
    since spline is population Nyúl which is non-affine)
```

---

## 13. Appendix A: Error and Correction History

This appendix documents the major errors made in earlier versions of this analysis and how they were corrected. It serves both as a record of intellectual evolution and as a guard against repeating the same mistakes.

### 13.1 Error 1: Original MRI-EDAIN's Mathematical Redundancy

**Origin**: v0 (the MICCAI submission) and Blueprint v1.

**Error**: The method introduced learnable scalars `(m, s)` to modulate per-volume z-score, but the per-volume statistics (μ_volume, σ_volume) were already computed in preprocessing and held fixed. This made `(m, s)` essentially a **global scalar pair** applied to already-standardized inputs.

**Why it failed**: Per the IN absorption theorem (§2.1), any global per-channel affine variation is exactly absorbed by the first Conv ∘ IN in the backbone. The two learnable scalars contributed nothing to network capacity.

**Correction**: v2 replaces the affine with input-conditional nonlinear monotone spline (§2.2 and onwards). The conditioning is per-volume, not global. The transform has nonlinear capacity that, **if preserved during training**, escapes IN absorption.

### 13.2 Error 2: First Naive Successor (U-Net + FC → (m, s))

**Origin**: The user's first attempted improvement before reaching v2.

**Error**: A small U-Net consumed the raw 3D volume; a global pool + FC head output `(m, s)`. The result was extremely unstable training.

**Why it failed (six diagnosed pathologies)**:
1. Gradient bottleneck: compressing 10⁷ voxels to 2 scalars → high-variance gradient
2. Scale-symmetry degeneracy: equivalent solutions form a 2D ridge
3. No identity prior: `(m, s)` started far from `(1, 0)`, breaking He-init assumptions
4. Hypernet magnitude proportionality (per MIP, Gonzalez Ortiz ICLR 2024)
5. Double moving-target dynamics (per Batch Renorm rationale, Ioffe 2017)
6. Underdetermined regression: many `(m, s)` yield the same Dice

**Correction**: v2 conditions on a compact 11D standardized summary (not raw image), uses zero-init residual hypernetwork output (anchors at population Nyúl), and outputs spline parameters (not scalars) so per-voxel curvature carries information.

### 13.3 Error 3: Blueprint v1 Overstated "Non-Absorbability"

**Origin**: Blueprint v1.

**Error**: v1 implied that "spline is monotone-nonlinear, therefore it cannot be absorbed by IN." This conflates **architectural capacity** with **trained behavior**.

**Why it was wrong**: The RQ-spline architecture has the **capacity** to be nonlinear, but training under Dice+CE with IN downstream provides no gradient pressure to maintain nonlinearity. The spline can collapse to affine even though it has the parameterization to be richer.

**Correction**: v2 explicitly distinguishes "architectural capacity" (always present) from "empirically realized nonlinearity" (must be diagnosed). The Phase I/II/III diagnostic protocol is built to measure the latter rather than assume it.

### 13.4 Error 4: v1 Used Universal 0.005 Dice Threshold

**Origin**: Blueprint v1 (inherited from the long collapse-analysis research report).

**Error**: Claimed that `ΔDice < 0.005` corresponds to the nnU-Net seed-to-seed noise floor and used this as a universal threshold for equivalence/collapse.

**Why it was wrong**: nnU-Net's seed-to-seed variance is dataset-dependent. On WORC (n=115), `σ_seed` can exceed 0.01. Augmentation, 5-fold splits, and checkpoint selection compound the variance. Using 0.005 universally produces false equivalence claims on smaller datasets.

**Correction** (P5): v2 computes `δ_min = max(0.005, 0.25 × σ_seed)` per dataset. TOST equivalence testing is used in place of naive thresholding.

### 13.5 Error 5: v1 Used Parameter-Space Anchor Loss

**Origin**: Blueprint v1.

**Error**: Anchor loss was `||θ - θ^(0)||²` in parameter space.

**Why it was wrong**: The RQ-spline parameterization has redundancies: softmax over width logits is shift-invariant (`softmax(θ) = softmax(θ + c·𝟙)`), so parameter distance is not isomorphic to function distance. Large `||θ - θ^(0)||` can yield identical spline functions.

**Correction** (P3): v2 uses function-space anchor `(1/L) Σ_ℓ [f_θ(t_ℓ) - f_θ^(0)(t_ℓ)]²` on a fixed sampling grid.

### 13.6 Error 6: v1 Used Identity Tails for Spline

**Origin**: Blueprint v1 (inherited from Durkan 2019 normalizing flow convention).

**Error**: Outside `[-B, B]`, the spline reduced to `f(x) = x`.

**Why it was wrong**: Identity tails preserve outliers — exactly contrary to the goal of intensity normalization. Outliers (bias-field residuals, artifacts, lesion edge effects) should be **compressed**, not passed through.

**Correction**: v2 uses clipped linear tails with slope `α_tail = 0.5 < 1`, compressing outliers while remaining differentiable.

### 13.7 Error 7: v1 Treated KL Anchor as Universally Beneficial

**Origin**: Blueprint v1 (inherited from EDAIN-KL paper, which worked on financial time series).

**Error**: Used `λ_KL = 10⁻³` as a default with the assumption that pulling foreground distributions toward N(0, 1) is beneficial.

**Why it was wrong**: MRI tumor foregrounds are **multi-modal mixtures** (fat, muscle, necrosis, tumor heterogeneity, scanner artifacts). Forcing them toward unimodal Gaussian can compress biologically meaningful tissue heterogeneity that is precisely what the segmentation network needs.

**Correction** (P4): v2 reduces `λ_KL` default to `10⁻⁴` (weak regularizer), requires `λ_KL = 0` ablation, and adds tumor contrast preservation `κ_i` as a monitored metric.

### 13.8 Error 8: v1 Used "Identity-at-Init" Terminology

**Origin**: Blueprint v1.

**Error**: Called the initialization "identity-at-init" when in fact the spline at init equals **population Nyúl mapping**, not identity `f(x) = x`.

**Why it was wrong**: Confusing terminology. Population Nyúl ≠ identity (unless population Nyúl happens to be identity, which it typically is not).

**Correction**: v2 renames consistently to "population-Nyúl anchor" or "population-Nyúl-at-init."

### 13.9 Error 9: v1 Missing Affine-Hypernet Baseline

**Origin**: Blueprint v1.

**Error**: Compared the spline method only against fixed normalization baselines (z-score, Nyúl, WhiteStripe). No baseline that uses **identical conditioning and hypernet but outputs only affine** existed.

**Why it was wrong**: Without this baseline, one cannot distinguish the contribution of "input-conditional learning" from "nonlinear capacity." A reviewer can correctly argue: "Maybe a simple affine-hypernet would achieve the same gains."

**Correction** (P2): v2 adds baseline #6 (AffineHypernetLayer) with identical conditioning and architecture, only outputting `(a, c)` instead of spline parameters.

### 13.10 Error 10: v1 Over-Interpreted Theoretical Literature

**Origin**: A long research report integrated into Blueprint v1.

**Error**: Cited implicit-bias theorems (Soudry 2018, Lyu & Li 2020), flat-minimum arguments (Keskar 2017, Dinh 2017), and Reinhold 2019 as **theoretical proof** that the spline would collapse to affine.

**Why it was wrong**:
- Soudry / Lyu & Li theorems apply to specific settings (separable data, homogeneous networks, exponential loss). Our setting (Dice+CE, spline hypernetwork, IN backbone) does not satisfy these conditions rigorously.
- Keskar discusses large-batch sharp minima; Dinh shows sharpness is reparameterization-dependent. The conclusion "affine basins are exponentially larger" is the user's reasoning, not a theorem.
- Reinhold 2019 studied MR **synthesis**, not segmentation. Extrapolating to segmentation is unjustified.

**Correction**: v2 cites these as **directional intuitions** rather than proofs. The actual claim is empirical: "These literatures suggest collapse is plausible; we therefore designed diagnostics (§8) to measure it." Mathematical proof is only claimed for IN's exact invariance under per-image affine (§2.1), which is algebra.

### 13.11 Error 11: v1 Cited Unverified Sources

**Origin**: Earlier research integration.

**Error**: Included a reference to "MDPI 2023 multi-site comparison" without specific title, authors, or DOI.

**Why it was wrong**: Hallucination risk. Reviewers can verify; if the reference does not exist or does not say what is claimed, the paper loses credibility.

**Correction**: v2 deletes this reference entirely. All retained citations have verified titles, authors, venues, and identifiers.

### 13.12 Summary of Corrections

| # | Domain | Correction Mechanism |
|---|--------|---------------------|
| 1 | Mathematical foundation | Replace global scalars with input-conditional spline |
| 2 | Hypernet stability | Use compact percentile summary, not raw image |
| 3 | Theoretical claims | Distinguish capacity from realized behavior |
| 4 | Statistical thresholds | Per-dataset δ_min from σ_seed |
| 5 | Anchor loss form | Function-space, not parameter-space |
| 6 | Spline tail behavior | Clipped linear, not identity |
| 7 | KL regularization | Reduce λ_KL, require zero ablation |
| 8 | Terminology | "Population-Nyúl anchor" not "identity anchor" |
| 9 | Baseline ladder | Add affine-hypernet kill-switch baseline |
| 10 | Citation strength | Demote theory to directional intuition |
| 11 | Citation accuracy | Delete all unverified references |

---

## 14. Appendix B: Citation Discipline

### 14.1 Strong Citations (Use These Confidently)

**Mathematical/methodological foundation**:
- Durkan, Bekasov, Murray, Papamakarios. *Neural Spline Flows*. NeurIPS 2019. arXiv:1906.04032. → RQ-spline parameterization.
- Gonzalez Ortiz, Guttag, Dalca. *Magnitude Invariant Parametrizations Improve Hypernetwork Learning*. ICLR 2024. arXiv:2304.07645. → Per-coordinate standardization mechanism.
- September, Sanna Passino, Goldmann, Hinel. *Extended Deep Adaptive Input Normalization*. AISTATS 2024. arXiv:2310.14720. → EDAIN-KL anchor concept (with caveat for medical imaging mismatch).
- Zhang, Dauphin, Ma. *Fixup Initialization*. ICLR 2019. arXiv:1901.09321. → Zero-init output principle.

**MRI normalization**:
- Nyúl, Udupa. *On Standardizing the MR Image Intensity Scale*. MRM 1999. → Foundational Nyúl method.
- Nyúl, Udupa, Zhang. *New Variants of a Method of MRI Scale Standardization*. IEEE TMI 2000. → 11-landmark variant.
- Shah, Xiao, Subbanna, Francis, Arnold, Collins, Arbel. *Evaluating Intensity Normalization on MRIs of Human Brain with Multiple Sclerosis*. Med Image Anal 2011, 15(2):267-282. → Modern 11-landmark percentile set.
- Shinohara et al. *Statistical Normalization Techniques for Magnetic Resonance Imaging*. NeuroImage Clinical 2014. → WhiteStripe.

**Backbone**:
- Isensee, Jaeger, Kohl, Petersen, Maier-Hein. *nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation*. Nature Methods 2021, 18(2):203-211. → nnU-Net default normalization.
- Cardoso et al. *MONAI: An Open-Source Framework for Deep Learning in Healthcare*. arXiv:2211.02701, 2022.

**Statistical methodology**:
- Maier-Hein, Reinke et al. *Metrics Reloaded: Recommendations for Image Analysis Validation*. Nature Methods 21(2):195-212, 2024. → Statistical evaluation protocol.
- Kirby, Gerlanc. *BootES: An R Package for Bootstrap Confidence Intervals on Effect Sizes*. Behavior Research Methods 45(4):905-927, 2013. → Effect size CIs.
- Benjamini, Hochberg. *Controlling the False Discovery Rate*. JRSS-B 57(1):289-300, 1995. → BH-FDR.

**Counterfactual analysis**:
- Bansal, Nakkiran, Barak. *Revisiting Model Stitching to Compare Neural Representations*. NeurIPS 2021. arXiv:2106.07682. → Replacement test methodology.

### 14.2 Cite With Caveats

These citations are valid but must be confined to what they actually support:

- **Reinhold et al.** *Evaluating the Impact of Intensity Normalization on MR Image Synthesis*. SPIE 2019. arXiv:1812.04652.
  - ✓ Cite for: comparing multiple normalization methods in MR image synthesis context
  - ✗ Do NOT cite for: claim that Nyúl ≈ z-score in segmentation
  
- **Kondrateva et al.** arXiv:2204.05278.
  - ✓ Cite for: their specific finding that nnU-Net is robust to normalization choice on their brain tumor segmentation datasets via Cliff's-Δ
  - ✗ Do NOT generalize to all segmentation tasks
  
- **Guo et al.** *Zero-DCE*. CVPR 2020.
  - ✓ Cite as: example of where learnable nonlinear curves are preserved because loss directly constrains image statistics
  - ✗ Do NOT cite as: proof that nonlinear curves will be preserved under Dice+CE

### 14.3 Demoted Theory (Cite as Directional Intuition Only)

- Soudry et al. *Implicit Bias of Gradient Descent on Separable Data*. JMLR 2018. → Frame as: "Theoretical work suggests..."
- Lyu & Li. *Gradient Descent Maximizes the Margin of Homogeneous Neural Networks*. ICLR 2020. → Same framing.
- Keskar et al. *On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima*. ICLR 2017. → Frame as: "Empirical observations on flat minima suggest..."
- Dinh et al. *Sharp Minima Can Generalize for Deep Nets*. ICML 2017. → Note that this **counters** simplistic flat-minimum-good narratives.

### 14.4 Deleted Citations

- "MDPI 2023 multi-site comparison" — hallucinated reference, no verifiable identifier.
- Any other reference lacking specific title + authors + venue + year.

### 14.5 Citation Discipline Rules (P-level)

**Rule 1**: Every specific claim has a specific citation. No vague "the literature suggests" without a paper.

**Rule 2**: Citations from one domain (e.g., synthesis) cannot be used to support claims in another domain (e.g., segmentation) without an explicit bridging argument.

**Rule 3**: Mathematical theorems cited must have prerequisites that hold in our setting. If not, cite as intuition only.

**Rule 4**: Engineering claims labeled `[ENG]` in the document. Literature-backed claims labeled `[LIT]`. Standard-practice claims labeled `[STD]`. Speculative claims labeled `[SPEC]`.

---

## 15. Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **Population Nyúl mapping** `f_{θ^(0)}` | The piecewise-linear mapping from input intensity percentiles to standard scale, computed once on the training set. |
| **Population-Nyúl anchor** | The initial state of the trained spline; equal to the population Nyúl mapping (NOT identity `f(x) = x`). |
| **Affine-hypernet baseline** | The kill-switch baseline that uses identical conditioning input and hypernetwork architecture but outputs only affine `(a, c)`. |
| **Non-affineness ratio** `r_i` | sqrt of (1 - R²) of the trained spline against its best linear fit. Near 0 → spline is approximately affine. |
| **Post-IN survival ratio** `η_i` | Ratio of feature-map difference (with vs. without spline) after IN to the same difference before IN. Near 0 → IN absorbed the spline change. |
| **Effective rank** | `exp(H(normalized singular values))` where H is Shannon entropy. Measures the "active dimensionality" of a matrix. |
| **Tumor contrast preservation** `κ_i` | Ratio of (tumor-to-normal contrast after spline) to (tumor-to-normal contrast on raw input). Near 1 → preserved. |
| **TOST** | Two One-Sided Tests for equivalence; rejects "different" hypothesis when both one-sided tests at margin δ are rejected. |
| **`δ_min`** | Dataset-specific minimum effect size of interest, computed as `max(0.005, 0.25 × σ_seed)`. |
| **`σ_seed`** | Standard deviation of Dice across ≥3 seeds for the baseline configuration. Estimated empirically per dataset. |
| **Function-space anchor** | Anchor loss computed as `(1/L) Σ_ℓ [f_θ(t_ℓ) - f_θ^(0)(t_ℓ)]²` on a fixed grid. |
| **Parameter-space anchor** | Anchor loss `||θ - θ^(0)||²`. **Rejected** in v2 due to softmax shift-invariance. |
| **Phase I / II / III** | The three diagnostic stages: online monitoring, one-shot confirmation, decisive retrain test. |
| **Plan A** | Continuation plan when diagnostics indicate the method works. Standard nonlinear spline framing. |
| **Plan B** | Pivot plan when diagnostics confirm collapse. Mechanistic study + non-IN backbone + OOD framing. |

---

## End of Blueprint

**Document Version**: 2.0 (post-critique, post-collapse-analysis, post-threshold-correction)

**Companion Documents**:
- Blueprint v1 (`MRI-EDAIN-Blueprint.md`) — superseded but retained for context
- Blueprint v2 (`MRI-EDAIN-Blueprint-v2.md`) — superseded by this document

**Intended Use**: Specification for local Claude code generation. Implement modules in order. Test in isolation. Integrate. Run the diagnostic protocol. Decide between Plan A and Plan B on multi-evidence concurrence.

**Six Principles Recap** (the document's backbone):

1. Non-absorption verification is mandatory, not assumed.
2. Affine-hypernetwork baseline is the kill-switch.
3. Function-space anchor, not parameter-space.
4. KL-to-N(0,1) is a weak regularizer with mandatory zero ablation.
5. Thresholds are dataset-specific, not universal.
6. Stop only on multi-evidence concurrence.
