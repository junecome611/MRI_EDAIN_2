"""MRIEDAINLayer: top-level layer wiring together percentile + standardizer +
hypernetwork + spline (blueprint section 3.1, 3.2).

One instance per modality. Holds:
    - theta_0 (frozen buffer, population Nyul anchor)
    - standardizer (frozen per-coordinate statistics for the conditioning vector)
    - hypernetwork (only learnable component besides spline params themselves)

The forward pass:
    1. compute gamma_raw on foreground (detached);
    2. apply per-coordinate standardisation (detached);
    3. hypernetwork -> delta theta;
    4. theta = theta_0 + delta;
    5. parameterise RQ-spline -> apply voxel-wise on foreground;
    6. return X_tilde + a diagnostics dict.

Background voxels are passed through unchanged (blueprint section 2.6).

Input convention:
    X    : (D, H, W) | (B, D, H, W) | (B, 1, D, H, W)
    mask : same spatial shape (boolean foreground mask)

For batched inputs each sample receives its own gamma_raw and its own spline.
Patches vs whole-volume: the layer just consumes whatever X is given. If the
trainer wants whole-volume gamma applied to patches, it should compute gamma
outside and pass via the `gamma_raw` kwarg.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mri_edain_v2.modules.hypernetwork import Hypernetwork
from mri_edain_v2.modules.percentile import PERCENTILES, percentile_summary
from mri_edain_v2.modules.rq_spline import (
    SplineParams,
    rq_spline_apply,
    rq_spline_parameterize,
)
from mri_edain_v2.modules.standardizer import CoordinateStandardizer


class MRIEDAINLayer(nn.Module):
    """Input-conditional RQ-spline normalization layer (single modality)."""

    def __init__(
        self,
        standardizer: CoordinateStandardizer,
        theta_0: torch.Tensor,
        K: int = 9,
        B_supp: float = 4.0,
        alpha_tail: float = 0.5,
        min_derivative: float = 1e-3,
        hypernet_hidden_dim: int = 64,
        hypernet_zero_init: bool = True,
        percentiles: tuple = PERCENTILES,
    ):
        super().__init__()
        if theta_0.ndim != 1 or theta_0.shape[0] != 3 * K - 1:
            raise ValueError(
                f"theta_0 must have shape ({3*K-1},) for K={K}, "
                f"got {tuple(theta_0.shape)}"
            )

        self.K = int(K)
        self.B_supp = float(B_supp)
        self.alpha_tail = float(alpha_tail)
        self.min_derivative = float(min_derivative)
        self.percentiles = tuple(percentiles)
        self.input_dim = len(self.percentiles)

        self.standardizer = standardizer  # nn.Module with frozen buffers
        self.hypernet = Hypernetwork(
            input_dim=self.input_dim,
            hidden_dim=hypernet_hidden_dim,
            output_dim=3 * self.K - 1,
            zero_init_output=hypernet_zero_init,
        )

        # FROZEN population Nyul anchor (blueprint section 2.4, 2.8).
        self.register_buffer("theta_0", theta_0.detach().clone().to(torch.float32))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_gammas(
        self, X_BDHW: torch.Tensor, mask_BDHW: torch.Tensor
    ) -> torch.Tensor:
        """Compute gamma_raw per-batch on foreground. Returns (B, 11), detached."""
        B = X_BDHW.shape[0]
        out = []
        for b in range(B):
            g = percentile_summary(
                X_BDHW[b], mask_BDHW[b], percentiles=self.percentiles
            )
            out.append(g)
        return torch.stack(out, dim=0)  # (B, 11), already detached

    def _theta_from_gamma(self, gamma_std: torch.Tensor) -> torch.Tensor:
        """Add hypernet residual onto theta_0 anchor."""
        delta = self.hypernet(gamma_std)
        # theta_0 is a 1-D buffer; broadcast across the batch.
        return self.theta_0 + delta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        gamma_raw: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply the learnable normalization.

        Args:
            X: input volume(s). Supported shapes:
                (D, H, W), (B, D, H, W), (B, 1, D, H, W).
            mask: foreground mask, same spatial layout as X. The channel dim
                (1) may be present or absent.
            gamma_raw: optional precomputed gamma. Shape (11,) for a single
                volume or (B, 11) for a batch. If None, computed inside.
            return_diagnostics: if False, returns ({}) for the second tuple
                element (slightly faster, used during sliding-window inference).

        Returns:
            (X_tilde, diagnostics)
                X_tilde has the same shape as X.
                diagnostics is a dict with detached tensors used by the
                trainer / DiagnosticLogger (blueprint section 8).
        """
        orig_ndim = X.ndim
        had_channel_dim = False

        # Normalize layout to (B, D, H, W).
        if orig_ndim == 3:
            X_b = X.unsqueeze(0)
            mask_b = mask.unsqueeze(0)
        elif orig_ndim == 4:
            X_b = X
            mask_b = mask
        elif orig_ndim == 5:
            if X.shape[1] != 1:
                raise ValueError(
                    "MRIEDAINLayer is single-modality; expected channel dim = 1, "
                    f"got {X.shape[1]}. Use one layer per modality."
                )
            X_b = X[:, 0]
            mask_b = mask[:, 0] if mask.ndim == 5 else mask
            had_channel_dim = True
        else:
            raise ValueError(f"Unsupported X.ndim = {orig_ndim}")

        if mask_b.shape != X_b.shape:
            raise ValueError(
                f"mask spatial shape {tuple(mask_b.shape)} does not match "
                f"X spatial shape {tuple(X_b.shape)}"
            )

        # 1-2. gamma_raw + per-coordinate standardisation.
        if gamma_raw is None:
            gammas_raw = self._compute_gammas(X_b, mask_b)  # (B, 11), detached
        else:
            if gamma_raw.ndim == 1:
                gammas_raw = gamma_raw.unsqueeze(0)
            else:
                gammas_raw = gamma_raw
            if gammas_raw.shape != (X_b.shape[0], self.input_dim):
                raise ValueError(
                    f"gamma_raw shape mismatch: expected "
                    f"({X_b.shape[0]}, {self.input_dim}), "
                    f"got {tuple(gammas_raw.shape)}"
                )
            gammas_raw = gammas_raw.detach()

        gammas_std = self.standardizer(gammas_raw)

        # 3-4. delta_theta + theta_0 anchor.
        theta = self._theta_from_gamma(gammas_std)  # (B, 3K-1), differentiable

        # 5. Parameterise and apply spline.
        params = rq_spline_parameterize(
            theta,
            K=self.K,
            B_supp=self.B_supp,
            alpha_tail=self.alpha_tail,
            min_derivative=self.min_derivative,
        )
        X_tilde_fg = rq_spline_apply(X_b, params)  # (B, D, H, W)

        # 6. Pass-through background voxels (blueprint section 2.6).
        mask_bool = mask_b.to(torch.bool)
        X_tilde = torch.where(mask_bool, X_tilde_fg, X_b)

        # Restore original layout.
        if orig_ndim == 3:
            X_tilde_out = X_tilde.squeeze(0)
        elif orig_ndim == 5 and had_channel_dim:
            X_tilde_out = X_tilde.unsqueeze(1)
        else:
            X_tilde_out = X_tilde

        diag: Dict[str, torch.Tensor] = {}
        if return_diagnostics:
            with torch.no_grad():
                diag = {
                    "gamma_raw": gammas_raw.detach(),
                    "gamma_std": gammas_std.detach(),
                    "theta": theta.detach(),
                }
        return X_tilde_out, diag

    # ------------------------------------------------------------------
    # Convenience accessors (used by trainer / diagnostics)
    # ------------------------------------------------------------------

    def current_spline_params(
        self, gamma_raw: torch.Tensor
    ) -> SplineParams:
        """Public helper returning SplineParams for inspection / diagnostics."""
        if gamma_raw.ndim == 1:
            gamma_raw = gamma_raw.unsqueeze(0)
        gamma_std = self.standardizer(gamma_raw.detach())
        theta = self._theta_from_gamma(gamma_std)
        return rq_spline_parameterize(
            theta,
            K=self.K,
            B_supp=self.B_supp,
            alpha_tail=self.alpha_tail,
            min_derivative=self.min_derivative,
        )

    def anchor_spline_params(self) -> SplineParams:
        """SplineParams for the frozen theta_0 anchor (population Nyul)."""
        return rq_spline_parameterize(
            self.theta_0,
            K=self.K,
            B_supp=self.B_supp,
            alpha_tail=self.alpha_tail,
            min_derivative=self.min_derivative,
        )

    def extra_repr(self) -> str:
        return (
            f"K={self.K}, B_supp={self.B_supp}, "
            f"alpha_tail={self.alpha_tail}, "
            f"input_dim={self.input_dim}"
        )
