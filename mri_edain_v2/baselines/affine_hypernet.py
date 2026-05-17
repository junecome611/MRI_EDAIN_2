"""Affine-Hypernet baseline -- THE KILL-SWITCH (blueprint section 4.6, P2).

Identical conditioning input + identical hypernet architecture as the main
method, but outputs only (a, c) and applies X_tilde = a * X + c.

Trained with the SAME optimizer, schedule, augmentations, and per-coordinate
standardizer as the main method. The point: if MRI-EDAIN v2 does not exceed
this baseline by > delta_min (TOST equivalence test), the nonlinear capacity
of the spline brings no measurable training advantage -- the framing dies.

Initialization: zero-init hypernet output. So at step 0:
    a = a_0 = 1.0 (since exp(0) = 1)
    c = c_0 = 0.0
which gives identity affine. (Note: this differs slightly from the blueprint
pseudocode where a_0 / c_0 are passed in from population stats; in our setting
the inputs are already per-volume z-scored upstream, so an identity affine at
init is the meaningful anchor. Population stats would just push toward
identity by construction.)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from mri_edain_v2.modules.hypernetwork import Hypernetwork
from mri_edain_v2.modules.percentile import PERCENTILES, percentile_summary
from mri_edain_v2.modules.standardizer import CoordinateStandardizer


class AffineHypernetLayer(nn.Module):
    """Per-image affine X_tilde = a * X + c with hypernet-predicted (a, c).

    `a` is constrained positive via a soft tanh:
        a = a_0 * exp(0.5 * tanh(delta_a))
    so at delta_a = 0 we get a = a_0, and the magnitude of `a` is bounded
    in [a_0 * exp(-0.5), a_0 * exp(0.5)] for stability.

    Args:
        standardizer: a fit CoordinateStandardizer instance (shared with the
            main method to ensure identical conditioning preprocessing).
        K_unused, B_supp_unused: accepted but ignored, kept in the signature
            so the trainer can swap baselines transparently.
        hypernet_hidden_dim: same as main method default.
        hypernet_zero_init: zero-init final layer (matches identity anchor).
        a_0, c_0: anchor values for (a, c); defaults give identity (a=1, c=0).
        percentiles: percentile slots for the conditioning vector.
    """

    def __init__(
        self,
        standardizer: CoordinateStandardizer,
        hypernet_hidden_dim: int = 64,
        hypernet_zero_init: bool = True,
        a_0: float = 1.0,
        c_0: float = 0.0,
        percentiles: tuple = PERCENTILES,
    ):
        super().__init__()
        self.percentiles = tuple(percentiles)
        self.input_dim = len(self.percentiles)

        self.standardizer = standardizer
        self.hypernet = Hypernetwork(
            input_dim=self.input_dim,
            hidden_dim=hypernet_hidden_dim,
            output_dim=2,  # delta_a, delta_c
            zero_init_output=hypernet_zero_init,
        )

        self.register_buffer("a_0", torch.tensor(float(a_0)))
        self.register_buffer("c_0", torch.tensor(float(c_0)))

    def _compute_gammas(
        self, X_BDHW: torch.Tensor, mask_BDHW: torch.Tensor
    ) -> torch.Tensor:
        B = X_BDHW.shape[0]
        out = []
        for b in range(B):
            g = percentile_summary(
                X_BDHW[b], mask_BDHW[b], percentiles=self.percentiles
            )
            out.append(g)
        return torch.stack(out, dim=0)

    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        gamma_raw: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        orig_ndim = X.ndim
        had_channel_dim = False

        if orig_ndim == 3:
            X_b = X.unsqueeze(0)
            mask_b = mask.unsqueeze(0)
        elif orig_ndim == 4:
            X_b = X
            mask_b = mask
        elif orig_ndim == 5:
            if X.shape[1] != 1:
                raise ValueError(
                    "AffineHypernetLayer is single-modality; expected channel "
                    f"dim = 1, got {X.shape[1]}."
                )
            X_b = X[:, 0]
            mask_b = mask[:, 0] if mask.ndim == 5 else mask
            had_channel_dim = True
        else:
            raise ValueError(f"Unsupported X.ndim = {orig_ndim}")

        if gamma_raw is None:
            gammas_raw = self._compute_gammas(X_b, mask_b)
        else:
            gammas_raw = (
                gamma_raw.unsqueeze(0) if gamma_raw.ndim == 1 else gamma_raw
            ).detach()

        gammas_std = self.standardizer(gammas_raw)
        deltas = self.hypernet(gammas_std)  # (B, 2)
        delta_a = deltas[:, 0]
        delta_c = deltas[:, 1]

        # Constrain a > 0 via bounded soft exp.
        a = self.a_0 * torch.exp(0.5 * torch.tanh(delta_a))   # (B,)
        c = self.c_0 + delta_c                                # (B,)

        # Broadcast over spatial dims.
        a_b = a.view(-1, 1, 1, 1)
        c_b = c.view(-1, 1, 1, 1)

        X_tilde_fg = a_b * X_b + c_b
        mask_bool = mask_b.to(torch.bool)
        X_tilde = torch.where(mask_bool, X_tilde_fg, X_b)

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
                    "a": a.detach(),
                    "c": c.detach(),
                }
        return X_tilde_out, diag

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, "
            f"a_0={float(self.a_0.item()):.4f}, c_0={float(self.c_0.item()):.4f}"
        )
