"""Function-space anchor loss (blueprint section 2.7, 4.5, principle P3).

L_anc = (1 / L) * sum_l [f_theta(t_l) - f_{theta_0}(t_l)]^2

The anchor regularises the function f_theta toward the population Nyul function
f_{theta_0}, evaluated on a fixed grid {t_l}. Parameter-space anchors
(||theta - theta_0||^2) are REJECTED because the RQ-spline parameterisation has
softmax shift-invariance symmetries: large parameter distances can correspond
to identical functions (blueprint error 13.5).
"""

from __future__ import annotations

from typing import Union

import torch
import torch.nn as nn

from mri_edain_v2.modules.rq_spline import (
    SplineParams,
    rq_spline_apply,
)


def function_space_anchor_loss(
    current_params: SplineParams,
    anchor_params: SplineParams,
    t_grid: torch.Tensor,
) -> torch.Tensor:
    """Compute L_anc on a precomputed grid.

    Args:
        current_params: SplineParams from the trained spline. May be batched
            (B, K+1) or unbatched (K+1,).
        anchor_params: SplineParams from theta_0 (unbatched, (K+1,)).
        t_grid: 1-D tensor of grid points in [-B_supp, +B_supp], shape (L,).

    Returns:
        Scalar tensor (mean over grid and batch).
    """
    if t_grid.ndim != 1:
        raise ValueError(f"t_grid must be 1-D, got {t_grid.ndim}-D")

    if current_params.is_batched:
        B = current_params.batch_size
        t_grid_b = t_grid.unsqueeze(0).expand(B, -1).contiguous()
        f_current = rq_spline_apply(t_grid_b, current_params)  # (B, L)
    else:
        f_current = rq_spline_apply(t_grid, current_params)  # (L,)

    f_anchor = rq_spline_apply(t_grid, anchor_params)  # (L,)

    diff = f_current - f_anchor
    return diff.pow(2).mean()


class FunctionSpaceAnchorLoss(nn.Module):
    """Module wrapper holding the fixed evaluation grid as a buffer.

    Args:
        grid_size: number of grid points (default 50; blueprint section 4.5).
        B_supp: spline support magnitude.
        grid_type: 'uniform' (default) or 'percentile_weighted' (ablation B10,
            not implemented yet -- raises if requested).
    """

    def __init__(
        self,
        grid_size: int = 50,
        B_supp: float = 4.0,
        grid_type: str = "uniform",
    ):
        super().__init__()
        if grid_type != "uniform":
            raise NotImplementedError(
                f"grid_type='{grid_type}' not implemented; only 'uniform' is "
                f"supported in this version. Ablation B10 is future work."
            )
        self.grid_size = int(grid_size)
        self.B_supp = float(B_supp)
        self.grid_type = grid_type
        self.register_buffer(
            "t_grid",
            torch.linspace(-self.B_supp, self.B_supp, self.grid_size),
        )

    def forward(
        self,
        current_params: SplineParams,
        anchor_params: SplineParams,
    ) -> torch.Tensor:
        return function_space_anchor_loss(current_params, anchor_params, self.t_grid)

    def extra_repr(self) -> str:
        return f"grid_size={self.grid_size}, B_supp={self.B_supp}, grid_type={self.grid_type}"
