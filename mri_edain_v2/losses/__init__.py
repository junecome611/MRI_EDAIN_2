"""Loss functions for MRI-EDAIN v2 (blueprint section 2.7, 4.5)."""

from mri_edain_v2.losses.anchor import (
    FunctionSpaceAnchorLoss,
    function_space_anchor_loss,
)
from mri_edain_v2.losses.kl import KLAnchorLoss
from mri_edain_v2.losses.combined import CombinedLoss

__all__ = [
    "FunctionSpaceAnchorLoss",
    "function_space_anchor_loss",
    "KLAnchorLoss",
    "CombinedLoss",
]
