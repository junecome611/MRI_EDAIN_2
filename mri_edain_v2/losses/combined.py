"""Composite training loss: L = L_seg + lambda_anc * L_anc + lambda_KL * L_KL
(blueprint section 2.7).

Loss scheduling for (lambda_anc, lambda_KL) is the trainer's responsibility
(blueprint section 5.1 three-phase schedule); this combiner just sums the
weighted components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CombinedLossOutput:
    """Per-step breakdown for logging."""
    total: torch.Tensor
    seg: torch.Tensor
    anchor: torch.Tensor
    kl: torch.Tensor
    lambda_anc: float
    lambda_kl: float


class CombinedLoss(nn.Module):
    """Sum-of-weighted-components combiner.

    Note: this module does NOT own the segmentation loss, the anchor loss, or
    the KL loss objects -- the trainer instantiates them and passes results
    in. This keeps the combiner agnostic to the seg loss choice (DiceCE,
    DiceFocal, etc.) and lets the caller cheaply skip components when their
    weight is zero.
    """

    def __init__(self, lambda_anc: float = 1e-2, lambda_kl: float = 0.0):
        super().__init__()
        self.lambda_anc = float(lambda_anc)
        self.lambda_kl = float(lambda_kl)

    def set_lambdas(self, lambda_anc: float, lambda_kl: float) -> None:
        """Update weights (called from the trainer phase scheduler)."""
        self.lambda_anc = float(lambda_anc)
        self.lambda_kl = float(lambda_kl)

    def forward(
        self,
        seg_loss: torch.Tensor,
        anchor_loss: Optional[torch.Tensor] = None,
        kl_loss: Optional[torch.Tensor] = None,
    ) -> CombinedLossOutput:
        zero = seg_loss.new_zeros(())
        anc = anchor_loss if anchor_loss is not None else zero
        kl = kl_loss if kl_loss is not None else zero

        total = seg_loss + self.lambda_anc * anc + self.lambda_kl * kl
        return CombinedLossOutput(
            total=total,
            seg=seg_loss.detach(),
            anchor=anc.detach(),
            kl=kl.detach(),
            lambda_anc=self.lambda_anc,
            lambda_kl=self.lambda_kl,
        )

    def extra_repr(self) -> str:
        return f"lambda_anc={self.lambda_anc}, lambda_kl={self.lambda_kl}"
