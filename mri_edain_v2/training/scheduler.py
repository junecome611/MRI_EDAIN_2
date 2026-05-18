"""Three-phase lambda scheduler (blueprint section 5.1).

Phase 0 (steps 0 - phase_0_end):
    - Hypernetwork FROZEN.
    - U-Net trains on f_{theta_0}(input).
    - lambda_anc fixed at lambda_anc_init.
    - lambda_kl = 0.

Phase 1 (phase_0_end - phase_1_end):
    - Hypernetwork unfrozen.
    - lambda_anc held at lambda_anc_init (strong anchor).
    - lambda_kl still 0.

Phase 2 (phase_1_end - total_steps):
    - lambda_anc cosine-decayed from lambda_anc_init -> lambda_anc_final.
    - lambda_kl ramped 0 -> lambda_kl_final over the first `kl_ramp_steps`.
"""

from __future__ import annotations

import math
from typing import Tuple


class LambdaScheduler:
    """Steps through (lambda_anc, lambda_kl) and reports current phase."""

    def __init__(
        self,
        total_steps: int = 250000,
        phase_0_end: int = 2500,
        phase_1_end: int = 25000,
        lambda_anc_init: float = 1.0e-2,
        lambda_anc_final: float = 1.0e-4,
        lambda_kl_final: float = 1.0e-4,
        kl_ramp_steps: int = 11250,  # 5% of phase 2 (default 225000 steps)
    ):
        # Allow phase_*_end > total_steps so callers can run "phase 0 forever"
        # (= baseline #5 RQSplineFixed) by passing very large boundaries.
        if not (0 < phase_0_end <= phase_1_end):
            raise ValueError(
                f"need 0 < phase_0_end={phase_0_end} <= phase_1_end={phase_1_end}; "
                f"total_steps={total_steps}"
            )
        self.total_steps = int(total_steps)
        self.phase_0_end = int(phase_0_end)
        self.phase_1_end = int(phase_1_end)
        self.lambda_anc_init = float(lambda_anc_init)
        self.lambda_anc_final = float(lambda_anc_final)
        self.lambda_kl_final = float(lambda_kl_final)
        self.kl_ramp_steps = int(kl_ramp_steps)

    def phase(self, step: int) -> int:
        if step < self.phase_0_end:
            return 0
        if step < self.phase_1_end:
            return 1
        return 2

    def hypernet_frozen(self, step: int) -> bool:
        return self.phase(step) == 0

    def lambdas(self, step: int) -> Tuple[float, float]:
        """Return (lambda_anc, lambda_kl) at `step`."""
        p = self.phase(step)
        if p in (0, 1):
            return self.lambda_anc_init, 0.0

        # Phase 2: cosine decay for lambda_anc, ramp for lambda_kl.
        phase_2_len = max(1, self.total_steps - self.phase_1_end)
        t = min(1.0, (step - self.phase_1_end) / phase_2_len)
        anc = self.lambda_anc_final + 0.5 * (self.lambda_anc_init - self.lambda_anc_final) * (
            1.0 + math.cos(math.pi * t)
        )

        kl_progress = min(
            1.0, (step - self.phase_1_end) / max(1, self.kl_ramp_steps)
        )
        kl = self.lambda_kl_final * kl_progress
        return float(anc), float(kl)

    def __repr__(self) -> str:
        return (
            f"LambdaScheduler(total={self.total_steps}, p0={self.phase_0_end}, "
            f"p1={self.phase_1_end}, anc[{self.lambda_anc_init}->"
            f"{self.lambda_anc_final}], kl_final={self.lambda_kl_final}, "
            f"kl_ramp={self.kl_ramp_steps})"
        )
