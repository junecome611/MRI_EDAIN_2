"""Exponential moving average of hypernet weights (blueprint section 5.3).

phi^EMA <- 0.99 * phi^EMA + 0.01 * phi   (every step after Phase 0)

At inference (validation, sliding window, OOF) the EMA copy is used in place
of the live weights for stability.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Tuple

import torch
import torch.nn as nn


class EMAWrapper:
    """Maintains an EMA shadow of `module` parameters.

    Usage::

        ema = EMAWrapper(model.hypernet, decay=0.99)
        # training loop
        for step in ...:
            loss.backward()
            optimizer.step()
            if not lambda_sched.hypernet_frozen(step):
                ema.update()
        # validation / inference
        with ema.swap_in(model.hypernet):
            run_validation(model)
    """

    def __init__(self, module: nn.Module, decay: float = 0.99):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must lie in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow: List[torch.Tensor] = [
            p.detach().clone() for p in module.parameters()
        ]
        self._param_refs = [id(p) for p in module.parameters()]
        self._num_params = len(self.shadow)

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """In-place EMA update of shadow buffers from `module`'s parameters."""
        live = list(module.parameters())
        if len(live) != self._num_params:
            raise RuntimeError(
                f"EMA module parameter count changed: was {self._num_params}, "
                f"now {len(live)}. Did you re-create the module?"
            )
        d = self.decay
        for shadow, p in zip(self.shadow, live):
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def swap_in(self, module: nn.Module) -> Iterator[None]:
        """Context manager: temporarily replace module's params with shadow.

        Restores live values on exit. Safe for inference; do NOT use within
        an active autograd graph (gradient tracking is disabled here).
        """
        live = list(module.parameters())
        if len(live) != self._num_params:
            raise RuntimeError(
                f"EMA module parameter count changed: was {self._num_params}, "
                f"now {len(live)}."
            )

        cached: List[torch.Tensor] = []
        try:
            with torch.no_grad():
                for shadow, p in zip(self.shadow, live):
                    cached.append(p.detach().clone())
                    p.data.copy_(shadow.data)
            yield
        finally:
            with torch.no_grad():
                for stash, p in zip(cached, live):
                    p.data.copy_(stash)

    def state_dict(self) -> List[torch.Tensor]:
        return [s.detach().clone() for s in self.shadow]

    def load_state_dict(self, shadow_list: List[torch.Tensor]) -> None:
        if len(shadow_list) != self._num_params:
            raise ValueError(
                f"Cannot load EMA state: expected {self._num_params} tensors, "
                f"got {len(shadow_list)}"
            )
        for sh, src in zip(self.shadow, shadow_list):
            sh.copy_(src)
