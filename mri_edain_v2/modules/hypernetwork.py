"""Hypernetwork that maps the conditioning vector gamma to spline params
(blueprint section 2.4).

Architecture: 2-layer GELU MLP with ZERO-init output layer.

The zero-init output is critical: at step 0,
    MLP_phi(gamma) = 0  =>  theta = theta^(0) + 0 = theta^(0)
which means the spline starts exactly at the population Nyul mapping. This is
the "population-Nyul-at-init" anchor (blueprint section 2.4, error 13.8).

Reference for zero-init principle:
    Zhang, Dauphin, Ma. "Fixup Initialization". ICLR 2019 (arXiv:1901.09321).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Hypernetwork(nn.Module):
    """2-layer GELU MLP. Output layer is zero-initialised.

    Args:
        input_dim:   dimensionality of conditioning vector (default 11).
        hidden_dim:  hidden width (default 64).
        output_dim:  dimensionality of the residual delta-theta returned.
                     For RQ-spline with K knots this is 3K - 1.
        zero_init_output: if True, weights AND bias of final layer are 0
                          (recommended; gives identity perturbation at init).
                          If False, final layer uses small-Kaiming init * 0.01
                          (ablation B7 in blueprint section 7).

    Returns from `forward(gamma)`: the residual to add to theta^(0), shape
    matching `output_dim` with the same leading batch dims as `gamma`.
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        output_dim: int = 26,  # = 3K - 1 for K = 9
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.zero_init_output = bool(zero_init_output)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self) -> None:
        # Kaiming init for hidden layers (GELU ~ ReLU-like for fan_in scaling).
        for layer in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

        if self.zero_init_output:
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)
        else:
            # Small-random (ablation B7): Kaiming * 0.01.
            nn.init.kaiming_normal_(self.fc3.weight, nonlinearity="relu")
            with torch.no_grad():
                self.fc3.weight.mul_(0.01)
            nn.init.zeros_(self.fc3.bias)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """Compute residual delta-theta.

        Args:
            gamma: (input_dim,) or (B, input_dim) standardised summary vector.

        Returns:
            delta_theta with same leading dims as gamma and trailing output_dim.
        """
        if gamma.shape[-1] != self.input_dim:
            raise ValueError(
                f"gamma last dim must be {self.input_dim}, got {gamma.shape[-1]}"
            )
        h = self.act(self.fc1(gamma))
        h = self.act(self.fc2(h))
        delta = self.fc3(h)
        return delta

    def extra_repr(self) -> str:
        return (
            f"in={self.input_dim}, hidden={self.hidden_dim}, "
            f"out={self.output_dim}, zero_init={self.zero_init_output}"
        )
