"""Per-coordinate standardization of the gamma vector (blueprint section 2.3).

Reference: Gonzalez Ortiz et al., "Magnitude Invariant Parametrizations Improve
Hypernetwork Learning", ICLR 2024 (arXiv:2304.07645).

Statistics (mu_i, sigma_i) are computed ONCE on the training set and FROZEN
during training (registered as non-trainable buffers).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CoordinateStandardizer(nn.Module):
    """Per-coordinate (per-percentile-slot) standardize: (gamma - mu) / sigma.

    Each of the 11 percentile dimensions gets its own (mu_i, sigma_i).
    Stats are fit by `.fit(training_gammas)` BEFORE training and never updated.

    Buffers:
        mu:    shape (n_dim,)
        sigma: shape (n_dim,)
        is_fit: shape (), 0 = not yet fit, 1 = fit.
    """

    def __init__(self, n_dim: int = 11, eps: float = 1e-8):
        super().__init__()
        self.n_dim = int(n_dim)
        self.eps = float(eps)
        self.register_buffer("mu", torch.zeros(self.n_dim))
        self.register_buffer("sigma", torch.ones(self.n_dim))
        self.register_buffer("is_fit", torch.zeros((), dtype=torch.uint8))

    @torch.no_grad()
    def fit(self, training_gammas: torch.Tensor) -> None:
        """Compute per-coordinate mean and std across N training scans.

        Args:
            training_gammas: tensor of shape (N, n_dim). Raw, not yet standardized.
        """
        if training_gammas.ndim != 2 or training_gammas.shape[1] != self.n_dim:
            raise ValueError(
                f"training_gammas must be (N, {self.n_dim}), got {tuple(training_gammas.shape)}"
            )
        g = training_gammas.to(torch.float32)
        mu = g.mean(dim=0)
        sigma = g.std(dim=0, unbiased=False)
        sigma = torch.clamp(sigma, min=self.eps)

        # In-place copy preserves buffer registration.
        self.mu.copy_(mu.to(self.mu.dtype))
        self.sigma.copy_(sigma.to(self.sigma.dtype))
        self.is_fit.fill_(1)

    def forward(self, gamma_raw: torch.Tensor) -> torch.Tensor:
        """Apply (gamma - mu) / sigma. Accepts (n_dim,) or (B, n_dim).

        The standardizer does not require gradients on (mu, sigma); buffers
        are not parameters. Input gamma_raw is detached by the caller
        (see PercentileSummary), so this op produces a detached output too.
        """
        if not bool(self.is_fit.item()):
            raise RuntimeError(
                "CoordinateStandardizer.fit(training_gammas) must be called "
                "before forward(). Stats are not initialised."
            )
        return (gamma_raw - self.mu) / (self.sigma + self.eps)

    def extra_repr(self) -> str:
        return f"n_dim={self.n_dim}, is_fit={bool(self.is_fit.item())}"
