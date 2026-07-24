"""
Regularization strategies for Q-function inversion.

- cutoff: hard truncation at ω_k^max (applied in transform step)
- tikhonov: smooth damping in Fourier space
- none: no regularization (debug/testing)
"""

import numpy as np
from numpy.typing import NDArray


def tikhonov_damping(
    q_hat_grid: NDArray,       # [N_xi, N_xi] complex
    xi_max: float,
    lambd: float = 0.01,
) -> NDArray:
    """Apply Tikhonov regularization in Fourier space.

    Q̂_reg(ξ) = Q̂(ξ) / (1 + λ|ξ|²)

    This smoothly damps high-frequency components, equivalent to
    minimizing ||P - F[Q]||² + λ||∇Q||² in real space.

    Args:
        q_hat_grid: Q̂ on Cartesian grid
        xi_max: Fourier-space half-width
        lambd: regularization parameter λ (>0)

    Returns:
        Regularized Q̂ grid
    """
    n_xi = q_hat_grid.shape[0]
    xi_1d = np.linspace(-xi_max, xi_max, n_xi)
    Xi_R, Xi_I = np.meshgrid(xi_1d, xi_1d, indexing="ij")
    xi_sq = Xi_R ** 2 + Xi_I ** 2

    damping = 1.0 / (1.0 + lambd * xi_sq)
    return q_hat_grid * damping


def soft_threshold(
    q_real: NDArray,
    threshold: float = 1e-8,
    d_alpha: float = 1.0,
) -> NDArray:
    """Remove small negative/noise values from real-space Q.

    Preserves integral normalization: ∫ Q d²α = 1.

    Args:
        q_real: Q(α) array
        threshold: values below this are zeroed
        d_alpha: grid spacing (for normalization)

    Returns:
        Cleaned Q
    """
    q = np.maximum(q_real, 0.0)
    q[q < threshold] = 0.0
    # Re-normalize preserving integral: ∫ Q d²α = Σ Q · dα² = 1
    total = np.sum(q) * d_alpha ** 2
    if total > 0:
        q /= total
    return q
