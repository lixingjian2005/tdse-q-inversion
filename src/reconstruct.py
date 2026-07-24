"""
2D inverse FFT reconstruction: Q̂(ξ) → Q(α).

Implements the continuous inverse Fourier transform:
    Q(α) = (1/(2π)²) ∫∫ Q̂(ξ) e^{-i ξ·α} d²ξ

using numpy's discrete ifft2 with correct scaling.
"""

import numpy as np
from numpy.typing import NDArray


def reconstruct_q(
    q_hat_grid: NDArray,   # [N_xi, N_xi] complex
    xi_max: float,
    n_alpha: int,
    alpha_min: float,
    alpha_max: float,
) -> NDArray:
    """Reconstruct Q(α) from Q̂(ξ) via 2D IFFT.

    The Fourier grid has N_ξ points spanning [-ξ_max, ξ_max] with spacing
    dξ = 2ξ_max/(N_ξ−1).  The standard FFT relationship gives:
        dα = 2π / (N_ξ · dξ)
    so the real-space grid covers ±N_ξ·dα/2 ≈ ±π/dξ.

    Args:
        q_hat_grid: Q̂ on Cartesian grid [N_ξ, N_ξ], zero-frequency at center
        xi_max: Fourier-space half-width (a.u.)
        n_alpha: output Q grid size per side
        alpha_min, alpha_max: output Q grid limits

    Returns:
        Q(α) array [N_alpha, N_alpha], real and non-negative
    """
    n_xi = q_hat_grid.shape[0]
    dxi = 2.0 * xi_max / (n_xi - 1) if n_xi > 1 else 1.0

    # ifftshift: move DC from center → corner for FFT
    q_hat_shifted = np.fft.ifftshift(q_hat_grid)

    # Use fft2 (forward, exp(-i...)), NOT ifft2 (inverse, exp(+i...))
    # Physics convention: Q(α) ∝ ∫ Q̂(ξ) e^{-iξ·α} d²ξ
    # numpy fft2: result = Σ X[k,l] exp(-2πi(k·p+l·q)/N)
    q_raw = np.fft.fft2(q_hat_shifted)  # [N_xi, N_xi]

    # fftshift: move zero-frequency back to center
    q_raw = np.fft.fftshift(q_raw)

    # Scale to match continuous Fourier convention:
    #   Continuous: Q(α) = (1/2π)² ∫ Q̂(ξ) e^{-iξ·α} d²ξ
    #   Discrete:   Q ≈ dξ² / (2π)² · FFT[Q̂_shifted]
    # fft2 has no 1/N² factor, so we don't compensate for it
    scaling = (dxi ** 2) / (2.0 * np.pi) ** 2
    q_real = np.real(q_raw) * scaling

    # Clamp negatives
    q_real = np.maximum(q_real, 0.0)

    # FFT natural real-space grid half-width:
    #   dα_fft = 2π/(N·dξ),  range = [-N·dα_fft/2, N·dα_fft/2] = [-π/dξ, π/dξ]
    alpha_fft_max = np.pi / dxi

    # Resize to target grid if needed
    if n_alpha != n_xi or not np.isclose(alpha_min, -alpha_fft_max) or \
       not np.isclose(alpha_max, alpha_fft_max):
        q_real = _resample_q(q_real, alpha_fft_max, n_alpha, alpha_min, alpha_max)

    # Normalize: ∫ Q d²α = 1
    d_alpha = (alpha_max - alpha_min) / (n_alpha - 1) if n_alpha > 1 else 1.0
    total = np.sum(q_real) * d_alpha ** 2
    if total > 0:
        q_real /= total

    return q_real


def _resample_q(
    q_in: NDArray,
    in_max: float,
    n_out: int,
    alpha_min: float,
    alpha_max: float,
) -> NDArray:
    """Resample Q grid to different size/range using real-space interpolation.

    Args:
        q_in: input Q on grid [-in_max, in_max] with n_in points
        in_max: half-width of input grid in α space
        n_out: output grid size
        alpha_min, alpha_max: output grid limits
    """
    from scipy.interpolate import RegularGridInterpolator

    n_in = q_in.shape[0]
    alpha_in = np.linspace(-in_max, in_max, n_in)
    alpha_out = np.linspace(alpha_min, alpha_max, n_out)

    interp = RegularGridInterpolator(
        (alpha_in, alpha_in), q_in,
        bounds_error=False, fill_value=0.0,
    )
    A_out_R, A_out_I = np.meshgrid(alpha_out, alpha_out, indexing="ij")
    pts = np.column_stack([A_out_R.ravel(), A_out_I.ravel()])
    q_out = interp(pts).reshape(n_out, n_out)

    return np.maximum(q_out, 0.0)
