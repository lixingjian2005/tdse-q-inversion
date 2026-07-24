"""
Fourier-space gridding: radial samples → Cartesian grid.

Takes the radial samples (ξ_R, ξ_I, Q̂) from the 1D FFT + deblur step
and interpolates them onto a regular Cartesian grid suitable for 2D IFFT.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from scipy.interpolate import griddata, RBFInterpolator


def build_xi_grid(
    n_xi: int, xi_max: float
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Build Cartesian Fourier-space grid.

    Args:
        n_xi: grid size per side
        xi_max: half-width of the grid

    Returns:
        xi_1d: 1D coordinate array [N_xi]
        dxi: grid spacing
        Xi_R, Xi_I: meshgrids [N_xi, N_xi]
    """
    xi_1d = np.linspace(-xi_max, xi_max, n_xi)
    dxi = xi_1d[1] - xi_1d[0]
    Xi_R, Xi_I = np.meshgrid(xi_1d, xi_1d, indexing="ij")
    return xi_1d, dxi, Xi_R, Xi_I


def grid_radial_to_cartesian(
    xi_R_samples: NDArray,    # [N_samples]
    xi_I_samples: NDArray,    # [N_samples]
    q_hat_samples: NDArray,   # [N_samples] complex
    n_xi: int,
    xi_max: float,
    method: str = "rbf",
) -> NDArray:
    """Interpolate radial Fourier samples onto Cartesian grid.

    Args:
        xi_R_samples, xi_I_samples: sample coordinates
        q_hat_samples: Q̂ values (complex)
        n_xi: Cartesian grid size per side
        xi_max: grid half-width
        method: interpolation method
            - 'bin': direct binning + Gaussian smoothing
            - 'linear': scipy griddata linear
            - 'nearest': scipy griddata nearest
            - 'rbf': radial basis function (thin-plate spline)

    Returns:
        Q̂_grid [N_xi, N_xi], complex
    """
    if method == "bin":
        return _grid_binning(
            xi_R_samples, xi_I_samples, q_hat_samples,
            n_xi, xi_max,
        )

    _, _, Xi_R, Xi_I = build_xi_grid(n_xi, xi_max)

    sample_points = np.column_stack([xi_R_samples, xi_I_samples])

    if method == "rbf":
        # Thin-plate spline RBF — good for scattered data
        # Interpolate real and imag separately
        try:
            rbf_real = RBFInterpolator(
                sample_points, np.real(q_hat_samples),
                kernel="thin_plate_spline",
            )
            rbf_imag = RBFInterpolator(
                sample_points, np.imag(q_hat_samples),
                kernel="thin_plate_spline",
            )
            grid_points = np.column_stack([Xi_R.ravel(), Xi_I.ravel()])
            real_grid = rbf_real(grid_points).reshape(n_xi, n_xi)
            imag_grid = rbf_imag(grid_points).reshape(n_xi, n_xi)
            q_hat_grid = real_grid + 1j * imag_grid
        except Exception:
            # Fallback to linear if RBF fails
            print("Warning: RBF interpolation failed, falling back to linear")
            method = "linear"

    if method in ("linear", "nearest"):
        q_hat_grid = griddata(
            sample_points, q_hat_samples,
            (Xi_R, Xi_I), method=method, fill_value=0.0 + 0j,
        )

    # Fill any NaN values with zero (handle complex separately)
    q_hat_grid = np.nan_to_num(np.real(q_hat_grid)) + \
                 1j * np.nan_to_num(np.imag(q_hat_grid))

    return q_hat_grid


def _grid_binning(
    xi_R_samples: NDArray,
    xi_I_samples: NDArray,
    q_hat_samples: NDArray,
    n_xi: int,
    xi_max: float,
) -> NDArray:
    """Grid radial samples via direct binning + Gaussian smoothing.

    Accumulates samples in histogram bins, then convolves with a
    narrow Gaussian to fill gaps. Handles sparse radial spoke data
    much better than Delaunay-based interpolation.
    """
    from scipy.ndimage import gaussian_filter

    xi_1d, dxi, _, _ = build_xi_grid(n_xi, xi_max)

    # Coherent accumulation (sum, not average!)
    # Q̂ values are complex amplitudes — they must be ADDED coherently.
    # Averaging would destroy phase information.
    q_hat_grid = np.zeros((n_xi, n_xi), dtype=complex)
    weight_grid = np.zeros((n_xi, n_xi), dtype=float)

    for k in range(len(xi_R_samples)):
        ir = int(np.round((xi_R_samples[k] + xi_max) / (2.0 * xi_max) * (n_xi - 1)))
        ii = int(np.round((xi_I_samples[k] + xi_max) / (2.0 * xi_max) * (n_xi - 1)))
        if 0 <= ir < n_xi and 0 <= ii < n_xi:
            q_hat_grid[ir, ii] += q_hat_samples[k]
            weight_grid[ir, ii] += 1.0

    # Normalize by bin count to avoid density bias
    # (compensates for varying sampling density)
    mask = weight_grid > 0
    q_hat_grid[mask] /= weight_grid[mask]

    # Gaussian smoothing to fill gaps (σ = 0.8 pixels — mild)
    sigma_pix = 0.8
    q_hat_grid = (
        gaussian_filter(np.real(q_hat_grid), sigma_pix) +
        1j * gaussian_filter(np.imag(q_hat_grid), sigma_pix)
    )

    return q_hat_grid


def coverage_density(
    xi_R_samples: NDArray,
    xi_I_samples: NDArray,
    n_xi: int,
    xi_max: float,
) -> Tuple[NDArray, float]:
    """Estimate Fourier-space sampling coverage.

    Returns:
        density_map: [N_xi, N_xi] sample count per cell
        coverage_fraction: fraction of cells with ≥1 sample
    """
    _, dxi, _, _ = build_xi_grid(n_xi, xi_max)

    density = np.zeros((n_xi, n_xi), dtype=int)
    for xr, xi in zip(xi_R_samples, xi_I_samples):
        ir = int((xr + xi_max) / (2 * xi_max) * n_xi)
        ii = int((xi + xi_max) / (2 * xi_max) * n_xi)
        if 0 <= ir < n_xi and 0 <= ii < n_xi:
            density[ir, ii] += 1

    coverage = np.sum(density > 0) / (n_xi * n_xi)
    return density, coverage
