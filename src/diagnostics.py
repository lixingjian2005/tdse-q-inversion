"""
Diagnostics and validation metrics for Q-inversion.

- Reconstruction error (L2 norm)
- Fourier-space coverage
- Resolution estimation
- Negativity check
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict


def l2_error(
    q_recon: NDArray, q_ref: NDArray
) -> float:
    """Relative L2 error between reconstructed and reference Q.

    E = ||Q_recon - Q_ref||_2 / ||Q_ref||_2
    """
    diff = q_recon - q_ref
    return float(np.sqrt(np.sum(diff ** 2)) / np.sqrt(np.sum(q_ref ** 2)))


def peak_shift(
    q_recon: NDArray, q_ref: NDArray,
    alpha_min: float, alpha_max: float,
) -> float:
    """Peak position deviation in α space (in grid units and a.u.).

    Returns:
        distance (a.u.) between reconstructed and reference peak positions
    """
    n = q_recon.shape[0]
    idx_ref = np.unravel_index(np.argmax(q_ref), q_ref.shape)
    idx_recon = np.unravel_index(np.argmax(q_recon), q_recon.shape)

    alpha_1d = np.linspace(alpha_min, alpha_max, n)

    pos_ref = np.array([alpha_1d[idx_ref[0]], alpha_1d[idx_ref[1]]])
    pos_recon = np.array([alpha_1d[idx_recon[0]], alpha_1d[idx_recon[1]]])

    return float(np.sqrt(np.sum((pos_recon - pos_ref) ** 2)))


def negativity_fraction(q_grid: NDArray) -> float:
    """Fraction of negative values (should be ~0 for valid Q)."""
    neg = q_grid[q_grid < 0]
    total = np.sum(np.abs(q_grid))
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(neg)) / total)


def resolution_estimate(
    xi_max: float, sigma_k: float
) -> Dict[str, float]:
    """Estimate spatial resolution from Fourier cutoff.

    Returns dict with:
        delta_alpha_min: diffraction-limited resolution
        effective_xi_max: effective Fourier radius
    """
    return {
        "delta_alpha_min": float(np.pi / xi_max) if xi_max > 0 else float("inf"),
        "effective_xi_max": float(xi_max),
        "equivalent_sigma_k": float(1.0 / xi_max) if xi_max > 0 else float("inf"),
    }


def forward_residual(
    p_input: NDArray,
    p_reconstructed: NDArray,
) -> float:
    """Compute relative residual of forward prediction.

    R = Σ|P_input - P_recon|² / Σ|P_input|²
    """
    return float(
        np.sum((p_input - p_reconstructed) ** 2) /
        np.sum(p_input ** 2)
    )


def print_report(
    q_recon: NDArray,
    q_ref: NDArray,
    p_input: NDArray,
    p_recon: NDArray,
    alpha_min: float,
    alpha_max: float,
    xi_max: float,
    sigma_k: float,
    coverage: float,
) -> str:
    """Generate a one-page diagnostic report string."""
    l2 = l2_error(q_recon, q_ref)
    shift = peak_shift(q_recon, q_ref, alpha_min, alpha_max)
    neg = negativity_fraction(q_recon)
    res = resolution_estimate(xi_max, sigma_k)
    fres = forward_residual(p_input, p_recon)

    lines = [
        "=" * 60,
        "  Q-INVERSION DIAGNOSTIC REPORT",
        "=" * 60,
        "",
        "  Reconstruction quality:",
        f"    L2 error (Q):              {l2:.6f}",
        f"    Peak shift (a.u.):          {shift:.4f}",
        f"    Negativity fraction:        {neg:.6e}",
        f"    Forward residual (P):       {fres:.6e}",
        "",
        "  Resolution:",
        f"    Fourier cutoff ξ_max:       {xi_max:.2f}",
        f"    Diffraction limit Δα_min:   {res['delta_alpha_min']:.4f}",
        f"    Equivalent σ_k:             {res['equivalent_sigma_k']:.4f}",
        "",
        "  Coverage:",
        f"    Fourier-space coverage:     {coverage:.2%}",
        "",
        "  Q statistics:",
        f"    max Q:  {np.max(q_recon):.6e}",
        f"    sum Q:  {np.sum(q_recon):.6e}",
        f"    ref max: {np.max(q_ref):.6e}",
        "=" * 60,
    ]
    return "\n".join(lines)
