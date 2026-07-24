"""
Debug: test forward-inverse consistency for delta-function Q(α).

If Q(α) = δ(α − α0), then P(k,τ) = Gaussian centered at k0 − κ(τ)·α0.
The inverse should recover Q̂(ξ) = exp(i ξ·α0) → Q(α) = δ(α − α0).

This test isolates the FFT/deblur/gridding/IFFT chain.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import QInvConfig
from src.forward_model import forward_pass, kappa_vector, generate_test_q
from src.transform import transform_all_tau, compute_frequency_axis
from src.gridding import grid_radial_to_cartesian, coverage_density
from src.reconstruct import reconstruct_q


def test_delta_q():
    """Test with a tight Gaussian Q (approx delta function)."""
    print("=" * 60)
    print("  DELTA-Q FORWARD-INVERSE TEST")
    print("=" * 60)

    # Parameters
    n_a = 128
    alpha_min, alpha_max = -4.0, 4.0
    alpha0 = 2.0 + 0j  # Q centered at (2,0)
    kappa = 1.0
    omega = 0.057
    nir_cycles = 4.0
    k0 = 1.0
    sigma_k = 0.05
    omega_k_max = 20.0  # generous cutoff

    n_tau = 32
    tau = np.linspace(0, 110.0, n_tau)
    n_k = 256

    # Alpha grid
    a1d = np.linspace(alpha_min, alpha_max, n_a)
    da = a1d[1] - a1d[0]
    aR, aI = np.meshgrid(a1d, a1d, indexing="ij")

    # Delta-like Q: very tight Gaussian
    width = da * 2  # slightly wider than one pixel
    alpha_c = aR + 1j * aI
    q_ref = np.exp(-np.abs(alpha_c - alpha0) ** 2 / width ** 2)
    q_ref /= np.sum(q_ref) * da ** 2

    print(f"Q ref: peak at ({alpha0.real}, {alpha0.imag}), "
          f"max={q_ref.max():.4f}, sum={np.sum(q_ref)*da**2:.6f}")

    # Forward pass
    k_shift = kappa * max(abs(alpha_min), abs(alpha_max)) + 3 * sigma_k
    k_min = k0 - k_shift
    k_max = k0 + k_shift
    k = np.linspace(k_min, k_max, n_k)
    dk = k[1] - k[0]

    kv = kappa_vector(tau, kappa, omega, nir_cycles)

    p = forward_pass(
        q_ref, alpha_min, alpha_max, tau, k,
        kappa, omega, nir_cycles, k0, sigma_k,
    )
    print(f"P forward: shape={p.shape}, range=[{p.min():.2e}, {p.max():.2e}]")

    # 1D FFT + deblur
    omega_k = compute_frequency_axis(n_k, dk)
    print(f"Frequency axis: dω={omega_k[1]-omega_k[0]:.4f}, "
          f"nyq={np.max(np.abs(omega_k)):.1f}")

    samples = transform_all_tau(
        p, tau, kv, k0, k_min, dk, sigma_k, omega_k_max,
    )
    n_samples = len(samples.xi_R)
    print(f"Radial samples: {n_samples} "
          f"({n_samples/(n_tau*n_k)*100:.1f}% of total)")

    # Gridding
    xi_max = omega_k_max * kappa  # ξ_max
    n_xi = 128
    print(f"Gridding: {n_samples} samples → {n_xi}×{n_xi}, ξ_max={xi_max:.1f}")

    q_hat_grid = grid_radial_to_cartesian(
        samples.xi_R, samples.xi_I, samples.q_hat,
        n_xi, xi_max, method="linear",
    )

    _, coverage = coverage_density(samples.xi_R, samples.xi_I, n_xi, xi_max)
    print(f"Coverage: {coverage:.1%}")

    # Reconstruct
    q_recon = reconstruct_q(q_hat_grid, xi_max, n_a, alpha_min, alpha_max)

    # Diagnostic
    print(f"\nQ recon: max={q_recon.max():.6f}, "
          f"sum={np.sum(q_recon)*da**2:.6f}")
    print(f"Q ref:   max={q_ref.max():.6f}, "
          f"sum={np.sum(q_ref)*da**2:.6f}")

    # Peak position
    idx_ref = np.unravel_index(np.argmax(q_ref), q_ref.shape)
    idx_recon = np.unravel_index(np.argmax(q_recon), q_recon.shape)
    pos_ref = np.array([a1d[idx_ref[0]], a1d[idx_ref[1]]])
    pos_recon = np.array([a1d[idx_recon[0]], a1d[idx_recon[1]]])
    print(f"Peak ref:   ({pos_ref[0]:.2f}, {pos_ref[1]:.2f})")
    print(f"Peak recon: ({pos_recon[0]:.2f}, {pos_recon[1]:.2f})")

    # L2 error
    l2 = np.sqrt(np.sum((q_recon - q_ref)**2)) / np.sqrt(np.sum(q_ref**2))
    print(f"L2 error: {l2:.6f}")

    # Check forward residual
    p_recon = forward_pass(
        q_recon, alpha_min, alpha_max, tau, k,
        kappa, omega, nir_cycles, k0, sigma_k,
    )
    fres = np.sum((p - p_recon)**2) / np.sum(p**2)
    print(f"Forward residual: {fres:.6e}")


if __name__ == "__main__":
    test_delta_q()
