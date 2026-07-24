"""
Systematic error analysis for Q-inversion pipeline.

Decomposes reconstruction error into contributions from:
  1. FT discretization (finite k grid, dk, FFT)
  2. Radial sampling (coverage gaps between spokes)
  3. Gridding method (binning/smoothing vs ideal)
  4. Cutoff (omega_k_max hard truncation)
  5. Forward model discretization (finite alpha grid)
"""

import numpy as np
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.forward_model import forward_pass, generate_test_q, kappa_vector
from src.transform import transform_all_tau, compute_frequency_axis
from src.gridding import grid_radial_to_cartesian, build_xi_grid, coverage_density
from src.reconstruct import reconstruct_q
from src.regularize import soft_threshold
from src.diagnostics import l2_error, peak_shift


def ideal_reference(q_ref, alpha_min, alpha_max, n_alpha):
    """Return reference Q (identity — perfect reconstruction)."""
    return q_ref.copy()


def analytic_qhat_grid(q_ref, alpha_min, alpha_max, n_xi, xi_max):
    """Build ideal Qhat from analytic 2D FT of Q_ref.

    Qhat(xi) = ∫ Q(α) exp(+i xi·α) d²α
    """
    n_a = q_ref.shape[0]
    a1d = np.linspace(alpha_min, alpha_max, n_a)
    da = a1d[1] - a1d[0]
    aR, aI = np.meshgrid(a1d, a1d, indexing='ij')

    xi_1d, dxi, Xi_R, Xi_I = build_xi_grid(n_xi, xi_max)
    q_hat = np.zeros((n_xi, n_xi), dtype=complex)

    for i in range(n_xi):
        for j in range(n_xi):
            phase = Xi_R[i,j] * aR + Xi_I[i,j] * aI
            q_hat[i,j] = np.sum(q_ref * np.exp(1j * phase)) * da**2

    return q_hat


def error_decomposition(cfg, qtype='coherent'):
    """Decompose reconstruction error into sources.

    Returns dict with L2 errors for each stage.
    """
    p = cfg.physics; n = cfg.numerics
    alpha_min, alpha_max, n_alpha = cfg.alpha_grid

    # Reference
    q_ref = generate_test_q(n_alpha, alpha_min, alpha_max, qtype, alpha0=3.0+0j)

    # ── Stage 0: Ideal (analytic Qhat → IFFT) ──
    t0 = time.perf_counter()
    q_hat_ideal = analytic_qhat_grid(q_ref, alpha_min, alpha_max, n.n_xi, cfg.xi_max)
    q_ideal = reconstruct_q(q_hat_ideal, cfg.xi_max, n_alpha, alpha_min, alpha_max)
    q_ideal = soft_threshold(q_ideal, d_alpha=cfg.d_alpha)
    l2_ideal = l2_error(q_ideal, q_ref)
    t_ideal = time.perf_counter() - t0

    # ── Stage 1: Forward + FFT + deblur (no gridding error) ──
    tau = np.linspace(p.tau_min, p.tau_max, n.n_tau)
    k_shift = cfg.kappa * abs(alpha_max) + 3.0 * p.sigma_k
    k_min = p.k0 - k_shift; k_max = p.k0 + k_shift
    k = np.linspace(k_min, k_max, n.n_k); dk = k[1] - k[0]
    kv = kappa_vector(tau, cfg.kappa, p.nir_omega, p.nir_cycles)

    p_input = forward_pass(q_ref, alpha_min, alpha_max, tau, k,
                           cfg.kappa, p.nir_omega, p.nir_cycles,
                           p.k0, p.sigma_k)

    omega_k = compute_frequency_axis(n.n_k, dk)
    samples = transform_all_tau(p_input, tau, kv, p.k0, k_min, dk,
                                p.sigma_k, cfg.omega_k_max)

    # ── Stage 2: Binning reconstruction ──
    t2 = time.perf_counter()
    q_hat_bin = grid_radial_to_cartesian(samples.xi_R, samples.xi_I,
                                          samples.q_hat, n.n_xi, cfg.xi_max,
                                          method='bin')
    q_bin = reconstruct_q(q_hat_bin, cfg.xi_max, n_alpha, alpha_min, alpha_max)
    q_bin = soft_threshold(q_bin, d_alpha=cfg.d_alpha)
    l2_bin = l2_error(q_bin, q_ref)
    t_bin = time.perf_counter() - t2

    # ── Stage 3: Linear interpolation ──
    t3 = time.perf_counter()
    try:
        q_hat_lin = grid_radial_to_cartesian(samples.xi_R, samples.xi_I,
                                              samples.q_hat, n.n_xi, cfg.xi_max,
                                              method='linear')
        q_lin = reconstruct_q(q_hat_lin, cfg.xi_max, n_alpha, alpha_min, alpha_max)
        q_lin = soft_threshold(q_lin, d_alpha=cfg.d_alpha)
        l2_lin = l2_error(q_lin, q_ref)
    except Exception:
        l2_lin = float('nan')
    t_lin = time.perf_counter() - t3

    # ── Stage 4: No cutoff (full ω_k range) ──
    omega_k_max_full = np.max(np.abs(omega_k)) * 0.8  # 80% of Nyquist
    samples_full = transform_all_tau(p_input, tau, kv, p.k0, k_min, dk,
                                     p.sigma_k, omega_k_max_full)
    q_hat_full = grid_radial_to_cartesian(samples_full.xi_R, samples_full.xi_I,
                                           samples_full.q_hat, n.n_xi,
                                           omega_k_max_full * cfg.kappa,
                                           method='bin')
    q_full = reconstruct_q(q_hat_full, omega_k_max_full * cfg.kappa,
                           n_alpha, alpha_min, alpha_max)
    q_full = soft_threshold(q_full, d_alpha=cfg.d_alpha)
    l2_nocut = l2_error(q_full, q_ref)

    # Coverage
    _, cov = coverage_density(samples.xi_R, samples.xi_I, n.n_xi, cfg.xi_max)

    return {
        'qtype': qtype,
        'l2_ideal': l2_ideal,
        'l2_bin': l2_bin,
        'l2_lin': l2_lin,
        'l2_nocut': l2_nocut,
        'cov': cov,
        'n_samples': len(samples.xi_R),
        'q_ref': q_ref,
        'q_ideal': q_ideal,
        'q_bin': q_bin,
        'timing': {'ideal': t_ideal, 'bin': t_bin, 'lin': t_lin},
    }


def parameter_sweep(base_cfg, param_name, values, qtype='coherent'):
    """Sweep a parameter and measure L2 error."""
    results = []
    for val in values:
        cfg = base_cfg
        # Set the parameter
        if param_name == 'n_tau':
            cfg.numerics.n_tau = val
        elif param_name == 'omega_k_max':
            cfg.numerics.omega_k_max = val
        elif param_name == 'n_xi':
            cfg.numerics.n_xi = val
        elif param_name == 'sigma_k':
            cfg.physics.sigma_k = val

        # Re-derive
        cfg.derive()

        try:
            r = error_decomposition(cfg, qtype)
            results.append({
                'value': val,
                'l2_bin': r['l2_bin'],
                'cov': r['cov'],
                'n_samples': r['n_samples'],
            })
            print(f"  {param_name}={val}: L2={r['l2_bin']:.4f}, cov={r['cov']:.1%}, "
                  f"samples={r['n_samples']}")
        except Exception as e:
            print(f"  {param_name}={val}: FAILED - {e}")

    return results


def print_report(decomp):
    """Print error decomposition report."""
    print(f"\n{'='*60}")
    print(f"  ERROR DECOMPOSITION: {decomp['qtype']}")
    print(f"{'='*60}")
    print(f"  Coverage: {decomp['cov']:.1%} ({decomp['n_samples']} samples)")
    print(f"")
    print(f"  L2 errors:")
    print(f"    Ideal (analytic Qhat):     {decomp['l2_ideal']:.4f}")
    print(f"    Full pipeline (bin):       {decomp['l2_bin']:.4f}")
    print(f"    Full pipeline (linear):    {decomp['l2_lin']:.4f}")
    print(f"    No cutoff (full ω range):  {decomp['l2_nocut']:.4f}")
    print(f"")
    print(f"  Error contributions:")

    if not np.isnan(decomp['l2_ideal']):
        print(f"    Discretization (ideal - 0):  {decomp['l2_ideal']:.4f}")
    print(f"    Sampling+gridding (bin - ideal): {decomp['l2_bin'] - decomp['l2_ideal']:.4f}")
    if not np.isnan(decomp['l2_lin']):
        print(f"    Gridding improvement (bin - linear): {decomp['l2_bin'] - decomp['l2_lin']:.4f}")
    print(f"    Cutoff effect (bin - nocut): {decomp['l2_bin'] - decomp['l2_nocut']:.4f}")
    print(f"")
    print(f"  Timing:")
    for k, v in decomp['timing'].items():
        print(f"    {k}: {v:.2f}s")


if __name__ == "__main__":
    from src.config import QInvConfig

    cfg = QInvConfig.from_namelist('examples/closed_loop_test.nml')

    print("=" * 60)
    print("  ERROR SOURCE ANALYSIS")
    print("=" * 60)

    # 1. Decompose error for coherent state
    decomp = error_decomposition(cfg, 'coherent')
    print_report(decomp)

    # 2. Parameter sweep: N_tau
    print(f"\n--- N_tau sweep ---")
    sweep_n_tau = parameter_sweep(cfg, 'n_tau', [32, 64, 128, 256], 'coherent')

    # 3. Parameter sweep: omega_k_max
    print(f"\n--- omega_k_max sweep ---")
    sweep_omega = parameter_sweep(cfg, 'omega_k_max', [6, 10, 15, 25], 'coherent')

    # 4. Parameter sweep: n_xi (grid resolution)
    print(f"\n--- n_xi sweep ---")
    sweep_nxi = parameter_sweep(cfg, 'n_xi', [64, 128, 256], 'coherent')

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Best L2: {min(r['l2_bin'] for r in sweep_n_tau):.4f} "
          f"(N_tau sweep)")
    print(f"  Best L2: {min(r['l2_bin'] for r in sweep_omega):.4f} "
          f"(omega_k_max sweep)")
