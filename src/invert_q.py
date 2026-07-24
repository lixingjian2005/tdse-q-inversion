#!/usr/bin/env python
"""
Q-function inversion from streaking P(k,τ).

Main pipeline:
    1. Read P(k,τ) data (or generate synthetic from known Q)
    2. 1D FFT + deblur + cutoff → Fourier radial samples
    3. Grid radial samples → Cartesian Q̂(ξ) grid
    4. Optional regularization (Tikhonov)
    5. 2D IFFT → Q(α)
    6. Diagnostics + visualization

Usage:
    # Closed-loop test (synthetic data)
    python invert_q.py --mode closed-loop --qtype coherent

    # From v5.1 output
    python invert_q.py --config examples/qinv_bsv.nml

    # Custom config
    python invert_q.py --config my_run.nml
"""

import sys
import os
import argparse
import time
import numpy as np

# Add parent to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import QInvConfig
from src.forward_model import (
    forward_pass, generate_test_q, kappa_vector, nir_envelope,
)
from src.transform import transform_all_tau
from src.gridding import grid_radial_to_cartesian, coverage_density
from src.reconstruct import reconstruct_q
from src.regularize import tikhonov_damping, soft_threshold
from src.io_data import write_q_function, write_p_comparison
from src.diagnostics import (
    l2_error, peak_shift, negativity_fraction,
    resolution_estimate, forward_residual, print_report,
)


def run_pipeline(
    config: QInvConfig,
    p_input: np.ndarray = None,
    q_ref: np.ndarray = None,
) -> dict:
    """Execute the full inversion pipeline.

    Args:
        config: QInvConfig with all parameters
        p_input: P(k,τ) data [N_τ, N_k], or None for synthetic
        q_ref: Reference Q [N_α, N_α] for validation, or None

    Returns:
        dict with all results (q_recon, p_recon, diagnostics, etc.)
    """
    p = config.physics
    n = config.numerics

    t_start = time.perf_counter()

    # ── 0. Build grids ──────────────────────────────────────────────

    # α grids (for forward model and output) — must come first
    alpha_min, alpha_max, n_alpha = config.alpha_grid

    # τ grid
    tau = np.linspace(p.tau_min, p.tau_max, n.n_tau)

    # k grid: need to cover momentum range including streaking shift
    # k_s = k0 - κ·α: with α_max and κ, shift up to ±κ·α_max
    # Add 3σ_k padding for Gaussian tails
    k_shift_max = config.kappa * abs(alpha_max) + 3.0 * p.sigma_k
    k_min = p.k0 - k_shift_max
    k_max = p.k0 + k_shift_max
    k = np.linspace(k_min, k_max, n.n_k)
    dk = k[1] - k[0]

    # κ vectors
    kv = kappa_vector(tau, config.kappa, p.nir_omega, p.nir_cycles)

    # ── 0a. Generate or load P(k,τ) ─────────────────────────────────

    if p_input is None:
        # Synthetic: generate from known Q
        qtype = getattr(config, "_qtype", "coherent")
        q_ref = generate_test_q(
            n_alpha, alpha_min, alpha_max, qtype,
            alpha0=3.0 + 0j,
        )
        p_input = forward_pass(
            q_ref, alpha_min, alpha_max, tau, k,
            config.kappa, p.nir_omega, p.nir_cycles,
            p.k0, p.sigma_k,
        )
        print(f"  Synthetic P generated: Q type={qtype}, "
              f"P range=[{p_input.min():.2e}, {p_input.max():.2e}]")

    # ── 1. 1D FFT + deblur ──────────────────────────────────────────

    print(f"  Transform: {n.n_tau} τ × {n.n_k} k → FFT + deblur "
          f"(ω_k^max={config.omega_k_max:.1f})")

    samples = transform_all_tau(
        p_input, tau, kv,
        p.k0, k_min, dk, p.sigma_k, config.omega_k_max,
    )
    n_samples = len(samples.xi_R)
    print(f"  Radial samples: {n_samples} kept "
          f"({n_samples / (n.n_tau * n.n_k) * 100:.1f}% of total)")

    # ── 2. Gridding ─────────────────────────────────────────────────

    xi_max = config.xi_max
    print(f"  Gridding: radial → {n.n_xi}×{n.n_xi} Cartesian, "
          f"ξ_max={xi_max:.1f}, method={n.interp_method}")

    q_hat_grid = grid_radial_to_cartesian(
        samples.xi_R, samples.xi_I, samples.q_hat,
        n.n_xi, xi_max, method=n.interp_method,
    )

    # Coverage diagnostic
    _, coverage = coverage_density(
        samples.xi_R, samples.xi_I, n.n_xi, xi_max,
    )
    print(f"  Fourier coverage: {coverage:.1%}")

    # ── 3. Regularization ───────────────────────────────────────────

    if n.regularize == "tikhonov":
        print(f"  Tikhonov: λ={n.tikhonov_lambda}")
        q_hat_grid = tikhonov_damping(q_hat_grid, xi_max, n.tikhonov_lambda)

    # ── 4. 2D IFFT → Q(α) ───────────────────────────────────────────

    print(f"  Reconstruct: 2D IFFT → Q({n_alpha}×{n_alpha})")
    q_recon = reconstruct_q(
        q_hat_grid, xi_max,
        n_alpha, alpha_min, alpha_max,
    )
    q_recon = soft_threshold(q_recon)

    # ── 5. Reconstruct P for residual check ─────────────────────────

    p_recon = forward_pass(
        q_recon, alpha_min, alpha_max, tau, k,
        config.kappa, p.nir_omega, p.nir_cycles,
        p.k0, p.sigma_k,
    )

    t_elapsed = time.perf_counter() - t_start
    print(f"  Pipeline complete: {t_elapsed:.2f}s")

    return {
        "q_recon": q_recon,
        "q_ref": q_ref,
        "p_input": p_input,
        "p_recon": p_recon,
        "tau": tau,
        "k": k,
        "q_hat_grid": q_hat_grid,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "xi_max": xi_max,
        "coverage": coverage,
        "sigma_k": p.sigma_k,
        "timing": t_elapsed,
    }


def run_closed_loop(config: QInvConfig, qtype: str = "coherent") -> dict:
    """Run closed-loop validation: known Q → synthetic P → invert → compare.

    Args:
        config: QInvConfig
        qtype: test Q distribution type ('coherent', 'bsv', 'thermal', 'gaussian_mixture')

    Returns:
        results dict
    """
    print(f"\n{'='*60}")
    print(f"  CLOSED-LOOP TEST: {qtype}")
    print(f"  {config.summary()}")
    print(f"{'='*60}\n")

    # Store qtype for pipeline
    config._qtype = qtype

    results = run_pipeline(config)

    # ── Diagnostics ─────────────────────────────────────────────────

    if results["q_ref"] is not None:
        diag_header = print_report(
            results["q_recon"], results["q_ref"],
            results["p_input"], results["p_recon"],
            results["alpha_min"], results["alpha_max"],
            results["xi_max"], results["sigma_k"],
            results["coverage"],
        )
        print("\n" + diag_header)

    # ── Output files ─────────────────────────────────────────────────
    io = config.io
    os.makedirs(io.output_dir, exist_ok=True)

    # Write Q function
    q_path = os.path.join(io.output_dir, f"{io.output_prefix}_qfunc.dat")
    write_q_function(
        q_path, results["q_recon"],
        results["alpha_min"], results["alpha_max"],
        f"Reconstructed Q, qtype={qtype}, "
        f"L2={l2_error(results['q_recon'], results['q_ref']):.4f}"
        if results["q_ref"] is not None else "Reconstructed Q",
    )
    print(f"\n  Q output: {q_path}")

    # Write P comparison
    p_path = os.path.join(io.output_dir, f"{io.output_prefix}_p_compare.dat")
    write_p_comparison(
        p_path, results["tau"], results["k"],
        results["p_input"], results["p_recon"],
    )
    print(f"  P comparison: {p_path}")

    # ── Plots ────────────────────────────────────────────────────────

    if io.plot:
        plot_results(results, io.output_dir, io.output_prefix, qtype)

    return results


def plot_results(
    results: dict, output_dir: str, prefix: str, qtype: str,
) -> None:
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    alpha_min = results["alpha_min"]
    alpha_max = results["alpha_max"]
    n_a = results["q_recon"].shape[0]
    extent = [alpha_min, alpha_max, alpha_min, alpha_max]

    # Top row: Q reference, reconstructed, difference
    ax = axes[0, 0]
    if results["q_ref"] is not None:
        im = ax.imshow(results["q_ref"].T, origin="lower", extent=extent,
                       cmap="inferno", aspect="equal")
        ax.set_title(f"Q ref ({qtype})")
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, "No reference", ha="center", va="center",
                transform=ax.transAxes)

    ax = axes[0, 1]
    im = ax.imshow(results["q_recon"].T, origin="lower", extent=extent,
                   cmap="inferno", aspect="equal")
    ax.set_title("Q reconstructed")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 2]
    if results["q_ref"] is not None:
        diff = results["q_recon"] - results["q_ref"]
        vmax = max(np.abs(diff.min()), np.abs(diff.max()))
        im = ax.imshow(diff.T, origin="lower", extent=extent,
                       cmap="RdBu_r", aspect="equal",
                       vmin=-vmax, vmax=vmax)
        ax.set_title("Q diff (recon − ref)")
        plt.colorbar(im, ax=ax)

    # Bottom row: P input, reconstructed, residual
    p_in = results["p_input"]
    p_recon = results["p_recon"]
    tau = results["tau"]
    k = results["k"]

    ax = axes[1, 0]
    im = ax.pcolormesh(k, tau, p_in, shading="auto", cmap="inferno")
    ax.set_title("P(k,τ) input")
    ax.set_xlabel("k (a.u.)"); ax.set_ylabel("τ (a.u.)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.pcolormesh(k, tau, p_recon, shading="auto", cmap="inferno")
    ax.set_title("P(k,τ) reconstructed")
    ax.set_xlabel("k (a.u.)"); ax.set_ylabel("τ (a.u.)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    p_res = p_in - p_recon
    vmax = max(np.abs(p_res.min()), np.abs(p_res.max()))
    im = ax.pcolormesh(k, tau, p_res, shading="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)
    ax.set_title("P residual (input − recon)")
    ax.set_xlabel("k (a.u.)"); ax.set_ylabel("τ (a.u.)")
    plt.colorbar(im, ax=ax)

    plt.suptitle(f"Q-Inversion Closed-Loop: {qtype}", fontsize=14, y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{prefix}_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {plot_path}")


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Q-function inversion from P(k,τ)",
    )
    parser.add_argument(
        "--config", "-c",
        default="",
        help="Path to namelist config file (.nml)",
    )
    parser.add_argument(
        "--mode",
        choices=["closed-loop", "file"],
        default="closed-loop",
        help="Run mode (default: closed-loop)",
    )
    parser.add_argument(
        "--qtype",
        default="coherent",
        help="Test Q distribution: coherent, bsv, thermal, gaussian_mixture",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Output directory",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        print(f"Loading config: {args.config}")
        config = QInvConfig.from_namelist(args.config)
    else:
        print("Using default config")
        config = QInvConfig.default()

    # Override output dir from CLI
    config.io.output_dir = args.output_dir

    if args.mode == "closed-loop":
        run_closed_loop(config, args.qtype)
    elif args.mode == "file":
        print("File mode not yet implemented. Use --mode closed-loop for testing.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
