"""
Analytic forward model: P(k,τ) = ∫ Q(α) K(k,τ|α) d²α

Implements the Gaussian-kernel forward model derived from SFA + saddle-point
approximation. Used for:
1. Closed-loop validation: known Q → synthetic P → inversion → compare
2. Forward prediction: evaluate model P for a given Q

The kernel is:
    K(k,τ|α) = 1/(√(2π) σ_k) · exp[-(k - k0 + κ(τ)·α)² / (2 σ_k²)]

where κ(τ) = κ · f(τ) · (sin ωτ, cos ωτ)^T
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional


# ═════════════════════════════════════════════════════════════════════════
# Kernel helpers
# ═════════════════════════════════════════════════════════════════════════

def nir_envelope(tau: NDArray, omega: float, nir_cycles: float) -> NDArray:
    """sin² NIR envelope: f(τ) = sin²(π·τ / (nir_cycles·T_nir)).

    Args:
        tau: delay array [N_tau], a.u.
        omega: NIR angular frequency, a.u.
        nir_cycles: number of NIR cycles in the envelope

    Returns:
        envelope values [N_tau]
    """
    t_nir = 2.0 * np.pi / omega
    tau_span = nir_cycles * t_nir
    # Guard against division by zero at endpoints
    phase = np.pi * tau / tau_span
    phase = np.clip(phase, 0.0, np.pi)
    return np.sin(phase) ** 2


def kappa_vector(
    tau: NDArray, kappa: float, omega: float, nir_cycles: float
) -> NDArray:
    """Streaking direction vectors κ(τ) for each delay.

    Args:
        tau: delay array [N_tau], a.u.
        kappa: streaking amplitude κ = 2E_N/ω, a.u.
        omega: NIR angular frequency, a.u.
        nir_cycles: NIR envelope cycles

    Returns:
        array shape [N_tau, 2]: components (κ_R, κ_I) in (α_R, α_I) plane
    """
    env = nir_envelope(tau, omega, nir_cycles)
    direction = np.column_stack([
        np.sin(omega * tau),
        np.cos(omega * tau),
    ])  # [N_tau, 2]
    return kappa * env[:, np.newaxis] * direction


def gaussian_kernel(
    k: NDArray,
    tau_idx: int,
    alpha_grid_R: NDArray,
    alpha_grid_I: NDArray,
    kappa_vec: NDArray,
    k0: float,
    sigma_k: float,
) -> NDArray:
    """Evaluate K(k, τ_j | α_pq) for all k at one τ.

    Args:
        k: momentum grid [N_k]
        tau_idx: index into kappa_vec
        alpha_grid_R: α_R meshgrid [N_alpha, N_alpha]
        alpha_grid_I: α_I meshgrid [N_alpha, N_alpha]
        kappa_vec: κ(τ) array [N_tau, 2]
        k0: field-free momentum
        sigma_k: momentum broadening

    Returns:
        kernel values [N_k, N_alpha, N_alpha]
    """
    kv = kappa_vec[tau_idx]  # [2]
    # dot product: κ_R·α_R + κ_I·α_I  → [N_alpha, N_alpha]
    dot = kv[0] * alpha_grid_R + kv[1] * alpha_grid_I
    # k_center = k0 - κ·α  → shape: broadcast [N_k, N_a, N_a]
    k_center = k0 - dot[np.newaxis, :, :]  # [1, N_a, N_a]
    k_expanded = k[:, np.newaxis, np.newaxis]  # [N_k, 1, 1]
    # Gaussian
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_k)
    exponent = -((k_expanded - k_center) ** 2) / (2.0 * sigma_k ** 2)
    return prefactor * np.exp(exponent)


# ═════════════════════════════════════════════════════════════════════════
# Forward pass
# ═════════════════════════════════════════════════════════════════════════

def forward_pass(
    q_grid: NDArray,
    alpha_min: float,
    alpha_max: float,
    tau_array: NDArray,
    k_array: NDArray,
    kappa: float,
    omega: float,
    nir_cycles: float,
    k0: float,
    sigma_k: float,
) -> NDArray:
    """Compute P(k,τ) from Q(α) via discretized integral.

    P(k_m, τ_j) = Σ_{p,q} Q(α_p, α_q) · K(k_m, τ_j | α_p, α_q) · Δα²

    Args:
        q_grid: Q values [N_α, N_α]
        alpha_min, alpha_max: α grid limits
        tau_array: τ grid [N_τ]
        k_array: k grid [N_k]
        kappa, omega, nir_cycles: kernel vector params
        k0: field-free momentum
        sigma_k: momentum broadening

    Returns:
        P(k,τ) array [N_τ, N_k]
    """
    n_alpha = q_grid.shape[0]
    n_tau = len(tau_array)
    n_k = len(k_array)

    # Alpha grid
    alpha_1d = np.linspace(alpha_min, alpha_max, n_alpha)
    d_alpha = alpha_1d[1] - alpha_1d[0]
    alpha_R, alpha_I = np.meshgrid(alpha_1d, alpha_1d, indexing="ij")

    # κ vectors
    kv = kappa_vector(tau_array, kappa, omega, nir_cycles)  # [N_tau, 2]

    # Output
    p_result = np.zeros((n_tau, n_k))

    for j in range(n_tau):
        K = gaussian_kernel(k_array, j, alpha_R, alpha_I, kv, k0, sigma_k)
        # Integrate over α: Σ K·Q·Δα²
        integral = np.sum(K * q_grid[np.newaxis, :, :], axis=(1, 2))  # [N_k]
        p_result[j, :] = integral * d_alpha ** 2

    return p_result


# ═════════════════════════════════════════════════════════════════════════
# Test Q distributions (for closed-loop validation)
# ═════════════════════════════════════════════════════════════════════════

def generate_test_q(
    n_alpha: int,
    alpha_min: float,
    alpha_max: float,
    qtype: str = "coherent",
    **kwargs,
) -> NDArray:
    """Generate known test Q(α) distributions.

    Args:
        n_alpha: grid size per side
        alpha_min, alpha_max: grid limits
        qtype: distribution type
            - 'coherent': single coherent state at α0
            - 'bsv': Bright Squeezed Vacuum Husimi-Q
            - 'gaussian_mixture': sum of Gaussians
            - 'thermal': thermal state Q(α) ∝ exp(-|α|²/(n̄+1))
        **kwargs: distribution-specific parameters
            - alpha0 (complex): for 'coherent', default 3+0j
            - r, theta (float): for 'bsv', squeezing params, default r=2, theta=0
            - centers, widths (list): for 'gaussian_mixture'

    Returns:
        Q array [n_alpha, n_alpha], normalized so ∫Q d²α = 1
    """
    alpha_1d = np.linspace(alpha_min, alpha_max, n_alpha)
    d_alpha = alpha_1d[1] - alpha_1d[0]
    alpha_R, alpha_I = np.meshgrid(alpha_1d, alpha_1d, indexing="ij")
    alpha_complex = alpha_R + 1j * alpha_I  # [N_a, N_a]

    if qtype == "coherent":
        # Q(α) = (1/π) exp(-|α - α0|²)   [Husimi Q of coherent state]
        alpha0 = kwargs.get("alpha0", 3.0 + 0.0j)
        q = (1.0 / np.pi) * np.exp(-np.abs(alpha_complex - alpha0) ** 2)

    elif qtype == "bsv":
        # BSV Husimi Q:
        # Q(α) = (1/(π cosh r)) exp[-|α|² - tanh r · Re(e^{-iθ} α²)]
        r = kwargs.get("r", 2.0)
        theta = kwargs.get("theta", 0.0)
        abs2 = np.abs(alpha_complex) ** 2
        alpha_sq = alpha_complex ** 2
        phase = np.exp(-1j * theta) * alpha_sq
        q = (1.0 / (np.pi * np.cosh(r))) * np.exp(
            -abs2 - np.tanh(r) * np.real(phase)
        )

    elif qtype == "thermal":
        # Q(α) = (1/(π(n̄+1))) exp(-|α|²/(n̄+1))
        nbar = kwargs.get("nbar", 1.0)
        q = (1.0 / (np.pi * (nbar + 1.0))) * np.exp(
            -np.abs(alpha_complex) ** 2 / (nbar + 1.0)
        )

    elif qtype == "gaussian_mixture":
        centers = kwargs.get("centers", [3.0 + 0j, -3.0 + 0j])
        widths = kwargs.get("widths", [0.5, 0.5])
        q = np.zeros_like(alpha_R, dtype=float)
        for c, w in zip(centers, widths):
            q += (1.0 / (np.pi * w ** 2)) * np.exp(
                -np.abs(alpha_complex - c) ** 2 / w ** 2
            )
        q /= len(centers)

    else:
        raise ValueError(f"Unknown qtype: {qtype}")

    # Normalize: ∫ Q d²α = 1
    q_sum = np.sum(q) * d_alpha ** 2
    q /= q_sum

    return q


# ═════════════════════════════════════════════════════════════════════════
# Quick test
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Quick smoke test: coherent state Q → P(k,τ)
    n_a = 64
    a_min, a_max = -8.0, 8.0
    q_true = generate_test_q(n_a, a_min, a_max, "coherent", alpha0=3.0 + 0j)

    tau = np.linspace(0, 110.0, 32)
    k = np.linspace(0.2, 2.0, 128)

    p = forward_pass(
        q_true, a_min, a_max, tau, k,
        kappa=1.4, omega=0.057, nir_cycles=4.0,
        k0=1.0, sigma_k=0.08,
    )

    print(f"P shape: {p.shape}, range: [{p.min():.2e}, {p.max():.2e}]")
    print(f"Total prob per tau: {p.sum(axis=1)[:5]}")

    # Quick plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.pcolormesh(np.linspace(a_min, a_max, n_a),
                   np.linspace(a_min, a_max, n_a),
                   q_true.T, shading="auto")
    ax1.set_title("Q(alpha) — Coherent state at alpha=3")
    ax1.set_xlabel("alpha_R"); ax1.set_ylabel("alpha_I")

    ax2.pcolormesh(k, tau, p, shading="auto")
    ax2.set_title("P(k, tau) — synthetic")
    ax2.set_xlabel("k (a.u.)"); ax2.set_ylabel("tau (a.u.)")

    plt.tight_layout()
    plt.savefig("output/smoke_forward.png", dpi=150)
    print("Saved output/smoke_forward.png")
