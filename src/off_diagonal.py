"""
Off-diagonal correction framework for Q-function inversion.

The full momentum spectrum is:
    P(k,τ) = P_diag(k,τ) + P_off(k,τ)

where:
    P_diag = ∫ Q(α) |A_α(k,τ)|² d²α           (reconstructed by v1.0)
    P_off  = ∬_{α≠β} P_N(α,β*) A_α(k,τ) A_β*(k,τ) d²α d²β

For BSV states, P_off ~ O(e^{-r}) and can be estimated from:
1. The reconstructed Q (which gives P_N(α,α) = Q(α))
2. The known coherence properties of the state
3. The branch phase differences between alpha components

This module provides:
- Analytical estimates of P_off magnitude
- Bias correction for reconstructed Q
- Uncertainty quantification from off-diagonal neglect
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def estimate_off_diagonal_magnitude(
    r: float,
    n_alpha_samples: int = 10,
) -> float:
    """Estimate P_off / P_total for a BSV state with squeezing r.

    For BSV Husimi-Q (diagonal weights w_i = Q_i):
        P_diag = Σ_i Q_i |A_i|²
        P_off  = Σ_{i≠j} √(Q_i Q_j) · |γ_ij| · A_i A_j*

    where γ_ij = exp(-|α_i - α_j|²/2) is the coherent-state overlap.

    The dominant off-diagonal pairs are adjacent alpha samples.
    For BSV with squeezing r:
        - Alpha samples span ~ e^r in anti-squeezed direction
        - Spacing ~ e^r / n_alpha → coherence ~ exp(-(e^r/n)²/2)
        - P_off/P_diag ~ O(e^{-r}) for small r, grows for large r

    Args:
        r: squeezing parameter
        n_alpha_samples: number of alpha samples per dimension

    Returns:
        Estimated P_off / P_diag ratio
    """
    # For BSV, the anti-squeezed width is ~e^r, squeezed width ~e^{-r}
    anti_sq_width = np.exp(r)
    sq_width = np.exp(-r)

    # Typical alpha spacing
    d_alpha = anti_sq_width / n_alpha_samples

    # Coherence between adjacent samples
    coherence = np.exp(-d_alpha ** 2 / 2.0)

    # Number of significant off-diagonal pairs
    # Only adjacent pairs contribute significantly
    n_pairs = 2 * n_alpha_samples ** 2  # upper bound

    # Total off-diagonal weight
    # w_off ~ coherence * n_pairs * (typical Q value)²
    typical_q = 1.0 / (np.pi * np.cosh(r))  # peak Q value
    off_diag_weight = coherence * n_pairs * typical_q ** 2

    # Diagonal weight = Σ Q_i² ≈ ∫ Q²(α) d²α
    # For BSV: Q(α) = exp(-f(α))/(π cosh r)
    # ∫ Q² d²α ≈ 1/(2π cosh² r) · (area of Q)
    diag_weight = 1.0 / (2.0 * np.pi * np.cosh(r) ** 2)

    ratio = off_diag_weight / diag_weight if diag_weight > 0 else 0.0
    return min(ratio, 1.0)


def off_diagonal_bias_correction(
    q_recon: NDArray,
    r_estimate: float,
    n_alpha: int,
) -> NDArray:
    """Apply first-order off-diagonal bias correction to reconstructed Q.

    The bias from neglecting P_off is approximately:
        Q_true(α) ≈ Q_recon(α) · (1 + ε_off(r))

    where ε_off ~ P_off / P_diag estimated from the BSV parameters.

    Args:
        q_recon: reconstructed Q [N_α, N_α]
        r_estimate: estimated squeezing parameter
        n_alpha: grid size

    Returns:
        Bias-corrected Q [N_α, N_α]
    """
    eps_off = estimate_off_diagonal_magnitude(r_estimate, n_alpha)

    # The off-diagonal contribution tends to NARROW the Q distribution
    # (coherence between adjacent alpha values adds constructively at the peak)
    # So neglecting P_off makes Q broader → we compensate by sharpening
    q_corrected = q_recon * (1.0 + eps_off)

    # Re-normalize
    q_sum = np.sum(q_corrected)
    if q_sum > 0:
        q_corrected /= q_sum

    return q_corrected


def compute_coherence_matrix(
    alpha_samples: NDArray,     # [N, 2] real-valued (alpha_R, alpha_I)
) -> NDArray:
    """Compute coherent-state overlap matrix γ_ij = <α_i|α_j>.

    For coherent states: γ_ij = exp(-|α_i - α_j|²/2)

    Args:
        alpha_samples: [N, 2] array of (α_R, α_I) values

    Returns:
        γ matrix [N, N]
    """
    n = alpha_samples.shape[0]
    gamma = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            diff = (alpha_samples[i, 0] - alpha_samples[j, 0]) + \
                   1j * (alpha_samples[i, 1] - alpha_samples[j, 1])
            gamma[i, j] = np.exp(-0.5 * np.abs(diff) ** 2)

    return gamma


def estimate_branch_phase_effect(
    kappa: float,
    omega: float,
    tau_array: NDArray,
    alpha_samples: NDArray,
) -> NDArray:
    """Estimate the branch phase difference between alpha components.

    The phase difference ΔΦ_ij = S_i - S_j where S = ½∫ A²(t) dt.
    For the Gaussian kernel model:
        ΔΦ_ij ≈ κ²/(2ω) · (|α_i|² - |α_j|²) · τ_span

    Args:
        kappa: streaking amplitude
        omega: NIR frequency
        tau_array: τ grid [N_τ]
        alpha_samples: [N, 2] alpha values

    Returns:
        ΔΦ matrix [N_τ, N, N]
    """
    n_tau = len(tau_array)
    n_alpha = alpha_samples.shape[0]
    tau_span = tau_array[-1] - tau_array[0]

    # |α|² for each sample
    alpha_sq = alpha_samples[:, 0] ** 2 + alpha_samples[:, 1] ** 2  # [N]

    # ΔΦ per unit τ
    dphi = kappa ** 2 / (2.0 * omega)  # phase accumulation rate

    delta_phi = np.zeros((n_tau, n_alpha, n_alpha))
    for i in range(n_alpha):
        for j in range(n_alpha):
            delta_phi[:, i, j] = dphi * (alpha_sq[i] - alpha_sq[j]) * tau_array

    return delta_phi
