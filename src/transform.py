"""
1D Fourier transform + deblur + cutoff for Q-function inversion.

Core operation (per τ):
    P̃(ω_k, τ) = FFT_k[P̃(k, τ)]                     [1D FFT]
    Q̂(ω_k κ(τ)) = exp(+i ω_k k0) exp(+ω_k² σ_k²/2) P̃ [deblur]
    Truncate at |ω_k| > ω_k^max                       [cutoff]

The output is a set of radial samples (ξ_R, ξ_I, Q̂_value) in Fourier space.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from dataclasses import dataclass


@dataclass
class FourierSamples:
    """Collection of Fourier-space radial samples.

    Attributes:
        xi_R, xi_I: Fourier-space coordinates [N_tau * N_k_valid]
        q_hat: Q̂ values at those coordinates
        omega_k: frequency values [N_k_valid]
    """
    xi_R: NDArray   # [N_samples]
    xi_I: NDArray   # [N_samples]
    q_hat: NDArray  # [N_samples]
    omega_k: NDArray  # [N_k]


def compute_frequency_axis(
    n_k: int, dk: float
) -> Tuple[NDArray, NDArray]:
    """Compute ω_k axis for 1D FFT.

    Args:
        n_k: number of k points
        dk: k spacing

    Returns:
        omega_k: frequency values [N_k], FFT-ordered (0, pos, neg)
        omega_k_full: frequencies [N_k] in natural order (-Nyq to +Nyq)
    """
    # FFT frequencies: 0, 1, ..., N/2-1, -N/2, ..., -1
    omega = 2.0 * np.pi * np.fft.fftfreq(n_k, dk)
    return omega


def transform_single_tau(
    p_tilde_tau: NDArray,   # [N_k]
    tau: float,
    kv_tau: NDArray,         # [2]
    omega_k: NDArray,        # [N_k]
    k0: float,
    k_min: float,
    dk: float,
    sigma_k: float,
    omega_k_max: float,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Perform 1D FFT + deblur for one τ value.

    Correct discrete implementation of:
        Q̂(ω_ℓ·κ(τ)) = dk · exp(+i ω_ℓ(k0−kmin)) · exp(+ω_ℓ²σ_k²/2) · FFT[P̃]_ℓ

    where ω_ℓ = 2πℓ/(N_k·dk) (FFT ordering: 0, pos, neg).

    Args:
        p_tilde_tau: P̃(k, τ) = P(k,τ)/J for this τ [N_k]
        tau: current τ value
        kv_tau: κ(τ) vector [2]
        omega_k: frequency axis [N_k] (FFT-ordered)
        k0: field-free momentum
        k_min: minimum k value in grid
        dk: k grid spacing
        sigma_k: momentum broadening
        omega_k_max: cutoff frequency

    Returns:
        xi_R: ξ_R coordinates for valid samples [N_valid]
        xi_I: ξ_I coordinates [N_valid]
        q_hat: Q̂ values [N_valid]
    """
    n_k = len(p_tilde_tau)

    # 1D FFT (without dk factor yet)
    p_fft = np.fft.fft(p_tilde_tau)  # [N_k]

    # Apply cutoff mask FIRST to avoid overflow in deblur
    mask = np.abs(omega_k) <= omega_k_max
    valid_idx = np.where(mask)[0]

    omega_valid = omega_k[valid_idx]
    p_fft_valid = p_fft[valid_idx]

    # Deblur:
    #   Q̂ = dk · exp(+i ω(k0−kmin)) · exp(+ω²σ²/2) · FFT[P̃]
    # Phase from k-grid offset: need exp(-i ω·k_min) from the FFT
    # and exp(+i ω·k0) from the deblur → combined exp(+i ω(k0−kmin))
    phase_fft = np.exp(-1j * omega_valid * k_min)  # FFT: P̃ = dk·FFT·exp(-iω kmin)
    phase_deblur = np.exp(+1j * omega_valid * k0)    # deblur: exp(+i ω k0)
    deblur = np.exp(0.5 * (omega_valid * sigma_k) ** 2)

    q_hat_valid = dk * phase_fft * phase_deblur * deblur * p_fft_valid

    # Map to 2D Fourier coordinates: ξ = ω_k · κ(τ)
    xi_R = omega_valid * kv_tau[0]
    xi_I = omega_valid * kv_tau[1]

    return xi_R, xi_I, q_hat_valid

    return xi_R, xi_I, q_hat_valid


def transform_all_tau(
    p_tilde: NDArray,          # [N_tau, N_k]
    tau_array: NDArray,        # [N_tau]
    kappa_vec: NDArray,        # [N_tau, 2]
    k0: float,
    k_min: float,
    dk: float,
    sigma_k: float,
    omega_k_max: float,
) -> FourierSamples:
    """Full 1D FFT + deblur for all τ values.

    Args:
        p_tilde: P̃(k,τ) [N_tau, N_k]
        tau_array: τ values [N_tau]
        kappa_vec: κ(τ) vectors [N_tau, 2]
        k0, k_min, dk: k grid parameters
        sigma_k, omega_k_max: kernel params

    Returns:
        FourierSamples with all radial samples
    """
    n_k = p_tilde.shape[1]
    omega_k = compute_frequency_axis(n_k, dk)

    xi_R_list, xi_I_list, q_hat_list = [], [], []

    for j in range(len(tau_array)):
        xi_r, xi_i, qh = transform_single_tau(
            p_tilde[j], tau_array[j], kappa_vec[j],
            omega_k, k0, k_min, dk, sigma_k, omega_k_max,
        )
        xi_R_list.append(xi_r)
        xi_I_list.append(xi_i)
        q_hat_list.append(qh)

    xi_R = np.concatenate(xi_R_list)
    xi_I = np.concatenate(xi_I_list)
    q_hat = np.concatenate(q_hat_list)

    return FourierSamples(
        xi_R=xi_R, xi_I=xi_I, q_hat=q_hat,
        omega_k=omega_k,
    )
