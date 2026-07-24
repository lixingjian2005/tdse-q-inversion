"""
Iterative reconstruction refinement via gradient descent.

Algorithm (v2.0):
    1. Q_0 = v1.0 Fourier-slice reconstruction
    2. For k = 0, 1, ..., N_iter-1:
        a. Forward:  P_model = forward_pass(Q_k)
        b. Residual: R = P_input - P_model
        c. Gradient: ∇C = K^T · R   (adjoint of forward operator)
        d. Update:   Q_{k+1} = Q_k - λ_k · ∇C
        e. Enforce:  Q_{k+1} = max(Q_{k+1}, 0)
        f. Normalize: ∫ Q_{k+1} d²α = 1
    3. Return Q_N

The adjoint K^T is:
    (K^T · R)(α_pq) = Σ_{τ,k} K(k,τ|α_pq) · R(k,τ)

which is a direct weighted back-projection without any deblur/FFT.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Callable
import time


def compute_adjoint_fft(
    residual: NDArray,           # [N_tau, N_k]
    tau_array: NDArray,          # [N_tau]
    k_array: NDArray,            # [N_k]
    kappa_vec: NDArray,          # [N_tau, 2]
    k0: float,
    sigma_k: float,
    n_xi: int,
    xi_max: float,
    n_alpha: int,
    alpha_min: float,
    alpha_max: float,
    d_alpha: float,
    interp_method: str = 'bin',
) -> NDArray:
    """Compute adjoint via FFT-based back-projection WITHOUT deblur.

    Uses the Fourier-slice machinery WITHOUT the exp(+ω²σ²/2) factor.
    This is the adjoint (not inverse) of the forward operator, making it
    numerically stable and suitable for gradient computation.

    The adjoint transforms:
        R(k,τ) → FFT → Grid → IFFT → ΔQ
    skipping the deblur step that would amplify high-frequency noise.
    """
    from src.transform import transform_all_tau, compute_frequency_axis

    n_k = len(k_array)
    dk = k_array[1] - k_array[0]
    k_min = k_array[0]
    omega_k = compute_frequency_axis(n_k, dk)

    # FFT + mask (NO deblur)
    omega_k_max_adjoint = xi_max / np.max(np.abs(kappa_vec)) if np.max(np.abs(kappa_vec)) > 0 else xi_max

    samples_adjoint = transform_all_tau(
        residual, tau_array, kappa_vec,
        k0, k_min, dk, sigma_k, omega_k_max_adjoint * 1.5,
        deblur=False,
    )

    # Grid
    from src.gridding import grid_radial_to_cartesian
    q_hat_adjoint = grid_radial_to_cartesian(
        samples_adjoint.xi_R, samples_adjoint.xi_I,
        samples_adjoint.q_hat,
        n_xi, xi_max, method=interp_method,
    )

    # IFFT (no deblur correction needed for adjoint)
    from src.reconstruct import reconstruct_q
    grad = reconstruct_q(q_hat_adjoint, xi_max, n_alpha, alpha_min, alpha_max)

    return grad


def iterative_refine(
    q_init: NDArray,
    p_input: NDArray,
    forward_func: Callable[[NDArray], NDArray],
    tau_array: NDArray,
    k_array: NDArray,
    alpha_1d: NDArray,
    kappa_vec: NDArray,
    k0: float,
    sigma_k: float,
    n_iter: int = 20,
    adjoint_kwargs: Optional[dict] = None,
    step_size: float = 0.3,
    step_decay: float = 0.95,
    positivity: bool = True,
    verbose: bool = True,
    callback: Optional[Callable] = None,
) -> dict:
    """Iterative gradient-descent refinement of Q-function reconstruction.

    Minimizes ||P_input - forward(Q)||² subject to Q ≥ 0, ∫Q = 1.

    Args:
        q_init: Initial Q [N_α, N_α]
        p_input: P(k,τ) [N_τ, N_k]
        forward_func: Q → P
        tau_array, k_array: τ and k grids
        alpha_1d: α grid (1D)
        kappa_vec, k0, sigma_k: kernel params
        n_iter: max iterations
        step_size, step_decay: gradient step parameters
        positivity: enforce Q ≥ 0
        verbose: print progress

    Returns:
        dict with q_final, history, converged
    """
    q_current = q_init.copy()
    q_best = q_init.copy()
    best_residual = float('inf')
    history = []
    converged = False
    d_alpha = alpha_1d[1] - alpha_1d[0]

    t_start = time.perf_counter()

    for iteration in range(n_iter):
        # ── Forward pass ──
        p_model = forward_func(q_current)

        # ── Residual ──
        residual = p_input - p_model
        rel_residual = (np.sqrt(np.sum(residual ** 2)) /
                        np.sqrt(np.sum(p_input ** 2)))

        # ── Gradient via FFT adjoint (stable, no deblur amplification) ──
        if adjoint_kwargs is None:
            adjoint_kwargs = {}
        gradient = compute_adjoint_fft(
            residual, tau_array, k_array,
            kappa_vec, k0, sigma_k,
            **adjoint_kwargs,
        )

        # ── Metrics ──
        metrics = {
            'iteration': iteration,
            'rel_residual': rel_residual,
            'step_size': step_size * (step_decay ** iteration),
            'q_max': float(np.max(q_current)),
            'q_neg': float(np.sum(np.abs(q_current[q_current < 0]))),
            'grad_norm': float(np.sqrt(np.sum(gradient ** 2))),
        }
        history.append(metrics)

        if verbose:
            print(f"  iter {iteration:3d}: residual={rel_residual:.4e}, "
                  f"Q_max={metrics['q_max']:.4f}, "
                  f"lambda={metrics['step_size']:.4f}, "
                  f"|grad|={metrics['grad_norm']:.2e}")

        if callback:
            callback(iteration, q_current, metrics)

        # ── Track best ──
        if rel_residual < best_residual:
            best_residual = rel_residual
            q_best = q_current.copy()

        # ── Convergence check ──
        if iteration > 2 and rel_residual < 1e-6:
            converged = True
            if verbose:
                print(f"  Converged at iteration {iteration}")
            break

        # ── Divergence check ──
        if iteration > 2 and rel_residual > history[0]['rel_residual'] * 5:
            q_current = q_best
            if verbose:
                print(f"  Diverged at iter {iteration}, restored best "
                      f"(residual={best_residual:.4e})")
            break

        # ── Gradient update ──
        lam = step_size * (step_decay ** iteration)
        q_current = q_current - lam * gradient

        # ── Positivity constraint ──
        if positivity:
            q_current = np.maximum(q_current, 0.0)

        # ── Re-normalize: ∫Q d²α = 1 ──
        q_sum = np.sum(q_current) * d_alpha ** 2
        if q_sum > 0:
            q_current /= q_sum

    t_elapsed = time.perf_counter() - t_start
    if verbose:
        print(f"  Gradient descent: {len(history)} iters in {t_elapsed:.1f}s"
              f", best residual={best_residual:.4e}")

    return {
        'q_final': q_best,
        'history': history,
        'converged': converged,
        'timing': t_elapsed,
    }


def make_forward_projector(
    tau: NDArray,
    k: NDArray,
    kappa: float,
    omega: float,
    nir_cycles: float,
    k0: float,
    sigma_k: float,
    alpha_min: float,
    alpha_max: float,
):
    """Create a forward-projection function with fixed geometry."""
    from src.forward_model import forward_pass

    def forward_project(q_grid: NDArray) -> NDArray:
        return forward_pass(
            q_grid, alpha_min, alpha_max, tau, k,
            kappa, omega, nir_cycles, k0, sigma_k,
        )
    return forward_project
