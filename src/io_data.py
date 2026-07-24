"""
Data I/O for Q-inversion project.

Handles:
- Reading P(k,τ) from v5.1 9-column format
- Reading P(k,τ) from compact matrix format
- Writing Q(α) output files
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
import os


def read_v51_probability(
    path: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Read v5.1 prj_probability_tau.dat or prj_tau_k_summary.dat.

    Format (9-column)::
        itau  tau  ik  k  itheta  theta  iphi  phi  probability

    Returns:
        tau_array: unique τ values [N_tau]
        k_array: unique k values [N_k]
        p_matrix: P(k,τ) [N_tau, N_k], angle-integrated
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    data = np.loadtxt(path, comments="#")

    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Columns: itau(0), tau(1), ik(2), k(3), itheta(4), theta(5),
    #          iphi(6), phi(7), prob(8)
    tau_vals = data[:, 1]
    k_vals = data[:, 3]
    prob_vals = data[:, 8]

    tau_array = np.unique(tau_vals)
    k_array = np.unique(k_vals)
    n_tau, n_k = len(tau_array), len(k_array)

    # Build matrix: sum over theta,phi for each (tau, k)
    p_matrix = np.zeros((n_tau, n_k))

    # Map values to indices
    tau_to_idx = {t: i for i, t in enumerate(tau_array)}
    k_to_idx = {k: i for i, k in enumerate(k_array)}

    for i in range(data.shape[0]):
        itau = tau_to_idx[tau_vals[i]]
        ik = k_to_idx[k_vals[i]]
        p_matrix[itau, ik] += prob_vals[i]

    return tau_array, k_array, p_matrix


def read_compact_matrix(
    path: str,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Read P(k,τ) from compact ASCII matrix format.

    First row: k grid values
    First column: τ grid values
    Matrix: P(k,τ) values

    Returns:
        tau_array, k_array, p_matrix
    """
    data = np.loadtxt(path, comments="#")
    k_array = data[0, 1:]
    tau_array = data[1:, 0]
    p_matrix = data[1:, 1:]
    return tau_array, k_array, p_matrix


def write_q_function(
    path: str,
    q_grid: NDArray,
    alpha_min: float,
    alpha_max: float,
    description: str = "",
) -> None:
    """Write reconstructed Q(α) to ASCII file.

    Format::
        # Q-function reconstruction
        # alpha_R_min  alpha_R_max  n_alpha_R
        # alpha_I_min  alpha_I_max  n_alpha_I
        # <description>
        # alpha_R  alpha_I  Q(alpha)
    """
    n_a = q_grid.shape[0]
    alpha_1d = np.linspace(alpha_min, alpha_max, n_a)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Q-function reconstruction\n")
        f.write(f"# {alpha_min:.6f}  {alpha_max:.6f}  {n_a}\n")
        f.write(f"# {alpha_min:.6f}  {alpha_max:.6f}  {n_a}\n")
        if description:
            f.write(f"# {description}\n")
        f.write("# alpha_R  alpha_I  Q(alpha_R, alpha_I)\n")

        for i in range(n_a):
            for j in range(n_a):
                f.write(f"  {alpha_1d[i]:.6f}  {alpha_1d[j]:.6f}"
                        f"  {q_grid[i, j]:.12e}\n")


def write_p_comparison(
    path: str,
    tau_array: NDArray,
    k_array: NDArray,
    p_input: NDArray,
    p_reconstructed: NDArray,
) -> None:
    """Write input vs reconstructed P(k,τ) for diagnostics.

    Format::
        # tau  k  P_input  P_recon  residual
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# tau  k  P_input  P_recon  residual\n")
        for j, tau in enumerate(tau_array):
            for m, k_val in enumerate(k_array):
                f.write(f"  {tau:.6f}  {k_val:.6f}"
                        f"  {p_input[j, m]:.12e}"
                        f"  {p_reconstructed[j, m]:.12e}"
                        f"  {p_input[j, m] - p_reconstructed[j, m]:.12e}\n")
