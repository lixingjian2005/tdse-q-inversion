"""
Configuration system for Q-function inversion.

Namelist-style parameter management with three groups:
  &physics  — laser/atom/kernel parameters
  &numerics — FFT sizes, grid resolution, regularization
  &io       — input/output paths and formats

Usage:
    config = QInvConfig.from_namelist("qinv.nml")
    config.validate()
"""

from dataclasses import dataclass, field
import math
import os


@dataclass
class PhysicsParams:
    """Physical parameters for the streaking kernel.

    The kernel is:
        K(k, tau|alpha) = J / (sqrt(2π) σ_k) *
            exp[-(k - k0 + κ(tau)·α)² / (2 σ_k²)]

    where κ(tau) = kappa * f(tau) * (sin(ωτ), cos(ωτ))^T
    """

    nir_omega: float = 0.057
    """NIR photon energy (a.u.), 800 nm → 0.057"""

    nir_e0: float = 0.04
    """NIR single-photon field amplitude E_N (a.u.)"""

    kappa: float = -1.0
    """Streaking amplitude κ = 2E_N/ω. Auto-computed if negative."""

    k0: float = 1.0
    """Field-free photoelectron momentum (a.u.). k0 = sqrt(2(Ω_XUV - I_p))"""

    sigma_k: float = 0.1
    """Momentum broadening σ_k (a.u.), ~1/τ_X (XUV pulse duration)"""

    ip: float = 0.5
    """Ionization potential (a.u.), H 1s → 0.5"""

    use_diagonal: bool = True
    """Use diagonal approximation (ignore off-diagonal coherence)"""

    nir_cycles: float = 4.0
    """Number of NIR cycles for envelope f(τ) = sin²(πτ/(nir_cycles*T_nir))"""

    tau_min: float = 0.0
    """Minimum τ value (a.u.). Auto-computed from nir_cycles if negative."""

    tau_max: float = -1.0
    """Maximum τ value (a.u.). Auto-computed from nir_cycles if negative."""


@dataclass
class NumericsParams:
    """Numerical parameters for the inversion pipeline."""

    n_k: int = 256
    """Number of k points in P(k,τ) data (also 1D FFT size)"""

    n_tau: int = 256
    """Number of τ (delay) samples per NIR cycle"""

    n_tau_cycles: int = 1
    """Number of NIR cycles covered by τ scan"""

    n_xi: int = 128
    """Fourier-space Cartesian grid size per side"""

    n_alpha: int = 128
    """Real-space Q(α) grid size per side"""

    alpha_max: float = -1.0
    """Real-space half-width for Q(α) grid. Auto-computed from xi_max if negative."""

    omega_k_max: float = -1.0
    """Cutoff frequency (a.u.). Auto-computed from SNR if negative."""

    snr_p: float = 100.0
    """Estimated SNR of input P(k,τ)"""

    snr_q_target: float = 10.0
    """Target SNR of reconstructed Q"""

    interp_method: str = "bin"
    """Gridding interpolation: 'linear', 'rbf', 'nearest'"""

    regularize: str = "cutoff"
    """Regularization strategy: 'cutoff', 'tikhonov', 'none'"""

    tikhonov_lambda: float = 0.01
    """Tikhonov regularization parameter λ"""


@dataclass
class IOParams:
    """Input/output parameters."""

    input_file: str = ""
    """Path to input P(k,τ) data file"""

    input_format: str = "v5.1"
    """Input format: 'v5.1' (9-column), 'compact' (matrix), 'synthetic' (no file)"""

    output_dir: str = "./output"
    """Output directory"""

    output_prefix: str = "qinv"
    """Output filename prefix"""

    plot: bool = True
    """Auto-generate visualization plots"""


@dataclass
class QInvConfig:
    """Master configuration object.

    Holds all parameters and provides:
    - from_namelist(path): parse a .nml file
    - validate(): check parameter consistency
    - derive(): compute dependent parameters
    """

    physics: PhysicsParams = field(default_factory=PhysicsParams)
    numerics: NumericsParams = field(default_factory=NumericsParams)
    io: IOParams = field(default_factory=IOParams)

    # ── derived quantities (populated by derive()) ──

    kappa: float = 0.0
    """Effective streaking amplitude κ"""

    omega_k_max: float = 0.0
    """Applied cutoff frequency"""

    xi_max: float = 0.0
    """Fourier-space half-width ξ_max"""

    d_alpha: float = 0.0
    """Real-space grid spacing Δα"""

    alpha_grid: tuple = field(default_factory=lambda: (0.0, 0.0, 0))
    """(alpha_min, alpha_max, n_alpha) for both directions"""

    # ── factory methods ──

    @classmethod
    def from_namelist(cls, path: str) -> "QInvConfig":
        """Parse a Fortran-style namelist file.

        Format example::

            &physics
              nir_omega = 0.057
              sigma_k = 0.08
            /
            &numerics
              n_k = 256
            /
            &io
              output_dir = './my_run'
            /
        """
        config = cls()
        config._parse_file(path)
        config.validate()
        config.derive()
        return config

    @classmethod
    def default(cls) -> "QInvConfig":
        """Factory with sensible defaults (closed-loop test)."""
        config = cls()
        config.validate()
        config.derive()
        return config

    def _parse_file(self, path: str) -> None:
        """Parse namelist file into parameter groups."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Parse &physics group
        self.physics = self._parse_group(text, "physics", PhysicsParams())

        # Parse &numerics group
        self.numerics = self._parse_group(text, "numerics", NumericsParams())

        # Parse &io group
        self.io = self._parse_group(text, "io", IOParams())

    def _parse_group(self, text: str, group_name: str, defaults):
        """Extract key=value pairs from a namelist group block.

        Supports:
        - Fortran-style booleans: .true. / .false.
        - Quoted strings: 'value' or "value"
        - Integer, float, boolean, string types
        - Comments: lines starting with ! or #
        """
        import re

        # Find group block: &name ... / (terminator on its own line)
        pattern = rf"&{group_name}\s*\n(.*?)\n\s*/"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return defaults

        block = match.group(1)

        # Remove comments
        block = re.sub(r"[!#].*", "", block)

        # Parse key = value pairs
        field_types = {
            f.name: (f.type, f.default)
            for f in type(defaults).__dataclass_fields__.values()  # noqa: E501
        }

        result = type(defaults)()

        for line in block.split("\n"):
            # Match key = value
            kv = re.match(r"\s*(\w+)\s*=\s*(.+)", line)
            if not kv:
                continue
            key, raw_value = kv.group(1).strip(), kv.group(2).strip()

            # Strip trailing comma
            raw_value = raw_value.rstrip(",").strip()

            if key not in field_types:
                continue  # skip unknown keys

            parsed = self._parse_value(raw_value, field_types[key][0])
            setattr(result, key, parsed)

        return result

    @staticmethod
    def _parse_value(raw: str, ftype: type):
        """Parse a raw string value into the target type."""
        # Boolean
        if ftype is bool:
            raw_lower = raw.lower()
            if raw_lower in (".true.", "true", "t", ".t.", "yes", "y"):
                return True
            elif raw_lower in (".false.", "false", "f", ".f.", "no", "n"):
                return False
            raise ValueError(f"Cannot parse boolean: {raw}")

        # Integer
        if ftype is int:
            return int(float(raw))  # tolerate '1.0' for int fields

        # Float
        if ftype is float:
            return float(raw)

        # String — strip surrounding quotes
        if ftype is str:
            if (raw.startswith("'") and raw.endswith("'")) or \
               (raw.startswith('"') and raw.endswith('"')):
                return raw[1:-1]
            return raw

        return raw

    # ── validation ──

    def validate(self) -> None:
        """Validate parameter consistency. Raises ValueError on issues."""
        p, n, io = self.physics, self.numerics, self.io

        if p.nir_omega <= 0:
            raise ValueError("nir_omega must be positive")
        if p.nir_e0 < 0:
            raise ValueError("nir_e0 must be non-negative")
        if p.sigma_k <= 0:
            raise ValueError("sigma_k must be positive")
        if p.ip <= 0:
            raise ValueError("ip must be positive")
        if p.nir_cycles <= 0:
            raise ValueError("nir_cycles must be positive")

        if n.n_k < 8:
            raise ValueError("n_k must be >= 8")
        if n.n_tau < 4:
            raise ValueError("n_tau must be >= 4")
        if n.n_xi < 16:
            raise ValueError("n_xi must be >= 16")
        if n.n_alpha < 16:
            raise ValueError("n_alpha must be >= 16")
        if n.snr_p <= 0:
            raise ValueError("snr_p must be positive")
        if n.snr_q_target <= 0:
            raise ValueError("snr_q_target must be positive")

        if n.interp_method not in ("linear", "rbf", "nearest", "bin"):
            raise ValueError(f"Unknown interp_method: {n.interp_method}")
        if n.regularize not in ("cutoff", "tikhonov", "none"):
            raise ValueError(f"Unknown regularize: {n.regularize}")

        if io.input_format not in ("v5.1", "compact", "synthetic"):
            raise ValueError(f"Unknown input_format: {io.input_format}")

    # ── derived parameters ──

    def derive(self) -> None:
        """Compute derived (dependent) parameters from raw inputs."""
        p, n = self.physics, self.numerics

        # kappa
        if p.kappa < 0:
            self.kappa = 2.0 * p.nir_e0 / p.nir_omega
        else:
            self.kappa = p.kappa

        # tau range
        t_nir = 2.0 * math.pi / p.nir_omega
        tau_span = p.nir_cycles * t_nir
        if p.tau_min < 0:
            p.tau_min = 0.0
        if p.tau_max < 0:
            p.tau_max = tau_span

        # envelope max
        tau_span_actual = p.tau_max - p.tau_min
        # max of sin²(πτ/tau_span) = 1.0 at τ = tau_span/2
        f_max = 1.0 if p.nir_cycles <= 0 else 1.0

        # effective kappa considering envelope
        kappa_max = self.kappa * f_max

        # cutoff frequency
        if n.omega_k_max < 0:
            self.omega_k_max = self._compute_cutoff(
                n.snr_p, n.snr_q_target, p.sigma_k
            )
        else:
            self.omega_k_max = n.omega_k_max

        # Fourier-space extent
        self.xi_max = self.omega_k_max * kappa_max

        # Real-space Q grid
        if n.alpha_max > 0:
            # User-specified alpha range
            self.d_alpha = 2.0 * n.alpha_max / (n.n_alpha - 1) if n.n_alpha > 1 else 1.0
        else:
            # Δα = 2π / (2·ξ_max)  (Nyquist in real space)
            self.d_alpha = math.pi / self.xi_max if self.xi_max > 0 else 1.0

        half_width = n.n_alpha * self.d_alpha / 2.0
        self.alpha_grid = (-half_width, half_width, n.n_alpha)

    @staticmethod
    def _compute_cutoff(snr_p: float, snr_q_target: float, sigma_k: float) -> float:
        """Compute cutoff frequency from SNR formula.

        ω_k^max = (1/σ_k) * sqrt(ln(SNR_P / SNR_Q_target))

        For SNR_P=100, SNR_Q_target=10, σ_k=0.08:
            ω_k^max ≈ 12.5 * sqrt(ln(10)) ≈ 19 a.u.⁻¹
        """
        if snr_p <= snr_q_target:
            return 1.0 / sigma_k  # fallback: 1-sigma radius
        ratio = snr_p / snr_q_target
        return math.sqrt(math.log(ratio)) / sigma_k

    def summary(self) -> str:
        """One-line parameter summary."""
        p, n = self.physics, self.numerics
        return (
            f"κ={self.kappa:.3f} σ_k={p.sigma_k:.3f} k0={p.k0:.3f} "
            f"ω_k^max={self.omega_k_max:.1f} ξ_max={self.xi_max:.1f} "
            f"Δα={self.d_alpha:.3f} N_τ={n.n_tau} N_k={n.n_k}"
        )


# ── quick test ──
if __name__ == "__main__":
    cfg = QInvConfig.default()
    print(cfg.summary())
    print(f"  alpha grid: {cfg.alpha_grid}")
