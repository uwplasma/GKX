"""
Thin wrappers to run either 'fourier' or 'dg' pipelines, returning (ts, state, diagnostics).
"""

from typing import Dict, Any, Tuple
import jax.numpy as jnp
from io_config import GridCfg 
from fourier import run_bank
from dg import assemble_A_real, initial_condition, solve_dg

def resolve_kgrid(grid: GridCfg, *, only_positive: bool = False) -> np.ndarray:
    """
    Build the Fourier wavenumber array from either:
      A) uniform spec: kmin, kmax, Nk
      B) periodic box: L, Nx  (uses FFT frequencies 2π/L * fftfreq(Nx, d=L/Nx))
    If both specs are present, A takes precedence.
    """
    # A) explicit uniform k-grid
    if grid.kmin is not None and grid.kmax is not None and grid.Nk is not None:
        kmin, kmax = float(grid.kmin), float(grid.kmax)
        Nk = int(grid.Nk)
        if Nk < 1:
            raise ValueError("Nk must be >= 1.")
        kvals = np.linspace(kmin, kmax, Nk, dtype=float)
        if only_positive:
            kvals = kvals[kvals >= 0]
        return kvals

    # B) derive from periodic box
    if grid.L is not None and grid.Nx is not None:
        L = float(grid.L)
        Nx = int(grid.Nx)
        if Nx < 1:
            raise ValueError("Nx must be >= 1.")
        dx = L / Nx
        # Standard FFT convention: wavenumbers in cycles per length -> multiply by 2π
        k = 2 * jnp.pi * np.fft.fftfreq(Nx, d=dx)
        # Optional: sort to increasing k if you prefer, or keep FFT order
        k = np.sort(k)
        if only_positive:
            k = k[k >= 0]
        return k

    # If we got here, the config is incomplete
    raise ValueError(
        "Fourier grid is underspecified. Provide either "
        "(kmin, kmax, Nk) for a uniform k-grid, or (L, Nx) to derive k from a periodic box."
    )

def run_fourier(cfg) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # kmin, kmax, Nk = cfg.grid.kmin, cfg.grid.kmax, cfg.grid.Nk
    # kvals = np.linspace(kmin, kmax, Nk)
    kvals = resolve_kgrid(cfg.grid, only_positive=False)  # or True if you want k ≥ 0
    ts, C_knt, Ek_kt = run_bank(kvals, cfg.hermite.N, cfg.hermite.nu0, cfg.hermite.hyper_p,
                                cfg.hermite.collide_cutoff, cfg.sim.backend,
                                cfg.sim.tmax, cfg.sim.nt,
                                cfg.init.amplitude, cfg.init.seed_c1)
    diag = {"k": kvals, "C_knt": C_knt, "Ek_kt": Ek_kt}
    return ts, diag

def run_dg(cfg) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    Nx, L = cfg.grid.Nx, cfg.grid.L
    N = cfg.hermite.N
    A_real, P = assemble_A_real(Nx, L, N,
                                cfg.hermite.nu0, cfg.hermite.hyper_p, cfg.hermite.collide_cutoff,
                                cfg.bc.kind)
    C0 = initial_condition(Nx, L, N, cfg.init.type, cfg.init.amplitude, cfg.init.k,
                           cfg.init.shift, cfg.init.seed_c1)
    ts, C_t = solve_dg(A_real, C0, cfg.sim.tmax, cfg.sim.nt, cfg.sim.backend)
    # Diagnostics: field from c0 via P
    E_xt = np.einsum("ij,njt->it", P, C_t[0:1, :, :])  # or P @ c0(t) for each t
    diag = {"C_t": C_t, "E_xt": E_xt}
    return ts, diag
