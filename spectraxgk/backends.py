# backends.py
"""
Thin wrappers to run either 'fourier' or 'dg' pipelines.
Always return (ts, diag_dict) with keys used by the unified plots.
"""

from typing import Dict, Any, Tuple
import jax.numpy as jnp
from io_config import GridCfg
from fourier import run_bank
from dg import assemble_A_real, initial_condition, solve_dg


def resolve_kgrid(grid: GridCfg, *, only_positive: bool = False) -> jnp.ndarray:
    """
    Build k-grid from either:
      A) uniform spec: kmin, kmax, Nk
      B) periodic box: L, Nx (FFT frequencies)
    A takes precedence if both present.
    """
    if grid.kmin is not None and grid.kmax is not None and grid.Nk is not None:
        kmin = jnp.asarray(grid.kmin, dtype=jnp.float64)
        kmax = jnp.asarray(grid.kmax, dtype=jnp.float64)
        Nk = int(grid.Nk)
        kvals = jnp.linspace(kmin, kmax, Nk, dtype=jnp.float64)
        if only_positive:
            kvals = kvals[kvals >= 0]
        return kvals

    if grid.L is not None and grid.Nx is not None:
        L = jnp.asarray(grid.L, dtype=jnp.float64)
        Nx = int(grid.Nx)
        dx = L / Nx
        k = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)  # float64 with x64 enabled
        k = jnp.sort(k)
        if only_positive:
            k = k[k >= 0]
        return k

    raise ValueError("Provide either (kmin,kmax,Nk) or (L,Nx) in [grid].")


def run_fourier(cfg) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    kvals = resolve_kgrid(cfg.grid, only_positive=False)
    ts, C_knt, Ek_kt = run_bank(
        kvals, cfg.hermite.N, cfg.hermite.nu0, cfg.hermite.hyper_p, cfg.hermite.collide_cutoff,
        cfg.sim.backend, cfg.sim.tmax, cfg.sim.nt, cfg.init.amplitude, cfg.init.seed_c1
    )
    return ts, {"k": kvals, "C_knt": C_knt, "Ek_kt": Ek_kt}


def run_dg(cfg) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    Nx, L = cfg.grid.Nx, cfg.grid.L
    N = cfg.hermite.N
    A_real, P = assemble_A_real(
        Nx, L, N,
        cfg.hermite.nu0, cfg.hermite.hyper_p, cfg.hermite.collide_cutoff,
        cfg.bc.kind
    )
    C0 = initial_condition(
        Nx, L, N, cfg.init.type, cfg.init.amplitude, cfg.init.k, cfg.init.shift, cfg.init.seed_c1
    )
    ts, C_t = solve_dg(A_real, C0, cfg.sim.tmax, cfg.sim.nt, cfg.sim.backend)  # (N,Nx,nt)
    # Field diagnostic (example projection)
    E_xt = jnp.einsum("ij,njt->it", P, C_t[0:1, :, :])    # shape (Nx, nt)
    return ts, {"C_t": C_t, "E_xt": E_xt, "x": jnp.linspace(0.0, L, Nx, dtype=jnp.float64)}
