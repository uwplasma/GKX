# backends.py
"""
Thin wrappers to run either 'fourier' or 'dg' pipelines.
Always return (ts, diag_dict) with keys used by the unified plots.
"""

from typing import Dict, Tuple
import jax.numpy as jnp
from io_config import GridCfg
from fourier import run_bank_multispecies
from dg import (
    assemble_A_real_multispecies,
    initial_condition_multispecies,
    solve_dg_multispecies,
)

def resolve_kgrid(grid: GridCfg, *, only_positive: bool = False) -> jnp.ndarray:
    """
    Build k-grid from the periodic box: L, Nx (FFT frequencies).
    """
    if grid.L is None or grid.Nx is None:
        raise ValueError("Provide (L, Nx) in [grid].")
    L  = jnp.asarray(grid.L, dtype=jnp.float64)
    Nx = int(grid.Nx)
    dx = L / Nx
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)
    k = jnp.sort(k)
    if only_positive:
        k = k[k >= 0]
    return k

def run_fourier(cfg):
    kvals = resolve_kgrid(cfg.grid, only_positive=False)
    # ALWAYS multispecies
    ts, C_kSnt, Ek_kt = run_bank_multispecies(
        kvals, cfg.hermite.N, cfg.species,
        cfg.sim.backend, cfg.sim.tmax, cfg.sim.nt,
    )
    return ts, {"k": kvals, "C_kSnt": C_kSnt, "Ek_kt": Ek_kt}

def run_dg(cfg) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    Nx, L = cfg.grid.Nx, cfg.grid.L
    N = cfg.hermite.N

    # Operator (same for all species; per-species params embedded in assembly)
    A_real, P = assemble_A_real_multispecies(
        Nx, L, N, cfg.species, cfg.bc.kind
    )

    # Per-species initial condition from species.amplitude & species.k
    C0_Snx = initial_condition_multispecies(
        Nx, L, N, cfg.species,          # per-species (amp, k)
        seed_c1=False                   # each species can have its own c1 seeding if you add it; global removed
    )

    # Evolve each species with same A_real
    ts, C_St = solve_dg_multispecies(
        A_real, C0_Snx, cfg.sim.tmax, cfg.sim.nt, cfg.sim.backend
    )  # (S, N, Nx, nt)

    # Species-summed diagnostics
    C_t  = C_St.sum(axis=0)                              # (N, Nx, nt)
    E_xt = jnp.einsum("ij,njt->it", P, C_t[0:1, :, :])   # (Nx, nt)
    x    = jnp.linspace(0.0, L, Nx, dtype=jnp.float64)
    return ts, {"C_t": C_t, "C_St": C_St, "E_xt": E_xt, "x": x}
