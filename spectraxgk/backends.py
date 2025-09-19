# backends.py

from typing import Dict, Tuple
import jax.numpy as jnp
from io_config import GridCfg
from fourier import (
    run_bank_multispecies_linear,
    run_bank_multispecies_nonlinear,
)
from dg import (
    assemble_A_real_multispecies,
    initial_condition_multispecies,
    solve_dg_multispecies_linear,
    solve_dg_multispecies_nonlinear,
)

def resolve_kgrid(grid: GridCfg, *, only_positive: bool = False) -> jnp.ndarray:
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
    if cfg.sim.nonlinear:
        ts, C_kSnt, Ek_kt = run_bank_multispecies_nonlinear(
            kvals=kvals,
            N=cfg.hermite.N,
            species=cfg.species,
            tmax=cfg.sim.tmax,
            nt=cfg.sim.nt,
            backend=cfg.sim.backend,
            dealias_frac=cfg.sim.dealias_frac,
        )
    else:
        ts, C_kSnt, Ek_kt = run_bank_multispecies_linear(
            kvals, cfg.hermite.N, cfg.species,
            cfg.sim.backend, cfg.sim.tmax, cfg.sim.nt,
        )
    return ts, {"k": kvals, "C_kSnt": C_kSnt, "Ek_kt": Ek_kt}

def run_dg(cfg) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    Nx, L = cfg.grid.Nx, cfg.grid.L
    N = cfg.hermite.N
    A_real, P = assemble_A_real_multispecies(Nx, L, N, cfg.species, cfg.bc.kind)

    C0_Snx = initial_condition_multispecies(
        Nx, L, N, cfg.species, getattr(cfg.init, "seed_c1", False),
    )  # (S,N,Nx)

    if cfg.sim.nonlinear:
        ts, C_St = solve_dg_multispecies_nonlinear(
            A_real=A_real, P=P, species=cfg.species,
            C0_Snx=C0_Snx, tmax=cfg.sim.tmax, nt=cfg.sim.nt, backend=cfg.sim.backend
        )
    else:
        ts, C_St = solve_dg_multispecies_linear(
            A_real=A_real, C0_Snx=C0_Snx,
            tmax=cfg.sim.tmax, nt=cfg.sim.nt, backend=cfg.sim.backend
        )

    C_t  = C_St.sum(axis=0)                               # (N, Nx, nt)
    E_xt = jnp.einsum("ij,njt->it", P, C_t[0:1, :, :])    # (Nx, nt)
    x    = jnp.linspace(0.0, L, Nx, dtype=jnp.float64)
    return ts, {"C_t": C_t, "C_St": C_St, "E_xt": E_xt, "x": x}
