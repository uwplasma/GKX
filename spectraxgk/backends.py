# backends.py
"""
Thin wrappers to run either 'fourier' or 'dg' pipelines (multi-species only).
Always return (ts, diag_dict) with keys used by the unified plots.
"""

import jax.numpy as jnp

from spectraxgk.dg import (
    assemble_A_real_multispecies,
    initial_condition_multispecies,
    solve_dg_multispecies,
    solve_dg_multispecies_nonlinear,
)
from spectraxgk.fourier import (
    run_bank_multispecies_linear,
    run_bank_multispecies_nonlinear,
)
from spectraxgk.io_config import GridCfg


def resolve_kgrid(grid: GridCfg, *, only_positive: bool = False) -> jnp.ndarray:
    """
    Build k-grid from the periodic box: L, Nx (FFT frequencies).
    """
    if grid.L is None or grid.Nx is None:
        raise ValueError("Provide grid.L and grid.Nx in [grid].")
    L = jnp.asarray(grid.L, dtype=jnp.float64)
    Nx = int(grid.Nx)
    dx = L / Nx
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(Nx, d=dx)  # float64 with x64 enabled
    k = jnp.sort(k)
    if only_positive:
        k = k[k >= 0]
    return k


def run_fourier(cfg):
    """
    Multi-species Fourier–Hermite entry point.
    Returns:
      ts, {
        "k": kvals,                 # (Nk,)
        "C_kSnt": C_kSnt,           # (Nk, S, N, nt)
        "Ek_kt": Ek_kt,             # (Nk, nt)
      }
    """
    kvals = resolve_kgrid(cfg.grid, only_positive=False)
    model = getattr(cfg.sim, "model", "linear").lower()

    if model == "nonlinear":
        ts, C_kSnt, Ek_kt = run_bank_multispecies_nonlinear(
            kvals=kvals,
            N=cfg.hermite.N,
            species=cfg.species,
            tmax=cfg.sim.tmax,
            nt=cfg.sim.nt,
            backend=cfg.sim.backend,
        )
    else:
        # Linear multispecies
        ts, C_kSnt, Ek_kt = run_bank_multispecies_linear(
            kvals=kvals,
            N=cfg.hermite.N,
            species=cfg.species,
            backend=cfg.sim.backend,
            tmax=cfg.sim.tmax,
            nt=cfg.sim.nt,
        )

    return ts, {"k": kvals, "C_kSnt": C_kSnt, "Ek_kt": Ek_kt}


def run_dg(cfg) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Multi-species DG entry point.
    Returns:
      ts, {
        "x": x,                     # (Nx,)
        "E_xt": E_xt,               # (Nx, nt)
        "C_St": C_St,               # (S, N, Nx, nt)
        "C_t": C_sum,               # (N, Nx, nt) species-summed (for diagnostics)
      }
    """
    Nx = int(cfg.grid.Nx)
    L = float(cfg.grid.L)
    N = int(cfg.hermite.N)
    model = getattr(cfg.sim, "model", "linear").lower()

    # Assemble global linear operator & Poisson map
    # (Ensure your dg.assemble_A_real_multispecies returns: A_real, P, D, x)
    A_real, P, D, x = assemble_A_real_multispecies(
        Nx=Nx,
        L=L,
        N=N,
        species=cfg.species,
        bc_kind=cfg.bc.kind,
    )

    # Per-species initial conditions (cos(2π k_s x/L) with amp_s; optional seed_c1 per species)
    C0_Snx = initial_condition_multispecies(
        Nx=Nx,
        L=L,
        N=N,
        species=cfg.species,
    )  # (S, N, Nx)

    # Evolve
    if model == "nonlinear":
        ts, C_St = solve_dg_multispecies_nonlinear(
            A_real=A_real,
            P=P,
            D=D,
            x=x,
            C0_Snx=C0_Snx,
            species=cfg.species,
            tmax=cfg.sim.tmax,
            nt=cfg.sim.nt,
            backend=cfg.sim.backend,
        )  # (S, N, Nx, nt)
    else:
        ts, C_St = solve_dg_multispecies(
            A_real=A_real,
            C0_Snx=C0_Snx,
            tmax=cfg.sim.tmax,
            nt=cfg.sim.nt,
            backend=cfg.sim.backend,
        )  # (S, N, Nx, nt)

    # Species-summed coefficients (for phase-mix & energy diagnostics)
    C_sum = C_St.sum(axis=0)  # (N, Nx, nt)

    # Field: E = P @ (Σ_s q_s c0^{(s)})
    # c0 per species: (S, Nx, nt)
    c0_Sxt = C_St[:, 0, :, :]  # (S, Nx, nt)
    q_vec = jnp.asarray(
        [float(getattr(sp, "q", -1.0)) for sp in cfg.species], dtype=jnp.float64
    )  # (S,)
    # ρ(x,t) = Σ_s q_s c0^{(s)}(x,t)
    rho_xt = jnp.tensordot(q_vec, c0_Sxt, axes=(0, 0))  # (Nx, nt)
    E_xt = jnp.einsum("ij,jt->it", P, rho_xt)  # (Nx, nt)

    return ts, {"C_t": C_sum, "C_St": C_St, "E_xt": E_xt, "x": x}
