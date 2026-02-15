"""Benchmark utilities for linear Cyclone base case comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from importlib import resources

import jax.numpy as jnp

from spectraxgk.analysis import ModeSelection, extract_mode, fit_growth_rate, select_ky_index
from spectraxgk.config import CycloneBaseCase
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.linear import LinearParams, integrate_linear


@dataclass(frozen=True)
class CycloneReference:
    ky: np.ndarray
    omega: np.ndarray
    gamma: np.ndarray


@dataclass(frozen=True)
class CycloneRunResult:
    t: np.ndarray
    phi_t: np.ndarray
    gamma: float
    omega: float
    ky: float
    selection: ModeSelection


def load_cyclone_reference() -> CycloneReference:
    """Load GX Cyclone base case reference data (adiabatic electrons)."""

    data_path = resources.files("spectraxgk").joinpath("data", "cyclone_gx_adiabatic_ref.csv")
    arr = np.loadtxt(data_path, delimiter=",", skiprows=1)
    ky = arr[:, 0]
    omega = arr[:, 1]
    gamma = arr[:, 2]
    return CycloneReference(ky=ky, omega=omega, gamma=gamma)


def run_cyclone_linear(
    ky_target: float = 0.3,
    Nl: int = 2,
    Nm: int = 4,
    dt: float = 0.05,
    steps: int = 200,
    params: LinearParams | None = None,
    cfg: CycloneBaseCase | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
) -> CycloneRunResult:
    """Run the linear Cyclone benchmark and extract growth rate."""

    cfg = cfg or CycloneBaseCase()
    params = params or LinearParams()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    ky_index = select_ky_index(np.asarray(grid.ky), ky_target)
    sel = ModeSelection(ky_index=ky_index, kx_index=0, z_index=0)

    G0 = np.zeros((Nl, Nm, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=np.complex64)
    G0[0, 0, sel.ky_index, sel.kx_index, :] = 1e-3 + 0.0j

    G0_jax = jnp.asarray(G0)
    _, phi_t = integrate_linear(G0_jax, grid, geom, params, dt=dt, steps=steps)

    phi_t_np = np.asarray(phi_t)
    t = np.arange(steps) * dt
    signal = extract_mode(phi_t_np, sel)
    gamma, omega = fit_growth_rate(t, signal, tmin=tmin, tmax=tmax)

    return CycloneRunResult(
        t=t,
        phi_t=phi_t_np,
        gamma=gamma,
        omega=omega,
        ky=float(grid.ky[sel.ky_index]),
        selection=sel,
    )
