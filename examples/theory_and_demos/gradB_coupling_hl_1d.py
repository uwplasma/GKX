#!/usr/bin/env python3
"""Minimal grad-B coupling demo using the linear solver."""

import numpy as np
import jax.numpy as jnp

from gkx.config import GeometryConfig, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.params import LinearParams, LinearTerms
from gkx.solvers.linear.integrators import integrate_linear


def main() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=1, Nz=128, Lx=6.28, Ly=6.28)
    grid = build_spectral_grid(grid_cfg)
    geom = SAlphaGeometry.from_config(GeometryConfig())
    params = LinearParams(tprim=0.0, fprim=0.0)
    terms = LinearTerms(streaming=1.0, mirror=0.0, curvature=0.0, gradb=1.0, diamagnetic=0.0)

    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

    _, phi_t = integrate_linear(G0, grid, geom, params, dt=0.1, steps=5, method="rk2", terms=terms)
    phi_np = np.asarray(phi_t)
    print("gradB demo phi_t shape:", phi_np.shape)
    print("phi_t min/max:", phi_np.min(), phi_np.max())


if __name__ == "__main__":
    main()
