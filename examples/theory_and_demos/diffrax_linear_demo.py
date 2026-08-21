"""Diffrax-based linear integration demo."""

import numpy as np
import jax.numpy as jnp

from gkx.config import CycloneBaseCase, GridConfig
from gkx.solvers.time.diffrax import integrate_linear_diffrax
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.params import LinearParams


def main() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=8, Nz=32, Lx=6.28, Ly=6.28)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    G0 = G0.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)

    _, phi_t = integrate_linear_diffrax(
        G0,
        grid,
        geom,
        params,
        dt=0.1,
        steps=4,
        method="Tsit5",
        adaptive=True,
        progress_bar=True,
    )

    phi_np = np.asarray(phi_t)
    print("diffrax demo phi_t shape:", phi_np.shape)
    print("phi_t min/max:", phi_np.min(), phi_np.max())


if __name__ == "__main__":
    main()
