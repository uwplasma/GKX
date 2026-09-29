"""The Jacobian benchmark's AD rows agree with its own finite differences."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.operators.linear.params import LinearParams
from gkx.solvers_nonlinear_state_integration import nonlinear_heat_flux_window
from gkx.terms.config import TermConfig
from support.paths import load_tool_script


def test_flux_jacobian_reverse_forward_and_fd_agree():
    jax.config.update("jax_enable_x64", True)
    module = load_tool_script("campaigns", "nonlinear_flux_jacobian")
    cfg = CycloneBaseCase(grid=GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0))
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    state = jnp.zeros((2, 2, grid.ky.size, cfg.grid.Nx, cfg.grid.Nz), jnp.complex128)
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 1, :].set((0.2 - 0.1j) * profile)
    params = LinearParams(tprim=jnp.asarray([2.49]), fprim=jnp.asarray([0.8]))

    def evaluate(p):
        return nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            p,
            dt=0.01,
            steps=6,
            method="rk3",
            terms=TermConfig(nonlinear=1.0),
            tail_steps=4,
        )

    objective, theta0 = module.make_objective(params, ["tprim", "fprim"], evaluate)
    record = module.differentiate(objective, theta0, [1.0e-3, 1.0e-5])

    reverse = np.asarray(record["reverse"])
    assert np.all(np.isfinite(reverse)) and np.all(reverse != 0.0)
    assert record["forward_reverse_relative_difference"] < 1.0e-10
    best = np.min([row["relative_error"] for row in record["fd"]], axis=0)
    assert np.all(best < 1.0e-6), best
    assert module.parameter_names(["tprim", "fprim"], 1) == ["tprim[0]", "fprim[0]"]
