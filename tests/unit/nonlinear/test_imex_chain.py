"""Kinetic-electron step: measured streaming bound and the per-chain IMEX operator."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from gkx import runtime as rt
from gkx.core_grid import build_spectral_grid
from gkx.geometry.core import apply_geometry_grid_defaults
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.nonlinear.policies import measured_streaming_frequency
from gkx.solvers_nonlinear_imex_chain import (
    ARS_TABLEAUX,
    build_chain_implicit_linear,
)
from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import TermConfig
from gkx.workflows.runtime.toml import load
from support.paths import REPO_ROOT

NL, NM = 1, 3


def _tiny_kinetic_electron_box(layout: str = "half"):
    cfg = load(REPO_ROOT / "examples/05_kinetic_electrons/case.toml")
    cfg = replace(
        cfg, grid=replace(cfg.grid, Nx=4, Ny=4, Nz=8, ntheta=8, ky_layout=layout)
    )
    geom = rt.build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    params = rt.build_runtime_linear_params(cfg, Nm=NM, geom=geom)
    cache = build_linear_cache(grid, geom, params, NL, NM)
    shape = (2, NL, NM, grid.ky.size, grid.kx.size, grid.z.size)
    return cfg, params, cache, shape


def _dense(op, shape, dtype):
    n = int(np.prod(shape))
    eye = jnp.eye(n, dtype=dtype).reshape((n, *shape))
    return np.asarray(jax.lax.map(op, eye)).reshape(n, n).T


def test_measured_streaming_frequency_is_the_operator_spectral_radius():
    _, params, cache, shape = _tiny_kinetic_electron_box()
    terms = TermConfig(
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
    )
    dtype = jnp.complex64

    def op(G):
        return assemble_rhs_cached(G, cache, params, terms=terms)[0]

    radius = np.max(np.abs(np.linalg.eigvals(_dense(op, shape, dtype))))
    measured = measured_streaming_frequency(params, cache, nl=NL, nm=NM)
    assert measured > 0.0
    np.testing.assert_allclose(measured, radius, rtol=1e-3)


def test_measured_streaming_frequency_skips_adiabatic_like_decks():
    _, params, cache, _ = _tiny_kinetic_electron_box()
    heavy = replace(params, vth=jnp.ones_like(jnp.asarray(params.vth)))
    assert measured_streaming_frequency(heavy, cache, nl=NL, nm=NM) == 0.0


def test_chain_solve_inverts_the_implicit_linear_rhs():
    cfg, params, cache, shape = _tiny_kinetic_electron_box()
    terms = rt.build_runtime_term_config(cfg)

    def rhs(G):
        return nonlinear_rhs_cached(G, cache, params, terms)[0]

    dt = 0.05
    op = build_chain_implicit_linear(
        rhs, shape, dt, modes=np.ones(shape[3:5], bool), scheme="imex-ars3"
    )
    key = jax.random.PRNGKey(3)
    G = (jax.random.normal(key, shape) + 1j * jax.random.normal(key + 1, shape)).astype(
        jnp.complex64
    )
    lin = 0.5 * (rhs(G) - rhs(-G))
    gamma = ARS_TABLEAUX["imex-ars3"][1][1][1]
    x = op.solve(G - gamma * dt * lin)
    np.testing.assert_allclose(x, G, atol=1e-3 * float(jnp.max(jnp.abs(G))))
