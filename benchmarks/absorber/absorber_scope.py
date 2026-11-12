"""Spectral radii and one-step amplification of the end absorber on a small kinetic-electron box.

Run ``python benchmarks/absorber/absorber_scope.py [rate]`` (float64, CPU). It prints the
radius of the streaming operator ``S``, of the absorber ``A``, of its diagonal part ``D`` and
of the remainder ``R = A - D``, and the spectral radius of the one-step amplification matrix
of explicit RK4, the integrating factor on ``D`` alone and the integrating factor on all of
``A`` (:mod:`absorber_exponential`), for a range of steps ``h``. A radius above one is an
unstable step.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from absorber_exponential import (  # noqa: E402
    build_absorber_exponential,
    lawson_rk4,
)

from gkx import runtime as rt  # noqa: E402
from gkx.core_grid import build_spectral_grid  # noqa: E402
from gkx.geometry.core import apply_geometry_grid_defaults  # noqa: E402
from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.linear.dissipation import end_damping_contribution  # noqa: E402
from gkx.terms.assembly import assemble_rhs_cached  # noqa: E402
from gkx.terms.config import TermConfig  # noqa: E402
from gkx.workflows.runtime.toml import load  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OFF = dict.fromkeys(
    "streaming mirror curvature gradb diamagnetic collisions hypercollisions end_damping".split(),
    0.0,
)


def build(rate: float, nl: int = 2, nm: int = 3):
    """A 4 x 4 x 8 box of the kinetic-electron Cyclone deck (example 05), float64."""
    cfg = load(os.path.join(ROOT, "examples/05_kinetic_electrons/case.toml"))
    cfg = replace(cfg, grid=replace(cfg.grid, Nx=4, Ny=4, Nz=8, ntheta=8))
    geom = rt.build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    params = rt.build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    params = replace(params, damp_ends_rate=rate)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    shape = (2, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    return cfg, params, cache, shape


def dense(op, shape):
    n = int(np.prod(shape))
    eye = jnp.eye(n, dtype=jnp.complex128).reshape((n, *shape))
    return np.asarray(jax.lax.map(op, eye)).reshape(n, n).T


def radius(m) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(m))))


def main(rate: float = 598.0) -> None:
    cfg, params, cache, shape = build(rate)
    terms = rt.build_runtime_term_config(cfg)
    fields = {"apar": terms.apar, "bpar": terms.bpar}
    stream = TermConfig(**{**OFF, "streaming": 1.0, **fields})
    absorb = TermConfig(**{**OFF, "end_damping": 1.0, **fields})
    rhs = lambda t: lambda g: assemble_rhs_cached(g, cache, params, terms=t)[0]  # noqa: E731
    s_op, a_op = rhs(stream), rhs(absorb)
    s, a = dense(s_op, shape), dense(a_op, shape)
    amp = params.end_damping_rate(cache.dz * cache.damp_profile.shape[-1], jnp.float64)
    diag_op = lambda g: end_damping_contribution(  # noqa: E731
        g,
        ky=cache.ky,
        damp_profile=cache.damp_profile,
        linked_damp_profile=cache.linked_damp_profile,
        damp_amp=amp,
        weight=jnp.asarray(1.0),
    )
    d = dense(diag_op, shape)
    r = a - d
    print(
        json.dumps(
            dict(
                rate=rate,
                rho_S=radius(s),
                rho_A=radius(a),
                rho_D=radius(d),
                rho_R=radius(r),
                rank_R=int(
                    np.sum(
                        np.linalg.svd(r, compute_uv=False) > 1e-9 * np.linalg.norm(r, 2)
                    )
                ),
                rho_S_plus_A=radius(s + a),
                rho_S_plus_R=radius(s + r),
            )
        ),
        flush=True,
    )
    expo = build_absorber_exponential(cache, params, absorb, shape, jnp.complex128)
    n = s.shape[0]
    eye = jnp.eye(n, dtype=jnp.complex128).reshape((n, *shape))
    ident, diag = np.eye(n), np.diag(d)
    for h in np.geomspace(2e-6, 1.5e-2, 10):
        m = h * (s + a)
        rk4 = ident + m + m @ m / 2 + m @ m @ m / 6 + m @ m @ m @ m / 24
        step = jax.jit(
            lambda e, h=h: jax.lax.map(lambda x: lawson_rk4(x, h, s_op, expo), e)
        )
        full = np.asarray(step(eye)).reshape(n, n).T
        e1, e2 = (
            np.exp(h * diag)[:, None],
            np.exp(h * diag / 2)[:, None],
        )  # diagonal part only
        nn = s + r
        k2 = nn @ (e2 * (ident + h / 2 * nn))
        k3 = nn @ (e2 * ident + h / 2 * k2)
        k4 = nn @ (e1 * ident + h * e2 * k3)
        diag_only = e1 * ident + h / 6 * (e1 * nn + 2 * e2 * (k2 + k3) + k4)
        print(
            json.dumps(
                dict(
                    h=float(h),
                    rk4=radius(rk4),
                    integrating_factor_on_D=radius(diag_only),
                    integrating_factor_on_A=radius(full),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 598.0)
