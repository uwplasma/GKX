"""Integrating factor on the end absorber against explicit RK4 on a small nonlinear box.

Run ``python benchmarks/absorber/absorber_nonlinear.py [T] [amplitude] [h_ref]`` (float64, CPU,
about two minutes). The box is the kinetic-electron Cyclone deck of ``examples/05``
at 8 x 8 perpendicular modes and ``(N_l, N_m) = (2, 4)``, started at an amplitude where the
bracket is 19% of the right-hand side. The reference is explicit RK4 at a step inside its
stability limit (``h_ref`` 2e-6 against ``2.785 / rho(S + A)``); its difference to ``h_ref / 2``
is printed. The integrating-factor step of :mod:`absorber_exponential` then runs at larger
steps and reports the relative error of the state, of its interior (no absorber) and of ``phi``.
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

from gkx.operators.linear.cache_builder import build_linear_cache  # noqa: E402
from gkx.operators.nonlinear.policies import measured_streaming_frequency  # noqa: E402
from gkx.solvers.nonlinear.diagnostics import nonlinear_rhs_cached  # noqa: E402
from gkx.workflows.nonlinear import _prepare_context  # noqa: E402
from gkx.workflows.runtime.toml import load  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def main(T: float = 0.05, amp: float = 3000.0, h_ref: float = 2.0e-6) -> None:
    cfg = load(os.path.join(ROOT, "examples/05_kinetic_electrons/case.toml"))
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nx=8, Ny=8, Nz=8),
        init=replace(cfg.init, init_amp=amp),
        run=replace(cfg.run, Nl=2, Nm=4),
    )
    Nl, Nm = cfg.run.Nl, cfg.run.Nm
    ctx, cfg = _prepare_context(
        cfg, ky_target=cfg.run.ky, kx_target=None, Nl=Nl, Nm=Nm, dt=None, steps=None,
        status=lambda _: None,
    )  # fmt: skip
    cache = build_linear_cache(ctx.grid, ctx.geom, ctx.params, Nl, Nm)
    params, terms, g0 = ctx.params, ctx.terms, ctx.G0
    kw = dict(
        compressed_real_fft=bool(cfg.time.compressed_real_fft),
        laguerre_mode=cfg.time.laguerre_nonlinear_mode,
    )

    def rhs(t):
        return jax.jit(lambda g: nonlinear_rhs_cached(g, cache, params, t, **kw)[0])

    full, no_absorber = rhs(terms), rhs(replace(terms, end_damping=0.0))
    off = dict.fromkeys(
        "streaming mirror curvature gradb diamagnetic collisions hypercollisions nonlinear".split(),
        0.0,
    )
    absorber_only = replace(terms, **off)
    expo = build_absorber_exponential(cache, params, absorber_only, g0.shape, g0.dtype)
    rho_s = measured_streaming_frequency(
        params, cache, nl=Nl, nm=Nm, hypercollisions=1.0
    )
    try:  # the absorber-aware bound; older trees measure streaming only
        rho_sa = measured_streaming_frequency(
            params, cache, nl=Nl, nm=Nm, hypercollisions=1.0, absorber_vs=0.0
        )
    except TypeError:
        rho_sa = None
    print(
        json.dumps(dict(shape=list(g0.shape), rho_S=rho_s, rho_S_plus_A=rho_sa)),
        flush=True,
    )

    def rk4(g, h, f):
        k1 = f(g)
        k2 = f(g + h / 2 * k1)
        k3 = f(g + h / 2 * k2)
        return g + h / 6 * (k1 + 2 * (k2 + k3) + f(g + h * k3))

    def run(step, n):
        g = g0
        for _ in range(n):
            g = step(g)
        return g

    ref = run(jax.jit(lambda g: rk4(g, h_ref, full)), int(round(T / h_ref)))
    half = run(jax.jit(lambda g: rk4(g, h_ref / 2, full)), int(round(2 * T / h_ref)))
    norm = jnp.linalg.norm(ref)
    print(
        json.dumps(
            dict(
                h_ref=h_ref,
                reference_self_difference=float(jnp.linalg.norm(half - ref) / norm),
            )
        ),
        flush=True,
    )
    interior = (expo.rate == 0.0)[None, None, None]

    def rel(a, b):
        a, b = jnp.asarray(a), jnp.asarray(b)
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    phi = lambda x: np.asarray(expo.fields(x)[0])  # noqa: E731
    for h in (5e-4, 1e-3, 2e-3, 4e-3):
        step = jax.jit(lambda g, h=h: lawson_rk4(g, h, no_absorber, expo))
        g = run(step, int(round(T / h)))
        print(
            json.dumps(
                dict(
                    h=h,
                    h_rho_S=h * rho_s,
                    state=rel(g, ref),
                    interior=rel(
                        jnp.where(interior, g, 0), jnp.where(interior, ref, 0)
                    ),
                    phi=rel(phi(g), phi(ref)),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main(*(float(a) for a in sys.argv[1:]))
