# ruff: noqa: E402
"""SOLVAX-DIRECT: the linear growth-rate gradient from one shift-invert factor.

PERF-LIT ranked item 5: "growth-rate gradient equal to dense to 1e-8 (x64) at
n=3,072 and faster than the adaptive gradient at n >= 3,072". The objective is
``gkx.solver_growth_rate_from_geometry`` on PERF-LIT's geometry (the shipped
nonlinear Cyclone deck's geometry, differentiated in the nine profile arrays of
``GEOMETRY_FIELDS``). Arms, one per process (``--arm``):

``direct``    ``solvax.sparse_eigenvalue`` (SOLVAX-DIRECT branch): the operator
              is assembled inside the trace by compressed products
              (``csr_data_from_products``), ``A - sigma I`` is factored once on
              the host, and the gradient is one VJP of the matrix-free operator
              at the right eigenvector, weighted by the left eigenvector from
              conjugate-transposed solves on the same factor. The factor cache
              is cleared before every timed call, so each call factors, as an
              optimizer step with new geometry would.
``dense``     GKX's shipped route: dense operator, ``lax.linalg.eig``, autodiff.
``adaptive``  GKX's matrix-free ``eigensolver="adaptive-propagator"`` objective
              vector (element 0), differentiated implicitly.

The shift is a warm start (``--shift``), as in an optimization loop: the
eigenvalue selected is the largest growth rate among ``--candidates`` nearest
it, and the value is checked against the dense arm's.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

T0_PROCESS = time.perf_counter()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

TOML = ROOT / "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_t400.toml"
GEOMETRY_FIELDS = (
    "bmag_profile",
    "bgrad_profile",
    "gds2_profile",
    "cv_profile",
    "gb_profile",
    "cv0_profile",
    "gb0_profile",
    "jacobian_profile",
    "grho_profile",
)


def log(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(
        f"[{time.perf_counter() - T0_PROCESS:8.1f}s rss<={rss:.2f}G] {msg}",
        file=sys.stderr,
        flush=True,
    )


def geometry_for(nz: int):
    """PERF-LIT's ``build_case(1, 4, nz, ...)`` geometry."""
    from gkx.core_grid import build_spectral_grid
    from gkx.geometry import ensure_flux_tube_geometry_data
    from gkx.runtime import build_runtime_geometry
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, _raw = load_runtime_from_toml(TOML)
    runtime = dataclasses.replace(
        runtime, grid=dataclasses.replace(runtime.grid, Nx=1, Ny=4, Nz=nz, ntheta=None)
    )
    grid = build_spectral_grid(runtime.grid)
    return ensure_flux_tube_geometry_data(build_runtime_geometry(runtime), grid.z)


def with_geometry(geometry, values):
    return dataclasses.replace(geometry, **dict(zip(GEOMETRY_FIELDS, values)))


def timed_calls(fn, args, repeats, before=None):
    calls = []
    out = None
    for _ in range(repeats):
        if before is not None:
            before()
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        calls.append(time.perf_counter() - t)
    return calls, out


def grad_norm(g) -> float:
    return float(sum(jnp.linalg.norm(x) ** 2 for x in g) ** 0.5)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("direct", "dense", "adaptive"))
    p.add_argument("--nz", type=int, default=96)
    p.add_argument("--nl", type=int, default=4)
    p.add_argument("--nm", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--shift", type=complex, default=complex(0.12, -0.25))
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--backend", default="mumps")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import gkx
    import solvax

    geometry = geometry_for(args.nz)
    geo_vars = [jnp.asarray(getattr(geometry, name)) for name in GEOMETRY_FIELDS]
    rec: dict = {
        "arm": args.arm,
        "Nz": args.nz,
        "Nl": args.nl,
        "Nm": args.nm,
        "environment": {
            "host": platform.node(),
            "loadavg": list(os.getloadavg()),
            "sha": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "jax": jax.__version__,
            "solvax": solvax.__version__,
            "solvax_file": str(Path(solvax.__file__).parent.name),
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "XLA_FLAGS")},
        },
    }

    if args.arm == "dense":

        def gamma(v):
            return gkx.solver_growth_rate_from_geometry(
                with_geometry(geometry, v), n_laguerre=args.nl, n_hermite=args.nm
            )

        from gkx.objectives.core import (
            _solver_geometry_context,
            _solver_operator_matrix,
        )

        ctx = _solver_geometry_context(
            geometry,
            selected_ky_index=1,
            n_laguerre=args.nl,
            n_hermite=args.nm,
            nx=1,
            ny=4,
            lx=6.0,
            ly=12.0,
            params_linear=None,
            terms=None,
        )
        lam = np.linalg.eigvals(np.asarray(_solver_operator_matrix(ctx)))
        top = lam[np.argmax(lam.real)]
        rec["lambda"] = [top.real, top.imag]
        rec["n"] = int(lam.size)
        value = jax.jit(gamma)
        vg = jax.jit(jax.value_and_grad(gamma))
        rec["value_s"], v = timed_calls(value, (geo_vars,), args.repeats)
        rec["value_and_grad_s"], (v, g) = timed_calls(vg, (geo_vars,), args.repeats)
    elif args.arm == "adaptive":

        def gamma(v):
            return gkx.solver_objective_vector_from_geometry(
                with_geometry(geometry, v),
                n_laguerre=args.nl,
                n_hermite=args.nm,
                eigensolver="adaptive-propagator",
            )[0]

        rec["value_s"], v = timed_calls(gamma, (geo_vars,), min(args.repeats, 2))
        rec["value_and_grad_s"], (v, g) = timed_calls(
            jax.value_and_grad(gamma), (geo_vars,), min(args.repeats, 2)
        )
    else:
        import operators as op
        from gkx.objectives.core import _linear_rhs_phi, _solver_geometry_context

        def context(v):
            return _solver_geometry_context(
                with_geometry(geometry, v),
                selected_ky_index=1,
                n_laguerre=args.nl,
                n_hermite=args.nm,
                nx=1,
                ny=4,
                lx=6.0,
                ly=12.0,
                params_linear=None,
                terms=None,
            )

        shape = context(geo_vars).state_shape
        n = int(np.prod(shape))
        rec["n"] = n

        def operator(v, x):
            ctx = context(v)
            return _linear_rhs_phi(x.reshape(ctx.state_shape), ctx)[0].reshape(-1)

        # Pattern and column groups: host-side, once, at the starting geometry.
        t = time.perf_counter()
        probe = jax.jit(lambda x: operator(geo_vars, x))
        full_shape = (1,) * (6 - len(shape)) + tuple(shape)
        local, chain = op.moment_graph(probe, full_shape)
        local = local | (np.eye(local.shape[0], dtype=bool) & ~chain)
        mask = op.pattern_from_graph(local, chain, shape[-1]).tocsr()
        mask.sort_indices()
        pattern = solvax.CsrPattern(mask.indptr, mask.indices, mask.shape)
        groups = op.kron_groups(local, chain, shape[-1])
        seeds = np.zeros((len(groups), n))
        for k, cols in enumerate(groups):
            seeds[k, cols] = 1.0
        seeds = jnp.asarray(seeds)
        rec["structure_s"] = time.perf_counter() - t
        rec["products"] = len(groups)
        rec["nnz"] = pattern.nnz
        options = solvax.HostFactorOptions(
            backend=args.backend,
            memory_limit_bytes=4 * 10**9 if args.backend == "mumps" else None,
        )

        def eigen(v):
            products = jax.vmap(lambda s: operator(v, s))(seeds)
            values = solvax.csr_data_from_products(pattern, groups, products)
            return solvax.sparse_eigenvalue(
                operator,
                v,
                pattern,
                values,
                args.shift,
                candidates=args.candidates,
                options=options,
                residual_tolerance=1e-9,
            )

        def gamma(v):
            return jnp.real(eigen(v).value)

        first = jax.jit(eigen)(geo_vars)
        lam = complex(first.value)
        rec["lambda"] = [lam.real, lam.imag]
        rec["residual"] = float(first.residual)
        rec["left_residual"] = float(first.left_residual)
        value = jax.jit(gamma)
        vg = jax.jit(jax.value_and_grad(gamma))
        clear = solvax.clear_factor_cache
        rec["value_s"], v = timed_calls(value, (geo_vars,), args.repeats, before=clear)
        rec["value_and_grad_s"], (v, g) = timed_calls(
            vg, (geo_vars,), args.repeats, before=clear
        )
        rec["factor_cache"] = solvax.factor_cache_info()

    rec["gamma"] = float(v)
    rec["grad_norm"] = grad_norm(g)
    rec["grad"] = [np.asarray(x).tolist() for x in g]
    rec["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    log(
        f"{args.arm}: gamma={rec['gamma']:.15g} value_s={rec['value_s']} vg_s={rec['value_and_grad_s']}"
    )
    text = json.dumps(rec, default=float)
    print(
        "RESULT "
        + json.dumps({k: v for k, v in rec.items() if k != "grad"}, default=float)
    )
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
