#!/usr/bin/env python3
"""Benchmark lineax GMRES against the current JAX GMRES path on a GKX implicit linear solve."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pandas as pd

from gkx.benchmarking.shared import (
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
    _build_initial_condition,
)
from gkx.config import CycloneBaseCase, GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.linear import LinearParams, LinearTerms, build_linear_cache
from gkx.solvers.linear.implicit import _build_implicit_operator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ky", type=float, default=0.3)
    p.add_argument("--Nl", type=int, default=8)
    p.add_argument("--Nm", type=int, default=16)
    p.add_argument("--Ny", type=int, default=7)
    p.add_argument("--Nz", type=int, default=96)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tol", type=float, default=1.0e-6)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--restart", type=int, default=20)
    p.add_argument("--preconditioner", type=str, default="diagonal")
    p.add_argument("--out", type=Path, default=None)
    return p


def _build_case(
    *, ky: float, Nl: int, Nm: int, Ny: int, Nz: int, dt: float, preconditioner: str
):
    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=1,
            Ny=int(Ny),
            Nz=int(Nz),
            Lx=62.8,
            Ly=62.8,
            y0=20.0,
            ntheta=32,
            nperiod=2,
            boundary="linked",
        )
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid_full = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid_full.ky) - float(ky))))
    grid = select_ky_grid(grid_full, ky_index)
    params = LinearParams(
        fprim=cfg.model.fprim,
        tprim=cfg.model.tprim_i,
        tprim_e=cfg.model.tprim_e,
        omega_d_scale=CYCLONE_OMEGA_D_SCALE,
        omega_star_scale=CYCLONE_OMEGA_STAR_SCALE,
        rho_star=CYCLONE_RHO_STAR,
        kpar_scale=float(geom.gradpar()),
        nu=cfg.model.nu_i,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=1.0 / 8.0,
    )
    cache = build_linear_cache(grid, geom, params, int(Nl), int(Nm))
    G0 = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=int(Nl),
        Nm=int(Nm),
        init_cfg=cfg.init,
    )
    G, shape, size, dt_val, precond_op, matvec, _squeeze = _build_implicit_operator(
        G0,
        cache,
        params,
        float(dt),
        LinearTerms(),
        preconditioner,
    )
    return G, shape, size, dt_val, precond_op, matvec


def _timed_jax_gmres(
    matvec,
    precond_op,
    rhs: jnp.ndarray,
    *,
    tol: float,
    maxiter: int,
    restart: int,
    use_preconditioner: bool,
):
    M = precond_op if use_preconditioner else None
    x0 = precond_op(rhs) if (use_preconditioner and precond_op is not None) else rhs
    solve = jax.jit(
        lambda b: jax.scipy.sparse.linalg.gmres(
            matvec,
            b,
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            restart=restart,
            M=M,
            solve_method="batched",
        )[0]
    )
    t0 = time.perf_counter()
    sol1 = solve(rhs)
    sol1.block_until_ready()
    t1 = time.perf_counter()
    sol2 = solve(rhs)
    sol2.block_until_ready()
    t2 = time.perf_counter()
    return sol1, t1 - t0, t2 - t1


def _timed_lineax_gmres(
    matvec,
    precond_op,
    rhs: jnp.ndarray,
    *,
    shape,
    tol: float,
    maxiter: int,
    restart: int,
):
    struct = jax.ShapeDtypeStruct(shape=rhs.shape, dtype=rhs.dtype)
    operator = lx.FunctionLinearOperator(matvec, struct)
    preconditioner = lx.IdentityLinearOperator(struct)
    options = {"y0": jnp.zeros_like(rhs), "preconditioner": preconditioner}
    solver = lx.GMRES(rtol=tol, atol=0.0, max_steps=maxiter, restart=restart)

    t0 = time.perf_counter()
    soln1 = lx.linear_solve(operator, rhs, solver, options=options, throw=False)
    soln1.value.block_until_ready()
    t1 = time.perf_counter()
    soln2 = lx.linear_solve(
        operator, rhs, solver, options=options, state=soln1.state, throw=False
    )
    soln2.value.block_until_ready()
    t2 = time.perf_counter()
    return soln1, soln2, t1 - t0, t2 - t1


def _residual_norm(matvec, x: jnp.ndarray, b: jnp.ndarray) -> float:
    r = matvec(x) - b
    return float(jnp.linalg.norm(r))


def main() -> None:
    args = build_parser().parse_args()
    (
        G,
        shape,
        size,
        _dt_val,
        precond_op,
        matvec,
    ) = _build_case(
        ky=float(args.ky),
        Nl=int(args.Nl),
        Nm=int(args.Nm),
        Ny=int(args.Ny),
        Nz=int(args.Nz),
        dt=float(args.dt),
        preconditioner=str(args.preconditioner),
    )
    rhs = G.reshape(size)

    jax_sol_pre, jax_pre_first_s, jax_pre_second_s = _timed_jax_gmres(
        matvec,
        precond_op,
        rhs,
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        restart=int(args.restart),
        use_preconditioner=True,
    )
    jax_sol_id, jax_id_first_s, jax_id_second_s = _timed_jax_gmres(
        matvec,
        precond_op,
        rhs,
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        restart=int(args.restart),
        use_preconditioner=False,
    )
    lineax_sol1, lineax_sol2, lineax_first_s, lineax_second_s = _timed_lineax_gmres(
        matvec,
        precond_op,
        rhs,
        shape=rhs.shape,
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        restart=int(args.restart),
    )

    rows = [
        {
            "solver": "jax_gmres_preconditioned",
            "phase": "first",
            "wall_s": jax_pre_first_s,
            "residual_norm": _residual_norm(matvec, jax_sol_pre, rhs),
        },
        {
            "solver": "jax_gmres_preconditioned",
            "phase": "second",
            "wall_s": jax_pre_second_s,
            "residual_norm": _residual_norm(matvec, jax_sol_pre, rhs),
        },
        {
            "solver": "jax_gmres_identity",
            "phase": "first",
            "wall_s": jax_id_first_s,
            "residual_norm": _residual_norm(matvec, jax_sol_id, rhs),
        },
        {
            "solver": "jax_gmres_identity",
            "phase": "second",
            "wall_s": jax_id_second_s,
            "residual_norm": _residual_norm(matvec, jax_sol_id, rhs),
        },
        {
            "solver": "lineax_gmres",
            "phase": "first",
            "wall_s": lineax_first_s,
            "residual_norm": _residual_norm(matvec, lineax_sol1.value, rhs),
            "result": str(lineax_sol1.result),
            "num_steps": int(np.asarray(lineax_sol1.stats.get("num_steps", -1))),
        },
        {
            "solver": "lineax_gmres",
            "phase": "second_reuse_state",
            "wall_s": lineax_second_s,
            "residual_norm": _residual_norm(matvec, lineax_sol2.value, rhs),
            "result": str(lineax_sol2.result),
            "num_steps": int(np.asarray(lineax_sol2.stats.get("num_steps", -1))),
        },
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
