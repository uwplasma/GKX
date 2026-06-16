"""IMEX nonlinear solve policies.

The public nonlinear facade builds operators and diagnostics.  This module owns
the small, reusable fixed-point predictor and GMRES solve step used by cached
and diagnostic IMEX paths.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

LinearRhsFn = Callable[..., tuple[jnp.ndarray, object]]
MatvecFn = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerFn = Callable[[jnp.ndarray], jnp.ndarray]


def imex_fixed_point_guess(
    G_in: jnp.ndarray,
    G_rhs: jnp.ndarray,
    *,
    linear_rhs_fn: LinearRhsFn,
    cache: object,
    params: object,
    linear_cfg: object,
    external_phi: jnp.ndarray | float | None,
    dt_val: jnp.ndarray,
    implicit_iters: int,
    implicit_relax: float,
) -> jnp.ndarray:
    """Build the fixed-point predictor used as the GMRES initial guess."""

    def body(_i, g):
        dG, _fields = linear_rhs_fn(
            g, cache, params, linear_cfg, external_phi=external_phi
        )
        g_next = G_rhs + dt_val * dG
        return (1.0 - implicit_relax) * g + implicit_relax * g_next

    return jax.lax.fori_loop(0, max(int(implicit_iters), 0), body, G_in)


def solve_imex_step(
    G_in: jnp.ndarray,
    G_rhs: jnp.ndarray,
    *,
    linear_rhs_fn: LinearRhsFn,
    cache: object,
    params: object,
    linear_cfg: object,
    external_phi: jnp.ndarray | float | None,
    dt_val: jnp.ndarray,
    implicit_iters: int,
    implicit_relax: float,
    matvec: MatvecFn,
    shape: tuple[int, ...],
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
    implicit_solve_method: str,
    precond_op: PreconditionerFn | None = None,
) -> jnp.ndarray:
    """Solve one IMEX linear system with a fixed-point predictor."""

    G_guess = imex_fixed_point_guess(
        G_in,
        G_rhs,
        linear_rhs_fn=linear_rhs_fn,
        cache=cache,
        params=params,
        linear_cfg=linear_cfg,
        external_phi=external_phi,
        dt_val=dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
    )
    sol, _info = jax.scipy.sparse.linalg.gmres(
        matvec,
        G_rhs.reshape(-1),
        x0=G_guess.reshape(-1),
        tol=implicit_tol,
        maxiter=implicit_maxiter,
        restart=implicit_restart,
        M=precond_op,
        solve_method=implicit_solve_method,
    )
    return sol.reshape(shape)


__all__ = ["imex_fixed_point_guess", "solve_imex_step"]
