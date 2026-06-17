"""IMEX nonlinear solve policies.

The public nonlinear facade builds operators and diagnostics.  This module owns
the small, reusable fixed-point predictor and GMRES solve step used by cached
and diagnostic IMEX paths.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from spectraxgk.solvers.nonlinear.explicit import (
    _SSPX3_ADT,
    _SSPX3_W1,
    _SSPX3_W2,
    _SSPX3_W3,
)

LinearRhsFn = Callable[..., tuple[jnp.ndarray, object]]
MatvecFn = Callable[[jnp.ndarray], jnp.ndarray]
FieldSolveFn = Callable[..., object]
NonlinearTermKernel = Callable[..., jnp.ndarray]
NonlinearTermFn = Callable[[jnp.ndarray], jnp.ndarray]
PreconditionerFn = Callable[[jnp.ndarray], jnp.ndarray]
ProjectFn = Callable[[jnp.ndarray], jnp.ndarray]
SolveStepFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
OperatorBuilderFn = Callable[..., Any]


def imex_fixed_point_guess(
    G_in: jnp.ndarray,
    G_rhs: jnp.ndarray,
    *,
    linear_rhs_fn: LinearRhsFn,
    cache: Any,
    params: Any,
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


def make_imex_nonlinear_term(
    cache: object,
    params: object,
    term_cfg: object,
    *,
    real_dtype: object | None = None,
    external_phi: jnp.ndarray | float | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
    fields_fn: FieldSolveFn,
    nonlinear_term_fn: NonlinearTermKernel,
    nonlinear_contribution_fn: NonlinearTermKernel | None = None,
) -> NonlinearTermFn:
    """Return the explicit nonlinear term closure used by IMEX scans."""

    extra_kwargs = (
        {}
        if nonlinear_contribution_fn is None
        else {"nonlinear_contribution_fn": nonlinear_contribution_fn}
    )

    def nonlinear_term(G_in: jnp.ndarray) -> jnp.ndarray:
        return nonlinear_term_fn(
            G_in,
            cache,
            params,
            term_cfg,
            real_dtype=real_dtype,
            external_phi=external_phi,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            fields_fn=fields_fn,
            **extra_kwargs,
        )

    return nonlinear_term


def make_imex_solve_step(
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
    precond_op: PreconditionerFn | None,
    solve_step_fn: Callable[..., jnp.ndarray] = solve_imex_step,
) -> SolveStepFn:
    """Return the GMRES solve-step closure used by IMEX scan policies."""

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        return solve_step_fn(
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
            matvec=matvec,
            shape=shape,
            implicit_tol=implicit_tol,
            implicit_maxiter=implicit_maxiter,
            implicit_restart=implicit_restart,
            implicit_solve_method=implicit_solve_method,
            precond_op=precond_op,
        )

    return solve_step


def advance_imex_nonlinear_state(
    G: jnp.ndarray,
    *,
    dt_val: jnp.ndarray,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: SolveStepFn,
    project_state: ProjectFn,
) -> jnp.ndarray:
    """Advance one IMEX nonlinear step with optional SSPX3 stage composition."""

    if method == "sspx3":

        def _euler_step(G_state: jnp.ndarray, dt_stage: jnp.ndarray) -> jnp.ndarray:
            rhs_stage = G_state + dt_stage * nonlinear_term(G_state)
            return solve_step(G_state, rhs_stage)

        G1 = _euler_step(G, _SSPX3_ADT * dt_val)
        G2_euler = _euler_step(G1, _SSPX3_ADT * dt_val)
        G2 = project_state(
            (1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler
        )
        G3 = _euler_step(G2, _SSPX3_ADT * dt_val)
        return (
            (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
            + _SSPX3_W3 * G1
            + (_SSPX3_W2 - 1.0) * G2
            + G3
        )

    rhs = G + dt_val * nonlinear_term(G)
    return solve_step(G, rhs)


def integrate_cached_imex_scan(
    G0: jnp.ndarray,
    cache: object,
    params: object,
    dt: float,
    steps: int,
    *,
    term_cfg: Any,
    linear_cfg: Any,
    linear_rhs_fn: LinearRhsFn,
    build_operator_fn: OperatorBuilderFn,
    build_implicit_operator_fn: Callable[..., tuple[Any, ...]] | None = None,
    fields_fn: FieldSolveFn,
    nonlinear_term_fn: NonlinearTermKernel,
    nonlinear_contribution_fn: NonlinearTermKernel,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_solve_method: str = "batched",
    implicit_preconditioner: str | None = None,
    implicit_operator: Any | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, Any]:
    """Run the cached IMEX nonlinear scan.

    The public facade injects field solves, operator construction, and RHS
    kernels so debug and monkeypatch seams stay outside this pure solver owner.
    """

    del show_progress  # Progress belongs to diagnostics/runtime scans.
    if implicit_operator is None:
        build_kwargs = {}
        if build_implicit_operator_fn is not None:
            build_kwargs["build_implicit_operator_fn"] = build_implicit_operator_fn
        implicit_operator = build_operator_fn(
            G0,
            cache,
            params,
            dt,
            terms=linear_cfg,
            implicit_preconditioner=implicit_preconditioner,
            compressed_real_fft=compressed_real_fft,
            **build_kwargs,
        )

    shape = implicit_operator.shape
    dt_val = implicit_operator.dt_val
    precond_op = implicit_operator.precond_op
    matvec = implicit_operator.matvec
    squeeze_species = implicit_operator.squeeze_species
    G = jnp.asarray(G0, dtype=implicit_operator.state_dtype)
    if squeeze_species and G.ndim == len(shape) - 1:
        G = G[None, ...]
    if G.shape != shape:
        raise ValueError(
            "implicit_operator shape mismatch: "
            f"expected {shape}, got {tuple(G.shape)}"
        )

    nonlinear_term = make_imex_nonlinear_term(
        cache,
        params,
        term_cfg,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_term_fn,
        nonlinear_contribution_fn=nonlinear_contribution_fn,
    )
    solve_step = make_imex_solve_step(
        linear_rhs_fn=linear_rhs_fn,
        cache=cache,
        params=params,
        linear_cfg=linear_cfg,
        external_phi=external_phi,
        dt_val=dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        matvec=matvec,
        shape=shape,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        implicit_solve_method=implicit_solve_method,
        precond_op=precond_op,
        solve_step_fn=solve_imex_step,
    )

    def step(G_in: jnp.ndarray, _unused: Any) -> tuple[jnp.ndarray, Any]:
        rhs = G_in + dt_val * nonlinear_term(G_in)
        G_new = solve_step(G_in, rhs)
        _dG_new, fields_new = linear_rhs_fn(
            G_new, cache, params, linear_cfg, external_phi=external_phi
        )
        return G_new, fields_new

    step_fn = jax.checkpoint(step) if checkpoint else step
    G_out, fields_t = jax.lax.scan(step_fn, G, None, length=steps)
    G_out = G_out[0] if squeeze_species else G_out
    return G_out, fields_t


__all__ = [
    "advance_imex_nonlinear_state",
    "imex_fixed_point_guess",
    "integrate_cached_imex_scan",
    "make_imex_nonlinear_term",
    "make_imex_solve_step",
    "solve_imex_step",
]
