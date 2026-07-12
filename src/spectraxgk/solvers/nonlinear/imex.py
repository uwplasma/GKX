"""IMEX nonlinear solve policies.

The public nonlinear facade builds operators and diagnostics.  This module owns
the small, reusable fixed-point predictor and GMRES solve step used by cached
and diagnostic IMEX paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from solvax import gmres

from spectraxgk.solvers.nonlinear.imex_diagnostics import (
    advance_imex_nonlinear_state,
    make_imex_diagnostic_step,
    run_imex_diagnostic_scan,
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
DiagnosticFn = Callable[..., Any]
CollisionSplitFn = Callable[[jnp.ndarray, Any, jnp.ndarray, str], jnp.ndarray]
DiagnosticStepFn = Callable[
    [tuple[Any, Any, Any, Any, Any], Any],
    tuple[tuple[Any, Any, Any, Any, Any], tuple[Any, Any]],
]
DiagnosticScanOutput = tuple[jnp.ndarray, tuple[Any, Any]]


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
    solution = gmres(
        matvec,
        G_rhs.reshape(-1),
        x0=G_guess.reshape(-1),
        precond=precond_op,
        restart=implicit_restart,
        rtol=implicit_tol,
        atol=0.0,
        max_restarts=implicit_maxiter,
    )
    return solution.x.reshape(shape)


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
            precond_op=precond_op,
        )

    return solve_step


def _resolve_imex_operator(
    *,
    implicit_operator: Any | None,
    G0: jnp.ndarray,
    cache: object,
    params: object,
    dt: float,
    linear_cfg: Any,
    implicit_preconditioner: str | None,
    compressed_real_fft: bool,
    build_operator_fn: OperatorBuilderFn,
    build_implicit_operator_fn: Callable[..., tuple[Any, ...]] | None,
) -> Any:
    """Build the implicit operator only when the caller did not provide one."""

    if implicit_operator is not None:
        return implicit_operator
    build_kwargs = {}
    if build_implicit_operator_fn is not None:
        build_kwargs["build_implicit_operator_fn"] = build_implicit_operator_fn
    return build_operator_fn(
        G0,
        cache,
        params,
        dt,
        terms=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
        **build_kwargs,
    )


def _state_for_imex_operator(
    G0: jnp.ndarray, implicit_operator: Any
) -> tuple[jnp.ndarray, tuple[int, ...], bool]:
    """Cast and shape the initial state to match the implicit operator."""

    shape = implicit_operator.shape
    squeeze_species = implicit_operator.squeeze_species
    G = jnp.asarray(G0, dtype=implicit_operator.state_dtype)
    if squeeze_species and G.ndim == len(shape) - 1:
        G = G[None, ...]
    if G.shape != shape:
        raise ValueError(
            "implicit_operator shape mismatch: "
            f"expected {shape}, got {tuple(G.shape)}"
        )
    return G, shape, squeeze_species


@dataclass(frozen=True)
class _CachedImexScanSetup:
    G: jnp.ndarray
    shape: tuple[int, ...]
    squeeze_species: bool
    dt_val: jnp.ndarray
    precond_op: PreconditionerFn | None
    matvec: MatvecFn


def _prepare_cached_imex_scan_setup(
    G0: jnp.ndarray,
    cache: object,
    params: object,
    dt: float,
    *,
    linear_cfg: Any,
    implicit_preconditioner: str | None,
    implicit_operator: Any | None,
    compressed_real_fft: bool,
    build_operator_fn: OperatorBuilderFn,
    build_implicit_operator_fn: Callable[..., tuple[Any, ...]] | None,
) -> _CachedImexScanSetup:
    """Resolve the implicit operator and initial state for cached IMEX scans."""

    operator = _resolve_imex_operator(
        implicit_operator=implicit_operator,
        G0=G0,
        cache=cache,
        params=params,
        dt=dt,
        linear_cfg=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        compressed_real_fft=compressed_real_fft,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=build_implicit_operator_fn,
    )
    G, shape, squeeze_species = _state_for_imex_operator(G0, operator)
    return _CachedImexScanSetup(
        G=G,
        shape=shape,
        squeeze_species=squeeze_species,
        dt_val=operator.dt_val,
        precond_op=operator.precond_op,
        matvec=operator.matvec,
    )


def _make_cached_imex_scan_step(
    *,
    setup: _CachedImexScanSetup,
    cache: object,
    params: object,
    term_cfg: Any,
    linear_cfg: Any,
    linear_rhs_fn: LinearRhsFn,
    fields_fn: FieldSolveFn,
    nonlinear_term_fn: NonlinearTermKernel,
    nonlinear_contribution_fn: NonlinearTermKernel,
    external_phi: jnp.ndarray | float | None,
    compressed_real_fft: bool,
    laguerre_mode: str,
    implicit_iters: int,
    implicit_relax: float,
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
) -> Callable[[jnp.ndarray, Any], tuple[jnp.ndarray, Any]]:
    """Build the cached IMEX scan body from explicit nonlinear and GMRES parts."""

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
        dt_val=setup.dt_val,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        matvec=setup.matvec,
        shape=setup.shape,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        precond_op=setup.precond_op,
        solve_step_fn=solve_imex_step,
    )

    def step(G_in: jnp.ndarray, _unused: Any) -> tuple[jnp.ndarray, Any]:
        rhs = G_in + setup.dt_val * nonlinear_term(G_in)
        G_new = solve_step(G_in, rhs)
        _dG_new, fields_new = linear_rhs_fn(
            G_new, cache, params, linear_cfg, external_phi=external_phi
        )
        return G_new, fields_new

    return step


def _run_cached_imex_scan(
    setup: _CachedImexScanSetup,
    step: Callable[[jnp.ndarray, Any], tuple[jnp.ndarray, Any]],
    *,
    steps: int,
    checkpoint: bool,
) -> tuple[jnp.ndarray, Any]:
    """Run the cached IMEX scan and restore single-species output rank."""

    step_fn = jax.checkpoint(step) if checkpoint else step
    G_out, fields_t = jax.lax.scan(step_fn, setup.G, None, length=steps)
    G_out = G_out[0] if setup.squeeze_species else G_out
    return G_out, fields_t



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
    setup = _prepare_cached_imex_scan_setup(
        G0,
        cache=cache,
        params=params,
        dt=dt,
        linear_cfg=linear_cfg,
        implicit_preconditioner=implicit_preconditioner,
        implicit_operator=implicit_operator,
        compressed_real_fft=compressed_real_fft,
        build_operator_fn=build_operator_fn,
        build_implicit_operator_fn=build_implicit_operator_fn,
    )
    step = _make_cached_imex_scan_step(
        setup=setup,
        cache=cache,
        params=params,
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        linear_rhs_fn=linear_rhs_fn,
        fields_fn=fields_fn,
        nonlinear_term_fn=nonlinear_term_fn,
        nonlinear_contribution_fn=nonlinear_contribution_fn,
        external_phi=external_phi,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
    )
    return _run_cached_imex_scan(setup, step, steps=steps, checkpoint=checkpoint)


__all__ = [
    "advance_imex_nonlinear_state",
    "imex_fixed_point_guess",
    "integrate_cached_imex_scan",
    "make_imex_diagnostic_step",
    "make_imex_nonlinear_term",
    "make_imex_solve_step",
    "run_imex_diagnostic_scan",
    "solve_imex_step",
]
