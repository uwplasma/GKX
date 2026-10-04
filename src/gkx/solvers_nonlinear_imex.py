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
import numpy as np
from solvax import (
    KrylovSolution,
    block_thomas_factor,
    block_thomas_solve,
    gmres,
    linear_solve,
)

from gkx.solvers_linear_implicit import (
    ImplicitSolveStats,
    _GmresStatus,
    _empty_implicit_solve_stats,
    _fold_implicit_solve_stats,
    _gmres_iteration_budget,
)
from gkx.solvers_nonlinear_imex_diagnostics import (
    StatsSolveStepFn,
    advance_imex_nonlinear_state,
    advance_imex_nonlinear_state_with_stats,
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
CachedImexCarry = tuple[jnp.ndarray, ImplicitSolveStats]


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


def _imex_gmres_solver(
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
    implicit_tol: float,
    implicit_maxiter: int,
    implicit_restart: int,
    precond_op: PreconditionerFn | None,
) -> Callable[[MatvecFn, jnp.ndarray], KrylovSolution]:
    """Return the predictor-seeded FGMRES solve shared by both IMEX step forms."""

    restart, max_restarts = _gmres_iteration_budget(implicit_maxiter, implicit_restart)

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

    def solve(operator: MatvecFn, rhs: jnp.ndarray) -> KrylovSolution:
        return gmres(
            operator,
            rhs,
            x0=G_guess.reshape(-1),
            precond=precond_op,
            restart=restart,
            rtol=implicit_tol,
            atol=0.0,
            max_restarts=max_restarts,
        )

    return solve


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
    """Solve one IMEX system with a predictor and an implicit solve VJP.

    The primal and transpose solves use the same tolerance-controlled FGMRES
    policy. Reverse mode differentiates the converged linear system through
    SOLVAX rather than tracing the dynamic Krylov stopping loop.
    ``implicit_maxiter`` counts iterations. This form returns only the state;
    :func:`solve_imex_step_with_stats` also returns the convergence summary.
    """

    gmres_solve = _imex_gmres_solver(
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
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        precond_op=precond_op,
    )

    def solver(operator: MatvecFn, rhs: jnp.ndarray) -> jnp.ndarray:
        return gmres_solve(operator, rhs).x

    solution = linear_solve(matvec, G_rhs.reshape(-1), solver)
    return solution.reshape(shape)


def solve_imex_step_with_stats(
    G_in: jnp.ndarray,
    G_rhs: jnp.ndarray,
    stats: ImplicitSolveStats,
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
) -> tuple[jnp.ndarray, ImplicitSolveStats]:
    """Solve one IMEX system as :func:`solve_imex_step` and fold its status.

    SOLVAX's true residual, iteration count and converged flag leave
    ``linear_solve`` as auxiliary data, so reverse mode still differentiates
    the solved system and never the stopping loop.
    """

    gmres_solve = _imex_gmres_solver(
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
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_restart=implicit_restart,
        precond_op=precond_op,
    )

    def solver(
        operator: MatvecFn, rhs: jnp.ndarray
    ) -> tuple[jnp.ndarray, _GmresStatus]:
        result = gmres_solve(operator, rhs)
        return result.x, _GmresStatus(
            result.residual_norm, result.iterations, result.converged
        )

    rhs_flat = G_rhs.reshape(-1)
    solution, info = linear_solve(matvec, rhs_flat, solver, has_aux=True)
    return solution.reshape(shape), _fold_implicit_solve_stats(stats, info, rhs_flat)


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


def _imex_solve_policy(
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
) -> dict[str, Any]:
    """Collect the solve keywords both IMEX step closures forward verbatim."""

    return {
        "linear_rhs_fn": linear_rhs_fn,
        "cache": cache,
        "params": params,
        "linear_cfg": linear_cfg,
        "external_phi": external_phi,
        "dt_val": dt_val,
        "implicit_iters": implicit_iters,
        "implicit_relax": implicit_relax,
        "matvec": matvec,
        "shape": shape,
        "implicit_tol": implicit_tol,
        "implicit_maxiter": implicit_maxiter,
        "implicit_restart": implicit_restart,
        "precond_op": precond_op,
    }


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

    policy = _imex_solve_policy(
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

    def solve_step(G_in: jnp.ndarray, G_rhs: jnp.ndarray) -> jnp.ndarray:
        return solve_step_fn(G_in, G_rhs, **policy)

    return solve_step


def make_imex_solve_step_with_stats(
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
    solve_step_fn: Callable[..., tuple[jnp.ndarray, ImplicitSolveStats]] = (
        solve_imex_step_with_stats
    ),
) -> StatsSolveStepFn:
    """Return :func:`make_imex_solve_step`'s closure, threading solve status.

    The closure maps ``(state, rhs, stats)`` to ``(next state, stats)``, the
    same shape the implicit linear scan's step uses, so a diagnostic scan can
    carry its convergence channel without a host callback inside the scan.
    """

    policy = _imex_solve_policy(
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

    def solve_step(
        G_in: jnp.ndarray, G_rhs: jnp.ndarray, stats: ImplicitSolveStats
    ) -> tuple[jnp.ndarray, ImplicitSolveStats]:
        return solve_step_fn(G_in, G_rhs, stats, **policy)

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
            f"implicit_operator shape mismatch: expected {shape}, got {tuple(G.shape)}"
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
) -> Callable[[CachedImexCarry, Any], tuple[CachedImexCarry, Any]]:
    """Build the cached IMEX scan body; the carry holds the solve status."""

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

    def step(carry: CachedImexCarry, _unused: Any) -> tuple[CachedImexCarry, Any]:
        G_in, solve_stats = carry
        rhs = G_in + setup.dt_val * nonlinear_term(G_in)
        G_new, solve_stats = solve_imex_step_with_stats(
            G_in,
            rhs,
            solve_stats,
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
        )
        _dG_new, fields_new = linear_rhs_fn(
            G_new, cache, params, linear_cfg, external_phi=external_phi
        )
        return (G_new, solve_stats), fields_new

    return step


def _run_cached_imex_scan(
    setup: _CachedImexScanSetup,
    step: Callable[[CachedImexCarry, Any], tuple[CachedImexCarry, Any]],
    *,
    steps: int,
    checkpoint: bool,
) -> tuple[jnp.ndarray, Any, ImplicitSolveStats]:
    """Run the cached IMEX scan and restore single-species output rank."""

    step_fn = jax.checkpoint(step) if checkpoint else step
    carry0 = (setup.G, _empty_implicit_solve_stats(setup.G.dtype))
    (G_out, solve_stats), fields_t = jax.lax.scan(step_fn, carry0, None, length=steps)
    G_out = G_out[0] if setup.squeeze_species else G_out
    return G_out, fields_t, solve_stats


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
    return_solve_stats: bool = False,
) -> tuple[Any, ...]:
    """Run the cached IMEX nonlinear scan.

    The public facade injects field solves, operator construction, and RHS
    kernels so debug and monkeypatch seams stay outside this pure solver owner.
    Returns ``(G_out, fields_t)``, or ``(G_out, fields_t, stats)`` with
    ``return_solve_stats=True``; ``stats`` summarizes every GMRES solve.
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
    G_out, fields_t, solve_stats = _run_cached_imex_scan(
        setup, step, steps=steps, checkpoint=checkpoint
    )
    if return_solve_stats:
        return G_out, fields_t, solve_stats
    return G_out, fields_t


# --- Per-chain implicit linear operator (imex-ars2 / imex-ars3) -------------
#
# Implicit linear operator per linked chain, for the ``imex-ars*`` methods.
#
# Kinetic-electron runs are step-limited by electron parallel streaming and the
# electromagnetic electron mode (|lambda| ~ 500-700 against drift rates ~ 1 on
# the Cyclone tutorial box), not by accuracy. The linear operator couples modes
# only along one ``(ky, kx)`` twist-shift chain, so its restriction to a chain is
# a dense matrix of size ``ns Nl Nm Nz L`` (``L`` chain links). This module
# materializes those matrices once, by probing the linear part of the run's own
# RHS (its odd part, so every linear term -- streaming, fields, mirror,
# drifts, dissipation, end damping -- enters exactly as the explicit route
# applies it), and inverts ``I - gamma dt L`` per chain.
#
# The time step is an Ascher-Ruuth-Spiteri IMEX Runge-Kutta scheme: L-stable
# SDIRK with one diagonal coefficient for the linear part (one factorization per
# dt), explicit for the nonlinear bracket. It is the plan's section 5.4 route with dense per-chain
# factors in place of the banded response-matrix solve: memory is
# ``sum over chains of (ns Nl Nm Nz L)^2`` complex entries, which fits the
# tutorial and moderate decks and not production resolution.


_G2 = 1.0 - 1.0 / 2.0**0.5
_D2 = 1.0 - 1.0 / (2.0 * _G2)
_G3 = 0.4358665215
_B1 = -1.5 * _G3**2 + 4.0 * _G3 - 0.25
_B2 = 1.5 * _G3**2 - 5.0 * _G3 + 1.25
# (explicit A, implicit A, explicit b, implicit b), Ascher-Ruuth-Spiteri 1997.
# ARS(2,2,2): two-stage explicit part, whose RK2 has no imaginary-axis interval
# (bracket advection is weakly unstable at any dt). ARS(3,4,3): third order,
# four explicit stages with an imaginary-axis interval, three RHS per step.
ARS_TABLEAUX: dict[str, Any] = {
    "imex-ars2": (
        ((0, 0, 0), (_G2, 0, 0), (_D2, 1 - _D2, 0)),
        ((0, 0, 0), (0, _G2, 0), (0, 1 - _G2, _G2)),
        (_D2, 1 - _D2, 0),
        (0, 1 - _G2, _G2),
    ),
    "imex-ars3": (
        (
            (0, 0, 0, 0),
            (_G3, 0, 0, 0),
            (0.3212788860, 0.3966543747, 0, 0),
            (-0.105858296, 0.5529291479, 0.5529291479, 0),
        ),
        (
            (0, 0, 0, 0),
            (0, _G3, 0, 0),
            (0, (1 - _G3) / 2, _G3, 0),
            (0, _B1, _B2, _G3),
        ),
        (0, _B1, _B2, _G3),
        (0, _B1, _B2, _G3),
    ),
}
IMEX_CHAIN_METHODS = frozenset(ARS_TABLEAUX)


@dataclass(frozen=True)
class _ChainGroup:
    ky: np.ndarray  # (chains, L) row indices
    kx: np.ndarray  # (chains, L) column indices
    factors: Any  # block-Thomas factors of I - gamma dt S, batch (chains*s*l)
    u: jnp.ndarray  # (chains, s, l, rows_m, N, nf*N): fields -> RHS rows
    w: jnp.ndarray  # (chains, nf, s, l, cols_m, N): moments -> fields (z-local)
    cap: jnp.ndarray  # (chains, nf*N, nf*N): (I - gamma dt Q B^-1 U)^-1


@dataclass(frozen=True)
class ChainImplicitLinear:
    """Structured ``(I - gamma dt L)^{-1}`` for the stiff linear terms.

    ``L = S + U Q`` per twist-shift chain: ``S`` is streaming with its
    dissipation (block-tridiagonal in Hermite, an ``N x N`` block per
    ``(species, Laguerre)``, ``N = links * Nz``) and ``U Q`` the field
    response (``Q`` z-local, rank ``nf N``). ``I - gamma dt S`` is factored by
    block Thomas and the fields enter by Woodbury, so memory is
    ``O(ns Nl Nm N^2)`` per chain instead of ``O((ns Nl Nm N)^2)``.
    """

    groups: tuple[_ChainGroup, ...]
    shape: tuple[int, ...]
    dt: float
    scheme: str
    rows_m: tuple[int, ...]
    cols_m: tuple[int, ...]
    linear: Callable[[jnp.ndarray], jnp.ndarray]  # L G (stiff terms only)

    @property
    def nbytes(self) -> int:
        leaves = jax.tree_util.tree_leaves(
            [(g.factors, g.u, g.w, g.cap) for g in self.groups]
        )
        return sum(int(x.nbytes) for x in leaves)

    @property
    def gamma_dt(self) -> float:
        return ARS_TABLEAUX[self.scheme][1][1][1] * self.dt

    def _bsolve(self, g: _ChainGroup, x: jnp.ndarray) -> jnp.ndarray:
        batch = int(np.prod(x.shape[:3]))
        y = jax.vmap(block_thomas_solve)(g.factors, x.reshape(batch, *x.shape[3:]))
        return y.reshape(x.shape)

    def _apply_inverse(self, g: _ChainGroup, x: jnp.ndarray) -> jnp.ndarray:
        x0 = self._bsolve(g, x)
        f = jnp.einsum("cfslmn,cslmn->cfn", g.w, x0[:, :, :, list(self.cols_m)])
        y = jnp.einsum("cij,cj->ci", g.cap, f.reshape(f.shape[0], -1))
        uy = jnp.einsum("cslmni,ci->cslmn", g.u, y)
        full = jnp.zeros_like(x0).at[:, :, :, list(self.rows_m)].set(uy)
        return x0 + self.gamma_dt * self._bsolve(g, full)

    def solve(self, R: jnp.ndarray) -> jnp.ndarray:
        """``(I - gamma dt L)^{-1} R`` on the chains (identity elsewhere)."""
        out = R
        for g in self.groups:
            x = _gather(g.ky, g.kx, R).astype(g.cap.dtype)
            out = _scatter(g.ky, g.kx, self._apply_inverse(g, x).astype(R.dtype), out)
        return out

    def ars_step(
        self,
        G: jnp.ndarray,
        dG: jnp.ndarray,
        rhs: Callable[[jnp.ndarray], jnp.ndarray],
        project: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """One ARS step; ``dG = rhs(G)`` is the full (linear + bracket) RHS."""
        a_exp, a_imp, b_exp, b_imp = ARS_TABLEAUX[self.scheme]
        dt, gam = self.dt, a_imp[1][1]
        # Stage 1 needs only the explicit part at G (ARS implicit weights on it
        # are zero): dG minus the stiff linear terms, no further RHS.
        lin: list[Any] = [None]
        non: list[Any] = [dG - self.linear(G)]
        for i in range(1, len(b_exp)):
            r = G + dt * sum(a_exp[i][j] * non[j] for j in range(i))
            r = r + dt * sum(a_imp[i][j] * lin[j] for j in range(1, i))
            y = project(self.solve(r))
            lin.append((y - r) / (gam * dt))  # y = r + gamma dt L y
            non.append(rhs(y) - lin[i] if b_exp[i] or i + 1 < len(b_exp) else 0.0)
        return G + dt * sum(
            b_exp[j] * non[j] + (b_imp[j] * lin[j] if j else 0.0)
            for j in range(len(b_exp))
        )


def _gather(ky: np.ndarray, kx: np.ndarray, G: jnp.ndarray) -> jnp.ndarray:
    """``(s, l, m, ky, kx, z)`` -> ``(chains, s, l, m, links * z)``."""
    x = jnp.moveaxis(G[:, :, :, ky, kx, :], 3, 0)  # (c, s, l, m, L, z)
    return x.reshape(*x.shape[:4], -1)


def _scatter(ky, kx, X: jnp.ndarray, out: jnp.ndarray) -> jnp.ndarray:
    x = X.reshape(*X.shape[:4], ky.shape[1], out.shape[-1])
    return out.at[:, :, :, ky, kx, :].set(jnp.moveaxis(x, 0, 3))


def _chains(lin: Callable, shape: tuple[int, ...], modes: np.ndarray, dtype) -> list:
    """Connected sets of the linear operator among the ``(ky, kx)`` ``modes``."""
    rng = np.random.default_rng(0)
    parent = {(int(r), int(c)): (int(r), int(c)) for r, c in np.argwhere(modes)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # Modes at different ky never couple, so one probe per kx column finds the
    # kx links of every row at once.
    for j in np.nonzero(modes.any(axis=0))[0]:
        rows = np.nonzero(modes[:, j])[0]
        v = np.zeros(shape, dtype=complex)
        v[:, :, :, rows, j, :] = rng.normal(size=v[:, :, :, rows, j, :].shape)
        out = np.abs(np.asarray(lin(jnp.asarray(v, dtype=dtype)))).max(
            axis=(0, 1, 2, 5)
        )
        tol = 1e-9 * max(float(out.max()), 1e-300)
        for r in rows:
            for q in np.nonzero(out[r] > tol)[0]:
                if (int(r), int(q)) in parent:
                    parent[find((int(r), int(q)))] = find((int(r), int(j)))
    sets: dict = {}
    for mode in parent:
        sets.setdefault(find(mode), []).append(mode)
    return [sorted(c) for c in sets.values()]


def _unit(n: int, j: jnp.ndarray, dtype) -> jnp.ndarray:
    return jnp.where(jnp.arange(n) == j, 1.0, 0.0).astype(dtype)


def _factor_chain_group(
    k, x, split, zero, zero_f, w_all, act, rows_m, cols_m, gdt, lin, dt, scheme
) -> _ChainGroup:
    """Probe ``S``, ``U`` and ``Q`` on one group of equal-length chains."""
    ns, nl, nm, *_r, nz = zero.shape
    dtype = zero.dtype
    c, n = k.shape[0], k.shape[1] * nz
    # S is block-tridiagonal in m: sources three apart never share an output
    # row, so three probe families (m mod 3) give every block, N probes each.
    lower, diag, upper = (jnp.zeros((c, ns, nl, nm, n, n), dtype) for _ in range(3))
    for res in range(3):
        src = np.arange(res, nm, 3)

        def s_column(j, src=src):
            e = (
                jnp.zeros((c, ns, nl, nm, n), dtype)
                .at[:, :, :, src]
                .set(_unit(n, j, dtype))
            )
            o = _gather(k, x, split(_scatter(k, x, e, zero), zero_f))
            pad = jnp.pad(o, ((0, 0),) * 3 + ((1, 1), (0, 0)))
            return tuple(pad[:, :, :, src + 1 + d] for d in (1, 0, -1))

        got = jax.lax.map(s_column, jnp.arange(n))
        for shift, part in zip((1, 0, -1), got):
            rows = src + shift
            ok = (rows >= 0) & (rows < nm)
            col = jnp.moveaxis(part, 0, -1)[:, :, :, ok]  # (c, s, l, src, N, N)
            if shift == 1:
                lower = lower.at[:, :, :, rows[ok]].set(col)
            elif shift == 0:
                diag = diag.at[:, :, :, rows[ok]].set(col)
            else:
                upper = upper.at[:, :, :, rows[ok]].set(col)
    eye = jnp.eye(n, dtype=dtype)
    factors = jax.vmap(block_thomas_factor)(
        (-gdt * lower).reshape(c * ns * nl, nm, n, n),
        (eye - gdt * diag).reshape(c * ns * nl, nm, n, n),
        (-gdt * upper).reshape(c * ns * nl, nm, n, n),
    )
    del lower, diag, upper

    def u_column(j):
        fg = jnp.broadcast_to(_unit(n, j, dtype), (c, n)).reshape(c, -1, nz)
        return jnp.stack(
            [
                _gather(k, x, split(zero, zero_f.at[f, k, x, :].set(fg)))[
                    :, :, :, list(rows_m)
                ]
                for f in act
            ],
            -1,
        )  # (c, s, l, rows_m, N, nf)

    u = jnp.moveaxis(jax.lax.map(u_column, jnp.arange(n)), 0, -1)
    u = u.reshape(c, ns, nl, len(rows_m), n, -1)  # last axis (f, N)
    wq = w_all[:, :, list(cols_m)][:, :, :, act][:, :, :, :, k, x, :]
    wq = jnp.transpose(wq, (4, 3, 0, 1, 2, 5, 6)).reshape(
        c, len(act), ns, nl, len(cols_m), n
    )
    g = _ChainGroup(k, x, factors, u, wq, jnp.zeros((c, 0, 0), dtype))
    op = ChainImplicitLinear((g,), zero.shape, float(dt), scheme, rows_m, cols_m, lin)

    def cap_column(i):
        ui = (
            jnp.zeros((c, ns, nl, nm, n), dtype)
            .at[:, :, :, list(rows_m)]
            .set(g.u[..., i])
        )
        y = op._bsolve(g, ui)[:, :, :, list(cols_m)]
        return jnp.einsum("cfslmn,cslmn->cfn", g.w, y).reshape(c, -1)

    qbu = jnp.moveaxis(jax.lax.map(cap_column, jnp.arange(len(act) * n)), 0, -1)
    cap = jnp.linalg.inv(jnp.eye(qbu.shape[-1], dtype=dtype) - gdt * qbu)
    return _ChainGroup(k, x, factors, u, wq, cap)


def build_chain_implicit_linear(
    split_rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    fields: Callable[[jnp.ndarray], jnp.ndarray],
    shape: tuple[int, ...],
    dt: float,
    *,
    modes: np.ndarray,
    scheme: str = "imex-ars3",
    dtype=jnp.complex64,
    check_tol: float = 1.0e-3,
) -> ChainImplicitLinear:
    """Probe the stiff linear terms per chain and factor ``I - gamma dt L``.

    ``fields(G)`` is the linear field solve, stacked ``(nf, Nky, Nkx, Nz)``;
    ``split_rhs(G, F)`` the implicit terms' RHS of ``G`` with the fields ``F``
    imposed. ``L G = split_rhs(G, fields(G))``; ``S = split_rhs(., 0)`` and
    ``U = split_rhs(0, .)`` are probed separately, ``Q`` from unit moments.
    ``modes`` (``Nky x Nkx`` bool) selects what is solved: every ``ky >= 0``
    mode, dealiased or not (leaving dealiased-out modes explicit makes their
    stiff terms explicit). ``ky < 0`` rows pass through; the projector
    rebuilds them. The factorization is checked by the residual of one solve.
    """
    ns, nl, nm, _nky, _nkx, nz = shape
    split, fld = jax.jit(split_rhs), jax.jit(fields)
    zero = jnp.zeros(shape, dtype)
    zero_f = jnp.zeros_like(fld(zero))

    def lin(v):
        return split(v, fld(v))

    # Q: the field solve is local in (ky, kx, z), so one unit moment per
    # (s, l, m) slot over every mode gives all of its weights.
    w_all = jax.lax.map(
        lambda i: fld(
            jnp.broadcast_to(
                jnp.zeros(ns * nl * nm, dtype)
                .at[i]
                .set(1)
                .reshape(ns, nl, nm, 1, 1, 1),
                shape,
            )
        ),
        jnp.arange(ns * nl * nm),
    ).reshape(ns, nl, nm, *zero_f.shape)  # (s, l, m, nf, ky, kx, z)
    w_np = np.abs(np.asarray(w_all))
    act = np.nonzero(w_np.max(axis=(0, 1, 2, 4, 5, 6)) > 0)[0]
    cols_m = tuple(int(m) for m in np.nonzero(w_np.max(axis=(0, 1, 3, 4, 5, 6)) > 0)[0])
    rng = np.random.default_rng(0)
    probe_f = jnp.asarray(rng.normal(size=zero_f.shape), dtype)
    resp = np.abs(np.asarray(split(zero, probe_f))).max(axis=(0, 1, 3, 4, 5))
    rows_m = tuple(int(m) for m in np.nonzero(resp > 1e-9 * max(resp.max(), 1e-300))[0])

    by_len: dict[int, list] = {}
    for chain in _chains(jax.jit(lin), shape, np.asarray(modes, dtype=bool), dtype):
        by_len.setdefault(len(chain), []).append(chain)
    idx = [
        (np.array(c)[..., 0], np.array(c)[..., 1]) for _, c in sorted(by_len.items())
    ]
    gdt = ARS_TABLEAUX[scheme][1][1][1] * float(dt)

    groups = [
        _factor_chain_group(
            k, x, split, zero, zero_f, w_all, act, rows_m, cols_m, gdt, lin, dt, scheme
        )
        for k, x in idx
    ]
    op = ChainImplicitLinear(
        tuple(groups), tuple(shape), float(dt), scheme, rows_m, cols_m, lin
    )
    # One solve's residual certifies every structural assumption at once
    # (Hermite tridiagonality, z-local fields, the chain partition).
    r = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype)
    mask = np.zeros(shape[3:5], bool)
    for k, x in idx:
        mask[k, x] = True
    on = jnp.asarray(mask, dtype)[None, None, None, :, :, None]
    r = r * on
    y = op.solve(r)
    # Only the solved rows: the RHS rebuilds ky < 0 rows from their partners.
    err = float(jnp.linalg.norm(on * (y - gdt * lin(y) - r)) / jnp.linalg.norm(r))
    if not err < check_tol:
        raise ValueError(
            f"structured implicit factor residual {err:.2e} > {check_tol:.0e}: the "
            "stiff operator is not block-tridiagonal in Hermite with z-local fields"
        )
    return op


__all__ = [
    "ARS_TABLEAUX",
    "ChainImplicitLinear",
    "IMEX_CHAIN_METHODS",
    "build_chain_implicit_linear",
    "advance_imex_nonlinear_state",
    "advance_imex_nonlinear_state_with_stats",
    "imex_fixed_point_guess",
    "integrate_cached_imex_scan",
    "make_imex_diagnostic_step",
    "make_imex_nonlinear_term",
    "make_imex_solve_step",
    "make_imex_solve_step_with_stats",
    "run_imex_diagnostic_scan",
    "solve_imex_step",
    "solve_imex_step_with_stats",
]
