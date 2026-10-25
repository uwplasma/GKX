"""IMEX nonlinear solve policies.

The public nonlinear facade builds operators and diagnostics.  This module owns
the small, reusable fixed-point predictor and GMRES solve step used by cached
and diagnostic IMEX paths.
"""

from __future__ import annotations

import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from functools import partial
from solvax import KrylovSolution, gmres, linear_solve
from typing import Any, Callable
from gkx.solvers_linear_implicit import (
    ImplicitSolveStats,
    _GmresStatus,
    _empty_implicit_solve_stats,
    _fold_implicit_solve_stats,
    _gmres_iteration_budget,
)
from gkx.solvers_nonlinear_explicit import (
    _SSPX3_ADT,
    _SSPX3_W1,
    _SSPX3_W2,
    _SSPX3_W3,
    cached_jit,
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


# --- Per-chain implicit linear operator (imex-ars3) ---------------------------
#
# Implicit linear operator per linked chain, for ``imex-ars3``.
#
# Kinetic-electron runs are step-limited by electron parallel streaming and the
# electromagnetic electron mode (|lambda| ~ 500-1500 against drift rates ~ 10),
# not by accuracy. Only the stiff part is implicit: parallel streaming with
# its dissipation (end damping, hypercollisions, hyperdiffusion) and the field
# response, L = S + U Q. It couples modes only along one (ky, kx) twist-shift
# chain; per chain, S is block-tridiagonal in Hermite with N x N blocks
# (N = links * Nz) and no species or Laguerre coupling, and Q is z-local
# (rank nf N). I - gamma dt S is factored by block Thomas, the fields enter
# by Woodbury. Every block is probed from the run's own RHS assembly, and one
# solve's residual certifies the structure. The electron mirror force is in
# S too (it set dt otherwise: |lambda| 57 against drifts ~3 at (4, 8)); it is
# z-local but couples l to l +- 1, so electron blocks are Nl N x Nl N.
# Drifts, the ion mirror, drive and collisions stay explicit with the bracket.
#
# The time step is an Ascher-Ruuth-Spiteri IMEX Runge-Kutta scheme: L-stable
# SDIRK with one diagonal coefficient for the linear part (one factorization
# per dt). Memory per chain is ~(Nm + 6) ns Nl N^2 complex entries against
# the dense (ns Nl Nm N)^2: 1.5 GB against 53 GB at 64x64x24, (Nl, Nm) =
# (4, 8).


_G3 = 0.4358665215
_B1 = -1.5 * _G3**2 + 4.0 * _G3 - 0.25
_B2 = 1.5 * _G3**2 - 5.0 * _G3 + 1.25
_GK = 1767732205903 / 4055673282236
_BK = (
    1471266399579 / 7840856788654,
    -4482444167858 / 7529755066697,
    11266239266428 / 11593286722821,
    _GK,
)
# (explicit A, implicit A, explicit b, implicit b, embedded b or None).
# Ascher-Ruuth-Spiteri 1997.
# ARS(3,4,3): third order, four explicit stages with an imaginary-axis
# interval, three RHS per step. (ARS(2,2,2) was removed: its RK2 explicit part
# has no imaginary-axis interval, so bracket advection is unstable at any dt.)
ARS_TABLEAUX: dict[str, Any] = {
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
        None,
    ),
    # Kennedy & Carpenter 2003, ARK3(2)4L[2]SA: the stability and cost of
    # ARS(3,4,3) (one more RHS) with an embedded second-order pair for dt
    # control. Shared b for both parts, ESDIRK implicit part.
    "imex-ark3": (
        (
            (0, 0, 0, 0),
            (2 * _GK, 0, 0, 0),
            (5535828885825 / 10492691773637, 788022342437 / 10882634858940, 0, 0),
            (
                6485989280629 / 16251701735622,
                -4246266847089 / 9704473918619,
                10755448449292 / 10357097424841,
                0,
            ),
        ),
        (
            (0, 0, 0, 0),
            (_GK, _GK, 0, 0),
            (2746238789719 / 10658868560708, -640167445237 / 6845629431997, _GK, 0),
            _BK,
        ),
        _BK,
        _BK,
        (
            2756255671327 / 12835298489170,
            -10771552573575 / 22201958757719,
            9247589265047 / 10645013368117,
            2193209047091 / 5459859503100,
        ),
    ),
}
IMEX_CHAIN_METHODS = frozenset(ARS_TABLEAUX)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["factors", "u", "w", "cap", "dr", "dc"],
    meta_fields=["ky", "kx"],
)
@dataclass(frozen=True)
class _ChainGroup:
    ky: tuple  # (chains, L) row indices (tuples: static in the pytree)
    kx: tuple  # (chains, L) column indices
    factors: Any  # per species group: (dinv, D, ladder scalars, mirror) of I - gdt S
    u: jnp.ndarray  # (chains, s, l, rows_m, N, nf*N): fields -> RHS rows
    w: jnp.ndarray  # (chains, nf, s, l, cols_m, N): moments -> fields (z-local)
    cap: jnp.ndarray  # (chains, nf*N, nf*N): (Dr (I - gamma dt Q B^-1 U) Dc)^-1
    dr: jnp.ndarray  # (chains, nf*N) row equilibration
    dc: jnp.ndarray  # (chains, nf*N) column equilibration


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["groups", "dt", "on"],
    meta_fields=["shape", "scheme", "rows_m", "cols_m", "linear", "spec"],
)
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
    spec: tuple[tuple[int, ...], ...] = ((0,),)  # species groups of the factor
    on: Any = None  # solved-row mask: refine each solve once (see ``solve``)

    @property
    def nbytes(self) -> int:
        leaves = jax.tree_util.tree_leaves(
            [(g.factors, g.u, g.w, g.cap, g.dr, g.dc) for g in self.groups]
        )
        return sum(int(x.nbytes) for x in leaves)

    @property
    def gamma_dt(self) -> float:
        return ARS_TABLEAUX[self.scheme][1][1][1] * self.dt

    def _bsolve(self, g: _ChainGroup, x: jnp.ndarray) -> jnp.ndarray:
        return _block_solve(g.factors, self.spec, x)

    def _apply_inverse(self, g: _ChainGroup, x: jnp.ndarray) -> jnp.ndarray:
        x0 = self._bsolve(g, x)
        f = jnp.einsum(
            "cfslmn,cslmn->cfn", g.w, x0[:, :, :, list(self.cols_m)], precision=_HI
        )
        y = g.dc * jnp.einsum(
            "cij,cj->ci", g.cap, g.dr * f.reshape(f.shape[0], -1), precision=_HI
        )
        uy = jnp.einsum("cslmni,ci->cslmn", g.u, y, precision=_HI)
        full = jnp.zeros_like(x0).at[:, :, :, list(self.rows_m)].set(uy)
        return x0 + self.gamma_dt * self._bsolve(g, full)

    def solve(self, R: jnp.ndarray) -> jnp.ndarray:
        """``(I - gamma dt L)^{-1} R`` on the chains (identity elsewhere).

        With ``on`` set, one step of iterative refinement (set at build when
        the check residual exceeds 5e-3; one step takes 3e-2 to 1e-5 and 0.8
        to 4e-3 at dt 0.2 and 0.8, 32x32, (4, 8), complex64).
        """
        y = self._solve(R)
        if self.on is None:
            return y
        return y - self._solve(self.on * (y - self.gamma_dt * self.linear(y) - R))

    def _solve(self, R: jnp.ndarray) -> jnp.ndarray:
        out = R
        for g in self.groups:
            ky, kx = np.asarray(g.ky), np.asarray(g.kx)
            x = _gather(ky, kx, R).astype(g.cap.dtype)
            out = _scatter(ky, kx, self._apply_inverse(g, x).astype(R.dtype), out)
        return out

    def ars_step(
        self,
        G: jnp.ndarray,
        dG: jnp.ndarray,
        rhs: Callable[[jnp.ndarray], jnp.ndarray],
        project: Callable[[jnp.ndarray], jnp.ndarray],
        *,
        with_error: bool = False,
    ) -> Any:
        """One IMEX step; ``dG = rhs(G)`` is the full (linear + bracket) RHS.

        ``with_error`` also returns the embedded error estimate (schemes with
        an embedded pair only).
        """
        a_exp, a_imp, b_exp, b_imp, b_hat = ARS_TABLEAUX[self.scheme]
        dt, gam = self.dt, a_imp[1][1]
        # Stage 1 is explicit: dG minus the stiff linear terms, no further RHS.
        lin: list[Any] = [self.linear(G)]
        non: list[Any] = [dG - lin[0]]

        def comb(be, bi):  # dt sum_j (be_j non_j + bi_j lin_j), ARS3 bits kept
            return dt * sum(
                (be[j] * non[j] if be[j] else 0.0) + (bi[j] * lin[j] if bi[j] else 0.0)
                for j in range(len(be))
            )

        for i in range(1, len(b_exp)):
            r = G + dt * sum(a_exp[i][j] * non[j] for j in range(i))
            if any(a_imp[i][:i]):
                r = r + dt * sum(a_imp[i][j] * lin[j] for j in range(i) if a_imp[i][j])
            y = project(self.solve(r))
            lin.append((y - r) / (gam * dt))  # y = r + gamma dt L y
            non.append(rhs(y) - lin[i] if b_exp[i] or i + 1 < len(b_exp) else 0.0)
        G_new = G + comb(b_exp, b_imp)
        if not with_error:
            return G_new
        d = [b - bh for b, bh in zip(b_imp, b_hat)]
        return G_new, comb(d, d)


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


# Full fp32 products: XLA otherwise lowers batched complex64 matmuls to TF32
# on Ampere, a 1e-3 error that the zonal A_par block (cond ~1e5) turns into
# an O(1) solve residual.
_HI = jax.lax.Precision.HIGHEST

# Probes evaluated per vmapped batch while building the factor: one probe is
# one RHS assembly, too small to fill a GPU alone.
_PROBE_BATCH = 8
_FACTORS: dict[Any, ChainImplicitLinear] = {}


def _mv(a: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("...ij,...j->...i", a, x, precision=_HI)


@jax.jit
def _thomas_factor(
    lower, diag, upper, e_lo, e_up, gdt
) -> tuple[tuple[jnp.ndarray, ...], Any]:
    """Block Thomas over Hermite for ``I - gdt S``; ``S`` blocks on axis 3.

    ``lower``, ``diag``, ``upper`` are the Laguerre-diagonal ``N x N`` blocks
    (``N = links * Nz``) batched over ``(chains, s, l)``. Every off-diagonal
    block is the chain's parallel-gradient matrix ``D`` (per species) times a
    Hermite-ladder scalar, plus, when ``e_lo``/``e_up`` (``(s, m, Nl, Nl,
    N)``, z-local) are given, the mirror force, which couples ``l`` to
    ``l +- 1``. Without the mirror the factor is one ``N x N`` inverse per
    ``(s, l, m)``; with it one ``Nl N x Nl N`` inverse per ``(s, m)`` (Nl x
    the memory). Only ``D``, the scalars, ``e`` and the Schur-complement
    inverses are stored. Inverses, not LU: batched triangular solves of
    blocks this small are launch-bound on GPU. One array per ``m`` (no stacked
    copy: 3.75 GB for the length-1 chains at 64x64, (8, 16)).
    """
    c, ns, nl, nm, n, _ = diag.shape
    k = 1 if e_lo is None else nl
    d = upper[:, :, 0, 0]  # (c, s, N, N): m = 0 -> 1, l = 0
    dd = jnp.sum(jnp.abs(d) ** 2, axis=(-2, -1))
    safe = jnp.where(dd > 0, dd, 1.0)[:, :, None, None]

    def scalars(blocks):
        a = jnp.einsum("csij,cslmij->cslm", jnp.conj(d), blocks, precision=_HI)
        return a / safe

    a_lo, a_up = scalars(lower), scalars(upper)
    err = jnp.stack(  # squared norms (defect, block) of lower and upper
        [
            jnp.sum(jnp.abs(x) ** 2)
            for blk, a in ((lower, a_lo), (upper, a_up))
            for x in (blk - a[..., None, None] * d[:, :, None, None], blk)
        ]
    )
    eye_k = jnp.eye(k, dtype=diag.dtype)

    def dense(blocks, e):  # (c, s, l, N, N) [+ e (s, k, k, N)] -> (c, s, b, kN, kN)
        x = blocks.reshape(c, ns, nl // k, k, n, n)
        x = jnp.einsum("csbkij,kq->csbkiqj", x, eye_k)
        if e is not None:
            x = (
                x
                + jnp.einsum("skqi,ij->skiqj", e, jnp.eye(n, dtype=x.dtype))[:, None][
                    None
                ]
            )
        return x.reshape(c, ns, nl // k, k * n, k * n)

    def off(a, e, m):
        return dense(a[:, :, :, m, None, None] * d[:, :, None], e[:, m])

    def mm(a, b):
        return jnp.matmul(a, b, precision=_HI)

    eye = jnp.eye(k * n, dtype=diag.dtype)
    dinv: list[jnp.ndarray] = []
    for m in range(nm):
        blk = eye - gdt * dense(diag[:, :, :, m], None)
        if m and e_lo is None:
            coef = gdt**2 * a_lo[:, :, :, m] * a_up[:, :, :, m - 1]
            blk = blk - coef[..., None, None] * mm(
                mm(d[:, :, None], dinv[-1]), d[:, :, None]
            )
        elif m:
            lo, up = off(a_lo, e_lo, m), off(a_up, e_up, m - 1)
            blk = blk - gdt**2 * mm(mm(lo, dinv[-1]), up)
        dinv.append(jnp.linalg.inv(blk))
    e = (None, None) if e_lo is None else (-gdt * e_lo, -gdt * e_up)
    return (tuple(dinv), d, -gdt * a_lo, -gdt * a_up, *e), err


def _thomas_solve(factors: tuple[jnp.ndarray, ...], x: jnp.ndarray) -> jnp.ndarray:
    dinv, d, lo, up, e_lo, e_up = factors
    c, ns, nl, nm, n = x.shape
    shape = (c, ns, dinv[0].shape[2], -1)

    def mv(m, v):  # dinv_m v on (c, s, l, N)
        return _mv(dinv[m], v.reshape(shape)).reshape(v.shape)

    def off(a, e, m, v):  # (a D + e) v, D shared over Laguerre
        y = a[:, :, :, m, None] * jnp.einsum("csij,cslj->csli", d, v, precision=_HI)
        if e is not None:
            y = y + jnp.einsum("skqn,csqn->cskn", e[:, m], v, precision=_HI)
        return y

    w = [mv(0, x[:, :, :, 0])]
    for m in range(1, nm):
        w.append(mv(m, x[:, :, :, m] - off(lo, e_lo, m, w[-1])))
    out = [w[-1]]
    for m in range(nm - 2, -1, -1):
        out.append(w[m] - mv(m, off(up, e_up, m, out[-1])))
    return jnp.stack(out[::-1], 3)


def _unit(n: int, j: jnp.ndarray, dtype) -> jnp.ndarray:
    return jnp.where(jnp.arange(n) == j, 1.0, 0.0).astype(dtype)


def _probe_columns(idx, nz: int, make: Callable, probe: Callable, take: Callable):
    """Columns ``j < N_g`` of a probed operator for every chain group at once.

    Groups hold disjoint modes and their chains do not couple, so one probe
    state carries local impulse ``j`` in every group with ``N_g > j``:
    ``max N_g`` probes instead of ``sum N_g`` (3.5x fewer at 64x64x24).
    ``probe`` is one compiled, vmapped RHS, ``_PROBE_BATCH`` states per call;
    ``make(k, x, n, j)`` scatters a group's impulse into a state and
    ``take(k, x, out)`` gathers its response. Indices ``j >= n`` make zero
    impulses.
    """
    ns = [k.shape[1] * nz for k, _ in idx]

    def make_all(j):
        v = None
        for (k, x), n in zip(idx, ns):
            v = make(k, x, n, j, v)
        return v

    make_all = cached_jit(jax.vmap(make_all))
    takes = [cached_jit(jax.vmap(partial(take, k, x))) for k, x in idx]
    cols: list[list] = [[] for _ in idx]
    # An RHS assembly holds ~9 states of intermediates: 8 probes of a 64x64,
    # (8, 16) state asked for 10 GB. Fewer per call above 0.4 GB of states.
    one = jax.eval_shape(lambda j: probe(make_all(j)), jnp.arange(1))
    size = np.prod(one.shape) * one.dtype.itemsize
    batch = int(np.clip(4e8 // size, 1, _PROBE_BATCH))
    for j0 in range(0, max(ns), batch):
        out = probe(make_all(jnp.arange(j0, j0 + batch)))
        for g, n in enumerate(ns):
            if j0 < n:
                # Host-side: the probed blocks of every group at once are
                # ~3x the factor (OOM at 64x64, (8, 16) with the mirror).
                cols[g].append(jax.device_get(takes[g](out)))
    return [
        jax.tree.map(lambda *a, n=n: np.concatenate(a)[:n], *c)
        for c, n in zip(cols, ns)
    ]


def _probe_stiff_blocks(idx, probe_s, probe_u, zero, zero_f, act, rows_m):
    """Per group: ``S``'s three Hermite diagonals and ``U``'s columns."""
    ns, nl, nm, *_r, nz = zero.shape
    dtype = zero.dtype
    s_parts = []
    # S is block-tridiagonal in m: sources three apart never share an output
    # row, so three probe families (m mod 3) give every block.
    for res in range(3):
        src = np.arange(res, nm, 3)

        def make(k, x, n, j, v, src=src):
            e = jnp.zeros((k.shape[0], ns, nl, nm, n), dtype)
            e = e.at[:, :, :, src].set(_unit(n, j, dtype))
            return _scatter(k, x, e, zero if v is None else v)

        def take(k, x, out, src=src):
            pad = jnp.pad(_gather(k, x, out), ((0, 0),) * 3 + ((1, 1), (0, 0)))
            return tuple(pad[:, :, :, src + 1 + d] for d in (1, 0, -1))

        s_parts.append(_probe_columns(idx, nz, make, probe_s, take))
    u_parts = []
    for f in act:

        def make_u(k, x, n, j, v, f=f):
            fg = jnp.broadcast_to(_unit(n, j, dtype), (k.shape[0], n))
            v = zero_f if v is None else v
            return v.at[f, k, x, :].set(fg.reshape(k.shape[0], -1, nz))

        def take_u(k, x, out):
            return _gather(k, x, out)[:, :, :, list(rows_m)]

        u_parts.append(_probe_columns(idx, nz, make_u, probe_u, take_u))
    return [
        (tuple(p[g] for p in s_parts), [u[g] for u in u_parts]) for g in range(len(idx))
    ]


def _factor_chain_group(
    k, x, s_parts, u_cols, zero, w_all, act, rows_m, cols_m, gdt, spec, e_mirror
) -> tuple[_ChainGroup, Any]:
    """Assemble, factor and couple the probed blocks of one chain group."""
    ns, nl, nm, *_r, nz = zero.shape
    dtype = zero.dtype
    c, n = k.shape[0], k.shape[1] * nz
    lower, diag, upper = (np.zeros((c, ns, nl, nm, n, n), dtype) for _ in range(3))
    for res, got in enumerate(s_parts):
        src = np.arange(res, nm, 3)
        for shift, part in zip((1, 0, -1), got):
            rows = src + shift
            ok = (rows >= 0) & (rows < nm)
            col = np.moveaxis(part, 0, -1)[:, :, :, ok]  # (c, s, l, src, N, N)
            {1: lower, 0: diag, -1: upper}[shift][:, :, :, rows[ok]] = col
    es = []
    for sp in spec:
        e: list[Any] = [None, None]
        if e_mirror is not None and sp[0] in e_mirror[0]:
            e = [jnp.tile(a[np.array(sp)], k.shape[1]) for a in e_mirror[1:]]
        es.append(e)
    u = jnp.moveaxis(jnp.stack(u_cols, -1), 0, -1)
    u = u.reshape(c, ns, nl, len(rows_m), n, -1)  # last axis (f, N)
    wq = w_all[:, :, list(cols_m)][:, :, :, act][:, :, :, :, k, x, :]
    wq = jnp.transpose(wq, (4, 3, 0, 1, 2, 5, 6)).reshape(
        c, len(act), ns, nl, len(cols_m), n
    )
    raw = ((lower, diag, upper), es)
    return _refactor_group(
        _ChainGroup(_static(k), _static(x), (), u, wq, *[jnp.zeros(())] * 3),
        raw,
        gdt,
        rows_m,
        cols_m,
        spec,
    ), raw


@partial(jax.jit, donate_argnums=0)
def _put(full: jnp.ndarray, part: jnp.ndarray, i: int) -> jnp.ndarray:
    return jax.lax.dynamic_update_slice_in_dim(full, part, i, 0)


def _defect(sq) -> float:
    """Relative defect from summed squared norms (defect, block) of lo and up."""
    sq = np.asarray(sq, np.float64)
    return float(np.sqrt(sq[::2] / np.maximum(sq[1::2], 1e-60)).max())


def _factor_chunked(sel, e, gdt, chunk_bytes: float = 4e8):
    """``_thomas_factor`` over chain slices of the host blocks.

    Chains do not couple, so slices factor independently into preallocated
    outputs: the peak is the factor plus one slice, not the factor plus every
    input block and temporary (10 GB for a 7 GB factor at 64x64, (8, 16)).
    """
    c = sel[0].shape[0]
    parts = np.array_split(
        np.arange(c), int(min(c, max(1, -(-sum(b.nbytes for b in sel) // chunk_bytes))))
    )
    full: Any = None
    err: Any = 0.0
    for ix in parts:
        a, b = int(ix[0]), int(ix[-1]) + 1
        fac, er = _thomas_factor(*[jnp.asarray(x[a:b]) for x in sel], *e, gdt)
        err = err + np.asarray(er, np.float64)
        if len(parts) == 1:
            return fac, _defect(err)
        dinv, d, lo, up, *ee = fac
        lead = [*dinv, d, lo, up]
        if full is None:
            full = [jnp.zeros((c, *x.shape[1:]), x.dtype) for x in lead]
        full = [_put(f, x, a) for f, x in zip(full, lead)]
    return (tuple(full[: len(dinv)]), *full[len(dinv) :], *ee), _defect(err)


def _refactor_group(g: _ChainGroup, raw, gdt, rows_m, cols_m, spec) -> _ChainGroup:
    """Factor one chain group at ``gdt`` from its probed (host) blocks."""
    blocks, es = raw
    facs, errs = [], []
    for sp, e in zip(spec, es):
        sel = [b[:, list(sp)] for b in blocks]
        fac, err = _factor_chunked(sel, e, gdt)
        facs.append(fac)
        errs.append(err)
    if not max(errs) < 1e-4:
        raise ValueError(
            f"Hermite off-diagonal blocks are not multiples of one parallel-"
            f"gradient matrix (defect {max(errs):.1e}); the structured factor "
            "needs it"
        )
    factors = tuple(facs)
    nm = blocks[1].shape[3]
    cap, dr, dc = _capacitance(factors, g.u, g.w, gdt, rows_m, cols_m, nm, spec)
    return dataclasses.replace(g, factors=factors, cap=cap, dr=dr, dc=dc)


def refactor_chain_implicit_linear(
    op: ChainImplicitLinear, dt: float, *, free: bool = False
) -> Any:
    """``op`` at a new ``dt``: refactor from the probed blocks, no new probes.

    Same pytree structure, so a scan compiled for ``op`` reuses its executable.
    ``free`` releases ``op``'s device factors first (two copies do not fit at
    production size); ``op`` is unusable afterwards.
    """
    if free:
        _FACTORS.clear()  # the reuse cache may hold ``op``
        for g in op.groups:
            for a in jax.tree.leaves((g.factors, g.cap, g.dr, g.dc)):
                a.delete()
    raws = op.__dict__["_raw"]
    gdt = ARS_TABLEAUX[op.scheme][1][1][1] * float(dt)
    groups = tuple(
        _refactor_group(g, r, gdt, op.rows_m, op.cols_m, op.spec)
        for g, r in zip(op.groups, raws)
    )
    new = dataclasses.replace(op, groups=groups, dt=float(dt))
    for key in ("_raw", "_rhs"):  # host-side extras, not pytree fields
        if key in op.__dict__:
            object.__setattr__(new, key, op.__dict__[key])
    return new


class ImexDtControl:
    """Chunk-level dt for the chain IMEX step (``fixed_dt = false``).

    Between chunks: ``dt = min(cfl 8.5 / w_nl, cfl 2.5 / rho, dt_err,
    dt_max)``. ``rho`` is the spectral radius of the explicit linear remainder
    (drifts, ion mirror, drive) by Arnoldi at startup and 2.5 the measured
    ARK3/ARS3 interval on it (0.82 x 3.6); ``w_nl`` the bracket CFL rate of the
    current fields; 8.5 = 5 x 1.7 is the saturated-turbulence limit measured
    on Cyclone KE (stable at dt 0.1, not 0.2, where the bracket rate is 80); ``dt_err`` the embedded
    ARK3(2) estimate against ``rtol``. The factor is rebuilt only when dt
    moves by more than 20% (down) or grows past 1.2x, and a non-finite chunk
    is rejected and redone at half the step.
    """

    def __init__(self, op, cfl_rate, *, dt_max, cfl=0.9, rtol=1e-2, iters=20):
        rhs, project = op.__dict__["_rhs"]
        self.cfl, self.rtol, self.dt_max = float(cfl), float(rtol), float(dt_max)

        def probe(op, G):
            dG, fields = rhs(G)
            _, err = op.ars_step(G, dG, lambda s: rhs(s)[0], project, with_error=True)
            rel = jnp.linalg.norm(err) / (self.rtol * jnp.linalg.norm(G) + 1e-30)
            return rel, cfl_rate(fields)

        self._probe = jax.jit(probe)
        scale = 1.0e-8  # rhs(s v) / s is linear in v to O(s |v|^2)

        @jax.jit
        def remainder(op, v):  # on projected states: the ky < 0 rows are rebuilt
            v = project(v)
            return project(rhs(scale * v)[0] / scale - op.linear(v))

        rng = np.random.default_rng(0)
        shape = op.shape
        v = jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape))
        q = [v / jnp.linalg.norm(v)]
        h = np.zeros((iters + 1, iters), complex)
        for j in range(iters):  # Arnoldi, as measured_streaming_frequency
            w = remainder(op, q[j].astype(op.groups[0].cap.dtype)).astype(q[j].dtype)
            for i, qi in enumerate(q):
                h[i, j] = complex(jnp.vdot(qi, w))
                w = w - h[i, j] * qi
            h[j + 1, j] = float(jnp.linalg.norm(w))
            q.append(w / max(h[j + 1, j], 1e-30))
        self.rho = float(np.abs(np.linalg.eigvals(h[:iters, :iters])).max())

    def propose(self, op, G) -> float:
        """Next dt from the state at the end of a chunk."""
        rel, w_nl = (float(x) for x in self._probe(op, G))
        dt = float(op.dt)
        dt_err = dt * min(1.5, max(0.5, 0.9 * max(rel, 1e-12) ** (-1.0 / 3.0)))
        dt_cfl = self.cfl * min(8.5 / max(w_nl, 1e-30), 2.5 / max(self.rho, 1e-30))
        self.last = dict(
            err=rel, bracket=w_nl, remainder=self.rho, dt_err=dt_err, dt_cfl=dt_cfl
        )
        return min(dt_err, dt_cfl, self.dt_max)

    def retune(self, op, dt: float):
        """``op`` at ``dt`` if it moved by more than 20%, else ``op``."""
        if dt < float(op.dt) / 1.2 or dt > 1.2 * float(op.dt):
            return refactor_chain_implicit_linear(op, dt, free=True)
        return op


def _static(a: np.ndarray) -> tuple:
    return tuple(map(tuple, np.asarray(a).tolist()))


def _block_solve(factors, spec, x: jnp.ndarray) -> jnp.ndarray:
    """``(I - gdt S)^-1 x`` per species group (species never couple in ``S``)."""
    if len(spec) == 1:
        return _thomas_solve(factors[0], x)
    out = x
    for fac, sp in zip(factors, spec):
        out = out.at[:, list(sp)].set(_thomas_solve(fac, x[:, list(sp)]))
    return out


@partial(jax.jit, static_argnames=("rows_m", "cols_m", "nm", "spec"))
def _capacitance(factors, u, wq, gdt, rows_m, cols_m, nm, spec):
    """Equilibrated ``(I - gdt Q B^-1 U)^-1`` and its row/column scalings."""
    c, ns, nl, _rows, n, k = u.shape
    dtype = u.dtype

    def column(i):
        ui = (
            jnp.zeros((c, ns, nl, nm, n), dtype)
            .at[:, :, :, list(rows_m)]
            .set(u[..., i])
        )
        y = _block_solve(factors, spec, ui)[:, :, :, list(cols_m)]
        return jnp.einsum("cfslmn,cslmn->cfn", wq, y, precision=_HI).reshape(c, -1)

    qbu = jnp.moveaxis(
        jax.lax.map(column, jnp.arange(k), batch_size=_PROBE_BATCH), 0, -1
    )
    cap = jnp.eye(k, dtype=dtype) - gdt * qbu
    # Equilibrate before inverting: near-zonal chains carry field blocks four
    # orders apart (A_par against phi), cond 2e8 raw and ~1e3 balanced.
    dr = jnp.ones(cap.shape[:2], jnp.real(cap).dtype)
    dc = jnp.ones_like(dr)
    for _ in range(8):
        dr = dr / jnp.sqrt(jnp.abs(dr[:, :, None] * cap * dc[:, None]).max(2))
        dc = dc / jnp.sqrt(jnp.abs(dr[:, :, None] * cap * dc[:, None]).max(1))
    cap = jnp.linalg.inv(dr[:, :, None] * cap * dc[:, None])
    return cap, dr.astype(dtype), dc.astype(dtype)


def build_chain_implicit_linear(
    split_rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    fields: Callable[[jnp.ndarray], jnp.ndarray],
    shape: tuple[int, ...],
    dt: float,
    *,
    modes: np.ndarray,
    scheme: str = "imex-ars3",
    dtype=jnp.complex64,
    check_tol: float | None = None,
    mirror_rhs: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> ChainImplicitLinear:
    """Probe the stiff linear terms per chain and factor ``I - gamma dt L``.

    ``fields(G)`` is the linear field solve, stacked ``(nf, Nky, Nkx, Nz)``;
    ``split_rhs(G, F)`` the implicit terms' RHS of ``G`` with the fields ``F``
    imposed. ``L G = split_rhs(G, fields(G))``; ``S = split_rhs(., 0)`` and
    ``U = split_rhs(0, .)`` are probed separately, ``Q`` from unit moments.
    ``modes`` (``Nky x Nkx`` bool) selects what is solved: every ``ky >= 0``
    mode, dealiased or not (leaving dealiased-out modes explicit makes their
    stiff terms explicit). ``ky < 0`` rows pass through; the projector
    rebuilds them. ``mirror_rhs`` is the part of ``split_rhs`` that is the
    mirror force (species it is nonzero for get the ``Nl N`` blocks). The
    factorization is checked by the residual of one solve.
    """
    ns, nl, nm, _nky, _nkx, nz = shape
    split, fld = cached_jit(split_rhs), cached_jit(fields)
    zero = jnp.zeros(shape, dtype)
    zero_f = jnp.zeros_like(fld(zero))
    # Graph digests cover geometry and parameters bitwise: same deck+dt, same factor.
    keyed = getattr(split, "key")(zero, zero_f), getattr(fld, "key")(zero)
    bits = np.asarray(modes, bool).tobytes()
    reuse = (
        None if None in keyed else (*keyed, shape, dt, scheme, dtype, check_tol, bits)
    )
    if reuse in _FACTORS:
        return _FACTORS[reuse]
    _FACTORS.clear()  # one factor alive at a time: it can fill the device

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
    for chain in _chains(cached_jit(lin), shape, np.asarray(modes, dtype=bool), dtype):
        by_len.setdefault(len(chain), []).append(chain)
    idx = [
        (np.array(c)[..., 0], np.array(c)[..., 1]) for _, c in sorted(by_len.items())
    ]
    gdt = ARS_TABLEAUX[scheme][1][1][1] * float(dt)

    spec: tuple[tuple[int, ...], ...] = (tuple(range(ns)),)
    e_mirror, s_only = None, split
    if mirror_rhs is not None:
        mir = cached_jit(mirror_rhs)
        r0, c0 = np.argwhere(modes)[-1]
        # z- and mode-local, the same at every mode: one probe per slot.
        ef = jax.lax.map(
            lambda i: mir(
                jnp.zeros(ns * nl * nm, dtype).at[i].set(1).reshape(ns, nl, nm, 1, 1, 1)
                * jnp.ones(shape, dtype),
                zero_f,
            )[..., r0, c0, :],
            jnp.arange(ns * nl * nm),
        )
        ef = np.asarray(ef).reshape(ns, nl, nm, ns, nl, nm, nz)
        mono = [i for i in range(ns) if np.abs(ef[i, :, :, i]).max() > 0]
        e_lo, e_up = (np.zeros((ns, nm, nl, nl, nz), ef.dtype) for _ in range(2))
        for i in mono:  # e[s, m, l, l'] couples (l', m -+ 1) into (l, m)
            e_lo[i, 1:] = np.diagonal(ef[i, :, :, i], 1, 1, 3).transpose(3, 1, 0, 2)
            e_up[i, :-1] = np.diagonal(ef[i, :, :, i], -1, 1, 3).transpose(3, 1, 0, 2)
        plain = tuple(i for i in range(ns) if i not in mono)
        spec = tuple(g for g in (plain, tuple(mono)) if g)
        e_mirror = (tuple(mono), jnp.asarray(e_lo, dtype), jnp.asarray(e_up, dtype))

        # S probes have F = 0, where the mirror is the probed z-local ef:
        # applying it is cheaper than a second RHS assembly per probe.
        ef_s = jnp.asarray(np.einsum("plmpqnz->pqnlmz", ef), dtype)

        def s_only(v, F):
            return split(v, F) - jnp.einsum("sqnlmz,slmyxz->sqnyxz", ef_s, v)

    probe_s = cached_jit(jax.vmap(lambda v: s_only(v, zero_f)))
    probe_u = cached_jit(jax.vmap(lambda F: split(zero, F)))
    probed = _probe_stiff_blocks(idx, probe_s, probe_u, zero, zero_f, act, rows_m)
    built = [
        _factor_chain_group(
            k, x, s_parts, u_cols, zero, w_all, act, rows_m, cols_m, gdt, spec, e_mirror
        )
        for (k, x), (s_parts, u_cols) in zip(idx, probed)
    ]
    groups = [g for g, _raw in built]
    mask = np.zeros(shape[3:5], bool)
    for k, x in idx:
        mask[k, x] = True
    on = jnp.asarray(mask, dtype)[None, None, None, :, :, None]
    op = ChainImplicitLinear(
        tuple(groups), tuple(shape), float(dt), scheme, rows_m, cols_m, lin, spec
    )
    # One solve's residual certifies every structural assumption at once
    # (Hermite tridiagonality, z-local fields, the chain partition).
    r = on * jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype)

    def residual(op):  # only the solved rows: ky < 0 rows are rebuilt
        y = op.solve(r)
        return float(jnp.linalg.norm(on * (y - gdt * lin(y) - r)) / jnp.linalg.norm(r))

    err = residual(op)
    # Refine (one more linear apply and solve per stage) only where needed:
    # 32x32, (4, 8), complex64: 1e-3 at dt 0.05, 3e-2 at 0.2, 0.8 at 0.8.
    if e_mirror is not None and err > 5.0e-3:
        op = dataclasses.replace(op, on=on)
        err = residual(op)
    # I - gamma dt L reaches cond 4e5 on the Cyclone tutorial box (A_par at
    # small k_perp), so a complex64 solve, dense or structured, leaves a
    # ~5e-3 residual; a broken structural assumption leaves O(1).
    if check_tol is None:
        check_tol = 5.0e-2 if jnp.finfo(dtype).bits <= 32 else 1.0e-6
    if not err < check_tol:
        raise ValueError(
            f"structured implicit factor residual {err:.2e} > {check_tol:.0e}: the "
            'stiff operator is not block-tridiagonal in Hermite with z-local fields; use method = "rk4-pi"'
        )
    object.__setattr__(op, "_raw", tuple(raw for _g, raw in built))
    _FACTORS.update({reuse: op} if reuse else {})
    return op


StatsSolveStepFn = Callable[
    [jnp.ndarray, jnp.ndarray, ImplicitSolveStats],
    tuple[jnp.ndarray, ImplicitSolveStats],
]
DiagnosticStepFn = Callable[
    [tuple[Any, ...], Any],
    tuple[tuple[Any, ...], tuple[Any, Any]],
]


def _advance_imex_nonlinear_state(
    G: jnp.ndarray,
    stats: Any,
    *,
    dt_val: jnp.ndarray,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: Callable[[jnp.ndarray, jnp.ndarray, Any], tuple[jnp.ndarray, Any]],
    project_state: ProjectFn,
) -> tuple[jnp.ndarray, Any]:
    """One IMEX step, threading whatever solve status ``solve_step`` carries.

    SSPX3 takes three implicit solves per step, so the status is threaded
    through all three rather than summarizing only the last one.
    """

    if method == "sspx3":

        def _euler_step(
            G_state: jnp.ndarray, carried: Any, dt_stage: jnp.ndarray
        ) -> tuple[jnp.ndarray, Any]:
            rhs_stage = G_state + dt_stage * nonlinear_term(G_state)
            return solve_step(G_state, rhs_stage, carried)

        G1, stats = _euler_step(G, stats, _SSPX3_ADT * dt_val)
        G2_euler, stats = _euler_step(G1, stats, _SSPX3_ADT * dt_val)
        G2 = project_state((1.0 - _SSPX3_W1) * G + (_SSPX3_W1 - 1.0) * G1 + G2_euler)
        G3, stats = _euler_step(G2, stats, _SSPX3_ADT * dt_val)
        return (
            (1.0 - _SSPX3_W2 - _SSPX3_W3) * G
            + _SSPX3_W3 * G1
            + (_SSPX3_W2 - 1.0) * G2
            + G3
        ), stats

    rhs = G + dt_val * nonlinear_term(G)
    return solve_step(G, rhs, stats)


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

    def _stats_free(
        G_state: jnp.ndarray, rhs: jnp.ndarray, carried: Any
    ) -> tuple[jnp.ndarray, Any]:
        return solve_step(G_state, rhs), carried

    G_new, _ = _advance_imex_nonlinear_state(
        G,
        None,
        dt_val=dt_val,
        method=method,
        nonlinear_term=nonlinear_term,
        solve_step=_stats_free,
        project_state=project_state,
    )
    return G_new


def advance_imex_nonlinear_state_with_stats(
    G: jnp.ndarray,
    stats: ImplicitSolveStats,
    *,
    dt_val: jnp.ndarray,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: StatsSolveStepFn,
    project_state: ProjectFn,
) -> tuple[jnp.ndarray, ImplicitSolveStats]:
    """Advance one IMEX step and fold every implicit solve into ``stats``."""

    return _advance_imex_nonlinear_state(
        G,
        stats,
        dt_val=dt_val,
        method=method,
        nonlinear_term=nonlinear_term,
        solve_step=solve_step,
        project_state=project_state,
    )


def make_imex_diagnostic_step(
    *,
    method: str,
    nonlinear_term: NonlinearTermFn,
    solve_step: SolveStepFn,
    project_state: ProjectFn,
    state_dtype: jnp.dtype,
    real_dtype: jnp.dtype,
    dt_val: jnp.ndarray,
    compute_fields_fn: FieldSolveFn,
    cache: Any,
    params: Any,
    term_cfg: Any,
    external_phi: jnp.ndarray | float | None,
    compute_diag_from_state: DiagnosticFn,
    diagnostics_stride: int,
    select_diagnostics_fn: Callable[..., Any],
    show_progress: bool,
    steps: int,
    progress_total: jnp.ndarray,
    emit_progress_fn: Callable[..., jnp.ndarray],
    use_collision_split: bool = False,
    damping: Any | None = None,
    collision_scheme: str = "implicit",
    apply_collision_split_fn: CollisionSplitFn | None = None,
    solve_step_with_stats: StatsSolveStepFn | None = None,
) -> DiagnosticStepFn:
    """Build one IMEX diagnostic scan step with injected runtime seams.

    With ``solve_step_with_stats`` the carry grows one
    :class:`~gkx.solvers_linear_implicit.ImplicitSolveStats` leaf and every
    implicit solve folds into it, which is how this route gets a convergence
    channel a traced step cannot raise on (queue row Q15). Without it the step
    keeps exactly the five-element carry, and the same solve, it had.
    """

    carries_solve_stats = solve_step_with_stats is not None

    def _advance(G: jnp.ndarray, stats: Any) -> tuple[jnp.ndarray, Any]:
        if solve_step_with_stats is None:
            return (
                advance_imex_nonlinear_state(
                    G,
                    dt_val=dt_val,
                    method=method,
                    nonlinear_term=nonlinear_term,
                    solve_step=solve_step,
                    project_state=project_state,
                ),
                stats,
            )
        return advance_imex_nonlinear_state_with_stats(
            G,
            stats,
            dt_val=dt_val,
            method=method,
            nonlinear_term=nonlinear_term,
            solve_step=solve_step_with_stats,
            project_state=project_state,
        )

    def step(
        carry: tuple[Any, ...],
        idx: Any,
    ) -> tuple[tuple[Any, ...], tuple[Any, Any]]:
        G, G_prev_step, fields_prev_step, diag_prev, t_prev = carry[:5]
        solve_stats = carry[5] if carries_solve_stats else None
        G_new, solve_stats = _advance(G, solve_stats)
        if use_collision_split and damping is not None:
            if apply_collision_split_fn is None:
                raise ValueError(
                    "apply_collision_split_fn is required when collision split is active"
                )
            G_new = apply_collision_split_fn(G_new, damping, dt_val, collision_scheme)
        G_new = project_state(G_new)
        # Keep scan carry dtype stable under mixed-precision scalar constants.
        G_new = jnp.asarray(G_new, dtype=state_dtype)
        t_new = jnp.asarray(t_prev + dt_val, dtype=real_dtype)
        fields_new = compute_fields_fn(
            G_new, cache, params, terms=term_cfg, external_phi=external_phi
        )

        def _compute_diag():
            return compute_diag_from_state(
                G_new, fields_new, G_prev_step, fields_prev_step, dt_val
            )

        diag = select_diagnostics_fn(
            idx,
            diagnostics_stride=diagnostics_stride,
            diag_prev=diag_prev,
            compute_diag_fn=_compute_diag,
            steps=steps,
        )
        G_new = emit_progress_fn(
            G_new,
            show_progress=show_progress,
            diag=diag,
            idx=idx,
            steps=steps,
            t_new=t_new,
            progress_total=progress_total,
        )
        next_carry: tuple[Any, ...] = (G_new, G_new, fields_new, diag, t_new)
        if carries_solve_stats:
            next_carry = (*next_carry, solve_stats)
        return next_carry, (diag, t_new)

    return step


def run_imex_diagnostic_scan(
    step_fn: DiagnosticStepFn,
    initial_carry: tuple[Any, ...],
    *,
    steps: int,
    checkpoint: bool,
    return_solve_stats: bool = False,
) -> DiagnosticScanOutput | tuple[Any, ...]:
    """Run the fixed-step IMEX diagnostic scan.

    ``return_solve_stats=True`` appends the carry's trailing
    :class:`~gkx.solvers_linear_implicit.ImplicitSolveStats`, or ``None`` when
    the step was built without one. A caller that does not ask never reads the
    leaf, so XLA drops the status arithmetic and the compiled scan is the graph
    it was before.
    """

    scan_step = jax.checkpoint(step_fn) if checkpoint else step_fn
    idx = jnp.arange(steps, dtype=jnp.int32)
    final_carry, scan_diag_out = jax.lax.scan(
        scan_step,
        initial_carry,
        idx,
        length=steps,
    )
    G_final = final_carry[0]
    if not return_solve_stats:
        return G_final, scan_diag_out
    solve_stats = final_carry[5] if len(final_carry) > 5 else None
    return G_final, scan_diag_out, solve_stats


__all__ = [
    "ARS_TABLEAUX",
    "ChainImplicitLinear",
    "IMEX_CHAIN_METHODS",
    "ImexDtControl",
    "StatsSolveStepFn",
    "advance_imex_nonlinear_state",
    "advance_imex_nonlinear_state_with_stats",
    "build_chain_implicit_linear",
    "imex_fixed_point_guess",
    "integrate_cached_imex_scan",
    "make_imex_diagnostic_step",
    "make_imex_nonlinear_term",
    "make_imex_solve_step",
    "make_imex_solve_step_with_stats",
    "refactor_chain_implicit_linear",
    "run_imex_diagnostic_scan",
    "solve_imex_step",
    "solve_imex_step_with_stats",
]
