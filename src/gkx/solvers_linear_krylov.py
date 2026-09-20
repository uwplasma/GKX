"""Public facade preserving the API above independently tested eigenmode kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from gkx.solvers_linear_adaptive_propagator import (
    AdaptivePropagatorSolution as AdaptivePropagatorSolution,
    adaptive_propagator_eigenpair,
    certifiable_residual_tolerance,
)
from gkx.solvers_linear_krylov_algorithms import (
    _advance_imex2,
    _apply_operator,
    _assemble_rhs_cached_novjp,
    _compute_damping,
    _linked_covered_mode_mask,
    _normalize,
    _project_to_linked_cover,
    _require_linked_cover_seed,
)
from gkx.solvers_linear_krylov_algorithms import (
    InnerSolveStats,
    _arnoldi,
    _shift_invert_eigenpair_with_inner_stats,
    _validate_shift_solve_method,
    build_shift_invert_preconditioner,
    _mode_family_sign,
    _omega_scale,
    _physical_omega,
    _select_by_overlap,
    _select_by_target,
    dominant_eigenpair_cached,
    dominant_eigenpair_power,
    dominant_eigenpair_propagator_cached,
    dominant_eigenpair_shift_invert_cached,
)
from gkx.solvers_linear_precond_pr3 import PR3_PRECOND_NAMES, build_pr3_factors


@dataclass(frozen=True)
class KrylovConfig:
    """Controls for the Krylov-based eigen solver.

    The default ``method="adaptive"`` is the residual-certified eigensolve the
    runtime already uses for generic contracts. Every method returns only a pair
    that passes an original-operator residual gate unless ``certify=False``.
    """

    krylov_dim: int = 24
    restarts: int = 2
    omega_min_factor: float = 0.0
    omega_target_factor: float = 0.0
    omega_cap_factor: float = 2.0
    omega_sign: int = 0
    method: str = "adaptive"
    # Matches the ``dominant_eigenpair(power_iters=40)`` signature default: one
    # route, one cost, whichever door it is entered by. The two used to disagree
    # by 5x, and the larger value bought nothing. Measured on the shipped Cyclone
    # deck at (Nl,Nm)=(4,8), ky=0.3, float32, against that rung's certified
    # adaptive eigenpair gamma=0.10128645: the route's residual is 9.501e-01 at
    # 40 applies and 9.467e-01 at 200, both against a 1.192e-04 gate, so 5x the
    # propagator applies moves the residual by 0.4% of an O(1) quantity and the
    # pair is rejected either way. The ladder only reaches 6.2e-03 at 5000
    # applies and then stalls -- 5.96e-03 at 10000 -- so no affordable setting
    # certifies and the value cannot be chosen for accuracy. It is therefore
    # chosen for cost, and for agreeing with the public signature.
    power_iters: int = 40
    power_dt: float = 0.01
    propagator_steps: int = 1
    shift: complex | None = None
    shift_source: str = "propagator"
    shift_tol: float = 1.0e-4
    shift_maxiter: int = 50
    shift_restart: int = 20
    # Compatibility alias: "batched", "incremental" and "flexible" are validated
    # but all run the one SOLVAX FGMRES solve; the label is not a compile key.
    shift_solve_method: str = "batched"
    shift_preconditioner: str | None = "auto"
    # Only ``pr3-cm`` reads the two fields below. ``shift_precond_alpha`` is its
    # Peaceman-Rachford parameter; None takes Q7's scalar symbol-bound rule
    # ``alpha = -sqrt(s1 d)``, which that row measured as good as a per-rung scan
    # at Nz=96. ``shift_precond_block_solve`` picks how its z-local block is
    # inverted: "auto" takes the exact block-Thomas plus Sherman-Morrison solve
    # when the block really is l-tridiagonal plus rank one and the dense batched
    # inverse when it is not, "block-thomas" refuses instead of falling back, and
    # "dense" is the control the fallback measures against.
    shift_precond_alpha: float | complex | None = None
    shift_precond_block_solve: str = "auto"
    shift_selection: str = "targeted"
    # Certified against the original operator in the working dtype, so this is
    # raised to that dtype's noise floor; the default is a float64 gate.
    shift_outer_residual_tol: float = 1.0e-6
    mode_family: str = "auto"
    fallback_method: str = "propagator"
    fallback_real_floor: float = -1.0e-6
    continuation: bool = False
    continuation_selection: str = "overlap"
    # Raw propagator, power and Arnoldi pairs have no convergence test of their
    # own; they are gated at the shift_outer_residual_tol outer gate and raise
    # on failure. False is the explicit opt-out that returns such a pair and
    # reports it as uncertified. Shift-invert, sparse and adaptive always gate.
    certify: bool = True


@dataclass(frozen=True)
class EigenSolveStatus:
    """Host-side status of the pair ``dominant_eigenpair`` returned.

    ``residual`` is the original-operator relative residual
    ``||A v - lambda v|| / max(||A v||, |lambda| ||v||)`` of the returned pair,
    ``tolerance`` the gate its route applied and ``certified`` whether it passed.
    Every gate raises on failure, so ``certified`` is false only for a raw
    route called with ``certify=False``. ``route`` names the method that
    produced the pair: a shift-invert fallback reports its fallback method.
    ``inner`` summarizes the inner FGMRES solves of the shift-invert build that
    produced the pair and is ``None`` for every other route.
    """

    method: str
    route: str
    residual: float
    tolerance: float
    certified: bool
    inner: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "route": self.route,
            "residual": self.residual,
            "tolerance": self.tolerance,
            "certified": self.certified,
            "inner": None if self.inner is None else dict(self.inner),
        }


def eigen_status_payload(status: EigenSolveStatus | None) -> dict[str, Any]:
    """Flatten an eigen solve status into scalar keys; ``None`` when none ran.

    This is the one spelling of those keys. A runtime result's ``summary()``
    and the saved ``*.summary.json`` both read it, so a run held in memory and
    the same run read back from disk report the same status under the same
    names.
    """

    inner = None if status is None else status.inner
    return {
        "eigen_route": None if status is None else status.route,
        "eigen_residual": None if status is None else status.residual,
        "eigen_tolerance": None if status is None else status.tolerance,
        "eigen_certified": None if status is None else status.certified,
        "eigen_inner_converged": None if inner is None else inner["converged"],
    }


_StatusCallback = Callable[[str], None] | None

# Base residual gate for the certified adaptive branch. The effective gate is
# floored by certifiable_residual_tolerance at what the working precision can
# express, so a converged complex64 eigenpair is not rejected for failing an
# f64-only tolerance.
_ADAPTIVE_BASE_TOL = 1.0e-9


def _status(status_callback: _StatusCallback, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def _normalized_config(options: Mapping[str, Any]) -> KrylovConfig:
    """Normalize public options once at the dispatch boundary."""
    value = options.__getitem__
    mode_family = str(value("mode_family"))
    omega_sign = int(value("omega_sign"))
    mode_family_sign = _mode_family_sign(mode_family)
    omega_sign_eff = omega_sign if omega_sign != 0 else mode_family_sign
    return KrylovConfig(
        method=str(value("method")).strip().lower(),
        krylov_dim=max(int(value("krylov_dim")), 1),
        restarts=max(int(value("restarts")), 1),
        omega_min_factor=float(value("omega_min_factor")),
        omega_target_factor=float(value("omega_target_factor")),
        omega_cap_factor=float(value("omega_cap_factor")),
        omega_sign=omega_sign_eff,
        power_iters=max(int(value("power_iters")), 1),
        power_dt=float(value("power_dt")),
        propagator_steps=max(int(value("propagator_steps")), 1),
        shift=value("shift"),
        shift_source=str(value("shift_source")),
        shift_tol=float(value("shift_tol")),
        shift_maxiter=max(int(value("shift_maxiter")), 1),
        shift_restart=max(int(value("shift_restart")), 1),
        shift_solve_method=str(value("shift_solve_method")),
        shift_preconditioner=value("shift_preconditioner"),
        shift_precond_alpha=value("shift_precond_alpha"),
        shift_precond_block_solve=str(value("shift_precond_block_solve")),
        shift_selection=str(value("shift_selection")),
        shift_outer_residual_tol=float(value("shift_outer_residual_tol")),
        mode_family=mode_family,
        fallback_method=str(value("fallback_method")),
        fallback_real_floor=float(value("fallback_real_floor")),
        certify=bool(value("certify")),
    )


def _power_branch(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _status(
        status_callback,
        "running power iteration seed with "
        f"iterations={cfg.power_iters} dt={cfg.power_dt:.6g}",
    )
    return dominant_eigenpair_power(
        v0, cache, params, term_cfg, iterations=cfg.power_iters, dt=cfg.power_dt
    )


def _propagator_branch(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    restarts: int | None = None,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    restarts_use = cfg.restarts if restarts is None else max(int(restarts), 1)
    _status(
        status_callback,
        "running propagator Arnoldi with "
        f"dt={cfg.power_dt:.6g} steps={cfg.propagator_steps} "
        f"horizon={cfg.power_dt * cfg.propagator_steps:.6g} "
        f"dim={cfg.krylov_dim} restarts={restarts_use}",
    )
    return dominant_eigenpair_propagator_cached(
        v0,
        v_ref,
        cache,
        params,
        term_cfg,
        krylov_dim=cfg.krylov_dim,
        restarts=restarts_use,
        dt=cfg.power_dt,
        propagator_steps=cfg.propagator_steps,
        omega_min_factor=cfg.omega_min_factor,
        omega_target_factor=cfg.omega_target_factor,
        omega_cap_factor=cfg.omega_cap_factor,
        omega_sign=cfg.omega_sign,
        select_overlap=bool(select_overlap),
    )


def _arnoldi_branch(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _status(
        status_callback,
        f"running plain Arnoldi with dim={cfg.krylov_dim} restarts={cfg.restarts}",
    )
    return dominant_eigenpair_cached(
        v0,
        v_ref,
        cache,
        params,
        term_cfg,
        krylov_dim=cfg.krylov_dim,
        restarts=cfg.restarts,
        omega_min_factor=cfg.omega_min_factor,
        omega_target_factor=cfg.omega_target_factor,
        omega_cap_factor=cfg.omega_cap_factor,
        omega_sign=cfg.omega_sign,
        select_overlap=bool(select_overlap),
    )


def _target_shift(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    cfg: KrylovConfig,
) -> jnp.ndarray:
    omega_target = cfg.omega_target_factor * _omega_scale(cache, params)
    if cfg.omega_sign != 0:
        omega_target = float(jnp.sign(cfg.omega_sign)) * jnp.abs(omega_target)
    return jnp.asarray(-1j * omega_target, dtype=v0.dtype)


def _shift_seed(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    shift_source_key = cfg.shift_source.strip().lower()
    if cfg.shift is None:
        if shift_source_key == "propagator":
            _status(status_callback, "estimating shift from propagator seed")
            return _propagator_branch(
                v0,
                v_ref,
                cache,
                params,
                term_cfg,
                cfg,
                None,
                restarts=1,
                select_overlap=False,
            )
        if shift_source_key == "target":
            _status(status_callback, "building target-frequency shift")
            return _target_shift(v0, cache, params, cfg), v0
        _status(status_callback, "estimating shift from power iteration seed")
        return _power_branch(v0, cache, params, term_cfg, cfg, None)

    sigma = jnp.asarray(cfg.shift, dtype=v0.dtype)
    if shift_source_key == "propagator":
        _status(status_callback, "using explicit shift with propagator seed vector")
        _shift_seed, v_seed = _propagator_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            None,
            restarts=1,
            select_overlap=select_overlap,
        )
        return sigma, v_seed
    if shift_source_key == "power":
        _status(
            status_callback, "using explicit shift with power-iteration seed vector"
        )
        _shift_seed, v_seed = _power_branch(v0, cache, params, term_cfg, cfg, None)
        return sigma, v_seed
    _status(status_callback, "using explicit shift with reference seed vector")
    return sigma, v_ref


def _shift_selection_flags(shift_selection: str) -> tuple[bool, bool]:
    selection_key = shift_selection.strip().lower()
    select_targeted = selection_key in {"targeted", "target", "auto", "default"}
    select_growth = selection_key in {"targeted", "growth", "auto", "default"}
    return select_targeted, select_growth


def _automatic_shift_preconditioner(params: LinearParams, term_cfg: Any) -> str:
    """Use the cheap streaming inverse unless electromagnetic fields require more."""

    apar = bool(np.asarray(term_cfg.apar) != 0.0)
    bpar = bool(np.asarray(term_cfg.bpar) != 0.0)
    beta = bool(np.asarray(getattr(params, "beta", 0.0)) != 0.0)
    fapar = bool(np.any(np.asarray(getattr(params, "fapar", 0.0)) != 0.0))
    electromagnetic = beta and (bpar or (apar and fapar))
    return "field-corrected" if electromagnetic else "hermite-line"


def _eigenpair_relative_residual(
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
) -> float:
    """Return the scale-invariant residual of one matrix-free eigenpair.

    A zero or non-finite eigenvector is never an eigenvector: its residual is
    infinite rather than the ``0 / floor`` a breakdown would otherwise report.
    """

    vector_norm = float(np.asarray(jnp.linalg.norm(eigenvector)))
    if not np.isfinite(vector_norm) or vector_norm == 0.0:
        return float("inf")
    operator_vec = _apply_operator(eigenvector, cache, params, term_cfg)
    numerator = jnp.linalg.norm(operator_vec - eigenvalue * eigenvector)
    denominator = jnp.maximum(
        jnp.maximum(
            jnp.linalg.norm(operator_vec),
            jnp.abs(eigenvalue) * jnp.linalg.norm(eigenvector),
        ),
        jnp.asarray(1.0e-30, dtype=jnp.real(eigenvector).dtype),
    )
    return float(np.asarray(numerator / denominator))


def _inner_solve_summary(stats: InnerSolveStats, tol: float) -> str:
    """Describe every inner GMRES solve of one shift-invert build on the host."""

    solves = int(np.asarray(stats.solves))
    unconverged = int(np.asarray(stats.unconverged_solves))
    residual = float(np.asarray(stats.max_relative_residual))
    return (
        f"inner converged={unconverged == 0} unconverged={unconverged}/{solves} "
        f"max_relative_residual={residual:.3g} tol={tol:.3g} "
        f"iterations={int(np.asarray(stats.total_iterations))}"
    )


def _inner_solve_status(
    stats: InnerSolveStats,
    tol: float,
    preconditioner: str,
    preconditioner_setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one shift-invert build's inner FGMRES solves as host scalars.

    ``preconditioner_setup`` carries what a preconditioner with host-side setup
    measured while building itself -- for ``pr3-cm``, which z-block solve its
    structural check selected and why, its parameter and the factors' size. It
    is ``None`` for every preconditioner that has no setup to report.
    """

    unconverged = int(np.asarray(stats.unconverged_solves))
    return {
        "converged": unconverged == 0,
        "unconverged_solves": unconverged,
        "solves": int(np.asarray(stats.solves)),
        "max_relative_residual": float(np.asarray(stats.max_relative_residual)),
        "total_iterations": int(np.asarray(stats.total_iterations)),
        "tolerance": float(tol),
        "preconditioner": str(preconditioner),
        "preconditioner_setup": (
            None if preconditioner_setup is None else dict(preconditioner_setup)
        ),
    }


def _certify_raw_eigenpair(
    pair: tuple[jnp.ndarray, jnp.ndarray],
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus]:
    """Gate a propagator, power or Arnoldi pair on its original-operator residual.

    These branches return a Rayleigh or Ritz pair without a convergence test, so
    a short subspace or horizon can return the wrong branch with a plausible
    growth rate. The pair is checked with the shift-invert outer gate's relative
    residual and dtype-floored tolerance. A failing pair raises unless
    ``cfg.certify`` is false, in which case it is returned and reported as
    uncertified.
    """

    eigenvalue, eigenvector = pair
    residual = _eigenpair_relative_residual(
        eigenvalue, eigenvector, cache, params, term_cfg
    )
    tolerance = certifiable_residual_tolerance(
        cfg.shift_outer_residual_tol, eigenvector.dtype
    )
    eig_host = complex(np.asarray(eigenvalue))
    certified = bool(
        np.isfinite(eig_host) and np.isfinite(residual) and residual <= tolerance
    )
    detail = f"residual={residual:.6g}, tolerance={tolerance:.6g}"
    _status(
        status_callback,
        f"{cfg.method} solve finished with "
        f"eig={eig_host.real:.6g}{eig_host.imag:+.6g}j {detail} certified={certified}",
    )
    status = EigenSolveStatus(
        method=cfg.method,
        route=cfg.method,
        residual=residual,
        tolerance=tolerance,
        certified=certified,
    )
    if certified:
        return eigenvalue, eigenvector, status
    if not cfg.certify:
        _status(
            status_callback,
            f"returning an UNCERTIFIED {cfg.method} eigenpair (certify=False): {detail}",
        )
        return eigenvalue, eigenvector, status
    raise RuntimeError(
        f"{cfg.method} eigenpair failed the outer residual gate: {detail}; use "
        "method='adaptive' for a certified solve or pass certify=False to accept "
        "an uncertified pair"
    )


def _shift_invert_fallback(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus] | None:
    fallback_key = cfg.fallback_method.strip().lower()
    _status(
        status_callback, f"shift-invert result rejected; falling back to {fallback_key}"
    )
    result: tuple[jnp.ndarray, jnp.ndarray] | None = None
    if fallback_key == "propagator":
        result = _propagator_branch(
            v0, v_ref, cache, params, term_cfg, cfg, None, select_overlap=False
        )
    elif fallback_key == "arnoldi":
        result = _arnoldi_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            None,
            select_overlap=select_overlap,
        )
    elif fallback_key == "power":
        result = _power_branch(v0, cache, params, term_cfg, cfg, None)
    if result is None:
        return None
    residual = _eigenpair_relative_residual(*result, cache, params, term_cfg)
    _status(status_callback, f"{fallback_key} fallback residual={residual:.3g}")
    residual_tol = certifiable_residual_tolerance(
        cfg.shift_outer_residual_tol, v0.dtype
    )
    if not np.isfinite(residual) or residual > residual_tol:
        return None
    status = EigenSolveStatus(
        method=cfg.method,
        route=fallback_key,
        residual=residual,
        tolerance=residual_tol,
        certified=True,
    )
    return result[0], result[1], status


def _shift_invert_branch(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus]:
    _validate_shift_solve_method(cfg.shift_solve_method)
    residual_tol = certifiable_residual_tolerance(
        cfg.shift_outer_residual_tol, v0.dtype
    )
    _status(
        status_callback,
        "preparing shift-invert solve with "
        f"dim={cfg.krylov_dim} restarts={cfg.restarts} "
        f"gmres_maxiter={cfg.shift_maxiter} restart={cfg.shift_restart} "
        f"tol={cfg.shift_tol:.3g}",
    )
    sigma, v_init = _shift_seed(
        v0,
        v_ref,
        cache,
        params,
        term_cfg,
        cfg,
        status_callback,
        select_overlap=select_overlap,
    )
    sigma_host = complex(np.asarray(sigma))
    _status(
        status_callback,
        f"shift-invert sigma={sigma_host.real:.6g}{sigma_host.imag:+.6g}j",
    )
    select_targeted, select_growth = _shift_selection_flags(cfg.shift_selection)
    requested = str(cfg.shift_preconditioner).strip().lower()
    automatic = requested in {"auto", "physics-auto", "physics_auto"}
    preconditioner = (
        _automatic_shift_preconditioner(params, term_cfg)
        if automatic
        else cfg.shift_preconditioner
    )
    preconditioners = (
        (preconditioner, "field-corrected")
        if automatic and preconditioner == "hermite-line"
        else (preconditioner,)
    )
    precond_meta: dict[str, Any] | None = None
    for attempt, mode in enumerate(preconditioners):
        precond_factors = None
        if str(mode).strip().lower() in PR3_PRECOND_NAMES:
            # pr3-cm is the one preconditioner with host-side setup: Nl*Nm probes
            # of the z-local operator and a factorization, built once per shift
            # outside the jit and handed in as an operand.
            _status(status_callback, "building pr3-cm factors")
            precond_factors, precond_meta = build_pr3_factors(
                v_init,
                cache,
                params,
                term_cfg,
                sigma,
                alpha=cfg.shift_precond_alpha,
                block_solve=cfg.shift_precond_block_solve,
            )
            _status(
                status_callback,
                "pr3-cm factors: "
                f"{precond_meta['block_solve']} solve, "
                f"alpha={precond_meta['alpha'][0]:.6g}"
                f"{precond_meta['alpha'][1]:+.6g}j "
                f"({precond_meta['alpha_source']}), "
                f"{precond_meta['factor_bytes'] / 1e6:.1f} MB; "
                f"{precond_meta['block_solve_reason']}",
            )
        _status(status_callback, f"running shift-invert Arnoldi ({mode})")
        eig_si, vec_si, inner_stats = _shift_invert_eigenpair_with_inner_stats(
            v_init,
            v_ref,
            cache,
            params,
            term_cfg,
            krylov_dim=cfg.krylov_dim,
            restarts=cfg.restarts,
            sigma=sigma,
            omega_min_factor=cfg.omega_min_factor,
            omega_target_factor=cfg.omega_target_factor,
            omega_cap_factor=cfg.omega_cap_factor,
            omega_sign=cfg.omega_sign,
            gmres_tol=cfg.shift_tol,
            gmres_maxiter=cfg.shift_maxiter,
            gmres_restart=cfg.shift_restart,
            shift_preconditioner=mode,
            select_targeted=select_targeted,
            select_growth=select_growth,
            select_overlap=bool(select_overlap),
            precond_factors=precond_factors,
        )
        eig_host = complex(np.asarray(eig_si))
        residual = _eigenpair_relative_residual(eig_si, vec_si, cache, params, term_cfg)
        inner = _inner_solve_summary(inner_stats, cfg.shift_tol)
        _status(
            status_callback,
            "shift-invert solve finished with "
            f"eig={eig_host.real:.6g}{eig_host.imag:+.6g}j residual={residual:.3g}; "
            f"{inner}",
        )
        nonfinite_pair = not np.isfinite(eig_host.real) or not np.isfinite(
            eig_host.imag
        )
        growth_floor_failed = select_growth and eig_host.real < cfg.fallback_real_floor
        residual_failed = not np.isfinite(residual) or residual > residual_tol
        need_fallback = nonfinite_pair or growth_floor_failed or residual_failed
        if not need_fallback or attempt + 1 == len(preconditioners):
            break
        _status(
            status_callback,
            "line solve rejected; retrying with exact low-moment field correction",
        )
    if need_fallback and cfg.fallback_method.strip().lower() != "none":
        fallback = _shift_invert_fallback(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
        if fallback is not None:
            return fallback
    if need_fallback:
        if residual_failed:
            raise RuntimeError(
                "shift-invert eigenpair failed the outer residual gate: "
                f"residual={residual:.6g}, tolerance={residual_tol:.6g}; {inner}"
            )
        if growth_floor_failed:
            raise RuntimeError(
                "shift-invert eigenpair failed the growth-selection floor: "
                f"growth={eig_host.real:.6g}, floor={cfg.fallback_real_floor:.6g}; "
                f"{inner}"
            )
        raise RuntimeError(
            "shift-invert eigenpair is non-finite: "
            f"eigenvalue={eig_host.real:.6g}{eig_host.imag:+.6g}j; {inner}"
        )
    status = EigenSolveStatus(
        method=cfg.method,
        route="shift_invert",
        residual=residual,
        tolerance=residual_tol,
        certified=True,
        inner=_inner_solve_status(inner_stats, cfg.shift_tol, str(mode), precond_meta),
    )
    return eig_si, vec_si, status


def _sparse_shift_invert_branch(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus]:
    """Factor the exact sparse shifted operator when matrix-free cold solves stall.

    On a linked cache only the chain-mode columns are assembled; eigenvectors
    are returned in the full state shape with exact zeros elsewhere.
    """

    covered = _linked_covered_mode_mask(cache)
    size = int(v0.size)
    rows = (
        None
        if covered is None
        else jnp.asarray(
            np.flatnonzero(np.broadcast_to(np.asarray(covered)[:, :, None], v0.shape))
        )
    )
    n = size if rows is None else int(rows.size)
    if n < 3:
        raise ValueError(
            "sparse shift-invert requires an operator of size at least three"
        )
    if cfg.shift is None:
        raise ValueError("sparse shift-invert requires a supplied or coarse-grid shift")
    residual_tol = certifiable_residual_tolerance(
        cfg.shift_outer_residual_tol, v0.dtype
    )
    from scipy.sparse import eye
    from solvax import SpluFactorization
    from solvax import (  # type: ignore[attr-defined]
        sparse_eigenpairs,
        sparse_operator_matrix,
    )

    def apply(state):
        if rows is None:
            return _apply_operator(state, cache, params, term_cfg)
        full = jnp.zeros((size,), dtype=v0.dtype).at[rows].set(state)
        image = _apply_operator(full.reshape(v0.shape), cache, params, term_cfg)
        return image.reshape(size)[rows]

    prototype = v0 if rows is None else v0.reshape(size)[rows]
    _status(status_callback, "assembling sparse operator in bounded column batches")
    matrix = sparse_operator_matrix(
        apply, prototype, batch_size=64, drop_tolerance=1.0e-14
    )
    shift = complex(cfg.shift)
    scope = "" if rows is None else f" on linked-chain modes ({n} of {size} unknowns)"
    _status(
        status_callback,
        f"factoring coupled sparse operator n={matrix.shape[0]} nnz={matrix.nnz}"
        + scope,
    )
    factor = SpluFactorization(
        matrix - shift * eye(matrix.shape[0], format="csr", dtype=matrix.dtype)
    )
    modes = sparse_eigenpairs(
        matrix,
        candidates=min(6, n - 2),
        shift=shift,
        initial=prototype,
        tolerance=min(cfg.shift_tol, 1.0e-10),
        maxiter=max(cfg.shift_maxiter, 20_000),
        residual_tolerance=residual_tol,
        factorization=factor,
    )
    eigenvectors = modes.eigenvectors
    if rows is not None:
        eigenvectors = (
            jnp.zeros((eigenvectors.shape[0], size), dtype=eigenvectors.dtype)
            .at[:, rows]
            .set(eigenvectors)
        )
    vectors = eigenvectors.reshape((eigenvectors.shape[0], *v0.shape))
    residuals = np.asarray(
        [
            _eigenpair_relative_residual(value, vector, cache, params, term_cfg)
            for value, vector in zip(modes.eigenvalues, vectors, strict=True)
        ]
    )
    certified = np.asarray(modes.converged) & np.isfinite(residuals)
    certified &= residuals <= residual_tol
    _targeted, select_growth = _shift_selection_flags(cfg.shift_selection)
    if select_overlap:
        scores = np.abs(
            np.asarray(vectors).reshape((vectors.shape[0], -1))
            @ np.asarray(v_ref).reshape(-1).conj()
        )
    elif select_growth:
        scores = np.asarray(modes.eigenvalues).real
    else:
        scores = -np.abs(np.asarray(modes.eigenvalues) - shift)
    if not np.any(certified):
        raise RuntimeError(
            "sparse shift-invert returned no original-operator-certified eigenpair"
        )
    selected = int(np.argmax(np.where(certified, scores, -np.inf)))
    _status(
        status_callback,
        f"sparse shift-invert residual={residuals[selected]:.3g}",
    )
    status = EigenSolveStatus(
        method=cfg.method,
        route="sparse_shift_invert",
        residual=float(residuals[selected]),
        tolerance=residual_tol,
        certified=True,
    )
    return modes.eigenvalues[selected], vectors[selected], status


def _adaptive_branch(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus]:
    """Run the residual-certified adaptive eigensolve; fail closed on rejection.

    Unlike the raw propagator/power/arnoldi branches, this path never returns
    an uncertified pair: the eigenvalue is either certified against the
    continuous operator by the adaptive solver's residual gate or the solve
    raises with the measured residual.
    """

    max_restarts = max(int(cfg.restarts), 4)
    tol = certifiable_residual_tolerance(_ADAPTIVE_BASE_TOL, v0.dtype)
    krylov_dim = max(min(int(cfg.krylov_dim), int(v0.size) - 1), 2)
    restart_krylov_dim = max(krylov_dim // 2, 2)
    candidate_count = min(2, krylov_dim, restart_krylov_dim)
    _status(
        status_callback,
        "running certified adaptive propagator with "
        f"dim={krylov_dim} max_restarts={max_restarts} tol={tol:.3g}",
    )
    solution = adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms,
        krylov_dim=krylov_dim,
        restart_krylov_dim=restart_krylov_dim,
        candidate_count=candidate_count,
        max_restarts=max_restarts,
        tol=tol,
    )
    eig_host = complex(np.asarray(solution.eigenvalue))
    residual = float(np.asarray(solution.residual))
    _status(
        status_callback,
        "adaptive solve finished with "
        f"eig={eig_host.real:.6g}{eig_host.imag:+.6g}j residual={residual:.3g} "
        f"converged={bool(solution.converged)} stable={bool(solution.stable)}",
    )
    if not (bool(solution.stable) and bool(solution.converged)):
        raise RuntimeError(
            "certified adaptive eigensolve rejected the dominant eigenpair: "
            f"stable={bool(solution.stable)} converged={bool(solution.converged)} "
            f"residual={residual:.6g} tolerance={tol:.6g}; refusing to report "
            "an uncertified growth rate"
        )
    # The adaptive gate divides by |lambda| ||v||; report the shared definition,
    # which is never larger, so a certified pair stays within its tolerance.
    status = EigenSolveStatus(
        method=cfg.method,
        route="adaptive",
        residual=_eigenpair_relative_residual(
            solution.eigenvalue,
            solution.eigenvector,
            cache,
            params,
            linear_terms_to_term_config(terms),
        ),
        tolerance=tol,
        certified=True,
    )
    return solution.eigenvalue, solution.eigenvector, status


def _dispatch_dominant_eigenpair(
    v0: jnp.ndarray,
    v_ref: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg,
    cfg: KrylovConfig,
    status_callback: _StatusCallback,
    *,
    select_overlap: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, EigenSolveStatus]:
    if cfg.method == "shift_invert":
        return _shift_invert_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
    if cfg.method == "sparse_shift_invert":
        return _sparse_shift_invert_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
    if cfg.method == "power":
        pair = _power_branch(v0, cache, params, term_cfg, cfg, status_callback)
    elif cfg.method == "propagator":
        pair = _propagator_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
    elif cfg.method == "arnoldi":
        pair = _arnoldi_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
    else:
        raise ValueError(
            "Krylov method must be adaptive, power, propagator, shift_invert, "
            "sparse_shift_invert, or arnoldi"
        )
    return _certify_raw_eigenpair(pair, cache, params, term_cfg, cfg, status_callback)


def dominant_eigenpair(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    v_ref: jnp.ndarray | None = None,
    select_overlap: bool = False,
    krylov_dim: int = 24,
    restarts: int = 2,
    omega_min_factor: float = 0.0,
    omega_target_factor: float = 0.0,
    omega_cap_factor: float = 2.0,
    omega_sign: int = 0,
    method: str = "adaptive",
    power_iters: int = 40,
    power_dt: float = 0.01,
    propagator_steps: int = 1,
    shift: complex | None = None,
    shift_source: str = "propagator",
    shift_tol: float = 1.0e-4,
    shift_maxiter: int = 50,
    shift_restart: int = 20,
    shift_solve_method: str = "batched",
    shift_preconditioner: str | None = "auto",
    shift_precond_alpha: float | complex | None = None,
    shift_precond_block_solve: str = "auto",
    shift_selection: str = "targeted",
    shift_outer_residual_tol: float = 1.0e-6,
    mode_family: str = "auto",
    fallback_method: str = "propagator",
    fallback_real_floor: float = -1.0e-6,
    certify: bool = True,
    status_callback: Callable[[str], None] | None = None,
    return_status: bool = False,
) -> tuple[Any, ...]:
    """Python wrapper for cached matrix-free and sparse eigen solvers.

    No method returns a pair above its original-operator residual gate: the
    adaptive, shift-invert and sparse routes raise on rejection, and the raw
    ``power``, ``propagator`` and ``arnoldi`` routes raise above
    ``certifiable_residual_tolerance(shift_outer_residual_tol, dtype)`` unless
    ``certify=False`` explicitly accepts an uncertified pair.

    Returns ``(eigenvalue, eigenvector)``; ``return_status=True`` appends the
    pair's :class:`EigenSolveStatus` (residual, gate, route, inner solves).
    """
    cfg = _normalized_config(locals())
    term_cfg = linear_terms_to_term_config(terms)
    v_ref_use = v0 if v_ref is None else v_ref
    covered = _linked_covered_mode_mask(cache)
    if covered is not None:
        v0 = _project_to_linked_cover(v0, covered)
        v_ref_use = _project_to_linked_cover(v_ref_use, covered)
        _require_linked_cover_seed(v0)
    _status(
        status_callback,
        f"krylov method={cfg.method} dim={cfg.krylov_dim} restarts={cfg.restarts}",
    )
    if cfg.method == "adaptive":
        solved = _adaptive_branch(v0, cache, params, terms, cfg, status_callback)
    else:
        solved = _dispatch_dominant_eigenpair(
            v0,
            v_ref_use,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
    return solved if return_status else solved[:2]


def dominant_eigenvalue(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: LinearTerms | None = None,
    *,
    krylov_dim: int = 24,
    restarts: int = 2,
) -> jnp.ndarray:
    eig, _vec = dominant_eigenpair(
        v0,
        cache,
        params,
        terms,
        krylov_dim=krylov_dim,
        restarts=restarts,
    )
    return eig


__all__ = [
    "KrylovConfig",
    "_advance_imex2",
    "_apply_operator",
    "_arnoldi",
    "_assemble_rhs_cached_novjp",
    "build_shift_invert_preconditioner",
    "_compute_damping",
    "_mode_family_sign",
    "_normalize",
    "_omega_scale",
    "_physical_omega",
    "_select_by_overlap",
    "_select_by_target",
    "adaptive_propagator_eigenpair",
    "certifiable_residual_tolerance",
    "dominant_eigenpair",
    "dominant_eigenpair_cached",
    "dominant_eigenpair_power",
    "dominant_eigenpair_propagator_cached",
    "dominant_eigenpair_shift_invert_cached",
    "dominant_eigenvalue",
]
