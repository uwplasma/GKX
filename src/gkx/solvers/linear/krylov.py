"""Public Krylov solver facade for linear gyrokinetic eigenmodes.

The compiled kernels live in focused eigenmode modules so that branch selection,
operator application, preconditioning, and Arnoldi iterations can be tested and
optimized independently.  This facade keeps the documented script import path
and the monkeypatch seams used by benchmark/runtime tests.
"""

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
from gkx.solvers.linear.krylov_algorithms import (
    _advance_imex2,
    _apply_operator,
    _assemble_rhs_cached_novjp,
    _compute_damping,
    _normalize,
)
from gkx.solvers.linear.krylov_algorithms import (
    _arnoldi,
    _build_shift_invert_precond,
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


@dataclass(frozen=True)
class KrylovConfig:
    """Controls for the Krylov-based eigen solver."""

    krylov_dim: int = 24
    restarts: int = 2
    omega_min_factor: float = 0.0
    omega_target_factor: float = 0.0
    omega_cap_factor: float = 2.0
    omega_sign: int = 0
    method: str = "propagator"
    power_iters: int = 200
    power_dt: float = 0.01
    shift: complex | None = None
    shift_source: str = "propagator"
    shift_tol: float = 1.0e-4
    shift_maxiter: int = 50
    shift_restart: int = 20
    shift_solve_method: str = "batched"
    shift_preconditioner: str | None = "damping"
    shift_selection: str = "targeted"
    shift_outer_residual_tol: float = 0.1
    mode_family: str = "auto"
    fallback_method: str = "propagator"
    fallback_real_floor: float = -1.0e-6
    continuation: bool = False
    continuation_selection: str = "overlap"


_StatusCallback = Callable[[str], None] | None


def _status(status_callback: _StatusCallback, message: str) -> None:
    if status_callback is not None:
        status_callback(message)


def _normalized_config(
    *,
    method: str,
    krylov_dim: int,
    restarts: int,
    omega_min_factor: float,
    omega_target_factor: float,
    omega_cap_factor: float,
    omega_sign: int,
    mode_family: str,
    power_iters: int,
    power_dt: float,
    shift: complex | None,
    shift_source: str,
    shift_tol: float,
    shift_maxiter: int,
    shift_restart: int,
    shift_solve_method: str,
    shift_preconditioner: str | None,
    shift_selection: str,
    shift_outer_residual_tol: float,
    fallback_method: str,
    fallback_real_floor: float,
) -> KrylovConfig:
    mode_family_sign = _mode_family_sign(mode_family)
    omega_sign_eff = int(omega_sign) if int(omega_sign) != 0 else mode_family_sign
    return KrylovConfig(
        method=method.strip().lower(),
        krylov_dim=max(int(krylov_dim), 1),
        restarts=max(int(restarts), 1),
        omega_min_factor=float(omega_min_factor),
        omega_target_factor=float(omega_target_factor),
        omega_cap_factor=float(omega_cap_factor),
        omega_sign=omega_sign_eff,
        power_iters=max(int(power_iters), 1),
        power_dt=float(power_dt),
        shift=shift,
        shift_source=shift_source,
        shift_tol=float(shift_tol),
        shift_maxiter=max(int(shift_maxiter), 1),
        shift_restart=max(int(shift_restart), 1),
        shift_solve_method=shift_solve_method,
        shift_preconditioner=shift_preconditioner,
        shift_selection=shift_selection,
        shift_outer_residual_tol=float(shift_outer_residual_tol),
        mode_family=mode_family,
        fallback_method=fallback_method,
        fallback_real_floor=float(fallback_real_floor),
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
        f"dt={cfg.power_dt:.6g} dim={cfg.krylov_dim} restarts={restarts_use}",
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


def _eigenpair_relative_residual(
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: Any,
) -> float:
    """Return the scale-invariant residual of one matrix-free eigenpair."""

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
) -> tuple[jnp.ndarray, jnp.ndarray] | None:
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
    if not np.isfinite(residual) or residual > cfg.shift_outer_residual_tol:
        return None
    return result


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    _status(status_callback, "running shift-invert Arnoldi")
    eig_si, vec_si = dominant_eigenpair_shift_invert_cached(
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
        gmres_solve_method=cfg.shift_solve_method,
        shift_preconditioner=cfg.shift_preconditioner,
        select_targeted=select_targeted,
        select_growth=select_growth,
        select_overlap=bool(select_overlap),
    )
    eig_host = complex(np.asarray(eig_si))
    residual = _eigenpair_relative_residual(eig_si, vec_si, cache, params, term_cfg)
    _status(
        status_callback,
        "shift-invert solve finished with "
        f"eig={eig_host.real:.6g}{eig_host.imag:+.6g}j residual={residual:.3g}",
    )
    nonfinite_pair = not np.isfinite(eig_host.real) or not np.isfinite(eig_host.imag)
    growth_floor_failed = select_growth and eig_host.real < cfg.fallback_real_floor
    residual_failed = (
        not np.isfinite(residual) or residual > cfg.shift_outer_residual_tol
    )
    need_fallback = nonfinite_pair or growth_floor_failed or residual_failed
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
                f"residual={residual:.6g}, tolerance={cfg.shift_outer_residual_tol:.6g}"
            )
        if growth_floor_failed:
            raise RuntimeError(
                "shift-invert eigenpair failed the growth-selection floor: "
                f"growth={eig_host.real:.6g}, floor={cfg.fallback_real_floor:.6g}"
            )
        raise RuntimeError(
            "shift-invert eigenpair is non-finite: "
            f"eigenvalue={eig_host.real:.6g}{eig_host.imag:+.6g}j"
        )
    return eig_si, vec_si


def _dominant_eigenpair_config_from_options(
    options: Mapping[str, Any],
) -> KrylovConfig:
    return _normalized_config(
        method=options["method"],
        krylov_dim=options["krylov_dim"],
        restarts=options["restarts"],
        omega_min_factor=options["omega_min_factor"],
        omega_target_factor=options["omega_target_factor"],
        omega_cap_factor=options["omega_cap_factor"],
        omega_sign=options["omega_sign"],
        mode_family=options["mode_family"],
        power_iters=options["power_iters"],
        power_dt=options["power_dt"],
        shift=options["shift"],
        shift_source=options["shift_source"],
        shift_tol=options["shift_tol"],
        shift_maxiter=options["shift_maxiter"],
        shift_restart=options["shift_restart"],
        shift_solve_method=options["shift_solve_method"],
        shift_preconditioner=options["shift_preconditioner"],
        shift_selection=options["shift_selection"],
        shift_outer_residual_tol=options["shift_outer_residual_tol"],
        fallback_method=options["fallback_method"],
        fallback_real_floor=options["fallback_real_floor"],
    )


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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if cfg.method == "power":
        return _power_branch(v0, cache, params, term_cfg, cfg, status_callback)
    if cfg.method == "propagator":
        return _propagator_branch(
            v0,
            v_ref,
            cache,
            params,
            term_cfg,
            cfg,
            status_callback,
            select_overlap=select_overlap,
        )
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
    if cfg.method != "arnoldi":
        raise ValueError(
            "Krylov method must be 'power', 'propagator', 'shift_invert', or 'arnoldi'"
        )
    return _arnoldi_branch(
        v0,
        v_ref,
        cache,
        params,
        term_cfg,
        cfg,
        status_callback,
        select_overlap=select_overlap,
    )


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
    method: str = "power",
    power_iters: int = 40,
    power_dt: float = 0.01,
    shift: complex | None = None,
    shift_source: str = "propagator",
    shift_tol: float = 1.0e-4,
    shift_maxiter: int = 50,
    shift_restart: int = 20,
    shift_solve_method: str = "batched",
    shift_preconditioner: str | None = "damping",
    shift_selection: str = "targeted",
    shift_outer_residual_tol: float = 0.1,
    mode_family: str = "auto",
    fallback_method: str = "propagator",
    fallback_real_floor: float = -1.0e-6,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Python wrapper for the cached Krylov solver."""
    cfg = _dominant_eigenpair_config_from_options(locals())
    term_cfg = linear_terms_to_term_config(terms)
    v_ref_use = v0 if v_ref is None else v_ref
    _status(
        status_callback,
        f"krylov method={cfg.method} dim={cfg.krylov_dim} restarts={cfg.restarts}",
    )
    return _dispatch_dominant_eigenpair(
        v0,
        v_ref_use,
        cache,
        params,
        term_cfg,
        cfg,
        status_callback,
        select_overlap=select_overlap,
    )


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
    "_build_shift_invert_precond",
    "_compute_damping",
    "_mode_family_sign",
    "_normalize",
    "_omega_scale",
    "_physical_omega",
    "_select_by_overlap",
    "_select_by_target",
    "dominant_eigenpair",
    "dominant_eigenpair_cached",
    "dominant_eigenpair_power",
    "dominant_eigenpair_propagator_cached",
    "dominant_eigenpair_shift_invert_cached",
    "dominant_eigenvalue",
]
