"""Public Krylov solver facade for linear gyrokinetic eigenmodes.

The compiled kernels live in focused eigenmode modules so that branch selection,
operator application, preconditioning, and Arnoldi iterations can be tested and
optimized independently.  This facade preserves the historical import path used
by scripts while keeping monkeypatch seams for benchmark/runtime tests.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.operators.linear.cache import LinearCache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.eigen_operator import (
    _advance_imex2,
    _apply_operator,
    _assemble_rhs_cached_novjp,
    _compute_damping,
    _normalize,
)
from spectraxgk.solvers.linear.eigen_policy import KrylovConfig
from spectraxgk.solvers.linear.eigen_preconditioners import _build_shift_invert_precond
from spectraxgk.solvers.linear.eigen_selection import (
    _mode_family_sign,
    _omega_scale,
    _physical_omega,
    _select_by_overlap,
    _select_by_target,
)
from spectraxgk.solvers.linear.krylov_algorithms import (
    _arnoldi,
    dominant_eigenpair_cached,
    dominant_eigenpair_power,
    dominant_eigenpair_propagator_cached,
    dominant_eigenpair_shift_invert_cached,
)

# Keep generated API docs anchored to the public import path rather than the
# small policy-owner module that physically defines the dataclass.
KrylovConfig.__module__ = __name__


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
    mode_family: str = "auto",
    fallback_method: str = "propagator",
    fallback_real_floor: float = -1.0e-6,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Python wrapper for the cached Krylov solver."""

    def _status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    term_cfg = linear_terms_to_term_config(terms)
    v_ref_use = v0 if v_ref is None else v_ref
    method_key = method.strip().lower()
    mode_family_sign = _mode_family_sign(mode_family)
    omega_sign_eff = int(omega_sign) if int(omega_sign) != 0 else mode_family_sign
    _status(
        f"krylov method={method_key} dim={max(int(krylov_dim), 1)} restarts={max(int(restarts), 1)}"
    )
    if method_key == "power":
        _status(f"running power iteration seed with iterations={max(int(power_iters), 1)} dt={float(power_dt):.6g}")
        return dominant_eigenpair_power(
            v0,
            cache,
            params,
            term_cfg,
            iterations=max(int(power_iters), 1),
            dt=float(power_dt),
        )
    if method_key == "propagator":
        _status(
            f"running propagator Arnoldi with dt={float(power_dt):.6g} dim={max(int(krylov_dim), 1)} restarts={max(int(restarts), 1)}"
        )
        return dominant_eigenpair_propagator_cached(
            v0,
            v_ref_use,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            restarts=max(int(restarts), 1),
            dt=float(power_dt),
            omega_min_factor=float(omega_min_factor),
            omega_target_factor=float(omega_target_factor),
            omega_cap_factor=float(omega_cap_factor),
            omega_sign=omega_sign_eff,
            select_overlap=bool(select_overlap),
        )
    if method_key == "shift_invert":
        restarts = max(int(restarts), 1)
        _status(
            f"preparing shift-invert solve with dim={max(int(krylov_dim), 1)} restarts={restarts} gmres_maxiter={max(int(shift_maxiter), 1)} restart={max(int(shift_restart), 1)} tol={float(shift_tol):.3g}"
        )
        if shift is None:
            shift_source_key = shift_source.strip().lower()
            if shift_source_key == "propagator":
                _status("estimating shift from propagator seed")
                shift_est, v_seed = dominant_eigenpair_propagator_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=max(int(krylov_dim), 1),
                    restarts=1,
                    dt=float(power_dt),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=False,
                )
                sigma = shift_est
                v_init = v_seed
            elif shift_source_key == "target":
                _status("building target-frequency shift")
                omega_scale = _omega_scale(cache, params)
                omega_target = float(omega_target_factor) * omega_scale
                if omega_sign_eff != 0:
                    omega_target = float(jnp.sign(omega_sign_eff)) * jnp.abs(omega_target)
                sigma = -1j * omega_target
                v_init = v0
            else:
                _status("estimating shift from power iteration seed")
                shift_est, v_seed = dominant_eigenpair_power(
                    v0,
                    cache,
                    params,
                    term_cfg,
                    iterations=max(int(power_iters), 1),
                    dt=float(power_dt),
                )
                sigma = shift_est
                v_init = v_seed
        else:
            sigma = jnp.asarray(shift, dtype=v0.dtype)
            shift_source_key = shift_source.strip().lower()
            if shift_source_key == "propagator":
                _status("using explicit shift with propagator seed vector")
                _shift_seed, v_seed = dominant_eigenpair_propagator_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=max(int(krylov_dim), 1),
                    restarts=1,
                    dt=float(power_dt),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=bool(select_overlap),
                )
                v_init = v_seed
            elif shift_source_key == "power":
                _status("using explicit shift with power-iteration seed vector")
                _shift_seed, v_seed = dominant_eigenpair_power(
                    v0,
                    cache,
                    params,
                    term_cfg,
                    iterations=max(int(power_iters), 1),
                    dt=float(power_dt),
                )
                v_init = v_seed
            else:
                _status("using explicit shift with reference seed vector")
                v_init = v_ref_use
        sigma_host = complex(np.asarray(sigma))
        _status(f"shift-invert sigma={sigma_host.real:.6g}{sigma_host.imag:+.6g}j")
        selection_key = shift_selection.strip().lower()
        select_targeted = selection_key in {"targeted", "target", "auto", "default"}
        select_growth = selection_key in {"targeted", "growth", "auto", "default"}
        _status("running shift-invert Arnoldi")
        eig_si, vec_si = dominant_eigenpair_shift_invert_cached(
            v_init,
            v_ref_use,
            cache,
            params,
            term_cfg,
            krylov_dim=krylov_dim,
            restarts=restarts,
            sigma=sigma,
            omega_min_factor=float(omega_min_factor),
            omega_target_factor=float(omega_target_factor),
            omega_cap_factor=float(omega_cap_factor),
            omega_sign=omega_sign_eff,
            gmres_tol=shift_tol,
            gmres_maxiter=max(int(shift_maxiter), 1),
            gmres_restart=max(int(shift_restart), 1),
            gmres_solve_method=shift_solve_method,
            shift_preconditioner=shift_preconditioner,
            select_targeted=select_targeted,
            select_growth=select_growth,
            select_overlap=bool(select_overlap),
        )
        eig_host = complex(np.asarray(eig_si))
        _status(f"shift-invert solve finished with eig={eig_host.real:.6g}{eig_host.imag:+.6g}j")
        fallback_key = fallback_method.strip().lower()
        need_fallback = (
            not np.isfinite(eig_host.real)
            or not np.isfinite(eig_host.imag)
            or eig_host.real < float(fallback_real_floor)
        )
        if need_fallback and fallback_key != "none":
            _status(f"shift-invert result rejected; falling back to {fallback_key}")
            if fallback_key == "propagator":
                return dominant_eigenpair_propagator_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=krylov_dim,
                    restarts=max(int(restarts), 1),
                    dt=float(power_dt),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=False,
                )
            if fallback_key == "arnoldi":
                return dominant_eigenpair_cached(
                    v0,
                    v_ref_use,
                    cache,
                    params,
                    term_cfg,
                    krylov_dim=krylov_dim,
                    restarts=max(int(restarts), 1),
                    omega_min_factor=float(omega_min_factor),
                    omega_target_factor=float(omega_target_factor),
                    omega_cap_factor=float(omega_cap_factor),
                    omega_sign=omega_sign_eff,
                    select_overlap=bool(select_overlap),
                )
            if fallback_key == "power":
                return dominant_eigenpair_power(
                    v0,
                    cache,
                    params,
                    term_cfg,
                    iterations=max(int(power_iters), 1),
                    dt=float(power_dt),
                )
        return eig_si, vec_si
    if method_key != "arnoldi":
        raise ValueError(
            "Krylov method must be 'power', 'propagator', 'shift_invert', or 'arnoldi'"
        )

    restarts = max(int(restarts), 1)
    _status(f"running plain Arnoldi with dim={max(int(krylov_dim), 1)} restarts={restarts}")
    return dominant_eigenpair_cached(
        v0,
        v_ref_use,
        cache,
        params,
        term_cfg,
        krylov_dim=krylov_dim,
        restarts=restarts,
        omega_min_factor=float(omega_min_factor),
        omega_target_factor=float(omega_target_factor),
        omega_cap_factor=float(omega_cap_factor),
        omega_sign=omega_sign_eff,
        select_overlap=bool(select_overlap),
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
