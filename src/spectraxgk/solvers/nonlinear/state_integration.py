"""Core nonlinear RHS and cached integrator drivers."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.core.extension_points import CollisionOperator
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import (
    build_linear_cache,
    update_linear_cache_for_sheared_kx,
)
from spectraxgk.operators.linear.params import LinearParams
from spectraxgk.operators.nonlinear.policies import (
    IMEXLinearOperator,
    _make_hermitian_projector,
    build_nonlinear_imex_operator,
)
from spectraxgk.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from spectraxgk.operators.nonlinear.projection import advance_shearing_coordinates
from spectraxgk.solvers.nonlinear.explicit import (
    integrate_cached_explicit_scan,
    integrate_nonlinear_scan,
)
from spectraxgk.solvers.nonlinear.imex import integrate_cached_imex_scan
from spectraxgk.terms.assembly import (
    _is_static_zero,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    compute_fields_cached,
)
from spectraxgk.terms.config import FieldState, TermConfig
from spectraxgk.terms.nonlinear import nonlinear_em_contribution


def _linear_rhs_jit_for_terms(term_cfg: TermConfig):
    """Return the narrowest compiled linear RHS path compatible with ``term_cfg``."""

    return linear_rhs_jit_for_terms_impl(
        term_cfg,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
    )


def nonlinear_rhs_cached(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    terms: TermConfig | None = None,
    *,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    collision_operator: CollisionOperator | None = None,
    radial_phase: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Compute the assembled nonlinear RHS and electromagnetic field state."""

    return nonlinear_rhs_cached_impl(
        G,
        cache,
        params,
        terms,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        collision_operator=collision_operator,
        radial_phase=radial_phase,
        electrostatic_rhs_fn=assemble_rhs_cached_electrostatic_jit,
        full_rhs_fn=assemble_rhs_cached_jit,
        is_static_zero_fn=_is_static_zero,
        nonlinear_contribution_fn=nonlinear_em_contribution,
    )


def _nonlinear_rhs_scan(
    G: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    compressed_real_fft: bool,
    laguerre_mode: str,
    collision_operator: CollisionOperator | None,
) -> tuple[jnp.ndarray, FieldState]:
    """Stable scan callable; arrays are dynamic while model switches are static."""

    return nonlinear_rhs_cached(
        G,
        cache,
        params,
        term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        collision_operator=collision_operator,
    )


def integrate_nonlinear_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
        if collision_operator is not None:
            raise NotImplementedError(
                "custom collision operators currently require explicit nonlinear integration"
            )
        result = integrate_nonlinear_imex_cached(
            G0,
            cache,
            params,
            dt,
            steps,
            terms=term_cfg,
            checkpoint=checkpoint,
            compressed_real_fft=compressed_real_fft,
            laguerre_mode=laguerre_mode,
            show_progress=show_progress,
        )
        return result if return_fields else result[0]

    project_state = None
    if compressed_real_fft:
        project_state = _make_hermitian_projector(
            np.asarray(cache.ky), int(np.asarray(cache.kx).size)
        )

    return integrate_cached_explicit_scan(
        G0,
        dt,
        steps,
        method=method,
        rhs_fn=_nonlinear_rhs_scan,
        rhs_args=(cache, params),
        rhs_static_args=(
            term_cfg,
            compressed_real_fft,
            laguerre_mode,
            collision_operator,
        ),
        scan_fn=integrate_nonlinear_scan,
        checkpoint=checkpoint,
        project_state=project_state,
        show_progress=show_progress,
        return_fields=return_fields,
    )


def integrate_nonlinear(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    method: str = "rk4",
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    show_progress: bool = False,
    return_fields: bool = True,
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system using built-in cache construction."""

    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            Nl, Nm = G0.shape[0], G0.shape[1]
        elif G0.ndim == 6:
            Nl, Nm = G0.shape[1], G0.shape[2]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or (Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom_eff, params, Nl, Nm)
    return integrate_nonlinear_cached(
        G0,
        cache,
        params,
        dt,
        steps,
        method=method,
        terms=terms,
        checkpoint=checkpoint,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        show_progress=show_progress,
        return_fields=return_fields,
        collision_operator=collision_operator,
    )


def integrate_nonlinear_sheared_euler(
    G0: jnp.ndarray,
    grid: SpectralGrid,
    geom: FluxTubeGeometryLike,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    shear_rate: jnp.ndarray | float,
    cache: LinearCache | None = None,
    terms: TermConfig | None = None,
    laguerre_mode: str = "grid",
    collision_operator: CollisionOperator | None = None,
) -> tuple[jnp.ndarray, FieldState]:
    """Integrate the periodic shearing-coordinate foundation with Euler steps.

    This research path intentionally supports only fixed-step Euler and the
    full-complex FFT. Higher-order stage-time routing and linked-boundary phases
    must pass separate gates before flow shear is exposed in runtime inputs.
    """

    if str(grid.boundary).lower() != "periodic" or bool(grid.non_twist):
        raise NotImplementedError(
            "sheared integration currently requires a periodic standard flux tube"
        )
    if steps < 1:
        raise ValueError("steps must be at least one")
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    if cache is None:
        if G0.ndim == 5:
            nl, nm = G0.shape[:2]
        elif G0.ndim == 6:
            nl, nm = G0.shape[1:3]
        else:
            raise ValueError(
                "G0 must have shape (Nl, Nm, Ny, Nx, Nz) or "
                "(Ns, Nl, Nm, Ny, Nx, Nz)"
            )
        cache = build_linear_cache(grid, geom_eff, params, int(nl), int(nm))

    term_cfg = terms or TermConfig()
    state_dtype = jnp.result_type(G0, jnp.complex64)
    real_dtype = jnp.real(jnp.empty((), dtype=state_dtype)).dtype
    dt_value = jnp.asarray(dt, dtype=real_dtype)
    rho_star = jnp.asarray(params.rho_star, dtype=real_dtype)

    def coordinates(state: jnp.ndarray, time: jnp.ndarray, previous: jnp.ndarray):
        return advance_shearing_coordinates(
            state,
            kx=grid.kx,
            ky=grid.ky,
            x0=grid.x0,
            shear_rate=shear_rate,
            previous_time=previous,
            time=time,
            dealias_mask=grid.dealias_mask,
        )

    def cache_at(update):
        return update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom_eff,
            params,
            rho_star * update.effective_kx,
        )

    def step(state: jnp.ndarray, index: jnp.ndarray):
        time = jnp.asarray(index, dtype=real_dtype) * dt_value
        current = coordinates(state, time, time)
        current_cache = cache_at(current)
        derivative, _ = nonlinear_rhs_cached(
            current.state,
            current_cache,
            params,
            term_cfg,
            compressed_real_fft=False,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            radial_phase=current.phase,
        )
        trial = current.state + dt_value * derivative
        new_time = time + dt_value
        advanced = coordinates(trial, new_time, time)
        new_cache = cache_at(advanced)
        _, fields = nonlinear_rhs_cached(
            advanced.state,
            new_cache,
            params,
            term_cfg,
            compressed_real_fft=False,
            laguerre_mode=laguerre_mode,
            collision_operator=collision_operator,
            radial_phase=advanced.phase,
        )
        return jnp.asarray(advanced.state, dtype=state_dtype), fields

    return jax.lax.scan(step, jnp.asarray(G0, dtype=state_dtype), jnp.arange(steps))


def integrate_nonlinear_imex_cached(
    G0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    dt: float,
    steps: int,
    *,
    terms: TermConfig | None = None,
    checkpoint: bool = False,
    implicit_tol: float = 1.0e-6,
    implicit_maxiter: int = 200,
    implicit_iters: int = 3,
    implicit_relax: float = 0.7,
    implicit_restart: int = 20,
    implicit_preconditioner: str | None = None,
    implicit_operator: IMEXLinearOperator | None = None,
    compressed_real_fft: bool = True,
    laguerre_mode: str = "grid",
    external_phi: jnp.ndarray | float | None = None,
    show_progress: bool = False,
) -> tuple[jnp.ndarray, FieldState]:
    """IMEX integrator: implicit linear operator, explicit nonlinear term."""

    term_cfg = terms or TermConfig()
    linear_cfg = replace(term_cfg, nonlinear=0.0)
    linear_rhs_fn = _linear_rhs_jit_for_terms(linear_cfg)
    return integrate_cached_imex_scan(
        G0,
        cache,
        params,
        dt,
        steps,
        term_cfg=term_cfg,
        linear_cfg=linear_cfg,
        linear_rhs_fn=linear_rhs_fn,
        build_operator_fn=build_nonlinear_imex_operator,
        build_implicit_operator_fn=_build_implicit_operator,
        fields_fn=compute_fields_cached,
        nonlinear_term_fn=nonlinear_em_term_cached_impl,
        nonlinear_contribution_fn=nonlinear_em_contribution,
        checkpoint=checkpoint,
        implicit_tol=implicit_tol,
        implicit_maxiter=implicit_maxiter,
        implicit_iters=implicit_iters,
        implicit_relax=implicit_relax,
        implicit_restart=implicit_restart,
        implicit_preconditioner=implicit_preconditioner,
        implicit_operator=implicit_operator,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
        external_phi=external_phi,
        show_progress=show_progress,
    )


__all__ = [
    "_linear_rhs_jit_for_terms",
    "integrate_nonlinear",
    "integrate_nonlinear_cached",
    "integrate_nonlinear_imex_cached",
    "integrate_nonlinear_sheared_euler",
    "nonlinear_rhs_cached",
]
