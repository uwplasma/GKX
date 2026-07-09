"""Core nonlinear RHS and cached integrator drivers."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import FluxTubeGeometryLike, ensure_flux_tube_geometry_data
from spectraxgk.core.grid import SpectralGrid
from spectraxgk.solvers.linear.implicit import _build_implicit_operator
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.cache_builder import build_linear_cache
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
) -> tuple[jnp.ndarray, FieldState]:
    """Stable scan callable; arrays are dynamic while model switches are static."""

    return nonlinear_rhs_cached(
        G,
        cache,
        params,
        term_cfg,
        compressed_real_fft=compressed_real_fft,
        laguerre_mode=laguerre_mode,
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
) -> tuple[jnp.ndarray, FieldState] | jnp.ndarray:
    """Integrate the nonlinear system using a cached geometry object."""

    term_cfg = terms or TermConfig()
    if method in {"imex", "semi-implicit"}:
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
        rhs_static_args=(term_cfg, compressed_real_fft, laguerre_mode),
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
    )


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
    implicit_solve_method: str = "batched",
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
        implicit_solve_method=implicit_solve_method,
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
    "nonlinear_rhs_cached",
]
