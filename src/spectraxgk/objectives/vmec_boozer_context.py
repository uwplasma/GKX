"""Shared VMEC/Boozer state-to-linear-solver context for objective gates."""

from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp

from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.diagnostics import fieldline_quadrature_weights, heat_flux_species
from spectraxgk.geometry.backend_discovery import discover_differentiable_geometry_backends
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.geometry.vmec_boozer_core import (
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
)
from spectraxgk.objectives.core import (
    _default_gradient_linear_params,
    _default_gradient_linear_terms,
)
from spectraxgk.objectives.vmec_state import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)
from spectraxgk.operators.linear.cache_builder import build_linear_cache
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.diagnostics.quasilinear_transport import effective_kperp2, phi_norm2
from spectraxgk.objectives.autodiff_validation import explicit_complex_operator_matrix


def _load_vmec_boozer_example(case_name: str) -> tuple[Any, Any, Any, Any]:
    """Load VMEC-JAX example inputs and the corresponding differentiable state."""

    discover_differentiable_geometry_backends()
    driver = importlib.import_module("vmec_jax.driver")
    config_mod = importlib.import_module("vmec_jax.config")
    static_mod = importlib.import_module("vmec_jax.static")
    wout_mod = importlib.import_module("vmec_jax.wout")

    input_path, wout_path = driver.example_paths(str(case_name))
    cfg_vmec, indata = config_mod.load_config(str(input_path))
    static = static_mod.build_static(cfg_vmec)
    wout = wout_mod.read_wout(wout_path)
    state = wout_mod.state_from_wout(wout)
    return static, indata, wout, state


def _resolve_mode21_state_parameter(
    state: Any,
    *,
    parameter_family: str,
    radial_index: int | None,
    mode_index: int,
) -> tuple[Any, int, int, tuple[str, ...]]:
    """Resolve the VMEC state coefficient addressed by a mode-21 gate."""

    base_coeff = _vmec_boozer_state_array(state, parameter_family)
    default_radial_index = int(base_coeff.shape[0] // 2)
    radial_index_int = (
        default_radial_index if radial_index is None else int(radial_index)
    )
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_coeff.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_coeff.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_names = (
        _vmec_boozer_state_parameter_name(
            parameter_family,
            radial_index_int,
            mode_index_int,
            default_mid_surface=default_radial_index,
        ),
    )
    return base_coeff, radial_index_int, mode_index_int, parameter_names


def _mode21_linear_grid(ntheta: int) -> tuple[CycloneBaseCase, Any]:
    """Build the compact linear grid used by VMEC/Boozer mode-21 gates."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=int(ntheta), Lx=6.0, Ly=12.0))
    return cfg, select_ky_grid(build_spectral_grid(cfg.grid), 1)


def _mode21_state_shape(
    *,
    n_laguerre: int,
    n_hermite: int,
    grid: Any,
) -> tuple[int, int, int, int, int]:
    """Return the flattened linear-state shape used by explicit matrices."""

    return (
        int(n_laguerre),
        int(n_hermite),
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )


def _make_mode21_geometry_for(
    *,
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    parameter_family: str,
    base_coeff: Any,
    radial_index_int: int,
    mode_index_int: int,
    surface_index: int | None,
    ntheta: int,
    mboz: int,
    nboz: int,
    surface_stencil_width: int | None,
):
    """Create the VMEC-state coefficient to solver geometry closure."""

    def geometry_for(x: jnp.ndarray):
        traced_state = _replace_vmec_boozer_state_coefficient(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            x[0],
        )
        mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
            traced_state,
            static,
            indata,
            wout,
            surface_index=surface_index,
            ntheta=int(ntheta),
            mboz=int(mboz),
            nboz=int(nboz),
            surface_stencil_width=surface_stencil_width,
        )
        return flux_tube_geometry_from_mapping(
            mapping,
            source_model="mode21_vmec_boozer_state",
            validate_finite=False,
        )

    return geometry_for


def _make_mode21_solver_closures(
    *,
    grid: Any,
    geometry_for: Any,
    params_linear: Any,
    terms: Any,
    n_laguerre: int,
    n_hermite: int,
    state_shape: tuple[int, int, int, int, int],
) -> tuple[Any, Any, Any]:
    """Create cache, RHS, and explicit-matrix closures for a mode-21 gate."""

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(
            grid, geometry_for(x), params_linear, int(n_laguerre), int(n_hermite)
        )

    def rhs_phi(state_arr: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            params_linear,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state_arr: rhs_phi(state_arr, cache)[0], state_shape
        )

    return cache_for, rhs_phi, matrix_fn


def _mode21_vmec_boozer_linear_context(  # pragma: no cover
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    parameter_family: str,
    surface_index: int | None,
    ntheta: int,
    mboz: int,
    nboz: int,
    surface_stencil_width: int | None,
    n_laguerre: int,
    n_hermite: int,
) -> dict[str, Any]:
    """Build shared VMEC/Boozer geometry and linear-RHS closures for gates."""

    static, indata, wout, state = _load_vmec_boozer_example(str(case_name))
    base_coeff, radial_index_int, mode_index_int, parameter_names = (
        _resolve_mode21_state_parameter(
            state,
            parameter_family=parameter_family,
            radial_index=radial_index,
            mode_index=mode_index,
        )
    )
    cfg, grid = _mode21_linear_grid(int(ntheta))
    state_shape = _mode21_state_shape(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        grid=grid,
    )
    params_linear = _default_gradient_linear_params()
    terms = _default_gradient_linear_terms()
    geometry_for = _make_mode21_geometry_for(
        state=state,
        static=static,
        indata=indata,
        wout=wout,
        parameter_family=parameter_family,
        base_coeff=base_coeff,
        radial_index_int=radial_index_int,
        mode_index_int=mode_index_int,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
    )
    cache_for, rhs_phi, matrix_fn = _make_mode21_solver_closures(
        grid=grid,
        geometry_for=geometry_for,
        params_linear=params_linear,
        terms=terms,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        state_shape=state_shape,
    )

    return {
        "case_name": str(case_name),
        "cfg": cfg,
        "grid": grid,
        "parameter_names": parameter_names,
        "parameter_indices": {
            str(parameter_family): [radial_index_int, mode_index_int]
        },
        "surface_index": surface_index,
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": surface_stencil_width,
        "n_laguerre": int(n_laguerre),
        "n_hermite": int(n_hermite),
        "state_shape": state_shape,
        "params_linear": params_linear,
        "geometry_for": geometry_for,
        "cache_for": cache_for,
        "rhs_phi": rhs_phi,
        "matrix_fn": matrix_fn,
    }


def _mode21_vmec_boozer_quasilinear_features(
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
    x: jnp.ndarray,
    context: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate quasilinear observables from one isolated linear eigenpair."""

    geom = context["geometry_for"](x)
    grid = context["grid"]
    params_linear = context["params_linear"]
    cache = build_linear_cache(
        grid,
        geom,
        params_linear,
        context["n_laguerre"],
        context["n_hermite"],
    )
    state_arr = jnp.reshape(eigenvector, context["state_shape"])
    _rhs, phi = context["rhs_phi"](state_arr, cache)
    zero_field = jnp.zeros_like(phi)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
    kperp_eff = effective_kperp2(phi, cache, vol_fac)
    heat_weight = jnp.real(
        jnp.sum(
            heat_flux_species(
                state_arr,
                phi,
                zero_field,
                zero_field,
                cache,
                grid,
                params_linear,
                flux_fac,
            )
        )
        / norm2
    )
    gamma = jnp.real(eigenvalue)
    ql_proxy = (
        gamma
        * heat_weight
        / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
    )
    return gamma, jnp.imag(eigenvalue), kperp_eff, heat_weight, ql_proxy


__all__ = [
    "_mode21_vmec_boozer_linear_context",
    "_mode21_vmec_boozer_quasilinear_features",
]
