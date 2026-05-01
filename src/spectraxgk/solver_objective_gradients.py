"""Production-adjacent solver-objective geometry-gradient gates.

These helpers validate gradients of actual SPECTRAX-GK linear-RHS observables
with respect to solver-ready geometry arrays.  They are deliberately stricter
than reduced optimization proxies, but still narrower than a full
``vmec_jax -> booz_xform_jax -> solver`` optimization claim.
"""

from __future__ import annotations

import importlib
from dataclasses import replace as dc_replace
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import gx_heat_flux_species, gx_particle_flux_species, gx_volume_factors
from spectraxgk.geometry.differentiable import (
    discover_differentiable_geometry_backends,
    flux_tube_geometry_from_mapping,
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
)
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_rhs_cached
from spectraxgk.quasilinear import effective_kperp2, phi_norm2


SOLVER_GEOMETRY_PARAMETER_NAMES = ("bmag_ripple", "curvature_drift_scale")
VMEC_BOOZER_STATE_PARAMETER_NAMES = ("Rcos_mid_surface_m1",)
SOLVER_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "linear_particle_flux_weight",
    "mixing_length_heat_flux_proxy",
)
VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES = ("gamma", "omega")
VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
)


def _vmec_boozer_state_parameter_name(radial_index: int, mode_index: int, *, default_mid_surface: int) -> str:
    if int(radial_index) == int(default_mid_surface) and int(mode_index) == 1:
        return VMEC_BOOZER_STATE_PARAMETER_NAMES[0]
    return f"Rcos_r{int(radial_index)}_m{int(mode_index)}"


def default_solver_geometry_design_params() -> jnp.ndarray:
    """Return the small geometry-design vector used by the release gate."""

    return jnp.asarray([0.05, 0.20], dtype=jnp.float32)


def solver_ready_geometry_mapping(params: jnp.ndarray, theta: jnp.ndarray) -> dict[str, Any]:
    """Map a two-parameter design vector into solver-ready flux-tube arrays."""

    p = jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector [bmag_ripple, curvature_drift_scale]")
    ripple = p[0]
    drift = p[1]
    theta_arr = jnp.asarray(theta)
    ones = jnp.ones_like(theta_arr)
    zeros = jnp.zeros_like(theta_arr)
    bmag = 1.0 + ripple * jnp.cos(theta_arr)
    return {
        "theta": theta_arr,
        "gradpar": 0.7 * ones,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta_arr),
        "gds2": 1.0 + 0.1 * ripple * jnp.cos(theta_arr),
        "gds21": 0.05 * ripple * jnp.sin(theta_arr),
        "gds22": 1.0 + 0.05 * ripple * jnp.cos(theta_arr),
        "cvdrift": drift * jnp.cos(theta_arr),
        "gbdrift": drift * jnp.cos(theta_arr),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones / (0.7 * bmag),
        "grho": ones,
        "q": 1.4,
        "s_hat": 0.0,
        "R0": 1.0,
        "nfp": 1,
    }


def _objective_gate_rows(
    report: dict[str, object],
    *,
    parameter_names: tuple[str, ...] = SOLVER_GEOMETRY_PARAMETER_NAMES,
    objective_names: tuple[str, ...] = SOLVER_OBJECTIVE_NAMES,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    implicit = np.asarray(report["jacobian_implicit"], dtype=float)
    finite_difference = np.asarray(report["jacobian_fd"], dtype=float)
    rows: list[dict[str, object]] = []
    for i, objective in enumerate(objective_names):
        for j, parameter in enumerate(parameter_names):
            fd_value = float(finite_difference[i, j])
            implicit_value = float(implicit[i, j])
            abs_error = abs(implicit_value - fd_value)
            rel_error = abs_error / max(abs(fd_value), float(atol))
            rows.append(
                {
                    "objective": objective,
                    "parameter": parameter,
                    "implicit": implicit_value,
                    "finite_difference": fd_value,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "atol": float(atol),
                    "rtol": float(rtol),
                    "passed": bool(abs_error <= float(atol) or rel_error <= float(rtol)),
                }
            )
    return rows


def linear_solver_geometry_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
) -> dict[str, object]:
    """Validate solver-objective geometry gradients on the actual linear RHS.

    The report differentiates a small electrostatic Cyclone-like linear
    operator with respect to geometry arrays entering the production cache. It
    uses implicit left/right eigenpair sensitivities and compares them with
    nearest-branch central finite differences.
    """

    p = default_solver_geometry_design_params() if params is None else jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    n_laguerre = 2
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    params_linear = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    theta = jnp.asarray(grid.z)

    def geometry_for(x: jnp.ndarray):
        return flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="solver_ready_geometry_gradient_gate",
            validate_finite=False,
        )

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(grid, geometry_for(x), params_linear, n_laguerre, n_hermite)

    def rhs_phi(state: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state,
            cache,
            params_linear,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(lambda state: rhs_phi(state, cache)[0], state_shape)

    def objective_fn(eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        geom = geometry_for(x)
        cache = cache_for(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state, cache)
        zero_field = jnp.zeros_like(phi)
        vol_fac, flux_fac = gx_volume_factors(geom, grid)
        norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        heat_weight = jnp.real(
            jnp.sum(
                gx_heat_flux_species(
                    state,
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
        particle_weight = jnp.real(
            jnp.sum(
                gx_particle_flux_species(
                    state,
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
        ql_proxy = gamma * heat_weight / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
        return jnp.asarray([gamma, jnp.imag(eigenvalue), kperp_eff, heat_weight, particle_weight, ql_proxy])

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(gate, rtol=rtol, atol=atol)
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name)) for name in SOLVER_OBJECTIVE_NAMES
    }
    linear_growth_gate = bool(by_objective["gamma"] and by_objective["omega"])
    quasilinear_weight_gate = bool(
        by_objective["linear_heat_flux_weight"] and by_objective["mixing_length_heat_flux_proxy"]
    )
    return {
        "kind": "linear_solver_geometry_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "solver_ready_geometry_contract",
        "claim_scope": (
            "actual_linear_rhs_solver_objectives; not yet a full "
            "vmec_jax_to_booz_xform_jax_to_solver gradient claim"
        ),
        "parameter_names": list(SOLVER_GEOMETRY_PARAMETER_NAMES),
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "params": np.asarray(p, dtype=float).tolist(),
        "grid": {"Nx": int(cfg.grid.Nx), "Ny": int(cfg.grid.Ny), "Nz": int(cfg.grid.Nz), "selected_ky_index": 1},
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(state_shape)),
        "linear_growth_gradient_gate": linear_growth_gate,
        "quasilinear_weight_gradient_gate": quasilinear_weight_gate,
        "nonlinear_window_gradient_gate": False,
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Promote this solver-ready geometry-gradient gate to actual mode-21 "
            "VMEC/Boozer state coefficients, then add nonlinear-window objective gradients."
        ),
    }


def mode21_vmec_boozer_linear_frequency_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 5.0e-2,
    atol: float = 2.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    """Validate a full VMEC/Boozer-state gradient of linear frequency.

    This is an offline manuscript artifact gate.  It perturbs one mid-surface
    VMEC Fourier coefficient, maps it through ``vmec_jax`` and
    ``booz_xform_jax`` into the mode-21 equal-arc flux-tube geometry contract,
    builds the SPECTRAX-GK linear RHS, and compares implicit eigenpair
    sensitivities against central finite differences.  Quasilinear flux-weight
    state gradients are intentionally not promoted here because the current
    full-chain diagnostic is substantially heavier and remains an optimization
    campaign lane.
    """

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
    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2 or int(base_Rcos.shape[1]) < 2:
        raise RuntimeError("vmec_jax state Rcos array must expose at least one non-axisymmetric mode")
    default_radial_index = int(base_Rcos.shape[0] // 2)
    radial_index_int = default_radial_index if radial_index is None else int(radial_index)
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_Rcos.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_Rcos.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_names = (
        _vmec_boozer_state_parameter_name(
            radial_index_int,
            mode_index_int,
            default_mid_surface=default_radial_index,
        ),
    )

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=int(ntheta), Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    n_laguerre = 1
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    params_linear = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def geometry_for(x: jnp.ndarray):
        traced_state = dc_replace(state, Rcos=base_Rcos.at[radial_index_int, mode_index_int].add(x[0]))
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

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(grid, geometry_for(x), params_linear, n_laguerre, n_hermite)

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state_arr: linear_rhs_cached(
                state_arr,
                cache,
                params_linear,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
            )[0],
            state_shape,
        )

    def objective_fn(eigenvalue: jnp.ndarray, _eigenvector: jnp.ndarray, _x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([jnp.real(eigenvalue), jnp.imag(eigenvalue)])

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=parameter_names,
        objective_names=VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES
    }
    return {
        "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc "
            "geometry -> SPECTRAX-GK linear-RHS eigenfrequency gradient"
        ),
        "case_name": str(case_name),
        "parameter_names": list(parameter_names),
        "objective_names": list(VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES),
        "parameter_indices": {"Rcos": [radial_index_int, mode_index_int]},
        "surface_index": None if surface_index is None else int(surface_index),
        "grid": {"Nx": int(cfg.grid.Nx), "Ny": int(cfg.grid.Ny), "Nz": int(cfg.grid.Nz), "selected_ky_index": 1},
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": None if surface_stencil_width is None else int(surface_stencil_width),
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(state_shape)),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": False,
        "nonlinear_window_gradient_gate": False,
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Promote the full-chain gate from eigenfrequency to quasilinear flux weights after "
            "the heavy Nl>=2 diagnostic is profiled and conditioned below manuscript runtime caps."
        ),
    }


def mode21_vmec_boozer_quasilinear_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 2.0e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    """Validate full VMEC/Boozer-state gradients of quasilinear observables.

    This offline manuscript gate is the production-gradient companion to
    :func:`mode21_vmec_boozer_linear_frequency_gradient_report`.  It uses a
    richer ``Nl=2, Nm=3`` moment basis so the electrostatic heat-flux weight is
    nonzero, then validates implicit eigenpair sensitivities of ``gamma``,
    ``omega``, ``<k_perp^2>``, the linear heat-flux weight, and the
    mixing-length heat-flux proxy against central finite differences.
    """

    start = time.perf_counter()
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
    base_Rcos = jnp.asarray(state.Rcos)
    if base_Rcos.ndim != 2 or int(base_Rcos.shape[1]) < 2:
        raise RuntimeError("vmec_jax state Rcos array must expose at least one non-axisymmetric mode")
    default_radial_index = int(base_Rcos.shape[0] // 2)
    radial_index_int = default_radial_index if radial_index is None else int(radial_index)
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_Rcos.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_Rcos.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_names = (
        _vmec_boozer_state_parameter_name(
            radial_index_int,
            mode_index_int,
            default_mid_surface=default_radial_index,
        ),
    )

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=int(ntheta), Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    n_laguerre = 2
    n_hermite = 3
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    params_linear = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def geometry_for(x: jnp.ndarray):
        traced_state = dc_replace(state, Rcos=base_Rcos.at[radial_index_int, mode_index_int].add(x[0]))
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

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(grid, geometry_for(x), params_linear, n_laguerre, n_hermite)

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
        return explicit_complex_operator_matrix(lambda state_arr: rhs_phi(state_arr, cache)[0], state_shape)

    def objective_fn(eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        geom = geometry_for(x)
        cache = build_linear_cache(grid, geom, params_linear, n_laguerre, n_hermite)
        state_arr = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state_arr, cache)
        zero_field = jnp.zeros_like(phi)
        vol_fac, flux_fac = gx_volume_factors(geom, grid)
        norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        heat_weight = jnp.real(
            jnp.sum(
                gx_heat_flux_species(
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
        ql_proxy = gamma * heat_weight / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
        return jnp.asarray([gamma, jnp.imag(eigenvalue), kperp_eff, heat_weight, ql_proxy])

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=parameter_names,
        objective_names=VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES
    }
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc "
            "geometry -> SPECTRAX-GK linear-RHS quasilinear heat-flux-weight gradient"
        ),
        "case_name": str(case_name),
        "parameter_names": list(parameter_names),
        "objective_names": list(VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES),
        "parameter_indices": {"Rcos": [radial_index_int, mode_index_int]},
        "surface_index": None if surface_index is None else int(surface_index),
        "grid": {"Nx": int(cfg.grid.Nx), "Ny": int(cfg.grid.Ny), "Nz": int(cfg.grid.Nz), "selected_ky_index": 1},
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": None if surface_stencil_width is None else int(surface_stencil_width),
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(state_shape)),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": bool(
            by_objective["linear_heat_flux_weight"] and by_objective["mixing_length_heat_flux_proxy"]
        ),
        "nonlinear_window_gradient_gate": False,
        "elapsed_seconds": float(time.perf_counter() - start),
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Use this as the full-chain quasilinear gradient gate for reduced linear/quasilinear "
            "stellarator objectives; keep full nonlinear-window VMEC/Boozer gradients as a separate future lane."
        ),
    }


__all__ = [
    "SOLVER_GEOMETRY_PARAMETER_NAMES",
    "SOLVER_OBJECTIVE_NAMES",
    "VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
    "default_solver_geometry_design_params",
    "linear_solver_geometry_gradient_report",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
    "solver_ready_geometry_mapping",
]
