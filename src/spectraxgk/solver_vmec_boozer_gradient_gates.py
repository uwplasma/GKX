"""VMEC/Boozer full-chain gradient gates for solver objectives."""

from __future__ import annotations

import importlib
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import fieldline_quadrature_weights, heat_flux_species
from spectraxgk.geometry.backend_discovery import discover_differentiable_geometry_backends
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.geometry.vmec_boozer_core import (
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
)
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.quasilinear import effective_kperp2, phi_norm2
from spectraxgk.solver_geometry_objectives import _objective_gate_rows
from spectraxgk.solver_nonlinear_window_objective import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)
from spectraxgk.solver_objective_core import (
    _default_gradient_linear_params,
    _default_gradient_linear_terms,
)
from spectraxgk.solver_vmec_state import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)

VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES = ("gamma", "omega")
VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
)
VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
    "nonlinear_window_heat_flux_mean",
    "nonlinear_window_heat_flux_cv",
    "nonlinear_window_heat_flux_trend",
)


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

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=int(ntheta), Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    state_shape = (
        int(n_laguerre),
        int(n_hermite),
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    params_linear = _default_gradient_linear_params()
    terms = _default_gradient_linear_terms()

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


def mode21_vmec_boozer_linear_frequency_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 5.0e-2,
    atol: float = 2.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
    _linear_context_fn: Any = _mode21_vmec_boozer_linear_context,
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

    context = _linear_context_fn(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=1,
        n_hermite=1,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, _eigenvector: jnp.ndarray, _x: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.asarray([jnp.real(eigenvalue), jnp.imag(eigenvalue)])

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
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
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
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
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 2.0e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
    _linear_context_fn: Any = _mode21_vmec_boozer_linear_context,
    _quasilinear_features_fn: Any = _mode21_vmec_boozer_quasilinear_features,
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
    context = _linear_context_fn(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=2,
        n_hermite=3,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = _quasilinear_features_fn(
            eigenvalue,
            eigenvector,
            x,
            context,
        )
        return jnp.asarray([gamma, omega, kperp_eff, heat_weight, ql_proxy])

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
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
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": bool(
            by_objective["linear_heat_flux_weight"]
            and by_objective["mixing_length_heat_flux_proxy"]
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


def mode21_vmec_boozer_nonlinear_window_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 7.5e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
    nonlinear_dt: float = 0.18,
    nonlinear_steps: int = 96,
    tail_fraction: float = 0.30,
    _linear_context_fn: Any = _mode21_vmec_boozer_linear_context,
    _quasilinear_features_fn: Any = _mode21_vmec_boozer_quasilinear_features,
    _window_metrics_fn: Any = _reduced_nonlinear_window_metrics_from_linear_observables,
) -> dict[str, object]:
    """Validate VMEC/Boozer-state gradients of a nonlinear-window estimator.

    The gate reuses the full ``vmec_jax`` state to ``booz_xform_jax`` to
    SPECTRAX-GK linear-RHS path from the quasilinear gradient gate, then feeds
    the isolated eigenpair observables into a differentiable late-time
    heat-flux-envelope estimator.  It is a reduced nonlinear-window
    differentiability gate; converged nonlinear turbulence windows and
    optimized-equilibrium nonlinear audits remain separate promotion gates.
    """

    start = time.perf_counter()
    context = _linear_context_fn(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=2,
        n_hermite=3,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = _quasilinear_features_fn(
            eigenvalue,
            eigenvector,
            x,
            context,
        )
        nl_mean, nl_cv, nl_trend = _window_metrics_fn(
            gamma,
            kperp_eff,
            heat_weight,
            dt=nonlinear_dt,
            steps=nonlinear_steps,
            tail_fraction=tail_fraction,
        )
        return jnp.asarray(
            [
                gamma,
                omega,
                kperp_eff,
                heat_weight,
                ql_proxy,
                nl_mean,
                nl_cv,
                nl_trend,
            ]
        )

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
        objective_names=VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES
    }
    nonlinear_window_gate = bool(
        by_objective["nonlinear_window_heat_flux_mean"]
        and by_objective["nonlinear_window_heat_flux_cv"]
        and by_objective["nonlinear_window_heat_flux_trend"]
    )
    return {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc geometry "
            "-> SPECTRAX-GK linear-RHS eigenpair -> reduced nonlinear-window estimator gradient"
        ),
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": bool(
            by_objective["linear_heat_flux_weight"]
            and by_objective["mixing_length_heat_flux_proxy"]
        ),
        "nonlinear_window_gradient_gate": nonlinear_window_gate,
        "nonlinear_window_config": {
            "model": "smooth_logistic_heat_flux_envelope_from_linear_observables",
            "dt": float(nonlinear_dt),
            "steps": int(nonlinear_steps),
            "tail_fraction": float(tail_fraction),
        },
        "elapsed_seconds": float(time.perf_counter() - start),
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Use this as a reduced nonlinear-window estimator-gradient gate only. Full stellarator "
            "heat-flux optimization still requires converged nonlinear SPECTRAX-GK window gradients "
            "or robust adjoint/finite-difference audits on optimized equilibria."
        ),
    }


__all__ = [
    "VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES",
    "VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES",
    "_mode21_vmec_boozer_linear_context",
    "_mode21_vmec_boozer_quasilinear_features",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_nonlinear_window_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
]
