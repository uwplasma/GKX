"""Solver-ready geometry-gradient gates for linear solver objectives."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import (
    fieldline_quadrature_weights,
    heat_flux_species,
    particle_flux_species,
)
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.quasilinear import effective_kperp2, phi_norm2
from spectraxgk.objectives.geometry import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    _objective_gate_rows,
    default_solver_geometry_design_params,
    solver_ready_geometry_mapping,
)
from spectraxgk.objectives.core import (
    SOLVER_OBJECTIVE_NAMES,
    solver_objective_vector_from_geometry,
    _default_gradient_linear_params,
    _default_gradient_linear_terms,
)


def _linear_eigenpair_quasilinear_features(
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


def solver_objective_branch_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
    n_laguerre: int = 2,
    n_hermite: int = 1,
    _quasilinear_features_fn: Any = _linear_eigenpair_quasilinear_features,
    _objective_vector_fn: Any = solver_objective_vector_from_geometry,
) -> dict[str, object]:
    """Validate branch continuity and AD/FD sensitivities for solver objectives.

    This gate is the lightweight counterpart of the VMEC/Boozer offline gates.
    It uses the solver-ready differentiable geometry contract so CI can check
    the objective path without optional geometry backends. The report requires
    the max-growth branch to stay dominant under central finite-difference
    perturbations and validates the objective sensitivities with the implicit
    left/right eigenpair method.
    """

    p = (
        default_solver_geometry_design_params()
        if params is None
        else jnp.asarray(params)
    )
    if p.ndim != 1 or int(p.size) != len(SOLVER_GEOMETRY_PARAMETER_NAMES):
        raise ValueError(
            f"params must be a length-{len(SOLVER_GEOMETRY_PARAMETER_NAMES)} vector"
        )
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    state_shape = (
        n_laguerre_int,
        n_hermite_int,
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    params_linear = _default_gradient_linear_params()
    terms = _default_gradient_linear_terms()
    theta = jnp.asarray(grid.z)

    def geometry_for(x: jnp.ndarray):
        return flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="solver_ready_branch_gradient_gate",
            validate_finite=False,
        )

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(
            grid, geometry_for(x), params_linear, n_laguerre_int, n_hermite_int
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

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = _quasilinear_features_fn(
            eigenvalue,
            eigenvector,
            x,
            {
                "geometry_for": geometry_for,
                "grid": grid,
                "params_linear": params_linear,
                "n_laguerre": n_laguerre_int,
                "n_hermite": n_hermite_int,
                "state_shape": state_shape,
                "rhs_phi": rhs_phi,
            },
        )
        geom = geometry_for(x)
        cache = cache_for(x)
        state_arr = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state_arr, cache)
        zero_field = jnp.zeros_like(phi)
        _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
        particle_weight = jnp.real(
            jnp.sum(
                particle_flux_species(
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
            / phi_norm2(
                phi, cache, params_linear, fieldline_quadrature_weights(geom, grid)[0]
            )
        )
        return jnp.asarray(
            [gamma, omega, kperp_eff, heat_weight, particle_weight, ql_proxy]
        )

    base_matrix = matrix_fn(p)
    base_eigs = np.asarray(jnp.linalg.eigvals(base_matrix))
    if base_eigs.ndim != 1 or base_eigs.size == 0:
        raise ValueError("matrix_fn must return at least one eigenvalue")
    base_index = int(np.argmax(np.real(base_eigs)))
    base_value = base_eigs[base_index]
    base_gap = (
        float("inf")
        if base_eigs.size == 1
        else float(np.min(np.abs(np.delete(base_eigs, base_index) - base_value)))
    )
    eye = jnp.eye(int(p.size), dtype=p.dtype)
    branch_rows = []
    for i, name in enumerate(SOLVER_GEOMETRY_PARAMETER_NAMES):
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            p_i = p + float(sign) * float(fd_step) * eye[i]
            eigs_i = np.asarray(jnp.linalg.eigvals(matrix_fn(p_i)))
            nearest_index = int(np.argmin(np.abs(eigs_i - base_value)))
            dominant_index = int(np.argmax(np.real(eigs_i)))
            nearest_value = eigs_i[nearest_index]
            nearest_gap = (
                float("inf")
                if eigs_i.size == 1
                else float(
                    np.min(np.abs(np.delete(eigs_i, nearest_index) - nearest_value))
                )
            )
            row_passed = bool(
                nearest_index == dominant_index and nearest_gap >= float(gap_floor)
            )
            branch_rows.append(
                {
                    "parameter": name,
                    "direction": label,
                    "nearest_index": nearest_index,
                    "dominant_index": dominant_index,
                    "nearest_real": float(np.real(nearest_value)),
                    "nearest_imag": float(np.imag(nearest_value)),
                    "nearest_gap": nearest_gap,
                    "passed": row_passed,
                }
            )

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=SOLVER_GEOMETRY_PARAMETER_NAMES,
        objective_names=SOLVER_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    value_vector = _objective_vector_fn(
        geometry_for(p),
        selected_ky_index=1,
        n_laguerre=n_laguerre_int,
        n_hermite=n_hermite_int,
        ny=cfg.grid.Ny,
    )
    value_np = np.asarray(value_vector, dtype=float)
    value_finite = bool(np.all(np.isfinite(value_np)))
    branch_passed = bool(
        base_gap >= float(gap_floor) and all(row["passed"] for row in branch_rows)
    )
    ad_fd_passed = bool(gate["passed"] and all(row["passed"] for row in rows))
    return {
        "kind": "solver_objective_branch_gradient_gate",
        "passed": bool(value_finite and branch_passed and ad_fd_passed),
        "source_scope": "solver_ready_geometry_contract",
        "claim_scope": (
            "solver-objective branch-continuity and implicit AD/FD gate on the "
            "solver-ready geometry contract; VMEC/Boozer production gates remain separate"
        ),
        "parameter_names": list(SOLVER_GEOMETRY_PARAMETER_NAMES),
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "params": np.asarray(p, dtype=float).tolist(),
        "grid": {
            "Nx": int(cfg.grid.Nx),
            "Ny": int(cfg.grid.Ny),
            "Nz": int(cfg.grid.Nz),
            "selected_ky_index": 1,
        },
        "n_laguerre": n_laguerre_int,
        "n_hermite": n_hermite_int,
        "state_size": int(np.prod(state_shape)),
        "value_evaluator_finite": value_finite,
        "value_evaluator_objectives": value_np.tolist(),
        "branch_continuity_gate": branch_passed,
        "base_selected_index": base_index,
        "base_eigenvalue_real": float(np.real(base_value)),
        "base_eigenvalue_imag": float(np.imag(base_value)),
        "base_eigenvalue_gap": base_gap,
        "branch_rows": branch_rows,
        "ad_fd_gate": ad_fd_passed,
        "objective_gates": rows,
        "eigenpair_gate": gate,
    }


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

    p = (
        default_solver_geometry_design_params()
        if params is None
        else jnp.asarray(params)
    )
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
        return build_linear_cache(
            grid, geometry_for(x), params_linear, n_laguerre, n_hermite
        )

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
        return explicit_complex_operator_matrix(
            lambda state: rhs_phi(state, cache)[0], state_shape
        )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        geom = geometry_for(x)
        cache = cache_for(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state, cache)
        zero_field = jnp.zeros_like(phi)
        vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
        norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        heat_weight = jnp.real(
            jnp.sum(
                heat_flux_species(
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
                particle_flux_species(
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
        ql_proxy = (
            gamma
            * heat_weight
            / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
        )
        return jnp.asarray(
            [
                gamma,
                jnp.imag(eigenvalue),
                kperp_eff,
                heat_weight,
                particle_weight,
                ql_proxy,
            ]
        )

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
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in SOLVER_OBJECTIVE_NAMES
    }
    linear_growth_gate = bool(by_objective["gamma"] and by_objective["omega"])
    quasilinear_weight_gate = bool(
        by_objective["linear_heat_flux_weight"]
        and by_objective["mixing_length_heat_flux_proxy"]
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
        "grid": {
            "Nx": int(cfg.grid.Nx),
            "Ny": int(cfg.grid.Ny),
            "Nz": int(cfg.grid.Nz),
            "selected_ky_index": 1,
        },
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


__all__ = [
    "_linear_eigenpair_quasilinear_features",
    "linear_solver_geometry_gradient_report",
    "solver_objective_branch_gradient_report",
]
