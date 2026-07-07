"""Solver-ready geometry-gradient gates for linear solver objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.autodiff_validation import (
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
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.operators.linear.rhs import linear_rhs_cached
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import LinearParams, LinearTerms
from spectraxgk.diagnostics.quasilinear_transport import effective_kperp2, phi_norm2
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


@dataclass(frozen=True)
class _SolverReadyLinearContext:
    cfg: CycloneBaseCase
    grid: Any
    n_laguerre: int
    n_hermite: int
    state_shape: tuple[int, int, int, int, int]
    params_linear: LinearParams
    terms: LinearTerms
    theta: jnp.ndarray
    source_model: str

    def geometry_for(self, x: jnp.ndarray) -> Any:
        return flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, self.theta),
            source_model=self.source_model,
            validate_finite=False,
        )

    def cache_for(self, x: jnp.ndarray) -> Any:
        return build_linear_cache(
            self.grid,
            self.geometry_for(x),
            self.params_linear,
            self.n_laguerre,
            self.n_hermite,
        )

    def rhs_phi(self, state: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state,
            cache,
            self.params_linear,
            terms=self.terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(self, x: jnp.ndarray) -> jnp.ndarray:
        cache = self.cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state: self.rhs_phi(state, cache)[0], self.state_shape
        )

    def quasilinear_feature_context(self) -> dict[str, Any]:
        return {
            "geometry_for": self.geometry_for,
            "grid": self.grid,
            "params_linear": self.params_linear,
            "n_laguerre": self.n_laguerre,
            "n_hermite": self.n_hermite,
            "state_shape": self.state_shape,
            "rhs_phi": self.rhs_phi,
        }


def _solver_ready_linear_context(
    *,
    n_laguerre: int,
    n_hermite: int,
    source_model: str,
    params_linear: LinearParams | None = None,
    terms: LinearTerms | None = None,
) -> _SolverReadyLinearContext:
    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    return _SolverReadyLinearContext(
        cfg=cfg,
        grid=grid,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        state_shape=(n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size),
        params_linear=(
            _default_gradient_linear_params()
            if params_linear is None
            else params_linear
        ),
        terms=_default_gradient_linear_terms() if terms is None else terms,
        theta=jnp.asarray(grid.z),
        source_model=source_model,
    )


def _linear_transport_weights(
    *,
    state: jnp.ndarray,
    phi: jnp.ndarray,
    cache: Any,
    grid: Any,
    params_linear: LinearParams,
    flux_fac: jnp.ndarray,
    norm2: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return heat and particle weights for one normalized linear eigenmode."""

    zero_field = jnp.zeros_like(phi)

    def normalized_flux(flux_fn: Any) -> jnp.ndarray:
        return jnp.real(
            jnp.sum(
                flux_fn(
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

    return normalized_flux(heat_flux_species), normalized_flux(particle_flux_species)


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
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
    kperp_eff = effective_kperp2(phi, cache, vol_fac)
    heat_weight, _particle_weight = _linear_transport_weights(
        state=state_arr,
        phi=phi,
        cache=cache,
        grid=grid,
        params_linear=params_linear,
        flux_fac=flux_fac,
        norm2=norm2,
    )
    gamma = jnp.real(eigenvalue)
    ql_proxy = (
        gamma
        * heat_weight
        / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
    )
    return gamma, jnp.imag(eigenvalue), kperp_eff, heat_weight, ql_proxy


def _validate_branch_gradient_inputs(
    params: jnp.ndarray | np.ndarray | None,
    *,
    n_laguerre: int,
    n_hermite: int,
) -> tuple[jnp.ndarray, int, int]:
    p = default_solver_geometry_design_params() if params is None else jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != len(SOLVER_GEOMETRY_PARAMETER_NAMES):
        raise ValueError(
            f"params must be a length-{len(SOLVER_GEOMETRY_PARAMETER_NAMES)} vector"
        )
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")
    return p, n_laguerre_int, n_hermite_int


def _solver_branch_objective_fn(
    context: _SolverReadyLinearContext,
    quasilinear_features_fn: Any,
):
    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = quasilinear_features_fn(
            eigenvalue,
            eigenvector,
            x,
            context.quasilinear_feature_context(),
        )
        geom = context.geometry_for(x)
        cache = context.cache_for(x)
        state_arr = jnp.reshape(eigenvector, context.state_shape)
        _rhs, phi = context.rhs_phi(state_arr, cache)
        vol_fac, flux_fac = fieldline_quadrature_weights(geom, context.grid)
        _heat_weight_check, particle_weight = _linear_transport_weights(
            state=state_arr,
            phi=phi,
            cache=cache,
            grid=context.grid,
            params_linear=context.params_linear,
            flux_fac=flux_fac,
            norm2=phi_norm2(phi, cache, context.params_linear, vol_fac),
        )
        return jnp.asarray(
            [gamma, omega, kperp_eff, heat_weight, particle_weight, ql_proxy]
        )

    return objective_fn


def _base_branch_eigensystem(
    matrix: jnp.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, float]:
    base_eigs = np.asarray(jnp.linalg.eigvals(matrix))
    if base_eigs.ndim != 1 or base_eigs.size == 0:
        raise ValueError("matrix_fn must return at least one eigenvalue")
    base_index = int(np.argmax(np.real(base_eigs)))
    base_value = base_eigs[base_index]
    base_gap = (
        float("inf")
        if base_eigs.size == 1
        else float(np.min(np.abs(np.delete(base_eigs, base_index) - base_value)))
    )
    return base_eigs, base_index, base_value, base_gap


def _branch_row_gap(eigs: np.ndarray, selected_index: int, selected_value: np.ndarray) -> float:
    if eigs.size == 1:
        return float("inf")
    return float(np.min(np.abs(np.delete(eigs, selected_index) - selected_value)))


def _branch_continuity_rows(
    *,
    context: _SolverReadyLinearContext,
    p: jnp.ndarray,
    base_value: np.ndarray,
    fd_step: float,
    gap_floor: float,
) -> list[dict[str, Any]]:
    eye = jnp.eye(int(p.size), dtype=p.dtype)
    branch_rows = []
    for i, name in enumerate(SOLVER_GEOMETRY_PARAMETER_NAMES):
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            p_i = p + float(sign) * float(fd_step) * eye[i]
            eigs_i = np.asarray(jnp.linalg.eigvals(context.matrix_fn(p_i)))
            nearest_index = int(np.argmin(np.abs(eigs_i - base_value)))
            dominant_index = int(np.argmax(np.real(eigs_i)))
            nearest_value = eigs_i[nearest_index]
            nearest_gap = _branch_row_gap(eigs_i, nearest_index, nearest_value)
            branch_rows.append(
                {
                    "parameter": name,
                    "direction": label,
                    "nearest_index": nearest_index,
                    "dominant_index": dominant_index,
                    "nearest_real": float(np.real(nearest_value)),
                    "nearest_imag": float(np.imag(nearest_value)),
                    "nearest_gap": nearest_gap,
                    "passed": bool(
                        nearest_index == dominant_index
                        and nearest_gap >= float(gap_floor)
                    ),
                }
            )
    return branch_rows


def _solver_objective_gate_payload(
    *,
    context: _SolverReadyLinearContext,
    objective_fn: Any,
    p: jnp.ndarray,
    fd_step: float,
    rtol: float,
    atol: float,
    gap_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = implicit_eigenpair_observable_sensitivity_report(
        context.matrix_fn,
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
    return gate, rows


def _solver_objective_value_vector(
    *,
    context: _SolverReadyLinearContext,
    p: jnp.ndarray,
    n_laguerre: int,
    n_hermite: int,
    objective_vector_fn: Any,
) -> tuple[np.ndarray, bool]:
    value_vector = objective_vector_fn(
        context.geometry_for(p),
        selected_ky_index=1,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        ny=context.cfg.grid.Ny,
    )
    value_np = np.asarray(value_vector, dtype=float)
    return value_np, bool(np.all(np.isfinite(value_np)))


def _pack_solver_branch_gradient_report(
    *,
    context: _SolverReadyLinearContext,
    p: jnp.ndarray,
    n_laguerre: int,
    n_hermite: int,
    value_np: np.ndarray,
    value_finite: bool,
    branch_passed: bool,
    ad_fd_passed: bool,
    base_index: int,
    base_value: np.ndarray,
    base_gap: float,
    branch_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, object]:
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
            "Nx": int(context.cfg.grid.Nx),
            "Ny": int(context.cfg.grid.Ny),
            "Nz": int(context.cfg.grid.Nz),
            "selected_ky_index": 1,
        },
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(context.state_shape)),
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

    p, n_laguerre_int, n_hermite_int = _validate_branch_gradient_inputs(
        params,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
    )
    context = _solver_ready_linear_context(
        n_laguerre=n_laguerre_int,
        n_hermite=n_hermite_int,
        source_model="solver_ready_branch_gradient_gate",
    )
    objective_fn = _solver_branch_objective_fn(context, _quasilinear_features_fn)
    _base_eigs, base_index, base_value, base_gap = _base_branch_eigensystem(
        context.matrix_fn(p)
    )
    branch_rows = _branch_continuity_rows(
        context=context,
        p=p,
        base_value=base_value,
        fd_step=fd_step,
        gap_floor=gap_floor,
    )
    gate, rows = _solver_objective_gate_payload(
        context=context,
        objective_fn=objective_fn,
        p=p,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    value_np, value_finite = _solver_objective_value_vector(
        context=context,
        p=p,
        n_laguerre=n_laguerre_int,
        n_hermite=n_hermite_int,
        objective_vector_fn=_objective_vector_fn,
    )
    branch_passed = bool(
        base_gap >= float(gap_floor) and all(row["passed"] for row in branch_rows)
    )
    ad_fd_passed = bool(gate["passed"] and all(row["passed"] for row in rows))
    return _pack_solver_branch_gradient_report(
        context=context,
        p=p,
        n_laguerre=n_laguerre_int,
        n_hermite=n_hermite_int,
        value_np=value_np,
        value_finite=value_finite,
        branch_passed=branch_passed,
        ad_fd_passed=ad_fd_passed,
        base_index=base_index,
        base_value=base_value,
        base_gap=base_gap,
        branch_rows=branch_rows,
        rows=rows,
        gate=gate,
    )


def _validate_linear_geometry_gradient_params(
    params: jnp.ndarray | np.ndarray | None,
) -> jnp.ndarray:
    p = default_solver_geometry_design_params() if params is None else jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector")
    return p


def _linear_solver_objective_fn(context: _SolverReadyLinearContext):
    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        geom = context.geometry_for(x)
        cache = context.cache_for(x)
        state = jnp.reshape(eigenvector, context.state_shape)
        _rhs, phi = context.rhs_phi(state, cache)
        vol_fac, flux_fac = fieldline_quadrature_weights(geom, context.grid)
        norm2 = phi_norm2(phi, cache, context.params_linear, vol_fac)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        heat_weight, particle_weight = _linear_transport_weights(
            state=state,
            phi=phi,
            cache=cache,
            grid=context.grid,
            params_linear=context.params_linear,
            flux_fac=flux_fac,
            norm2=norm2,
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

    return objective_fn


def _linear_solver_gradient_gate_payload(
    *,
    context: _SolverReadyLinearContext,
    objective_fn: Any,
    p: jnp.ndarray,
    fd_step: float,
    rtol: float,
    atol: float,
    gap_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    gate = implicit_eigenpair_observable_sensitivity_report(
        context.matrix_fn,
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
    return gate, rows, by_objective


def _pack_linear_solver_geometry_gradient_report(
    *,
    context: _SolverReadyLinearContext,
    p: jnp.ndarray,
    n_laguerre: int,
    n_hermite: int,
    gate: dict[str, Any],
    rows: list[dict[str, Any]],
    by_objective: Mapping[str, bool],
) -> dict[str, object]:
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
            "Nx": int(context.cfg.grid.Nx),
            "Ny": int(context.cfg.grid.Ny),
            "Nz": int(context.cfg.grid.Nz),
            "selected_ky_index": 1,
        },
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(context.state_shape)),
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

    p = _validate_linear_geometry_gradient_params(params)
    n_laguerre = 2
    n_hermite = 1
    context = _solver_ready_linear_context(
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        source_model="solver_ready_geometry_gradient_gate",
    )
    gate, rows, by_objective = _linear_solver_gradient_gate_payload(
        context=context,
        objective_fn=_linear_solver_objective_fn(context),
        p=p,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    return _pack_linear_solver_geometry_gradient_report(
        context=context,
        p=p,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
        gate=gate,
        rows=rows,
        by_objective=by_objective,
    )


__all__ = [
    "_linear_eigenpair_quasilinear_features",
    "linear_solver_geometry_gradient_report",
    "solver_objective_branch_gradient_report",
]
