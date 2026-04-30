"""Production-adjacent solver-objective geometry-gradient gates.

These helpers validate gradients of actual SPECTRAX-GK linear-RHS observables
with respect to solver-ready geometry arrays.  They are deliberately stricter
than reduced optimization proxies, but still narrower than a full
``vmec_jax -> booz_xform_jax -> solver`` optimization claim.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import gx_heat_flux_species, gx_particle_flux_species, gx_volume_factors
from spectraxgk.geometry.differentiable import flux_tube_geometry_from_mapping
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache, linear_rhs_cached
from spectraxgk.quasilinear import effective_kperp2, phi_norm2


SOLVER_GEOMETRY_PARAMETER_NAMES = ("bmag_ripple", "curvature_drift_scale")
SOLVER_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "linear_particle_flux_weight",
    "mixing_length_heat_flux_proxy",
)


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


def _objective_gate_rows(report: dict[str, object], *, rtol: float, atol: float) -> list[dict[str, object]]:
    implicit = np.asarray(report["jacobian_implicit"], dtype=float)
    finite_difference = np.asarray(report["jacobian_fd"], dtype=float)
    rows: list[dict[str, object]] = []
    for i, objective in enumerate(SOLVER_OBJECTIVE_NAMES):
        for j, parameter in enumerate(SOLVER_GEOMETRY_PARAMETER_NAMES):
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


__all__ = [
    "SOLVER_GEOMETRY_PARAMETER_NAMES",
    "SOLVER_OBJECTIVE_NAMES",
    "default_solver_geometry_design_params",
    "linear_solver_geometry_gradient_report",
    "solver_ready_geometry_mapping",
]
