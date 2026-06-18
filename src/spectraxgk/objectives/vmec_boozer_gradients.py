"""VMEC/Boozer full-chain gradient gates for solver objectives."""

from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import implicit_eigenpair_observable_sensitivity_report
from spectraxgk.objectives.geometry import _objective_gate_rows
from spectraxgk.objectives.nonlinear_window import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)
from spectraxgk.objectives.vmec_boozer_context import (
    _mode21_vmec_boozer_linear_context,
    _mode21_vmec_boozer_quasilinear_features,
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
