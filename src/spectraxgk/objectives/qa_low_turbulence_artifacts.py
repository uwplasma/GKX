"""JSON-ready payload builders for reduced QA low-turbulence comparisons."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
)
from spectraxgk.objectives.qa_low_turbulence_model import (
    _qa_low_turbulence_core,
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_window_metrics,
)
from spectraxgk.objectives.qa_low_turbulence_optimizer import optimize_qa_low_turbulence
from spectraxgk.objectives.stellarator import PARAMETER_NAMES, _validate_params


def reduced_boundary_surface(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> dict[str, Any]:
    """Return a reduced max-mode-1 boundary surface for visualization."""

    cfg = config or QALowTurbulenceConfig()
    p = np.asarray(_validate_params(params), dtype=float)
    minor_shift, elong_shift, ripple, shear_shift = p
    aspect = float(_qa_low_turbulence_core(p.tolist(), cfg)["aspect"])
    theta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_nzeta), endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    major_radius = 1.0
    minor_radius = major_radius / max(aspect, 1.0e-6)
    elongation = 1.0 + 0.26 * float(elong_shift)
    helical = float(ripple)
    nfp = int(cfg.n_field_periods)
    visual_helical_radial = 1.35 * helical
    visual_helical_vertical = 1.05 * helical
    axis_radius = major_radius * (1.0 + 0.22 * helical * np.cos(nfp * zz))
    axis_height = minor_radius * 1.10 * helical * np.sin(nfp * zz)
    radius = axis_radius + minor_radius * (
        np.cos(tt)
        + visual_helical_radial * np.cos(tt - nfp * zz)
        + 0.030 * float(shear_shift) * np.cos(2.0 * tt)
        + 0.060 * float(minor_shift) * np.cos(tt + nfp * zz)
    )
    height = axis_height + minor_radius * (
        elongation * np.sin(tt)
        + visual_helical_vertical * np.sin(tt - nfp * zz)
        + 0.040 * float(shear_shift) * np.sin(2.0 * tt)
    )
    x = radius * np.cos(zz)
    y = radius * np.sin(zz)
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "x": x.tolist(),
        "y": y.tolist(),
        "z": height.tolist(),
        "visual_helical_radial_amplitude": float(visual_helical_radial),
        "visual_helical_vertical_amplitude": float(visual_helical_vertical),
        "reduced_boundary_scope": "max-mode-1 visualization, not a solved VMEC equilibrium",
    }


def reduced_lcfs_bmag(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
) -> dict[str, Any]:
    """Return a reduced LCFS ``|B|`` map for QA visualization."""

    cfg = config or QALowTurbulenceConfig()
    p = np.asarray(_validate_params(params), dtype=float)
    core = _qa_low_turbulence_core(p.tolist(), cfg)
    theta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_ntheta), endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, int(cfg.surface_nzeta), endpoint=False)
    tt, zz = np.meshgrid(theta, zeta, indexing="ij")
    nfp = int(cfg.n_field_periods)
    ripple = float(p[2])
    elong = float(p[1])
    qa_amp = float(core["qa_residual"])
    bmag = (
        1.0
        + 0.055 * np.cos(tt)
        + 0.018 * elong * np.cos(2.0 * tt)
        + 0.220 * ripple * np.cos(tt - nfp * zz)
        + 0.030 * qa_amp * np.cos(2.0 * tt - nfp * zz)
    )
    return {
        "theta": theta.tolist(),
        "zeta": zeta.tolist(),
        "bmag": bmag.tolist(),
        "reduced_bmag_scope": "synthetic LCFS |B| map from reduced QA controls",
    }


def _window_mean_for_gradient(
    params: Sequence[float],
    config: QALowTurbulenceConfig,
    *,
    density_gradient: float,
    temperature_gradient: float,
) -> tuple[float, float, float, float]:
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        config,
        density_gradient=density_gradient,
        temperature_gradient=temperature_gradient,
    )
    window = qa_low_turbulence_window_metrics(
        times,
        heat_flux,
        tail_fraction=config.nonlinear_tail_fraction,
    )
    core = _qa_low_turbulence_core(params, config)
    return (
        float(window["mean"]),
        float(window["cv"]),
        float(window["trend"]),
        float(core["growth_rate"]),
    )


def _scan_density_gradient(
    params: Sequence[float],
    config: QALowTurbulenceConfig,
) -> dict[str, Any]:
    gradients = np.asarray(config.scan_density_gradients, dtype=float)
    means = []
    cvs = []
    trends = []
    gammas = []
    for aln in gradients:
        mean, cv, trend, gamma = _window_mean_for_gradient(
            params,
            config,
            density_gradient=float(aln),
            temperature_gradient=config.fixed_temperature_gradient,
        )
        means.append(mean)
        cvs.append(cv)
        trends.append(trend)
        gammas.append(gamma)
    slope = float(np.polyfit(gradients, np.asarray(means, dtype=float), deg=1)[0])
    return {
        "density_gradient_axis": gradients.tolist(),
        "fixed_temperature_gradient": float(config.fixed_temperature_gradient),
        "heat_flux_mean": means,
        "heat_flux_cv": cvs,
        "heat_flux_trend": trends,
        "growth_rate": gammas,
        "linear_slope_dQ_d_a_over_Ln": slope,
    }


def _long_window_convergence_gate(
    times: np.ndarray,
    heat_flux: np.ndarray,
    window: dict[str, Any],
    config: QALowTurbulenceConfig,
) -> dict[str, Any]:
    """Return explicit convergence checks for the fixed-gradient trace."""

    t = np.asarray(times, dtype=float)
    q = np.asarray(heat_flux, dtype=float)
    start = int(window["start_index"])
    tail = q[start:]
    if tail.size < 4:
        raise ValueError(
            "long-window convergence gate requires at least four late-window samples"
        )
    split = max(1, tail.size // 2)
    first_mean = float(np.mean(tail[:split]))
    second_mean = float(np.mean(tail[split:]))
    full_mean = float(np.mean(tail))
    denom = max(abs(full_mean), 1.0e-14)
    half_mean_rel_change = abs(second_mean - first_mean) / denom
    running_mean = np.cumsum(tail) / np.arange(1, tail.size + 1, dtype=float)
    checkpoint = max(0, int(round(0.75 * (tail.size - 1))))
    running_mean_rel_change = (
        abs(float(running_mean[-1]) - float(running_mean[checkpoint])) / denom
    )
    tmax = float(t[-1])
    passed = bool(
        tmax >= float(config.long_window_min_time)
        and float(window["cv"]) <= float(config.long_window_max_cv)
        and float(window["trend"]) <= float(config.long_window_max_trend)
        and half_mean_rel_change <= float(config.long_window_max_half_mean_rel_change)
    )
    return {
        "passed": passed,
        "tmax": tmax,
        "minimum_tmax": float(config.long_window_min_time),
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "half_window_relative_mean_change": float(half_mean_rel_change),
        "running_mean_checkpoint_fraction": 0.75,
        "running_mean_relative_change_since_checkpoint": float(running_mean_rel_change),
        "max_cv": float(config.long_window_max_cv),
        "max_trend": float(config.long_window_max_trend),
        "max_half_window_relative_mean_change": float(
            config.long_window_max_half_mean_rel_change
        ),
    }


def _fixed_trace_payload(
    params: Sequence[float], config: QALowTurbulenceConfig
) -> dict[str, Any]:
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        config,
        density_gradient=config.fixed_density_gradient,
        temperature_gradient=config.fixed_temperature_gradient,
    )
    window = qa_low_turbulence_window_metrics(
        times,
        heat_flux,
        tail_fraction=config.nonlinear_tail_fraction,
    )
    times_np = np.asarray(times, dtype=float)
    heat_flux_np = np.asarray(heat_flux, dtype=float)
    window_payload = {key: float(value) for key, value in window.items()}
    window_payload["start_index"] = int(window["start_index"])
    convergence_gate = _long_window_convergence_gate(
        times_np, heat_flux_np, window_payload, config
    )
    return {
        "density_gradient": float(config.fixed_density_gradient),
        "temperature_gradient": float(config.fixed_temperature_gradient),
        "trace_kind": "smooth_reduced_nonlinear_envelope_not_full_turbulent_gk",
        "trace_equation": "dE/dt = 2 gamma E - alpha E^2; Q_env = W E",
        "times": [float(x) for x in times_np],
        "heat_flux": [float(x) for x in heat_flux_np],
        "window": window_payload,
        "long_window_convergence": convergence_gate,
    }


def qa_low_turbulence_comparison_payload(
    config: QALowTurbulenceConfig | None = None,
    *,
    finite_difference_workers: int = 1,
) -> dict[str, Any]:
    """Build the full JSON-ready aspect-6 QA low-turbulence comparison."""

    cfg = config or QALowTurbulenceConfig()
    control = optimize_qa_low_turbulence(
        includes_nonlinear_heat_flux=False,
        config=cfg,
        finite_difference_workers=finite_difference_workers,
    )
    transport = optimize_qa_low_turbulence(
        includes_nonlinear_heat_flux=True,
        config=cfg,
        finite_difference_workers=finite_difference_workers,
    )
    results = [control.to_dict(), transport.to_dict()]
    obs_index = {name: i for i, name in enumerate(QA_LOW_TURBULENCE_OBSERVABLE_NAMES)}
    control_q = float(control.final_observables[obs_index["nonlinear_heat_flux_mean"]])
    transport_q = float(
        transport.final_observables[obs_index["nonlinear_heat_flux_mean"]]
    )
    reduction = 1.0 - transport_q / max(control_q, 1.0e-14)
    design_payloads = []
    for result in results:
        params = result["final_params"]
        design_payloads.append(
            {
                "design_name": result["design_name"],
                "final_params": params,
                "final_observables": result["final_observables"],
                "density_gradient_scan": _scan_density_gradient(params, cfg),
                "fixed_gradient_trace": _fixed_trace_payload(params, cfg),
                "surface": reduced_boundary_surface(params, cfg),
                "lcfs_bmag": reduced_lcfs_bmag(params, cfg),
            }
        )
    all_gates_passed = all(
        bool(result["scalar_gradient_gate"]["passed"])
        and bool(result["residual_gradient_gate"]["passed"])
        and bool(result["observable_gradient_gate"]["passed"])
        for result in results
    )
    long_window_gates_passed = all(
        bool(design["fixed_gradient_trace"]["long_window_convergence"]["passed"])
        for design in design_payloads
    )
    constraints_passed = all(
        abs(result["final_observables"][obs_index["aspect"]] - cfg.target_aspect)
        / cfg.target_aspect
        < 2.5e-2
        and result["final_observables"][obs_index["mean_iota"]]
        >= cfg.iota_operating_floor - 2.0e-3
        and result["final_observables"][obs_index["qa_residual"]] < 2.5e-2
        for result in results
    )
    transport_passed = bool(transport_q <= 0.95 * control_q)
    passed = bool(
        all_gates_passed
        and constraints_passed
        and transport_passed
        and long_window_gates_passed
    )
    return {
        "kind": "qa_low_turbulence_comparison",
        "claim_level": (
            "reduced_differentiable_qa_low_turbulence_comparison_"
            "not_full_vmec_nonlinear_transport_optimization"
        ),
        "target_aspect": float(cfg.target_aspect),
        "minimum_iota": float(cfg.min_iota),
        "operating_iota_floor": float(cfg.iota_operating_floor),
        "fixed_density_gradient": float(cfg.fixed_density_gradient),
        "fixed_temperature_gradient": float(cfg.fixed_temperature_gradient),
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": list(QA_LOW_TURBULENCE_OBSERVABLE_NAMES),
        "results": results,
        "designs": design_payloads,
        "comparison_metrics": {
            "control_design_heat_flux_mean": control_q,
            "transport_design_heat_flux_mean": transport_q,
            "relative_heat_flux_reduction_at_fixed_gradients": float(reduction),
            "constraints_passed": bool(constraints_passed),
            "transport_reduction_gate_passed": transport_passed,
            "long_window_gates_passed": bool(long_window_gates_passed),
            "ad_fd_gates_passed": bool(all_gates_passed),
            "passed": passed,
            "reduced_differentiable_plumbing_passed": bool(all_gates_passed),
            "full_vmec_nonlinear_differentiable_plumbing_passed": False,
        },
        "differentiable_plumbing": {
            "stages": [
                "reduced QA controls",
                "geometry constraints and reduced LCFS visualization",
                "linear ITG feature map",
                "quasilinear mixing-length diagnostic",
                "long-window differentiable nonlinear heat-flux envelope",
                "weighted optimization residuals",
                "scalar, residual, and observable AD-vs-FD gates",
            ],
            "all_scalar_objective_gates_passed": all(
                bool(result["scalar_gradient_gate"]["passed"]) for result in results
            ),
            "all_residual_jacobian_gates_passed": all(
                bool(result["residual_gradient_gate"]["passed"]) for result in results
            ),
            "all_observable_jacobian_gates_passed": all(
                bool(result["observable_gradient_gate"]["passed"]) for result in results
            ),
            "passed": bool(all_gates_passed),
        },
        "model_equations": {
            "objective": (
                "||r||^2 with aspect, minimum-iota, operating-iota, QA, QA-compatible "
                "helical-shaping, regularization, and optional sqrt(weight * late-window reduced "
                "nonlinear heat flux) residuals"
            ),
            "nonlinear_envelope": "dE/dt = 2 gamma E - alpha E^2; Q_env = W_i E; fixed-step RK2",
            "gradient_scan": "fixed a/L_T while scanning a/L_n and refitting late-window Q_env means",
        },
        "config": asdict(cfg),
        "scope_notes": [
            "The surface and |B| maps are reduced max-mode-1 visualizations, not solved VMEC equilibria.",
            "The nonlinear heat-flux objective is a differentiable envelope used for optimization plumbing.",
            "Production nonlinear claims still require long post-transient replicated SPECTRAX-GK windows.",
        ],
    }


__all__ = [
    "_fixed_trace_payload",
    "_long_window_convergence_gate",
    "_scan_density_gradient",
    "qa_low_turbulence_comparison_payload",
    "reduced_boundary_surface",
    "reduced_lcfs_bmag",
]
