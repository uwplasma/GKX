"""Reduced and VMEC/Boozer ITG objective tables for stellarator studies."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.core import (
    SolverScalarObjective,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.portfolio_contracts import aggregate_objective_portfolio
from spectraxgk.objectives.portfolio_sensitivity import (
    objective_portfolio_sensitivity_report,
)
from spectraxgk.objectives.stellarator_contracts import (
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGSampleSet,
)
from spectraxgk.objectives.stellarator_reduced import (
    _sampled_qa_itg_fields,
    _validate_params,
    default_stellarator_initial_params,
    qa_max_mode1_observables,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
)


def _precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return FD tolerances that are strict in x64 and stable in float32."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return max(float(fd_step), 5.0e-3), 5.0e-2, 6.0e-3


def stellarator_itg_sample_objective_table(
    params: jnp.ndarray | Sequence[float],
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    config: StellaratorITGOptimizationConfig | None = None,
    sample_set: StellaratorITGSampleSet | None = None,
) -> jnp.ndarray:
    """Return ``(surface, alpha, ky, objective)`` reduced ITG objective rows.

    This is the backend-free rehearsal of the production VMEC/Boozer sample
    table. It keeps the optimizer and gate semantics identical to the future
    real-geometry path while remaining cheap enough for CI.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    samples = sample_set or StellaratorITGSampleSet()
    fields = _sampled_qa_itg_fields(params, cfg, samples)
    if not objectives:
        raise ValueError("objectives must contain at least one objective name")
    columns = []
    for objective in objectives:
        key = str(objective).strip().lower()
        if key not in fields:
            raise ValueError(f"unknown portfolio objective {objective!r}")
        columns.append(fields[key])
    return jnp.stack(columns, axis=-1)


def stellarator_itg_reduced_portfolio_objective(
    params: jnp.ndarray | Sequence[float],
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    config: StellaratorITGOptimizationConfig | None = None,
    sample_set: StellaratorITGSampleSet | None = None,
    *,
    objective_weights: Sequence[float] | None = None,
) -> jnp.ndarray:
    """Reduce a sampled ITG growth/QL portfolio to one differentiable scalar."""

    samples = sample_set or StellaratorITGSampleSet()
    table = stellarator_itg_sample_objective_table(params, objectives, config, samples)
    return aggregate_objective_portfolio(
        table,
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=objective_weights,
        reduction=samples.reduction,
    )


def stellarator_itg_vmec_boozer_sample_objective_table_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    sample_set: StellaratorITGSampleSet | None = None,
    **vmec_boozer_options: Any,
) -> jnp.ndarray:
    """Return real VMEC/Boozer/SPECTRAX-GK rows on a ``StellaratorITGSampleSet``.

    This is the production bridge counterpart to
    :func:`stellarator_itg_sample_objective_table`: the sample axes are
    physical toroidal-flux, field-line alpha, and ``k_y rho_i`` values, while
    the objective columns are selected from the solver objective vector.
    """

    samples = sample_set or StellaratorITGSampleSet()
    objective_names = tuple(str(objective).strip().lower() for objective in objectives)
    if not objective_names:
        raise ValueError("objectives must contain at least one objective name")
    flat_table, _metadata = vmec_boozer_solver_objective_table_with_metadata_from_state(
        state,
        static,
        indata,
        wout,
        torflux_values=samples.surfaces,
        alphas=samples.alphas,
        ky_values=samples.ky_values,
        **vmec_boozer_options,
    )
    columns = [
        jnp.asarray(
            [
                solver_scalar_objective_from_vector(
                    row, cast(SolverScalarObjective, objective)
                )
                for row in flat_table
            ]
        )
        for objective in objective_names
    ]
    flat_objectives = jnp.stack(columns, axis=-1)
    return jnp.reshape(
        flat_objectives,
        (
            len(samples.surfaces),
            len(samples.alphas),
            len(samples.ky_values),
            len(objective_names),
        ),
    )


def stellarator_itg_vmec_boozer_portfolio_objective_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    sample_set: StellaratorITGSampleSet | None = None,
    *,
    objective_weights: Sequence[float] | None = None,
    **vmec_boozer_options: Any,
) -> jnp.ndarray:
    """Reduce real VMEC/Boozer/SPECTRAX-GK ITG rows to one portfolio scalar."""

    samples = sample_set or StellaratorITGSampleSet()
    table = stellarator_itg_vmec_boozer_sample_objective_table_from_state(
        state,
        static,
        indata,
        wout,
        objectives,
        samples,
        **vmec_boozer_options,
    )
    return aggregate_objective_portfolio(
        table,
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=objective_weights,
        reduction=samples.reduction,
    )


def stellarator_itg_portfolio_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    config: StellaratorITGOptimizationConfig | None = None,
    sample_set: StellaratorITGSampleSet | None = None,
    *,
    objective_weights: Sequence[float] | None = None,
    step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> dict[str, Any]:
    """AD/FD, conditioning, and covariance gate for the reduced ITG portfolio."""

    cfg = config or StellaratorITGOptimizationConfig()
    samples = sample_set or StellaratorITGSampleSet()
    objective_names = tuple(str(objective).strip().lower() for objective in objectives)
    fd_step, default_rtol, default_atol = _precision_gate_tolerances(cfg.fd_step)
    report = objective_portfolio_sensitivity_report(
        lambda x: stellarator_itg_sample_objective_table(
            x, objective_names, cfg, samples
        ),
        _validate_params(params),
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=objective_weights,
        reduction=samples.reduction,
        step=fd_step if step is None else float(step),
        rtol=default_rtol if rtol is None else float(rtol),
        atol=default_atol if atol is None else float(atol),
        min_rank=len(PARAMETER_NAMES),
        condition_number_limit=1.0e8,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    return {
        "kind": "stellarator_itg_portfolio_sensitivity_report",
        "claim_level": "reduced_multi_surface_alpha_ky_objective_gate_not_full_vmec_production",
        "passed": bool(report["passed"]),
        "parameter_names": list(PARAMETER_NAMES),
        "objective_names": list(objective_names),
        "sample_set": samples.to_dict(),
        "backend_boundary": (
            "same reducer/gate contract intended for vmec_jax -> booz_xform_jax "
            "objective rows after geometry parity passes"
        ),
        "portfolio_report": report,
    }


def _portfolio_sample_rows(
    sample_set: StellaratorITGSampleSet,
    *,
    sample_weights: np.ndarray,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i, surface in enumerate(sample_set.surfaces):
        for j, alpha in enumerate(sample_set.alphas):
            for k, ky in enumerate(sample_set.ky_values):
                rows.append(
                    {
                        "surface": float(surface),
                        "alpha": float(alpha),
                        "ky": float(ky),
                        "weight": float(sample_weights[i, j, k]),
                    }
                )
    return rows


def _normalized_axis_weights(values: Sequence[float] | None, size: int) -> np.ndarray:
    if values is None:
        return np.full((int(size),), 1.0 / float(size), dtype=float)
    arr = np.asarray(values, dtype=float)
    return arr / float(np.sum(arr))


def _normalized_sample_weight_array(sample_set: StellaratorITGSampleSet) -> np.ndarray:
    surface = _normalized_axis_weights(
        sample_set.surface_weights, len(sample_set.surfaces)
    )
    alpha = _normalized_axis_weights(sample_set.alpha_weights, len(sample_set.alphas))
    ky = _normalized_axis_weights(sample_set.ky_weights, len(sample_set.ky_values))
    return surface[:, None, None] * alpha[None, :, None] * ky[None, None, :]


def _normalized_objective_weights(
    objective_weights: Sequence[float] | None,
    size: int,
) -> np.ndarray:
    if objective_weights is None:
        return np.full((int(size),), 1.0 / float(size), dtype=float)
    arr = np.asarray(objective_weights, dtype=float)
    if arr.ndim != 1 or arr.size != int(size) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"objective_weights must be a finite length-{int(size)} vector"
        )
    if np.any(arr < 0.0) or float(np.sum(arr)) <= 0.0:
        raise ValueError("objective_weights must be non-negative with positive sum")
    return arr / float(np.sum(arr))


def stellarator_itg_portfolio_gate_payload(
    params: jnp.ndarray | Sequence[float] | None = None,
    objectives: Sequence[str] = ("growth", "quasilinear_flux"),
    config: StellaratorITGOptimizationConfig | None = None,
    sample_set: StellaratorITGSampleSet | None = None,
    *,
    objective_weights: Sequence[float] | None = None,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Return the JSON-ready reduced ITG portfolio gate artifact payload."""

    cfg = config or StellaratorITGOptimizationConfig()
    samples = sample_set or StellaratorITGSampleSet()
    p = (
        default_stellarator_initial_params()
        if params is None
        else _validate_params(params)
    )
    objective_names = tuple(str(objective).strip().lower() for objective in objectives)
    table = np.asarray(
        stellarator_itg_sample_objective_table(p, objective_names, cfg, samples),
        dtype=float,
    )
    obj_weights = _normalized_objective_weights(objective_weights, table.shape[-1])
    sample_weights = _normalized_sample_weight_array(samples)
    sample_values = np.sum(table * obj_weights[None, None, None, :], axis=-1)
    reduced_value = float(
        stellarator_itg_reduced_portfolio_objective(
            p,
            objective_names,
            cfg,
            samples,
            objective_weights=objective_weights,
        )
    )
    report = stellarator_itg_portfolio_sensitivity_report(
        p,
        objective_names,
        cfg,
        samples,
        objective_weights=objective_weights,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    return {
        "kind": "stellarator_itg_portfolio_gate",
        "claim_level": "reduced_multi_surface_alpha_ky_objective_gate_not_full_vmec_production",
        "source_scope": "reduced_qa_max_mode1_surrogate_rows",
        "production_nonlinear_optimization_claim": False,
        "passed": bool(report["passed"]),
        "parameter_names": list(PARAMETER_NAMES),
        "objective_names": list(objective_names),
        "initial_params": [float(value) for value in np.asarray(p)],
        "sample_set": samples.to_dict(),
        "samples": _portfolio_sample_rows(samples, sample_weights=sample_weights),
        "objective_weights": obj_weights.tolist(),
        "base_value": reduced_value,
        "base_sample_values": sample_values.ravel().tolist(),
        "base_objective_table": table.reshape((-1, table.shape[-1])).tolist(),
        "base_objective_tensor": table.tolist(),
        "portfolio_report": report["portfolio_report"],
        "config": asdict(cfg),
        "next_action": (
            "Replace the reduced surrogate row producer with real vmec_jax -> "
            "booz_xform_jax -> SPECTRAX-GK rows, then rerun the same gate "
            "with held-out surface/alpha samples before optimization claims."
        ),
    }


def stellarator_itg_density_gradient_scan(
    params: jnp.ndarray | Sequence[float],
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    density_gradients: Sequence[float] | None = None,
    temperature_gradient: float | None = None,
) -> dict[str, Any]:
    """Return a reduced ITG response scan versus normalized density gradient.

    The scan uses the same explicit drive inputs as the reduced growth,
    quasilinear, and nonlinear-envelope objectives. It is intended for
    candidate ranking and figure QA before promotion to solved VMEC/Boozer
    nonlinear SPECTRAX-GK scans.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    gradients = np.asarray(
        cfg.scan_density_gradients if density_gradients is None else density_gradients,
        dtype=float,
    )
    if gradients.ndim != 1 or gradients.size == 0 or not np.all(np.isfinite(gradients)):
        raise ValueError("density_gradients must be a finite non-empty vector")
    if np.any(gradients <= 0.0):
        raise ValueError("density_gradients must be positive")
    alti = (
        cfg.reference_temperature_gradient
        if temperature_gradient is None
        else float(temperature_gradient)
    )
    means: list[float] = []
    cvs: list[float] = []
    trends: list[float] = []
    growth_rates: list[float] = []
    quasilinear_fluxes: list[float] = []
    for aln in gradients:
        obs = qa_max_mode1_observables(
            params,
            cfg,
            density_gradient=float(aln),
            temperature_gradient=alti,
        )
        means.append(float(obs["nonlinear_heat_flux_mean"]))
        cvs.append(float(obs["nonlinear_heat_flux_cv"]))
        trends.append(float(obs["nonlinear_heat_flux_trend"]))
        growth_rates.append(float(obs["growth_rate"]))
        quasilinear_fluxes.append(float(obs["quasilinear_heat_flux"]))
    slope = float(np.polyfit(gradients, np.asarray(means, dtype=float), deg=1)[0])
    return {
        "density_gradient_axis": gradients.tolist(),
        "fixed_temperature_gradient": float(alti),
        "heat_flux_mean": means,
        "heat_flux_cv": cvs,
        "heat_flux_trend": trends,
        "growth_rate": growth_rates,
        "quasilinear_heat_flux": quasilinear_fluxes,
        "linear_slope_dQ_d_a_over_Ln": slope,
        "scope": "reduced_max_mode1_density_gradient_response_not_full_nonlinear_scan",
    }


__all__ = [
    "stellarator_itg_density_gradient_scan",
    "stellarator_itg_portfolio_gate_payload",
    "stellarator_itg_portfolio_sensitivity_report",
    "stellarator_itg_reduced_portfolio_objective",
    "stellarator_itg_sample_objective_table",
    "stellarator_itg_vmec_boozer_portfolio_objective_from_state",
    "stellarator_itg_vmec_boozer_sample_objective_table_from_state",
]
