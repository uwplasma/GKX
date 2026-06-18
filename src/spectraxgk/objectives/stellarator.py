"""Differentiable stellarator ITG objective-reduction utilities.

The functions here provide a small, fully JAX-differentiable optimization
contract for QA, max-mode-1 stellarator studies. They are intentionally
separate from the production runtime drivers: this module is the gradient,
conditioning, and UQ gate used before promoting a full VMEC/Boozer/nonlinear
gyrokinetic loop into a release claim.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.validation.autodiff import autodiff_finite_difference_report, covariance_diagnostics
from spectraxgk.geometry.backend_discovery import discover_differentiable_geometry_backends
from spectraxgk.parallel import independent_map
from spectraxgk.objectives.core import (
    SolverScalarObjective,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
)
from spectraxgk.objectives.portfolio_contracts import aggregate_objective_portfolio
from spectraxgk.objectives.portfolio_sensitivity import objective_portfolio_sensitivity_report
from spectraxgk.objectives.stellarator_contracts import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    StellaratorITGSampleSet,
    StellaratorObjectiveKind,
)
from spectraxgk.objectives.stellarator_reduced import (
    _qa_core_features,
    _sampled_qa_itg_fields,
    _validate_params,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    qa_max_mode1_observables,
    qa_observable_vector,
    smooth_positive,
)


_RESIDUAL_CONDITION_NUMBER_LIMIT = 1.0e4



def _precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return FD tolerances that are strict in x64 and stable in float32."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return max(float(fd_step), 5.0e-3), 5.0e-2, 6.0e-3


def _residual_precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return residual-Jacobian FD tolerances that stay local near zero residuals."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return min(float(fd_step), 1.0e-4), 5.0e-2, 6.0e-3


def _conditioning_gate_from_covariance(
    covariance: dict[str, Any],
    *,
    min_rank: int,
    condition_number_limit: float,
) -> dict[str, Any]:
    """Return a pass/fail gate for Gauss-Newton residual conditioning."""

    singular = np.asarray(covariance.get("jacobian_singular_values", ()), dtype=float)
    rank = int(covariance.get("sensitivity_map_rank", 0))
    condition_number = float(covariance.get("jacobian_condition_number", float("inf")))
    finite_singular_values = bool(singular.size > 0 and np.all(np.isfinite(singular)))
    finite_condition = bool(np.isfinite(condition_number))
    smallest = float(singular[-1]) if finite_singular_values else 0.0
    limit = float(condition_number_limit)
    if int(min_rank) < 1:
        raise ValueError("min_rank must be >= 1")
    if limit <= 0.0:
        raise ValueError("condition_number_limit must be positive")
    passed = bool(
        finite_singular_values
        and finite_condition
        and rank >= int(min_rank)
        and condition_number <= limit
        and smallest > 0.0
    )
    return {
        "passed": passed,
        "finite_singular_values": finite_singular_values,
        "finite_condition_number": finite_condition,
        "sensitivity_map_rank": rank,
        "min_rank": int(min_rank),
        "rank_deficiency": int(max(int(min_rank) - rank, 0)),
        "jacobian_condition_number": condition_number,
        "condition_number_limit": limit,
        "smallest_singular_value": smallest,
    }












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
        jnp.asarray([
            solver_scalar_objective_from_vector(row, cast(SolverScalarObjective, objective))
            for row in flat_table
        ])
        for objective in objective_names
    ]
    flat_objectives = jnp.stack(columns, axis=-1)
    return jnp.reshape(
        flat_objectives,
        (len(samples.surfaces), len(samples.alphas), len(samples.ky_values), len(objective_names)),
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
        lambda x: stellarator_itg_sample_objective_table(x, objective_names, cfg, samples),
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
    surface = _normalized_axis_weights(sample_set.surface_weights, len(sample_set.surfaces))
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
        raise ValueError(f"objective_weights must be a finite length-{int(size)} vector")
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
    p = default_stellarator_initial_params() if params is None else _validate_params(params)
    objective_names = tuple(str(objective).strip().lower() for objective in objectives)
    table = np.asarray(stellarator_itg_sample_objective_table(p, objective_names, cfg, samples), dtype=float)
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
    alti = cfg.reference_temperature_gradient if temperature_gradient is None else float(temperature_gradient)
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


def stellarator_itg_objective(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return the scalar constrained QA + ITG objective for one optimization."""

    residual = stellarator_itg_objective_residual_vector(params, kind, config)
    return jnp.dot(residual, residual)


def stellarator_itg_objective_residual_names(kind: StellaratorObjectiveKind) -> tuple[str, ...]:
    """Return stable residual names for the weighted QA + ITG objective."""

    if kind not in ("growth", "quasilinear_flux", "nonlinear_heat_flux"):
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    return (
        "aspect_constraint",
        "iota_constraint",
        "qa_constraint",
        *(f"regularization_{name}" for name in PARAMETER_NAMES),
        f"{kind}_transport_objective",
    )


def stellarator_itg_objective_residual_vector(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
) -> jnp.ndarray:
    """Return weighted residuals whose squared norm is the optimization objective.

    This is the correct local residual map for Gauss-Newton covariance and
    identifiability diagnostics. Using the initial-to-final observable
    displacement would overstate uncertainty because it measures optimizer
    travel rather than the residual left at the optimized point.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    obs = _qa_core_features(p, cfg)
    dtype = p.dtype
    aspect_res = jnp.sqrt(jnp.asarray(cfg.aspect_weight, dtype=dtype)) * (
        (obs["aspect"] - cfg.target_aspect) / cfg.target_aspect
    )
    iota_res = jnp.sqrt(jnp.asarray(cfg.iota_weight, dtype=dtype)) * (obs["mean_iota"] - cfg.target_iota)
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    if kind == "growth":
        turbulence = obs["growth_rate"]
    elif kind == "quasilinear_flux":
        turbulence = obs["quasilinear_heat_flux"]
    elif kind == "nonlinear_heat_flux":
        times, heat_flux = nonlinear_heat_flux_trace(p, cfg)
        turbulence = nonlinear_heat_flux_window_metrics(
            times,
            heat_flux,
            tail_fraction=cfg.nonlinear_tail_fraction,
        )["mean"]
    else:
        raise ValueError(f"unknown stellarator objective kind {kind!r}")
    turbulence_res = jnp.sqrt(
        jnp.maximum(
            jnp.asarray(cfg.turbulence_weight, dtype=dtype) * turbulence,
            jnp.asarray(0.0, dtype=dtype),
        )
    )
    return jnp.concatenate(
        [
            jnp.asarray([aspect_res, iota_res, qa_res], dtype=dtype),
            reg_res,
            jnp.asarray([turbulence_res], dtype=dtype),
        ]
    )


def stellarator_itg_residual_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    kind: StellaratorObjectiveKind,
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    step: float | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    min_rank: int = len(PARAMETER_NAMES),
    condition_number_limit: float = _RESIDUAL_CONDITION_NUMBER_LIMIT,
    covariance_regularization: float = 1.0e-8,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Check residual-Jacobian AD/FD parity and local conditioning.

    The scalar objective gradient can pass even when the residual sensitivity
    map is rank-deficient. This gate validates the full weighted residual map
    used by Gauss-Newton covariance and UQ diagnostics.
    """

    cfg = config or StellaratorITGOptimizationConfig()
    p = _validate_params(params)
    default_step, default_rtol, default_atol = _residual_precision_gate_tolerances(cfg.fd_step)
    fd_step = default_step if step is None else float(step)
    fd_rtol = default_rtol if rtol is None else float(rtol)
    fd_atol = default_atol if atol is None else float(atol)

    def residual_fn(x: jnp.ndarray) -> jnp.ndarray:
        return stellarator_itg_objective_residual_vector(x, kind, cfg)

    residual_gate = autodiff_finite_difference_report(
        residual_fn,
        p,
        step=fd_step,
        rtol=fd_rtol,
        atol=fd_atol,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    jac = np.asarray(residual_gate["jacobian_ad"], dtype=float)
    residual = np.asarray(residual_fn(p), dtype=float)
    covariance = covariance_diagnostics(jac, residual, regularization=covariance_regularization)
    covariance["source"] = "weighted_objective_residual"
    covariance["residual_names"] = list(stellarator_itg_objective_residual_names(kind))
    conditioning_gate = _conditioning_gate_from_covariance(
        covariance,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
    )
    covariance["conditioning_gate"] = conditioning_gate
    return {
        "kind": "stellarator_itg_residual_sensitivity_report",
        "objective_kind": kind,
        "passed": bool(residual_gate["passed"] and conditioning_gate["passed"]),
        "parameter_names": list(PARAMETER_NAMES),
        "residual_names": list(stellarator_itg_objective_residual_names(kind)),
        "finite_difference_gate": residual_gate,
        "conditioning_gate": conditioning_gate,
        "covariance": covariance,
    }


def optimize_stellarator_itg(
    kind: StellaratorObjectiveKind,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    config: StellaratorITGOptimizationConfig | None = None,
    *,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> StellaratorITGOptimizationResult:
    """Optimize one differentiable stellarator ITG objective with Adam."""

    backend_info = discover_differentiable_geometry_backends()
    base_cfg = config or StellaratorITGOptimizationConfig()
    cfg = base_cfg.with_kind_defaults(kind)
    p0 = default_stellarator_initial_params() if initial_params is None else _validate_params(initial_params)
    p = jnp.asarray(p0)
    value_and_grad = jax.jit(jax.value_and_grad(lambda x: stellarator_itg_objective(x, kind, cfg)))
    obs_fn = jax.jit(lambda x: qa_observable_vector(x, cfg))

    beta1 = jnp.asarray(0.9, dtype=p.dtype)
    beta2 = jnp.asarray(0.99, dtype=p.dtype)
    eps = jnp.asarray(1.0e-8, dtype=p.dtype)
    lr = jnp.asarray(cfg.learning_rate, dtype=p.dtype)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    history: list[dict[str, Any]] = []
    initial_value = float(stellarator_itg_objective(p, kind, cfg))

    for step in range(int(cfg.steps) + 1):
        value, grad = value_and_grad(p)
        obs = obs_fn(p)
        history.append(
            {
                "step": int(step),
                "objective": float(value),
                "params": np.asarray(p).tolist(),
                "observables": np.asarray(obs).tolist(),
                "gradient_norm": float(jnp.linalg.norm(grad)),
            }
        )
        if step == int(cfg.steps):
            break
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** (step + 1))
        v_hat = v / (1.0 - beta2 ** (step + 1))
        p = p - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        p = jnp.clip(p, -0.8, 0.8)

    final_value = float(stellarator_itg_objective(p, kind, cfg))
    initial_obs = qa_observable_vector(p0, cfg)
    final_obs = qa_observable_vector(p, cfg)
    fd_step, rtol, atol = _precision_gate_tolerances(cfg.fd_step)
    gradient_gate = autodiff_finite_difference_report(
        lambda x: stellarator_itg_objective(x, kind, cfg),
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    residual_sensitivity = stellarator_itg_residual_sensitivity_report(
        p,
        kind,
        cfg,
        min_rank=len(PARAMETER_NAMES),
        condition_number_limit=_RESIDUAL_CONDITION_NUMBER_LIMIT,
        covariance_regularization=1.0e-8,
        finite_difference_workers=finite_difference_workers,
        finite_difference_executor=finite_difference_executor,
    )
    covariance = dict(residual_sensitivity["covariance"])
    covariance["residual_jacobian_gate"] = residual_sensitivity["finite_difference_gate"]
    covariance["residual_sensitivity_passed"] = bool(residual_sensitivity["passed"])

    nonlinear_trace = None
    if kind == "nonlinear_heat_flux":
        times0, heat0 = nonlinear_heat_flux_trace(p0, cfg)
        times1, heat1 = nonlinear_heat_flux_trace(p, cfg)
        summary0 = nonlinear_heat_flux_window_metrics(times0, heat0, tail_fraction=cfg.nonlinear_tail_fraction)
        summary1 = nonlinear_heat_flux_window_metrics(times1, heat1, tail_fraction=cfg.nonlinear_tail_fraction)
        nonlinear_trace = {
            "times": np.asarray(times1).tolist(),
            "initial_heat_flux": np.asarray(heat0).tolist(),
            "final_heat_flux": np.asarray(heat1).tolist(),
            "initial_window": {
                "mean": float(summary0["mean"]),
                "cv": float(summary0["cv"]),
                "trend": float(summary0["trend"]),
                "start_index": int(summary0["start_index"]),
            },
            "final_window": {
                "mean": float(summary1["mean"]),
                "cv": float(summary1["cv"]),
                "trend": float(summary1["trend"]),
                "start_index": int(summary1["start_index"]),
            },
        }

    return StellaratorITGOptimizationResult(
        objective_kind=kind,
        parameter_names=PARAMETER_NAMES,
        observable_names=OBSERVABLE_NAMES,
        initial_params=tuple(float(x) for x in np.asarray(p0)),
        final_params=tuple(float(x) for x in np.asarray(p)),
        initial_objective=initial_value,
        final_objective=final_value,
        initial_observables=tuple(float(x) for x in np.asarray(initial_obs)),
        final_observables=tuple(float(x) for x in np.asarray(final_obs)),
        history=tuple(history),
        gradient_gate=gradient_gate,
        covariance=covariance,
        nonlinear_trace=nonlinear_trace,
        config=asdict(cfg),
        backend_info=backend_info,
    )


def _optimize_stellarator_itg_task(
    task: tuple[
        StellaratorObjectiveKind,
        tuple[float, ...] | None,
        StellaratorITGOptimizationConfig | None,
        int,
        str,
    ],
) -> StellaratorITGOptimizationResult:
    """Run one optimization task for ordered independent objective comparisons."""

    kind, initial_params, config, fd_workers, fd_executor = task
    initial = None if initial_params is None else jnp.asarray(initial_params)
    return optimize_stellarator_itg(
        kind,
        initial_params=initial,
        config=config,
        finite_difference_workers=fd_workers,
        finite_difference_executor=fd_executor,
    )


def compare_stellarator_itg_objectives(
    kinds: Sequence[StellaratorObjectiveKind] = ("growth", "quasilinear_flux", "nonlinear_heat_flux"),
    *,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    config: StellaratorITGOptimizationConfig | None = None,
    workers: int = 1,
    parallel_executor: str = "thread",
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Run the three objective reductions from a shared starting point."""

    kind_list = list(kinds)
    initial_tuple = None if initial_params is None else tuple(float(x) for x in np.asarray(_validate_params(initial_params)))
    tasks = [
        (kind, initial_tuple, config, int(finite_difference_workers), str(finite_difference_executor))
        for kind in kind_list
    ]
    results = independent_map(_optimize_stellarator_itg_task, tasks, workers=workers, executor=parallel_executor)
    return {
        "claim_level": "reduced_objective_optimization_comparison_not_full_production_vmec_gk",
        "production_nonlinear_optimization_claim": False,
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": list(OBSERVABLE_NAMES),
        "results": [result.to_dict() for result in results],
        "backend_info": discover_differentiable_geometry_backends(),
        "parallel": {
            "requested_workers": int(workers),
            "effective_workers": int(min(max(int(workers), 1), max(len(kind_list), 1))),
            "executor": str(parallel_executor).strip().lower(),
            "finite_difference_workers": int(finite_difference_workers),
            "finite_difference_executor": str(finite_difference_executor).strip().lower(),
            "identity_contract": "parallel objective reports must preserve serial ordering and values",
        },
    }


__all__ = [
    "OBSERVABLE_NAMES",
    "PARAMETER_NAMES",
    "StellaratorITGOptimizationConfig",
    "StellaratorITGOptimizationResult",
    "StellaratorITGSampleSet",
    "StellaratorObjectiveKind",
    "compare_stellarator_itg_objectives",
    "default_stellarator_initial_params",
    "nonlinear_heat_flux_trace",
    "nonlinear_heat_flux_window_metrics",
    "optimize_stellarator_itg",
    "qa_max_mode1_observables",
    "qa_observable_vector",
    "stellarator_itg_density_gradient_scan",
    "stellarator_itg_portfolio_gate_payload",
    "smooth_positive",
    "stellarator_itg_portfolio_sensitivity_report",
    "stellarator_itg_residual_sensitivity_report",
    "stellarator_itg_objective",
    "stellarator_itg_objective_residual_names",
    "stellarator_itg_objective_residual_vector",
    "stellarator_itg_reduced_portfolio_objective",
    "stellarator_itg_sample_objective_table",
    "stellarator_itg_vmec_boozer_portfolio_objective_from_state",
    "stellarator_itg_vmec_boozer_sample_objective_table_from_state",
]
