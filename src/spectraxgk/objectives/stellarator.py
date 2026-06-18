"""Public facade and optimizer loop for reduced stellarator ITG objectives."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.backend_discovery import (
    discover_differentiable_geometry_backends,
)
from spectraxgk.objectives.stellarator_contracts import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    StellaratorITGSampleSet,
    StellaratorObjectiveKind,
)
from spectraxgk.objectives.stellarator_reduced import (
    _validate_params,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    qa_max_mode1_observables,
    qa_observable_vector,
    smooth_positive,
)
from spectraxgk.objectives.stellarator_residuals import (
    _RESIDUAL_CONDITION_NUMBER_LIMIT,
    _precision_gate_tolerances,
    stellarator_itg_objective,
    stellarator_itg_objective_residual_names,
    stellarator_itg_objective_residual_vector,
    stellarator_itg_residual_sensitivity_report,
)
from spectraxgk.objectives import stellarator_tables as _stellarator_tables
from spectraxgk.objectives.stellarator_tables import (
    stellarator_itg_density_gradient_scan,
    stellarator_itg_portfolio_gate_payload,
    stellarator_itg_portfolio_sensitivity_report,
    stellarator_itg_reduced_portfolio_objective,
    stellarator_itg_sample_objective_table,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
)
from spectraxgk.parallel import independent_map
from spectraxgk.validation.autodiff import autodiff_finite_difference_report


def _sync_table_dependencies() -> None:
    """Preserve the historical monkeypatch seam on the stellarator facade."""

    _stellarator_tables.vmec_boozer_solver_objective_table_with_metadata_from_state = (
        vmec_boozer_solver_objective_table_with_metadata_from_state
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
    """Facade wrapper for real VMEC/Boozer/SPECTRAX-GK objective rows."""

    _sync_table_dependencies()
    return _stellarator_tables.stellarator_itg_vmec_boozer_sample_objective_table_from_state(
        state,
        static,
        indata,
        wout,
        objectives,
        sample_set,
        **vmec_boozer_options,
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
    """Facade wrapper for the real VMEC/Boozer/SPECTRAX-GK portfolio scalar."""

    _sync_table_dependencies()
    return (
        _stellarator_tables.stellarator_itg_vmec_boozer_portfolio_objective_from_state(
            state,
            static,
            indata,
            wout,
            objectives,
            sample_set,
            objective_weights=objective_weights,
            **vmec_boozer_options,
        )
    )


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
    p0 = (
        default_stellarator_initial_params()
        if initial_params is None
        else _validate_params(initial_params)
    )
    p = jnp.asarray(p0)
    value_and_grad = jax.jit(
        jax.value_and_grad(lambda x: stellarator_itg_objective(x, kind, cfg))
    )
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
    covariance["residual_jacobian_gate"] = residual_sensitivity[
        "finite_difference_gate"
    ]
    covariance["residual_sensitivity_passed"] = bool(residual_sensitivity["passed"])

    nonlinear_trace = None
    if kind == "nonlinear_heat_flux":
        times0, heat0 = nonlinear_heat_flux_trace(p0, cfg)
        times1, heat1 = nonlinear_heat_flux_trace(p, cfg)
        summary0 = nonlinear_heat_flux_window_metrics(
            times0, heat0, tail_fraction=cfg.nonlinear_tail_fraction
        )
        summary1 = nonlinear_heat_flux_window_metrics(
            times1, heat1, tail_fraction=cfg.nonlinear_tail_fraction
        )
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
    kinds: Sequence[StellaratorObjectiveKind] = (
        "growth",
        "quasilinear_flux",
        "nonlinear_heat_flux",
    ),
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
    initial_tuple = (
        None
        if initial_params is None
        else tuple(float(x) for x in np.asarray(_validate_params(initial_params)))
    )
    tasks = [
        (
            kind,
            initial_tuple,
            config,
            int(finite_difference_workers),
            str(finite_difference_executor),
        )
        for kind in kind_list
    ]
    results = independent_map(
        _optimize_stellarator_itg_task,
        tasks,
        workers=workers,
        executor=parallel_executor,
    )
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
            "finite_difference_executor": str(finite_difference_executor)
            .strip()
            .lower(),
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
    "smooth_positive",
    "stellarator_itg_density_gradient_scan",
    "stellarator_itg_portfolio_gate_payload",
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
