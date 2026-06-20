"""Public facade and optimizer loop for reduced stellarator ITG objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class _AdamState:
    """State carried by the reduced Adam optimizer."""

    params: jnp.ndarray
    first_moment: jnp.ndarray
    second_moment: jnp.ndarray


@dataclass(frozen=True)
class _StellaratorAdamRun:
    """Completed reduced stellarator optimization trace."""

    initial_params: jnp.ndarray
    final_params: jnp.ndarray
    initial_objective: float
    history: list[dict[str, Any]]


def _sync_table_dependencies() -> None:
    """Sync patchable table dependencies on the stellarator facade."""

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


def _initial_stellarator_params(
    initial_params: jnp.ndarray | Sequence[float] | None,
) -> jnp.ndarray:
    """Return validated reduced stellarator optimization parameters."""

    p0 = (
        default_stellarator_initial_params()
        if initial_params is None
        else _validate_params(initial_params)
    )
    return jnp.asarray(p0)


def _adam_update(
    state: _AdamState,
    grad: jnp.ndarray,
    *,
    step: int,
    learning_rate: jnp.ndarray,
) -> _AdamState:
    """Apply the clipped Adam update used by reduced stellarator examples."""

    beta1 = jnp.asarray(0.9, dtype=state.params.dtype)
    beta2 = jnp.asarray(0.99, dtype=state.params.dtype)
    eps = jnp.asarray(1.0e-8, dtype=state.params.dtype)
    m = beta1 * state.first_moment + (1.0 - beta1) * grad
    v = beta2 * state.second_moment + (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - beta1 ** (step + 1))
    v_hat = v / (1.0 - beta2 ** (step + 1))
    params = state.params - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
    return _AdamState(
        params=jnp.clip(params, -0.8, 0.8),
        first_moment=m,
        second_moment=v,
    )


def _run_stellarator_adam(
    kind: StellaratorObjectiveKind,
    cfg: StellaratorITGOptimizationConfig,
    initial_params: jnp.ndarray,
) -> _StellaratorAdamRun:
    """Run the reduced differentiable Adam loop and collect JSON-ready history."""

    value_and_grad = jax.jit(
        jax.value_and_grad(lambda x: stellarator_itg_objective(x, kind, cfg))
    )
    obs_fn = jax.jit(lambda x: qa_observable_vector(x, cfg))
    state = _AdamState(
        params=jnp.asarray(initial_params),
        first_moment=jnp.zeros_like(initial_params),
        second_moment=jnp.zeros_like(initial_params),
    )
    learning_rate = jnp.asarray(cfg.learning_rate, dtype=state.params.dtype)
    history: list[dict[str, Any]] = []
    initial_value = float(stellarator_itg_objective(state.params, kind, cfg))

    for step in range(int(cfg.steps) + 1):
        value, grad = value_and_grad(state.params)
        obs = obs_fn(state.params)
        history.append(
            {
                "step": int(step),
                "objective": float(value),
                "params": np.asarray(state.params).tolist(),
                "observables": np.asarray(obs).tolist(),
                "gradient_norm": float(jnp.linalg.norm(grad)),
            }
        )
        if step == int(cfg.steps):
            break
        state = _adam_update(
            state,
            grad,
            step=step,
            learning_rate=learning_rate,
        )

    return _StellaratorAdamRun(
        initial_params=jnp.asarray(initial_params),
        final_params=state.params,
        initial_objective=initial_value,
        history=history,
    )


def _stellarator_gradient_gate(
    p: jnp.ndarray,
    kind: StellaratorObjectiveKind,
    cfg: StellaratorITGOptimizationConfig,
    *,
    workers: int,
    executor: str,
) -> dict[str, Any]:
    """Run the AD/finite-difference gate for the optimized scalar objective."""

    fd_step, rtol, atol = _precision_gate_tolerances(cfg.fd_step)
    return autodiff_finite_difference_report(
        lambda x: stellarator_itg_objective(x, kind, cfg),
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=executor,
    )


def _stellarator_covariance_report(
    p: jnp.ndarray,
    kind: StellaratorObjectiveKind,
    cfg: StellaratorITGOptimizationConfig,
    *,
    workers: int,
    executor: str,
) -> dict[str, Any]:
    """Build residual sensitivity and local covariance metadata."""

    residual_sensitivity = stellarator_itg_residual_sensitivity_report(
        p,
        kind,
        cfg,
        min_rank=len(PARAMETER_NAMES),
        condition_number_limit=_RESIDUAL_CONDITION_NUMBER_LIMIT,
        covariance_regularization=1.0e-8,
        finite_difference_workers=workers,
        finite_difference_executor=executor,
    )
    covariance = dict(residual_sensitivity["covariance"])
    covariance["residual_jacobian_gate"] = residual_sensitivity[
        "finite_difference_gate"
    ]
    covariance["residual_sensitivity_passed"] = bool(residual_sensitivity["passed"])
    return covariance


def _stellarator_nonlinear_trace_report(
    kind: StellaratorObjectiveKind,
    p0: jnp.ndarray,
    p: jnp.ndarray,
    cfg: StellaratorITGOptimizationConfig,
) -> dict[str, Any] | None:
    """Build nonlinear-window trace metadata for the reduced nonlinear objective."""

    if kind != "nonlinear_heat_flux":
        return None
    times0, heat0 = nonlinear_heat_flux_trace(p0, cfg)
    times1, heat1 = nonlinear_heat_flux_trace(p, cfg)
    summary0 = nonlinear_heat_flux_window_metrics(
        times0, heat0, tail_fraction=cfg.nonlinear_tail_fraction
    )
    summary1 = nonlinear_heat_flux_window_metrics(
        times1, heat1, tail_fraction=cfg.nonlinear_tail_fraction
    )
    return {
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
    p0 = _initial_stellarator_params(initial_params)
    run = _run_stellarator_adam(kind, cfg, p0)

    final_value = float(stellarator_itg_objective(run.final_params, kind, cfg))
    initial_obs = qa_observable_vector(run.initial_params, cfg)
    final_obs = qa_observable_vector(run.final_params, cfg)
    gradient_gate = _stellarator_gradient_gate(
        run.final_params,
        kind,
        cfg,
        workers=finite_difference_workers,
        executor=finite_difference_executor,
    )
    covariance = _stellarator_covariance_report(
        run.final_params,
        kind,
        cfg,
        workers=finite_difference_workers,
        executor=finite_difference_executor,
    )
    nonlinear_trace = _stellarator_nonlinear_trace_report(
        kind,
        run.initial_params,
        run.final_params,
        cfg,
    )

    return StellaratorITGOptimizationResult(
        objective_kind=kind,
        parameter_names=PARAMETER_NAMES,
        observable_names=OBSERVABLE_NAMES,
        initial_params=tuple(float(x) for x in np.asarray(run.initial_params)),
        final_params=tuple(float(x) for x in np.asarray(run.final_params)),
        initial_objective=run.initial_objective,
        final_objective=final_value,
        initial_observables=tuple(float(x) for x in np.asarray(initial_obs)),
        final_observables=tuple(float(x) for x in np.asarray(final_obs)),
        history=tuple(run.history),
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
