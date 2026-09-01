"""Explicit workflow helpers for the stellarator ITG optimization examples.

The public package keeps :func:`gkx.optimize_stellarator_itg` as the
compact API.  The examples use this module instead so readers can see the same
steps used in the VMEC-JAX ``QA_optimization.py`` script: choose editable
problem constants, assemble a constrained objective, run the optimizer, audit
AD/finite-difference parity, then write diagnostic artifacts explicitly.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from gkx import (
    STELLARATOR_ITG_OBSERVABLE_NAMES,
    STELLARATOR_ITG_PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    StellaratorITGSampleSet,
    StellaratorObjectiveKind,
    autodiff_finite_difference_report,
    default_stellarator_initial_params,
    discover_differentiable_geometry_backends,
    independent_map,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    qa_observable_vector,
    stellarator_itg_objective,
    stellarator_itg_portfolio_gate_payload,
    stellarator_itg_residual_sensitivity_report,
)

from _stellarator_itg_plotting import write_portfolio_gate_artifacts

PORTFOLIO_OBJECTIVES = ("growth", "quasilinear_flux")


def precision_gate_tolerances(fd_step: float) -> tuple[float, float, float]:
    """Return FD tolerances matching the release optimization API."""

    if bool(getattr(jax.config, "jax_enable_x64", False)):
        return float(fd_step), 5.0e-3, 6.0e-4
    return max(float(fd_step), 5.0e-3), 5.0e-2, 6.0e-3


def run_stellarator_itg_adam(
    objective_kind: StellaratorObjectiveKind,
    *,
    config: StellaratorITGOptimizationConfig,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
    progress: bool = True,
) -> StellaratorITGOptimizationResult:
    """Run the explicit constrained Adam workflow used by the example scripts."""

    backend_info = discover_differentiable_geometry_backends()
    p0 = (
        default_stellarator_initial_params()
        if initial_params is None
        else jnp.asarray(initial_params)
    )
    p = jnp.asarray(p0)
    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda x: stellarator_itg_objective(x, objective_kind, config)
        )
    )
    obs_fn = jax.jit(lambda x: qa_observable_vector(x, config))

    beta1 = jnp.asarray(0.9, dtype=p.dtype)
    beta2 = jnp.asarray(0.99, dtype=p.dtype)
    eps = jnp.asarray(1.0e-8, dtype=p.dtype)
    learning_rate = jnp.asarray(config.learning_rate, dtype=p.dtype)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    history: list[dict[str, Any]] = []
    initial_value = float(stellarator_itg_objective(p, objective_kind, config))
    report_stride = max(1, int(config.steps) // 10)

    if progress:
        print("\nAssembled GKX reduced stellarator ITG objective:")
        print(f"  objective:        {objective_kind}")
        print(f"  target aspect:    {config.target_aspect:.6g}")
        print(f"  target iota:      {config.target_iota:.6g}")
        print(f"  Adam steps/lr:    {int(config.steps)} / {config.learning_rate:.3e}")
        print(f"  initial ||r||^2:  {initial_value:.6e}")

    for step in range(int(config.steps) + 1):
        value, grad = value_and_grad(p)
        obs = obs_fn(p)
        history.append(
            {
                "step": int(step),
                "objective": float(value),
                "params": np.asarray(p, dtype=float).tolist(),
                "observables": np.asarray(obs, dtype=float).tolist(),
                "gradient_norm": float(jnp.linalg.norm(grad)),
            }
        )
        if progress and (
            step == 0 or step == int(config.steps) or step % report_stride == 0
        ):
            print(
                f"  step {step:4d}: objective={float(value):.6e}, "
                f"|grad|={float(jnp.linalg.norm(grad)):.3e}"
            )
        if step == int(config.steps):
            break
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** (step + 1))
        v_hat = v / (1.0 - beta2 ** (step + 1))
        p = p - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        p = jnp.clip(p, -0.8, 0.8)

    final_value = float(stellarator_itg_objective(p, objective_kind, config))
    initial_obs = qa_observable_vector(p0, config)
    final_obs = qa_observable_vector(p, config)
    fd_step, rtol, atol = precision_gate_tolerances(config.fd_step)
    gradient_gate = autodiff_finite_difference_report(
        lambda x: stellarator_itg_objective(x, objective_kind, config),
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
        parallel_executor=finite_difference_executor,
    )
    residual_sensitivity = stellarator_itg_residual_sensitivity_report(
        p,
        objective_kind,
        config,
        finite_difference_workers=finite_difference_workers,
        finite_difference_executor=finite_difference_executor,
    )
    covariance = dict(residual_sensitivity["covariance"])
    covariance["residual_jacobian_gate"] = residual_sensitivity[
        "finite_difference_gate"
    ]
    covariance["residual_sensitivity_passed"] = bool(residual_sensitivity["passed"])

    nonlinear_trace = None
    if objective_kind == "nonlinear_heat_flux":
        times0, heat0 = nonlinear_heat_flux_trace(p0, config)
        times1, heat1 = nonlinear_heat_flux_trace(p, config)
        summary0 = nonlinear_heat_flux_window_metrics(
            times0, heat0, tail_fraction=config.nonlinear_tail_fraction
        )
        summary1 = nonlinear_heat_flux_window_metrics(
            times1, heat1, tail_fraction=config.nonlinear_tail_fraction
        )
        nonlinear_trace = {
            "times": np.asarray(times1, dtype=float).tolist(),
            "initial_heat_flux": np.asarray(heat0, dtype=float).tolist(),
            "final_heat_flux": np.asarray(heat1, dtype=float).tolist(),
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

    if progress:
        print("\nFinal diagnostics:")
        print(f"  objective:        {initial_value:.6e} -> {final_value:.6e}")
        print(f"  scalar AD/FD:     {gradient_gate['passed']}")
        print(f"  residual AD/FD:   {residual_sensitivity['passed']}")

    return StellaratorITGOptimizationResult(
        objective_kind=objective_kind,
        parameter_names=STELLARATOR_ITG_PARAMETER_NAMES,
        observable_names=STELLARATOR_ITG_OBSERVABLE_NAMES,
        initial_params=tuple(float(x) for x in np.asarray(p0, dtype=float)),
        final_params=tuple(float(x) for x in np.asarray(p, dtype=float)),
        initial_objective=initial_value,
        final_objective=final_value,
        initial_observables=tuple(
            float(x) for x in np.asarray(initial_obs, dtype=float)
        ),
        final_observables=tuple(float(x) for x in np.asarray(final_obs, dtype=float)),
        history=tuple(history),
        gradient_gate=gradient_gate,
        covariance=covariance,
        nonlinear_trace=nonlinear_trace,
        config=asdict(config),
        backend_info=backend_info,
    )


def write_optional_portfolio_artifacts(
    *,
    result: StellaratorITGOptimizationResult,
    out_base: Path,
    portfolio: bool = False,
    surfaces: tuple[float, ...] | None = None,
    alphas: tuple[float, ...] | None = None,
    ky_values: tuple[float, ...] | None = None,
    objective_weights: tuple[float, ...] | None = None,
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> Path | None:
    """Write the optional reduced multi-surface/field-line gate for linear/QL objectives."""

    if not portfolio:
        return None
    if objective_weights is not None and len(objective_weights) != len(
        PORTFOLIO_OBJECTIVES
    ):
        raise ValueError(
            "objective_weights must provide two values: growth,quasilinear_flux"
        )
    defaults = StellaratorITGSampleSet()
    sample_set = StellaratorITGSampleSet(
        surfaces=defaults.surfaces if surfaces is None else surfaces,
        alphas=defaults.alphas if alphas is None else alphas,
        ky_values=defaults.ky_values if ky_values is None else ky_values,
    )
    params = np.asarray(result.final_params, dtype=float)
    cfg = StellaratorITGOptimizationConfig(**result.config)
    payload = stellarator_itg_portfolio_gate_payload(
        params,
        PORTFOLIO_OBJECTIVES,
        cfg,
        sample_set,
        objective_weights=objective_weights,
        finite_difference_workers=finite_difference_workers,
        finite_difference_executor=finite_difference_executor,
    )
    payload["optimization_objective_kind"] = result.objective_kind
    payload["optimized_params"] = params.tolist()
    payload["optimization_initial_params"] = [
        float(value) for value in np.asarray(result.initial_params, dtype=float)
    ]
    out = out_base.with_name(f"{out_base.name}_portfolio_gate")
    write_portfolio_gate_artifacts(payload, out)
    return out


def _run_compare_task(
    task: tuple[str, dict[str, Any]],
) -> StellaratorITGOptimizationResult:
    kind, kwargs = task
    return run_stellarator_itg_adam(kind, **kwargs)


def compare_scripted_stellarator_itg_objectives(
    kinds: Sequence[StellaratorObjectiveKind],
    *,
    config: StellaratorITGOptimizationConfig,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    workers: int = 1,
    parallel_executor: str = "thread",
    finite_difference_workers: int = 1,
    finite_difference_executor: str = "thread",
) -> dict[str, Any]:
    """Run ordered independent objective examples without using the compact package optimizer."""

    initial_tuple = (
        None
        if initial_params is None
        else tuple(float(x) for x in np.asarray(initial_params, dtype=float))
    )
    tasks = [
        (
            str(kind),
            {
                "config": config.with_kind_defaults(kind),
                "initial_params": initial_tuple,
                "finite_difference_workers": int(finite_difference_workers),
                "finite_difference_executor": str(finite_difference_executor),
                "progress": False,
            },
        )
        for kind in kinds
    ]
    results = independent_map(
        _run_compare_task, tasks, workers=workers, executor=parallel_executor
    )
    return {
        "claim_level": "reduced_objective_optimization_comparison_not_full_production_vmec_gk",
        "production_nonlinear_optimization_claim": False,
        "baseline_scope": (
            "reduced max-mode-1 QA controls; solved-boundary baseline should be the final WOUT "
            "from vmex/examples/optimization/QA_optimization.py or the constraints-only VMEX driver"
        ),
        "parameter_names": list(STELLARATOR_ITG_PARAMETER_NAMES),
        "observable_names": list(STELLARATOR_ITG_OBSERVABLE_NAMES),
        "results": [result.to_dict() for result in results],
        "backend_info": discover_differentiable_geometry_backends(),
        "parallel": {
            "requested_workers": int(workers),
            "effective_workers": int(min(max(int(workers), 1), max(len(tasks), 1))),
            "executor": str(parallel_executor).strip().lower(),
            "finite_difference_workers": int(finite_difference_workers),
            "finite_difference_executor": str(finite_difference_executor)
            .strip()
            .lower(),
            "identity_contract": "parallel objective reports must preserve serial ordering and values",
        },
    }
