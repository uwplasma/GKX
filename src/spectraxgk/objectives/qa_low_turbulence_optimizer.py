"""Optimizer loop for reduced QA low-turbulence objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.qa_low_turbulence_model import (
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    QALowTurbulenceResult,
    _sensitivity_reports,
    default_qa_low_turbulence_initial_params,
    qa_low_turbulence_objective,
    qa_low_turbulence_observable_vector,
)
from spectraxgk.objectives.stellarator import PARAMETER_NAMES, _validate_params


@dataclass(frozen=True)
class _AdamState:
    """Adam optimizer state for the reduced QA objective."""

    params: jnp.ndarray
    first_moment: jnp.ndarray
    second_moment: jnp.ndarray


@dataclass(frozen=True)
class _QAOptimizationTrace:
    """Reduced QA optimizer trace and final differentiable state."""

    initial_objective: jnp.ndarray
    final_objective: jnp.ndarray
    final_params: jnp.ndarray
    history: tuple[dict[str, Any], ...]


def _history_row(
    step: int,
    params: jnp.ndarray,
    objective: jnp.ndarray,
    grad: jnp.ndarray,
    config: QALowTurbulenceConfig,
) -> dict[str, Any]:
    obs = np.asarray(qa_low_turbulence_observable_vector(params, config), dtype=float)
    return {
        "step": int(step),
        "objective": float(objective),
        "gradient_norm": float(jnp.linalg.norm(grad)),
        "params": [float(x) for x in np.asarray(params, dtype=float)],
        "observables": [float(x) for x in obs],
    }


def _qa_low_turbulence_grad_fn(
    cfg: QALowTurbulenceConfig,
    *,
    includes_nonlinear_heat_flux: bool,
) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    return jax.value_and_grad(
        lambda x: qa_low_turbulence_objective(
            x,
            cfg,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        )
    )


def _adam_update(
    state: _AdamState,
    grad: jnp.ndarray,
    *,
    step: int,
    learning_rate: float,
) -> _AdamState:
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    first_moment = beta1 * state.first_moment + (1.0 - beta1) * grad
    second_moment = beta2 * state.second_moment + (1.0 - beta2) * (grad * grad)
    first_hat = first_moment / (1.0 - beta1**step)
    second_hat = second_moment / (1.0 - beta2**step)
    params = state.params - learning_rate * first_hat / (jnp.sqrt(second_hat) + eps)
    return _AdamState(
        params=params,
        first_moment=first_moment,
        second_moment=second_moment,
    )


def _run_qa_low_turbulence_adam(
    initial_params: jnp.ndarray,
    cfg: QALowTurbulenceConfig,
    *,
    includes_nonlinear_heat_flux: bool,
) -> _QAOptimizationTrace:
    grad_fn = _qa_low_turbulence_grad_fn(
        cfg,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
    )
    state = _AdamState(
        params=jnp.asarray(initial_params),
        first_moment=jnp.zeros_like(initial_params),
        second_moment=jnp.zeros_like(initial_params),
    )
    history: list[dict[str, Any]] = []
    objective0, grad0 = grad_fn(state.params)
    history.append(_history_row(0, state.params, objective0, grad0, cfg))

    for step in range(1, int(cfg.steps) + 1):
        _, grad = grad_fn(state.params)
        state = _adam_update(
            state,
            grad,
            step=step,
            learning_rate=float(cfg.learning_rate),
        )
        if step % 5 == 0 or step == int(cfg.steps):
            objective_i, grad_i = grad_fn(state.params)
            history.append(_history_row(step, state.params, objective_i, grad_i, cfg))

    final_objective, final_grad = grad_fn(state.params)
    if history[-1]["step"] != int(cfg.steps):
        history.append(
            _history_row(int(cfg.steps), state.params, final_objective, final_grad, cfg)
        )
    return _QAOptimizationTrace(
        initial_objective=objective0,
        final_objective=final_objective,
        final_params=state.params,
        history=tuple(history),
    )


def _qa_low_turbulence_result(
    *,
    cfg: QALowTurbulenceConfig,
    initial_params: jnp.ndarray,
    trace: _QAOptimizationTrace,
    includes_nonlinear_heat_flux: bool,
    scalar_gate: dict[str, Any],
    residual_gate: dict[str, Any],
    observable_gate: dict[str, Any],
    covariance: dict[str, Any],
) -> QALowTurbulenceResult:
    design_name = (
        "qa_plus_nonlinear_heat_flux"
        if includes_nonlinear_heat_flux
        else "qa_constraints"
    )
    return QALowTurbulenceResult(
        design_name=design_name,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        parameter_names=tuple(PARAMETER_NAMES),
        observable_names=tuple(QA_LOW_TURBULENCE_OBSERVABLE_NAMES),
        initial_params=tuple(float(x) for x in np.asarray(initial_params, dtype=float)),
        final_params=tuple(float(x) for x in np.asarray(trace.final_params, dtype=float)),
        initial_objective=float(trace.initial_objective),
        final_objective=float(trace.final_objective),
        initial_observables=tuple(
            float(x)
            for x in np.asarray(
                qa_low_turbulence_observable_vector(initial_params, cfg),
                dtype=float,
            )
        ),
        final_observables=tuple(
            float(x)
            for x in np.asarray(
                qa_low_turbulence_observable_vector(trace.final_params, cfg),
            )
        ),
        history=trace.history,
        residual_gradient_gate=residual_gate,
        scalar_gradient_gate=scalar_gate,
        observable_gradient_gate=observable_gate,
        covariance=covariance,
        config=asdict(cfg),
    )


def optimize_qa_low_turbulence(
    *,
    includes_nonlinear_heat_flux: bool,
    config: QALowTurbulenceConfig | None = None,
    initial_params: jnp.ndarray | Sequence[float] | None = None,
    finite_difference_workers: int = 1,
) -> QALowTurbulenceResult:
    """Optimize one reduced QA low-turbulence design with Adam."""

    cfg = config or QALowTurbulenceConfig()
    initial_p = (
        default_qa_low_turbulence_initial_params()
        if initial_params is None
        else _validate_params(initial_params)
    )
    trace = _run_qa_low_turbulence_adam(
        jnp.asarray(initial_p),
        cfg,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
    )
    scalar_gate, residual_gate, observable_gate, covariance = _sensitivity_reports(
        trace.final_params,
        cfg,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        finite_difference_workers=finite_difference_workers,
    )
    return _qa_low_turbulence_result(
        cfg=cfg,
        initial_params=jnp.asarray(initial_p),
        trace=trace,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        scalar_gate=scalar_gate,
        residual_gate=residual_gate,
        observable_gate=observable_gate,
        covariance=covariance,
    )


__all__ = ["optimize_qa_low_turbulence"]
