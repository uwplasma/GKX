"""Optimizer loop for reduced QA low-turbulence objectives."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    QALowTurbulenceResult,
)
from spectraxgk.objectives.qa_low_turbulence_model import (
    default_qa_low_turbulence_initial_params,
    qa_low_turbulence_observable_vector,
)
from spectraxgk.objectives.qa_low_turbulence_residuals import (
    _sensitivity_reports,
    qa_low_turbulence_objective,
)
from spectraxgk.objectives.stellarator import PARAMETER_NAMES, _validate_params


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
    p = jnp.asarray(initial_p)
    m = jnp.zeros_like(p)
    v = jnp.zeros_like(p)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.0e-8
    grad_fn = jax.value_and_grad(
        lambda x: qa_low_turbulence_objective(
            x,
            cfg,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        )
    )
    history: list[dict[str, Any]] = []
    objective0, grad0 = grad_fn(p)
    history.append(_history_row(0, p, objective0, grad0, cfg))

    for step in range(1, int(cfg.steps) + 1):
        objective, grad = grad_fn(p)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        mhat = m / (1.0 - beta1**step)
        vhat = v / (1.0 - beta2**step)
        p = p - cfg.learning_rate * mhat / (jnp.sqrt(vhat) + eps)
        if step % 5 == 0 or step == int(cfg.steps):
            new_objective, new_grad = grad_fn(p)
            history.append(_history_row(step, p, new_objective, new_grad, cfg))

    final_objective, final_grad = grad_fn(p)
    if history[-1]["step"] != int(cfg.steps):
        history.append(
            _history_row(int(cfg.steps), p, final_objective, final_grad, cfg)
        )
    scalar_gate, residual_gate, observable_gate, covariance = _sensitivity_reports(
        p,
        cfg,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        finite_difference_workers=finite_difference_workers,
    )
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
        initial_params=tuple(float(x) for x in np.asarray(initial_p, dtype=float)),
        final_params=tuple(float(x) for x in np.asarray(p, dtype=float)),
        initial_objective=float(objective0),
        final_objective=float(final_objective),
        initial_observables=tuple(
            float(x)
            for x in np.asarray(
                qa_low_turbulence_observable_vector(initial_p, cfg), dtype=float
            )
        ),
        final_observables=tuple(
            float(x) for x in np.asarray(qa_low_turbulence_observable_vector(p, cfg))
        ),
        history=tuple(history),
        residual_gradient_gate=residual_gate,
        scalar_gradient_gate=scalar_gate,
        observable_gradient_gate=observable_gate,
        covariance=covariance,
        config=asdict(cfg),
    )


__all__ = ["optimize_qa_low_turbulence"]
