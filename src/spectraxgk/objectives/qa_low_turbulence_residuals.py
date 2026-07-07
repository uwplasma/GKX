"""Residual and sensitivity gates for reduced QA low-turbulence objectives."""

from __future__ import annotations

from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
)
from spectraxgk.objectives.qa_low_turbulence_model import (
    _fd_gate_tolerances,
    _qa_low_turbulence_core,
    qa_low_turbulence_observable_vector,
    qa_low_turbulence_observables,
)
from spectraxgk.objectives.stellarator import PARAMETER_NAMES, _validate_params
from spectraxgk.objectives.autodiff_validation import (
    autodiff_finite_difference_report,
    covariance_diagnostics,
)


def qa_low_turbulence_residual_names(
    *,
    includes_nonlinear_heat_flux: bool,
) -> tuple[str, ...]:
    """Return the stable residual names for the comparison objective."""

    names = (
        "aspect_constraint",
        "minimum_iota_floor",
        "operating_iota_floor",
        "quasisymmetry_residual",
        "qa_helical_shaping_amplitude",
        *(f"regularization_{name}" for name in PARAMETER_NAMES),
    )
    if includes_nonlinear_heat_flux:
        return (*names, "reduced_nonlinear_heat_flux")
    return names


def qa_low_turbulence_residual_vector(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    includes_nonlinear_heat_flux: bool,
) -> jnp.ndarray:
    """Return weighted residuals for the aspect-6 QA low-turbulence objective."""

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    obs = qa_low_turbulence_observables(p, cfg)
    dtype = p.dtype
    aspect_res = jnp.sqrt(jnp.asarray(cfg.aspect_weight, dtype=dtype)) * (
        (obs["aspect"] - cfg.target_aspect) / cfg.target_aspect
    )
    iota_res = (
        jnp.sqrt(jnp.asarray(cfg.iota_floor_weight, dtype=dtype))
        * obs["iota_floor_violation"]
    )
    operating_iota_res = (
        jnp.sqrt(jnp.asarray(cfg.iota_operating_weight, dtype=dtype))
        * obs["iota_operating_floor_violation"]
    )
    qa_res = jnp.sqrt(jnp.asarray(cfg.qa_weight, dtype=dtype)) * obs["qa_residual"]
    helical_res = (
        jnp.sqrt(jnp.asarray(cfg.helical_shaping_weight, dtype=dtype))
        * (_qa_low_turbulence_core(p, cfg)["helical_mismatch"])
    )
    reg_res = jnp.sqrt(jnp.asarray(cfg.regularization, dtype=dtype)) * p
    parts = [
        jnp.asarray(
            [aspect_res, iota_res, operating_iota_res, qa_res, helical_res], dtype=dtype
        ),
        reg_res,
    ]
    if includes_nonlinear_heat_flux:
        q_res = jnp.sqrt(
            jnp.maximum(
                jnp.asarray(cfg.nonlinear_weight, dtype=dtype)
                * obs["nonlinear_heat_flux_mean"],
                jnp.asarray(0.0, dtype=dtype),
            )
        )
        parts.append(jnp.asarray([q_res], dtype=dtype))
    return jnp.concatenate(parts)


def qa_low_turbulence_objective(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    includes_nonlinear_heat_flux: bool,
) -> jnp.ndarray:
    """Return the scalar reduced QA comparison objective."""

    residual = qa_low_turbulence_residual_vector(
        params,
        config,
        includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
    )
    return jnp.dot(residual, residual)


def qa_low_turbulence_observable_sensitivity_report(
    params: jnp.ndarray | Sequence[float],
    config: QALowTurbulenceConfig | None = None,
    *,
    finite_difference_workers: int = 1,
) -> dict[str, Any]:
    """Check the full controls-to-observables differentiable plumbing.

    This gate is stricter than the scalar objective check: it differentiates
    the full reduced observable vector, including the long-window nonlinear
    heat-flux mean, CV, and trend, and compares the JAX Jacobian against
    central finite differences.
    """

    cfg = config or QALowTurbulenceConfig()
    p = _validate_params(params)
    fd_step, rtol, atol = _fd_gate_tolerances(cfg.fd_step)
    report = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_observable_vector(x, cfg),
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    report["observable_names"] = list(QA_LOW_TURBULENCE_OBSERVABLE_NAMES)
    report["parameter_names"] = list(PARAMETER_NAMES)
    report["kind"] = "qa_low_turbulence_observable_sensitivity_report"
    report["claim_level"] = (
        "full_reduced_controls_to_linear_quasilinear_nonlinear_observable_"
        "differentiability_gate"
    )
    return report


def _sensitivity_reports(
    params: jnp.ndarray,
    config: QALowTurbulenceConfig,
    *,
    includes_nonlinear_heat_flux: bool,
    finite_difference_workers: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    fd_step, rtol, atol = _fd_gate_tolerances(config.fd_step)
    scalar_gate = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_objective(
            x,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        params,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    residual_gate = autodiff_finite_difference_report(
        lambda x: qa_low_turbulence_residual_vector(
            x,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        params,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        workers=finite_difference_workers,
    )
    jac = np.asarray(residual_gate["jacobian_ad"], dtype=float)
    residual = np.asarray(
        qa_low_turbulence_residual_vector(
            params,
            config,
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        ),
        dtype=float,
    )
    covariance = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    covariance["residual_names"] = list(
        qa_low_turbulence_residual_names(
            includes_nonlinear_heat_flux=includes_nonlinear_heat_flux,
        )
    )
    covariance["source"] = "qa_low_turbulence_weighted_residuals"
    observable_gate = qa_low_turbulence_observable_sensitivity_report(
        params,
        config,
        finite_difference_workers=finite_difference_workers,
    )
    return scalar_gate, residual_gate, observable_gate, covariance


__all__ = [
    "_sensitivity_reports",
    "qa_low_turbulence_objective",
    "qa_low_turbulence_observable_sensitivity_report",
    "qa_low_turbulence_residual_names",
    "qa_low_turbulence_residual_vector",
]
