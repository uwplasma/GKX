"""Solver-ready geometry objective gates used by differentiability tests."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.autodiff_checks import observable_gradient_validation_report
from spectraxgk.geometry.flux_tube_contract import flux_tube_geometry_from_mapping
from spectraxgk.objectives.core import SOLVER_OBJECTIVE_NAMES


SOLVER_GEOMETRY_PARAMETER_NAMES = ("bmag_ripple", "curvature_drift_scale")
TINY_OBJECTIVE_NAMES = (
    "mean_bmag",
    "drift_rms",
    "metric_weighted_bmag_proxy",
)


def default_solver_geometry_design_params() -> jnp.ndarray:
    """Return the small geometry-design vector used by the release gate."""

    return jnp.asarray([0.05, 0.20], dtype=jnp.float32)


def solver_ready_geometry_mapping(params: jnp.ndarray, theta: jnp.ndarray) -> dict[str, Any]:
    """Map a two-parameter design vector into solver-ready flux-tube arrays."""

    p = jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector [bmag_ripple, curvature_drift_scale]")
    ripple = p[0]
    drift = p[1]
    theta_arr = jnp.asarray(theta)
    ones = jnp.ones_like(theta_arr)
    zeros = jnp.zeros_like(theta_arr)
    bmag = 1.0 + ripple * jnp.cos(theta_arr)
    return {
        "theta": theta_arr,
        "gradpar": 0.7 * ones,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta_arr),
        "gds2": 1.0 + 0.1 * ripple * jnp.cos(theta_arr),
        "gds21": 0.05 * ripple * jnp.sin(theta_arr),
        "gds22": 1.0 + 0.05 * ripple * jnp.cos(theta_arr),
        "cvdrift": drift * jnp.cos(theta_arr),
        "gbdrift": drift * jnp.cos(theta_arr),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones / (0.7 * bmag),
        "grho": ones,
        "q": 1.4,
        "s_hat": 0.0,
        "R0": 1.0,
        "nfp": 1,
    }


def tiny_differentiable_objective_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-4,
    rtol: float = 2.0e-4,
    atol: float = 2.0e-6,
) -> dict[str, object]:
    """Validate a tiny differentiable objective on the solver-ready geometry map.

    This is a lightweight objective-observable gate for CI and documentation. It
    checks the reusable AD/finite-difference report path without running the
    linear eigensolver or optional VMEC/Boozer backends.
    """

    p = default_solver_geometry_design_params() if params is None else jnp.asarray(params)
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector")
    theta = jnp.linspace(-jnp.pi, jnp.pi, 16, endpoint=False, dtype=p.dtype)

    def objective_observables(x: jnp.ndarray) -> jnp.ndarray:
        geom = flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="tiny_solver_ready_objective_gradient_gate",
            validate_finite=False,
        )
        bmag = jnp.asarray(geom.bmag_profile)
        gds2 = jnp.asarray(geom.gds2_profile)
        drift = jnp.asarray(geom.cv_profile)
        gb = jnp.asarray(geom.gb_profile)
        drift_rms = jnp.sqrt(jnp.mean(drift * drift + gb * gb))
        metric_weighted_bmag = jnp.mean(gds2 * bmag) + 0.25 * x[1] * x[1]
        return jnp.asarray([jnp.mean(bmag), drift_rms, metric_weighted_bmag])

    report = observable_gradient_validation_report(
        objective_observables,
        p,
        fd_step=float(fd_step),
        rtol=float(rtol),
        atol=float(atol),
        observable_names=TINY_OBJECTIVE_NAMES,
        param_names=SOLVER_GEOMETRY_PARAMETER_NAMES,
        relative_floor=1.0e-12,
        report_kind="tiny_solver_ready_objective_gradient_ad_fd_gate",
    )
    report.update(
        {
            "source_scope": "solver_ready_geometry_contract",
            "claim_scope": (
                "tiny differentiable objective-observable gate only; not a "
                "linear eigenpair, VMEC/Boozer, or nonlinear transport claim"
            ),
        }
    )
    return report


def _objective_gate_rows(
    report: dict[str, object],
    *,
    parameter_names: tuple[str, ...] = SOLVER_GEOMETRY_PARAMETER_NAMES,
    objective_names: tuple[str, ...] = SOLVER_OBJECTIVE_NAMES,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    implicit = np.asarray(report["jacobian_implicit"], dtype=float)
    finite_difference = np.asarray(report["jacobian_fd"], dtype=float)
    rows: list[dict[str, object]] = []
    for i, objective in enumerate(objective_names):
        for j, parameter in enumerate(parameter_names):
            fd_value = float(finite_difference[i, j])
            implicit_value = float(implicit[i, j])
            abs_error = abs(implicit_value - fd_value)
            rel_error = abs_error / max(abs(fd_value), float(atol))
            rows.append(
                {
                    "objective": objective,
                    "parameter": parameter,
                    "implicit": implicit_value,
                    "finite_difference": fd_value,
                    "abs_error": abs_error,
                    "rel_error": rel_error,
                    "atol": float(atol),
                    "rtol": float(rtol),
                    "passed": bool(abs_error <= float(atol) or rel_error <= float(rtol)),
                }
            )
    return rows


__all__ = [
    "SOLVER_GEOMETRY_PARAMETER_NAMES",
    "TINY_OBJECTIVE_NAMES",
    "_objective_gate_rows",
    "default_solver_geometry_design_params",
    "solver_ready_geometry_mapping",
    "tiny_differentiable_objective_gradient_report",
]
