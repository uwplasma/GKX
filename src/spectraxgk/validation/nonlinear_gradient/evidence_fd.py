"""Finite-difference turbulence-gradient evidence reports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import math

from spectraxgk.validation.nonlinear_gradient.evidence_core import (
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    _ensemble_statistics_row,
    _finite_float,
    _gate,
    _json_number,
    _paired_replicate_fd_diagnostics,
)


def nonlinear_turbulence_gradient_finite_difference_report(
    *,
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    delta_parameter: float,
    parameter_name: str,
    baseline_path: str | None = None,
    plus_path: str | None = None,
    minus_path: str | None = None,
    config: NonlinearTurbulenceGradientFiniteDifferenceConfig | None = None,
) -> dict[str, Any]:
    """Build a production long-window central finite-difference gradient gate.

    Inputs must be replicated ``nonlinear_window_ensemble_report`` payloads for
    the same nonlinear case and analysis window, differing only by the perturbed
    parameter.  The report computes the central finite-difference heat-flux
    gradient and checks that the response is resolved above ensemble
    uncertainty before allowing any turbulence-gradient claim.
    """

    cfg = config or NonlinearTurbulenceGradientFiniteDifferenceConfig()
    delta = float(delta_parameter)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_parameter must be finite and positive")

    rows = {
        "minus": _ensemble_statistics_row(minus, path=minus_path),
        "baseline": _ensemble_statistics_row(baseline, path=baseline_path),
        "plus": _ensemble_statistics_row(plus, path=plus_path),
    }
    means = {
        name: _finite_float(row.get("ensemble_mean")) for name, row in rows.items()
    }
    sems = {name: _finite_float(row.get("combined_sem")) for name, row in rows.items()}
    finite_means = all(value is not None for value in means.values())
    finite_sems = all(value is not None for value in sems.values())

    if finite_means:
        assert means["minus"] is not None
        assert means["baseline"] is not None
        assert means["plus"] is not None
        minus_mean = float(means["minus"])
        baseline_mean = float(means["baseline"])
        plus_mean = float(means["plus"])
        central_gradient = (plus_mean - minus_mean) / (2.0 * delta)
        forward_gradient = (plus_mean - baseline_mean) / delta
        backward_gradient = (baseline_mean - minus_mean) / delta
        response = abs(plus_mean - minus_mean)
        response_fraction = response / max(abs(baseline_mean), float(cfg.value_floor))
        fd_asymmetry_rel = abs(forward_gradient - backward_gradient) / max(
            abs(central_gradient),
            float(cfg.value_floor),
        )
        fd_condition_number = (abs(plus_mean) + abs(minus_mean)) / max(
            response,
            float(cfg.value_floor),
        )
    else:
        central_gradient = math.nan
        forward_gradient = math.nan
        backward_gradient = math.nan
        response = math.nan
        response_fraction = math.nan
        fd_asymmetry_rel = math.nan
        fd_condition_number = math.nan

    if finite_sems:
        assert sems["plus"] is not None
        assert sems["minus"] is not None
        gradient_uncertainty = math.sqrt(
            float(sems["plus"]) ** 2 + float(sems["minus"]) ** 2
        ) / (2.0 * delta)
        gradient_uncertainty_rel = gradient_uncertainty / max(
            abs(central_gradient) if math.isfinite(central_gradient) else 0.0,
            float(cfg.value_floor),
        )
    else:
        gradient_uncertainty = math.nan
        gradient_uncertainty_rel = math.nan

    source_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        n_reports = _finite_float(row.get("n_reports"))
        source_gates.extend(
            [
                _gate(
                    f"{name}_ensemble_kind",
                    row.get("kind") == "nonlinear_window_ensemble_report",
                    f"kind={row.get('kind')}",
                ),
                _gate(
                    f"{name}_ensemble_passed",
                    bool(row["passed"]),
                    f"path={row.get('path')}",
                ),
                _gate(
                    f"{name}_ensemble_replicated",
                    n_reports is not None and n_reports >= int(cfg.min_window_reports),
                    f"n_reports={n_reports} min={cfg.min_window_reports}",
                ),
            ]
        )
    window_gates: list[dict[str, Any]] = []
    for name, row in rows.items():
        mean_rel_spread = _finite_float(row.get("mean_rel_spread"))
        combined_sem_rel = _finite_float(row.get("combined_sem_rel"))
        window_gates.extend(
            [
                _gate(
                    f"{name}_window_mean_spread",
                    mean_rel_spread is not None
                    and mean_rel_spread <= float(cfg.max_window_mean_rel_spread),
                    f"mean_rel_spread={mean_rel_spread} max={cfg.max_window_mean_rel_spread}",
                ),
                _gate(
                    f"{name}_window_sem",
                    combined_sem_rel is not None
                    and combined_sem_rel <= float(cfg.max_window_combined_sem_rel),
                    f"combined_sem_rel={combined_sem_rel} max={cfg.max_window_combined_sem_rel}",
                ),
            ]
        )

    gradient_gates = [
        _gate("finite_window_means", finite_means, f"means={means}"),
        _gate("finite_window_uncertainties", finite_sems, f"combined_sem={sems}"),
        _gate(
            "fd_response_resolved",
            math.isfinite(response_fraction)
            and response_fraction >= float(cfg.min_fd_response_fraction),
            f"response_fraction={response_fraction} min={cfg.min_fd_response_fraction}",
        ),
        _gate(
            "fd_asymmetry_bounded",
            math.isfinite(fd_asymmetry_rel)
            and fd_asymmetry_rel <= float(cfg.max_fd_asymmetry_rel),
            f"fd_asymmetry_rel={fd_asymmetry_rel} max={cfg.max_fd_asymmetry_rel}",
        ),
        _gate(
            "fd_condition_number_bounded",
            math.isfinite(fd_condition_number)
            and fd_condition_number <= float(cfg.max_fd_condition_number),
            f"fd_condition_number={fd_condition_number} max={cfg.max_fd_condition_number}",
        ),
        _gate(
            "gradient_uncertainty_bounded",
            math.isfinite(gradient_uncertainty_rel)
            and gradient_uncertainty_rel <= float(cfg.max_gradient_uncertainty_rel),
            f"gradient_uncertainty_rel={gradient_uncertainty_rel} max={cfg.max_gradient_uncertainty_rel}",
        ),
    ]
    gates = [*source_gates, *window_gates, *gradient_gates]
    passed = all(bool(gate["passed"]) for gate in gates)
    return {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "claim_scope": (
            "production_long_window nonlinear turbulence gradient from matched replicated "
            "post-transient heat-flux windows"
        ),
        "parameter_name": str(parameter_name),
        "delta_parameter": delta,
        "passed": passed,
        "production_nonlinear_window_gradient_gate": passed,
        "nonlinear_turbulence_gradient_gate": passed,
        "metrics": {
            "central_gradient": _json_number(central_gradient),
            "forward_gradient": _json_number(forward_gradient),
            "backward_gradient": _json_number(backward_gradient),
            "response": _json_number(response),
            "response_fraction": _json_number(response_fraction),
            "fd_asymmetry_rel": _json_number(fd_asymmetry_rel),
            "asymmetry_rel": _json_number(fd_asymmetry_rel),
            "fd_condition_number": _json_number(fd_condition_number),
            "condition_number": _json_number(fd_condition_number),
            "gradient_uncertainty": _json_number(gradient_uncertainty),
            "gradient_uncertainty_rel": _json_number(gradient_uncertainty_rel),
            "gradient_relative_uncertainty": _json_number(gradient_uncertainty_rel),
            "baseline_window_mean": means["baseline"],
            "plus_window_mean": means["plus"],
            "minus_window_mean": means["minus"],
            "baseline_window_sem": sems["baseline"],
            "plus_window_sem": sems["plus"],
            "minus_window_sem": sems["minus"],
        },
        "source_ensembles": rows,
        "paired_replicate_diagnostics": _paired_replicate_fd_diagnostics(
            rows=rows,
            delta=delta,
            value_floor=float(cfg.value_floor),
        ),
        "config": asdict(cfg),
        "gates": gates,
        "blockers": [gate["metric"] for gate in gates if not bool(gate["passed"])],
    }


__all__ = ["nonlinear_turbulence_gradient_finite_difference_report"]
