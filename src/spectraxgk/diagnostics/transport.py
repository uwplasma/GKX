"""Species transport and turbulent-heating diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import jax.numpy as jnp

from spectraxgk.core.grid import SpectralGrid
from spectraxgk.diagnostics.channels import (
    _heat_flux_channel_contrib_species,
    _particle_flux_channel_contrib_species,
    _turbulent_heating_contrib_species,
)
from spectraxgk.diagnostics.metadata import (
    NonlinearTurbulenceGradientFiniteDifferenceConfig,
    _ensemble_statistics_row,
    _finite_float,
    _gate,
    _json_number,
    _paired_replicate_fd_diagnostics,
)
from spectraxgk.operators.linear.cache_model import LinearCache
from spectraxgk.operators.linear.params import LinearParams

__all__ = [
    "heat_flux_channel_species",
    "heat_flux_species",
    "heat_flux_total",
    "nonlinear_turbulence_gradient_finite_difference_report",
    "particle_flux_channel_species",
    "particle_flux_species",
    "particle_flux_total",
    "turbulent_heating_species",
    "turbulent_heating_total",
]


def heat_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Heat-flux diagnostic per species (gyroBohm units)."""

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


def heat_flux_channel_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar heat-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _heat_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def heat_flux_total(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Total heat-flux diagnostic."""

    return jnp.sum(
        heat_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
    )


def particle_flux_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Particle-flux diagnostic per species."""

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return jnp.sum(es_contrib + apar_contrib + bpar_contrib, axis=(1, 2, 3))


def particle_flux_channel_species(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ES, Apar, and Bpar particle-flux channels per species."""

    es_contrib, apar_contrib, bpar_contrib = _particle_flux_channel_contrib_species(
        G,
        phi,
        apar,
        bpar,
        cache,
        grid,
        params,
        flux_fac,
        use_dealias=use_dealias,
        flux_scale=flux_scale,
    )
    return (
        jnp.sum(es_contrib, axis=(1, 2, 3)),
        jnp.sum(apar_contrib, axis=(1, 2, 3)),
        jnp.sum(bpar_contrib, axis=(1, 2, 3)),
    )


def particle_flux_total(
    G: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    flux_fac: jnp.ndarray,
    *,
    use_dealias: bool = True,
    flux_scale: float = 1.0,
) -> jnp.ndarray:
    """Total particle-flux diagnostic."""

    return jnp.sum(
        particle_flux_species(
            G,
            phi,
            apar,
            bpar,
            cache,
            grid,
            params,
            flux_fac,
            use_dealias=use_dealias,
            flux_scale=flux_scale,
        )
    )


def turbulent_heating_species(
    G: jnp.ndarray,
    G_old: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    dt: jnp.ndarray | float,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Turbulent-heating diagnostic per species."""

    contrib = _turbulent_heating_contrib_species(
        G,
        G_old,
        phi,
        apar,
        bpar,
        phi_old,
        apar_old,
        bpar_old,
        cache,
        grid,
        params,
        vol_fac,
        dt,
        use_dealias=use_dealias,
    )
    return jnp.sum(contrib, axis=(1, 2, 3))


def turbulent_heating_total(
    G: jnp.ndarray,
    G_old: jnp.ndarray,
    phi: jnp.ndarray,
    apar: jnp.ndarray,
    bpar: jnp.ndarray,
    phi_old: jnp.ndarray,
    apar_old: jnp.ndarray,
    bpar_old: jnp.ndarray,
    cache: LinearCache,
    grid: SpectralGrid,
    params: LinearParams,
    vol_fac: jnp.ndarray,
    dt: jnp.ndarray | float,
    *,
    use_dealias: bool = True,
) -> jnp.ndarray:
    """Total turbulent-heating diagnostic."""

    return jnp.sum(
        turbulent_heating_species(
            G,
            G_old,
            phi,
            apar,
            bpar,
            phi_old,
            apar_old,
            bpar_old,
            cache,
            grid,
            params,
            vol_fac,
            dt,
            use_dealias=use_dealias,
        )
    )


# ---- finite-difference evidence reports ----


@dataclass(frozen=True)
class _FiniteDifferenceMetrics:
    central_gradient: float
    forward_gradient: float
    backward_gradient: float
    response: float
    response_fraction: float
    fd_asymmetry_rel: float
    fd_condition_number: float
    gradient_uncertainty: float
    gradient_uncertainty_rel: float


def _validated_fd_inputs(
    config: NonlinearTurbulenceGradientFiniteDifferenceConfig | None,
    delta_parameter: float,
) -> tuple[NonlinearTurbulenceGradientFiniteDifferenceConfig, float]:
    cfg = config or NonlinearTurbulenceGradientFiniteDifferenceConfig()
    delta = float(delta_parameter)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_parameter must be finite and positive")
    return cfg, delta


def _window_rows(
    *,
    baseline: dict[str, Any],
    plus: dict[str, Any],
    minus: dict[str, Any],
    baseline_path: str | None,
    plus_path: str | None,
    minus_path: str | None,
) -> dict[str, dict[str, Any]]:
    return {
        "minus": _ensemble_statistics_row(minus, path=minus_path),
        "baseline": _ensemble_statistics_row(baseline, path=baseline_path),
        "plus": _ensemble_statistics_row(plus, path=plus_path),
    }


def _window_mean_sem(
    rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    means = {
        name: _finite_float(row.get("ensemble_mean")) for name, row in rows.items()
    }
    sems = {name: _finite_float(row.get("combined_sem")) for name, row in rows.items()}
    return means, sems


def _required_float(values: dict[str, float | None], key: str) -> float:
    value = values[key]
    assert value is not None
    return float(value)


def _fd_transport_response(
    *,
    means: dict[str, float | None],
    delta: float,
    value_floor: float,
) -> tuple[float, float, float, float, float, float, float]:
    finite_means = all(value is not None for value in means.values())
    if not finite_means:
        return (math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan)
    minus_mean = _required_float(means, "minus")
    baseline_mean = _required_float(means, "baseline")
    plus_mean = _required_float(means, "plus")
    central_gradient = (plus_mean - minus_mean) / (2.0 * delta)
    forward_gradient = (plus_mean - baseline_mean) / delta
    backward_gradient = (baseline_mean - minus_mean) / delta
    response = abs(plus_mean - minus_mean)
    response_fraction = response / max(abs(baseline_mean), value_floor)
    fd_asymmetry_rel = abs(forward_gradient - backward_gradient) / max(
        abs(central_gradient),
        value_floor,
    )
    fd_condition_number = (abs(plus_mean) + abs(minus_mean)) / max(
        response,
        value_floor,
    )
    return (
        central_gradient,
        forward_gradient,
        backward_gradient,
        response,
        response_fraction,
        fd_asymmetry_rel,
        fd_condition_number,
    )


def _fd_gradient_uncertainty(
    *,
    sems: dict[str, float | None],
    central_gradient: float,
    delta: float,
    value_floor: float,
) -> tuple[float, float]:
    finite_sems = all(value is not None for value in sems.values())
    if not finite_sems:
        return math.nan, math.nan
    gradient_uncertainty = math.sqrt(
        _required_float(sems, "plus") ** 2 + _required_float(sems, "minus") ** 2
    ) / (2.0 * delta)
    gradient_uncertainty_rel = gradient_uncertainty / max(
        abs(central_gradient) if math.isfinite(central_gradient) else 0.0,
        value_floor,
    )
    return gradient_uncertainty, gradient_uncertainty_rel


def _finite_difference_metrics(
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
    delta: float,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> _FiniteDifferenceMetrics:
    value_floor = float(cfg.value_floor)
    (
        central_gradient,
        forward_gradient,
        backward_gradient,
        response,
        response_fraction,
        fd_asymmetry_rel,
        fd_condition_number,
    ) = _fd_transport_response(
        means=means,
        delta=delta,
        value_floor=value_floor,
    )
    gradient_uncertainty, gradient_uncertainty_rel = _fd_gradient_uncertainty(
        sems=sems,
        central_gradient=central_gradient,
        delta=delta,
        value_floor=value_floor,
    )
    return _FiniteDifferenceMetrics(
        central_gradient=central_gradient,
        forward_gradient=forward_gradient,
        backward_gradient=backward_gradient,
        response=response,
        response_fraction=response_fraction,
        fd_asymmetry_rel=fd_asymmetry_rel,
        fd_condition_number=fd_condition_number,
        gradient_uncertainty=gradient_uncertainty,
        gradient_uncertainty_rel=gradient_uncertainty_rel,
    )


def _source_ensemble_gates(
    rows: dict[str, dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
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
    return source_gates


def _window_quality_gates(
    rows: dict[str, dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
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
    return window_gates


def _gradient_resolution_gates(
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    finite_means = all(value is not None for value in means.values())
    finite_sems = all(value is not None for value in sems.values())
    gradient_gates = [
        _gate("finite_window_means", finite_means, f"means={means}"),
        _gate("finite_window_uncertainties", finite_sems, f"combined_sem={sems}"),
        _gate(
            "fd_response_resolved",
            math.isfinite(metrics.response_fraction)
            and metrics.response_fraction >= float(cfg.min_fd_response_fraction),
            f"response_fraction={metrics.response_fraction} min={cfg.min_fd_response_fraction}",
        ),
        _gate(
            "fd_asymmetry_bounded",
            math.isfinite(metrics.fd_asymmetry_rel)
            and metrics.fd_asymmetry_rel <= float(cfg.max_fd_asymmetry_rel),
            f"fd_asymmetry_rel={metrics.fd_asymmetry_rel} max={cfg.max_fd_asymmetry_rel}",
        ),
        _gate(
            "fd_condition_number_bounded",
            math.isfinite(metrics.fd_condition_number)
            and metrics.fd_condition_number <= float(cfg.max_fd_condition_number),
            f"fd_condition_number={metrics.fd_condition_number} max={cfg.max_fd_condition_number}",
        ),
        _gate(
            "gradient_uncertainty_bounded",
            math.isfinite(metrics.gradient_uncertainty_rel)
            and metrics.gradient_uncertainty_rel
            <= float(cfg.max_gradient_uncertainty_rel),
            f"gradient_uncertainty_rel={metrics.gradient_uncertainty_rel} max={cfg.max_gradient_uncertainty_rel}",
        ),
    ]
    return gradient_gates


def _fd_metrics_payload(
    metrics: _FiniteDifferenceMetrics,
    *,
    means: dict[str, float | None],
    sems: dict[str, float | None],
) -> dict[str, Any]:
    return {
        "central_gradient": _json_number(metrics.central_gradient),
        "forward_gradient": _json_number(metrics.forward_gradient),
        "backward_gradient": _json_number(metrics.backward_gradient),
        "response": _json_number(metrics.response),
        "response_fraction": _json_number(metrics.response_fraction),
        "fd_asymmetry_rel": _json_number(metrics.fd_asymmetry_rel),
        "asymmetry_rel": _json_number(metrics.fd_asymmetry_rel),
        "fd_condition_number": _json_number(metrics.fd_condition_number),
        "condition_number": _json_number(metrics.fd_condition_number),
        "gradient_uncertainty": _json_number(metrics.gradient_uncertainty),
        "gradient_uncertainty_rel": _json_number(metrics.gradient_uncertainty_rel),
        "gradient_relative_uncertainty": _json_number(metrics.gradient_uncertainty_rel),
        "baseline_window_mean": means["baseline"],
        "plus_window_mean": means["plus"],
        "minus_window_mean": means["minus"],
        "baseline_window_sem": sems["baseline"],
        "plus_window_sem": sems["plus"],
        "minus_window_sem": sems["minus"],
    }


def _finite_difference_gates(
    *,
    rows: dict[str, dict[str, Any]],
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> list[dict[str, Any]]:
    return [
        *_source_ensemble_gates(rows, cfg),
        *_window_quality_gates(rows, cfg),
        *_gradient_resolution_gates(
            means=means,
            sems=sems,
            metrics=metrics,
            cfg=cfg,
        ),
    ]


def _pack_finite_difference_report(
    *,
    parameter_name: str,
    delta: float,
    rows: dict[str, dict[str, Any]],
    means: dict[str, float | None],
    sems: dict[str, float | None],
    metrics: _FiniteDifferenceMetrics,
    gates: list[dict[str, Any]],
    cfg: NonlinearTurbulenceGradientFiniteDifferenceConfig,
) -> dict[str, Any]:
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
        "metrics": _fd_metrics_payload(metrics, means=means, sems=sems),
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

    cfg, delta = _validated_fd_inputs(config, delta_parameter)
    rows = _window_rows(
        baseline=baseline,
        plus=plus,
        minus=minus,
        baseline_path=baseline_path,
        plus_path=plus_path,
        minus_path=minus_path,
    )
    means, sems = _window_mean_sem(rows)
    metrics = _finite_difference_metrics(
        means=means,
        sems=sems,
        delta=delta,
        cfg=cfg,
    )
    gates = _finite_difference_gates(
        rows=rows,
        means=means,
        sems=sems,
        metrics=metrics,
        cfg=cfg,
    )
    return _pack_finite_difference_report(
        parameter_name=parameter_name,
        delta=delta,
        rows=rows,
        means=means,
        sems=sems,
        metrics=metrics,
        gates=gates,
        cfg=cfg,
    )
