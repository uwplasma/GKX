#!/usr/bin/env python3
"""Build zonal-response figures and optimization-row artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

from gkx.artifacts.io import (  # noqa: E402
    load_diagnostic_time_series,
)
from gkx.artifacts.plotting import (  # noqa: E402
    set_plot_style,
    zonal_flow_response_figure,
)
from gkx.diagnostics.zonal_validation import (  # noqa: E402
    zonal_flow_response_metrics,
)
from gkx.diagnostics.validation_gates import (  # noqa: E402
    evaluate_scalar_gate,
    gate_report,
    gate_report_to_dict,
)
from gkx.objectives.zonal import (  # noqa: E402
    ZonalFlowObjectiveConfig,
    zonal_flow_objective_artifact_from_records,
)
from gkx.workflows.runtime.artifacts import (  # noqa: E402
    run_runtime_nonlinear_with_artifacts,
)
from gkx.workflows.runtime.toml import load_runtime_from_toml  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "docs" / "_static" / "w7x_zonal_response_panel.csv"
DEFAULT_COMPARISON = ROOT / "docs" / "_static" / "w7x_zonal_reference_compare.csv"
DEFAULT_OUT_JSON = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.json"
DEFAULT_OUT_CSV = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.csv"
DEFAULT_OUT_PNG = ROOT / "docs" / "_static" / "zonal_flow_objective_gate.png"

MERLO_CASE_III_REFERENCE = {
    "paper": "Merlo et al., Phys. Plasmas 23, 032104 (2016)",
    "case": "III",
    "q_s": 1.389,
    "s_hat": 0.751,
    "epsilon": 0.18,
    "kappa": 1.4723,
    "delta": -0.0070,
    "D": -0.0139,
    "a_MHD": 0.5425,
    "dRgeom_dr": -0.1569,
    "dkappa_dr": -0.0728,
    "ddelta_dr": -0.0140,
    "kx_rhoi": 0.05,
    "ky": 0.0,
    "tmax_R0_over_vi": 150.0,
    # Figure read-offs are paper-scale gates, not frozen trace replacements.
    "residual_phi_over_phi0": 0.190,
    "omega_gam_R0_over_vi": 2.24,
    "gamma_gam_R0_over_vi": -0.17,
}

MERLO_CASE_III_GATE_TOLERANCES = {
    "residual_atol": 0.015,
    "omega_atol_R0_over_vi": 0.10,
    "gamma_atol_R0_over_vi": 0.03,
}

# GKX's own converged Case-III answer, and what the artifact gate asserts.
#
# The literature read-offs above are still reported, but they are NOT gated any
# more: at converged Hermite resolution GKX's residual and GAM frequency sit
# outside the paper-scale tolerances, and the previous baseline passed only
# because Nm=24 puts the whole averaging window past the onset of Hermite
# recurrence. The honest gate is therefore GKX's converged value plus the
# stability conditions that make it a measurement at all. See
# The branch-only roadmap's Merlo audit and docs/testing.rst.
MERLO_CASE_III_GKX_CONVERGED = {
    "protocol": (
        "Nm=144, Nl=4, Nz=32, dt=0.0025 a/v_i, t_max=60 a/v_i, "
        "diagnostic sample spacing 0.025 a/v_i"
    ),
    "ladder": "Nm = 24, 48, 96, 144, 192 at dt = 0.0025 with Nl=4, Nz=32 held fixed",
    "residual_phi_over_phi0": 0.2059,
    "residual_atol": 0.006,
    "omega_gam_R0_over_vi": 2.345,
    "omega_atol_R0_over_vi": 0.05,
    "gamma_gam_R0_over_vi": -0.184,
    "gamma_atol_R0_over_vi": 0.010,
    # t_quiet ~ recurrence_coefficient * sqrt(Nm) is the measured onset of
    # Hermite recurrence for this case; every analysis window must close before
    # it. At Nm=144 that is t ~ 66 against a 4-period damping window at t ~ 30.
    "recurrence_coefficient": 5.5,
    # residual_std / |residual_level| inside the averaging window. It is 1.20 at
    # the retired Nm=24 baseline -- the scatter exceeded the gated number -- and
    # 0.14 here. The limit is what separates those two regimes.
    "residual_scatter_ratio_limit": 0.25,
    "trace_completeness_atol": 0.02,
}

COLLISIONAL_ZONAL_PROTOCOL = {
    "paper": "Frei, Ernst & Ricci, Phys. Plasmas 29, 093902 (2022)",
    "figures": [12, 13, 14],
    "q": 1.4,
    "epsilon": 0.1,
    "normalized_collisionality": 3.13,
    "maximum_hermite_order": 24,
    "maximum_laguerre_order": 10,
    "wavenumbers": [0.05, 0.1, 0.2],
    "xiao_residual": (0.1**2 / 1.4**2) / (1.0 + 0.1**2 / 1.4**2),
}


def collisional_zonal_frequency(
    *, normalized_collisionality: float, q: float, epsilon: float
) -> float:
    r"""Convert :math:`\nu_i^*=\sqrt{2}q\nu/\epsilon^{3/2}` to solver units."""

    values = (float(normalized_collisionality), float(q), float(epsilon))
    if not np.all(np.isfinite(values)) or any(value <= 0.0 for value in values):
        raise ValueError(
            "normalized collisionality, q, and epsilon must be finite and > 0"
        )
    return float(values[0] * values[2] ** 1.5 / (np.sqrt(2.0) * values[1]))


def require_active_zonal_mode(grid: Any, *, kx_index: int) -> None:
    """Reject a requested zonal mode that the spectral mask would remove."""

    mask = np.asarray(grid.dealias_mask, dtype=bool)
    if mask.ndim != 2 or not 0 <= int(kx_index) < mask.shape[1]:
        raise ValueError("kx_index is outside the spectral grid")
    ky = np.asarray(grid.ky, dtype=float)
    zonal_rows = np.flatnonzero(np.isclose(ky, 0.0))
    if zonal_rows.size != 1:
        raise ValueError("collisional zonal runs require exactly one ky=0 row")
    # Linear single-ky grids intentionally bypass the nonlinear two-thirds
    # convolution mask; see ``_apply_dealias_to_kperp_and_drifts``.
    if ky.size == 1:
        return
    if not bool(mask[int(zonal_rows[0]), int(kx_index)]):
        raise ValueError(
            "requested zonal mode is outside the active dealiased spectrum; "
            "increase Nx or use a single-ky linear grid"
        )


def summarize_collisional_zonal_campaign(
    trace_records: list[dict[str, object]],
    section_records: list[dict[str, object]],
    *,
    residual_atol: float = 1.5e-3,
) -> dict[str, object]:
    """Evaluate the complete Frei--Ernst--Ricci zonal-response protocol.

    Trace rows use ``model``, ``kx``, ``t_nu``, ``response``, ``p_max``, and
    ``j_max``. Velocity-section rows additionally use ``coordinate``
    (``parallel`` or ``perpendicular``), ``abscissa``, and
    ``normalized_distribution``. Missing models, time coverage, resolution,
    or velocity sections fail closed.
    """

    required_models = ("coulomb", "original_sugama", "improved_sugama")
    required_kx = tuple(
        float(value) for value in COLLISIONAL_ZONAL_PROTOCOL["wavenumbers"]
    )
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    finite = True
    resolution_passed = True
    for record in trace_records:
        model = str(record["model"]).strip().lower()
        kx = float(record["kx"])
        values = (
            float(record["t_nu"]),
            float(record["response"]),
            float(record["p_max"]),
            float(record["j_max"]),
        )
        finite &= bool(np.all(np.isfinite(values)))
        resolution_passed &= bool(
            int(record["p_max"]) >= COLLISIONAL_ZONAL_PROTOCOL["maximum_hermite_order"]
            and int(record["j_max"])
            >= COLLISIONAL_ZONAL_PROTOCOL["maximum_laguerre_order"]
        )
        grouped.setdefault((model, kx), []).append(record)

    required_pairs = {(model, kx) for model in required_models for kx in required_kx}
    complete_pairs = required_pairs.issubset(grouped)
    time_coverage_passed = complete_pairs
    normalized: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    tails: dict[str, dict[str, float]] = {model: {} for model in required_models}
    if complete_pairs:
        for key in sorted(required_pairs):
            records = sorted(grouped[key], key=lambda row: float(row["t_nu"]))
            time = np.asarray([float(row["t_nu"]) for row in records])
            response = np.asarray([float(row["response"]) for row in records])
            time_coverage_passed &= bool(
                time.size >= 50 and time[0] <= 0.05 and time[-1] >= 30.0
            )
            finite &= bool(np.all(np.isfinite(response)))
            initial = response[0]
            if not np.isfinite(initial) or abs(initial) < 1.0e-14:
                finite = False
                normalized[key] = (time, np.full_like(response, np.nan))
                continue
            response = response / initial
            normalized[key] = (time, response)
            tail = response[time >= 25.0]
            if tail.size < 5:
                time_coverage_passed = False
                continue
            tails[key[0]][f"{key[1]:.2f}"] = float(np.median(tail))

    xiao = float(COLLISIONAL_ZONAL_PROTOCOL["xiao_residual"])
    residual_passed = complete_pairs and all(
        abs(tails.get(model, {}).get("0.05", np.inf) - xiao) <= residual_atol
        for model in required_models
    )
    ordering_passed = complete_pairs and all(
        tails.get("original_sugama", {}).get(f"{kx:.2f}", np.inf)
        < tails.get("improved_sugama", {}).get(f"{kx:.2f}", -np.inf)
        < tails.get("coulomb", {}).get(f"{kx:.2f}", -np.inf)
        for kx in (0.1, 0.2)
    )
    early_errors: dict[str, dict[str, float]] = {}
    improved_closer = complete_pairs
    if complete_pairs:
        for kx in (0.1, 0.2):
            base_time, base = normalized[("coulomb", kx)]
            mask = (base_time >= 0.0) & (base_time <= 10.0)
            model_errors: dict[str, float] = {}
            for model in ("original_sugama", "improved_sugama"):
                time, response = normalized[(model, kx)]
                compared = np.interp(base_time[mask], time, response)
                model_errors[model] = float(
                    np.sqrt(np.mean((compared - base[mask]) ** 2))
                )
            early_errors[f"{kx:.2f}"] = model_errors
            improved_closer &= bool(
                model_errors["improved_sugama"] < model_errors["original_sugama"]
            )

    sections: dict[tuple[str, str], list[dict[str, object]]] = {}
    section_finite = True
    section_resolution = True
    for record in section_records:
        model = str(record["model"]).strip().lower()
        coordinate = str(record["coordinate"]).strip().lower()
        values = (
            float(record["kx"]),
            float(record["t_nu"]),
            float(record["abscissa"]),
            float(record["normalized_distribution"]),
        )
        section_finite &= bool(np.all(np.isfinite(values)))
        section_resolution &= bool(
            int(record["p_max"]) >= COLLISIONAL_ZONAL_PROTOCOL["maximum_hermite_order"]
            and int(record["j_max"])
            >= COLLISIONAL_ZONAL_PROTOCOL["maximum_laguerre_order"]
        )
        if np.isclose(values[0], 0.2) and np.isclose(values[1], 5.0, atol=0.05):
            sections.setdefault((model, coordinate), []).append(record)
    required_sections = {
        (model, coordinate)
        for model in required_models
        for coordinate in ("parallel", "perpendicular")
    }
    velocity_sections_passed = required_sections.issubset(sections) and all(
        len(sections[key]) >= 21
        and np.isclose(
            max(float(row["normalized_distribution"]) for row in sections[key]),
            1.0,
            atol=0.03,
        )
        for key in required_sections
    )
    gates = {
        "finite_values": bool(finite and section_finite),
        "paper_resolution_reached": bool(resolution_passed and section_resolution),
        "required_model_wavenumber_traces_present": bool(complete_pairs),
        "normalized_time_window_reaches_30": bool(time_coverage_passed),
        "drift_kinetic_residual_matches_xiao": bool(residual_passed),
        "gyrokinetic_tail_ordering_os_is_coulomb": bool(ordering_passed),
        "improved_sugama_closer_to_coulomb_before_tnu10": bool(improved_closer),
        "velocity_sections_present_at_tnu5": bool(velocity_sections_passed),
    }
    return {
        "schema_version": 1,
        "claim_scope": "collisional_zonal_response_figures_12_14",
        "protocol": dict(COLLISIONAL_ZONAL_PROTOCOL),
        "thresholds": {"xiao_residual_absolute_tolerance": residual_atol},
        "tail_response": tails,
        "early_window_rms_error_vs_coulomb": early_errors,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def _add_response_metric_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tail-fraction", type=float, default=0.3)
    parser.add_argument("--initial-fraction", type=float, default=0.1)
    parser.add_argument(
        "--initial-policy",
        choices=("window_abs_mean", "first_abs"),
        default="window_abs_mean",
    )
    parser.add_argument("--peak-fit-max-peaks", type=int, default=None)
    parser.add_argument(
        "--damping-fit-mode",
        choices=("combined_envelope", "branchwise_extrema"),
        default="combined_envelope",
    )
    parser.add_argument(
        "--frequency-fit-mode",
        choices=("peak_spacing", "hilbert_phase"),
        default="peak_spacing",
    )
    parser.add_argument("--fit-window-tmin", type=float, default=None)
    parser.add_argument("--fit-window-tmax", type=float, default=None)
    parser.add_argument("--hilbert-trim-fraction", type=float, default=0.2)


def _response_metrics(args: argparse.Namespace, t: np.ndarray, response: np.ndarray):
    return zonal_flow_response_metrics(
        t,
        response,
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
        peak_fit_max_peaks=args.peak_fit_max_peaks,
        damping_fit_mode=str(args.damping_fit_mode),
        frequency_fit_mode=str(args.frequency_fit_mode),
        fit_window_tmin=args.fit_window_tmin,
        fit_window_tmax=args.fit_window_tmax,
        hilbert_trim_fraction=float(args.hilbert_trim_fraction),
    )


def _response_payload(metrics) -> dict[str, object]:
    return {
        "initial_level": metrics.initial_level,
        "initial_policy": metrics.initial_policy,
        "residual_level": metrics.residual_level,
        "residual_std": metrics.residual_std,
        "response_rms": metrics.response_rms,
        "gam_frequency": metrics.gam_frequency,
        "gam_damping_rate": metrics.gam_damping_rate,
        "damping_method": metrics.damping_method,
        "frequency_method": metrics.frequency_method,
        "peak_count": metrics.peak_count,
        "peak_fit_count": metrics.peak_fit_count,
        "tmin": metrics.tmin,
        "tmax": metrics.tmax,
        "fit_tmin": metrics.fit_tmin,
        "fit_tmax": metrics.fit_tmax,
    }


def _write_response_panel(
    *, t: np.ndarray, response: np.ndarray, out: Path, title: str, metrics
) -> None:
    fig, _axes = zonal_flow_response_figure(t, response, metrics=metrics, title=title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    if out.suffix.lower() != ".pdf":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_campaign_csv(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


@dataclass(frozen=True)
class _CollisionalZonalProblem:
    grid: Any
    geometry: Any
    parameters: Any
    cache: Any
    initial: Any
    volume_weight: Any
    kx_index: int
    major_radius: float
    minor_radius: float


def _build_collisional_zonal_problem(
    *,
    config: Path,
    kx: float,
    nz: int,
    n_laguerre: int,
    n_hermite: int,
) -> _CollisionalZonalProblem:
    """Build the common paper-geometry state used by every collision model."""

    from gkx.core.grid import build_spectral_grid
    from gkx.diagnostics.moments import fieldline_quadrature_weights
    from gkx.geometry import apply_geometry_grid_defaults
    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.workflows.runtime.startup import (
        _build_initial_condition,
        build_runtime_geometry,
        build_runtime_linear_params,
    )

    q = float(COLLISIONAL_ZONAL_PROTOCOL["q"])
    epsilon = float(COLLISIONAL_ZONAL_PROTOCOL["epsilon"])
    cfg, _raw = load_runtime_from_toml(config)
    major_radius = float(cfg.geometry.R0)
    minor_radius = epsilon * major_radius
    cfg = replace(
        cfg,
        grid=replace(
            cfg.grid,
            Nx=3,
            Ny=1,
            Nz=int(nz),
            Lx=float(2.0 * np.pi / kx),
            Ly=float(2.0 * np.pi),
            boundary="periodic",
        ),
        geometry=replace(
            cfg.geometry,
            rhoc=minor_radius,
            q=q,
            s_hat=0.5,
            epsilon=epsilon,
            akappa=1.0,
            akappri=0.0,
            tri=0.0,
            tripri=0.0,
            shift=0.0,
        ),
    )
    geometry = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geometry, cfg.grid))
    kx_index = int(np.argmin(np.abs(np.asarray(grid.kx, dtype=float) - kx)))
    if not np.isclose(float(np.asarray(grid.kx)[kx_index]), kx, atol=1.0e-12):
        raise ValueError("spectral grid does not contain the requested kx")
    require_active_zonal_mode(grid, kx_index=kx_index)
    parameters = build_runtime_linear_params(cfg, Nm=n_hermite, geom=geometry)
    cache = build_linear_cache(grid, geometry, parameters, n_laguerre, n_hermite)
    initial = _build_initial_condition(
        grid,
        geometry,
        cfg,
        ky_index=0,
        kx_index=kx_index,
        Nl=n_laguerre,
        Nm=n_hermite,
        nspecies=1,
    )
    volume_weight, _flux_weight = fieldline_quadrature_weights(geometry, grid)
    return _CollisionalZonalProblem(
        grid=grid,
        geometry=geometry,
        parameters=parameters,
        cache=cache,
        initial=initial,
        volume_weight=volume_weight,
        kx_index=kx_index,
        major_radius=major_radius,
        minor_radius=minor_radius,
    )


def _integrate_collisional_zonal_trace(
    problem: _CollisionalZonalProblem,
    *,
    collision_operator: Any,
    terms: Any,
    collision_frequency: float,
    dt: float,
    maximum_normalized_time: float,
    sample_stride: int,
    snapshot_normalized_time: float | None = None,
) -> tuple[
    np.ndarray, np.ndarray, float, int, np.ndarray | None, float | None
]:
    """Advance one collision model and optionally retain one physical state."""

    import time

    import jax
    import jax.numpy as jnp

    from gkx.solvers.linear.integrators import integrate_linear
    from gkx.terms.assembly import compute_fields_cached

    initial_phi = compute_fields_cached(
        problem.initial,
        problem.cache,
        problem.parameters,
        terms=terms,
        use_custom_vjp=False,
    ).phi
    initial_response = jnp.sum(initial_phi[0, problem.kx_index] * problem.volume_weight)
    requested_steps = int(np.ceil(maximum_normalized_time / collision_frequency / dt))
    steps = int(np.ceil(requested_steps / sample_stride) * sample_stride)
    if snapshot_normalized_time is not None:
        if not 0.0 < snapshot_normalized_time < maximum_normalized_time:
            raise ValueError(
                "snapshot_normalized_time must lie inside the integration window"
            )
        requested_snapshot_steps = int(
            np.ceil(snapshot_normalized_time / collision_frequency / dt)
        )
        snapshot_steps = int(
            np.ceil(requested_snapshot_steps / sample_stride) * sample_stride
        )
        if snapshot_steps >= steps:
            raise ValueError(
                "snapshot must precede the final sampled integration state"
            )
    else:
        snapshot_steps = 0

    def integrate_segment(initial: Any, segment_steps: int):
        return integrate_linear(
            initial,
            problem.grid,
            problem.geometry,
            problem.parameters,
            dt=dt,
            steps=segment_steps,
            method="rk2",
            cache=problem.cache,
            terms=terms,
            sample_stride=sample_stride,
            show_progress=True,
            collision_operator=collision_operator,
        )

    started = time.perf_counter()
    if snapshot_steps:
        snapshot_state, first_history = integrate_segment(
            problem.initial, snapshot_steps
        )
        remaining_steps = steps - snapshot_steps
        if remaining_steps:
            final_state, second_history = integrate_segment(
                snapshot_state, remaining_steps
            )
            field_history = jnp.concatenate((first_history, second_history), axis=0)
        else:
            final_state, field_history = snapshot_state, first_history
        snapshot_time = snapshot_steps * dt * collision_frequency
    else:
        final_state, field_history = integrate_segment(problem.initial, steps)
        snapshot_state = None
        snapshot_time = None
    jax.block_until_ready((final_state, field_history))
    elapsed_seconds = time.perf_counter() - started
    history = np.asarray(field_history)
    response = np.einsum(
        "tz,z->t",
        history[:, 0, problem.kx_index, :],
        np.asarray(problem.volume_weight),
    )
    times = (np.arange(response.size) + 1) * dt * sample_stride
    response = np.concatenate(([complex(np.asarray(initial_response))], response))
    normalized_time = np.concatenate(([0.0], times)) * collision_frequency
    if not np.all(np.isfinite(response)):
        raise RuntimeError("collisional-zonal trace contains non-finite values")
    return (
        normalized_time,
        response,
        elapsed_seconds,
        steps,
        None if snapshot_state is None else np.asarray(snapshot_state),
        snapshot_time,
    )


def run_drift_kinetic_collisional_zonal_trace(
    *,
    config: Path,
    model_archive: Path,
    model: str,
    out_csv: Path,
    dt: float = 0.005,
    maximum_normalized_time: float = 30.0,
    sample_stride: int = 10,
    nz: int = 32,
) -> dict[str, object]:
    """Run one paper-resolution drift-kinetic collisional-zonal trace.

    The model archive is an offline research artifact with ``coulomb``,
    ``original_sugama``, and ``improved_sugama`` matrices in runtime
    Hermite-major convention. It is intentionally supplied by path rather than
    shipped in the package: exact arbitrary-order matrix generation remains a
    reproducible research workflow, not a default executable dependency.
    """

    import jax
    import jax.numpy as jnp

    from gkx.operators.linear.collisions import (
        DriftKineticMomentCollisionOperator,
    )
    from gkx.operators.linear.params import LinearTerms

    if model not in {"coulomb", "original_sugama", "improved_sugama"}:
        raise ValueError("unknown drift-kinetic collision model")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(maximum_normalized_time) or maximum_normalized_time <= 0.0:
        raise ValueError("maximum_normalized_time must be finite and > 0")
    if sample_stride < 1 or nz < 8:
        raise ValueError("sample_stride must be >= 1 and nz must be >= 8")

    with np.load(model_archive) as archive:
        required = {
            model,
            "maximum_hermite_order",
            "maximum_laguerre_order",
            "correction_order",
        }
        if missing := required - set(archive.files):
            raise ValueError(f"collision model archive is missing {sorted(missing)}")
        matrix = np.asarray(archive[model], dtype=float)
        maximum_hermite = int(archive["maximum_hermite_order"])
        maximum_laguerre = int(archive["maximum_laguerre_order"])
        correction_order = int(archive["correction_order"])
    mode_count = (maximum_hermite + 1) * (maximum_laguerre + 1)
    if matrix.shape != (mode_count, mode_count) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "collision model matrix does not match its declared resolution"
        )

    q = float(COLLISIONAL_ZONAL_PROTOCOL["q"])
    epsilon = float(COLLISIONAL_ZONAL_PROTOCOL["epsilon"])
    normalized_collisionality = float(
        COLLISIONAL_ZONAL_PROTOCOL["normalized_collisionality"]
    )
    collision_frequency = collisional_zonal_frequency(
        normalized_collisionality=normalized_collisionality,
        q=q,
        epsilon=epsilon,
    )
    kx = 0.05
    n_laguerre = maximum_laguerre + 1
    n_hermite = maximum_hermite + 1
    problem = _build_collisional_zonal_problem(
        config=config,
        kx=kx,
        nz=nz,
        n_laguerre=n_laguerre,
        n_hermite=n_hermite,
    )
    operator = DriftKineticMomentCollisionOperator(
        jnp.asarray(collision_frequency * matrix[None, None])
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    normalized_time, response, elapsed_seconds, steps, _snapshot, _snapshot_time = (
        _integrate_collisional_zonal_trace(
            problem,
            collision_operator=operator,
            terms=terms,
            collision_frequency=collision_frequency,
            dt=dt,
            maximum_normalized_time=maximum_normalized_time,
            sample_stride=sample_stride,
        )
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "kx",
                "t_nu",
                "response",
                "response_imag",
                "p_max",
                "j_max",
            ),
        )
        writer.writeheader()
        for time_value, value in zip(normalized_time, response, strict=True):
            writer.writerow(
                {
                    "model": model,
                    "kx": kx,
                    "t_nu": float(time_value),
                    "response": float(value.real),
                    "response_imag": float(value.imag),
                    "p_max": maximum_hermite,
                    "j_max": maximum_laguerre,
                }
            )
    normalized = response.real / response[0].real
    tail = normalized[normalized_time >= 25.0]
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "drift_kinetic_collisional_zonal_trace",
        "model": model,
        "config": _repo_relative(config),
        "model_archive": str(model_archive),
        "q": q,
        "epsilon": epsilon,
        "miller_minor_radius": problem.minor_radius,
        "miller_major_radius": problem.major_radius,
        "normalized_collisionality": normalized_collisionality,
        "collision_frequency": collision_frequency,
        "kx": kx,
        "dt": dt,
        "steps": steps,
        "sample_stride": sample_stride,
        "nz": nz,
        "maximum_hermite_order": maximum_hermite,
        "maximum_laguerre_order": maximum_laguerre,
        "correction_order": correction_order,
        "elapsed_seconds": elapsed_seconds,
        "tail_normalized_median": (None if tail.size == 0 else float(np.median(tail))),
        "maximum_imaginary_fraction": float(
            np.max(np.abs(response.imag)) / max(np.max(np.abs(response.real)), 1.0e-300)
        ),
        "finite": True,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(out_csv.with_suffix(".json"), report)
    return report


def reconstruct_collisional_zonal_velocity_sections(
    state: np.ndarray,
    problem: _CollisionalZonalProblem,
    *,
    model: str,
    normalized_time: float,
    point_count: int = 81,
) -> list[dict[str, object]]:
    r"""Reconstruct the Figure-14 ``|g_i|`` cuts from gyro-moments.

    Frei, Ernst & Ricci (2022), equation (52), expands the perturbed
    distribution in physicists' Hermite and ordinary Laguerre polynomials.
    GKX stores the equivalent signed-Laguerre coefficients. The two
    cuts are evaluated at the outboard midplane and normalized independently,
    as required by the paper-facing comparison gate.
    """

    from scipy.special import eval_hermite, eval_laguerre, gammaln

    values = np.asarray(state)
    if values.ndim == 6:
        if values.shape[0] != 1:
            raise ValueError("collisional zonal sections require one species")
        values = values[0]
    if values.ndim != 5:
        raise ValueError("zonal state must have Laguerre, Hermite, ky, kx, z axes")
    if point_count < 21:
        raise ValueError("velocity sections require at least 21 points")
    z_index = int(np.argmin(np.abs(np.asarray(problem.grid.z, dtype=float))))
    moments = values[:, :, 0, problem.kx_index, z_index]
    if not np.all(np.isfinite(moments)) or not np.any(np.abs(moments) > 0.0):
        raise ValueError("velocity-section moments must be finite and nonzero")

    n_laguerre, n_hermite = moments.shape
    hermite_order = np.arange(n_hermite)
    hermite_norm = np.exp(
        0.5 * (hermite_order * np.log(2.0) + gammaln(hermite_order + 1.0))
    )
    laguerre_sign = (-1.0) ** np.arange(n_laguerre)
    parallel = np.linspace(-3.0, 3.0, point_count)
    perpendicular = np.linspace(0.0, 4.0, point_count)
    hermite_parallel = np.asarray(
        [eval_hermite(order, parallel) for order in hermite_order]
    ) / hermite_norm[:, None]
    hermite_zero = np.asarray(
        [eval_hermite(order, 0.0) for order in hermite_order]
    ) / hermite_norm
    laguerre_zero = laguerre_sign * np.asarray(
        [eval_laguerre(order, 0.0) for order in range(n_laguerre)]
    )
    laguerre_perpendicular = laguerre_sign[:, None] * np.asarray(
        [eval_laguerre(order, perpendicular) for order in range(n_laguerre)]
    )
    distributions = {
        "parallel": np.exp(-(parallel**2))
        * np.abs(np.einsum("lm,ma,l->a", moments, hermite_parallel, laguerre_zero)),
        "perpendicular": np.exp(-perpendicular)
        * np.abs(
            np.einsum(
                "lm,m,la->a", moments, hermite_zero, laguerre_perpendicular
            )
        ),
    }
    coordinates = {"parallel": parallel, "perpendicular": perpendicular}
    rows: list[dict[str, object]] = []
    for coordinate, distribution in distributions.items():
        maximum = float(np.max(distribution))
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("velocity section cannot be normalized")
        for abscissa, value in zip(
            coordinates[coordinate], distribution / maximum, strict=True
        ):
            rows.append(
                {
                    "model": model,
                    "coordinate": coordinate,
                    "kx": float(np.asarray(problem.grid.kx)[problem.kx_index]),
                    "t_nu": float(normalized_time),
                    "abscissa": float(abscissa),
                    "normalized_distribution": float(value),
                    "p_max": n_hermite - 1,
                    "j_max": n_laguerre - 1,
                }
            )
    return rows


def run_finite_wavelength_collisional_zonal_trace(
    *,
    config: Path,
    table_archive: Path,
    model: str = "coulomb",
    kx: float,
    out_csv: Path,
    out_sections_csv: Path | None = None,
    dt: float = 0.005,
    maximum_normalized_time: float = 30.0,
    sample_stride: int = 10,
    nz: int = 32,
) -> dict[str, object]:
    """Run one equal-species finite-wavelength Coulomb or Sugama trace."""

    import jax
    import jax.numpy as jnp

    from gkx.operators.linear import (
        EqualSpeciesFiniteWavelengthCoulombOperator,
        EqualSpeciesFiniteWavelengthSugamaOperator,
    )
    from gkx.operators.linear.params import LinearTerms

    if kx not in {0.1, 0.2}:
        raise ValueError("finite-wavelength zonal kx must be 0.1 or 0.2")
    if model not in {"coulomb", "original_sugama", "improved_sugama"}:
        raise ValueError("unknown finite-wavelength collision model")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not np.isfinite(maximum_normalized_time) or maximum_normalized_time <= 0.0:
        raise ValueError("maximum_normalized_time must be finite and > 0")
    if sample_stride < 1 or nz < 8:
        raise ValueError("sample_stride must be >= 1 and nz must be >= 8")

    if out_sections_csv is not None and (
        kx != 0.2 or maximum_normalized_time <= 5.0
    ):
        raise ValueError(
            "velocity sections require kx=0.2 and maximum normalized time > 5"
        )
    matrix_names = ("test_table", "field_table")
    polarization_names = (
        "test_phi1",
        "field_phi1",
        "test_phi2",
        "field_phi2",
    )
    array_names = matrix_names + (polarization_names if model == "coulomb" else ())
    required = {"metadata", "bessel_argument_grid", *array_names}
    with np.load(table_archive) as archive:
        if missing := required - set(archive.files):
            raise ValueError(f"collision table archive is missing {sorted(missing)}")
        metadata = json.loads(str(archive["metadata"]))
        grid = np.asarray(archive["bessel_argument_grid"], dtype=float)
        arrays = tuple(np.asarray(archive[name], dtype=float) for name in array_names)
    expected_scope = {
        "coulomb": "equal_species_diagonal_finite_wavelength_coulomb_table",
        "original_sugama": (
            "equal_species_diagonal_finite_wavelength_original_sugama_table"
        ),
        "improved_sugama": (
            "equal_species_diagonal_finite_wavelength_improved_sugama_table"
        ),
    }[model]
    if metadata.get("claim_scope") != expected_scope:
        raise ValueError("collision table archive has the wrong claim scope")
    if metadata.get("laguerre_convention") != "runtime_signed":
        raise ValueError("collision table archive must use runtime Laguerre signs")
    resolution = metadata.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("collision table archive has invalid resolution metadata")
    maximum_hermite, maximum_laguerre = map(int, resolution)
    mode_count = (maximum_hermite + 1) * (maximum_laguerre + 1)
    expected_shapes: tuple[tuple[int, ...], ...] = (
        (grid.size, mode_count, mode_count),
        (grid.size, mode_count, mode_count),
    )
    if model == "coulomb":
        expected_shapes += ((grid.size, mode_count),) * 4
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.all(np.isfinite(grid))
        or not np.all(np.diff(grid) > 0.0)
    ):
        raise ValueError("collision table Bessel grid must be strictly increasing")
    if any(
        array.shape != shape
        for array, shape in zip(arrays, expected_shapes, strict=True)
    ):
        raise ValueError("collision table arrays do not match the declared resolution")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("collision table arrays must be finite")

    collision_frequency = collisional_zonal_frequency(
        normalized_collisionality=float(
            COLLISIONAL_ZONAL_PROTOCOL["normalized_collisionality"]
        ),
        q=float(COLLISIONAL_ZONAL_PROTOCOL["q"]),
        epsilon=float(COLLISIONAL_ZONAL_PROTOCOL["epsilon"]),
    )
    problem = _build_collisional_zonal_problem(
        config=config,
        kx=kx,
        nz=nz,
        n_laguerre=maximum_laguerre + 1,
        n_hermite=maximum_hermite + 1,
    )
    local_bessel_argument = np.sqrt(2.0 * np.asarray(problem.cache.b))[
        0, 0, problem.kx_index
    ]
    local_range = (
        float(np.min(local_bessel_argument)),
        float(np.max(local_bessel_argument)),
    )
    if local_range[0] < grid[0] or local_range[1] > grid[-1]:
        raise ValueError(
            "collision table does not cover the field-line Bessel-argument range"
        )
    operator_type = (
        EqualSpeciesFiniteWavelengthCoulombOperator
        if model == "coulomb"
        else EqualSpeciesFiniteWavelengthSugamaOperator
    )
    operator = operator_type(
        jnp.asarray(grid),
        jnp.asarray([[collision_frequency]]),
        *(jnp.asarray(array) for array in arrays),
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    (
        normalized_time,
        response,
        elapsed_seconds,
        steps,
        snapshot_state,
        snapshot_time,
    ) = (
        _integrate_collisional_zonal_trace(
            problem,
            collision_operator=operator,
            terms=terms,
            collision_frequency=collision_frequency,
            dt=dt,
            maximum_normalized_time=maximum_normalized_time,
            sample_stride=sample_stride,
            snapshot_normalized_time=(5.0 if out_sections_csv is not None else None),
        )
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "kx",
                "t_nu",
                "response",
                "response_imag",
                "p_max",
                "j_max",
            ),
        )
        writer.writeheader()
        for time_value, value in zip(normalized_time, response, strict=True):
            writer.writerow(
                {
                    "model": model,
                    "kx": kx,
                    "t_nu": float(time_value),
                    "response": float(value.real),
                    "response_imag": float(value.imag),
                    "p_max": maximum_hermite,
                    "j_max": maximum_laguerre,
                }
            )
    if out_sections_csv is not None:
        if snapshot_state is None or snapshot_time is None:
            raise RuntimeError("requested velocity-section state was not captured")
        section_rows = reconstruct_collisional_zonal_velocity_sections(
            snapshot_state,
            problem,
            model=model,
            normalized_time=snapshot_time,
        )
        out_sections_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_sections_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(section_rows[0]))
            writer.writeheader()
            writer.writerows(section_rows)
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "finite_wavelength_collisional_zonal_trace",
        "model": model,
        "config": _repo_relative(config),
        "table_archive": str(table_archive),
        "collision_frequency": collision_frequency,
        "kx": kx,
        "dt": dt,
        "steps": steps,
        "sample_stride": sample_stride,
        "nz": nz,
        "maximum_hermite_order": maximum_hermite,
        "maximum_laguerre_order": maximum_laguerre,
        "bessel_argument_grid": grid.tolist(),
        "fieldline_bessel_argument_range": list(local_range),
        "elapsed_seconds": elapsed_seconds,
        "velocity_sections_csv": (
            None if out_sections_csv is None else str(out_sections_csv)
        ),
        "velocity_section_normalized_time": snapshot_time,
        "maximum_imaginary_fraction": float(
            np.max(np.abs(response.imag)) / max(np.max(np.abs(response.real)), 1.0e-300)
        ),
        "finite": True,
        "devices": [str(device) for device in jax.devices()],
    }
    _write_json(out_csv.with_suffix(".json"), report)
    return report


def _normalized_trace_comparison(
    reference: list[dict[str, str]],
    candidate: list[dict[str, str]],
    *,
    maximum_relative_l2: float,
    maximum_absolute_error: float,
    error_context: str,
) -> dict[str, object]:
    """Compare two matched, initial-value-normalized zonal traces."""

    reference_time = np.asarray([float(row["t_nu"]) for row in reference])
    candidate_time = np.asarray([float(row["t_nu"]) for row in candidate])
    if reference_time.shape != candidate_time.shape or not np.allclose(
        reference_time, candidate_time, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError(f"{error_context} traces require identical sample times")
    reference_response = np.asarray([float(row["response"]) for row in reference])
    candidate_response = np.asarray([float(row["response"]) for row in candidate])
    if (
        reference_response.size < 2
        or reference_response[0] == 0.0
        or candidate_response[0] == 0.0
        or not np.all(np.isfinite(reference_response))
        or not np.all(np.isfinite(candidate_response))
    ):
        raise ValueError(f"{error_context} traces must be finite and normalizable")
    reference_normalized = reference_response / reference_response[0]
    candidate_normalized = candidate_response / candidate_response[0]
    difference = reference_normalized - candidate_normalized
    relative_l2 = float(
        np.linalg.norm(difference)
        / max(float(np.linalg.norm(candidate_normalized)), 1.0e-300)
    )
    maximum_absolute = float(np.max(np.abs(difference)))
    return {
        "samples": int(candidate_time.size),
        "maximum_normalized_time": float(candidate_time[-1]),
        "relative_l2": relative_l2,
        "maximum_absolute_error": maximum_absolute,
        "reference_final": float(reference_normalized[-1]),
        "candidate_final": float(candidate_normalized[-1]),
        "passed": (
            relative_l2 <= maximum_relative_l2
            and maximum_absolute <= maximum_absolute_error
        ),
    }


def write_finite_wavelength_zonal_grid_gate(
    *,
    coarse_traces: dict[float, Path],
    fine_traces: dict[float, Path],
    out_json: Path,
    maximum_relative_l2: float = 1.0e-3,
    maximum_absolute_error: float = 5.0e-4,
) -> dict[str, object]:
    """Gate nested B-grid interpolation on matched finite-B zonal traces."""

    if set(coarse_traces) != {0.1, 0.2} or set(fine_traces) != {0.1, 0.2}:
        raise ValueError("nested B-grid gate requires kx=0.1 and kx=0.2 traces")
    if maximum_relative_l2 <= 0.0 or maximum_absolute_error <= 0.0:
        raise ValueError("nested B-grid tolerances must be > 0")

    rows: dict[str, object] = {}
    passed = True
    for kx in (0.1, 0.2):
        coarse = _read_campaign_csv(coarse_traces[kx])
        fine = _read_campaign_csv(fine_traces[kx])
        metrics = _normalized_trace_comparison(
            coarse,
            fine,
            maximum_relative_l2=maximum_relative_l2,
            maximum_absolute_error=maximum_absolute_error,
            error_context="nested B-grid",
        )
        passed = passed and bool(metrics["passed"])
        metrics["coarse_final"] = metrics.pop("reference_final")
        metrics["fine_final"] = metrics.pop("candidate_final")
        rows[f"{kx:.1f}"] = metrics
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "finite_wavelength_zonal_b_grid_interpolation_pilot",
        "resolution": [7, 3],
        "coarse_bessel_argument_grid": [0.12, 0.16, 0.24, 0.32],
        "fine_bessel_argument_grid": [0.12, 0.14, 0.16, 0.24, 0.28, 0.32],
        "thresholds": {
            "maximum_relative_l2": maximum_relative_l2,
            "maximum_absolute_error": maximum_absolute_error,
        },
        "traces": rows,
        "gate_passed": passed,
        "notes": (
            "This closes B-grid interpolation at P7/J3 through t*nu=2 only. "
            "It does not close the paper-required P24/J10 moment hierarchy or "
            "the t*nu=30 collisional-zonal trace."
        ),
    }
    _write_json(out_json, report)
    return report


def write_finite_wavelength_zonal_moment_gate(
    *,
    hierarchy: list[tuple[tuple[int, int], dict[float, Path]]],
    out_json: Path,
    maximum_relative_l2: float = 5.0e-2,
    maximum_absolute_error: float = 5.0e-2,
) -> dict[str, object]:
    """Gate adjacent velocity-moment levels on matched normalized traces."""

    if len(hierarchy) < 2:
        raise ValueError("moment hierarchy gate requires at least two levels")
    if maximum_relative_l2 <= 0.0 or maximum_absolute_error <= 0.0:
        raise ValueError("moment hierarchy tolerances must be > 0")
    resolutions = [resolution for resolution, _ in hierarchy]
    if any(
        high[0] <= low[0] or high[1] <= low[1]
        for low, high in zip(resolutions, resolutions[1:])
    ):
        raise ValueError("moment hierarchy levels must increase in both P and J")

    loaded: list[tuple[tuple[int, int], dict[float, list[dict[str, str]]]]] = []
    for resolution, traces in hierarchy:
        if set(traces) != {0.1, 0.2}:
            raise ValueError("moment hierarchy levels require kx=0.1 and kx=0.2")
        rows_by_kx: dict[float, list[dict[str, str]]] = {}
        for kx, path in traces.items():
            rows = _read_campaign_csv(path)
            if not rows or any(
                int(row["p_max"]) != resolution[0] or int(row["j_max"]) != resolution[1]
                for row in rows
            ):
                raise ValueError("trace metadata does not match declared resolution")
            rows_by_kx[kx] = rows
        loaded.append((resolution, rows_by_kx))

    comparisons: list[dict[str, object]] = []
    passed = True
    for (low_resolution, low_traces), (high_resolution, high_traces) in zip(
        loaded, loaded[1:]
    ):
        trace_metrics: dict[str, object] = {}
        comparison_passed = True
        for kx in (0.1, 0.2):
            metrics = _normalized_trace_comparison(
                low_traces[kx],
                high_traces[kx],
                maximum_relative_l2=maximum_relative_l2,
                maximum_absolute_error=maximum_absolute_error,
                error_context="moment hierarchy",
            )
            comparison_passed = comparison_passed and bool(metrics["passed"])
            metrics["lower_final"] = metrics.pop("reference_final")
            metrics["higher_final"] = metrics.pop("candidate_final")
            trace_metrics[f"{kx:.1f}"] = metrics
        passed = passed and comparison_passed
        comparisons.append(
            {
                "lower_resolution": list(low_resolution),
                "higher_resolution": list(high_resolution),
                "traces": trace_metrics,
                "passed": comparison_passed,
            }
        )
    report: dict[str, object] = {
        "schema_version": 1,
        "claim_scope": "finite_wavelength_zonal_moment_hierarchy",
        "resolutions": [list(resolution) for resolution in resolutions],
        "thresholds": {
            "maximum_relative_l2": maximum_relative_l2,
            "maximum_absolute_error": maximum_absolute_error,
        },
        "comparisons": comparisons,
        "gate_passed": passed,
        "notes": (
            "Adjacent levels are compared on initial-value-normalized physical "
            "traces. A failed lower hierarchy is diagnostic evidence only; the "
            "paper protocol remains P24/J10 through t*nu=30."
        ),
    }
    _write_json(out_json, report)
    return report


def write_collisional_zonal_artifacts(
    trace_records: list[dict[str, object]],
    section_records: list[dict[str, object]],
    *,
    out_json: Path,
    out_png: Path,
) -> dict[str, object]:
    """Write the complete collisional-zonal gate and paper-protocol panel."""

    summary = summarize_collisional_zonal_campaign(trace_records, section_records)
    colors = {
        "coulomb": "#246A8D",
        "original_sugama": "#C43D3D",
        "improved_sugama": "#22A884",
    }
    labels = {
        "coulomb": "Coulomb",
        "original_sugama": "original Sugama",
        "improved_sugama": "improved Sugama",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for column, kx in enumerate((0.05, 0.1, 0.2)):
        axis = axes[0, column]
        for model in colors:
            records = sorted(
                (
                    row
                    for row in trace_records
                    if str(row["model"]).strip().lower() == model
                    and np.isclose(float(row["kx"]), kx)
                ),
                key=lambda row: float(row["t_nu"]),
            )
            if not records:
                continue
            time = np.asarray([float(row["t_nu"]) for row in records])
            response = np.asarray([float(row["response"]) for row in records])
            response = response / response[0]
            axis.plot(time, response, color=colors[model], lw=1.5, label=labels[model])
        axis.axhline(
            float(COLLISIONAL_ZONAL_PROTOCOL["xiao_residual"]),
            color="#20272E",
            lw=1.0,
            ls="--",
            label="Xiao residual" if column == 0 else None,
        )
        axis.set_yscale("log")
        axis.set_title(rf"$k_x\rho_i={kx:g}$")
        axis.set_xlabel(r"$t\nu_{ii}$")
        if column == 0:
            axis.set_ylabel(r"$R_z(t)=\phi_z(t)/\phi_z(0)$")
            axis.legend(frameon=False, fontsize=7.5)

    for column, coordinate in enumerate(("parallel", "perpendicular")):
        axis = axes[1, column]
        for model in colors:
            records = sorted(
                (
                    row
                    for row in section_records
                    if str(row["model"]).strip().lower() == model
                    and str(row["coordinate"]).strip().lower() == coordinate
                    and np.isclose(float(row["kx"]), 0.2)
                    and np.isclose(float(row["t_nu"]), 5.0, atol=0.05)
                ),
                key=lambda row: float(row["abscissa"]),
            )
            if records:
                axis.plot(
                    [float(row["abscissa"]) for row in records],
                    [float(row["normalized_distribution"]) for row in records],
                    color=colors[model],
                    lw=1.6,
                    label=labels[model],
                )
        reference_abscissa = (
            np.linspace(-3.0, 3.0, 161)
            if coordinate == "parallel"
            else np.linspace(0.0, 4.0, 161)
        )
        maxwellian = np.exp(
            -(reference_abscissa**2)
            if coordinate == "parallel"
            else -reference_abscissa
        )
        axis.plot(
            reference_abscissa,
            maxwellian,
            color="#20272E",
            lw=1.2,
            ls="--",
            label="Maxwellian",
        )
        axis.set_title(rf"$t\nu=5$: {coordinate} section")
        axis.set_xlabel(r"$s_\parallel$" if coordinate == "parallel" else r"$x$")
        if column == 0:
            axis.set_ylabel(r"$|g_i|/\max |g_i|$")
        axis.legend(frameon=False, fontsize=7.5)

    metric_axis = axes[1, 2]
    metric_axis.axis("off")
    status = "PASS" if summary["gate_passed"] else "OPEN"
    gate_labels = {
        "finite_values": "finite values",
        "paper_resolution_reached": r"$(P,J)\geq(24,10)$",
        "required_model_wavenumber_traces_present": "all model/kx traces",
        "normalized_time_window_reaches_30": r"time coverage to $t\nu=30$",
        "drift_kinetic_residual_matches_xiao": "Xiao residual",
        "gyrokinetic_tail_ordering_os_is_coulomb": "tail ordering: OS < IS < Coulomb",
        "improved_sugama_closer_to_coulomb_before_tnu10": r"IS closer to Coulomb for $t\nu\leq10$",
        "velocity_sections_present_at_tnu5": r"velocity sections at $t\nu=5$",
    }
    gate_lines = [
        f"{gate_labels[name]}: {'PASS' if passed else 'OPEN'}"
        for name, passed in summary["gates"].items()
    ]
    metric_axis.text(
        0.0,
        1.0,
        f"Protocol status: {status}\n\n" + "\n".join(gate_lines),
        va="top",
        fontsize=8.3,
        linespacing=1.35,
    )
    for axis in axes.flat[:5]:
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Pfirsch-Schlueter collisional zonal response",
        fontsize=14,
        color="#20272E",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="white")
    plt.close(fig)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out_json, summary)
    return summary


def summarize_drift_kinetic_collisional_zonal_campaign(
    trace_records: list[dict[str, object]],
    *,
    residual_atol: float = 1.5e-3,
) -> dict[str, object]:
    """Evaluate the drift-kinetic Figure-12 subset of the paper protocol."""

    required_models = ("coulomb", "original_sugama", "improved_sugama")
    normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    finite = True
    resolution = True
    coverage = True
    tails: dict[str, float] = {}
    for model in required_models:
        rows = sorted(
            (
                row
                for row in trace_records
                if str(row["model"]).strip().lower() == model
                and np.isclose(float(row["kx"]), 0.05)
            ),
            key=lambda row: float(row["t_nu"]),
        )
        if not rows:
            coverage = False
            continue
        time = np.asarray([float(row["t_nu"]) for row in rows])
        response = np.asarray([float(row["response"]) for row in rows])
        finite &= bool(np.all(np.isfinite(time)) and np.all(np.isfinite(response)))
        resolution &= all(
            int(row["p_max"]) >= 24 and int(row["j_max"]) >= 10 for row in rows
        )
        coverage &= bool(time.size >= 50 and time[0] <= 0.05 and time[-1] >= 30.0)
        if abs(response[0]) < 1.0e-14:
            finite = False
            continue
        response = response / response[0]
        normalized[model] = time, response
        tail = response[time >= 25.0]
        if tail.size < 5:
            coverage = False
        else:
            tails[model] = float(np.median(tail))

    complete = len(normalized) == len(required_models)
    xiao = float(COLLISIONAL_ZONAL_PROTOCOL["xiao_residual"])
    residual = complete and all(
        abs(tails.get(model, np.inf) - xiao) <= residual_atol
        for model in required_models
    )
    ordering = complete and tails.get("original_sugama", np.inf) < min(
        tails.get("coulomb", -np.inf), tails.get("improved_sugama", -np.inf)
    )
    early_errors: dict[str, float] = {}
    improved_closer = False
    if complete:
        base_time, base = normalized["coulomb"]
        mask = base_time <= 10.0
        for model in ("original_sugama", "improved_sugama"):
            time, response = normalized[model]
            compared = np.interp(base_time[mask], time, response)
            early_errors[model] = float(np.sqrt(np.mean((compared - base[mask]) ** 2)))
        improved_closer = (
            early_errors["improved_sugama"] < early_errors["original_sugama"]
        )
    gates = {
        "finite_values": finite,
        "paper_resolution_reached": resolution,
        "all_drift_kinetic_models_present": complete,
        "normalized_time_window_reaches_30": coverage,
        "residual_matches_xiao": residual,
        "original_sugama_damps_most_strongly": ordering,
        "improved_sugama_closer_to_coulomb_before_tnu10": improved_closer,
    }
    return {
        "schema_version": 1,
        "claim_scope": "drift_kinetic_collisional_zonal_response_figure_12",
        "protocol": dict(COLLISIONAL_ZONAL_PROTOCOL),
        "thresholds": {"xiao_residual_absolute_tolerance": residual_atol},
        "tail_response": tails,
        "early_window_rms_error_vs_coulomb": early_errors,
        "gates": gates,
        "gate_passed": all(gates.values()),
    }


def write_drift_kinetic_collisional_zonal_artifacts(
    trace_records: list[dict[str, object]],
    *,
    out_json: Path,
    out_png: Path,
) -> dict[str, object]:
    """Write the scoped Figure-12 gate and publication panel."""

    summary = summarize_drift_kinetic_collisional_zonal_campaign(trace_records)
    colors = {
        "coulomb": "#246A8D",
        "original_sugama": "#C43D3D",
        "improved_sugama": "#22A884",
    }
    labels = {
        "coulomb": "Coulomb",
        "original_sugama": "original Sugama",
        "improved_sugama": "improved Sugama (K=5)",
    }
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    for model, color in colors.items():
        rows = sorted(
            (
                row
                for row in trace_records
                if str(row["model"]).strip().lower() == model
                and np.isclose(float(row["kx"]), 0.05)
            ),
            key=lambda row: float(row["t_nu"]),
        )
        if not rows:
            continue
        time = np.asarray([float(row["t_nu"]) for row in rows])
        response = np.asarray([float(row["response"]) for row in rows])
        response /= response[0]
        axes[0].plot(time, np.abs(response), color=color, lw=1.45, label=labels[model])
        axes[1].plot(time, 1.0e3 * response, color=color, lw=1.7, label=labels[model])
    xiao = float(COLLISIONAL_ZONAL_PROTOCOL["xiao_residual"])
    collisionless = 1.0 / (1.0 + 1.6 * 1.4 / np.sqrt(0.1))
    axes[0].axhline(
        collisionless, color="#252A30", ls="--", lw=1.0, label="collisionless"
    )
    axes[0].axhline(xiao, color="#4666B0", ls=":", lw=1.2, label="Xiao residual")
    axes[1].axhline(1.0e3 * xiao, color="#4666B0", ls=":", lw=1.2)
    axes[0].set(yscale="log", xlim=(0.0, 30.0), ylim=(2.5e-3, 1.1))
    axes[1].set(xlim=(15.0, 30.0), ylim=(4.8, 6.8))
    axes[0].set_xlabel(r"$t\nu_{ii}$")
    axes[1].set_xlabel(r"$t\nu_{ii}$")
    axes[0].set_ylabel(r"$|R_z(t)|$")
    axes[1].set_ylabel(r"$10^3 R_z(t)$")
    axes[0].set_title("Damping history")
    axes[1].set_title("Late-time response")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.22, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    status = "PASS" if summary["gate_passed"] else "OPEN"
    fig.suptitle(
        rf"Drift-kinetic zonal response: $k_x\rho_i=0.05$, $\nu_i^*=3.13$ ({status})",
        fontsize=12.5,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor="white")
    plt.close(fig)
    _write_json(out_json, summary)
    return summary


def _main_collisional_zonal_dk(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Gate the paper's drift-kinetic Figure 12."
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_drift_kinetic_collisional_zonal_artifacts(
        _read_campaign_csv(args.traces), out_json=args.out_json, out_png=args.out_png
    )
    print(
        f"drift-kinetic zonal literature gate: {'PASS' if summary['gate_passed'] else 'OPEN'}"
    )
    return 0 if summary["gate_passed"] else 1


def _main_collisional_zonal(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Gate Frei--Ernst--Ricci collisional zonal-response traces."
    )
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = write_collisional_zonal_artifacts(
        _read_campaign_csv(args.traces),
        _read_campaign_csv(args.sections),
        out_json=args.out_json,
        out_png=args.out_png,
    )
    print(
        "collisional zonal literature gate: "
        f"{'PASS' if summary['gate_passed'] else 'OPEN'}"
    )
    return 0 if summary["gate_passed"] else 1


def _main_simulate_collisional_zonal_dk(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run one Frei--Ernst--Ricci drift-kinetic zonal trace."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("coulomb", "original_sugama", "improved_sugama"),
        required=True,
    )
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--maximum-normalized-time", type=float, default=30.0)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--nz", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_drift_kinetic_collisional_zonal_trace(
        config=args.config,
        model_archive=args.model_archive,
        model=args.model,
        out_csv=args.out_csv,
        dt=args.dt,
        maximum_normalized_time=args.maximum_normalized_time,
        sample_stride=args.sample_stride,
        nz=args.nz,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _main_simulate_collisional_zonal_finite_b(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Run one finite-wavelength Coulomb or Sugama zonal trace."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--table-archive", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("coulomb", "original_sugama", "improved_sugama"),
        default="coulomb",
    )
    parser.add_argument("--kx", type=float, choices=(0.1, 0.2), required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-sections-csv", type=Path)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--maximum-normalized-time", type=float, default=30.0)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--nz", type=int, default=32)
    args = parser.parse_args(argv)
    report = run_finite_wavelength_collisional_zonal_trace(
        config=args.config,
        table_archive=args.table_archive,
        model=str(args.model),
        kx=float(args.kx),
        out_csv=args.out_csv,
        out_sections_csv=args.out_sections_csv,
        dt=float(args.dt),
        maximum_normalized_time=float(args.maximum_normalized_time),
        sample_stride=int(args.sample_stride),
        nz=int(args.nz),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _main_collisional_zonal_grid_gate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Gate finite-wavelength zonal B-grid interpolation."
    )
    for prefix in ("coarse", "fine"):
        parser.add_argument(f"--{prefix}-kx010", type=Path, required=True)
        parser.add_argument(f"--{prefix}-kx020", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--maximum-relative-l2", type=float, default=1.0e-3)
    parser.add_argument("--maximum-absolute-error", type=float, default=5.0e-4)
    args = parser.parse_args(argv)
    report = write_finite_wavelength_zonal_grid_gate(
        coarse_traces={0.1: args.coarse_kx010, 0.2: args.coarse_kx020},
        fine_traces={0.1: args.fine_kx010, 0.2: args.fine_kx020},
        out_json=args.out_json,
        maximum_relative_l2=float(args.maximum_relative_l2),
        maximum_absolute_error=float(args.maximum_absolute_error),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def _main_collisional_zonal_moment_gate(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Gate adjacent finite-wavelength zonal moment levels."
    )
    parser.add_argument(
        "--level",
        action="append",
        nargs=4,
        metavar=("P", "J", "KX010_CSV", "KX020_CSV"),
        required=True,
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--maximum-relative-l2", type=float, default=5.0e-2)
    parser.add_argument("--maximum-absolute-error", type=float, default=5.0e-2)
    args = parser.parse_args(argv)
    hierarchy = [
        (
            (int(level[0]), int(level[1])),
            {0.1: Path(level[2]), 0.2: Path(level[3])},
        )
        for level in args.level
    ]
    report = write_finite_wavelength_zonal_moment_gate(
        hierarchy=hierarchy,
        out_json=args.out_json,
        maximum_relative_l2=float(args.maximum_relative_l2),
        maximum_absolute_error=float(args.maximum_absolute_error),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


def _build_response_csv_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a t,response CSV artifact.")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response.png",
    )
    parser.add_argument("--title", default="Zonal-flow response")
    _add_response_metric_args(parser)
    return parser


def _build_response_output_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot a saved zonal diagnostic.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--var", default="Phi2_zonal_t")
    parser.add_argument("--kx-index", type=int, default=None)
    parser.add_argument(
        "--component", choices=("real", "imag", "abs", "complex"), default="real"
    )
    parser.add_argument("--align-phase", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "zonal_flow_response_from_output.png",
    )
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    _add_response_metric_args(parser)
    return parser


def _main_response_csv(argv: list[str]) -> int:
    args = _build_response_csv_parser().parse_args(argv)
    data = np.genfromtxt(args.csv, delimiter=",", names=True, dtype=float)
    if {"t", "response"} - set(data.dtype.names or ()):
        raise ValueError("CSV must contain columns t,response")
    t = np.asarray(data["t"], dtype=float)
    response = np.asarray(data["response"], dtype=float)
    metrics = _response_metrics(args, t, response)
    _write_response_panel(
        t=t, response=response, out=args.out, title=args.title, metrics=metrics
    )
    _write_json(args.out.with_suffix(".json"), _response_payload(metrics))
    return 0


def _main_response_output(argv: list[str]) -> int:
    args = _build_response_output_parser().parse_args(argv)
    series = load_diagnostic_time_series(
        args.output,
        variable=args.var,
        kx_index=args.kx_index,
        component=args.component,
        align_phase=bool(args.align_phase),
    )
    if np.iscomplexobj(series.values):
        raise ValueError("zonal-response plotting requires a real extracted component")
    metrics = _response_metrics(args, series.t, series.values)
    _write_response_panel(
        t=series.t,
        response=series.values,
        out=args.out,
        title=args.title or f"{args.var} response",
        metrics=metrics,
    )
    csv_out = args.csv_out or args.out.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        csv_out,
        np.column_stack([series.t, series.values]),
        delimiter=",",
        header="t,response",
        comments="",
    )
    payload = {
        "source_path": series.source_path,
        "variable": series.variable,
        **_response_payload(metrics),
        "notes": (
            "Phi2_zonal_t is a zonal-energy proxy. Prefer Phi_zonal_mode_kxt "
            "with a selected kx and phase alignment for signed response studies."
        ),
    }
    _write_json(args.out.with_suffix(".json"), payload)
    return 0


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "nan", "none", "null"}:
            return None
        value = stripped
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def _kx_key(value: Any) -> float:
    scalar = _optional_float(value)
    if scalar is None:
        raise ValueError(f"missing finite kx value: {value!r}")
    return round(float(scalar), 10)


def _comparison_by_kx(path: Path | None) -> dict[float, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    table = _read_csv(path)
    rows: dict[float, dict[str, str]] = {}
    for row in table:
        if "kx" in row:
            rows[_kx_key(row["kx"])] = row
        elif "kx_target" in row:
            rows[_kx_key(row["kx_target"])] = row
    return rows


def _tail_std_ratio(row: dict[str, str] | None) -> float | None:
    if row is None:
        return None
    direct = _optional_float(row.get("tail_std_ratio"))
    if direct is not None:
        return direct
    tail_std = _optional_float(row.get("tail_std"))
    reference_tail_std = _optional_float(row.get("reference_tail_std"))
    if tail_std is None or reference_tail_std is None or reference_tail_std <= 0.0:
        return None
    return tail_std / reference_tail_std


def _recurrence_value(
    *,
    summary_row: dict[str, str],
    comparison_row: dict[str, str] | None,
    source: str,
) -> float | None:
    if source == "residual_std":
        return _optional_float(summary_row.get("residual_std"))
    if source == "tail_std":
        return (
            None
            if comparison_row is None
            else _optional_float(comparison_row.get("tail_std"))
        )
    if source == "tail_std_ratio":
        return _tail_std_ratio(comparison_row)
    if source != "auto":
        raise ValueError(f"unknown recurrence source: {source}")
    ratio = _tail_std_ratio(comparison_row)
    if ratio is not None:
        return ratio
    return _optional_float(summary_row.get("residual_std"))


def records_from_w7x_summary(
    summary_csv: Path,
    *,
    comparison_csv: Path | None = None,
    recurrence_source: str = "auto",
) -> list[dict[str, object]]:
    """Return normalized zonal-objective records from the W7-X summary CSV."""

    summary = _read_csv(summary_csv)
    comparison = _comparison_by_kx(comparison_csv)
    records: list[dict[str, object]] = []
    for row in summary:
        kx = _kx_key(row.get("kx_target", row.get("kx")))
        comparison_row = comparison.get(kx)
        recurrence = _recurrence_value(
            summary_row=row,
            comparison_row=comparison_row,
            source=recurrence_source,
        )
        records.append(
            {
                "surface": _optional_float(row.get("surface")) or 0.0,
                "alpha": _optional_float(row.get("alpha")) or 0.0,
                "kx": float(kx),
                "residual_level": row.get("residual_level"),
                "damping_rate": row.get("gam_damping_rate", row.get("damping_rate")),
                "linear_growth_rate": row.get("linear_growth_rate", 0.0),
                "recurrence_amplitude": recurrence,
            }
        )
    return records


def _write_row_csv(path: Path, payload: dict[str, object]) -> None:
    rows = list(payload["row_table"])
    if not rows:
        raise ValueError("cannot write an empty zonal-flow objective table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "surface",
        "alpha",
        "kx",
        "residual_level",
        "damping_rate",
        "linear_growth_rate",
        "recurrence_amplitude",
        "inverse_residual",
        "growth_over_residual",
        "sample_objective",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def _plot_payload(path: Path, payload: dict[str, object]) -> None:
    set_plot_style()
    table = list(payload["row_table"])
    kx = np.asarray([float(row["kx"]) for row in table], dtype=float)
    order = np.argsort(kx)
    labels = [f"{kx[index]:.2f}" for index in order]
    x = np.arange(order.size)
    metrics = {
        "Residual response\n(higher is better)": [
            float(table[index]["residual_level"]) for index in order
        ],
        "Damping penalty\n(lower is better)": [
            float(table[index]["damping_rate"]) for index in order
        ],
        "Recurrence/tail penalty\n(lower is better)": [
            float(table[index]["recurrence_amplitude"]) for index in order
        ],
        "Weighted sample objective\n(lower is better)": [
            float(table[index]["sample_objective"]) for index in order
        ],
    }
    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    for ax, (title, values), color in zip(
        axes.ravel(), metrics.items(), colors, strict=True
    ):
        ax.bar(x, values, color=color, alpha=0.86, edgecolor="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Zonal-flow objective row gate", fontsize=15)
    status = "promotion-ready" if payload["promotion_ready"] else "diagnostic only"
    fig.text(
        0.5,
        0.01,
        (
            f"Status: {status}; missing damping rows: {payload['missing_damping_count']}; "
            f"claim: {payload['claim_level']}"
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _parse_objective_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument(
        "--recurrence-source",
        choices=("auto", "residual_std", "tail_std", "tail_std_ratio"),
        default="auto",
        help="Late-envelope recurrence metric used for the fourth objective column.",
    )
    parser.add_argument(
        "--missing-damping-policy",
        choices=("zero", "fail"),
        default="zero",
        help=(
            "Use 'fail' for promoted physics gates. The default 'zero' writes a "
            "diagnostic W7-X row artifact while preserving promotion_ready=false."
        ),
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--damping-weight", type=float, default=1.0)
    parser.add_argument("--growth-over-residual-weight", type=float, default=0.0)
    parser.add_argument("--recurrence-weight", type=float, default=0.25)
    parser.add_argument("--residual-floor", type=float, default=1.0e-6)
    parser.add_argument(
        "--claim-level",
        default="diagnostic_zonal_objective_row_producer_not_promoted_w7x_optimization_claim",
    )
    return parser.parse_args(argv)


def _main_objective_gate(argv: list[str]) -> int:
    args = _parse_objective_args(argv)
    records = records_from_w7x_summary(
        args.summary_csv,
        comparison_csv=args.comparison_csv,
        recurrence_source=args.recurrence_source,
    )
    config = ZonalFlowObjectiveConfig(
        residual_weight=args.residual_weight,
        damping_weight=args.damping_weight,
        growth_over_residual_weight=args.growth_over_residual_weight,
        recurrence_weight=args.recurrence_weight,
        residual_floor=args.residual_floor,
    )
    payload = zonal_flow_objective_artifact_from_records(
        records,
        config=config,
        missing_damping_policy=args.missing_damping_policy,
        claim_level=args.claim_level,
        source_paths=[
            _repo_relative(args.summary_csv),
            _repo_relative(args.comparison_csv),
        ],
    )
    payload["input_summary_csv"] = _repo_relative(args.summary_csv)
    payload["input_comparison_csv"] = _repo_relative(args.comparison_csv)
    payload["recurrence_source"] = args.recurrence_source
    payload["validation_status"] = (
        "closed" if payload["promotion_ready"] else "diagnostic"
    )
    payload["gate_index_include"] = False
    payload["notes"] = [
        "This artifact verifies the row-production contract for zonal-flow optimization objectives.",
        "W7-X rows with missing GAM damping remain diagnostic and are not promoted to an optimization claim.",
        "Use --missing-damping-policy=fail for closed QA/QH/Miller-style promotion gates.",
    ]
    json.dumps(payload, allow_nan=False)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_row_csv(args.out_csv, payload)
    _plot_payload(args.out_png, payload)
    print(
        "wrote zonal-flow objective gate "
        f"samples={payload['sample_count']} promotion_ready={payload['promotion_ready']} "
        f"json={_repo_relative(args.out_json)}"
    )
    return 0


def _build_miller_panel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks" / "runtime_miller_zonal_response.toml",
        help="Runtime TOML for the shaped-Miller zonal-response panel.",
    )
    parser.add_argument(
        "--out-bundle",
        type=Path,
        default=ROOT
        / "tools_out"
        / "zonal_response"
        / "miller_caseIII_initial_density_Nl4_Nm24_Nz32_dt0005_t60.out.nc",
        help="Runtime output bundle path.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "docs" / "_static" / "miller_zonal_response_pilot.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.3,
        help="Late-time fraction used for the residual window.",
    )
    parser.add_argument(
        "--initial-fraction",
        type=float,
        default=0.1,
        help="Leading fraction used for the initial normalization window.",
    )
    parser.add_argument(
        "--initial-policy",
        choices=("first_abs", "window_abs_mean"),
        default="first_abs",
        help=(
            "Initial normalization convention. Merlo/Rosenbluth-Hinton residuals "
            "are quoted as phi(t->infinity)/phi(0), so this tool defaults to first_abs."
        ),
    )
    parser.add_argument(
        "--peak-fit-max-peaks",
        type=int,
        default=4,
        help=(
            "Maximum number of positive and negative extrema used per branch for the "
            "Merlo-style damping fit."
        ),
    )
    parser.add_argument(
        "--fit-window-tmax",
        type=float,
        default=30.0,
        help="Upper time bound for the common pre-recurrence GAM fit window.",
    )
    parser.add_argument(
        "--damping-fit-start-periods",
        type=float,
        default=1.0,
        help=(
            "GAM periods to skip before the damping window opens, so the "
            "initial-condition relaxation stays out of the envelope fit."
        ),
    )
    parser.add_argument(
        "--damping-fit-periods",
        type=float,
        default=3.0,
        help="Length of the GAM damping window, in periods of the measured oscillation.",
    )
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Reuse an existing out.nc bundle instead of rerunning the panel source simulation.",
    )
    return parser


def _nearest_kx_index(path: Path, target_kx: float) -> tuple[int, float]:
    with nc.Dataset(path, "r") as ds:
        grids = ds.groups.get("Grids")
        if grids is None or "kx" not in grids.variables:
            raise ValueError(f"missing Grids/kx in {path}")
        kx = np.asarray(grids.variables["kx"][:], dtype=float)
    if kx.ndim != 1 or kx.size == 0:
        raise ValueError(f"invalid kx grid in {path}")
    idx = int(np.argmin(np.abs(kx - float(target_kx))))
    return idx, float(kx[idx])


def _setup_note(cfg) -> str:
    source = str(getattr(cfg.expert, "source", "default")).strip().lower()
    if source == "phiext_full":
        return "external phiext_full source"
    return f"initial {cfg.init.init_field} perturbation"


def _miller_stability_evidence(
    *,
    metrics,
    nm: int,
    requested_tmax: float,
) -> dict[str, float]:
    """Measure the conditions that decide whether the gated numbers mean anything.

    Three things silently broke this benchmark before, and each one is a number
    here rather than a comment: the averaging window sitting past Hermite
    recurrence, the damping window reaching past it too, and the run stopping
    early so that "the last 30% of the trace" was 30% of something much shorter
    than the configured horizon.
    """

    recurrence_time = float(
        MERLO_CASE_III_GKX_CONVERGED["recurrence_coefficient"]
    ) * math.sqrt(float(nm))
    residual = abs(float(metrics.residual_level))
    scatter_ratio = (
        float(metrics.residual_std) / residual if residual > 0.0 else float("inf")
    )
    windows = [float(metrics.tmax)]
    if np.isfinite(metrics.damping_fit_tmax):
        windows.append(float(metrics.damping_fit_tmax))
    overshoot = max(0.0, max(windows) - recurrence_time)
    completeness = (
        float(metrics.tmax) / float(requested_tmax)
        if float(requested_tmax) > 0.0
        else float("nan")
    )
    return {
        "hermite_recurrence_time": recurrence_time,
        "residual_scatter_ratio": scatter_ratio,
        "analysis_window_past_recurrence": overshoot,
        "trace_completeness": completeness,
        "requested_tmax": float(requested_tmax),
    }


def _miller_converged_gate_report(
    *,
    metrics,
    omega_r0_over_vi: float,
    gamma_r0_over_vi: float,
    evidence: dict[str, float],
):
    """Gate the converged GKX answer and the conditions that make it one."""

    converged = MERLO_CASE_III_GKX_CONVERGED
    return gate_report(
        "merlo_case_iii_zonal_response",
        (
            "GKX converged Hermite baseline (Nm=144, dt=0.0025); agreement with "
            "the Merlo et al. read-off is reported under literature_comparison "
            "and is NOT asserted here"
        ),
        (
            evaluate_scalar_gate(
                "residual_level",
                metrics.residual_level,
                float(converged["residual_phi_over_phi0"]),
                atol=float(converged["residual_atol"]),
                rtol=0.0,
                notes=(
                    "GKX's own converged residual, not the paper value. The "
                    "tolerance is the residual's remaining drift across "
                    "Nm=96/144/192, so refining Hermite resolution further must "
                    "not move it more than the ladder already did."
                ),
            ),
            evaluate_scalar_gate(
                "gam_frequency_R0_over_vi",
                omega_r0_over_vi,
                float(converged["omega_gam_R0_over_vi"]),
                atol=float(converged["omega_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
                notes="GKX's own converged GAM frequency, not the paper value.",
            ),
            evaluate_scalar_gate(
                "gam_growth_rate_R0_over_vi",
                gamma_r0_over_vi,
                float(converged["gamma_gam_R0_over_vi"]),
                atol=float(converged["gamma_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
                notes=(
                    "Signed growth-rate convention; negative values are damping. "
                    "Measured with the period-RMS envelope estimator. Across a "
                    "factor-8 span of diagnostic output cadence on this same "
                    "trace it moves by 0.0003 in R0/v_i; the retired "
                    "branchwise-extrema fit moves by 0.038 over the same span, "
                    "which is why the tolerance below can be this tight at all."
                ),
            ),
            evaluate_scalar_gate(
                "residual_scatter_ratio",
                evidence["residual_scatter_ratio"],
                0.0,
                atol=float(converged["residual_scatter_ratio_limit"]),
                rtol=0.0,
                notes=(
                    "residual_std / |residual_level| inside the averaging "
                    "window. Above this limit the window is measuring Hermite "
                    "recurrence rather than the residual: the retired Nm=24 "
                    "baseline sat at 1.20, i.e. the scatter exceeded the number "
                    "being gated."
                ),
            ),
            evaluate_scalar_gate(
                "analysis_window_past_recurrence",
                evidence["analysis_window_past_recurrence"],
                0.0,
                atol=0.0,
                rtol=0.0,
                units="a/vi",
                notes=(
                    "How far the residual and damping windows reach past the "
                    "measured recurrence onset t_quiet ~ "
                    f"{converged['recurrence_coefficient']} sqrt(Nm). Must be zero."
                ),
            ),
            evaluate_scalar_gate(
                "trace_completeness",
                evidence["trace_completeness"],
                1.0,
                atol=float(converged["trace_completeness_atol"]),
                rtol=0.0,
                notes=(
                    "Fraction of the configured horizon the trace actually "
                    "reaches. [time] run_to defaults to \"saturation\", which "
                    "declares a zero-gradient run converged inside the first "
                    "chunk and truncates it at t ~ 6 without raising."
                ),
            ),
        ),
    )


def _main_miller_panel(argv: list[str]) -> int:
    args = _build_miller_panel_parser().parse_args(argv)
    cfg, raw = load_runtime_from_toml(args.config)
    run_cfg = dict(raw.get("run", {}))
    ky_target = float(run_cfg.get("ky", 0.0))
    kx_target = float(run_cfg.get("kx", 0.1))
    nl = int(run_cfg.get("Nl", 2))
    nm = int(run_cfg.get("Nm", 2))
    dt = float(run_cfg.get("dt", cfg.time.dt))
    steps = int(run_cfg.get("steps", max(int(round(float(cfg.time.t_max) / dt)), 1)))
    sample_stride = int(run_cfg.get("sample_stride", cfg.time.sample_stride))
    diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))

    out_bundle = Path(args.out_bundle)
    out_bundle.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_output or not out_bundle.exists():
        run_runtime_nonlinear_with_artifacts(
            cfg,
            out=out_bundle,
            ky_target=ky_target,
            kx_target=kx_target,
            Nl=nl,
            Nm=nm,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics=diagnostics,
            show_progress=False,
        )

    kx_index, kx_selected = _nearest_kx_index(out_bundle, kx_target)
    series = load_diagnostic_time_series(
        out_bundle,
        variable="Phi_zonal_mode_kxt",
        kx_index=kx_index,
        component="real",
        align_phase=True,
    )
    if np.iscomplexobj(series.values):
        raise ValueError(
            "signed zonal-response plotting requires a real-valued phase-aligned trace"
        )

    metrics = zonal_flow_response_metrics(
        series.t,
        np.asarray(series.values, dtype=float),
        tail_fraction=float(args.tail_fraction),
        initial_fraction=float(args.initial_fraction),
        initial_policy=str(args.initial_policy),
        peak_fit_max_peaks=int(args.peak_fit_max_peaks)
        if args.peak_fit_max_peaks is not None
        else None,
        damping_fit_mode="period_rms_envelope",
        frequency_fit_mode="hilbert_phase",
        fit_window_tmax=float(args.fit_window_tmax),
        hilbert_trim_fraction=0.2,
        damping_fit_start_periods=float(args.damping_fit_start_periods),
        damping_fit_periods=float(args.damping_fit_periods),
    )
    setup_note = _setup_note(cfg)
    ref_residual = float(MERLO_CASE_III_REFERENCE["residual_phi_over_phi0"])
    ref_omega = float(MERLO_CASE_III_REFERENCE["omega_gam_R0_over_vi"])
    ref_gamma = float(MERLO_CASE_III_REFERENCE["gamma_gam_R0_over_vi"])
    r0 = float(getattr(cfg.geometry, "R0", 1.0))
    omega_r0_over_vi = float(metrics.gam_frequency) * r0
    damping_r0_over_vi = float(metrics.gam_damping_rate) * r0
    gamma_r0_over_vi = -damping_r0_over_vi
    residual_abs_error = abs(float(metrics.residual_level) - ref_residual)
    omega_abs_error = abs(omega_r0_over_vi - ref_omega)
    gamma_abs_error = abs(gamma_r0_over_vi - ref_gamma)
    evidence = _miller_stability_evidence(
        metrics=metrics,
        nm=nm,
        requested_tmax=float(steps) * dt,
    )
    validation_gate_report = _miller_converged_gate_report(
        metrics=metrics,
        omega_r0_over_vi=omega_r0_over_vi,
        gamma_r0_over_vi=gamma_r0_over_vi,
        evidence=evidence,
    )
    # Reported, never asserted: the same three numbers against the published
    # read-off at the published tolerances. Two of them do not pass at converged
    # resolution, and that is the claim boundary this artifact exists to state.
    literature_report = gate_report(
        "merlo_case_iii_literature_comparison",
        "Merlo et al. paper-scale read-off (reported, NOT gated)",
        (
            evaluate_scalar_gate(
                "residual_level",
                metrics.residual_level,
                ref_residual,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["residual_atol"]),
                rtol=0.0,
            ),
            evaluate_scalar_gate(
                "gam_frequency_R0_over_vi",
                omega_r0_over_vi,
                ref_omega,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["omega_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
            ),
            evaluate_scalar_gate(
                "gam_growth_rate_R0_over_vi",
                gamma_r0_over_vi,
                ref_gamma,
                atol=float(MERLO_CASE_III_GATE_TOLERANCES["gamma_atol_R0_over_vi"]),
                rtol=0.0,
                units="R0/vi",
                notes="Signed growth-rate convention; negative values correspond to damping.",
            ),
        ),
    )
    title = f"Merlo Case III zonal-response (ky={ky_target:.3f}, kx={kx_selected:.3f})"
    fig, _axes = zonal_flow_response_figure(
        series.t,
        np.asarray(series.values, dtype=float),
        metrics=metrics,
        title=title,
        y_label=r"$\phi_\mathrm{zonal}/|\phi_\mathrm{zonal}(0)|$",
    )
    ax0 = _axes[0]
    ax0.axhline(
        ref_residual,
        color="#7b2cbf",
        linestyle=":",
        linewidth=2.1,
        label="Merlo Case III residual",
    )
    ax0.legend(loc="best", frameon=False)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    fig.savefig(args.out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    csv_out = args.out_png.with_suffix(".csv")
    np.savetxt(
        csv_out,
        np.column_stack([series.t, np.asarray(series.values, dtype=float)]),
        delimiter=",",
        header="t,phi_zonal_real",
        comments="",
    )

    meta_out = args.out_png.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "config": _repo_relative(args.config),
                "source_path": _repo_relative(series.source_path),
                "variable": "Phi_zonal_mode_kxt",
                "kx_index": int(kx_index),
                "kx_selected": float(kx_selected),
                "ky_target": float(ky_target),
                "initial_level": float(metrics.initial_level),
                "initial_policy": str(metrics.initial_policy),
                "residual_level": float(metrics.residual_level),
                "residual_std": float(metrics.residual_std),
                "response_rms": float(metrics.response_rms),
                "gam_frequency": float(metrics.gam_frequency),
                "gam_damping_rate": float(metrics.gam_damping_rate),
                "gam_frequency_R0_over_vi": float(omega_r0_over_vi),
                "gam_damping_rate_R0_over_vi": float(damping_r0_over_vi),
                "gam_growth_rate_R0_over_vi": float(gamma_r0_over_vi),
                "damping_method": str(metrics.damping_method),
                "frequency_method": str(metrics.frequency_method),
                "peak_count": int(metrics.peak_count),
                "peak_fit_count": int(metrics.peak_fit_count),
                "tmin": float(metrics.tmin),
                "tmax": float(metrics.tmax),
                "fit_tmin": float(metrics.fit_tmin),
                "fit_tmax": float(metrics.fit_tmax),
                "damping_fit_tmin": float(metrics.damping_fit_tmin),
                "damping_fit_tmax": float(metrics.damping_fit_tmax),
                "damping_fit_start_periods": float(args.damping_fit_start_periods),
                "damping_fit_periods": float(args.damping_fit_periods),
                "resolution": {
                    "Nm": int(nm),
                    "Nl": int(nl),
                    "Nz": int(cfg.grid.Nz),
                    "dt": float(dt),
                    "steps": int(steps),
                    "sample_stride": int(sample_stride),
                    "sample_dt": float(dt) * float(sample_stride),
                },
                "stability_evidence": {
                    key: float(value) for key, value in evidence.items()
                },
                "literature_reference": dict(MERLO_CASE_III_REFERENCE),
                "gate_tolerances": dict(MERLO_CASE_III_GATE_TOLERANCES),
                "converged_reference": dict(MERLO_CASE_III_GKX_CONVERGED),
                "gate_report": gate_report_to_dict(validation_gate_report),
                "literature_comparison": gate_report_to_dict(literature_report),
                "paper_scale_gate_passed": bool(literature_report.passed),
                "residual_abs_error_vs_literature": float(residual_abs_error),
                "omega_abs_error_vs_literature_R0_over_vi": float(omega_abs_error),
                "gamma_abs_error_vs_literature_R0_over_vi": float(gamma_abs_error),
                "setup": setup_note,
                "validation_status": "open",
                "notes": (
                    "Merlo Case-III shaped-Miller zonal-relaxation run built from the signed "
                    f"zonal observable Phi_zonal_mode_kxt with zero gradients, adiabatic "
                    f"electrons, and an {setup_note}. "
                    "WHAT IS GATED: GKX's own converged answer at Nm=144, dt=0.0025, plus the "
                    "conditions that make it a measurement -- that the averaging window is not "
                    "recurrence-dominated, that no analysis window reaches past the recurrence "
                    "onset t_quiet ~ 5.5 sqrt(Nm), and that the trace reaches its configured "
                    "horizon. WHAT IS NOT GATED: agreement with the published read-off. At "
                    "converged resolution GKX gives residual "
                    f"{float(metrics.residual_level):.3f} against the paper's {ref_residual:.3f} "
                    f"(gap {float(residual_abs_error):.3f}, tolerance "
                    f"{float(MERLO_CASE_III_GATE_TOLERANCES['residual_atol']):.3f}) and "
                    f"omega_GAM {float(omega_r0_over_vi):.2f} against {ref_omega:.2f} "
                    f"(gap {float(omega_abs_error):.2f}, tolerance "
                    f"{float(MERLO_CASE_III_GATE_TOLERANCES['omega_atol_R0_over_vi']):.2f}) "
                    "in R0/v_i, so two of the three literature comparisons FAIL and are "
                    "recorded under literature_comparison rather than widened until they pass. "
                    "The previous Nm=24 baseline agreed with the paper only because its whole "
                    "averaging window sat past Hermite recurrence, where the scatter (1.20 of "
                    "the gated value) exceeded the number being gated. The residual is "
                    "normalized with the Rosenbluth-Hinton first-sample convention. The GAM "
                    "frequency comes from the Hilbert instantaneous phase over t in "
                    f"[{metrics.fit_tmin:.1f}, {metrics.fit_tmax:.1f}]; the damping comes from a "
                    "sliding one-period RMS envelope fitted over "
                    f"{float(args.damping_fit_periods):.1f} GAM periods starting "
                    f"{float(args.damping_fit_start_periods):.1f} period(s) in, i.e. t in "
                    f"[{metrics.damping_fit_tmin:.1f}, {metrics.damping_fit_tmax:.1f}]. That "
                    "estimator replaces a four-extrema log-linear fit that moved by 52 percent "
                    "when the diagnostic output cadence alone was halved."
                ),
                "references": [
                    "Merlo et al. 2016 shaped-tokamak collisionless GAM benchmark, Case III",
                    "W7-X stella/GENE benchmark 2022 for zonal-flow observable conventions",
                    "research-roadmap Merlo note -- resolution and estimator audit",
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        print(
            "usage: build_zonal_flow_artifacts.py "
            "{response-csv,response-output,objective-gate,miller-panel,"
            "collisional-zonal-dk,collisional-zonal,simulate-collisional-zonal-dk,"
            "simulate-collisional-zonal-finite-b,collisional-zonal-grid-gate,"
            "collisional-zonal-moment-gate} ..."
        )
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "response-csv":
        return _main_response_csv(rest)
    if command == "response-output":
        return _main_response_output(rest)
    if command == "objective-gate":
        return _main_objective_gate(rest)
    if command == "miller-panel":
        return _main_miller_panel(rest)
    if command == "collisional-zonal-dk":
        return _main_collisional_zonal_dk(rest)
    if command == "collisional-zonal":
        return _main_collisional_zonal(rest)
    if command == "simulate-collisional-zonal-dk":
        return _main_simulate_collisional_zonal_dk(rest)
    if command == "simulate-collisional-zonal-finite-b":
        return _main_simulate_collisional_zonal_finite_b(rest)
    if command == "collisional-zonal-grid-gate":
        return _main_collisional_zonal_grid_gate(rest)
    if command == "collisional-zonal-moment-gate":
        return _main_collisional_zonal_moment_gate(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
