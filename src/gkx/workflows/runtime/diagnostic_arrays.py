"""Runtime diagnostic array validation, composition, and cost summaries.

These helpers operate on already-sampled diagnostic payloads. They deliberately
stay host-side: runtime drivers use them to fail fast on invalid artifacts, to
combine adaptive chunks without mixing that array bookkeeping into the
linear-fit and quasilinear-finalization code, and to summarise what a run's
simulated time cost before the payload is handed to the artifact writers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields as dataclass_fields
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from gkx.diagnostics import (
    ResolvedDiagnostics,
    SimulationDiagnostics,
    total_energy,
)
from gkx.diagnostics.analysis import (
    CFLLimiterReport,
    CFLScales,
    cfl_limiter_report,
    cfl_scales_from_array,
)

__all__ = [
    "TimestepCostReport",
    "concat_runtime_diagnostics",
    "slice_runtime_diagnostics",
    "stride_runtime_diagnostics",
    "timestep_cost_payload",
    "timestep_cost_report",
    "validate_finite_runtime_diagnostics",
]


def _first_nonfinite_sample(
    value: np.ndarray | jnp.ndarray, *, nsamples: int
) -> int | None:
    arr = np.asarray(value)
    if arr.size == 0 or np.isfinite(arr).all():
        return None
    if arr.ndim >= 1 and arr.shape[0] == nsamples:
        finite_by_sample = np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=1)
        bad = np.flatnonzero(~finite_by_sample)
        if bad.size:
            return int(bad[0])
    return 0


def validate_finite_runtime_diagnostics(
    diag: SimulationDiagnostics, *, label: str = "runtime"
) -> None:
    """Raise if a runtime diagnostic chunk contains NaN or infinite values.

    Long validation runs can otherwise continue for thousands of fixed steps
    after the first unstable sample. This host-side guard keeps the expensive
    artifact path fail-fast and reports the first offending diagnostic channel.
    """

    t_arr = np.asarray(diag.t, dtype=float)
    nsamples = int(t_arr.size)
    fields_to_check = [
        "t",
        "dt_t",
        "gamma_t",
        "omega_t",
        "Wg_t",
        "Wphi_t",
        "Wapar_t",
        "heat_flux_t",
        "particle_flux_t",
        "energy_t",
        "heat_flux_species_t",
        "particle_flux_species_t",
        "turbulent_heating_t",
        "turbulent_heating_species_t",
        "phi_mode_t",
    ]
    for name in fields_to_check:
        value = getattr(diag, name, None)
        if value is None:
            continue
        sample = _first_nonfinite_sample(value, nsamples=nsamples)
        if sample is None:
            continue
        t_text = ""
        if t_arr.size and sample < t_arr.size and np.isfinite(t_arr[sample]):
            t_text = f" at t={float(t_arr[sample]):.6g}"
        raise RuntimeError(
            f"{label} produced non-finite diagnostics in {name} at sample {sample}{t_text}"
        )

    if diag.resolved is None:
        return
    for field in dataclass_fields(ResolvedDiagnostics):
        value = getattr(diag.resolved, field.name)
        if value is None:
            continue
        sample = _first_nonfinite_sample(value, nsamples=nsamples)
        if sample is None:
            continue
        t_text = ""
        if t_arr.size and sample < t_arr.size and np.isfinite(t_arr[sample]):
            t_text = f" at t={float(t_arr[sample]):.6g}"
        raise RuntimeError(
            f"{label} produced non-finite diagnostics in resolved.{field.name} at sample {sample}{t_text}"
        )


def slice_runtime_diagnostics(
    diag: SimulationDiagnostics, stop: int
) -> SimulationDiagnostics:
    """Return the first ``stop`` diagnostic samples."""

    if stop < 0:
        raise ValueError("stop must be >= 0")

    def _slice_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        return np.asarray(arr)[:stop, ...]

    def _slice_resolved(
        resolved: ResolvedDiagnostics | None,
    ) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            value = getattr(resolved, field.name)
            payload[field.name] = (
                None if value is None else np.asarray(value)[:stop, ...]
            )
        return ResolvedDiagnostics(**payload)

    dt_t = np.asarray(diag.dt_t)[:stop]
    Wg_t = np.asarray(diag.Wg_t)[:stop]
    Wphi_t = np.asarray(diag.Wphi_t)[:stop]
    Wapar_t = np.asarray(diag.Wapar_t)[:stop]
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=np.asarray(diag.t)[:stop],
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=np.asarray(diag.gamma_t)[:stop],
        omega_t=np.asarray(diag.omega_t)[:stop],
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=np.asarray(diag.heat_flux_t)[:stop],
        particle_flux_t=np.asarray(diag.particle_flux_t)[:stop],
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ),
        heat_flux_species_t=_slice_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_slice_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_slice_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_slice_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_slice_optional(diag.phi_mode_t),
        # Run-constant, so it survives slicing/striding untouched.
        cfl_scales=diag.cfl_scales,
        resolved=_slice_resolved(diag.resolved),
    )


def stride_runtime_diagnostics(
    diag: SimulationDiagnostics,
    *,
    stride: int,
    offset: int = 0,
    keep_last: bool = False,
) -> SimulationDiagnostics:
    """Apply the runtime output stride to chunk diagnostics.

    ``offset`` selects the first sample kept, which is what lets the stride be
    applied per chunk instead of after concatenation. Striding a chunk that
    starts at global sample ``g`` with ``offset = (-g) % stride`` keeps exactly
    the global indices ``0, stride, 2*stride, ...`` -- the same samples the
    post-concatenation stride would have kept, at a fraction of the peak memory,
    because the discarded samples are never accumulated in the first place.
    """

    stride_use = int(max(stride, 1))
    offset_use = int(offset) % stride_use
    if stride_use == 1:
        return diag
    indices = np.arange(offset_use, np.asarray(diag.t).size, stride_use)
    if keep_last and (indices.size == 0 or indices[-1] != np.asarray(diag.t).size - 1):
        indices = np.append(indices, np.asarray(diag.t).size - 1)
    def _take(arr: np.ndarray | jnp.ndarray) -> np.ndarray:
        return np.asarray(arr)[indices, ...].copy()

    def _stride_optional(arr: np.ndarray | jnp.ndarray | None) -> np.ndarray | None:
        return None if arr is None else _take(arr)

    def _stride_resolved(
        resolved: ResolvedDiagnostics | None,
    ) -> ResolvedDiagnostics | None:
        if resolved is None:
            return None
        payload: dict[str, np.ndarray | None] = {
            field.name: _stride_optional(getattr(resolved, field.name))
            for field in dataclass_fields(ResolvedDiagnostics)
        }
        return ResolvedDiagnostics(**payload)

    dt_t = _take(diag.dt_t)
    Wg_t = _take(diag.Wg_t)
    Wphi_t = _take(diag.Wphi_t)
    Wapar_t = _take(diag.Wapar_t)
    if dt_t.size == 0:
        dt_mean = np.asarray(0.0, dtype=float)
    else:
        dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=_take(diag.t),
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=_take(diag.gamma_t),
        omega_t=_take(diag.omega_t),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=_take(diag.heat_flux_t),
        particle_flux_t=_take(diag.particle_flux_t),
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ).copy(),
        heat_flux_species_t=_stride_optional(diag.heat_flux_species_t),
        particle_flux_species_t=_stride_optional(diag.particle_flux_species_t),
        turbulent_heating_t=_stride_optional(diag.turbulent_heating_t),
        turbulent_heating_species_t=_stride_optional(diag.turbulent_heating_species_t),
        phi_mode_t=_stride_optional(diag.phi_mode_t),
        cfl_scales=diag.cfl_scales,
        resolved=_stride_resolved(diag.resolved),
    )


def concat_runtime_diagnostics(
    diags: Sequence[SimulationDiagnostics],
) -> SimulationDiagnostics:
    """Concatenate one or more diagnostic chunks."""

    if not diags:
        raise ValueError("at least one diagnostic chunk is required")

    def _concat(name: str) -> np.ndarray:
        return np.concatenate(
            [np.asarray(getattr(diag, name)) for diag in diags], axis=0
        )

    def _concat_optional(name: str) -> np.ndarray | None:
        values = [getattr(diag, name) for diag in diags]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                f"inconsistent optional diagnostic {name}: every concatenated chunk must either provide it or omit it"
            )
        return np.concatenate(
            [np.asarray(value) for value in values if value is not None], axis=0
        )

    def _concat_resolved() -> ResolvedDiagnostics | None:
        values = [diag.resolved for diag in diags]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(
                "inconsistent resolved diagnostics: every concatenated chunk must either provide resolved data or omit it"
            )
        payload: dict[str, np.ndarray | None] = {}
        for field in dataclass_fields(ResolvedDiagnostics):
            series = [
                None if value is None else getattr(value, field.name)
                for value in values
            ]
            if all(item is None for item in series):
                payload[field.name] = None
            elif any(item is None for item in series):
                raise ValueError(
                    f"inconsistent resolved diagnostic {field.name}: every concatenated chunk must either provide it or omit it"
                )
            else:
                payload[field.name] = np.concatenate(
                    [np.asarray(item) for item in series if item is not None],
                    axis=0,
                )
        return ResolvedDiagnostics(**payload)

    dt_t = _concat("dt_t")
    Wg_t = _concat("Wg_t")
    Wphi_t = _concat("Wphi_t")
    Wapar_t = _concat("Wapar_t")
    dt_mean = np.asarray(np.mean(dt_t), dtype=float)
    return SimulationDiagnostics(
        t=_concat("t"),
        dt_t=dt_t,
        dt_mean=dt_mean,
        gamma_t=_concat("gamma_t"),
        omega_t=_concat("omega_t"),
        Wg_t=Wg_t,
        Wphi_t=Wphi_t,
        Wapar_t=Wapar_t,
        heat_flux_t=_concat("heat_flux_t"),
        particle_flux_t=_concat("particle_flux_t"),
        energy_t=np.asarray(
            total_energy(jnp.asarray(Wg_t), jnp.asarray(Wphi_t), jnp.asarray(Wapar_t))
        ),
        heat_flux_species_t=_concat_optional("heat_flux_species_t"),
        particle_flux_species_t=_concat_optional("particle_flux_species_t"),
        turbulent_heating_t=_concat_optional("turbulent_heating_t"),
        turbulent_heating_species_t=_concat_optional("turbulent_heating_species_t"),
        phi_mode_t=_concat_optional("phi_mode_t"),
        cfl_scales=next(
            (d.cfl_scales for d in diags if d.cfl_scales is not None), None
        ),
        resolved=_concat_resolved(),
    )


@dataclass(frozen=True)
class TimestepCostReport:
    """Cost per unit of simulated time, and the dt trajectory behind it."""

    n_samples: int
    t_start: float
    t_end: float
    t_span: float
    steps: float
    steps_are_exact: bool
    steps_per_unit_time: float
    wall_seconds: float | None
    wall_seconds_per_unit_time: float | None
    dt_initial: float
    dt_min: float
    dt_max: float
    dt_final: float
    dt_collapse_ratio: float
    steps_per_unit_time_first_quarter: float
    steps_per_unit_time_last_quarter: float
    cost_growth_ratio: float
    cfl: CFLLimiterReport | None
    notes: tuple[str, ...]


def _steps_over(t: np.ndarray, dt: np.ndarray) -> float:
    """Steps across the recorded samples, as the sum of gap over step size.

    ``dt[i]`` is the step that produced ``t[i]``, so on an unstrided series
    every term is exactly one and the sum is ``len(t) - 1``.
    """

    return 0.0 if t.size < 2 else float(np.sum(np.diff(t) / dt[1:]))


def _rate_in_window(t: np.ndarray, dt: np.ndarray, lo: float, hi: float) -> float:
    """Steps per unit simulated time over the ``[lo, hi]`` slice of a run."""

    mask = (t >= lo) & (t <= hi)
    t_win, dt_win = t[mask], dt[mask]
    span = float(t_win[-1] - t_win[0]) if t_win.size >= 2 else 0.0
    return _steps_over(t_win, dt_win) / span if span > 0.0 else float("nan")


def _timestep_cost_notes(
    n_samples: int, *, exact: bool, limiter: CFLLimiterReport | None
) -> tuple[str, ...]:
    """Non-fatal observations worth surfacing next to the cost numbers."""

    notes: list[str] = []
    if not exact:
        notes.append(
            "dt series is strided: the step count is an estimate that charges "
            "each gap to its final step size"
        )
    if limiter is not None and limiter.samples_at_dt_floor:
        notes.append(
            f"dt was pinned at the configured floor for "
            f"{limiter.samples_at_dt_floor} of {n_samples} samples: the "
            "requested CFL condition was not honoured there"
        )
    if limiter is not None and limiter.samples_attributed == 0:
        notes.append(
            "every dt sample sat at the configured ceiling, so this run was "
            "capped by dt_max rather than limited by any CFL term"
        )
    return tuple(notes)


def timestep_cost_report(
    t: np.ndarray,
    dt: np.ndarray,
    *,
    wall_seconds: float | None = None,
    scales: CFLScales | None = None,
) -> TimestepCostReport:
    """Summarise what one unit of simulated time cost, and why.

    Nothing here gates. Within a healthy nonlinear run dt collapses by
    construction as the initial condition saturates, so a collapse-ratio limit
    would fire on every well-behaved surface; the ratio is reported and left to
    the reader. The one note emitted is objective rather than tuned: a step
    pinned at ``dt_min`` means the requested CFL condition was not honoured.
    """

    t_arr = np.asarray(t, dtype=float).reshape(-1)
    dt_arr = np.asarray(dt, dtype=float).reshape(-1)
    if t_arr.size != dt_arr.size:
        raise ValueError("t and dt must have the same length")
    keep = np.isfinite(t_arr) & np.isfinite(dt_arr) & (dt_arr > 0.0)
    t_arr, dt_arr = t_arr[keep], dt_arr[keep]
    if t_arr.size == 0:
        raise ValueError("t and dt must contain at least one usable sample")
    span = float(t_arr[-1] - t_arr[0])
    steps = _steps_over(t_arr, dt_arr)
    # Compare against the unstrided answer directly, with a tolerance loose
    # enough for a float32 cumulative time axis but far tighter than the
    # factor-of-stride gap a strided series would show.
    exact = bool(
        t_arr.size < 2 or abs(steps - (t_arr.size - 1)) <= 1e-3 * (t_arr.size - 1)
    )
    quarter = span / 4.0
    early = _rate_in_window(t_arr, dt_arr, t_arr[0], t_arr[0] + quarter)
    late = _rate_in_window(t_arr, dt_arr, t_arr[-1] - quarter, t_arr[-1])
    limiter = None if scales is None else cfl_limiter_report(dt_arr, scales)
    wall = None if wall_seconds is None else float(wall_seconds)
    return TimestepCostReport(
        n_samples=int(t_arr.size),
        t_start=float(t_arr[0]),
        t_end=float(t_arr[-1]),
        t_span=span,
        steps=steps,
        steps_are_exact=exact,
        steps_per_unit_time=steps / span if span > 0.0 else float("nan"),
        wall_seconds=wall,
        wall_seconds_per_unit_time=(
            wall / span if wall is not None and span > 0.0 else None
        ),
        dt_initial=float(dt_arr[0]),
        dt_min=float(np.min(dt_arr)),
        dt_max=float(np.max(dt_arr)),
        dt_final=float(dt_arr[-1]),
        dt_collapse_ratio=float(dt_arr[0]) / float(np.min(dt_arr)),
        steps_per_unit_time_first_quarter=early,
        steps_per_unit_time_last_quarter=late,
        cost_growth_ratio=late / early if early > 0.0 else float("nan"),
        cfl=limiter,
        notes=_timestep_cost_notes(int(t_arr.size), exact=exact, limiter=limiter),
    )


def timestep_cost_payload(
    diag: SimulationDiagnostics, *, wall_seconds: float | None = None
) -> dict[str, Any]:
    """Return the JSON-ready cost block for one run's recorded diagnostics.

    Returns an empty payload rather than raising when the series is unusable,
    so a summary artifact never fails to write because of a diagnostic field.
    """

    try:
        report = timestep_cost_report(
            np.asarray(diag.t),
            np.asarray(diag.dt_t),
            wall_seconds=wall_seconds,
            scales=cfl_scales_from_array(diag.cfl_scales),
        )
    except ValueError:
        return {}
    payload = _json_safe(asdict(report))
    payload["notes"] = list(report.notes)
    return payload


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the summary stays valid JSON.

    ``NaN`` and ``Infinity`` are Python-json extensions that strict parsers
    reject, and the cost block is deliberately full of ratios that are
    undefined on a degenerate run.
    """

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value
