"""High-level benchmark helpers for scans and eigenfunction extraction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import netCDF4 as nc
import numpy as np

from spectraxgk.analysis import (
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
)
from spectraxgk.benchmarks import LinearRunResult, LinearScanResult
from spectraxgk.grids import SpectralGrid, build_spectral_grid


@dataclass(frozen=True)
class ScanAndModeResult:
    scan: LinearScanResult
    eigenfunction: np.ndarray
    grid: SpectralGrid
    ky_selected: float
    tmin: float | None
    tmax: float | None


@dataclass(frozen=True)
class LateTimeLinearMetrics:
    """Late-time growth/frequency metrics for a linear run."""

    gamma_fit: float
    omega_fit: float
    gamma_tail_mean: float
    omega_tail_mean: float
    gamma_tail_std: float
    omega_tail_std: float
    tmin: float | None
    tmax: float | None
    nsamples: int
    signal_source: str


@dataclass(frozen=True)
class NonlinearWindowMetrics:
    """Windowed transport/envelope metrics for a nonlinear run."""

    tmin: float
    tmax: float
    nsamples: int
    heat_flux_mean: float
    heat_flux_std: float
    heat_flux_rms: float
    wphi_mean: float
    wphi_std: float
    wg_mean: float
    wg_std: float
    phi_mode_envelope_mean: float | None
    phi_mode_envelope_std: float | None
    phi_mode_envelope_max: float | None


@dataclass(frozen=True)
class ZonalFlowResponseMetrics:
    """Late-time residual and GAM-envelope metrics for zonal-flow responses."""

    initial_level: float
    initial_policy: str
    residual_level: float
    residual_std: float
    response_rms: float
    gam_frequency: float
    gam_damping_rate: float
    damping_method: str
    frequency_method: str
    peak_count: int
    peak_fit_count: int
    tmin: float
    tmax: float
    fit_tmin: float
    fit_tmax: float
    peak_times: np.ndarray
    peak_envelope: np.ndarray
    max_peak_times: np.ndarray
    max_peak_values: np.ndarray
    min_peak_times: np.ndarray
    min_peak_values: np.ndarray


@dataclass(frozen=True)
class ObservedOrderMetrics:
    """Observed-order convergence summary from step sizes and errors."""

    step_sizes: np.ndarray
    errors: np.ndarray
    orders: np.ndarray
    asymptotic_order: float


@dataclass(frozen=True)
class BranchContinuationMetrics:
    """Continuity summary for a scanned linear branch."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    rel_gamma_jumps: np.ndarray
    rel_omega_jumps: np.ndarray
    max_rel_gamma_jump: float
    max_rel_omega_jump: float
    min_successive_overlap: float | None


@dataclass(frozen=True)
class ScalarGateResult:
    """Pass/fail result for one benchmark observable.

    The tolerance convention follows ``numpy.isclose``: a metric passes when
    ``abs_error <= atol + rtol * abs(reference)``. This keeps near-zero
    frequency and marginal-growth gates explicit through ``atol`` rather than
    hiding them behind unstable relative errors.
    """

    metric: str
    observed: float
    reference: float
    abs_error: float
    rel_error: float
    atol: float
    rtol: float
    passed: bool
    units: str
    notes: str


@dataclass(frozen=True)
class GateReport:
    """Collection of scalar gates for one validation artifact."""

    case: str
    source: str
    gates: tuple[ScalarGateResult, ...]
    passed: bool
    max_abs_error: float
    max_rel_error: float


@dataclass(frozen=True)
class EigenfunctionComparisonMetrics:
    """Phase-aligned eigenfunction comparison summary."""

    overlap: float
    relative_l2: float
    phase_shift: float


@dataclass(frozen=True)
class EigenfunctionReferenceBundle:
    """Frozen reference eigenfunction bundle for manuscript-grade overlays."""

    theta: np.ndarray
    mode: np.ndarray
    source: str
    case: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class DiagnosticTimeSeries:
    """Single benchmark-facing time series loaded from an ``out.nc`` artifact."""

    t: np.ndarray
    values: np.ndarray
    variable: str
    source_path: str


def evaluate_scalar_gate(
    metric: str,
    observed: float,
    reference: float,
    *,
    atol: float,
    rtol: float,
    units: str = "",
    notes: str = "",
) -> ScalarGateResult:
    """Evaluate one scalar benchmark gate.

    Use this helper for publication-facing metrics such as growth rates,
    frequencies, windowed heat fluxes, zonal residuals, and damping rates. The
    explicit ``atol``/``rtol`` pair forces each artifact to document whether its
    tolerance is absolute, relative, or both.
    """

    obs = float(observed)
    ref = float(reference)
    atol_f = float(atol)
    rtol_f = float(rtol)
    if atol_f < 0.0 or rtol_f < 0.0:
        raise ValueError("atol and rtol must be non-negative")
    abs_error = float(abs(obs - ref)) if np.isfinite(obs) and np.isfinite(ref) else float("inf")
    if np.isfinite(ref) and abs(ref) > 0.0:
        rel_error = float(abs_error / abs(ref))
    else:
        rel_error = 0.0 if abs_error == 0.0 else float("inf")
    tolerance = atol_f + rtol_f * abs(ref)
    passed = bool(np.isfinite(obs) and np.isfinite(ref) and abs_error <= tolerance)
    return ScalarGateResult(
        metric=str(metric),
        observed=obs,
        reference=ref,
        abs_error=abs_error,
        rel_error=rel_error,
        atol=atol_f,
        rtol=rtol_f,
        passed=passed,
        units=str(units),
        notes=str(notes),
    )


def gate_report(case: str, source: str, gates: list[ScalarGateResult] | tuple[ScalarGateResult, ...]) -> GateReport:
    """Summarize a set of scalar gates for one artifact."""

    gate_tuple = tuple(gates)
    if not gate_tuple:
        raise ValueError("gate report requires at least one scalar gate")
    finite_abs = [gate.abs_error for gate in gate_tuple if np.isfinite(gate.abs_error)]
    finite_rel = [gate.rel_error for gate in gate_tuple if np.isfinite(gate.rel_error)]
    return GateReport(
        case=str(case),
        source=str(source),
        gates=gate_tuple,
        passed=all(gate.passed for gate in gate_tuple),
        max_abs_error=float(max(finite_abs)) if finite_abs else float("inf"),
        max_rel_error=float(max(finite_rel)) if finite_rel else float("inf"),
    )


def gate_report_to_dict(report: GateReport) -> dict[str, object]:
    """Return a JSON-serializable representation of a gate report."""

    return {
        "case": report.case,
        "source": report.source,
        "passed": bool(report.passed),
        "max_abs_error": float(report.max_abs_error),
        "max_rel_error": float(report.max_rel_error),
        "gates": [
            {
                "metric": gate.metric,
                "observed": float(gate.observed),
                "reference": float(gate.reference),
                "abs_error": float(gate.abs_error),
                "rel_error": float(gate.rel_error),
                "atol": float(gate.atol),
                "rtol": float(gate.rtol),
                "passed": bool(gate.passed),
                "units": gate.units,
                "notes": gate.notes,
            }
            for gate in report.gates
        ],
    }


def linear_metrics_gate_report(
    observed: LateTimeLinearMetrics,
    reference: LateTimeLinearMetrics,
    *,
    case: str,
    source: str,
    gamma_atol: float = 0.0,
    gamma_rtol: float = 0.05,
    omega_atol: float = 0.0,
    omega_rtol: float = 0.05,
) -> GateReport:
    """Gate late-time linear growth and frequency metrics."""

    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "gamma_fit",
                observed.gamma_fit,
                reference.gamma_fit,
                atol=gamma_atol,
                rtol=gamma_rtol,
                units="v_t/R",
            ),
            evaluate_scalar_gate(
                "omega_fit",
                observed.omega_fit,
                reference.omega_fit,
                atol=omega_atol,
                rtol=omega_rtol,
                units="v_t/R",
            ),
        ),
    )


def nonlinear_window_gate_report(
    observed: NonlinearWindowMetrics,
    reference: NonlinearWindowMetrics,
    *,
    case: str,
    source: str,
    rtol: float = 0.1,
    atol: float = 0.0,
    include_envelope: bool = True,
) -> GateReport:
    """Gate windowed nonlinear transport and field-energy metrics."""

    metrics = ("heat_flux_mean", "heat_flux_rms", "wphi_mean", "wg_mean")
    gates: list[ScalarGateResult] = []
    for metric in metrics:
        gates.append(
            evaluate_scalar_gate(
                metric,
                getattr(observed, metric),
                getattr(reference, metric),
                atol=atol,
                rtol=rtol,
            )
        )
    if (
        include_envelope
        and observed.phi_mode_envelope_mean is not None
        and reference.phi_mode_envelope_mean is not None
    ):
        gates.append(
            evaluate_scalar_gate(
                "phi_mode_envelope_mean",
                observed.phi_mode_envelope_mean,
                reference.phi_mode_envelope_mean,
                atol=atol,
                rtol=rtol,
            )
        )
    return gate_report(case, source, gates)


def zonal_response_gate_report(
    observed: ZonalFlowResponseMetrics,
    reference: ZonalFlowResponseMetrics,
    *,
    case: str,
    source: str,
    residual_atol: float,
    residual_rtol: float = 0.0,
    frequency_atol: float,
    frequency_rtol: float = 0.0,
    damping_atol: float,
    damping_rtol: float = 0.0,
) -> GateReport:
    """Gate Rosenbluth-Hinton/GAM-style response observables."""

    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "residual_level",
                observed.residual_level,
                reference.residual_level,
                atol=residual_atol,
                rtol=residual_rtol,
            ),
            evaluate_scalar_gate(
                "gam_frequency",
                observed.gam_frequency,
                reference.gam_frequency,
                atol=frequency_atol,
                rtol=frequency_rtol,
                units="v_t/R",
            ),
            evaluate_scalar_gate(
                "gam_damping_rate",
                observed.gam_damping_rate,
                reference.gam_damping_rate,
                atol=damping_atol,
                rtol=damping_rtol,
                units="v_t/R",
            ),
        ),
    )


def eigenfunction_gate_report(
    comparison: EigenfunctionComparisonMetrics,
    *,
    case: str,
    source: str,
    min_overlap: float = 0.95,
    max_relative_l2: float = 0.25,
) -> GateReport:
    """Gate a phase-aligned eigenfunction comparison.

    The ideal reference is overlap equal to one and relative L2 mismatch equal
    to zero. ``min_overlap`` and ``max_relative_l2`` make the acceptance policy
    explicit for manuscript overlays and branch-identity checks.
    """

    min_overlap_f = float(min_overlap)
    max_relative_l2_f = float(max_relative_l2)
    if not 0.0 <= min_overlap_f <= 1.0:
        raise ValueError("min_overlap must be in [0, 1]")
    if max_relative_l2_f < 0.0:
        raise ValueError("max_relative_l2 must be non-negative")
    return gate_report(
        case,
        source,
        (
            evaluate_scalar_gate(
                "eigenfunction_overlap",
                comparison.overlap,
                1.0,
                atol=1.0 - min_overlap_f,
                rtol=0.0,
                notes=f"Passes when overlap >= {min_overlap_f:.6g}.",
            ),
            evaluate_scalar_gate(
                "eigenfunction_relative_l2",
                comparison.relative_l2,
                0.0,
                atol=max_relative_l2_f,
                rtol=0.0,
                notes=f"Passes when relative L2 <= {max_relative_l2_f:.6g}.",
            ),
        ),
    )


def normalize_eigenfunction(eigenfunction: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Normalize an eigenfunction by its value at theta=0 (nearest z=0)."""

    idx = int(np.argmin(np.abs(z)))
    scale = eigenfunction[idx]
    if scale == 0:
        return eigenfunction
    return eigenfunction / scale


def phase_align_eigenfunction(eigenfunction: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, float]:
    """Phase-align ``eigenfunction`` to ``reference`` using the global complex phase."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    phase = np.vdot(lhs, rhs)
    if abs(phase) <= 1.0e-30:
        return lhs, 0.0
    phase_shift = float(np.angle(phase))
    return lhs * np.exp(1j * phase_shift), phase_shift


def compare_eigenfunctions(eigenfunction: np.ndarray, reference: np.ndarray) -> EigenfunctionComparisonMetrics:
    """Return normalized overlap and relative L2 error after global phase alignment."""

    lhs = np.asarray(eigenfunction, dtype=np.complex128)
    rhs = np.asarray(reference, dtype=np.complex128)
    if lhs.shape != rhs.shape:
        raise ValueError("eigenfunction and reference must have the same shape")
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return EigenfunctionComparisonMetrics(
            overlap=float("nan"),
            relative_l2=float("nan"),
            phase_shift=0.0,
        )
    lhs_aligned, phase_shift = phase_align_eigenfunction(lhs, rhs)
    overlap = float(np.abs(np.vdot(lhs, rhs)) / (lhs_norm * rhs_norm))
    rel_l2 = float(np.linalg.norm(lhs_aligned - rhs) / rhs_norm)
    return EigenfunctionComparisonMetrics(
        overlap=overlap,
        relative_l2=rel_l2,
        phase_shift=phase_shift,
    )


def save_eigenfunction_reference_bundle(
    path: str | Path,
    *,
    theta: np.ndarray,
    mode: np.ndarray,
    source: str,
    case: str,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write a frozen reference eigenfunction bundle as ``.npz``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        theta=np.asarray(theta, dtype=float),
        mode=np.asarray(mode, dtype=np.complex128),
        source=np.asarray(str(source)),
        case=np.asarray(str(case)),
        metadata_json=np.asarray(json.dumps(metadata or {}, sort_keys=True)),
    )
    return out


def load_eigenfunction_reference_bundle(path: str | Path) -> EigenfunctionReferenceBundle:
    """Load a frozen reference eigenfunction bundle."""

    data = np.load(Path(path), allow_pickle=False)
    metadata_json = str(np.asarray(data["metadata_json"]).item()) if "metadata_json" in data else "{}"
    return EigenfunctionReferenceBundle(
        theta=np.asarray(data["theta"], dtype=float),
        mode=np.asarray(data["mode"], dtype=np.complex128),
        source=str(np.asarray(data["source"]).item()),
        case=str(np.asarray(data["case"]).item()),
        metadata=json.loads(metadata_json),
    )


def load_diagnostic_time_series(
    path: str | Path,
    *,
    variable: str,
    diagnostics_group: str = "Diagnostics",
    time_group: str = "Grids",
    time_var: str = "time",
    kx_index: int | None = None,
    component: str = "real",
    align_phase: bool = False,
) -> DiagnosticTimeSeries:
    """Load a 1D diagnostics time series from a GX-style ``out.nc`` artifact."""

    src = Path(path)
    with nc.Dataset(src) as ds:
        diag_group = ds.groups.get(diagnostics_group)
        if diag_group is None:
            raise ValueError(f"missing NetCDF group {diagnostics_group!r} in {src}")
        if variable not in diag_group.variables:
            raise ValueError(f"missing diagnostics variable {variable!r} in {src}")
        var = diag_group.variables[variable]
        raw = np.asarray(var[:])
        dims = tuple(getattr(var, "dimensions", ()))
        if dims and dims[-1] == "ri":
            raw = np.asarray(raw[..., 0] + 1j * raw[..., 1], dtype=np.complex128)
        if raw.ndim == 1:
            values = raw
        elif raw.ndim == 2:
            if kx_index is None:
                raise ValueError(f"diagnostics variable {variable!r} requires kx_index for 2D extraction")
            values = raw[:, int(kx_index)]
        else:
            raise ValueError(f"diagnostics variable {variable!r} must reduce to a 1D time series")

        if time_group in ds.groups and time_var in ds.groups[time_group].variables:
            t = np.asarray(ds.groups[time_group].variables[time_var][:], dtype=float)
        elif time_var in ds.variables:
            t = np.asarray(ds.variables[time_var][:], dtype=float)
        else:
            raise ValueError(f"missing time variable {time_group}/{time_var} in {src}")

    if t.ndim != 1 or t.size != values.size:
        raise ValueError(f"time axis for {variable!r} must be one-dimensional and match the diagnostics length")

    values_arr = np.asarray(values)
    if np.iscomplexobj(values_arr):
        if align_phase:
            finite = np.isfinite(values_arr)
            nz = finite & (np.abs(values_arr) > 1.0e-30)
            if np.any(nz):
                first = values_arr[np.flatnonzero(nz)[0]]
                values_arr = values_arr * np.exp(-1j * np.angle(first))
        component_key = str(component).lower()
        if component_key == "complex":
            selected = values_arr
        elif component_key == "real":
            selected = np.real(values_arr)
        elif component_key == "imag":
            selected = np.imag(values_arr)
        elif component_key == "abs":
            selected = np.abs(values_arr)
        else:
            raise ValueError("component must be one of {'real', 'imag', 'abs', 'complex'}")
    else:
        if component not in {"real", "abs"}:
            raise ValueError("real diagnostics only support component='real' or 'abs'")
        selected = np.abs(values_arr) if component == "abs" else np.asarray(values_arr, dtype=float)

    return DiagnosticTimeSeries(
        t=t,
        values=np.asarray(selected),
        variable=str(variable),
        source_path=str(src),
    )


def _tail_window(t: np.ndarray, tail_fraction: float) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(tail_fraction) <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    start = max(0, int(np.floor((1.0 - float(tail_fraction)) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[start:] = True
    if not np.any(mask):
        mask[-1] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def late_time_window(t: np.ndarray, *, tail_fraction: float = 0.4) -> tuple[float, float]:
    """Return the start/end of a late-time tail window.

    This is the windowing convention used for manuscript-facing eigenfunction
    extraction when the growth-rate fit window is not the same object as the
    late-time mode-shape window.
    """

    _mask, tmin, tmax = _tail_window(np.asarray(t, dtype=float), float(tail_fraction))
    if tmin is None or tmax is None:
        raise ValueError("late-time window requires a non-empty time axis")
    return float(tmin), float(tmax)


def infer_triple_dealiased_ny(nky_positive: int) -> int:
    """Infer the full ``Ny`` from the number of positive ``k_y`` points.

    GX-style reference outputs typically store only the non-negative
    ``k_y`` branch. For the linked-boundary spectral grid used here, the
    corresponding real-space ``Ny`` follows ``Ny = 3 * (nky - 1) + 1``.
    """

    nky = int(nky_positive)
    if nky < 2:
        raise ValueError("nky_positive must be >= 2")
    return 3 * (nky - 1) + 1


def _tail_stats(arr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _leading_window(
    t: np.ndarray,
    lead_fraction: float,
) -> tuple[np.ndarray, float | None, float | None]:
    if not 0.0 < float(lead_fraction) <= 1.0:
        raise ValueError("lead_fraction must be in (0, 1]")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")
    if t.size == 0:
        raise ValueError("t must be non-empty")
    stop = max(1, int(np.ceil(float(lead_fraction) * t.size)))
    mask = np.zeros_like(t, dtype=bool)
    mask[:stop] = True
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]) if tt.size else None, float(tt[-1]) if tt.size else None


def _explicit_time_window(
    t: np.ndarray,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    mask = np.ones_like(t, dtype=bool)
    if tmin is not None:
        mask &= t >= float(tmin)
    if tmax is not None:
        mask &= t <= float(tmax)
    if not np.any(mask):
        raise ValueError("explicit fit window is empty")
    tt = np.asarray(t[mask], dtype=float)
    return mask, float(tt[0]), float(tt[-1])


def _analytic_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("signal must be a non-empty one-dimensional array")
    spec = np.fft.fft(x)
    filt = np.zeros(x.size, dtype=float)
    if x.size % 2 == 0:
        filt[0] = 1.0
        filt[x.size // 2] = 1.0
        filt[1 : x.size // 2] = 2.0
    else:
        filt[0] = 1.0
        filt[1 : (x.size + 1) // 2] = 2.0
    return np.fft.ifft(spec * filt)


def zonal_flow_response_metrics(
    t: np.ndarray,
    response: np.ndarray,
    *,
    tail_fraction: float = 0.3,
    initial_fraction: float = 0.1,
    initial_policy: str = "window_abs_mean",
    peak_fit_max_peaks: int | None = None,
    damping_fit_mode: str = "combined_envelope",
    frequency_fit_mode: str = "peak_spacing",
    fit_window_tmin: float | None = None,
    fit_window_tmax: float | None = None,
    hilbert_trim_fraction: float = 0.2,
) -> ZonalFlowResponseMetrics:
    """Estimate residual level and GAM envelope metrics from a zonal response.

    The input ``response`` should be a scalar zonal observable such as zonal
    potential or a normalized zonal-energy proxy on a uniform time trace.
    ``initial_policy="first_abs"`` follows Rosenbluth-Hinton/GAM convention by
    normalizing to the initial potential magnitude; ``"window_abs_mean"`` keeps
    the older robust behavior for generic noisy traces.
    """

    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError("t and response must be one-dimensional arrays of equal length")
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four samples")

    finite = np.isfinite(t_arr) & np.isfinite(resp)
    t_arr = t_arr[finite]
    resp = resp[finite]
    if t_arr.size < 4:
        raise ValueError("zonal-flow response requires at least four finite samples")

    policy = str(initial_policy).strip().lower().replace("-", "_")
    if policy not in {"window_abs_mean", "first_abs"}:
        raise ValueError("initial_policy must be one of {'window_abs_mean', 'first_abs'}")
    if peak_fit_max_peaks is not None and int(peak_fit_max_peaks) <= 0:
        raise ValueError("peak_fit_max_peaks must be > 0 when provided")
    damping_mode = str(damping_fit_mode).strip().lower().replace("-", "_")
    if damping_mode not in {"combined_envelope", "branchwise_extrema"}:
        raise ValueError("damping_fit_mode must be one of {'combined_envelope', 'branchwise_extrema'}")
    frequency_mode = str(frequency_fit_mode).strip().lower().replace("-", "_")
    if frequency_mode not in {"peak_spacing", "hilbert_phase"}:
        raise ValueError("frequency_fit_mode must be one of {'peak_spacing', 'hilbert_phase'}")
    if not 0.0 <= float(hilbert_trim_fraction) < 0.5:
        raise ValueError("hilbert_trim_fraction must be in [0, 0.5)")

    tail_mask, tail_tmin, tail_tmax = _tail_window(t_arr, float(tail_fraction))
    tail_vals = resp[tail_mask]
    if tail_vals.size == 0:
        raise ValueError("response windows must be non-empty")

    if policy == "first_abs":
        initial_level = float(abs(resp[0]))
    else:
        lead_mask, _lead_tmin, _lead_tmax = _leading_window(t_arr, float(initial_fraction))
        initial_vals = resp[lead_mask]
        if initial_vals.size == 0:
            raise ValueError("response windows must be non-empty")
        initial_level = float(np.mean(np.abs(initial_vals)))
    if initial_level <= 0.0 or not np.isfinite(initial_level):
        raise ValueError("initial response level must be finite and positive")

    residual = float(np.mean(tail_vals))
    residual_std = float(np.std(tail_vals))
    response_norm = resp / initial_level
    residual_norm = residual / initial_level
    residual_std_norm = residual_std / initial_level
    response_rms = float(np.sqrt(np.mean(np.square(response_norm[tail_mask]))))

    detrended_norm = response_norm - residual_norm
    fit_mask, fit_tmin, fit_tmax = _explicit_time_window(
        t_arr,
        tmin=fit_window_tmin,
        tmax=fit_window_tmax,
    )

    max_peak_idx = np.asarray([], dtype=int)
    min_peak_idx = np.asarray([], dtype=int)
    peak_idx = np.asarray([], dtype=int)
    if detrended_norm.size >= 3:
        max_peak_idx = (
            np.flatnonzero(
                (detrended_norm[1:-1] > detrended_norm[:-2])
                & (detrended_norm[1:-1] >= detrended_norm[2:])
                & (detrended_norm[1:-1] > 1.0e-12)
            )
            + 1
        )
        min_peak_idx = (
            np.flatnonzero(
                (detrended_norm[1:-1] < detrended_norm[:-2])
                & (detrended_norm[1:-1] <= detrended_norm[2:])
                & (detrended_norm[1:-1] < -1.0e-12)
            )
            + 1
        )
        peak_idx = np.sort(np.concatenate([max_peak_idx, min_peak_idx]))
    peak_times = t_arr[peak_idx]
    peak_values = np.abs(detrended_norm[peak_idx])
    max_peak_times = t_arr[max_peak_idx]
    max_peak_values = response_norm[max_peak_idx]
    min_peak_times = t_arr[min_peak_idx]
    min_peak_values = response_norm[min_peak_idx]

    gam_frequency = float("nan")
    gam_damping = float("nan")
    peak_fit_count = 0
    if damping_mode == "combined_envelope":
        peak_fit_times = peak_times[fit_mask[peak_idx]]
        peak_fit_values = peak_values[fit_mask[peak_idx]]
        if peak_fit_max_peaks is not None and peak_fit_times.size:
            nfit = min(int(peak_fit_max_peaks), int(peak_fit_times.size))
            peak_fit_times = peak_fit_times[:nfit]
            peak_fit_values = peak_fit_values[:nfit]
        peak_fit_count = int(peak_fit_times.size)
        valid = np.isfinite(peak_fit_values) & (peak_fit_values > 0.0)
        if np.count_nonzero(valid) >= 2:
            slope, _offset = np.polyfit(peak_fit_times[valid], np.log(peak_fit_values[valid]), 1)
            gam_damping = float(-slope)
    else:
        branch_gammas: list[float] = []
        branch_counts: list[int] = []
        for branch_idx in (max_peak_idx, min_peak_idx):
            idx = branch_idx[fit_mask[branch_idx]]
            if peak_fit_max_peaks is not None and idx.size:
                idx = idx[: min(int(peak_fit_max_peaks), int(idx.size))]
            amp = np.abs(detrended_norm[idx])
            valid = np.isfinite(amp) & (amp > 0.0)
            if np.count_nonzero(valid) >= 2:
                slope, _offset = np.polyfit(t_arr[idx][valid], np.log(amp[valid]), 1)
                branch_gammas.append(float(-slope))
                branch_counts.append(int(np.count_nonzero(valid)))
        if branch_gammas:
            gam_damping = float(np.mean(branch_gammas))
            peak_fit_count = int(np.sum(branch_counts))

    fit_peak_times = peak_times[fit_mask[peak_idx]]
    if peak_fit_max_peaks is not None and damping_mode == "combined_envelope" and fit_peak_times.size:
        fit_peak_times = fit_peak_times[: min(int(peak_fit_max_peaks), int(fit_peak_times.size))]

    if frequency_mode == "peak_spacing":
        freq_peak_times = fit_peak_times if fit_peak_times.size >= 2 else peak_times[fit_mask[peak_idx]]
        if freq_peak_times.size >= 2:
            dt_peaks = np.diff(freq_peak_times)
            dt_peaks = dt_peaks[np.isfinite(dt_peaks) & (dt_peaks > 0.0)]
            if dt_peaks.size:
                gam_frequency = float(np.pi / np.mean(dt_peaks))
    else:
        fit_t = t_arr[fit_mask]
        fit_signal = detrended_norm[fit_mask]
        if fit_t.size >= 8:
            analytic = _analytic_signal(fit_signal)
            phase = np.unwrap(np.angle(analytic))
            omega = np.gradient(phase, fit_t)
            trim = int(np.floor(float(hilbert_trim_fraction) * fit_t.size))
            trim_mask = np.ones_like(fit_t, dtype=bool)
            if trim > 0:
                trim_mask[:trim] = False
                trim_mask[-trim:] = False
            amp = np.abs(analytic)
            valid = np.isfinite(omega) & np.isfinite(amp) & (amp > 1.0e-6) & trim_mask
            if np.count_nonzero(valid) >= 2:
                gam_frequency = float(np.mean(omega[valid]))

    return ZonalFlowResponseMetrics(
        initial_level=initial_level,
        initial_policy=policy,
        residual_level=residual_norm,
        residual_std=residual_std_norm,
        response_rms=response_rms,
        gam_frequency=gam_frequency,
        gam_damping_rate=gam_damping,
        damping_method=damping_mode,
        frequency_method=frequency_mode,
        peak_count=int(peak_times.size),
        peak_fit_count=int(peak_fit_count),
        tmin=float(tail_tmin if tail_tmin is not None else t_arr[0]),
        tmax=float(tail_tmax if tail_tmax is not None else t_arr[-1]),
        fit_tmin=float(fit_tmin),
        fit_tmax=float(fit_tmax),
        peak_times=np.asarray(peak_times, dtype=float),
        peak_envelope=np.asarray(peak_values, dtype=float),
        max_peak_times=np.asarray(max_peak_times, dtype=float),
        max_peak_values=np.asarray(max_peak_values, dtype=float),
        min_peak_times=np.asarray(min_peak_times, dtype=float),
        min_peak_values=np.asarray(min_peak_values, dtype=float),
    )


def late_time_linear_metrics(
    result: object,
    *,
    tail_fraction: float = 0.5,
    mode_method: str = "project",
) -> LateTimeLinearMetrics:
    """Return late-time growth/frequency metrics from a linear benchmark/runtime result."""

    t = getattr(result, "t", None)
    if t is None:
        gamma = float(getattr(result, "gamma"))
        omega = float(getattr(result, "omega"))
        return LateTimeLinearMetrics(
            gamma_fit=gamma,
            omega_fit=omega,
            gamma_tail_mean=gamma,
            omega_tail_mean=omega,
            gamma_tail_std=0.0,
            omega_tail_std=0.0,
            tmin=None,
            tmax=None,
            nsamples=1,
            signal_source="scalar",
        )

    t_arr = np.asarray(t, dtype=float)
    mask, tmin, tmax = _tail_window(t_arr, tail_fraction)

    gamma_series = getattr(result, "gamma_t", None)
    omega_series = getattr(result, "omega_t", None)
    signal_source = "scalar"
    gamma_fit = float(getattr(result, "gamma"))
    omega_fit = float(getattr(result, "omega"))

    signal = getattr(result, "signal", None)
    if signal is not None:
        signal_arr = np.asarray(signal, dtype=np.complex128)
        signal_source = "signal"
    elif hasattr(result, "phi_t") and hasattr(result, "selection"):
        signal_arr = np.asarray(
            extract_mode_time_series(
                np.asarray(getattr(result, "phi_t")),
                getattr(result, "selection"),
                method=mode_method,
            ),
            dtype=np.complex128,
        )
        signal_source = f"phi_t:{mode_method}"
    else:
        signal_arr = None

    if signal_arr is not None:
        finite = np.isfinite(signal_arr)
        signal_tail = signal_arr[mask & finite]
        t_tail = t_arr[mask & finite]
        if t_tail.size >= 2:
            gamma_fit, omega_fit = fit_growth_rate(t_tail, signal_tail)

    if gamma_series is not None:
        gamma_mean, gamma_std = _tail_stats(np.asarray(gamma_series), mask)
    else:
        gamma_mean, gamma_std = gamma_fit, 0.0
    if omega_series is not None:
        omega_mean, omega_std = _tail_stats(np.asarray(omega_series), mask)
    else:
        omega_mean, omega_std = omega_fit, 0.0

    nsamples = int(np.count_nonzero(mask))
    return LateTimeLinearMetrics(
        gamma_fit=float(gamma_fit),
        omega_fit=float(omega_fit),
        gamma_tail_mean=float(gamma_mean),
        omega_tail_mean=float(omega_mean),
        gamma_tail_std=float(gamma_std),
        omega_tail_std=float(omega_std),
        tmin=tmin,
        tmax=tmax,
        nsamples=nsamples,
        signal_source=signal_source,
    )


def windowed_nonlinear_metrics(
    result: object,
    *,
    start_fraction: float = 0.5,
) -> NonlinearWindowMetrics:
    """Return late-window transport and envelope metrics from a nonlinear runtime result."""

    diagnostics = getattr(result, "diagnostics", result)
    if diagnostics is None:
        raise ValueError("nonlinear diagnostics are required")
    if not 0.0 <= float(start_fraction) < 1.0:
        raise ValueError("start_fraction must be in [0, 1)")
    t = np.asarray(getattr(diagnostics, "t", None), dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("diagnostics.t must be a non-empty one-dimensional array")
    tail_fraction = max(np.finfo(float).eps, 1.0 - float(start_fraction))
    mask, tmin, tmax = _tail_window(t, tail_fraction)
    heat_flux = np.asarray(getattr(diagnostics, "heat_flux_t"), dtype=float)[mask]
    wphi = np.asarray(getattr(diagnostics, "Wphi_t"), dtype=float)[mask]
    wg = np.asarray(getattr(diagnostics, "Wg_t"), dtype=float)[mask]
    heat_flux = heat_flux[np.isfinite(heat_flux)]
    wphi = wphi[np.isfinite(wphi)]
    wg = wg[np.isfinite(wg)]
    if heat_flux.size == 0 or wphi.size == 0 or wg.size == 0:
        raise ValueError("windowed diagnostics must contain finite heat/Wphi/Wg samples")

    phi_mode = getattr(diagnostics, "phi_mode_t", None)
    envelope_mean: float | None = None
    envelope_std: float | None = None
    envelope_max: float | None = None
    if phi_mode is not None:
        envelope = np.abs(np.asarray(phi_mode)[mask])
        envelope = envelope[np.isfinite(envelope)]
        if envelope.size:
            envelope_mean = float(np.mean(envelope))
            envelope_std = float(np.std(envelope))
            envelope_max = float(np.max(envelope))

    return NonlinearWindowMetrics(
        tmin=float(tmin if tmin is not None else t[0]),
        tmax=float(tmax if tmax is not None else t[-1]),
        nsamples=int(np.count_nonzero(mask)),
        heat_flux_mean=float(np.mean(heat_flux)),
        heat_flux_std=float(np.std(heat_flux)),
        heat_flux_rms=float(np.sqrt(np.mean(np.square(heat_flux)))),
        wphi_mean=float(np.mean(wphi)),
        wphi_std=float(np.std(wphi)),
        wg_mean=float(np.mean(wg)),
        wg_std=float(np.std(wg)),
        phi_mode_envelope_mean=envelope_mean,
        phi_mode_envelope_std=envelope_std,
        phi_mode_envelope_max=envelope_max,
    )


def estimate_observed_order(step_sizes: np.ndarray, errors: np.ndarray) -> ObservedOrderMetrics:
    """Estimate observed order from successive step-size refinements."""

    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if h.ndim != 1 or err.ndim != 1 or h.size != err.size or h.size < 2:
        raise ValueError("step_sizes and errors must be one-dimensional arrays of equal length >= 2")
    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(err)):
        raise ValueError("step_sizes and errors must be finite")
    if np.any(h <= 0.0):
        raise ValueError("step_sizes must be positive")
    if np.any(err <= 0.0):
        raise ValueError("errors must be positive")

    orders: list[float] = []
    for i in range(h.size - 1):
        if np.isclose(h[i], h[i + 1]):
            raise ValueError("successive step sizes must differ")
        orders.append(float(np.log(err[i] / err[i + 1]) / np.log(h[i] / h[i + 1])))
    orders_arr = np.asarray(orders, dtype=float)
    return ObservedOrderMetrics(
        step_sizes=h,
        errors=err,
        orders=orders_arr,
        asymptotic_order=float(orders_arr[-1]),
    )


def observed_order_gate_report(
    metrics: ObservedOrderMetrics,
    *,
    case: str,
    source: str,
    min_asymptotic_order: float,
    max_final_error: float | None = None,
    order_atol: float = 1.0e-12,
) -> GateReport:
    """Gate an observed-order convergence study.

    ``min_asymptotic_order`` encodes the expected method/order floor, while
    ``max_final_error`` can be used for publication figures where both rate and
    absolute accuracy matter.
    """

    min_order = float(min_asymptotic_order)
    order_tol = float(order_atol)
    if min_order < 0.0 or order_tol < 0.0:
        raise ValueError("min_asymptotic_order and order_atol must be non-negative")
    gates = [
        evaluate_scalar_gate(
            "observed_order_deficit",
            max(0.0, min_order - float(metrics.asymptotic_order)),
            0.0,
            atol=order_tol,
            rtol=0.0,
            notes=f"Passes when asymptotic observed order >= {min_order:.6g}.",
        )
    ]
    if max_final_error is not None:
        final_error_limit = float(max_final_error)
        if final_error_limit < 0.0:
            raise ValueError("max_final_error must be non-negative")
        gates.append(
            evaluate_scalar_gate(
                "final_error",
                float(metrics.errors[-1]),
                0.0,
                atol=final_error_limit,
                rtol=0.0,
                notes=f"Passes when final-grid error <= {final_error_limit:.6g}.",
            )
        )
    return gate_report(case, source, gates)


def branch_continuity_metrics(
    ky: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    *,
    successive_overlap: np.ndarray | None = None,
    floor_fraction: float = 1.0e-8,
) -> BranchContinuationMetrics:
    """Compute branch-continuity diagnostics for a linear scan.

    The relative jump normalization uses a local scale from adjacent values,
    with a floor tied to the largest value in the scan. This avoids false
    blow-ups near marginal points while still flagging branch jumps.
    """

    ky_arr = np.asarray(ky, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    omega_arr = np.asarray(omega, dtype=float)
    if ky_arr.ndim != 1 or gamma_arr.ndim != 1 or omega_arr.ndim != 1:
        raise ValueError("ky, gamma, and omega must be one-dimensional arrays")
    if not (ky_arr.size == gamma_arr.size == omega_arr.size):
        raise ValueError("ky, gamma, and omega must have equal length")
    if ky_arr.size < 2:
        raise ValueError("branch continuity requires at least two ky samples")
    if np.any(~np.isfinite(ky_arr)) or np.any(~np.isfinite(gamma_arr)) or np.any(~np.isfinite(omega_arr)):
        raise ValueError("ky, gamma, and omega must be finite")
    floor = float(floor_fraction)
    if floor < 0.0:
        raise ValueError("floor_fraction must be non-negative")

    def _relative_jumps(values: np.ndarray) -> np.ndarray:
        jumps = np.abs(np.diff(values))
        global_floor = max(float(np.nanmax(np.abs(values))) * floor, 1.0e-30)
        local_scale = np.maximum(np.maximum(np.abs(values[:-1]), np.abs(values[1:])), global_floor)
        return jumps / local_scale

    overlap_min: float | None = None
    if successive_overlap is not None:
        overlap = np.asarray(successive_overlap, dtype=float)
        if overlap.ndim != 1 or overlap.size != ky_arr.size - 1:
            raise ValueError("successive_overlap must have length len(ky) - 1")
        if np.any(~np.isfinite(overlap)):
            raise ValueError("successive_overlap must be finite")
        overlap_min = float(np.min(overlap))

    gamma_jumps = _relative_jumps(gamma_arr)
    omega_jumps = _relative_jumps(omega_arr)
    return BranchContinuationMetrics(
        ky=ky_arr,
        gamma=gamma_arr,
        omega=omega_arr,
        rel_gamma_jumps=gamma_jumps,
        rel_omega_jumps=omega_jumps,
        max_rel_gamma_jump=float(np.max(gamma_jumps)),
        max_rel_omega_jump=float(np.max(omega_jumps)),
        min_successive_overlap=overlap_min,
    )


def branch_continuity_gate_report(
    metrics: BranchContinuationMetrics,
    *,
    case: str,
    source: str,
    max_rel_gamma_jump: float,
    max_rel_omega_jump: float,
    min_successive_overlap: float | None = None,
) -> GateReport:
    """Gate branch-continuation diagnostics for branch-followed scans."""

    gamma_limit = float(max_rel_gamma_jump)
    omega_limit = float(max_rel_omega_jump)
    if gamma_limit < 0.0 or omega_limit < 0.0:
        raise ValueError("maximum relative jumps must be non-negative")
    gates = [
        evaluate_scalar_gate(
            "max_rel_gamma_jump",
            metrics.max_rel_gamma_jump,
            0.0,
            atol=gamma_limit,
            rtol=0.0,
            notes=f"Passes when adjacent gamma jumps <= {gamma_limit:.6g}.",
        ),
        evaluate_scalar_gate(
            "max_rel_omega_jump",
            metrics.max_rel_omega_jump,
            0.0,
            atol=omega_limit,
            rtol=0.0,
            notes=f"Passes when adjacent omega jumps <= {omega_limit:.6g}.",
        ),
    ]
    if min_successive_overlap is not None:
        min_overlap = float(min_successive_overlap)
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_successive_overlap must be in [0, 1]")
        observed = float("nan") if metrics.min_successive_overlap is None else float(metrics.min_successive_overlap)
        gates.append(
            evaluate_scalar_gate(
                "successive_overlap_deficit",
                max(0.0, min_overlap - observed) if np.isfinite(observed) else float("nan"),
                0.0,
                atol=0.0,
                rtol=0.0,
                notes=f"Passes when successive eigenfunction overlap >= {min_overlap:.6g}.",
            )
        )
    return gate_report(case, source, gates)


def run_linear_scan(
    *,
    ky_values: np.ndarray,
    run_linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
) -> LinearScanResult:
    """Run a linear scan over ky values."""

    gammas: list[float] = []
    omegas: list[float] = []
    ky_out: list[float] = []
    for i, ky in enumerate(ky_values):
        if resolution_policy is not None:
            Nl_i, Nm_i = resolution_policy(float(ky))
        else:
            Nl_i, Nm_i = int(Nl), int(Nm)
        dt_i = float(dt[i]) if isinstance(dt, np.ndarray) else float(dt)
        steps_i = int(steps[i]) if isinstance(steps, np.ndarray) else int(steps)
        tmin_i = tmin[i] if isinstance(tmin, np.ndarray) else tmin
        tmax_i = tmax[i] if isinstance(tmax, np.ndarray) else tmax
        krylov_cfg_use = krylov_policy(float(ky)) if krylov_policy is not None else krylov_cfg
        result = run_linear_fn(
            ky_target=float(ky),
            cfg=cfg,
            Nl=int(Nl_i),
            Nm=int(Nm_i),
            dt=dt_i,
            steps=steps_i,
            method=method,
            solver=solver,
            krylov_cfg=krylov_cfg_use,
            auto_window=auto_window,
            tmin=tmin_i,
            tmax=tmax_i,
            **window_kw,
            **(run_kwargs or {}),
        )
        gammas.append(float(result.gamma))
        omegas.append(float(result.omega))
        ky_out.append(float(result.ky))

    return LinearScanResult(ky=np.array(ky_out), gamma=np.array(gammas), omega=np.array(omegas))


def run_scan_and_mode(
    *,
    ky_values: np.ndarray,
    scan_fn,
    linear_fn,
    cfg,
    Nl: int,
    Nm: int,
    dt: float | np.ndarray,
    steps: int | np.ndarray,
    method: str,
    solver: str,
    mode_solver: str,
    krylov_cfg,
    window_kw: dict,
    tmin: float | np.ndarray | None = None,
    tmax: float | np.ndarray | None = None,
    auto_window: bool = True,
    run_kwargs: dict | None = None,
    mode_kwargs: dict | None = None,
    resolution_policy: Callable[[float], tuple[int, int]] | None = None,
    krylov_policy: Callable[[float], object] | None = None,
    select_ky: Callable[[LinearScanResult], float] | None = None,
) -> ScanAndModeResult:
    """Run a scan and extract a representative eigenfunction."""

    if select_ky is None:
        select_ky = lambda scan: float(scan.ky[int(np.nanargmax(scan.gamma))])

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=linear_fn,
        cfg=cfg,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        solver=solver,
        krylov_cfg=krylov_cfg,
        window_kw=window_kw,
        tmin=tmin,
        tmax=tmax,
        auto_window=auto_window,
        run_kwargs=run_kwargs,
        resolution_policy=resolution_policy,
        krylov_policy=krylov_policy,
    )
    ky_sel = float(select_ky(scan))
    if resolution_policy is not None:
        Nl_mode, Nm_mode = resolution_policy(ky_sel)
    else:
        Nl_mode, Nm_mode = int(Nl), int(Nm)
    idx = int(np.argmin(np.abs(scan.ky - ky_sel)))
    dt_mode = float(dt[idx]) if isinstance(dt, np.ndarray) else float(dt)
    steps_mode = int(steps[idx]) if isinstance(steps, np.ndarray) else int(steps)
    run: LinearRunResult = linear_fn(
        cfg=cfg,
        ky_target=ky_sel,
        Nl=int(Nl_mode),
        Nm=int(Nm_mode),
        dt=dt_mode,
        steps=steps_mode,
        method=method,
        solver=mode_solver,
        **window_kw,
        **(mode_kwargs or {}),
    )
    grid = build_spectral_grid(cfg.grid)
    if run.t.size < 2:
        tmin_fit = None
        tmax_fit = None
    else:
        signal = extract_mode_time_series(run.phi_t, run.selection, method="project")
        _g, _w, tmin_fit, tmax_fit = fit_growth_rate_auto(run.t, signal, **window_kw)
    z_np = np.asarray(grid.z)
    eig = extract_eigenfunction(
        run.phi_t, run.t, run.selection, z=z_np, method="snapshot", tmin=tmin_fit, tmax=tmax_fit
    )
    return ScanAndModeResult(
        scan=scan,
        eigenfunction=eig,
        grid=grid,
        ky_selected=ky_sel,
        tmin=tmin_fit,
        tmax=tmax_fit,
    )
