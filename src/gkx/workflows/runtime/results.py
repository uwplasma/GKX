"""Runtime result containers and small assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gkx.diagnostics.modes import ModeSelection
from gkx.diagnostics import SimulationDiagnostics
from gkx.terms.config import FieldState


def _dataset_payload(
    coords: dict[str, Any], data_vars: dict[str, Any], attrs: dict[str, Any]
) -> dict[str, Any]:
    """Return an xarray-compatible mapping without importing xarray.

    GKX does not depend on xarray, and the dependency review keeps it that way,
    so a result cannot hand back a real ``Dataset``. It hands back exactly the
    three keyword arguments ``xarray.Dataset`` accepts, so a caller who has
    xarray writes ``xarray.Dataset(**result.to_dataset())`` and a caller who
    does not still gets named arrays with their coordinates attached.
    """

    drop_empty = {k: v for k, v in data_vars.items() if v is not None}
    return {
        "coords": {k: v for k, v in coords.items() if v is not None},
        "data_vars": {k: (dims, np.asarray(v)) for k, (dims, v) in drop_empty.items()},
        "attrs": {k: v for k, v in attrs.items() if v is not None},
    }


class _ResultArtifacts:
    """Shared ``save``/``plot``/``print_summary`` behaviour for public results.

    Each result already had a writer and a figure builder; what it lacked was a
    way to reach them from the object itself. These delegate rather than
    reimplement, so the artifact bytes a result writes through ``save`` are the
    bytes the runtime already wrote.
    """

    @staticmethod
    def _artifact_writer(io_module: Any) -> Any:
        """Return the writer in ``gkx.artifacts.io`` that owns this result."""

        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        """Return this result's typed scalar diagnostics."""

        raise NotImplementedError

    def save(self, path: str | Path) -> dict[str, str]:
        """Write this result's artifact bundle and return the paths written."""

        from gkx.artifacts import io as artifacts_io

        writer = self._artifact_writer(artifacts_io)
        return writer(path, self)

    def plot(self) -> Any:
        """Return the standard figure for this result."""

        from gkx.artifacts.plotting import plot as _plot

        return _plot(self)

    def print_summary(self, *, stream: Any = None) -> None:
        """Print the same scalar summary that ``save`` records."""

        print(self.summary_text(), file=stream)

    def summary_text(self) -> str:
        """Return the one-line-per-field scalar summary as text."""

        return "\n".join(f"{k}: {v}" for k, v in self.summary().items())


@dataclass(frozen=True)
class RuntimeLinearResult(_ResultArtifacts):
    """Result container for runtime linear runs."""

    ky: float
    gamma: float
    omega: float
    selection: ModeSelection
    t: np.ndarray | None = None
    signal: np.ndarray | None = None
    field_history: np.ndarray | None = None
    state: np.ndarray | None = None
    z: np.ndarray | None = None
    eigenfunction: np.ndarray | None = None
    fit_window_tmin: float | None = None
    fit_window_tmax: float | None = None
    fit_signal_used: str | None = None
    # Fit-quality diagnostics over the selected window (None for eigensolves).
    gamma_stderr: float | None = None
    omega_stderr: float | None = None
    fit_r2: float | None = None
    fit_settled: bool | None = None
    quasilinear: dict[str, Any] | None = None

    @staticmethod
    def _artifact_writer(io_module: Any) -> Any:
        return io_module.write_runtime_linear_artifacts

    def summary(self) -> dict[str, Any]:
        """Return the typed scalar diagnostics, including fit status."""

        return {
            "kind": "linear",
            "ky": float(self.ky),
            "gamma": float(self.gamma),
            "omega": float(self.omega),
            "gamma_stderr": self.gamma_stderr,
            "omega_stderr": self.omega_stderr,
            "fit_r2": self.fit_r2,
            "fit_settled": self.fit_settled,
            "fit_signal_used": self.fit_signal_used,
            "fit_window_tmin": self.fit_window_tmin,
            "fit_window_tmax": self.fit_window_tmax,
        }

    def to_dataset(self) -> dict[str, Any]:
        """Return the trace and eigenfunction as an xarray-compatible mapping."""

        return _dataset_payload(
            coords={"t": self.t, "z": self.z},
            data_vars={
                "signal": ("t", self.signal) if self.signal is not None else None,
                "eigenfunction": (
                    ("z", self.eigenfunction)
                    if self.eigenfunction is not None
                    else None
                ),
            },
            attrs=self.summary(),
        )


@dataclass(frozen=True)
class RuntimeLinearScanResult(_ResultArtifacts):
    """Result container for runtime linear ky scans."""

    ky: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    quasilinear: tuple[dict[str, Any], ...] | None = None
    parallel: dict[str, Any] | None = None
    # Provenance for a state-carrying scan: visit order and how many points
    # were seeded from a neighbour. None when every point started cold.
    warm_start: dict[str, Any] | None = None

    @staticmethod
    def _artifact_writer(io_module: Any) -> Any:
        return io_module.write_runtime_linear_scan_artifacts

    def summary(self) -> dict[str, Any]:
        """Return scan extent and the peak growth rate over the scanned k_y."""

        gamma = np.asarray(self.gamma)
        ky = np.asarray(self.ky)
        peak = int(np.argmax(gamma)) if gamma.size else None
        return {
            "kind": "linear_scan",
            "n_points": int(ky.size),
            "ky_min": float(ky.min()) if ky.size else None,
            "ky_max": float(ky.max()) if ky.size else None,
            "gamma_peak": float(gamma[peak]) if peak is not None else None,
            "ky_at_gamma_peak": float(ky[peak]) if peak is not None else None,
        }

    def to_dataset(self) -> dict[str, Any]:
        """Return growth and frequency against k_y."""

        return _dataset_payload(
            coords={"ky": np.asarray(self.ky)},
            data_vars={
                "gamma": ("ky", self.gamma),
                "omega": ("ky", self.omega),
            },
            attrs=self.summary(),
        )


@dataclass(frozen=True)
class RuntimeParameterScanResult:
    """Ordered linear results for a named scalar configuration parameter."""

    parameter_name: str
    values: np.ndarray
    gamma: np.ndarray
    omega: np.ndarray
    runs: tuple[RuntimeLinearResult, ...]


@dataclass(frozen=True)
class RuntimeNonlinearResult(_ResultArtifacts):
    """Result container for runtime nonlinear runs."""

    t: np.ndarray
    diagnostics: SimulationDiagnostics | None
    phi2: np.ndarray | None = None
    fields: FieldState | None = None
    state: np.ndarray | None = None
    ky_selected: float | None = None
    kx_selected: float | None = None
    # Wall-clock seconds spent integrating this result, when the caller
    # measured it. Reported per unit of simulated time so a straggler surface
    # is comparable against its siblings without normalising by hand.
    wall_seconds: float | None = None
    # Run-to-saturation stop decision: measured heat-flux window, mean +/- SEM,
    # and whether the run stopped before t_max. None when the run was not
    # driven by the saturation stop policy.
    saturation: dict[str, Any] | None = None

    @staticmethod
    def _artifact_writer(io_module: Any) -> Any:
        return io_module.write_runtime_nonlinear_table_artifacts

    def summary(self) -> dict[str, Any]:
        """Return scalar diagnostics and the saturation verdict.

        ``saturated`` is reported as the stop policy recorded it, including
        ``False`` and ``None``. A result must not present an unsaturated window
        as an accepted value, so the status travels with the number rather than
        being inferred by a reader.
        """

        t = np.asarray(self.t)
        saturation = self.saturation or {}
        return {
            "kind": "nonlinear",
            "t_final": float(t[-1]) if t.size else None,
            "n_samples": int(t.size),
            "ky_selected": self.ky_selected,
            "kx_selected": self.kx_selected,
            "wall_seconds": self.wall_seconds,
            "saturated": saturation.get("saturated"),
            "heat_flux_mean": saturation.get("mean_flux"),
            "heat_flux_sem": saturation.get("sem"),
            "window_tmin": saturation.get("window_tmin"),
            "window_tmax": saturation.get("window_tmax"),
        }

    def to_dataset(self) -> dict[str, Any]:
        """Return the sampled time series as an xarray-compatible mapping."""

        return _dataset_payload(
            coords={"t": np.asarray(self.t)},
            data_vars={"phi2": ("t", self.phi2) if self.phi2 is not None else None},
            attrs=self.summary(),
        )


LinearResult = RuntimeLinearResult
NonlinearResult = RuntimeNonlinearResult
ScanResult = RuntimeLinearScanResult


def nonlinear_field_phi2(fields: FieldState) -> np.ndarray:
    """Return the mean electrostatic energy density from final fields."""

    return np.asarray(jnp.mean(jnp.abs(fields.phi) ** 2))


def build_runtime_nonlinear_result(
    *,
    t: np.ndarray,
    diagnostics: SimulationDiagnostics | None,
    fields: FieldState | None,
    state: np.ndarray | None,
    ky_selected: float | None,
    kx_selected: float | None,
    summarize_fields: bool,
    saturation: dict[str, Any] | None = None,
) -> RuntimeNonlinearResult:
    """Build a runtime nonlinear result with optional final-field summary."""

    phi2 = None
    t_out = np.asarray(t)
    diag_out = diagnostics
    if summarize_fields:
        if fields is None:
            raise RuntimeError("final fields are required when summarize_fields=True")
        phi2 = nonlinear_field_phi2(fields)
        t_out = np.asarray([])
        diag_out = None
    return RuntimeNonlinearResult(
        t=t_out,
        diagnostics=diag_out,
        phi2=phi2,
        fields=fields,
        state=state,
        ky_selected=ky_selected,
        kx_selected=kx_selected,
        saturation=saturation,
    )
