"""Compatibility facade for quasilinear calibration artifacts.

Core calibration reports, spectrum integration, and nonlinear-window artifact
IO live in focused modules. This facade keeps the historical import path stable
for tools, tests, and public API exports.
"""

from __future__ import annotations

# ruff: noqa: F401

from spectraxgk.validation.quasilinear.calibration_core import (
    QuasilinearCalibrationPoint,
    _holdout_window_convergence_summary,
    _point_from_mapping,
    _split_metrics,
    apply_heat_flux_scale,
    fit_train_heat_flux_scale,
    quasilinear_calibration_report,
)
from spectraxgk.validation.quasilinear.calibration_io import (
    _NETCDF_HEAT_FLUX_COLUMNS,
    _csv_heat_flux_trace,
    _netcdf_heat_flux_trace,
    _netcdf_heat_flux_variable,
    _netcdf_variable,
    _read_csv_window_heat_flux,
    _read_netcdf_window_heat_flux,
    _resolve_summary_artifact,
    _window_convergence_config,
    _window_from_summary,
    calibration_point_from_nonlinear_window_summary,
    calibration_point_from_spectrum_and_nonlinear_window,
    write_quasilinear_calibration_report,
)
from spectraxgk.validation.quasilinear.calibration_spectrum import (
    integrated_quasilinear_flux_from_spectrum,
)

__all__ = [
    "QuasilinearCalibrationPoint",
    "apply_heat_flux_scale",
    "calibration_point_from_nonlinear_window_summary",
    "calibration_point_from_spectrum_and_nonlinear_window",
    "fit_train_heat_flux_scale",
    "integrated_quasilinear_flux_from_spectrum",
    "quasilinear_calibration_report",
    "write_quasilinear_calibration_report",
]
