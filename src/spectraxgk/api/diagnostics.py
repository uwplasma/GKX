"""Public diagnostics API exports."""

from spectraxgk.diagnostics.normalization import (
    DiagnosticNorm,
    NormalizationContract,
    apply_diagnostic_normalization,
    get_normalization_contract,
)
from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode,
    extract_mode_time_series,
    extract_eigenfunction,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_fit_window,
    select_ky_index,
)
from spectraxgk.diagnostics import SimulationDiagnostics

growth_rate_from_phi = instantaneous_growth_rate_from_phi

__all__ = [
    "DiagnosticNorm",
    "NormalizationContract",
    "get_normalization_contract",
    "apply_diagnostic_normalization",
    "SimulationDiagnostics",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "instantaneous_growth_rate_from_phi",
    "growth_rate_from_phi",
    "select_fit_window",
    "ModeSelection",
    "extract_mode",
    "extract_mode_time_series",
    "extract_eigenfunction",
    "select_ky_index",
]
