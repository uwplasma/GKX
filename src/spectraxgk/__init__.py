"""SPECTRAX-GK: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from spectraxgk._version import __version__
from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all, gamma0
from spectraxgk.operators import hermite_streaming
from spectraxgk.linear import (
    LinearCache,
    LinearParams,
    build_linear_cache,
    integrate_linear,
    linear_rhs,
    linear_rhs_cached,
)
from spectraxgk.analysis import (
    ModeSelection,
    extract_mode,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    select_fit_window,
    select_ky_index,
)
from spectraxgk.benchmarks import (
    CycloneComparison,
    CycloneReference,
    CycloneRunResult,
    CycloneScanResult,
    compare_cyclone_to_reference,
    load_cyclone_reference,
    run_cyclone_linear,
    run_cyclone_scan,
)
from spectraxgk.plotting import cyclone_comparison_figure, cyclone_reference_figure, set_plot_style

__all__ = [
    "__version__",
    "CycloneBaseCase",
    "GridConfig",
    "TimeConfig",
    "SAlphaGeometry",
    "J_l_all",
    "gamma0",
    "hermite_streaming",
    "LinearParams",
    "LinearCache",
    "build_linear_cache",
    "linear_rhs",
    "linear_rhs_cached",
    "integrate_linear",
    "load_cyclone_reference",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "select_fit_window",
    "CycloneReference",
    "CycloneScanResult",
    "CycloneComparison",
    "run_cyclone_linear",
    "run_cyclone_scan",
    "compare_cyclone_to_reference",
    "CycloneRunResult",
    "ModeSelection",
    "extract_mode",
    "extract_mode_time_series",
    "select_ky_index",
    "cyclone_reference_figure",
    "cyclone_comparison_figure",
    "set_plot_style",
]
