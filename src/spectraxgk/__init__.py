"""SPECTRAX-GK: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from spectraxgk._version import __version__
from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all, gamma0
from spectraxgk.operators import hermite_streaming
from spectraxgk.linear import LinearParams, linear_rhs, integrate_linear
from spectraxgk.analysis import fit_growth_rate, ModeSelection, extract_mode, select_ky_index
from spectraxgk.benchmarks import load_cyclone_reference, CycloneReference, run_cyclone_linear, CycloneRunResult
from spectraxgk.plotting import cyclone_reference_figure, set_plot_style

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
    "linear_rhs",
    "integrate_linear",
    "load_cyclone_reference",
    "fit_growth_rate",
    "CycloneReference",
    "run_cyclone_linear",
    "CycloneRunResult",
    "ModeSelection",
    "extract_mode",
    "select_ky_index",
    "cyclone_reference_figure",
    "set_plot_style",
]

