"""SPECTRAX-GK: a JAX gyrokinetic solver with Hermite-Laguerre velocity space."""

from spectraxgk._version import __version__
from spectraxgk.config import CycloneBaseCase, GridConfig, TimeConfig
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.gyroaverage import J_l_all, gamma0
from spectraxgk.operators import hermite_streaming
from spectraxgk.linear import LinearParams, linear_rhs
from spectraxgk.benchmarks import load_cyclone_reference, fit_growth_rate, CycloneReference

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
    "load_cyclone_reference",
    "fit_growth_rate",
    "CycloneReference",
]

