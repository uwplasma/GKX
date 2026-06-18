"""Cyclone benchmark facade.

The implementation is split into single-mode and ky-scan runners, while this
module keeps the historical import path and test monkeypatch seams stable.
"""

# ruff: noqa: F401

from __future__ import annotations

from functools import wraps

from dataclasses import replace
from typing import Callable

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    CYCLONE_KRYLOV_DEFAULT,
    CYCLONE_OMEGA_D_SCALE,
    CYCLONE_OMEGA_STAR_SCALE,
    CYCLONE_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import CycloneRunResult, CycloneScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    _apply_reference_hypercollisions,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    should_use_ky_batch,
)
from spectraxgk.config import (
    CycloneBaseCase,
    InitializationConfig,
    TimeConfig,
    resolve_cfl_fac,
)
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit,
)
from spectraxgk.linear import integrate_linear, integrate_linear_diagnostics
from spectraxgk.operators.linear.cache import build_linear_cache
from spectraxgk.operators.linear.params import (
    LinearParams,
    LinearTerms,
    linear_terms_to_term_config,
)
from spectraxgk.solvers.linear.krylov import KrylovConfig, dominant_eigenpair
from spectraxgk.solvers.time.runners import integrate_linear_from_config
from spectraxgk.terms.assembly import compute_fields_cached
from spectraxgk.validation.benchmarks import cyclone_linear as _cyclone_linear
from spectraxgk.validation.benchmarks import cyclone_scan as _cyclone_scan

_PATCHABLE_NAMES = (
    "replace",
    "Callable",
    "jnp",
    "np",
    "ModeSelection",
    "ModeSelectionBatch",
    "extract_mode_time_series",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "instantaneous_growth_rate_from_phi",
    "select_ky_index",
    "CYCLONE_KRYLOV_DEFAULT",
    "CYCLONE_OMEGA_D_SCALE",
    "CYCLONE_OMEGA_STAR_SCALE",
    "CYCLONE_RHO_STAR",
    "_iter_ky_batches",
    "_resolve_streaming_window",
    "_normalize_growth_rate",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_build_initial_condition",
    "CycloneRunResult",
    "CycloneScanResult",
    "_midplane_index",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "_apply_reference_hypercollisions",
    "ScanFitWindowPolicy",
    "apply_auto_fit_scan_policy",
    "indexed_float_value",
    "normalize_fit_signal",
    "normalize_solver_key",
    "resolve_scan_mode_method",
    "should_use_ky_batch",
    "CycloneBaseCase",
    "InitializationConfig",
    "TimeConfig",
    "resolve_cfl_fac",
    "integrate_linear_diffrax_streaming",
    "SAlphaGeometry",
    "build_spectral_grid",
    "select_ky_grid",
    "ExplicitTimeConfig",
    "integrate_linear_explicit",
    "integrate_linear",
    "integrate_linear_diagnostics",
    "build_linear_cache",
    "LinearParams",
    "LinearTerms",
    "linear_terms_to_term_config",
    "KrylovConfig",
    "dominant_eigenpair",
    "integrate_linear_from_config",
    "compute_fields_cached",
)


def _sync_impl_hooks(module: object) -> None:
    namespace = globals()
    for name in _PATCHABLE_NAMES:
        if name in namespace and hasattr(module, name):
            setattr(module, name, namespace[name])


def _sync_all_hooks() -> None:
    _sync_impl_hooks(_cyclone_linear)
    _sync_impl_hooks(_cyclone_scan)
    _cyclone_scan.run_cyclone_linear = globals()["run_cyclone_linear"]


@wraps(_cyclone_linear.run_cyclone_linear)
def run_cyclone_linear(*args, **kwargs):
    _sync_impl_hooks(_cyclone_linear)
    return _cyclone_linear.run_cyclone_linear(*args, **kwargs)


@wraps(_cyclone_scan.run_cyclone_scan)
def run_cyclone_scan(*args, **kwargs):
    _sync_all_hooks()
    return _cyclone_scan.run_cyclone_scan(*args, **kwargs)


__all__ = ["run_cyclone_linear", "run_cyclone_scan"]
