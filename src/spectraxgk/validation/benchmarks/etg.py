"""ETG benchmark facade.

The implementation is split into single-mode and ky-scan runners while this
module keeps the historical import path and test monkeypatch seams stable.
"""

# ruff: noqa: F401

from __future__ import annotations

from functools import wraps

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    ModeSelectionBatch,
    fit_growth_rate,
    fit_growth_rate_auto,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    ETG_KRYLOV_DEFAULT,
    ETG_OMEGA_D_SCALE,
    ETG_OMEGA_STAR_SCALE,
    ETG_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import (
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import _midplane_index
from spectraxgk.validation.benchmarks.species import (
    _electron_only_params,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    normalize_fit_signal,
    normalize_solver_key,
    resolve_scan_mode_method,
    scan_window_valid,
    should_use_ky_batch,
)
from spectraxgk.config import ETGBaseCase, TimeConfig
from spectraxgk.solvers.time.diffrax import (
    integrate_linear_diffrax,
    integrate_linear_diffrax_streaming,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
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
from spectraxgk.validation.benchmarks import etg_linear as _etg_linear
from spectraxgk.validation.benchmarks import etg_scan as _etg_scan

_PATCHABLE_NAMES = (
    "replace",
    "jnp",
    "np",
    "ModeSelection",
    "ModeSelectionBatch",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "instantaneous_growth_rate_from_phi",
    "select_ky_index",
    "ETG_KRYLOV_DEFAULT",
    "ETG_OMEGA_D_SCALE",
    "ETG_OMEGA_STAR_SCALE",
    "ETG_RHO_STAR",
    "_iter_ky_batches",
    "_resolve_streaming_window",
    "_extract_mode_only_signal",
    "_normalize_growth_rate",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_build_initial_condition",
    "LinearRunResult",
    "LinearScanResult",
    "_midplane_index",
    "_electron_only_params",
    "_two_species_params",
    "ScanFitWindowPolicy",
    "apply_auto_fit_scan_policy",
    "indexed_float_value",
    "normalize_fit_signal",
    "normalize_solver_key",
    "resolve_scan_mode_method",
    "scan_window_valid",
    "should_use_ky_batch",
    "ETGBaseCase",
    "TimeConfig",
    "integrate_linear_diffrax",
    "integrate_linear_diffrax_streaming",
    "SAlphaGeometry",
    "build_spectral_grid",
    "select_ky_grid",
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
    _sync_impl_hooks(_etg_linear)
    _sync_impl_hooks(_etg_scan)
    _etg_scan.run_etg_linear = globals()["run_etg_linear"]


@wraps(_etg_linear.run_etg_linear)
def run_etg_linear(*args, **kwargs):
    _sync_impl_hooks(_etg_linear)
    return _etg_linear.run_etg_linear(*args, **kwargs)


@wraps(_etg_scan.run_etg_scan)
def run_etg_scan(*args, **kwargs):
    _sync_all_hooks()
    return _etg_scan.run_etg_scan(*args, **kwargs)


__all__ = ["run_etg_linear", "run_etg_scan"]
