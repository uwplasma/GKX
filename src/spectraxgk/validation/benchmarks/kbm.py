"""KBM benchmark facade.

The implementation is split into beta-scan, single-point, and ky-scan runners.
This module keeps the historical import path and test monkeypatch seams stable.
"""

# ruff: noqa: F401

from __future__ import annotations

from functools import wraps

from dataclasses import replace
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from spectraxgk.diagnostics.analysis import (
    ModeSelection,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    windowed_growth_rate_from_omega_series,
    instantaneous_growth_rate_from_phi,
    select_ky_index,
)
from spectraxgk.validation.benchmarks.defaults import (
    KBM_KRYLOV_DEFAULT,
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
)
from spectraxgk.validation.benchmarks.batching import _resolve_streaming_window
from spectraxgk.validation.benchmarks.fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _select_fit_signal,
    _select_fit_signal_auto,
)
from spectraxgk.validation.benchmarks.initialization import _build_initial_condition
from spectraxgk.validation.benchmarks.reference import LinearRunResult, LinearScanResult
from spectraxgk.validation.benchmarks.solver_policy import (
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.validation.benchmarks.species import (
    _linked_boundary_end_damping,
    _two_species_params,
)
from spectraxgk.validation.benchmarks.scan import (
    ScanFitWindowPolicy,
    apply_auto_fit_scan_policy,
    indexed_float_value,
    indexed_scan_value,
    normalize_fit_signal,
    normalize_solver_key,
    scan_window_valid,
)
from spectraxgk.config import KBMBaseCase, TimeConfig, resolve_cfl_fac
from spectraxgk.solvers.time.diffrax import integrate_linear_diffrax_streaming
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    build_flux_tube_geometry,
)
from spectraxgk.core.grid import build_spectral_grid, select_ky_grid
from spectraxgk.solvers.time.explicit import (
    ExplicitTimeConfig,
    integrate_linear_explicit_diagnostics,
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
from spectraxgk.validation.benchmarks import kbm_beta as _kbm_beta
from spectraxgk.validation.benchmarks import kbm_linear as _kbm_linear
from spectraxgk.validation.benchmarks import kbm_scan as _kbm_scan

_PATCHABLE_NAMES = (
    "replace",
    "Sequence",
    "jnp",
    "np",
    "ModeSelection",
    "extract_mode_time_series",
    "fit_growth_rate",
    "fit_growth_rate_auto",
    "windowed_growth_rate_from_omega_series",
    "instantaneous_growth_rate_from_phi",
    "select_ky_index",
    "KBM_KRYLOV_DEFAULT",
    "KBM_OMEGA_D_SCALE",
    "KBM_OMEGA_STAR_SCALE",
    "KBM_RHO_STAR",
    "_resolve_streaming_window",
    "_extract_mode_only_signal",
    "_normalize_growth_rate",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_build_initial_condition",
    "LinearRunResult",
    "LinearScanResult",
    "_kbm_use_multi_target_krylov",
    "_midplane_index",
    "select_kbm_solver_auto",
    "_linked_boundary_end_damping",
    "_two_species_params",
    "ScanFitWindowPolicy",
    "apply_auto_fit_scan_policy",
    "indexed_float_value",
    "indexed_scan_value",
    "normalize_fit_signal",
    "normalize_solver_key",
    "scan_window_valid",
    "KBMBaseCase",
    "TimeConfig",
    "resolve_cfl_fac",
    "integrate_linear_diffrax_streaming",
    "SAlphaGeometry",
    "apply_geometry_grid_defaults",
    "build_flux_tube_geometry",
    "build_spectral_grid",
    "select_ky_grid",
    "ExplicitTimeConfig",
    "integrate_linear_explicit_diagnostics",
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
    _sync_impl_hooks(_kbm_beta)
    _sync_impl_hooks(_kbm_linear)
    _sync_impl_hooks(_kbm_scan)
    _kbm_scan.run_kbm_beta_scan = globals()["run_kbm_beta_scan"]


@wraps(_kbm_beta.run_kbm_beta_scan)
def run_kbm_beta_scan(*args, **kwargs):
    _sync_impl_hooks(_kbm_beta)
    return _kbm_beta.run_kbm_beta_scan(*args, **kwargs)


@wraps(_kbm_linear.run_kbm_linear)
def run_kbm_linear(*args, **kwargs):
    _sync_impl_hooks(_kbm_linear)
    return _kbm_linear.run_kbm_linear(*args, **kwargs)


@wraps(_kbm_scan.run_kbm_scan)
def run_kbm_scan(*args, **kwargs):
    _sync_all_hooks()
    return _kbm_scan.run_kbm_scan(*args, **kwargs)


__all__ = ["run_kbm_beta_scan", "run_kbm_linear", "run_kbm_scan"]
