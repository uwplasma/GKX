"""Compatibility facade for benchmark helper policies.

Implementation is split into focused benchmark helper modules while preserving
legacy imports from ``spectraxgk.benchmark_helpers`` and the public
``spectraxgk.benchmarks`` facade.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from spectraxgk.analysis import (
    extract_mode_time_series,
    fit_growth_rate_auto_with_stats,
)
import spectraxgk.benchmark_fit_signals as _fit_signals
from spectraxgk.benchmark_batching import (
    _is_array_like,
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.benchmark_fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
)
from spectraxgk.benchmark_initialization import (
    _build_gaussian_profile,
    _build_initial_condition,
    _kinetic_reference_init_cfg,
)
from spectraxgk.benchmark_reference import (
    CycloneComparison,
    CycloneReference,
    CycloneRunResult,
    CycloneScanResult,
    LinearRunResult,
    LinearScanResult,
    _load_reference_with_header,
    compare_cyclone_to_reference,
    load_cyclone_reference,
    load_cyclone_reference_kinetic,
    load_etg_reference,
    load_kbm_reference,
    load_tem_reference,
)
from spectraxgk.benchmark_solver_policy import (
    KBM_GX_SOLVER_LOCK,
    KBM_GX_SOLVER_LOCK_TOL,
    _kbm_use_multi_target_krylov,
    _midplane_index,
    select_kbm_solver_auto,
)
from spectraxgk.benchmark_species import (
    REFERENCE_DAMP_ENDS_AMP,
    REFERENCE_DAMP_ENDS_WIDTHFRAC,
    REFERENCE_NU_HYPER_L,
    REFERENCE_NU_HYPER_M,
    REFERENCE_P_HYPER_L,
    REFERENCE_P_HYPER_M,
    _apply_gx_hypercollisions,
    _electron_only_params,
    _gx_linked_end_damping,
    _gx_p_hyper_m,
    _two_species_params,
)


_DEFAULT_EXTRACT_MODE_TIME_SERIES = extract_mode_time_series
_DEFAULT_FIT_GROWTH_RATE_AUTO_WITH_STATS = fit_growth_rate_auto_with_stats


def _call_with_facade_mode_extractor(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run fit-signal helpers through the compatibility facade extractor.

    Older tests and user code monkeypatch ``spectraxgk.benchmark_helpers``
    directly.  The focused implementation lives in ``benchmark_fit_signals``;
    this thin bridge preserves the legacy indirection without duplicating the
    fit-selection policy.
    """

    if extract_mode_time_series is _DEFAULT_EXTRACT_MODE_TIME_SERIES:
        return func(*args, **kwargs)
    original = _fit_signals.extract_mode_time_series
    _fit_signals.extract_mode_time_series = extract_mode_time_series
    try:
        return func(*args, **kwargs)
    finally:
        _fit_signals.extract_mode_time_series = original


def _call_with_facade_growth_fit(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    if fit_growth_rate_auto_with_stats is _DEFAULT_FIT_GROWTH_RATE_AUTO_WITH_STATS:
        return func(*args, **kwargs)
    original = _fit_signals.fit_growth_rate_auto_with_stats
    _fit_signals.fit_growth_rate_auto_with_stats = fit_growth_rate_auto_with_stats
    try:
        return func(*args, **kwargs)
    finally:
        _fit_signals.fit_growth_rate_auto_with_stats = original


def _select_fit_signal(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_mode_extractor(
        _fit_signals._select_fit_signal, *args, **kwargs
    )


def _score_fit_signal_auto(*args: Any, **kwargs: Any) -> Any:
    return _call_with_facade_growth_fit(
        _fit_signals._score_fit_signal_auto, *args, **kwargs
    )


def _select_fit_signal_auto(*args: Any, **kwargs: Any) -> Any:
    original_extract = _fit_signals.extract_mode_time_series
    original_score = _fit_signals._score_fit_signal_auto
    patch_extract = extract_mode_time_series is not _DEFAULT_EXTRACT_MODE_TIME_SERIES
    patch_score = (
        fit_growth_rate_auto_with_stats is not _DEFAULT_FIT_GROWTH_RATE_AUTO_WITH_STATS
    )

    def _facade_score(*score_args: Any, **score_kwargs: Any) -> Any:
        return _call_with_facade_growth_fit(original_score, *score_args, **score_kwargs)

    if patch_extract:
        _fit_signals.extract_mode_time_series = extract_mode_time_series
    if patch_score:
        _fit_signals._score_fit_signal_auto = _facade_score
    try:
        return _fit_signals._select_fit_signal_auto(*args, **kwargs)
    finally:
        if patch_extract:
            _fit_signals.extract_mode_time_series = original_extract
        if patch_score:
            _fit_signals._score_fit_signal_auto = original_score


__all__ = [
    "REFERENCE_NU_HYPER_L",
    "REFERENCE_NU_HYPER_M",
    "REFERENCE_P_HYPER_L",
    "REFERENCE_P_HYPER_M",
    "REFERENCE_DAMP_ENDS_AMP",
    "REFERENCE_DAMP_ENDS_WIDTHFRAC",
    "KBM_GX_SOLVER_LOCK",
    "KBM_GX_SOLVER_LOCK_TOL",
    "CycloneComparison",
    "CycloneReference",
    "CycloneRunResult",
    "CycloneScanResult",
    "LinearRunResult",
    "LinearScanResult",
    "_apply_gx_hypercollisions",
    "_build_gaussian_profile",
    "_build_initial_condition",
    "_electron_only_params",
    "_extract_mode_only_signal",
    "extract_mode_time_series",
    "_gx_linked_end_damping",
    "_gx_p_hyper_m",
    "_is_array_like",
    "_iter_ky_batches",
    "_kbm_use_multi_target_krylov",
    "_kinetic_reference_init_cfg",
    "_load_reference_with_header",
    "_midplane_index",
    "_normalize_growth_rate",
    "_resolve_streaming_window",
    "_score_fit_signal_auto",
    "_select_fit_signal",
    "_select_fit_signal_auto",
    "_two_species_params",
    "compare_cyclone_to_reference",
    "fit_growth_rate_auto_with_stats",
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
    "select_kbm_solver_auto",
]
