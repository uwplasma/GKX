"""Compatibility facade for benchmark helper policies.

Implementation is split into focused benchmark helper modules while preserving
legacy imports from ``spectraxgk.benchmark_helpers`` and the public
``spectraxgk.benchmarks`` facade.
"""

from __future__ import annotations

from spectraxgk.benchmark_batching import (
    _is_array_like,
    _iter_ky_batches,
    _resolve_streaming_window,
)
from spectraxgk.benchmark_fit_signals import (
    _extract_mode_only_signal,
    _normalize_growth_rate,
    _score_fit_signal_auto,
    _select_fit_signal,
    _select_fit_signal_auto,
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
    "load_cyclone_reference",
    "load_cyclone_reference_kinetic",
    "load_etg_reference",
    "load_kbm_reference",
    "load_tem_reference",
    "select_kbm_solver_auto",
]
