"""Public quasilinear transport facade.

The implementation lives in :mod:`spectraxgk.diagnostics.quasilinear_transport`
with the diagnostic owners.  This module remains the stable public import path
used by examples and downstream scripts.
"""

from __future__ import annotations

from spectraxgk.diagnostics.quasilinear_transport import (
    QuasilinearTransportResult,
    compute_quasilinear_from_linear_state,
    effective_kperp2,
    mixing_length_amplitude2_jax,
    normalize_quasilinear_channels,
    phi_norm2,
    quasilinear_feature_objective,
    saturation_amplitude2,
    saturated_flux_from_linear_weight,
    shape_aware_power_law_objective,
    spectral_phi_weights,
)

__all__ = [
    "QuasilinearTransportResult",
    "compute_quasilinear_from_linear_state",
    "effective_kperp2",
    "mixing_length_amplitude2_jax",
    "normalize_quasilinear_channels",
    "phi_norm2",
    "quasilinear_feature_objective",
    "saturation_amplitude2",
    "saturated_flux_from_linear_weight",
    "shape_aware_power_law_objective",
    "spectral_phi_weights",
]
