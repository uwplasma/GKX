"""Stable facade for nonlinear spectral parallelization helpers."""

from __future__ import annotations

from spectraxgk.operators.nonlinear.spectral_brackets import (
    _field_from_spectral_density,
    _field_from_state,
    _pencil_fft2,
    _pencil_ifft2,
    _pencil_nonlinear_spectral_rhs,
    _pencil_nonlinear_spectral_rhs_z_chunked,
    _pencil_spectral_bracket,
    _pencil_spectral_bracket_z_chunked,
    _serial_nonlinear_spectral_rhs,
    _spectral_bracket,
    _spectral_rhs_from_bracket,
    _spectral_wave_numbers,
)
from spectraxgk.operators.nonlinear.spectral_layout import (
    _chunk_offsets,
    _logical_spectral_tiles,
    _normalize_spectral_tile_bounds,
    _reconstruct_logical_spectral_tiles,
    _spectral_layout_round_trip,
    _spectral_tile_bounds,
    _split_reassemble,
    _validate_chunks,
    _validate_spectral_state_shape,
    deterministic_nonlinear_spectral_state,
)
from spectraxgk.operators.nonlinear.spectral_tolerances import (
    _host_max_abs_rel_error,
    _host_staged_array_for_sharding,
    _max_abs_rel_error,
    _within_abs_or_rel_tolerance,
)
from spectraxgk.operators.nonlinear.spectral_work_models import (
    device_z_pencil_fft_batch_pressure_model,
    nonlinear_spectral_domain_work_model,
    nonlinear_spectral_pencil_work_model,
)

__all__ = [
    "_chunk_offsets",
    "_field_from_spectral_density",
    "_field_from_state",
    "_host_max_abs_rel_error",
    "_host_staged_array_for_sharding",
    "_logical_spectral_tiles",
    "_max_abs_rel_error",
    "_normalize_spectral_tile_bounds",
    "_pencil_fft2",
    "_pencil_ifft2",
    "_pencil_nonlinear_spectral_rhs",
    "_pencil_nonlinear_spectral_rhs_z_chunked",
    "_pencil_spectral_bracket",
    "_pencil_spectral_bracket_z_chunked",
    "_reconstruct_logical_spectral_tiles",
    "_serial_nonlinear_spectral_rhs",
    "_spectral_bracket",
    "_spectral_layout_round_trip",
    "_spectral_rhs_from_bracket",
    "_spectral_tile_bounds",
    "_spectral_wave_numbers",
    "_split_reassemble",
    "_validate_chunks",
    "_validate_spectral_state_shape",
    "_within_abs_or_rel_tolerance",
    "deterministic_nonlinear_spectral_state",
    "device_z_pencil_fft_batch_pressure_model",
    "nonlinear_spectral_domain_work_model",
    "nonlinear_spectral_pencil_work_model",
]
