"""Stable facade for nonlinear spectral parallelization helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

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
from spectraxgk.operators.nonlinear.spectral_work_models import (
    device_z_pencil_fft_batch_pressure_model,
    nonlinear_spectral_domain_work_model,
    nonlinear_spectral_pencil_work_model,
)


def _max_abs_rel_error(
    reference: jax.Array,
    candidate: jax.Array,
    *,
    atol: float,
) -> tuple[float, float]:
    if tuple(reference.shape) != tuple(candidate.shape):
        return float("inf"), float("inf")
    abs_error = jnp.abs(candidate - reference)
    scale = jnp.maximum(
        jnp.abs(reference), jnp.asarray(atol, dtype=jnp.real(abs_error).dtype)
    )
    rel_error = abs_error / scale
    return float(jnp.max(abs_error)), float(jnp.max(rel_error))


def _host_max_abs_rel_error(
    reference: jax.Array,
    candidate: jax.Array,
    *,
    atol: float,
) -> tuple[float, float]:
    """Return max errors after materializing arrays on the host."""

    reference_host = np.asarray(jax.device_get(reference))
    candidate_host = np.asarray(jax.device_get(candidate))
    if reference_host.shape != candidate_host.shape:
        return float("inf"), float("inf")
    abs_error = np.abs(candidate_host - reference_host)
    scale = np.maximum(np.abs(reference_host), float(atol))
    rel_error = abs_error / scale
    return float(np.max(abs_error)), float(np.max(rel_error))


def _within_abs_or_rel_tolerance(
    max_abs_error: float,
    max_rel_error: float,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Return an allclose-style scalar gate for recorded max errors."""

    return bool(max_abs_error <= float(atol) or max_rel_error <= float(rtol))


def _host_staged_array_for_sharding(array: jax.Array) -> np.ndarray:
    """Return a host-backed array before applying explicit device sharding.

    On the CUDA stack used for the current device-z diagnostic, direct
    ``device_put`` from a single-device JAX array into a z-sharded
    ``NamedSharding`` can misplace the second z shard. Host staging keeps the
    identity gate about the candidate nonlinear route instead of about that
    source-device resharding behavior.
    """

    return np.asarray(jax.device_get(array))


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
