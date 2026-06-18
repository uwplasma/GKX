"""Fail-closed reports for logical nonlinear spectral decomposition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectraxgk.operators.nonlinear.parallel_contracts import (
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralRHSIdentityReport,
)
from spectraxgk.operators.nonlinear.spectral_core import (
    _chunk_offsets,
    _field_from_state,
    _max_abs_rel_error,
    _normalize_spectral_tile_bounds,
    _spectral_bracket,
    _spectral_layout_round_trip,
    _spectral_tile_bounds,
    _validate_chunks,
    _validate_spectral_state_shape,
)


def _nonlinear_spectral_report_blockers(
    serial_fft_roundtrip: jax.Array,
    communicated_fft_roundtrip: jax.Array,
    serial_bracket: jax.Array,
    communicated_bracket: jax.Array,
    serial_field: jax.Array,
    communicated_field: jax.Array,
    *,
    state_shape: tuple[int, ...],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[str, ...]:
    blockers: list[str] = []

    try:
        normalized_state_shape = _validate_spectral_state_shape(tuple(state_shape))
    except ValueError:
        normalized_state_shape = None
        blockers.append("state_shape_invalid")

    if normalized_state_shape is not None:
        _nl, _nm, ny, nx, nz = normalized_state_shape
        try:
            _validate_chunks(ny, y_chunks, name="y_chunks")
        except ValueError:
            blockers.append("y_chunks_invalid")
        try:
            _validate_chunks(nx, x_chunks, name="x_chunks")
        except ValueError:
            blockers.append("x_chunks_invalid")

        expected_field_shape = (ny, nx, nz)
        state_arrays = (
            ("serial_fft_roundtrip", serial_fft_roundtrip),
            ("communicated_fft_roundtrip", communicated_fft_roundtrip),
            ("serial_bracket", serial_bracket),
            ("communicated_bracket", communicated_bracket),
        )
        field_arrays = (
            ("serial_field", serial_field),
            ("communicated_field", communicated_field),
        )
        for name, arr in state_arrays:
            if tuple(arr.shape) != normalized_state_shape:
                blockers.append(f"{name}_shape_mismatch")
        for name, arr in field_arrays:
            if tuple(arr.shape) != expected_field_shape:
                blockers.append(f"{name}_shape_mismatch")

    return tuple(blockers)


def nonlinear_spectral_communication_identity_report(
    serial_fft_roundtrip: jax.Array,
    communicated_fft_roundtrip: jax.Array,
    serial_bracket: jax.Array,
    communicated_bracket: jax.Array,
    serial_field: jax.Array,
    communicated_field: jax.Array,
    *,
    state_shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralCommunicationReport:
    """Compare spectral communication outputs and fail closed on mismatches."""

    blocked_reasons = _nonlinear_spectral_report_blockers(
        serial_fft_roundtrip,
        communicated_fft_roundtrip,
        serial_bracket,
        communicated_bracket,
        serial_field,
        communicated_field,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    fft_abs, fft_rel = _max_abs_rel_error(
        serial_fft_roundtrip,
        communicated_fft_roundtrip,
        atol=atol,
    )
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket,
        communicated_bracket,
        atol=atol,
    )
    field_abs, field_rel = _max_abs_rel_error(
        serial_field,
        communicated_field,
        atol=atol,
    )
    identity_passed = bool(
        not blocked_reasons
        and fft_abs <= float(atol)
        and fft_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
    )
    return NonlinearSpectralCommunicationReport(
        state_shape=state_shape,
        y_chunks=tuple(int(item) for item in y_chunks),
        x_chunks=tuple(int(item) for item in x_chunks),
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        atol=float(atol),
        rtol=float(rtol),
        fft_max_abs_error=fft_abs,
        fft_max_rel_error=fft_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic spectral communication identity gate only; "
            "split/reassemble layout simulation with no production routing or speedup claim"
        ),
        blocked_reasons=blocked_reasons,
    )


def nonlinear_spectral_communication_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralCommunicationReport:
    """Validate FFT, bracket, and field layout identity under split/reassemble."""

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    communicated_state = _spectral_layout_round_trip(
        state_hat,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    serial_fft = jnp.fft.fft2(jnp.fft.ifft2(state_hat, axes=(-3, -2)), axes=(-3, -2))
    communicated_fft = jnp.fft.fft2(
        jnp.fft.ifft2(communicated_state, axes=(-3, -2)),
        axes=(-3, -2),
    )
    serial_field = _field_from_state(state_hat)
    communicated_field = _field_from_state(communicated_state)
    serial_bracket = _spectral_bracket(state_hat, serial_field)
    communicated_bracket = _spectral_bracket(communicated_state, communicated_field)

    return nonlinear_spectral_communication_identity_report(
        serial_fft,
        communicated_fft,
        serial_bracket,
        communicated_bracket,
        serial_field,
        communicated_field,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )


def _nonlinear_spectral_rhs_report_blockers(
    serial_reconstruction: jax.Array,
    logical_reconstruction: jax.Array,
    serial_field: jax.Array,
    logical_field: jax.Array,
    serial_bracket: jax.Array,
    logical_bracket: jax.Array,
    serial_rhs: jax.Array,
    logical_rhs: jax.Array,
    *,
    state_shape: tuple[int, ...],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    tile_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[str, ...]:
    blockers: list[str] = []

    try:
        normalized_state_shape = _validate_spectral_state_shape(tuple(state_shape))
    except ValueError:
        normalized_state_shape = None
        blockers.append("state_shape_invalid")

    normalized_y_chunks: tuple[int, ...] | None = None
    normalized_x_chunks: tuple[int, ...] | None = None
    if normalized_state_shape is not None:
        _nl, _nm, ny, nx, nz = normalized_state_shape
        try:
            normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
        except ValueError:
            blockers.append("y_chunks_invalid")
        try:
            normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
        except ValueError:
            blockers.append("x_chunks_invalid")

        expected_field_shape = (ny, nx, nz)
        state_arrays = (
            ("serial_reconstruction", serial_reconstruction),
            ("logical_reconstruction", logical_reconstruction),
            ("serial_bracket", serial_bracket),
            ("logical_bracket", logical_bracket),
            ("serial_rhs", serial_rhs),
            ("logical_rhs", logical_rhs),
        )
        field_arrays = (
            ("serial_field", serial_field),
            ("logical_field", logical_field),
        )
        for name, arr in state_arrays:
            if tuple(arr.shape) != normalized_state_shape:
                blockers.append(f"{name}_shape_mismatch")
        for name, arr in field_arrays:
            if tuple(arr.shape) != expected_field_shape:
                blockers.append(f"{name}_shape_mismatch")

    if normalized_y_chunks is not None and normalized_x_chunks is not None:
        if tile_bounds != _spectral_tile_bounds(
            normalized_y_chunks, normalized_x_chunks
        ):
            blockers.append("tile_bounds_not_row_major")

    return tuple(blockers)


def nonlinear_spectral_rhs_identity_report(
    serial_reconstruction: jax.Array,
    logical_reconstruction: jax.Array,
    serial_field: jax.Array,
    logical_field: jax.Array,
    serial_bracket: jax.Array,
    logical_bracket: jax.Array,
    serial_rhs: jax.Array,
    logical_rhs: jax.Array,
    *,
    state_shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    tile_bounds: tuple[tuple[int, int, int, int], ...] | None = None,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralRHSIdentityReport:
    """Compare serial and logical-shard spectral RHS outputs fail-closed."""

    normalized_y_chunks = tuple(int(item) for item in y_chunks)
    normalized_x_chunks = tuple(int(item) for item in x_chunks)
    effective_tile_bounds = (
        _spectral_tile_bounds(normalized_y_chunks, normalized_x_chunks)
        if tile_bounds is None
        else _normalize_spectral_tile_bounds(tile_bounds)
    )
    blocked_reasons = _nonlinear_spectral_rhs_report_blockers(
        serial_reconstruction,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        tile_bounds=effective_tile_bounds,
    )
    reconstruction_abs, reconstruction_rel = _max_abs_rel_error(
        serial_reconstruction,
        logical_reconstruction,
        atol=atol,
    )
    field_abs, field_rel = _max_abs_rel_error(
        serial_field,
        logical_field,
        atol=atol,
    )
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket,
        logical_bracket,
        atol=atol,
    )
    rhs_abs, rhs_rel = _max_abs_rel_error(
        serial_rhs,
        logical_rhs,
        atol=atol,
    )
    identity_passed = bool(
        not blocked_reasons
        and reconstruction_abs <= float(atol)
        and reconstruction_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and rhs_abs <= float(atol)
        and rhs_rel <= float(rtol)
    )
    return NonlinearSpectralRHSIdentityReport(
        state_shape=state_shape,
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        tile_bounds=effective_tile_bounds,
        atol=float(atol),
        rtol=float(rtol),
        reconstruction_max_abs_error=reconstruction_abs,
        reconstruction_max_rel_error=reconstruction_rel,
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        rhs_max_abs_error=rhs_abs,
        rhs_max_rel_error=rhs_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear spectral RHS identity gate only; "
            "logical output-tile reconstruction with existing bracket contribution "
            "and no production routing or speedup claim"
        ),
        blocked_reasons=blocked_reasons,
    )


__all__ = [
    "_nonlinear_spectral_report_blockers",
    "_nonlinear_spectral_rhs_report_blockers",
    "nonlinear_spectral_communication_identity_gate",
    "nonlinear_spectral_communication_identity_report",
    "nonlinear_spectral_rhs_identity_report",
]
