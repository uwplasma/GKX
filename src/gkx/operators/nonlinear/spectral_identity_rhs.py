"""Logical RHS routing for nonlinear spectral decomposition diagnostics."""

from __future__ import annotations

import jax

from gkx.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralRHSIdentityReport,
)
from gkx.operators.nonlinear.spectral_core import (
    _chunk_offsets,
    _field_from_state,
    _logical_spectral_tiles,
    _max_abs_rel_error,
    _pencil_nonlinear_spectral_rhs,
    _reconstruct_logical_spectral_tiles,
    _serial_nonlinear_spectral_rhs,
    _spectral_bracket,
    _spectral_rhs_from_bracket,
    _spectral_tile_bounds,
    _validate_chunks,
    _validate_spectral_state_shape,
    nonlinear_spectral_pencil_work_model,
)
from gkx.operators.nonlinear.spectral_identity_reports import (
    nonlinear_spectral_rhs_identity_report,
)


def _logical_sharded_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    state_tiles = _logical_spectral_tiles(
        state_hat,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    reconstructed_state = _reconstruct_logical_spectral_tiles(
        state_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    field = _field_from_state(reconstructed_state)
    bracket = _spectral_bracket(reconstructed_state, field)
    bracket_tiles = _logical_spectral_tiles(
        bracket,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    logical_bracket = _reconstruct_logical_spectral_tiles(
        bracket_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    rhs_tiles = tuple(_spectral_rhs_from_bracket(tile) for tile in bracket_tiles)
    logical_rhs = _reconstruct_logical_spectral_tiles(
        rhs_tiles,
        y_axis=-3,
        x_axis=-2,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    return reconstructed_state, field, logical_bracket, logical_rhs


def nonlinear_spectral_rhs_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralRHSIdentityReport:
    """Validate serial-vs-logical-shard nonlinear spectral RHS identity.

    This diagnostic route owns and reassembles spectral ``(y, x)`` output tiles
    in row-major order. It deliberately does not install distributed FFT
    runtime routing or make a speedup claim.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    (
        logical_reconstruction,
        logical_field,
        logical_bracket,
        logical_rhs,
    ) = _logical_sharded_nonlinear_spectral_rhs(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )

    return nonlinear_spectral_rhs_identity_report(
        state_hat,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        atol=atol,
        rtol=rtol,
    )


def logical_decomposed_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> tuple[jax.Array, NonlinearSpectralRHSIdentityReport]:
    """Return the logical-shard nonlinear spectral RHS after identity gating.

    The returned RHS uses the logical decomposed route only when it is exactly
    equivalent to the serial reference under the provided tolerances. Otherwise
    the serial RHS is returned and ``decomposed_path_enabled`` is false. This is
    still a local diagnostic route, not a distributed runtime implementation.
    """

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    (
        logical_reconstruction,
        logical_field,
        logical_bracket,
        logical_rhs,
    ) = _logical_sharded_nonlinear_spectral_rhs(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
    )
    report = nonlinear_spectral_rhs_identity_report(
        state_hat,
        logical_reconstruction,
        serial_field,
        logical_field,
        serial_bracket,
        logical_bracket,
        serial_rhs,
        logical_rhs,
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = logical_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


def nonlinear_spectral_pencil_rhs_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilRHSIdentityReport:
    """Validate serial-vs-pencil nonlinear spectral RHS identity."""

    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    work_model = nonlinear_spectral_pencil_work_model(
        state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        max_communication_to_fft_work_ratio=max_communication_to_fft_work_ratio,
        min_predicted_speedup=min_predicted_speedup,
    )

    serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(state_hat)
    pencil_field, pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
        state_hat
    )
    field_abs, field_rel = _max_abs_rel_error(serial_field, pencil_field, atol=atol)
    bracket_abs, bracket_rel = _max_abs_rel_error(
        serial_bracket, pencil_bracket, atol=atol
    )
    rhs_abs, rhs_rel = _max_abs_rel_error(serial_rhs, pencil_rhs, atol=atol)
    identity_passed = bool(
        field_abs <= float(atol)
        and field_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
        and rhs_abs <= float(atol)
        and rhs_rel <= float(rtol)
    )
    decomposed_path_enabled = bool(
        identity_passed and work_model.production_speedup_feasible
    )
    blocked_reasons: list[str] = []
    if not identity_passed:
        blocked_reasons.append("pencil_rhs_identity_failed")
    blocked_reasons.extend(work_model.feasibility_blockers)

    return NonlinearSpectralPencilRHSIdentityReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        atol=float(atol),
        rtol=float(rtol),
        field_max_abs_error=field_abs,
        field_max_rel_error=field_rel,
        bracket_max_abs_error=bracket_abs,
        bracket_max_rel_error=bracket_rel,
        rhs_max_abs_error=rhs_abs,
        rhs_max_rel_error=rhs_rel,
        identity_passed=identity_passed,
        decomposed_path_enabled=decomposed_path_enabled,
        work_model=work_model,
        claim_scope=(
            "diagnostic pencil-FFT nonlinear spectral RHS identity gate; "
            "axis-wise FFT staging with no global reconstruction, but no "
            "production speedup claim without profiler-backed scaling"
        ),
        blocked_reasons=tuple(blocked_reasons),
    )


def pencil_decomposed_nonlinear_spectral_rhs(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
) -> tuple[jax.Array, NonlinearSpectralPencilRHSIdentityReport]:
    """Return the pencil nonlinear spectral RHS after identity/model gating."""

    _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
        state_hat
    )
    _pencil_field, _pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
        state_hat
    )
    report = nonlinear_spectral_pencil_rhs_identity_gate(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    gated_rhs = pencil_rhs if report.decomposed_path_enabled else serial_rhs
    return gated_rhs, report


__all__ = [
    "_logical_sharded_nonlinear_spectral_rhs",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_pencil_rhs_identity_gate",
    "nonlinear_spectral_rhs_identity_gate",
    "pencil_decomposed_nonlinear_spectral_rhs",
]
