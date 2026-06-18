"""Logical RHS routing for nonlinear spectral decomposition diagnostics."""

from __future__ import annotations

import jax

from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralRHSIdentityReport,
)
from spectraxgk.operators.nonlinear.spectral_core import (
    _field_from_state,
    _logical_spectral_tiles,
    _reconstruct_logical_spectral_tiles,
    _serial_nonlinear_spectral_rhs,
    _spectral_bracket,
    _spectral_rhs_from_bracket,
    _spectral_tile_bounds,
    _validate_chunks,
    _validate_spectral_state_shape,
)
from spectraxgk.operators.nonlinear.spectral_identity_reports import (
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


__all__ = [
    "_logical_sharded_nonlinear_spectral_rhs",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_rhs_identity_gate",
]
