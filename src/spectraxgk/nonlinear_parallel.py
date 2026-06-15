"""Public facade for nonlinear parallelization contracts and identity gates.

The production-facing paths in this module remain policy metadata. Numerical
routes are conservative diagnostic prototypes: they only enable decomposed
updates after direct numerical identity against serial reference operations.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp

from spectraxgk.nonlinear_parallel_contracts import (
    _STRATEGIES,
    _STRATEGY_BY_NAME,
    NonlinearDomainDecompositionPlan,
    NonlinearDomainIdentityReport,
    NonlinearDomainTransportWindowReport,
    NonlinearParallelStrategy,
    NonlinearParallelStrategyName,
    NonlinearSpectralCommunicationReport,
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDevicePencilRHSIdentityReport,
    NonlinearSpectralDevicePencilTransportWindowReport,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralIntegratorIdentityReport,
    NonlinearSpectralPencilRHSIdentityReport,
    NonlinearSpectralPencilTransportWindowReport,
    NonlinearSpectralPencilWorkModel,
    NonlinearSpectralRHSIdentityReport,
    ParallelReadiness,
)


from spectraxgk.nonlinear_parallel_domain import (
    _relative_trace_error,
    _trace_drift,
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    prototype_nonlinear_domain_decomposed_step,
    prototype_nonlinear_domain_serial_step,
)


from spectraxgk.nonlinear_parallel_spectral_core import (
    _chunk_offsets as _chunk_offsets,
    _field_from_spectral_density as _field_from_spectral_density,
    _field_from_state as _field_from_state,
    _host_max_abs_rel_error as _host_max_abs_rel_error,
    _host_staged_array_for_sharding as _host_staged_array_for_sharding,
    _logical_spectral_tiles as _logical_spectral_tiles,
    _max_abs_rel_error as _max_abs_rel_error,
    _normalize_spectral_tile_bounds as _normalize_spectral_tile_bounds,
    _pencil_fft2 as _pencil_fft2,
    _pencil_ifft2 as _pencil_ifft2,
    _pencil_nonlinear_spectral_rhs as _pencil_nonlinear_spectral_rhs,
    _pencil_nonlinear_spectral_rhs_z_chunked as _pencil_nonlinear_spectral_rhs_z_chunked,
    _pencil_spectral_bracket as _pencil_spectral_bracket,
    _pencil_spectral_bracket_z_chunked as _pencil_spectral_bracket_z_chunked,
    _reconstruct_logical_spectral_tiles as _reconstruct_logical_spectral_tiles,
    _serial_nonlinear_spectral_rhs as _serial_nonlinear_spectral_rhs,
    _spectral_bracket as _spectral_bracket,
    _spectral_layout_round_trip as _spectral_layout_round_trip,
    _spectral_rhs_from_bracket as _spectral_rhs_from_bracket,
    _spectral_tile_bounds as _spectral_tile_bounds,
    _spectral_wave_numbers as _spectral_wave_numbers,
    _split_reassemble as _split_reassemble,
    _validate_chunks as _validate_chunks,
    _validate_spectral_state_shape as _validate_spectral_state_shape,
    _within_abs_or_rel_tolerance as _within_abs_or_rel_tolerance,
    deterministic_nonlinear_spectral_state,
    device_z_pencil_fft_batch_pressure_model,
    nonlinear_spectral_domain_work_model,
    nonlinear_spectral_pencil_work_model,
)

from spectraxgk.nonlinear_parallel_device_z import (
    _append_spectral_physical_observable_vector as _append_spectral_physical_observable_vector,
    _append_spectral_physical_observables as _append_spectral_physical_observables,
    _device_z_pencil_shard_map_observables_fn as _device_z_pencil_shard_map_observables_fn,
    _device_z_pencil_shard_map_rhs_fn as _device_z_pencil_shard_map_rhs_fn,
    _device_z_sharding_for_spectral_state as _device_z_sharding_for_spectral_state,
    _spectral_physical_transport_observable_sums as _spectral_physical_transport_observable_sums,
    _spectral_physical_transport_observable_vector_from_sums as _spectral_physical_transport_observable_vector_from_sums,
    _spectral_physical_transport_observables as _spectral_physical_transport_observables,
    device_z_pencil_nonlinear_spectral_rhs,
    device_z_pencil_nonlinear_spectral_transport_window_identity_gate,
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
    """Validate serial-vs-pencil nonlinear spectral RHS identity.

    The pencil route uses stacked derivative transforms and explicit axis
    transposes, avoiding the logical tile reconstruction used by the older
    diagnostic route.  Passing this report only enables the routed path when
    the communication model also predicts plausible speedup.
    """

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
    pencil_field, pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(state_hat)
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












def nonlinear_spectral_pencil_transport_window_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 1.0e-5,
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilTransportWindowReport:
    """Validate a serial-vs-pencil nonlinear transport window.

    The trace includes physical-space density/``E_r`` transport proxy values in
    addition to free energy, field energy, bracket RMS, and final state identity.
    This is a routing identity gate for the pseudo-spectral nonlinear kernel,
    not a calibrated turbulent heat-flux validation.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
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

    serial_state = state_hat
    pencil_state = state_hat
    serial_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }
    pencil_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "physical_flux": [],
        "bracket_rms": [],
    }
    _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
        serial_state
    )
    _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(
        pencil_state
    )
    _append_spectral_physical_observables(serial_traces, serial_state, serial_bracket)
    _append_spectral_physical_observables(pencil_traces, pencil_state, pencil_bracket)

    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    for _ in range(int(steps)):
        _serial_field, serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state,
        )
        _pencil_field, pencil_bracket, pencil_rhs = _pencil_nonlinear_spectral_rhs(
            pencil_state,
        )
        serial_state = serial_state + dt_array * serial_rhs
        pencil_state = pencil_state + dt_array * pencil_rhs
        _serial_field, serial_bracket, _serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state,
        )
        _pencil_field, pencil_bracket, _pencil_rhs = _pencil_nonlinear_spectral_rhs(
            pencil_state,
        )
        _append_spectral_physical_observables(
            serial_traces, serial_state, serial_bracket
        )
        _append_spectral_physical_observables(
            pencil_traces, pencil_state, pencil_bracket
        )

    state_abs, state_rel = _max_abs_rel_error(serial_state, pencil_state, atol=atol)
    serial_free = tuple(serial_traces["free_energy"])
    pencil_free = tuple(pencil_traces["free_energy"])
    serial_field_energy = tuple(serial_traces["field_energy"])
    pencil_field_energy = tuple(pencil_traces["field_energy"])
    serial_physical_flux = tuple(serial_traces["physical_flux"])
    pencil_physical_flux = tuple(pencil_traces["physical_flux"])
    serial_bracket_rms = tuple(serial_traces["bracket_rms"])
    pencil_bracket_rms = tuple(pencil_traces["bracket_rms"])
    free_abs, free_rel = _relative_trace_error(serial_free, pencil_free, floor=atol)
    field_abs, field_rel = _relative_trace_error(
        serial_field_energy,
        pencil_field_energy,
        floor=atol,
    )
    flux_abs, flux_rel = _relative_trace_error(
        serial_physical_flux,
        pencil_physical_flux,
        floor=atol,
    )
    bracket_abs, bracket_rel = _relative_trace_error(
        serial_bracket_rms,
        pencil_bracket_rms,
        floor=atol,
    )
    identity_passed = bool(
        state_abs <= float(atol)
        and state_rel <= float(rtol)
        and free_abs <= float(atol)
        and free_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and flux_abs <= float(atol)
        and flux_rel <= float(rtol)
        and bracket_abs <= float(atol)
        and bracket_rel <= float(rtol)
    )
    decomposed_path_enabled = bool(
        identity_passed and work_model.production_speedup_feasible
    )
    blocked_reasons: list[str] = []
    if not identity_passed:
        blocked_reasons.append("pencil_transport_window_identity_failed")
    blocked_reasons.extend(work_model.feasibility_blockers)

    return NonlinearSpectralPencilTransportWindowReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        final_state_max_abs_error=state_abs,
        final_state_max_rel_error=state_rel,
        free_energy_trace_max_abs_error=free_abs,
        free_energy_trace_max_rel_error=free_rel,
        field_energy_trace_max_abs_error=field_abs,
        field_energy_trace_max_rel_error=field_rel,
        physical_flux_trace_max_abs_error=flux_abs,
        physical_flux_trace_max_rel_error=flux_rel,
        bracket_rms_trace_max_abs_error=bracket_abs,
        bracket_rms_trace_max_rel_error=bracket_rel,
        serial_free_energy_drift=_trace_drift(serial_free),
        pencil_free_energy_drift=_trace_drift(pencil_free),
        identity_passed=identity_passed,
        decomposed_path_enabled=decomposed_path_enabled,
        work_model=work_model,
        claim_scope=(
            "diagnostic serial-vs-pencil nonlinear physical-space transport-window "
            "identity gate; includes a density-times-radial-E-field transport "
            "proxy, but is not an absolute nonlinear turbulent heat-flux claim"
        ),
        blocked_reasons=tuple(blocked_reasons),
        serial_free_energy_trace=serial_free,
        pencil_free_energy_trace=pencil_free,
        serial_field_energy_trace=serial_field_energy,
        pencil_field_energy_trace=pencil_field_energy,
        serial_physical_flux_trace=serial_physical_flux,
        pencil_physical_flux_trace=pencil_physical_flux,
        serial_bracket_rms_trace=serial_bracket_rms,
        pencil_bracket_rms_trace=pencil_bracket_rms,
    )


def _spectral_integrator_observables(
    state_hat: jax.Array,
) -> tuple[float, float, float]:
    field = _field_from_state(state_hat)
    free_energy = float(jnp.sum(jnp.abs(state_hat) ** 2))
    field_energy = float(jnp.sum(jnp.abs(field) ** 2))
    bracket = _spectral_bracket(state_hat, field)
    flux_proxy = float(jnp.mean(jnp.abs(_spectral_rhs_from_bracket(bracket))))
    return free_energy, field_energy, flux_proxy


def _append_spectral_observables(
    traces: dict[str, list[float]],
    state_hat: jax.Array,
) -> None:
    free_energy, field_energy, flux_proxy = _spectral_integrator_observables(state_hat)
    traces["free_energy"].append(free_energy)
    traces["field_energy"].append(field_energy)
    traces["flux_proxy"].append(flux_proxy)


def nonlinear_spectral_integrator_identity_gate(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> NonlinearSpectralIntegratorIdentityReport:
    """Validate serial-vs-logical-shard nonlinear spectral integration.

    This gate compares a fixed-step explicit-Euler micro-integration using the
    same nonlinear spectral RHS on serial and logical tiled routes. It verifies
    final-state identity and per-step free-energy, field-energy, and flux-proxy
    traces. Passing this gate is necessary but not sufficient for production
    nonlinear domain parallelization.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    serial_state = state_hat
    logical_state = state_hat
    serial_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "flux_proxy": [],
    }
    logical_traces: dict[str, list[float]] = {
        "free_energy": [],
        "field_energy": [],
        "flux_proxy": [],
    }
    _append_spectral_observables(serial_traces, serial_state)
    _append_spectral_observables(logical_traces, logical_state)
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    blocked_reasons: list[str] = []

    for _ in range(int(steps)):
        _serial_field, _serial_bracket, serial_rhs = _serial_nonlinear_spectral_rhs(
            serial_state
        )
        logical_rhs, rhs_report = logical_decomposed_nonlinear_spectral_rhs(
            logical_state,
            y_chunks=y_chunks,
            x_chunks=x_chunks,
            atol=atol,
            rtol=rtol,
        )
        blocked_reasons.extend(rhs_report.blocked_reasons)
        if not rhs_report.identity_passed:
            blocked_reasons.append("per_step_rhs_identity_failed")
        serial_state = serial_state + dt_array * serial_rhs
        logical_state = logical_state + dt_array * logical_rhs
        _append_spectral_observables(serial_traces, serial_state)
        _append_spectral_observables(logical_traces, logical_state)

    state_abs, state_rel = _max_abs_rel_error(serial_state, logical_state, atol=atol)
    serial_free = tuple(serial_traces["free_energy"])
    logical_free = tuple(logical_traces["free_energy"])
    serial_field_energy = tuple(serial_traces["field_energy"])
    logical_field_energy = tuple(logical_traces["field_energy"])
    serial_flux = tuple(serial_traces["flux_proxy"])
    logical_flux = tuple(logical_traces["flux_proxy"])
    free_abs, free_rel = _relative_trace_error(serial_free, logical_free, floor=atol)
    field_abs, field_rel = _relative_trace_error(
        serial_field_energy,
        logical_field_energy,
        floor=atol,
    )
    flux_abs, flux_rel = _relative_trace_error(serial_flux, logical_flux, floor=atol)
    unique_blockers = tuple(sorted(set(blocked_reasons)))
    identity_passed = bool(
        not unique_blockers
        and state_abs <= float(atol)
        and state_rel <= float(rtol)
        and free_abs <= float(atol)
        and free_rel <= float(rtol)
        and field_abs <= float(atol)
        and field_rel <= float(rtol)
        and flux_abs <= float(atol)
        and flux_rel <= float(rtol)
    )
    return NonlinearSpectralIntegratorIdentityReport(
        state_shape=state_shape,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        y_offsets=_chunk_offsets(y_chunks),
        x_offsets=_chunk_offsets(x_chunks),
        tile_bounds=_spectral_tile_bounds(y_chunks, x_chunks),
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
        final_state_max_abs_error=state_abs,
        final_state_max_rel_error=state_rel,
        free_energy_trace_max_abs_error=free_abs,
        free_energy_trace_max_rel_error=free_rel,
        field_energy_trace_max_abs_error=field_abs,
        field_energy_trace_max_rel_error=field_rel,
        flux_proxy_trace_max_abs_error=flux_abs,
        flux_proxy_trace_max_rel_error=flux_rel,
        serial_free_energy_drift=_trace_drift(serial_free),
        logical_free_energy_drift=_trace_drift(logical_free),
        identity_passed=identity_passed,
        decomposed_path_enabled=identity_passed,
        claim_scope=(
            "diagnostic nonlinear spectral integrator identity gate only; "
            "fixed-step serial-vs-logical-shard RHS routing with final-state "
            "and transport-proxy trace identity, no production distributed FFT "
            "routing or speedup claim"
        ),
        blocked_reasons=unique_blockers,
        serial_free_energy_trace=serial_free,
        logical_free_energy_trace=logical_free,
        serial_field_energy_trace=serial_field_energy,
        logical_field_energy_trace=logical_field_energy,
        serial_flux_proxy_trace=serial_flux,
        logical_flux_proxy_trace=logical_flux,
    )


def integrate_logical_decomposed_nonlinear_spectral(
    state_hat: jax.Array,
    *,
    y_chunks: tuple[int, ...] = (3, 3),
    x_chunks: tuple[int, ...] = (2, 2),
    dt: float = 0.005,
    steps: int = 4,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> tuple[jax.Array, NonlinearSpectralIntegratorIdentityReport]:
    """Integrate with the logical decomposed spectral route after identity gating.

    This is the callable route behind the nonlinear spectral integrator identity
    artifact.  Each step requests the logical tiled RHS; if the local RHS gate
    fails, that step uses the serial RHS instead.  The returned report is the
    independent serial-vs-routed identity gate for the full fixed-step window.
    Passing this function's report is required before timing the route, but it
    is still not a production distributed-FFT implementation or speedup claim.
    """

    if int(steps) < 1:
        raise ValueError("steps must be at least one")
    state_shape = _validate_spectral_state_shape(tuple(state_hat.shape))
    _nl, _nm, ny, nx, _nz = state_shape
    y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")

    routed_state = state_hat
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state_hat).dtype)
    for _ in range(int(steps)):
        logical_rhs, rhs_report = logical_decomposed_nonlinear_spectral_rhs(
            routed_state,
            y_chunks=y_chunks,
            x_chunks=x_chunks,
            atol=atol,
            rtol=rtol,
        )
        if rhs_report.decomposed_path_enabled:
            step_rhs = logical_rhs
        else:
            _field, _bracket, step_rhs = _serial_nonlinear_spectral_rhs(routed_state)
        routed_state = routed_state + dt_array * step_rhs

    report = nonlinear_spectral_integrator_identity_gate(
        state_hat,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        dt=dt,
        steps=steps,
        atol=atol,
        rtol=rtol,
    )
    if report.decomposed_path_enabled:
        return routed_state, report

    serial_state = state_hat
    for _ in range(int(steps)):
        _field, _bracket, serial_rhs = _serial_nonlinear_spectral_rhs(serial_state)
        serial_state = serial_state + dt_array * serial_rhs
    return serial_state, report


def nonlinear_parallel_strategies() -> tuple[NonlinearParallelStrategy, ...]:
    """Return all nonlinear parallelization strategy contracts."""

    return _STRATEGIES


def nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> NonlinearParallelStrategy:
    """Return the contract for a named nonlinear parallelization strategy."""

    try:
        return _STRATEGY_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown nonlinear parallelization strategy {name!r}"
        ) from exc


def classify_nonlinear_parallel_strategy(
    name: NonlinearParallelStrategyName,
) -> ParallelReadiness:
    """Return the release-readiness classification for a strategy."""

    return nonlinear_parallel_strategy(name).readiness


def release_ready_nonlinear_parallel_strategies() -> tuple[
    NonlinearParallelStrategy, ...
]:
    """Return production-facing strategies that do not alter solver layout."""

    return tuple(strategy for strategy in _STRATEGIES if strategy.release_ready)


__all__ = [
    "NonlinearDomainDecompositionPlan",
    "NonlinearDomainIdentityReport",
    "NonlinearDomainTransportWindowReport",
    "NonlinearParallelStrategy",
    "NonlinearParallelStrategyName",
    "NonlinearSpectralCommunicationReport",
    "NonlinearSpectralDevicePencilFFTBatchModel",
    "NonlinearSpectralDevicePencilRHSIdentityReport",
    "NonlinearSpectralDevicePencilTransportWindowReport",
    "NonlinearSpectralDomainWorkModel",
    "NonlinearSpectralIntegratorIdentityReport",
    "NonlinearSpectralPencilRHSIdentityReport",
    "NonlinearSpectralPencilTransportWindowReport",
    "NonlinearSpectralPencilWorkModel",
    "NonlinearSpectralRHSIdentityReport",
    "ParallelReadiness",
    "build_nonlinear_domain_decomposition_plan",
    "classify_nonlinear_parallel_strategy",
    "deterministic_nonlinear_domain_state",
    "deterministic_nonlinear_spectral_state",
    "device_z_pencil_fft_batch_pressure_model",
    "device_z_pencil_nonlinear_spectral_rhs",
    "device_z_pencil_nonlinear_spectral_transport_window_identity_gate",
    "integrate_logical_decomposed_nonlinear_spectral",
    "nonlinear_domain_identity_report",
    "nonlinear_domain_parallel_identity_gate",
    "nonlinear_domain_transport_window_identity_gate",
    "nonlinear_parallel_strategies",
    "nonlinear_parallel_strategy",
    "nonlinear_spectral_communication_identity_gate",
    "nonlinear_spectral_communication_identity_report",
    "nonlinear_spectral_domain_work_model",
    "logical_decomposed_nonlinear_spectral_rhs",
    "nonlinear_spectral_integrator_identity_gate",
    "nonlinear_spectral_pencil_rhs_identity_gate",
    "nonlinear_spectral_pencil_transport_window_identity_gate",
    "nonlinear_spectral_pencil_work_model",
    "nonlinear_spectral_rhs_identity_gate",
    "nonlinear_spectral_rhs_identity_report",
    "pencil_decomposed_nonlinear_spectral_rhs",
    "prototype_nonlinear_domain_decomposed_step",
    "prototype_nonlinear_domain_serial_step",
    "release_ready_nonlinear_parallel_strategies",
]
