"""Stable facade for nonlinear spectral parallelization helpers."""

from __future__ import annotations

import jax
import math
import jax.numpy as jnp
import numpy as np
from typing import Any

from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralPencilWorkModel,
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


def _spectral_wave_numbers(ny: int, nx: int, dtype: Any) -> tuple[jax.Array, jax.Array]:
    ky = jnp.fft.fftfreq(ny, d=1.0 / float(ny)).astype(dtype)
    kx = jnp.fft.fftfreq(nx, d=1.0 / float(nx)).astype(dtype)
    return ky, kx


def _field_from_spectral_density(density_hat: jax.Array) -> jax.Array:
    ny, nx, _nz = (int(item) for item in density_hat.shape)
    real_dtype = jnp.real(density_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    kperp2 = ky[:, None, None] ** 2 + kx[None, :, None] ** 2
    phi_hat = density_hat / (1.0 + kperp2)
    return phi_hat.at[0, 0, :].set(0.0)


def _field_from_state(state_hat: jax.Array) -> jax.Array:
    density_hat = jnp.sum(state_hat[:, 0, :, :, :], axis=0)
    return _field_from_spectral_density(density_hat)


def _spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_dx = jnp.fft.ifft2(1j * kx_state * state_hat, axes=(-3, -2))
    state_dy = jnp.fft.ifft2(1j * ky_state * state_hat, axes=(-3, -2))
    phi_dx = jnp.fft.ifft2(1j * kx_field * phi_hat, axes=(0, 1))
    phi_dy = jnp.fft.ifft2(1j * ky_field * phi_hat, axes=(0, 1))
    bracket_xy = (
        phi_dx[None, None, :, :, :] * state_dy - phi_dy[None, None, :, :, :] * state_dx
    )
    return jnp.fft.fft2(bracket_xy, axes=(-3, -2))


def _pencil_ifft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D inverse FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.ifft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.ifft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_fft2(arr: jax.Array, *, y_axis: int, x_axis: int) -> jax.Array:
    """Return a 2D forward FFT through explicit x-then-y pencil stages."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    x_transformed = jnp.fft.fft(arr, axis=x_axis)
    transposed = jnp.swapaxes(x_transformed, y_axis, x_axis)
    y_transformed = jnp.fft.fft(transposed, axis=x_axis)
    return jnp.swapaxes(y_transformed, y_axis, x_axis)


def _pencil_spectral_bracket(state_hat: jax.Array, phi_hat: jax.Array) -> jax.Array:
    """Return the pseudo-spectral bracket using pencil FFT staging.

    This function is the local algorithmic route that a distributed pencil FFT
    implementation should follow: stack derivative operands, transform through
    explicit axis-transpose stages, multiply in physical space, and transform
    the bracket back without first reconstructing logical output tiles.
    """

    _nl, _nm, ny, nx, _nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    real_dtype = jnp.real(state_hat).dtype
    ky, kx = _spectral_wave_numbers(ny, nx, real_dtype)
    ky_state = ky[None, None, :, None, None]
    kx_state = kx[None, None, None, :, None]
    ky_field = ky[:, None, None]
    kx_field = kx[None, :, None]

    state_grad_hat = jnp.stack(
        [1j * kx_state * state_hat, 1j * ky_state * state_hat],
        axis=0,
    )
    state_grad_xy = _pencil_ifft2(state_grad_hat, y_axis=-3, x_axis=-2)
    state_dx = state_grad_xy[0]
    state_dy = state_grad_xy[1]

    field_grad_hat = jnp.stack(
        [1j * kx_field * phi_hat, 1j * ky_field * phi_hat],
        axis=0,
    )
    field_grad_xy = _pencil_ifft2(field_grad_hat, y_axis=1, x_axis=2)
    phi_dx = field_grad_xy[0]
    phi_dy = field_grad_xy[1]

    bracket_xy = (
        phi_dx[None, None, :, :, :] * state_dy - phi_dy[None, None, :, :, :] * state_dx
    )
    return _pencil_fft2(bracket_xy, y_axis=-3, x_axis=-2)


def _pencil_spectral_bracket_z_chunked(
    state_hat: jax.Array,
    phi_hat: jax.Array,
    *,
    z_chunk_size: int,
) -> jax.Array:
    """Return the pencil bracket by processing independent z slabs.

    The nonlinear bracket has no coupling along ``z`` inside this local
    pseudo-spectral micro-route. Chunking the local z extent therefore preserves
    the operator while reducing cuFFT batched-plan pressure on GPUs.
    """

    _nl, _nm, _ny, _nx, nz = _validate_spectral_state_shape(tuple(state_hat.shape))
    chunk_size = int(z_chunk_size)
    if chunk_size < 1:
        raise ValueError("z_chunk_size must be at least one")
    if chunk_size >= nz:
        return _pencil_spectral_bracket(state_hat, phi_hat)

    bracket_chunks: list[jax.Array] = []
    for start in range(0, nz, chunk_size):
        size = min(chunk_size, nz - start)
        state_chunk = jax.lax.dynamic_slice_in_dim(
            state_hat,
            start,
            size,
            axis=-1,
        )
        phi_chunk = jax.lax.dynamic_slice_in_dim(
            phi_hat,
            start,
            size,
            axis=-1,
        )
        bracket_chunks.append(_pencil_spectral_bracket(state_chunk, phi_chunk))
    return jnp.concatenate(bracket_chunks, axis=-1)


def _spectral_rhs_from_bracket(bracket_hat: jax.Array) -> jax.Array:
    """Return the ExB advection contribution used by the identity micro-route."""

    return -bracket_hat


def _serial_nonlinear_spectral_rhs(
    state_hat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _pencil_nonlinear_spectral_rhs(
    state_hat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _pencil_spectral_bracket(state_hat, field)
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def _pencil_nonlinear_spectral_rhs_z_chunked(
    state_hat: jax.Array,
    *,
    z_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    field = _field_from_state(state_hat)
    bracket = _pencil_spectral_bracket_z_chunked(
        state_hat,
        field,
        z_chunk_size=z_chunk_size,
    )
    rhs = _spectral_rhs_from_bracket(bracket)
    return field, bracket, rhs


def nonlinear_spectral_domain_work_model(
    state_shape: tuple[int, int, int, int, int],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    max_communication_to_owned_work_ratio: float = 0.5,
) -> NonlinearSpectralDomainWorkModel:
    """Estimate communication pressure for the current logical spectral route.

    The current diagnostic route reconstructs global spectral state/bracket
    arrays before returning owned output tiles. That is useful for identity
    gating, but it implies allgather/broadcast traffic that can dominate the
    owned tile work. This model is a conservative fail-closed screen for that
    route; it is not a performance prediction for a future distributed FFT.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(state_shape))
    normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    num_tiles = len(normalized_y_chunks) * len(normalized_x_chunks)
    state_elements = int(nl * nm * ny * nx * nz)
    field_elements = int(ny * nx * nz)
    communication_factor = max(num_tiles - 1, 0)
    state_allgather_elements = int(state_elements * communication_factor)
    bracket_allgather_elements = int(state_elements * communication_factor)
    field_broadcast_elements = int(field_elements * communication_factor)
    total_communication_elements = (
        state_allgather_elements + bracket_allgather_elements + field_broadcast_elements
    )
    owned_state_elements = state_elements
    ratio = (
        float(total_communication_elements) / float(owned_state_elements)
        if owned_state_elements > 0
        else float("inf")
    )
    efficiency_ceiling = 1.0 / (1.0 + ratio) if ratio >= 0.0 else 0.0

    blockers: list[str] = []
    if num_tiles < 2:
        blockers.append("single_tile_no_domain_decomposition")
    if ratio > float(max_communication_to_owned_work_ratio):
        blockers.append("global_reconstruction_communication_dominates_owned_work")
    production_speedup_feasible = bool(not blockers)

    return NonlinearSpectralDomainWorkModel(
        state_shape=(nl, nm, ny, nx, nz),
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        tile_bounds=_spectral_tile_bounds(normalized_y_chunks, normalized_x_chunks),
        num_tiles=num_tiles,
        state_elements=state_elements,
        field_elements=field_elements,
        owned_state_elements_per_step=owned_state_elements,
        state_allgather_elements_per_step=state_allgather_elements,
        bracket_allgather_elements_per_step=bracket_allgather_elements,
        field_broadcast_elements_per_step=field_broadcast_elements,
        total_communication_elements_per_step=total_communication_elements,
        communication_to_owned_work_ratio=ratio,
        parallel_efficiency_ceiling=efficiency_ceiling,
        max_communication_to_owned_work_ratio=float(
            max_communication_to_owned_work_ratio
        ),
        production_speedup_feasible=production_speedup_feasible,
        feasibility_blockers=tuple(blockers),
        claim_scope=(
            "diagnostic communication/work model for the current global-reconstruction "
            "logical spectral route; not a distributed FFT performance claim"
        ),
    )


def nonlinear_spectral_pencil_work_model(
    state_shape: tuple[int, int, int, int, int],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    max_communication_to_fft_work_ratio: float = 0.35,
    min_predicted_speedup: float = 1.5,
) -> NonlinearSpectralPencilWorkModel:
    """Estimate communication pressure for a pencil-FFT bracket route.

    The pencil route avoids global state/bracket reconstruction. Its remaining
    distributed cost is the all-to-all transpose traffic needed by axis-wise 2D
    FFTs plus a field-reduction/broadcast. This model is intentionally simple
    and fail-closed: it must pass before any strong-scaling timing is treated as
    a meaningful candidate for production promotion.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(state_shape))
    normalized_y_chunks = _validate_chunks(ny, y_chunks, name="y_chunks")
    normalized_x_chunks = _validate_chunks(nx, x_chunks, name="x_chunks")
    num_tiles = len(normalized_y_chunks) * len(normalized_x_chunks)
    state_elements = int(nl * nm * ny * nx * nz)
    field_elements = int(ny * nx * nz)

    # Fused bracket payload: inverse FFTs for two state gradients, inverse FFTs
    # for two field gradients, and one forward FFT of the bracket.
    transform_payload = int(3 * state_elements + 2 * field_elements)
    sent_fraction = 1.0 - (1.0 / float(num_tiles)) if num_tiles > 0 else 1.0
    pencil_transpose_elements = int(round(transform_payload * sent_fraction))
    fft_extent = max(int(ny * nx), 2)
    approximate_fft_work = float(transform_payload) * math.log2(float(fft_extent))
    ratio = (
        float(pencil_transpose_elements) / approximate_fft_work
        if approximate_fft_work > 0.0
        else float("inf")
    )
    efficiency_ceiling = 1.0 / (1.0 + ratio) if ratio >= 0.0 else 0.0
    predicted_speedup = float(num_tiles) * efficiency_ceiling

    blockers: list[str] = []
    if num_tiles < 2:
        blockers.append("single_tile_no_domain_decomposition")
    if ratio > float(max_communication_to_fft_work_ratio):
        blockers.append("pencil_transpose_communication_dominates_fft_work")
    if predicted_speedup < float(min_predicted_speedup):
        blockers.append("predicted_speedup_below_gate")

    return NonlinearSpectralPencilWorkModel(
        state_shape=(nl, nm, ny, nx, nz),
        y_chunks=normalized_y_chunks,
        x_chunks=normalized_x_chunks,
        y_offsets=_chunk_offsets(normalized_y_chunks),
        x_offsets=_chunk_offsets(normalized_x_chunks),
        num_tiles=num_tiles,
        state_elements=state_elements,
        field_elements=field_elements,
        transform_payload_elements_per_step=transform_payload,
        pencil_transpose_elements_per_step=pencil_transpose_elements,
        global_reconstruction_elements_per_step=0,
        approximate_fft_work_units_per_step=approximate_fft_work,
        communication_to_fft_work_ratio=ratio,
        parallel_efficiency_ceiling=efficiency_ceiling,
        predicted_speedup_ceiling=predicted_speedup,
        max_communication_to_fft_work_ratio=float(max_communication_to_fft_work_ratio),
        min_predicted_speedup=float(min_predicted_speedup),
        production_speedup_feasible=bool(not blockers),
        feasibility_blockers=tuple(blockers),
        claim_scope=(
            "communication/work model for a pencil-FFT nonlinear bracket route "
            "with explicit transpose stages and no global reconstruction; not a "
            "runtime speedup claim without identity and profiler artifacts"
        ),
    )


def _largest_power_of_two_not_above(value: int) -> int:
    if int(value) < 1:
        return 0
    return 1 << (int(value).bit_length() - 1)


def device_z_pencil_fft_batch_pressure_model(
    state_shape: tuple[int, int, int, int, int],
    *,
    device_count: int,
    max_fft_batch_count: int = 65_536,
    z_chunk_size: int | None = None,
) -> NonlinearSpectralDevicePencilFFTBatchModel:
    """Estimate cuFFT batch pressure for the device-z pencil micro-route.

    The profiler traces showed that large GPU cases can fail before timing when
    axis-wise FFTs create too-large batched cuFFT plans. This backend-free model
    predicts that pressure and suggests a local ``z_chunk_size`` that keeps the
    largest state-gradient axis FFT batch below ``max_fft_batch_count``. It is a
    profiling preflight, not a speedup or physics claim.
    """

    nl, nm, ny, nx, nz = _validate_spectral_state_shape(tuple(state_shape))
    count = int(device_count)
    max_batch = int(max_fft_batch_count)
    if count < 1:
        raise ValueError("device_count must be at least one")
    if max_batch < 1:
        raise ValueError("max_fft_batch_count must be at least one")
    if z_chunk_size is not None and int(z_chunk_size) < 1:
        raise ValueError("z_chunk_size must be at least one")

    blockers: list[str] = []
    if nz % count != 0:
        blockers.append("z_extent_not_divisible_by_device_count")
        local_z_extent = int(math.ceil(float(nz) / float(count)))
    else:
        local_z_extent = int(nz // count)

    max_axis_extent = int(max(ny, nx))
    batch_per_z_plane = int(nl * nm * max_axis_extent)
    unchunked_batch = int(batch_per_z_plane * local_z_extent)
    chunking_required = bool(unchunked_batch > max_batch)
    max_planes_under_cap = int(max_batch // max(batch_per_z_plane, 1))
    if chunking_required:
        suggested = _largest_power_of_two_not_above(max_planes_under_cap)
        if suggested < 1:
            suggested = 1
            blockers.append("fft_batch_pressure_exceeds_single_z_plane")
        suggested = min(int(suggested), int(local_z_extent))
    else:
        suggested = None

    requested_chunk = int(z_chunk_size) if z_chunk_size is not None else suggested
    if requested_chunk is None:
        effective_chunk = int(local_z_extent)
    else:
        effective_chunk = int(min(max(requested_chunk, 1), local_z_extent))
    chunked_batch = int(batch_per_z_plane * effective_chunk)
    chunking_active = bool(effective_chunk < local_z_extent)
    if chunking_required and chunked_batch > max_batch:
        blockers.append("z_chunk_size_still_exceeds_fft_batch_cap")
    profiling_candidate = bool(not blockers)

    return NonlinearSpectralDevicePencilFFTBatchModel(
        state_shape=(nl, nm, ny, nx, nz),
        device_count=count,
        local_z_extent=local_z_extent,
        max_fft_axis_extent=max_axis_extent,
        max_fft_batch_count=max_batch,
        unchunked_fft_batch_count=unchunked_batch,
        suggested_z_chunk_size=suggested,
        effective_z_chunk_size=effective_chunk,
        chunked_fft_batch_count=chunked_batch,
        chunking_required=chunking_required,
        chunking_active=chunking_active,
        disable_gpu_preallocation_recommended=bool(chunking_required),
        profiling_candidate=profiling_candidate,
        feasibility_blockers=tuple(blockers),
        claim_scope=(
            "cuFFT batch-pressure preflight for device-z pencil profiling; "
            "suggests z chunking before launching expensive CPU/GPU runs and "
            "does not constitute a nonlinear speedup claim"
        ),
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
    "_largest_power_of_two_not_above",
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
