"""Communication and profiling work models for nonlinear spectral routes."""

from __future__ import annotations

import math

from spectraxgk.operators.nonlinear.parallel_contracts_spectral import (
    NonlinearSpectralDevicePencilFFTBatchModel,
    NonlinearSpectralDomainWorkModel,
    NonlinearSpectralPencilWorkModel,
)
from spectraxgk.operators.nonlinear.spectral_layout import (
    _chunk_offsets,
    _spectral_tile_bounds,
    _validate_chunks,
)
from spectraxgk.operators.nonlinear.spectral_state import _validate_spectral_state_shape


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


__all__ = [
    "_largest_power_of_two_not_above",
    "device_z_pencil_fft_batch_pressure_model",
    "nonlinear_spectral_domain_work_model",
    "nonlinear_spectral_pencil_work_model",
]
