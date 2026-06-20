"""Logical spectral layout and tile utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _validate_chunks(
    axis_size: int, chunks: tuple[int, ...], *, name: str
) -> tuple[int, ...]:
    if not chunks:
        raise ValueError(f"{name} must contain at least one chunk")
    normalized = tuple(int(item) for item in chunks)
    if any(item <= 0 for item in normalized):
        raise ValueError(f"{name} entries must be positive")
    if sum(normalized) != int(axis_size):
        raise ValueError(f"{name} must sum to the decomposed axis size")
    return normalized


def _chunk_offsets(chunks: tuple[int, ...]) -> tuple[int, ...]:
    offsets: list[int] = []
    start = 0
    for chunk in chunks:
        offsets.append(start)
        start += int(chunk)
    return tuple(offsets)


def _split_reassemble(
    arr: jax.Array, *, axis: int, chunks: tuple[int, ...]
) -> jax.Array:
    canonical_axis = axis % arr.ndim
    normalized_chunks = _validate_chunks(
        int(arr.shape[canonical_axis]),
        chunks,
        name="chunks",
    )
    split_points = []
    offset = 0
    for chunk in normalized_chunks[:-1]:
        offset += chunk
        split_points.append(offset)
    return jnp.concatenate(
        jnp.split(arr, split_points, axis=canonical_axis), axis=canonical_axis
    )


def _spectral_layout_round_trip(
    arr: jax.Array,
    *,
    y_axis: int,
    x_axis: int,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> jax.Array:
    """Simulate the split/transposed/reassembled layout changes used by FFTs."""

    y_axis = y_axis % arr.ndim
    x_axis = x_axis % arr.ndim
    communicated = _split_reassemble(arr, axis=y_axis, chunks=y_chunks)
    transposed = jnp.swapaxes(communicated, y_axis, x_axis)
    reassembled = _split_reassemble(transposed, axis=x_axis, chunks=y_chunks)
    reassembled = _split_reassemble(reassembled, axis=y_axis, chunks=x_chunks)
    return jnp.swapaxes(reassembled, y_axis, x_axis)


def _spectral_tile_bounds(
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    y_offsets = _chunk_offsets(y_chunks)
    x_offsets = _chunk_offsets(x_chunks)
    return tuple(
        (y_start, y_start + y_size, x_start, x_start + x_size)
        for y_start, y_size in zip(y_offsets, y_chunks, strict=True)
        for x_start, x_size in zip(x_offsets, x_chunks, strict=True)
    )


def _normalize_spectral_tile_bounds(
    tile_bounds: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return validated fixed-width tile bounds for mypy and runtime checks."""

    normalized: list[tuple[int, int, int, int]] = []
    for item in tile_bounds:
        if len(item) != 4:
            raise ValueError("each spectral tile bound must contain four integers")
        y_start, y_stop, x_start, x_stop = (int(value) for value in item)
        normalized.append((y_start, y_stop, x_start, x_stop))
    return tuple(normalized)


def _logical_spectral_tiles(
    arr: jax.Array,
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    y_axis: int,
    x_axis: int,
) -> tuple[jax.Array, ...]:
    canonical_y_axis = y_axis % arr.ndim
    canonical_x_axis = x_axis % arr.ndim
    normalized_y_chunks = _validate_chunks(
        int(arr.shape[canonical_y_axis]),
        y_chunks,
        name="y_chunks",
    )
    normalized_x_chunks = _validate_chunks(
        int(arr.shape[canonical_x_axis]),
        x_chunks,
        name="x_chunks",
    )

    tiles: list[jax.Array] = []
    for y_start, y_stop, x_start, x_stop in _spectral_tile_bounds(
        normalized_y_chunks,
        normalized_x_chunks,
    ):
        y_tile = jax.lax.dynamic_slice_in_dim(
            arr,
            y_start,
            y_stop - y_start,
            axis=canonical_y_axis,
        )
        tiles.append(
            jax.lax.dynamic_slice_in_dim(
                y_tile,
                x_start,
                x_stop - x_start,
                axis=canonical_x_axis,
            )
        )
    return tuple(tiles)


def _reconstruct_logical_spectral_tiles(
    tiles: tuple[jax.Array, ...],
    *,
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    y_axis: int,
    x_axis: int,
) -> jax.Array:
    if len(tiles) != len(y_chunks) * len(x_chunks):
        raise ValueError("tile count must match y_chunks by x_chunks")
    if not tiles:
        raise ValueError("at least one tile is required")

    canonical_y_axis = y_axis % tiles[0].ndim
    canonical_x_axis = x_axis % tiles[0].ndim
    rows = []
    tile_index = 0
    for _y_chunk in y_chunks:
        row_tiles = []
        for _x_chunk in x_chunks:
            row_tiles.append(tiles[tile_index])
            tile_index += 1
        rows.append(jnp.concatenate(row_tiles, axis=canonical_x_axis))
    return jnp.concatenate(rows, axis=canonical_y_axis)


__all__ = [
    "_chunk_offsets",
    "_logical_spectral_tiles",
    "_normalize_spectral_tile_bounds",
    "_reconstruct_logical_spectral_tiles",
    "_spectral_layout_round_trip",
    "_spectral_tile_bounds",
    "_split_reassemble",
    "_validate_chunks",
]
