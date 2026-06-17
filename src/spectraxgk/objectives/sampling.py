"""Sampling-axis and weighting helpers for solver-objective gates."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from spectraxgk.config import GridConfig
from spectraxgk.core.grid import build_spectral_grid


def _surface_index_tuple(value: int | None | tuple[int | None, ...] | list[int | None]) -> tuple[int | None, ...]:
    if value is None:
        return (None,)
    if isinstance(value, int):
        return (int(value),)
    result = tuple(None if item is None else int(item) for item in value)
    if not result:
        raise ValueError("surface_indices must contain at least one entry")
    return result


def _surface_sample_axis(
    surface_indices: int | None | tuple[int | None, ...] | list[int | None],
    torflux_values: float | tuple[float, ...] | list[float] | None,
) -> tuple[dict[str, float | int | None], ...]:
    if torflux_values is not None:
        if not (surface_indices is None or surface_indices == (None,)):
            raise TypeError("use torflux_values or surface_indices, not both")
        return tuple(
            {"surface_index": None, "torflux": float(torflux)}
            for torflux in _float_tuple(torflux_values, name="torflux_values")
        )
    return tuple({"surface_index": surface_index} for surface_index in _surface_index_tuple(surface_indices))


def _int_tuple(value: int | tuple[int, ...] | list[int], *, name: str) -> tuple[int, ...]:
    result: tuple[int, ...]
    if isinstance(value, int):
        result = (int(value),)
    else:
        result = tuple(int(item) for item in value)
    if not result:
        raise ValueError(f"{name} must contain at least one entry")
    return result


def _float_tuple(value: float | tuple[float, ...] | list[float], *, name: str) -> tuple[float, ...]:
    result: tuple[float, ...]
    if isinstance(value, (float, int)):
        result = (float(value),)
    else:
        result = tuple(float(item) for item in value)
    if not result:
        raise ValueError(f"{name} must contain at least one entry")
    if not np.all(np.isfinite(np.asarray(result, dtype=float))):
        raise ValueError(f"{name} must be finite")
    return result


def solver_grid_options_from_ky_values(
    ky_values: float | tuple[float, ...] | list[float],
    *,
    ky_base: float | None = None,
    min_ny: int = 4,
) -> dict[str, object]:
    """Return solver grid options for physical ``k_y rho_i`` scan values.

    The linear objective evaluator selects FFT row indices, while user-facing
    optimization and validation studies should be specified in physical
    ``k_y rho_i``. This helper fixes that contract explicitly: values must be
    positive integer multiples of the base spacing, ``Ly = 2*pi/ky_base``, and
    ``Ny`` is chosen large enough that all requested modes are represented as
    positive FFT rows.
    """

    values = _float_tuple(ky_values, name="ky_values")
    value_array = np.asarray(values, dtype=float)
    if np.any(value_array <= 0.0):
        raise ValueError("ky_values must be positive")
    base = float(np.min(value_array) if ky_base is None else ky_base)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("ky_base must be positive and finite")
    ratios = value_array / base
    indices = np.rint(ratios).astype(int)
    if np.any(indices < 1) or not np.allclose(ratios, indices, rtol=5.0e-10, atol=5.0e-12):
        raise ValueError("ky_values must be positive integer multiples of ky_base")
    if len(set(int(item) for item in indices)) != int(indices.size):
        raise ValueError("ky_values map to duplicate selected ky indices")
    ny = max(int(min_ny), 2 * int(np.max(indices)) + 2)
    if ny < 3:
        raise ValueError("min_ny must leave room for at least one positive ky mode")
    ly = float(2.0 * np.pi / base)
    grid = build_spectral_grid(GridConfig(Nx=1, Ny=ny, Nz=1, Lx=1.0, Ly=ly))
    selected = tuple(int(item) for item in indices)
    resolved = np.asarray(grid.ky, dtype=float)[list(selected)]
    if not np.allclose(resolved, value_array, rtol=5.0e-6, atol=5.0e-8):
        raise RuntimeError("internal ky grid construction did not reproduce requested ky_values")
    return {
        "ky_base": base,
        "ly": ly,
        "ny": int(ny),
        "selected_ky_indices": selected,
        "resolved_ky_values": tuple(float(item) for item in resolved),
    }


def _ky_sample_axis(
    selected_ky_indices: int | tuple[int, ...] | list[int],
    ky_values: float | tuple[float, ...] | list[float] | None,
    *,
    ky_base: float | None,
    objective_kwargs: dict[str, Any],
) -> tuple[tuple[dict[str, float | int], ...], dict[str, Any], dict[str, object] | None]:
    if ky_values is None:
        return (
            tuple({"selected_ky_index": index} for index in _int_tuple(selected_ky_indices, name="selected_ky_indices")),
            objective_kwargs,
            None,
        )
    if selected_ky_indices != (1,):
        raise TypeError("use ky_values or selected_ky_indices, not both")
    grid_options = solver_grid_options_from_ky_values(
        ky_values,
        ky_base=ky_base,
        min_ny=int(objective_kwargs.get("ny", 4)),
    )
    selected_grid = cast(tuple[int, ...], grid_options["selected_ky_indices"])
    resolved_grid = cast(tuple[float, ...], grid_options["resolved_ky_values"])
    selected = tuple(int(item) for item in selected_grid)
    resolved = tuple(float(item) for item in resolved_grid)
    requested = _float_tuple(ky_values, name="ky_values")
    updated_kwargs = dict(objective_kwargs)
    updated_kwargs["ny"] = int(cast(int, grid_options["ny"]))
    updated_kwargs["ly"] = float(cast(float, grid_options["ly"]))
    rows = tuple(
        {
            "selected_ky_index": index,
            "ky": requested_value,
            "selected_ky": resolved_value,
            "ky_abs_error": abs(resolved_value - requested_value),
        }
        for index, requested_value, resolved_value in zip(selected, requested, resolved, strict=True)
    )
    return rows, updated_kwargs, grid_options


def _aggregate_weights(weights: tuple[float, ...] | list[float] | np.ndarray | None, n_samples: int) -> np.ndarray:
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if weights is None:
        return np.full(n_samples, 1.0 / float(n_samples), dtype=float)
    array = np.asarray(weights, dtype=float).reshape(-1)
    if int(array.size) != int(n_samples):
        raise ValueError("weights must have one entry per aggregate sample")
    if not np.all(np.isfinite(array)):
        raise ValueError("weights must be finite")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return array / total


def _aggregate_sample_metadata(
    surface_indices: tuple[int | None, ...],
    alphas: tuple[float, ...],
    selected_ky_indices: tuple[int, ...],
    weights: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    weight_index = 0
    for surface_index in surface_indices:
        for alpha in alphas:
            for selected_ky_index in selected_ky_indices:
                rows.append(
                    {
                        "surface_index": None if surface_index is None else int(surface_index),
                        "alpha": float(alpha),
                        "selected_ky_index": int(selected_ky_index),
                        "weight": float(weights[weight_index]),
                    }
                )
                weight_index += 1
    return rows


__all__ = [
    "_aggregate_sample_metadata",
    "_aggregate_weights",
    "_float_tuple",
    "_int_tuple",
    "_ky_sample_axis",
    "_surface_index_tuple",
    "_surface_sample_axis",
    "solver_grid_options_from_ky_values",
]
