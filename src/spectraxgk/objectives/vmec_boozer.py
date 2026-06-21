"""VMEC/Boozer objective-table plumbing for differentiable solver gates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import flux_tube_geometry_from_vmec_boozer_state
from spectraxgk.objectives.core import (
    SolverScalarObjective,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.sampling import (
    _aggregate_weights,
    _float_tuple,
    _ky_sample_axis,
    _surface_sample_axis,
)

_VMEC_BOOZER_GEOMETRY_OPTION_KEYS = {
    "surface_index",
    "torflux",
    "alpha",
    "ntheta",
    "mboz",
    "nboz",
    "jit",
    "surface_stencil_width",
    "reference_length",
    "reference_b",
    "source_model",
    "validate_finite",
}
_SOLVER_OBJECTIVE_OPTION_KEYS = {
    "selected_ky_index",
    "n_laguerre",
    "n_hermite",
    "nx",
    "ny",
    "lx",
    "ly",
    "params_linear",
    "terms",
}


def _split_vmec_boozer_objective_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split bridge options from solver-objective options and reject typos."""

    unknown = (
        set(kwargs) - _VMEC_BOOZER_GEOMETRY_OPTION_KEYS - _SOLVER_OBJECTIVE_OPTION_KEYS
    )
    if unknown:
        raise TypeError(f"unknown VMEC/Boozer objective options: {sorted(unknown)!r}")
    geometry_kwargs = {
        key: kwargs[key] for key in _VMEC_BOOZER_GEOMETRY_OPTION_KEYS if key in kwargs
    }
    objective_kwargs = {
        key: kwargs[key] for key in _SOLVER_OBJECTIVE_OPTION_KEYS if key in kwargs
    }
    return geometry_kwargs, objective_kwargs


def vmec_boozer_solver_objective_vector_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    geometry_fn: Callable[..., Any] | None = None,
    objective_vector_fn: Callable[..., jnp.ndarray] | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives from the in-memory VMEC/Boozer bridge."""

    geometry_kwargs, objective_kwargs = _split_vmec_boozer_objective_kwargs(kwargs)
    geom = (geometry_fn or flux_tube_geometry_from_vmec_boozer_state)(
        state,
        static,
        indata,
        wout,
        **geometry_kwargs,
    )
    evaluator = objective_vector_fn or solver_objective_vector_from_geometry
    return evaluator(geom, **objective_kwargs)


def vmec_boozer_solver_objective_table_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    table_with_metadata_fn: Callable[..., tuple[jnp.ndarray, list[dict[str, object]]]]
    | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives over a surface/field-line/``k_y`` table."""

    table_builder = (
        table_with_metadata_fn
        or vmec_boozer_solver_objective_table_with_metadata_from_state
    )
    table, _metadata = table_builder(
        state,
        static,
        indata,
        wout,
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        **kwargs,
    )
    return table


def _surface_geometry_kwargs(
    base_geometry_kwargs: Mapping[str, Any],
    surface: Mapping[str, object],
    *,
    alpha: float,
) -> tuple[dict[str, Any], int | None, float | None]:
    """Return geometry kwargs plus normalized surface metadata."""

    geom_kwargs = dict(base_geometry_kwargs)
    surface_index_raw = surface.get("surface_index")
    torflux_raw = surface.get("torflux")
    surface_index = None if surface_index_raw is None else int(cast(Any, surface_index_raw))
    torflux = None if torflux_raw is None else float(cast(Any, torflux_raw))
    if surface_index is not None:
        geom_kwargs["surface_index"] = surface_index
    if torflux is not None:
        geom_kwargs["torflux"] = torflux
    geom_kwargs["alpha"] = float(alpha)
    return geom_kwargs, surface_index, torflux


def _objective_row_metadata(
    *,
    surface_index: int | None,
    torflux: float | None,
    alpha: float,
    selected_ky_index: int,
    ky_sample: Mapping[str, object],
) -> dict[str, object]:
    """Return one row of table metadata using the public artifact schema."""

    row_metadata: dict[str, object] = {
        "surface_index": surface_index,
        "alpha": float(alpha),
        "selected_ky_index": int(selected_ky_index),
    }
    if torflux is not None:
        row_metadata["torflux"] = float(torflux)
        row_metadata["surface"] = float(torflux)
    for key in ("ky", "selected_ky", "ky_abs_error"):
        value = ky_sample.get(key)
        if value is not None:
            row_metadata[key] = float(cast(Any, value))
    return row_metadata


def _attach_ky_grid_options(
    metadata: list[dict[str, object]],
    ky_grid_options: Mapping[str, object] | None,
) -> None:
    """Attach physical-``k_y`` grid metadata to every table row in place."""

    if ky_grid_options is None:
        return
    options = {
        "ky_base": float(cast(float, ky_grid_options["ky_base"])),
        "ly": float(cast(float, ky_grid_options["ly"])),
        "ny": int(cast(int, ky_grid_options["ny"])),
    }
    for row in metadata:
        row["ky_grid_options"] = options


def vmec_boozer_solver_objective_table_with_metadata_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    geometry_fn: Callable[..., Any] | None = None,
    objective_vector_fn: Callable[..., jnp.ndarray] | None = None,
    **kwargs: Any,
) -> tuple[jnp.ndarray, list[dict[str, object]]]:
    """Evaluate VMEC/Boozer objective rows and return sample metadata."""

    mutable_kwargs = dict(kwargs)
    if "selected_ky_index" in mutable_kwargs:
        if selected_ky_indices != (1,) or ky_values is not None:
            raise TypeError(
                "use selected_ky_indices, not both selected_ky_index and "
                "selected_ky_indices"
            )
        selected_ky_indices = int(mutable_kwargs.pop("selected_ky_index"))
    geometry_kwargs, objective_kwargs = _split_vmec_boozer_objective_kwargs(
        mutable_kwargs
    )
    surfaces = _surface_sample_axis(surface_indices, torflux_values)
    alpha_values = _float_tuple(alphas, name="alphas")
    ky_samples, objective_kwargs, ky_grid_options = _ky_sample_axis(
        selected_ky_indices,
        ky_values,
        ky_base=ky_base,
        objective_kwargs=objective_kwargs,
    )

    geometry_builder = geometry_fn or flux_tube_geometry_from_vmec_boozer_state
    objective_evaluator = objective_vector_fn or solver_objective_vector_from_geometry
    rows: list[jnp.ndarray] = []
    metadata: list[dict[str, object]] = []
    for surface in surfaces:
        for alpha in alpha_values:
            geom_kwargs, surface_index, torflux = _surface_geometry_kwargs(
                geometry_kwargs,
                surface,
                alpha=alpha,
            )
            geom = geometry_builder(
                state,
                static,
                indata,
                wout,
                **geom_kwargs,
            )
            for ky_sample in ky_samples:
                obj_kwargs = dict(objective_kwargs)
                selected_ky_index = int(ky_sample["selected_ky_index"])
                obj_kwargs["selected_ky_index"] = selected_ky_index
                rows.append(objective_evaluator(geom, **obj_kwargs))
                metadata.append(
                    _objective_row_metadata(
                        surface_index=surface_index,
                        torflux=torflux,
                        alpha=alpha,
                        selected_ky_index=selected_ky_index,
                        ky_sample=ky_sample,
                    )
                )
    if not rows:
        raise RuntimeError("VMEC/Boozer objective table produced no samples")
    _attach_ky_grid_options(metadata, ky_grid_options)
    return jnp.stack(rows), metadata


def vmec_boozer_aggregate_scalar_objective_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    table_fn: Callable[..., jnp.ndarray] | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Reduce a VMEC/Boozer multi-point objective table to one scalar."""

    table_builder = table_fn or vmec_boozer_solver_objective_table_from_state
    table = table_builder(
        state,
        static,
        indata,
        wout,
        surface_indices=surface_indices,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        **kwargs,
    )
    values = jnp.asarray(
        [solver_scalar_objective_from_vector(row, objective) for row in table]
    )
    if str(reduction) == "mean":
        return jnp.mean(values)
    if str(reduction) == "weighted_mean":
        normalized = _aggregate_weights(weights, int(values.size))
        return jnp.sum(values * jnp.asarray(normalized, dtype=values.dtype))
    if str(reduction) == "max":
        return jnp.max(values)
    raise ValueError("reduction must be one of 'mean', 'weighted_mean', or 'max'")


def vmec_boozer_scalar_objective_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    objective: SolverScalarObjective = "growth",
    vector_fn: Callable[..., jnp.ndarray] | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate one scalar optimization objective on the VMEC/Boozer path."""

    vector_builder = vector_fn or vmec_boozer_solver_objective_vector_from_state
    vector = vector_builder(
        state,
        static,
        indata,
        wout,
        **kwargs,
    )
    return solver_scalar_objective_from_vector(vector, objective)


__all__ = [
    "_split_vmec_boozer_objective_kwargs",
    "vmec_boozer_aggregate_scalar_objective_from_state",
    "vmec_boozer_scalar_objective_from_state",
    "vmec_boozer_solver_objective_table_from_state",
    "vmec_boozer_solver_objective_table_with_metadata_from_state",
    "vmec_boozer_solver_objective_vector_from_state",
]
