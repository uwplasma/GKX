"""Sample-table and scalar-reduction kernels for VMEC transport objectives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.vmec_boozer_core import (
    flux_tube_geometry_from_vmec_boozer_state,
)
from spectraxgk.objectives.core import (
    SOLVER_OBJECTIVE_NAMES,
    solver_growth_rate_from_geometry,
)
from spectraxgk.objectives.portfolio import aggregate_objective_portfolio
from spectraxgk.objectives.stellarator import StellaratorITGSampleSet, smooth_positive
from spectraxgk.objectives.vmec_transport_config import (
    VMECJAXTransportObjectiveConfig,
    _pin_current_optional_backend_paths,
)

_SOLVER_OBJECTIVE_INDEX = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}


def _solver_table_to_nonlinear_window_proxy(
    table: jnp.ndarray,
    config: VMECJAXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Map linear solver rows to a smooth reduced nonlinear heat-flux proxy."""

    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    gamma = jnp.asarray(table[..., idx["gamma"]])
    kperp_eff2 = jnp.asarray(table[..., idx["kperp_eff2"]])
    heat_weight = jnp.asarray(table[..., idx["linear_heat_flux_weight"]])
    gamma_plus = smooth_positive(gamma, beta=18.0)
    saturation = 1.0 + 2.2 * jnp.maximum(kperp_eff2, 0.0) + 0.15 * gamma_plus
    mean_energy = (
        2.0
        * gamma_plus
        / jnp.maximum(
            saturation,
            jnp.asarray(config.nonlinear_saturation_floor, dtype=gamma_plus.dtype),
        )
    )
    return float(config.nonlinear_csat) * jnp.maximum(heat_weight, 0.0) * mean_energy


def _normalized_axis_weights(values: Sequence[float] | None, size: int) -> np.ndarray:
    """Return normalized static weights for one separable sample axis."""

    if int(size) <= 0:
        raise ValueError("axis size must be positive")
    if values is None:
        return np.full((int(size),), 1.0 / float(size), dtype=float)
    arr = np.asarray(tuple(float(value) for value in values), dtype=float)
    if arr.shape != (int(size),) or not np.all(np.isfinite(arr)):
        raise ValueError(f"weights must be a finite length-{int(size)} vector")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return arr / total


def _surface_chunk_sample_sets(
    sample_set: StellaratorITGSampleSet,
    *,
    chunk_size: int,
) -> tuple[tuple[StellaratorITGSampleSet, float], ...]:
    """Split a sample set by surface while preserving weighted-mean algebra."""

    surfaces = tuple(float(value) for value in sample_set.surfaces)
    if int(chunk_size) <= 0 or int(chunk_size) >= len(surfaces):
        return ((sample_set, 1.0),)
    if sample_set.reduction not in ("weighted_mean", "mean"):
        raise ValueError(
            "surface chunking currently supports only mean or weighted_mean reductions"
        )
    surface_weights = _normalized_axis_weights(
        sample_set.surface_weights, len(surfaces)
    )
    chunks: list[tuple[StellaratorITGSampleSet, float]] = []
    for start in range(0, len(surfaces), int(chunk_size)):
        indices = tuple(range(start, min(start + int(chunk_size), len(surfaces))))
        chunk_surfaces = tuple(surfaces[index] for index in indices)
        if sample_set.reduction == "mean":
            chunk_weight = float(len(indices) / len(surfaces))
            chunk_surface_weights = None
        else:
            chunk_weight = float(np.sum(surface_weights[list(indices)]))
            chunk_surface_weights = (
                None
                if sample_set.surface_weights is None
                else tuple(
                    float(sample_set.surface_weights[index]) for index in indices
                )
            )
        chunks.append(
            (
                StellaratorITGSampleSet(
                    surfaces=chunk_surfaces,
                    alphas=sample_set.alphas,
                    ky_values=sample_set.ky_values,
                    surface_weights=chunk_surface_weights,
                    alpha_weights=sample_set.alpha_weights,
                    ky_weights=sample_set.ky_weights,
                    reduction=sample_set.reduction,
                ),
                chunk_weight,
            )
        )
    return tuple(chunks)


def _static_grid_options_from_ky_values(
    ky_values: Sequence[float],
    *,
    min_ny: int,
) -> dict[str, Any]:
    """Resolve physical ``ky`` samples without creating traced JAX arrays."""

    values = np.asarray(tuple(float(value) for value in ky_values), dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("ky_values must be a finite non-empty vector")
    if np.any(values <= 0.0):
        raise ValueError("ky_values must be positive")
    base = float(np.min(values))
    ratios = values / base
    indices = np.rint(ratios).astype(int)
    if np.any(indices < 1) or not np.allclose(
        ratios, indices, rtol=5.0e-10, atol=5.0e-12
    ):
        raise ValueError(
            "ky_values must be positive integer multiples of their minimum value"
        )
    if len(set(int(item) for item in indices)) != int(indices.size):
        raise ValueError("ky_values map to duplicate selected ky indices")
    ny = max(int(min_ny), 2 * int(np.max(indices)) + 2)
    return {
        "ky_base": base,
        "ly": float(2.0 * np.pi / base),
        "ny": int(ny),
        "selected_ky_indices": tuple(int(item) for item in indices),
    }


def _geometry_transport_weights(
    geom: Any,
    *,
    selected_ky_index: int,
    ly: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return trace-safe geometry weights for VMEC-JAX transport residuals."""

    theta = jnp.asarray(getattr(geom, "theta"))
    dtype = theta.dtype
    bmag = jnp.asarray(getattr(geom, "bmag_profile", jnp.ones_like(theta)), dtype=dtype)
    jac = jnp.abs(
        jnp.asarray(
            getattr(geom, "jacobian_profile", jnp.ones_like(theta)), dtype=dtype
        )
    )
    norm = jnp.maximum(jnp.sum(jac), jnp.asarray(1.0e-30, dtype=dtype))
    weights = jac / norm
    gds2 = jnp.asarray(getattr(geom, "gds2_profile", jnp.ones_like(theta)), dtype=dtype)
    gds21 = jnp.asarray(
        getattr(geom, "gds21_profile", jnp.zeros_like(theta)), dtype=dtype
    )
    gds22 = jnp.asarray(
        getattr(geom, "gds22_profile", jnp.ones_like(theta)), dtype=dtype
    )
    cv = jnp.asarray(getattr(geom, "cv_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb = jnp.asarray(getattr(geom, "gb_profile", jnp.zeros_like(theta)), dtype=dtype)
    cv0 = jnp.asarray(getattr(geom, "cv0_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb0 = jnp.asarray(getattr(geom, "gb0_profile", jnp.zeros_like(theta)), dtype=dtype)
    mean_b = jnp.sum(weights * bmag)
    ripple = jnp.sqrt(
        jnp.sum(
            weights
            * (
                bmag / jnp.maximum(jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=dtype))
                - 1.0
            )
            ** 2
        )
    )
    metric = jnp.sqrt(jnp.sum(weights * (gds2**2 + 2.0 * gds21**2 + gds22**2)))
    drift = jnp.sqrt(jnp.sum(weights * (cv**2 + gb**2 + cv0**2 + gb0**2)))
    ky = jnp.asarray(2.0 * np.pi * float(selected_ky_index) / float(ly), dtype=dtype)
    kperp_eff2 = jnp.maximum(
        ky**2 * jnp.sum(weights * jnp.maximum(gds2, jnp.asarray(1.0e-8, dtype=dtype))),
        jnp.asarray(1.0e-8, dtype=dtype),
    )
    heat_weight = 0.08 + 0.35 * ripple + 0.08 * metric + 0.22 * drift
    particle_weight = 0.25 * heat_weight
    return kperp_eff2, heat_weight, particle_weight


def _transport_sample_geometry(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig,
    *,
    torflux: float,
    alpha: float,
) -> Any:
    return flux_tube_geometry_from_vmec_boozer_state(
        state,
        static,
        indata,
        wout_reference,
        torflux=float(torflux),
        alpha=float(alpha),
        ntheta=int(config.ntheta),
        mboz=int(config.mboz),
        nboz=int(config.nboz),
        reference_length=config.reference_length,
        reference_b=config.reference_b,
        validate_finite=bool(config.validate_finite),
    )


def _solver_objective_row(gamma: jnp.ndarray) -> jnp.ndarray:
    row = jnp.zeros(
        (len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype
    )
    return row.at[_SOLVER_OBJECTIVE_INDEX["gamma"]].set(gamma)


def _transport_weighted_solver_row(
    geom: Any,
    gamma: jnp.ndarray,
    *,
    selected_ky_index: int,
    ly: float,
) -> jnp.ndarray:
    kperp, heat_weight, particle_weight = _geometry_transport_weights(
        geom,
        selected_ky_index=int(selected_ky_index),
        ly=float(ly),
    )
    ql_proxy = (
        gamma
        * heat_weight
        / jnp.maximum(kperp, jnp.asarray(1.0e-12, dtype=jnp.asarray(gamma).dtype))
    )
    row = _solver_objective_row(gamma)
    row = row.at[_SOLVER_OBJECTIVE_INDEX["kperp_eff2"]].set(kperp)
    row = row.at[_SOLVER_OBJECTIVE_INDEX["linear_heat_flux_weight"]].set(heat_weight)
    row = row.at[_SOLVER_OBJECTIVE_INDEX["linear_particle_flux_weight"]].set(
        particle_weight
    )
    return row.at[_SOLVER_OBJECTIVE_INDEX["mixing_length_heat_flux_proxy"]].set(
        ql_proxy
    )


def _transport_feature_table_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig,
    grid_options: dict[str, Any],
) -> jnp.ndarray:
    """Return solver-objective rows with trace-safe growth-rate derivatives."""

    samples = config.sample_set
    selected_indices = cast(tuple[int, ...], grid_options["selected_ky_indices"])
    rows: list[jnp.ndarray] = []
    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            geom = _transport_sample_geometry(
                state,
                static,
                indata,
                wout_reference,
                torflux=float(torflux),
                alpha=float(alpha),
                config=config,
            )
            for selected_ky_index in selected_indices:
                gamma = solver_growth_rate_from_geometry(
                    geom,
                    selected_ky_index=int(selected_ky_index),
                    n_laguerre=int(config.n_laguerre),
                    n_hermite=int(config.n_hermite),
                    nx=int(config.nx),
                    ny=int(grid_options["ny"]),
                    ly=float(grid_options["ly"]),
                )
                if config.kind == "growth":
                    rows.append(_solver_objective_row(gamma))
                    continue
                rows.append(
                    _transport_weighted_solver_row(
                        geom,
                        gamma,
                        selected_ky_index=int(selected_ky_index),
                        ly=float(grid_options["ly"]),
                    )
                )
    if not rows:
        raise RuntimeError("VMEC-JAX transport objective produced no sample rows")
    return jnp.reshape(
        jnp.stack(rows),
        (
            len(samples.surfaces),
            len(samples.alphas),
            len(samples.ky_values),
            len(SOLVER_OBJECTIVE_NAMES),
        ),
    )


def _apply_objective_transform(
    value: jnp.ndarray,
    config: VMECJAXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Return a dimensionless transport residual with optional safe scaling."""

    raw = jnp.asarray(value)
    if config.objective_transform == "raw":
        return raw
    scale = jnp.asarray(float(config.objective_scale), dtype=raw.dtype)
    scaled = raw / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=raw.dtype))
    if config.objective_transform == "scaled":
        return scaled
    return jnp.sign(scaled) * jnp.log1p(jnp.abs(scaled))


def _transport_objective_raw_value_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    cfg: VMECJAXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Evaluate the untransformed scalar transport objective."""

    samples = cfg.sample_set
    solver_options = cfg.objective_options()
    grid_options = _static_grid_options_from_ky_values(
        samples.ky_values,
        min_ny=int(solver_options.get("ny", cfg.ny)),
    )
    solver_options["ny"] = int(grid_options["ny"])
    solver_options["ly"] = float(grid_options["ly"])
    table = _transport_feature_table_from_state(
        state,
        static,
        indata,
        wout_reference,
        cfg,
        grid_options,
    )
    if cfg.kind == "nonlinear_window_heat_flux":
        objective_table = _solver_table_to_nonlinear_window_proxy(table, cfg)[..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    elif cfg.kind == "growth":
        objective_table = table[..., SOLVER_OBJECTIVE_NAMES.index("gamma")][..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    else:
        objective_table = table[
            ..., SOLVER_OBJECTIVE_NAMES.index("mixing_length_heat_flux_proxy")
        ][..., None]
        weights = (1.0,) if cfg.objective_weights is None else cfg.objective_weights
    value = aggregate_objective_portfolio(
        objective_table,
        surface_weights=samples.surface_weights,
        alpha_weights=samples.alpha_weights,
        ky_weights=samples.ky_weights,
        objective_weights=weights,
        reduction=samples.reduction,
    )
    return jnp.asarray(value)


def _chunked_transport_objective_raw_value_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    cfg: VMECJAXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Evaluate a weighted-mean raw objective one surface chunk at a time."""

    raw_value = None
    for chunk_sample_set, chunk_weight in _surface_chunk_sample_sets(
        cfg.sample_set,
        chunk_size=int(cfg.surface_chunk_size),
    ):
        chunk_cfg = replace(
            cfg,
            sample_set=chunk_sample_set,
            surface_chunk_size=0,
            objective_transform="raw",
            objective_scale=1.0,
        )
        chunk_value = _transport_objective_raw_value_from_state(
            state,
            static,
            indata,
            wout_reference,
            chunk_cfg,
        )
        weighted = (
            jnp.asarray(float(chunk_weight), dtype=jnp.asarray(chunk_value).dtype)
            * chunk_value
        )
        raw_value = weighted if raw_value is None else raw_value + weighted
    if raw_value is None:
        raise RuntimeError("surface chunking produced no objective chunks")
    return jnp.asarray(raw_value)


def vmec_jax_transport_objective_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig | None = None,
) -> jnp.ndarray:
    """Evaluate a scalar SPECTRAX-GK transport objective from a VMEC-JAX state."""

    _pin_current_optional_backend_paths()
    cfg = config or VMECJAXTransportObjectiveConfig()
    if int(cfg.surface_chunk_size) > 0:
        value = _chunked_transport_objective_raw_value_from_state(
            state,
            static,
            indata,
            wout_reference,
            cfg,
        )
    else:
        value = _transport_objective_raw_value_from_state(
            state,
            static,
            indata,
            wout_reference,
            cfg,
        )
    return _apply_objective_transform(value, cfg)


__all__ = [
    "_apply_objective_transform",
    "_chunked_transport_objective_raw_value_from_state",
    "_geometry_transport_weights",
    "_solver_table_to_nonlinear_window_proxy",
    "_static_grid_options_from_ky_values",
    "_surface_chunk_sample_sets",
    "_transport_feature_table_from_state",
    "_transport_objective_raw_value_from_state",
    "vmec_jax_transport_objective_from_state",
]
