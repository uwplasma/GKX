"""VMEC-JAX QA optimization hooks for SPECTRAX-GK transport objectives.

The classes in this module are optional glue for examples that start from a
``vmec_jax`` fixed-boundary optimization and append a SPECTRAX-GK transport
term. They are intentionally small: VMEC-JAX remains responsible for aspect,
iota, and quasisymmetry constraints, while SPECTRAX-GK supplies the local ITG
linear/quasilinear or reduced nonlinear-window transport residual.

The gradient contract is deliberately conservative. The ``growth`` objective
uses eigenvalue-only differentiation. The quasilinear and reduced nonlinear
window objectives used by the VMEC-JAX optimizer differentiate the same solver
growth rate and combine it with differentiable geometry-level transport
weights. The eigenfunction-resolved quasilinear weights remain an audit path,
not the traced VMEC-JAX optimizer residual, because nonsymmetric eigenvector AD
is not supported by JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Sequence, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry.differentiable import (
    flux_tube_geometry_from_vmec_boozer_state,
    prewarm_vmec_boozer_equal_arc_cache,
)
from spectraxgk.solver_eigen_objectives import (
    dominant_eigenvalue_branch_locality_report,
)
from spectraxgk.solver_objective_core import (
    SOLVER_OBJECTIVE_NAMES,
    solver_growth_rate_from_geometry,
    solver_linear_operator_matrix_from_geometry,
)
from spectraxgk.stellarator_objective_portfolio import aggregate_objective_portfolio
from spectraxgk.stellarator_optimization import StellaratorITGSampleSet, smooth_positive


VMECJAXTransportObjectiveKind = Literal[
    "growth",
    "quasilinear_flux",
    "nonlinear_window_heat_flux",
]
VMECJAXTransportObjectiveTransform = Literal["raw", "scaled", "log1p"]


def _module_search_root(module_name: str) -> Path | None:
    """Return the import root for an already importable optional backend."""

    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    raw_file = getattr(module, "__file__", None)
    if raw_file is not None:
        return Path(str(raw_file)).resolve(strict=False).parent.parent
    raw_paths = getattr(module, "__path__", None)
    if raw_paths is None:
        return None
    for raw in raw_paths:
        path = Path(str(raw)).resolve(strict=False)
        if path.exists():
            return path
    return None


def _pin_current_optional_backend_paths() -> None:
    """Keep geometry discovery on the same optional backends VMEC-JAX imported.

    The differentiable-geometry bridge intentionally prefers explicit local
    checkouts over globally installed packages.  When examples run from a fresh
    temporary clone while another VMEC-JAX checkout exists in ``$HOME``, that
    preference can otherwise evict the VMEC-JAX module that owns the traced
    optimization state.  Pinning the currently importable backend paths makes
    the VMEC-JAX/SPECTRAX-GK objective reproducible without requiring users to
    hand-set environment variables.
    """

    if not (os.environ.get("SPECTRAX_VMEC_JAX_PATH") or os.environ.get("VMEC_JAX_PATH")):
        root = _module_search_root("vmec_jax")
        if root is not None:
            os.environ.setdefault("SPECTRAX_VMEC_JAX_PATH", str(root))
    if not (
        os.environ.get("SPECTRAX_BOOZ_XFORM_JAX_PATH")
        or os.environ.get("BOOZ_XFORM_JAX_PATH")
    ):
        root = _module_search_root("booz_xform_jax")
        if root is not None:
            os.environ.setdefault("SPECTRAX_BOOZ_XFORM_JAX_PATH", str(root))


@dataclass(frozen=True)
class VMECJAXTransportObjectiveConfig:
    """Configuration for VMEC-JAX to SPECTRAX-GK objective evaluation."""

    kind: VMECJAXTransportObjectiveKind = "nonlinear_window_heat_flux"
    sample_set: StellaratorITGSampleSet = field(default_factory=StellaratorITGSampleSet)
    objective_weights: tuple[float, ...] | None = None
    ntheta: int = 24
    mboz: int = 21
    nboz: int = 21
    n_laguerre: int = 2
    n_hermite: int = 3
    nx: int = 1
    ny: int = 4
    nonlinear_csat: float = 0.85
    nonlinear_saturation_floor: float = 1.0e-10
    reference_length: float | None = None
    reference_b: float | None = None
    objective_transform: VMECJAXTransportObjectiveTransform = "raw"
    objective_scale: float = 1.0
    surface_chunk_size: int = 0
    validate_finite: bool = True

    @property
    def gradient_scope(self) -> str:
        """Return the differentiated part of this objective."""

        if self.kind == "growth":
            return "eigenvalue_growth_ad"
        return "eigenvalue_growth_ad_with_geometry_transport_weights"

    def __post_init__(self) -> None:
        if self.kind not in ("growth", "quasilinear_flux", "nonlinear_window_heat_flux"):
            raise ValueError(f"unknown VMEC-JAX transport objective kind {self.kind!r}")
        if int(self.ntheta) < 4:
            raise ValueError("ntheta must be >= 4")
        if int(self.mboz) < 21 or int(self.nboz) < 21:
            raise ValueError("mboz and nboz must be at least 21 for paper-facing QA optimization")
        if int(self.n_laguerre) < 1 or int(self.n_hermite) < 1:
            raise ValueError("n_laguerre and n_hermite must be positive")
        if int(self.nx) < 1 or int(self.ny) < 3:
            raise ValueError("nx must be positive and ny must be at least 3")
        if float(self.nonlinear_csat) <= 0.0:
            raise ValueError("nonlinear_csat must be positive")
        if self.objective_transform not in ("raw", "scaled", "log1p"):
            raise ValueError(f"unknown VMEC-JAX transport objective transform {self.objective_transform!r}")
        if float(self.objective_scale) <= 0.0:
            raise ValueError("objective_scale must be positive")
        if int(self.surface_chunk_size) < 0:
            raise ValueError("surface_chunk_size must be non-negative")
        if int(self.surface_chunk_size) > 0 and self.sample_set.reduction not in ("weighted_mean", "mean"):
            raise ValueError("surface_chunk_size currently supports only mean or weighted_mean reductions")

    def objective_options(self) -> dict[str, Any]:
        """Return SPECTRAX-GK solver options for this objective."""

        options: dict[str, Any] = {
            "ntheta": int(self.ntheta),
            "mboz": int(self.mboz),
            "nboz": int(self.nboz),
            "n_laguerre": int(self.n_laguerre),
            "n_hermite": int(self.n_hermite),
            "nx": int(self.nx),
            "ny": int(self.ny),
            "reference_length": self.reference_length,
            "reference_b": self.reference_b,
            "validate_finite": bool(self.validate_finite),
        }
        return {key: value for key, value in options.items() if value is not None}


def _reference_wout_from_context(ctx: Any) -> Any:
    """Return the minimal WOUT metadata needed by the VMEC/Boozer bridge."""

    cfg = getattr(getattr(ctx, "static", None), "cfg", None)
    nfp = int(getattr(cfg, "nfp", 1))
    return SimpleNamespace(
        signgs=int(getattr(ctx, "signgs", 1)),
        nfp=nfp,
        Aminor_p=1.0,
        phi=np.asarray([0.0, -np.pi], dtype=float),
    )


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
    mean_energy = 2.0 * gamma_plus / jnp.maximum(
        saturation,
        jnp.asarray(config.nonlinear_saturation_floor, dtype=gamma_plus.dtype),
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
        raise ValueError("surface chunking currently supports only mean or weighted_mean reductions")
    surface_weights = _normalized_axis_weights(sample_set.surface_weights, len(surfaces))
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
                else tuple(float(sample_set.surface_weights[index]) for index in indices)
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
    if np.any(indices < 1) or not np.allclose(ratios, indices, rtol=5.0e-10, atol=5.0e-12):
        raise ValueError("ky_values must be positive integer multiples of their minimum value")
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
    jac = jnp.abs(jnp.asarray(getattr(geom, "jacobian_profile", jnp.ones_like(theta)), dtype=dtype))
    norm = jnp.maximum(jnp.sum(jac), jnp.asarray(1.0e-30, dtype=dtype))
    weights = jac / norm
    gds2 = jnp.asarray(getattr(geom, "gds2_profile", jnp.ones_like(theta)), dtype=dtype)
    gds21 = jnp.asarray(getattr(geom, "gds21_profile", jnp.zeros_like(theta)), dtype=dtype)
    gds22 = jnp.asarray(getattr(geom, "gds22_profile", jnp.ones_like(theta)), dtype=dtype)
    cv = jnp.asarray(getattr(geom, "cv_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb = jnp.asarray(getattr(geom, "gb_profile", jnp.zeros_like(theta)), dtype=dtype)
    cv0 = jnp.asarray(getattr(geom, "cv0_profile", jnp.zeros_like(theta)), dtype=dtype)
    gb0 = jnp.asarray(getattr(geom, "gb0_profile", jnp.zeros_like(theta)), dtype=dtype)
    mean_b = jnp.sum(weights * bmag)
    ripple = jnp.sqrt(
        jnp.sum(weights * (bmag / jnp.maximum(jnp.abs(mean_b), jnp.asarray(1.0e-30, dtype=dtype)) - 1.0) ** 2)
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
    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            geom = flux_tube_geometry_from_vmec_boozer_state(
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
                    row = jnp.zeros((len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype)
                    row = row.at[idx["gamma"]].set(gamma)
                    rows.append(row)
                    continue
                kperp, heat_weight, particle_weight = _geometry_transport_weights(
                    geom,
                    selected_ky_index=int(selected_ky_index),
                    ly=float(grid_options["ly"]),
                )
                ql_proxy = gamma * heat_weight / jnp.maximum(
                    kperp,
                    jnp.asarray(1.0e-12, dtype=jnp.asarray(gamma).dtype),
                )
                row = jnp.zeros((len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype)
                row = row.at[idx["gamma"]].set(gamma)
                row = row.at[idx["kperp_eff2"]].set(kperp)
                row = row.at[idx["linear_heat_flux_weight"]].set(heat_weight)
                row = row.at[idx["linear_particle_flux_weight"]].set(particle_weight)
                row = row.at[idx["mixing_length_heat_flux_proxy"]].set(ql_proxy)
                rows.append(row)
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
        objective_table = table[..., SOLVER_OBJECTIVE_NAMES.index("mixing_length_heat_flux_proxy")][..., None]
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
        weighted = jnp.asarray(float(chunk_weight), dtype=jnp.asarray(chunk_value).dtype) * chunk_value
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


def vmec_jax_transport_growth_branch_locality_report_from_states(
    base_state: Any,
    plus_state: Any,
    minus_state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMECJAXTransportObjectiveConfig | None = None,
    *,
    step: float,
    gap_floor: float = 1.0e-8,
    slope_rtol: float = 1.0e-2,
    slope_atol: float = 1.0e-8,
    max_samples: int = 0,
) -> dict[str, object]:
    """Check dominant-growth eigenbranch locality for VMEC/Boozer samples.

    The optimizer-facing transport residual can only use the implicit
    dominant-eigenvalue gradient when the same eigenbranch is locally selected.
    This report evaluates the exact SPECTRAX-GK linear operator matrix at the
    base, plus, and minus VMEC final states for each configured
    surface/alpha/``k_y`` sample, then delegates branch classification to
    :func:`dominant_eigenvalue_branch_locality_report`.
    """

    _pin_current_optional_backend_paths()
    cfg = config or VMECJAXTransportObjectiveConfig(kind="growth")
    step_f = float(step)
    if step_f <= 0.0:
        raise ValueError("step must be positive")
    max_samples_int = int(max_samples)
    if max_samples_int < 0:
        raise ValueError("max_samples must be non-negative")

    samples = cfg.sample_set
    grid_options = _static_grid_options_from_ky_values(
        samples.ky_values,
        min_ny=int(cfg.ny),
    )
    selected_indices = cast(tuple[int, ...], grid_options["selected_ky_indices"])
    rows: list[dict[str, object]] = []
    total_sample_count = int(len(samples.surfaces) * len(samples.alphas) * len(samples.ky_values))

    def geom_for(state: Any, *, torflux: float, alpha: float) -> Any:
        return flux_tube_geometry_from_vmec_boozer_state(
            state,
            static,
            indata,
            wout_reference,
            torflux=float(torflux),
            alpha=float(alpha),
            ntheta=int(cfg.ntheta),
            mboz=int(cfg.mboz),
            nboz=int(cfg.nboz),
            reference_length=cfg.reference_length,
            reference_b=cfg.reference_b,
            validate_finite=bool(cfg.validate_finite),
        )

    for torflux in samples.surfaces:
        for alpha in samples.alphas:
            for ky_value, selected_ky_index in zip(samples.ky_values, selected_indices, strict=True):
                metadata = {
                    "surface": float(torflux),
                    "alpha": float(alpha),
                    "ky": float(ky_value),
                    "selected_ky_index": int(selected_ky_index),
                }
                try:
                    base_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(base_state, torflux=float(torflux), alpha=float(alpha)),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    plus_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(plus_state, torflux=float(torflux), alpha=float(alpha)),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    minus_matrix = solver_linear_operator_matrix_from_geometry(
                        geom_for(minus_state, torflux=float(torflux), alpha=float(alpha)),
                        selected_ky_index=int(selected_ky_index),
                        n_laguerre=int(cfg.n_laguerre),
                        n_hermite=int(cfg.n_hermite),
                        nx=int(cfg.nx),
                        ny=int(grid_options["ny"]),
                        ly=float(grid_options["ly"]),
                    )
                    branch = dominant_eigenvalue_branch_locality_report(
                        base_matrix,
                        plus_matrix,
                        minus_matrix,
                        step=step_f,
                        gap_floor=float(gap_floor),
                        slope_rtol=float(slope_rtol),
                        slope_atol=float(slope_atol),
                    )
                    rows.append(
                        {
                            **metadata,
                            "passed": bool(branch["passed"]),
                            "classification": str(branch["classification"]),
                            "branch_locality": branch,
                        }
                    )
                except Exception as exc:  # pragma: no cover - exercised by optional backends.
                    rows.append(
                        {
                            **metadata,
                            "passed": False,
                            "classification": "branch_locality_evaluation_error",
                            "error": str(exc),
                        }
                    )
                if max_samples_int > 0 and len(rows) >= max_samples_int:
                    break
            if max_samples_int > 0 and len(rows) >= max_samples_int:
                break
        if max_samples_int > 0 and len(rows) >= max_samples_int:
            break

    finite = bool(rows and all("error" not in row for row in rows))
    passed = bool(finite and len(rows) == total_sample_count and all(bool(row["passed"]) for row in rows))
    blockers: list[str] = []
    if not rows:
        blockers.append("no_branch_locality_samples")
    if len(rows) < total_sample_count:
        blockers.append("branch_locality_sample_set_truncated")
    if any(str(row.get("classification")) == "branch_locality_evaluation_error" for row in rows):
        blockers.append("branch_locality_evaluation_error")
    if any(not bool(row.get("passed", False)) for row in rows):
        blockers.append("branch_locality_mismatch_or_underisolated")
    classifications = sorted({str(row.get("classification")) for row in rows})
    return {
        "kind": "vmec_jax_transport_growth_branch_locality_report",
        "claim_scope": (
            "VMEC/Boozer final-state perturbation -> SPECTRAX-GK linear operator "
            "dominant-growth branch locality; not a full transport-gradient promotion by itself"
        ),
        "passed": passed,
        "finite": finite,
        "classification": (
            "all_samples_dominant_growth_branch_locally_consistent"
            if passed
            else "growth_branch_locality_failed_or_incomplete"
        ),
        "step": step_f,
        "gap_floor": float(gap_floor),
        "slope_rtol": float(slope_rtol),
        "slope_atol": float(slope_atol),
        "sample_count": total_sample_count,
        "evaluated_sample_count": len(rows),
        "truncated": bool(len(rows) < total_sample_count),
        "classifications": classifications,
        "blockers": sorted(set(blockers)),
        "sample_set": samples.to_dict(),
        "spectrax_config": {
            "ntheta": int(cfg.ntheta),
            "mboz": int(cfg.mboz),
            "nboz": int(cfg.nboz),
            "n_laguerre": int(cfg.n_laguerre),
            "n_hermite": int(cfg.n_hermite),
            "nx": int(cfg.nx),
            "ny": int(grid_options["ny"]),
            "ly": float(grid_options["ly"]),
        },
        "rows": rows,
        "next_action": (
            "growth-branch locality is admissible for these samples"
            if passed
            else (
                "keep VMEC/SPECTRAX transport-gradient optimization fail-closed; "
                "reduce finite-difference steps, regularize branch selection, or "
                "use explicit branch tracking before promotion"
            )
        ),
    }


@dataclass(frozen=True)
class VMECJAXSpectraxTransportObjective:
    """Least-squares objective object for ``vmec_jax`` QA optimizers."""

    config: VMECJAXTransportObjectiveConfig = field(default_factory=VMECJAXTransportObjectiveConfig)
    wout_reference: Any | None = None
    name: str = "spectraxgk_transport"

    def J(self, ctx: Any, state: Any) -> jnp.ndarray:
        """Return the scalar transport objective for VMEC-JAX callbacks."""

        wout_ref = self.wout_reference if self.wout_reference is not None else _reference_wout_from_context(ctx)
        return vmec_jax_transport_objective_from_state(
            state,
            ctx.static,
            ctx.indata,
            wout_ref,
            self.config,
        )

    def to_objective_term(self, *, target: float | np.ndarray, residual_weight: float) -> Any:
        """Return a VMEC-JAX ``ObjectiveTerm`` when VMEC-JAX is installed."""

        _pin_current_optional_backend_paths()
        objective_term = getattr(importlib.import_module("vmec_jax"), "ObjectiveTerm")

        def _prepare(ctx: Any) -> Any:
            _pin_current_optional_backend_paths()
            wout_ref = self.wout_reference if self.wout_reference is not None else _reference_wout_from_context(ctx)
            prewarm_vmec_boozer_equal_arc_cache(
                ctx.static,
                wout_ref,
                mboz=int(self.config.mboz),
                nboz=int(self.config.nboz),
            )
            return objective_term(
                self.name,
                self.J,
                target=target,
                weight=residual_weight,
                metadata={
                    "objective_family": "spectraxgk_transport",
                    "spectraxgk_transport_kind": self.config.kind,
                    "gradient_scope": self.config.gradient_scope,
                    "mboz": int(self.config.mboz),
                    "nboz": int(self.config.nboz),
                    "ntheta": int(self.config.ntheta),
                    "surface_chunk_size": int(self.config.surface_chunk_size),
                    "objective_transform": self.config.objective_transform,
                    "objective_scale": float(self.config.objective_scale),
                },
            )

        return objective_term(
            self.name,
            self.J,
            target=target,
            weight=residual_weight,
            metadata={
                "objective_family": "spectraxgk_transport",
                "spectraxgk_transport_kind": self.config.kind,
                "gradient_scope": self.config.gradient_scope,
                "mboz": int(self.config.mboz),
                "nboz": int(self.config.nboz),
                "ntheta": int(self.config.ntheta),
                "surface_chunk_size": int(self.config.surface_chunk_size),
                "objective_transform": self.config.objective_transform,
                "objective_scale": float(self.config.objective_scale),
            },
            prepare=_prepare,
        )


def spectrax_transport_objective_tuple(
    *,
    weight: float,
    config: VMECJAXTransportObjectiveConfig | None = None,
    wout_reference: Any | None = None,
    target: float = 0.0,
) -> tuple[Any, float, float]:
    """Return a tuple that can be appended to ``LeastSquaresProblem.from_tuples``."""

    objective = VMECJAXSpectraxTransportObjective(
        config=config or VMECJAXTransportObjectiveConfig(),
        wout_reference=wout_reference,
    )
    return (objective.J, float(target), float(weight))
