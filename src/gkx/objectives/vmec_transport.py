"""Differentiable VMEC-JAX transport objectives and sample reductions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import asdict

from gkx.geometry.vmec_boozer_core import (
    flux_tube_geometry_from_vmec_boozer_state,
)
from gkx.objectives.core import (
    SOLVER_OBJECTIVE_NAMES,
    solver_growth_rate_from_geometry,
)


PortfolioReduction = Literal["weighted_mean", "mean", "max"]


@dataclass(frozen=True)
class StellaratorObjectivePortfolioContract:
    """Static shape/weight contract for a reduced objective portfolio."""

    n_surfaces: int
    n_alphas: int
    n_ky: int
    n_objectives: int
    reduction: PortfolioReduction
    uses_sample_weights: bool
    uses_separable_sample_weights: bool
    uses_objective_weights: bool

    @property
    def row_shape(self) -> tuple[int, int, int, int]:
        """Expected objective-table shape ``(surface, alpha, ky, objective)``."""

        return (self.n_surfaces, self.n_alphas, self.n_ky, self.n_objectives)

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        """Expected sample-weight shape ``(surface, alpha, ky)``."""

        return (self.n_surfaces, self.n_alphas, self.n_ky)

    @property
    def n_samples(self) -> int:
        """Number of surface/alpha/ky samples in the portfolio."""

        return self.n_surfaces * self.n_alphas * self.n_ky

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        payload = asdict(self)
        payload["row_shape"] = list(self.row_shape)
        payload["sample_shape"] = list(self.sample_shape)
        payload["n_samples"] = int(self.n_samples)
        return payload


def _is_real_numeric_dtype(dtype: jnp.dtype) -> bool:
    return bool(
        jnp.issubdtype(dtype, jnp.number)
        and not jnp.issubdtype(dtype, jnp.complexfloating)
    )


def _floating_dtype(*arrays: jnp.ndarray) -> jnp.dtype:
    return jnp.result_type(*(arrays + (jnp.asarray(1.0),)))


def _objective_rows(objective_rows: Any) -> jnp.ndarray:
    rows = jnp.asarray(objective_rows)
    if int(rows.ndim) != 4:
        raise ValueError(
            "objective_rows must have shape (n_surface, n_alpha, n_ky, n_objective)"
        )
    if any(int(size) < 1 for size in rows.shape):
        raise ValueError("objective_rows dimensions must all be positive")
    if not _is_real_numeric_dtype(rows.dtype):
        raise TypeError("objective_rows must be a real numeric array")
    return rows


def _concrete_numpy_array(value: jnp.ndarray | Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float)
    except (
        Exception
    ) as exc:  # pragma: no cover - exercised by JAX tracers under jit/grad.
        class_name = type(exc).__name__
        if "Tracer" in class_name or "Concretization" in class_name:
            return None
        raise


def _validate_concrete_weights(weights: jnp.ndarray | Any, *, name: str) -> None:
    concrete = _concrete_numpy_array(weights)
    if concrete is None:
        return
    if not np.all(np.isfinite(concrete)):
        raise ValueError(f"{name} must be finite")
    if np.any(concrete < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if float(np.sum(concrete)) <= 0.0:
        raise ValueError(f"{name} must have positive sum")


def _normalized_vector(
    weights: Any | None,
    *,
    size: int,
    name: str,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if weights is None:
        return jnp.full((int(size),), 1.0 / float(size), dtype=dtype)
    array = jnp.asarray(weights, dtype=dtype)
    if int(array.ndim) != 1 or int(array.shape[0]) != int(size):
        raise ValueError(f"{name} must be a length-{int(size)} vector")
    _validate_concrete_weights(array, name=name)
    return array / jnp.sum(array)


def _normalized_sample_weights(
    objective_rows: jnp.ndarray,
    *,
    sample_weights: Any | None,
    surface_weights: Any | None,
    alpha_weights: Any | None,
    ky_weights: Any | None,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    n_surface, n_alpha, n_ky, _n_objective = (
        int(size) for size in objective_rows.shape
    )
    has_sample_weights = sample_weights is not None
    has_axis_weights = any(
        weight is not None for weight in (surface_weights, alpha_weights, ky_weights)
    )
    if has_sample_weights and has_axis_weights:
        raise ValueError(
            "provide either sample_weights or separable surface/alpha/ky weights, not both"
        )
    if has_sample_weights:
        array = jnp.asarray(sample_weights, dtype=dtype)
        if tuple(int(size) for size in array.shape) != (n_surface, n_alpha, n_ky):
            raise ValueError(
                "sample_weights must have shape (n_surface, n_alpha, n_ky)"
            )
        _validate_concrete_weights(array, name="sample_weights")
        return array / jnp.sum(array)

    surface = _normalized_vector(
        surface_weights, size=n_surface, name="surface_weights", dtype=dtype
    )
    alpha = _normalized_vector(
        alpha_weights, size=n_alpha, name="alpha_weights", dtype=dtype
    )
    ky = _normalized_vector(ky_weights, size=n_ky, name="ky_weights", dtype=dtype)
    return surface[:, None, None] * alpha[None, :, None] * ky[None, None, :]


def portfolio_sample_weight_tensor(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
) -> jnp.ndarray:
    """Return normalized sample weights with shape ``(surface, alpha, ky)``."""

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    return _normalized_sample_weights(
        rows,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        dtype=dtype,
    )


def portfolio_objective_weight_vector(
    objective_rows: Any,
    *,
    objective_weights: Any | None = None,
) -> jnp.ndarray:
    """Return normalized objective-column weights."""

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    return _normalized_vector(
        objective_weights,
        size=int(rows.shape[-1]),
        name="objective_weights",
        dtype=dtype,
    )


def validate_objective_portfolio_contract(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
) -> StellaratorObjectivePortfolioContract:
    """Validate row/weight shapes and concrete finite non-negative weights."""

    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")
    if reduction == "mean" and any(
        weight is not None
        for weight in (
            sample_weights,
            surface_weights,
            alpha_weights,
            ky_weights,
            objective_weights,
        )
    ):
        raise ValueError("mean reduction does not accept weights; use weighted_mean")
    if reduction == "max" and any(
        weight is not None
        for weight in (sample_weights, surface_weights, alpha_weights, ky_weights)
    ):
        raise ValueError("max reduction does not accept sample weights")

    rows = _objective_rows(objective_rows)
    _ = portfolio_sample_weight_tensor(
        rows,
        sample_weights=sample_weights if reduction == "weighted_mean" else None,
        surface_weights=surface_weights if reduction == "weighted_mean" else None,
        alpha_weights=alpha_weights if reduction == "weighted_mean" else None,
        ky_weights=ky_weights if reduction == "weighted_mean" else None,
    )
    _ = portfolio_objective_weight_vector(
        rows,
        objective_weights=objective_weights
        if reduction in ("weighted_mean", "max")
        else None,
    )

    n_surface, n_alpha, n_ky, n_objective = (int(size) for size in rows.shape)
    return StellaratorObjectivePortfolioContract(
        n_surfaces=n_surface,
        n_alphas=n_alpha,
        n_ky=n_ky,
        n_objectives=n_objective,
        reduction=reduction,
        uses_sample_weights=sample_weights is not None and reduction == "weighted_mean",
        uses_separable_sample_weights=any(
            weight is not None
            for weight in (surface_weights, alpha_weights, ky_weights)
        )
        and reduction == "weighted_mean",
        uses_objective_weights=objective_weights is not None
        and reduction in ("weighted_mean", "max"),
    )


def aggregate_objective_portfolio(
    objective_rows: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
    validate: bool = True,
) -> jnp.ndarray:
    """Reduce a ``(surface, alpha, ky, objective)`` table to one scalar.

    ``weighted_mean`` normalizes sample and objective weights; ``mean`` is
    unweighted, while ``max`` selects the worst objective-weighted sample.
    """

    if validate:
        validate_objective_portfolio_contract(
            objective_rows,
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
            reduction=reduction,
        )

    rows = _objective_rows(objective_rows)
    dtype = _floating_dtype(rows)
    values = rows.astype(dtype)

    if reduction == "mean":
        return jnp.mean(values)
    if reduction == "weighted_mean":
        sample = _normalized_sample_weights(
            values,
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            dtype=dtype,
        )
        objective = _normalized_vector(
            objective_weights,
            size=int(values.shape[-1]),
            name="objective_weights",
            dtype=dtype,
        )
        return jnp.sum(values * sample[..., None] * objective)
    if reduction == "max":
        objective = _normalized_vector(
            objective_weights,
            size=int(values.shape[-1]),
            name="objective_weights",
            dtype=dtype,
        )
        return jnp.max(jnp.sum(values * objective, axis=-1))
    raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")


VMEXTransportObjectiveKind = Literal[
    "growth",
    "quasilinear_flux",
    "nonlinear_window_heat_flux",
]
VMEXTransportObjectiveTransform = Literal["raw", "scaled", "log1p"]


def _smooth_positive(x: jnp.ndarray | float, *, beta: float = 18.0) -> jnp.ndarray:
    """Smooth positive part used to keep objectives differentiable near marginality."""

    arr = jnp.asarray(x)
    beta_arr = jnp.asarray(beta, dtype=arr.dtype)
    return jax.nn.softplus(beta_arr * arr) / beta_arr


@dataclass(frozen=True)
class StellaratorITGSampleSet:
    """Reduced multi-surface/multi-alpha/multi-``k_y`` ITG portfolio contract."""

    surfaces: tuple[float, ...] = (0.50, 0.64, 0.78)
    alphas: tuple[float, ...] = (0.0, 1.0471975511965976)
    ky_values: tuple[float, ...] = (0.10, 0.30, 0.50)
    surface_weights: tuple[float, ...] | None = None
    alpha_weights: tuple[float, ...] | None = None
    ky_weights: tuple[float, ...] | None = None
    reduction: PortfolioReduction = "weighted_mean"

    def __post_init__(self) -> None:
        for name, values in (
            ("surfaces", self.surfaces),
            ("alphas", self.alphas),
            ("ky_values", self.ky_values),
        ):
            arr = np.asarray(values, dtype=float)
            if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a non-empty finite vector")
        if np.any(np.asarray(self.ky_values, dtype=float) <= 0.0):
            raise ValueError("ky_values must be positive")
        for name, weights, expected in (
            ("surface_weights", self.surface_weights, len(self.surfaces)),
            ("alpha_weights", self.alpha_weights, len(self.alphas)),
            ("ky_weights", self.ky_weights, len(self.ky_values)),
        ):
            if weights is None:
                continue
            arr = np.asarray(weights, dtype=float)
            if arr.ndim != 1 or arr.size != expected or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must be a finite length-{expected} vector")
            if np.any(arr < 0.0) or float(np.sum(arr)) <= 0.0:
                raise ValueError(f"{name} must be non-negative with positive sum")
        if self.reduction not in ("weighted_mean", "mean", "max"):
            raise ValueError("reduction must be weighted_mean, mean, or max")

    @property
    def n_samples(self) -> int:
        """Number of surface/alpha/ky samples in the rectangular portfolio."""

        return len(self.surfaces) * len(self.alphas) * len(self.ky_values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "surfaces": list(self.surfaces),
            "alphas": list(self.alphas),
            "ky_values": list(self.ky_values),
            "surface_weights": None
            if self.surface_weights is None
            else list(self.surface_weights),
            "alpha_weights": None
            if self.alpha_weights is None
            else list(self.alpha_weights),
            "ky_weights": None if self.ky_weights is None else list(self.ky_weights),
            "reduction": self.reduction,
            "n_samples": self.n_samples,
        }


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
    """Keep geometry discovery on the same optional backends already imported.

    The differentiable-geometry bridge intentionally prefers explicit local
    checkouts over globally installed packages.  When examples run from a fresh
    temporary clone while another vmex checkout exists in ``$HOME``, that
    preference can otherwise evict the vmex module that owns the traced
    optimization state.  Pinning the currently importable backend paths makes
    the vmex/GKX objective reproducible without requiring users to
    hand-set environment variables (the historical ``*_VMEX_PATH``
    environment-variable names are retained).
    """

    if not (os.environ.get("GKX_VMEX_PATH") or os.environ.get("VMEX_PATH")):
        root = _module_search_root("vmex")
        if root is not None:
            os.environ.setdefault("GKX_VMEX_PATH", str(root))
    if not (
        os.environ.get("GKX_BOOZ_XFORM_JAX_PATH")
        or os.environ.get("BOOZ_XFORM_JAX_PATH")
    ):
        root = _module_search_root("booz_xform_jax")
        if root is not None:
            os.environ.setdefault("GKX_BOOZ_XFORM_JAX_PATH", str(root))


@dataclass(frozen=True)
class VMEXTransportObjectiveConfig:
    """Configuration for VMEC-JAX to GKX objective evaluation."""

    kind: VMEXTransportObjectiveKind = "nonlinear_window_heat_flux"
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
    objective_transform: VMEXTransportObjectiveTransform = "raw"
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
        if self.kind not in (
            "growth",
            "quasilinear_flux",
            "nonlinear_window_heat_flux",
        ):
            raise ValueError(f"unknown VMEC-JAX transport objective kind {self.kind!r}")
        if int(self.ntheta) < 4:
            raise ValueError("ntheta must be >= 4")
        if int(self.mboz) < 21 or int(self.nboz) < 21:
            raise ValueError(
                "mboz and nboz must be at least 21 for paper-facing QA optimization"
            )
        if int(self.n_laguerre) < 1 or int(self.n_hermite) < 1:
            raise ValueError("n_laguerre and n_hermite must be positive")
        if int(self.nx) < 1 or int(self.ny) < 3:
            raise ValueError("nx must be positive and ny must be at least 3")
        if float(self.nonlinear_csat) <= 0.0:
            raise ValueError("nonlinear_csat must be positive")
        if self.objective_transform not in ("raw", "scaled", "log1p"):
            raise ValueError(
                f"unknown VMEC-JAX transport objective transform {self.objective_transform!r}"
            )
        if float(self.objective_scale) <= 0.0:
            raise ValueError("objective_scale must be positive")
        if int(self.surface_chunk_size) < 0:
            raise ValueError("surface_chunk_size must be non-negative")
        if int(self.surface_chunk_size) > 0 and self.sample_set.reduction not in (
            "weighted_mean",
            "mean",
        ):
            raise ValueError(
                "surface_chunk_size currently supports only mean or weighted_mean reductions"
            )

    def objective_options(self) -> dict[str, Any]:
        """Return GKX solver options for this objective."""

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


_SOLVER_OBJECTIVE_INDEX = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}


def _solver_table_to_nonlinear_window_proxy(
    table: jnp.ndarray,
    config: VMEXTransportObjectiveConfig,
) -> jnp.ndarray:
    """Map linear solver rows to a smooth reduced nonlinear heat-flux proxy."""

    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    gamma = jnp.asarray(table[..., idx["gamma"]])
    kperp_eff2 = jnp.asarray(table[..., idx["kperp_eff2"]])
    heat_weight = jnp.asarray(table[..., idx["linear_heat_flux_weight"]])
    gamma_plus = _smooth_positive(gamma, beta=18.0)
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
    config: VMEXTransportObjectiveConfig,
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
    row = jnp.zeros((len(SOLVER_OBJECTIVE_NAMES),), dtype=jnp.asarray(gamma).dtype)
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
    config: VMEXTransportObjectiveConfig,
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
    config: VMEXTransportObjectiveConfig,
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
    cfg: VMEXTransportObjectiveConfig,
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
    cfg: VMEXTransportObjectiveConfig,
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


def vmex_transport_objective_from_state(
    state: Any,
    static: Any,
    indata: Any,
    wout_reference: Any,
    config: VMEXTransportObjectiveConfig | None = None,
) -> jnp.ndarray:
    """Evaluate a scalar GKX transport objective from a VMEC-JAX state."""

    _pin_current_optional_backend_paths()
    cfg = config or VMEXTransportObjectiveConfig()
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


@dataclass(frozen=True)
class VMEXGKXTransportObjective:
    """Evaluate a configured transport metric from a solved VMEC-JAX state."""

    config: VMEXTransportObjectiveConfig = field(
        default_factory=VMEXTransportObjectiveConfig
    )
    wout_reference: Any | None = None

    def J(self, ctx: Any, state: Any) -> jnp.ndarray:
        """Return the scalar transport objective for VMEC-JAX callbacks."""

        wout_ref = (
            self.wout_reference
            if self.wout_reference is not None
            else _reference_wout_from_context(ctx)
        )
        return vmex_transport_objective_from_state(
            state,
            ctx.static,
            ctx.indata,
            wout_ref,
            self.config,
        )


__all__ = [
    "VMEXGKXTransportObjective",
    "VMEXTransportObjectiveConfig",
    "VMEXTransportObjectiveKind",
    "VMEXTransportObjectiveTransform",
    "vmex_transport_objective_from_state",
    "PortfolioReduction",
    "StellaratorObjectivePortfolioContract",
    "aggregate_objective_portfolio",
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "validate_objective_portfolio_contract",
]
