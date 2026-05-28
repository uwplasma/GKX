"""Backend-free reduced objective portfolios for stellarator optimization.

This module only reduces already-evaluated objective rows.  It intentionally
does not import VMEC, Boozer, or solver backends so the same contract can be
used by fast CI fixtures and by production VMEC/Boozer objective drivers after
they have built a per-surface/per-alpha/per-ky objective table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import autodiff_finite_difference_report, covariance_diagnostics


PortfolioReduction = Literal["weighted_mean", "mean", "max"]

_GROWTH_OBJECTIVE_NAMES = frozenset(("gamma", "growth", "growth_rate"))
_QUASILINEAR_OBJECTIVE_NAMES = frozenset(
    (
        "quasilinear_flux",
        "quasilinear_heat_flux",
        "mixing_length_heat_flux_proxy",
        "linear_heat_flux_weight",
    )
)


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


@dataclass(frozen=True)
class ReducedPortfolioArtifactGuardConfig:
    """Requirements for promoting real VMEC/Boozer reduced-portfolio rows."""

    min_alphas: int = 2
    min_ky: int = 2
    min_objectives: int = 1
    min_boozer_mode: int = 21
    require_growth_objective: bool = True
    require_quasilinear_objective: bool = True
    require_vmec_paths: bool = True
    value_rtol: float = 1.0e-8
    value_atol: float = 1.0e-8

    def __post_init__(self) -> None:
        if int(self.min_alphas) < 1:
            raise ValueError("min_alphas must be >= 1")
        if int(self.min_ky) < 1:
            raise ValueError("min_ky must be >= 1")
        if int(self.min_objectives) < 1:
            raise ValueError("min_objectives must be >= 1")
        if int(self.min_boozer_mode) < 1:
            raise ValueError("min_boozer_mode must be >= 1")
        if float(self.value_rtol) < 0.0 or float(self.value_atol) < 0.0:
            raise ValueError("value tolerances must be non-negative")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation."""

        return asdict(self)


def _is_real_numeric_dtype(dtype: jnp.dtype) -> bool:
    return bool(jnp.issubdtype(dtype, jnp.number) and not jnp.issubdtype(dtype, jnp.complexfloating))


def _floating_dtype(*arrays: jnp.ndarray) -> jnp.dtype:
    return jnp.result_type(*(arrays + (jnp.asarray(1.0),)))


def _objective_rows(objective_rows: Any) -> jnp.ndarray:
    rows = jnp.asarray(objective_rows)
    if int(rows.ndim) != 4:
        raise ValueError("objective_rows must have shape (n_surface, n_alpha, n_ky, n_objective)")
    if any(int(size) < 1 for size in rows.shape):
        raise ValueError("objective_rows dimensions must all be positive")
    if not _is_real_numeric_dtype(rows.dtype):
        raise TypeError("objective_rows must be a real numeric array")
    return rows


def _parameter_vector(params: Any) -> jnp.ndarray:
    p = jnp.asarray(params)
    if int(p.ndim) != 1:
        raise ValueError("params must be a one-dimensional vector")
    if int(p.shape[0]) < 1:
        raise ValueError("params must contain at least one parameter")
    if not _is_real_numeric_dtype(p.dtype):
        raise TypeError("params must be a real numeric vector")
    return jnp.asarray(p, dtype=_floating_dtype(p))


def _concrete_numpy_array(value: jnp.ndarray | Any) -> np.ndarray | None:
    try:
        return np.asarray(value, dtype=float)
    except Exception as exc:  # pragma: no cover - exercised by JAX tracers under jit/grad.
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
    n_surface, n_alpha, n_ky, _n_objective = (int(size) for size in objective_rows.shape)
    has_sample_weights = sample_weights is not None
    has_axis_weights = any(weight is not None for weight in (surface_weights, alpha_weights, ky_weights))
    if has_sample_weights and has_axis_weights:
        raise ValueError("provide either sample_weights or separable surface/alpha/ky weights, not both")
    if has_sample_weights:
        array = jnp.asarray(sample_weights, dtype=dtype)
        if tuple(int(size) for size in array.shape) != (n_surface, n_alpha, n_ky):
            raise ValueError("sample_weights must have shape (n_surface, n_alpha, n_ky)")
        _validate_concrete_weights(array, name="sample_weights")
        return array / jnp.sum(array)

    surface = _normalized_vector(surface_weights, size=n_surface, name="surface_weights", dtype=dtype)
    alpha = _normalized_vector(alpha_weights, size=n_alpha, name="alpha_weights", dtype=dtype)
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
    """Validate static row/weight contracts and return portfolio metadata.

    Concrete weights must be finite, non-negative, and have positive sum.  Under
    JAX tracing, value-level weight checks are deferred to the caller, but shape
    contracts remain enforced from static array shapes.
    """

    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("reduction must be one of 'weighted_mean', 'mean', or 'max'")
    if reduction == "mean" and any(
        weight is not None
        for weight in (sample_weights, surface_weights, alpha_weights, ky_weights, objective_weights)
    ):
        raise ValueError("mean reduction does not accept weights; use weighted_mean")
    if reduction == "max" and any(
        weight is not None for weight in (sample_weights, surface_weights, alpha_weights, ky_weights)
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
        objective_weights=objective_weights if reduction in ("weighted_mean", "max") else None,
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
            weight is not None for weight in (surface_weights, alpha_weights, ky_weights)
        )
        and reduction == "weighted_mean",
        uses_objective_weights=objective_weights is not None and reduction in ("weighted_mean", "max"),
    )


def _conditioning_gate(
    jacobian: np.ndarray,
    *,
    min_rank: int | None,
    condition_number_limit: float,
) -> dict[str, object]:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional array")
    limit = float(condition_number_limit)
    if limit <= 0.0:
        raise ValueError("condition_number_limit must be positive")
    expected_rank = int(min_rank) if min_rank is not None else int(min(jac.shape))
    if expected_rank < 1:
        raise ValueError("min_rank must be >= 1")
    if expected_rank > int(min(jac.shape)):
        raise ValueError("min_rank cannot exceed min(jacobian.shape)")

    finite = bool(np.all(np.isfinite(jac)))
    singular_values = np.linalg.svd(jac, compute_uv=False) if finite else np.asarray([], dtype=float)
    rank = int(np.linalg.matrix_rank(jac)) if finite else 0
    if singular_values.size == 0 or float(singular_values[-1]) <= 0.0:
        condition_number = float("inf")
        smallest = 0.0
    else:
        smallest = float(singular_values[-1])
        condition_number = float(singular_values[0] / singular_values[-1])
    passed = bool(finite and rank >= expected_rank and condition_number <= limit)
    return {
        "passed": passed,
        "finite_jacobian": finite,
        "sensitivity_map_rank": rank,
        "min_rank": expected_rank,
        "rank_deficiency": int(max(expected_rank - rank, 0)),
        "jacobian_condition_number": condition_number,
        "condition_number_limit": limit,
        "smallest_singular_value": smallest,
        "singular_values": singular_values.tolist(),
    }


def objective_portfolio_sensitivity_report(
    objective_row_fn: Callable[[jnp.ndarray], Any],
    params: Any,
    *,
    sample_weights: Any | None = None,
    surface_weights: Any | None = None,
    alpha_weights: Any | None = None,
    ky_weights: Any | None = None,
    objective_weights: Any | None = None,
    reduction: PortfolioReduction = "weighted_mean",
    step: float = 1.0e-4,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
    min_rank: int | None = None,
    condition_number_limit: float = 1.0e8,
    covariance_regularization: float = 1.0e-9,
    workers: int = 1,
    parallel_executor: str = "thread",
) -> dict[str, object]:
    """AD/FD and conditioning report for a reduced objective-row portfolio.

    ``objective_row_fn`` is the backend boundary: production callers can wire a
    VMEC/Boozer/quasilinear row builder into this gate while tests can use a
    cheap fixture. The report checks both the final scalar reduction and the
    unreduced row sensitivity map so a passing scalar gradient cannot hide a
    rank-deficient or badly conditioned objective table.
    """

    p = _parameter_vector(params)

    def row_table(x: jnp.ndarray) -> jnp.ndarray:
        rows = _objective_rows(objective_row_fn(x))
        return rows.astype(_floating_dtype(x, rows))

    base_rows = row_table(p)
    contract = validate_objective_portfolio_contract(
        base_rows,
        sample_weights=sample_weights,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
        reduction=reduction,
    )

    def scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
        return aggregate_objective_portfolio(
            row_table(x),
            sample_weights=sample_weights,
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
            reduction=reduction,
            validate=True,
        )

    def row_vector_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ravel(row_table(x))

    scalar_gradient_gate = autodiff_finite_difference_report(
        scalar_fn,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    row_jacobian_gate = autodiff_finite_difference_report(
        row_vector_fn,
        p,
        step=step,
        rtol=rtol,
        atol=atol,
        workers=workers,
        parallel_executor=parallel_executor,
    )
    row_jacobian = np.asarray(row_jacobian_gate["jacobian_ad"], dtype=float)
    row_residual = np.asarray(row_vector_fn(p), dtype=float)
    conditioning_gate = _conditioning_gate(
        row_jacobian,
        min_rank=min_rank,
        condition_number_limit=condition_number_limit,
    )
    covariance = covariance_diagnostics(
        row_jacobian,
        row_residual,
        regularization=covariance_regularization,
    )
    covariance["source"] = "objective_portfolio_rows"

    return {
        "kind": "objective_portfolio_sensitivity_report",
        "passed": bool(
            scalar_gradient_gate["passed"]
            and row_jacobian_gate["passed"]
            and conditioning_gate["passed"]
        ),
        "portfolio_contract": contract.to_dict(),
        "parameter_count": int(p.size),
        "base_value": float(scalar_fn(p)),
        "base_row_norm": float(jnp.linalg.norm(row_vector_fn(p))),
        "scalar_gradient_gate": scalar_gradient_gate,
        "row_jacobian_gate": row_jacobian_gate,
        "conditioning_gate": conditioning_gate,
        "covariance": covariance,
    }


def _artifact_text(payload: dict[str, Any], *keys: str) -> str:
    return " ".join(str(payload.get(key, "")) for key in keys).lower()


def _as_finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nested(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(array.size > 0 and np.all(np.isfinite(array)))


def _sample_key(sample: dict[str, Any]) -> tuple[str, float, float]:
    surface = sample.get("surface_index", "__mid_surface__")
    if surface is None and sample.get("torflux") is not None:
        surface_key = f"torflux={_as_finite_float(sample.get('torflux'), name='sample torflux'):.16g}"
    elif surface is None and sample.get("surface") is not None:
        surface_key = f"surface={_as_finite_float(sample.get('surface'), name='sample surface'):.16g}"
    else:
        surface_key = "__mid_surface__" if surface is None else str(surface)
    alpha = _as_finite_float(sample.get("alpha"), name="sample alpha")
    ky_value = sample.get("ky", sample.get("ky_index", sample.get("selected_ky_index")))
    ky = _as_finite_float(ky_value, name="sample ky")
    return surface_key, alpha, ky


def _axis_indices(samples: list[dict[str, Any]]) -> tuple[list[str], list[float], list[float]]:
    surfaces = sorted({_sample_key(sample)[0] for sample in samples})
    alphas = sorted({_sample_key(sample)[1] for sample in samples})
    kys = sorted({_sample_key(sample)[2] for sample in samples})
    return surfaces, alphas, kys


def _artifact_sample_values(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    samples_raw = payload.get("samples")
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ValueError("payload must contain a non-empty samples list")
    samples = [item for item in samples_raw if isinstance(item, dict)]
    if len(samples) != len(samples_raw):
        raise ValueError("all samples must be dictionaries")
    values = np.asarray(payload.get("base_sample_values"), dtype=float)
    if values.ndim != 1 or int(values.shape[0]) != len(samples):
        raise ValueError("base_sample_values must be a length-n_samples vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("base_sample_values must be finite")
    return samples, values


def _artifact_sample_value_tensor(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[float], list[float]]:
    samples, values = _artifact_sample_values(payload)
    surfaces, alphas, kys = _axis_indices(samples)
    shape = (len(surfaces), len(alphas), len(kys))
    table = np.full(shape, np.nan, dtype=float)
    weights = np.full(shape, np.nan, dtype=float)
    seen: set[tuple[int, int, int]] = set()
    surface_index = {value: index for index, value in enumerate(surfaces)}
    alpha_index = {value: index for index, value in enumerate(alphas)}
    ky_index = {value: index for index, value in enumerate(kys)}
    for sample, value in zip(samples, values, strict=True):
        surface, alpha, ky = _sample_key(sample)
        idx = (surface_index[surface], alpha_index[alpha], ky_index[ky])
        if idx in seen:
            raise ValueError("samples must not contain duplicate surface/alpha/ky rows")
        seen.add(idx)
        table[idx] = value
        weights[idx] = _as_finite_float(sample.get("weight", 1.0), name="sample weight")
    if len(seen) != int(np.prod(shape)) or not np.all(np.isfinite(table)):
        raise ValueError("samples must form a complete rectangular surface/alpha/ky table")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("sample weights must be finite, non-negative, and have positive sum")
    return table[..., None], weights, surfaces, alphas, kys


def _artifact_full_objective_table(payload: dict[str, Any], *, n_samples: int) -> np.ndarray:
    table = np.asarray(payload.get("base_objective_table"), dtype=float)
    if table.ndim != 2:
        raise ValueError("base_objective_table must be a two-dimensional array")
    if int(table.shape[0]) != int(n_samples):
        raise ValueError("base_objective_table row count must match samples")
    if not np.all(np.isfinite(table)):
        raise ValueError("base_objective_table must be finite")
    return table


def _artifact_objective_name_gate(
    objective_names: list[str],
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    lowered = {name.lower() for name in objective_names}
    has_growth = bool(lowered & _GROWTH_OBJECTIVE_NAMES)
    has_quasilinear = bool(lowered & _QUASILINEAR_OBJECTIVE_NAMES)
    passed = bool(
        len(objective_names) >= int(config.min_objectives)
        and (has_growth or not config.require_growth_objective)
        and (has_quasilinear or not config.require_quasilinear_objective)
    )
    return {
        "passed": passed,
        "objective_names": objective_names,
        "n_objectives": len(objective_names),
        "min_objectives": int(config.min_objectives),
        "has_growth_objective": has_growth,
        "has_quasilinear_objective": has_quasilinear,
        "requires_growth_objective": bool(config.require_growth_objective),
        "requires_quasilinear_objective": bool(config.require_quasilinear_objective),
    }


def _artifact_provenance_gate(
    payload: dict[str, Any],
    *,
    config: ReducedPortfolioArtifactGuardConfig,
) -> dict[str, object]:
    options_raw = payload.get("options")
    options: dict[str, Any] = options_raw if isinstance(options_raw, dict) else {}
    mboz = int(payload.get("mboz") or options.get("mboz") or 0)
    nboz = int(payload.get("nboz") or options.get("nboz") or 0)
    text = _artifact_text(payload, "kind", "artifact_kind", "source_scope", "claim_scope", "builder")
    has_vmec_boozer_scope = "vmec_boozer" in text or "vmec/boozer" in text
    input_path = str(payload.get("input_path", ""))
    wout_path = str(payload.get("wout_path", ""))
    has_paths = bool(input_path and wout_path)
    mode_gate = bool(mboz >= int(config.min_boozer_mode) and nboz >= int(config.min_boozer_mode))
    return {
        "passed": bool(
            has_vmec_boozer_scope
            and mode_gate
            and (has_paths or not config.require_vmec_paths)
        ),
        "has_vmec_boozer_scope": has_vmec_boozer_scope,
        "requires_vmec_paths": bool(config.require_vmec_paths),
        "has_input_and_wout_paths": has_paths,
        "input_path": input_path,
        "wout_path": wout_path,
        "mboz": mboz,
        "nboz": nboz,
        "min_boozer_mode": int(config.min_boozer_mode),
    }


def _artifact_claim_scope_gate(payload: dict[str, Any]) -> dict[str, object]:
    objective = str(payload.get("objective", payload.get("objective_kind", ""))).lower()
    text = _artifact_text(payload, "claim_scope", "next_action")
    has_safe_disclaimer = any(
        phrase in text
        for phrase in (
            "not a nonlinear",
            "not a production nonlinear",
            "not an optimized-equilibrium nonlinear",
            "nonlinear transport optimization still requires",
        )
    )
    no_nonlinear_objective = objective not in {
        "nonlinear_heat_flux",
        "nonlinear_window_heat_flux_mean",
        "production_nonlinear_heat_flux",
    }
    no_trace = payload.get("nonlinear_trace") in (None, {}, [])
    return {
        "passed": bool(no_nonlinear_objective and no_trace and has_safe_disclaimer),
        "objective": objective,
        "nonlinear_trace_absent": no_trace,
        "has_nonproduction_disclaimer": has_safe_disclaimer,
    }


def _artifact_fd_gate(payload: dict[str, Any]) -> dict[str, object]:
    finite_fields = {
        name: _finite_nested(payload.get(name))
        for name in (
            "minus_sample_values",
            "base_sample_values",
            "plus_sample_values",
            "minus_objective_table",
            "base_objective_table",
            "plus_objective_table",
        )
    }
    scalar_fields = {
        name: bool(np.isfinite(float(payload.get(name, np.nan))))
        for name in (
            "base_value",
            "minus_value",
            "plus_value",
            "central_derivative",
            "response_abs",
            "curvature_ratio",
        )
    }
    passed = bool(
        payload.get("passed", False)
        and payload.get("finite_values", True)
        and payload.get("finite_difference_consistent", False)
        and payload.get("response_resolved", True)
        and all(finite_fields.values())
        and all(scalar_fields.values())
    )
    return {
        "passed": passed,
        "artifact_passed": bool(payload.get("passed", False)),
        "finite_difference_consistent": bool(payload.get("finite_difference_consistent", False)),
        "response_resolved": bool(payload.get("response_resolved", True)),
        "finite_array_fields": finite_fields,
        "finite_scalar_fields": scalar_fields,
    }


def _gradient_artifact_gate(gradient_artifacts: list[dict[str, Any]]) -> dict[str, object]:
    objective_names: set[str] = set()
    n_gates = 0
    n_passed = 0
    finite = True
    for artifact in gradient_artifacts:
        for gate in artifact.get("objective_gates", []) if isinstance(artifact, dict) else []:
            if not isinstance(gate, dict):
                finite = False
                continue
            n_gates += 1
            objective_names.add(str(gate.get("objective", "")).lower())
            n_passed += int(bool(gate.get("passed", False)))
            finite = bool(finite and all(_finite_nested(gate.get(name)) for name in ("implicit", "finite_difference", "abs_error", "rel_error")))
    has_growth = bool(objective_names & _GROWTH_OBJECTIVE_NAMES)
    has_quasilinear = bool(objective_names & _QUASILINEAR_OBJECTIVE_NAMES)
    passed = bool(n_gates > 0 and n_passed == n_gates and finite and has_growth and has_quasilinear)
    return {
        "passed": passed,
        "n_gradient_artifacts": len(gradient_artifacts),
        "n_objective_gates": n_gates,
        "n_passed_objective_gates": n_passed,
        "finite_ad_fd_values": finite,
        "objective_names": sorted(objective_names),
        "has_growth_ad_fd_gate": has_growth,
        "has_quasilinear_ad_fd_gate": has_quasilinear,
    }


def reduced_portfolio_artifact_guard_report(
    row_artifact: dict[str, Any],
    *,
    gradient_artifacts: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    config: ReducedPortfolioArtifactGuardConfig | None = None,
) -> dict[str, object]:
    """Validate a real VMEC/Boozer reduced-portfolio artifact before promotion.

    The guard is backend-free: it consumes already-generated JSON payloads,
    rebuilds a ``(surface, alpha, ky, objective)`` reducer table from real
    VMEC/Boozer sample rows, and checks that provenance, coverage, FD/AD
    diagnostics, and nonlinear-claim boundaries are explicit.
    """

    cfg = config or ReducedPortfolioArtifactGuardConfig()
    sample_rows, sample_weights, surfaces, alphas, kys = _artifact_sample_value_tensor(row_artifact)
    samples, _values = _artifact_sample_values(row_artifact)
    full_table = _artifact_full_objective_table(row_artifact, n_samples=len(samples))
    objective_names = [str(item) for item in row_artifact.get("objective_names", [])]
    objective_name_gate = _artifact_objective_name_gate(objective_names, config=cfg)
    reduction = str(row_artifact.get("reduction", "weighted_mean"))
    if reduction not in ("weighted_mean", "mean", "max"):
        raise ValueError("artifact reduction must be weighted_mean, mean, or max")
    reducer_reduction = cast(PortfolioReduction, reduction)
    if reduction == "weighted_mean":
        reduced_value = float(
            aggregate_objective_portfolio(
                sample_rows,
                sample_weights=sample_weights,
                reduction=reducer_reduction,
            )
        )
        contract = validate_objective_portfolio_contract(
            sample_rows,
            sample_weights=sample_weights,
            reduction=reducer_reduction,
        )
    else:
        reduced_value = float(aggregate_objective_portfolio(sample_rows, reduction=reducer_reduction))
        contract = validate_objective_portfolio_contract(sample_rows, reduction=reducer_reduction)
    artifact_value = _as_finite_float(row_artifact.get("base_value"), name="base_value")
    reducer_matches = bool(np.isclose(reduced_value, artifact_value, rtol=cfg.value_rtol, atol=cfg.value_atol))
    coverage_gate = {
        "passed": bool(len(alphas) >= int(cfg.min_alphas) and len(kys) >= int(cfg.min_ky)),
        "surface_labels": surfaces,
        "alphas": alphas,
        "ky_values": kys,
        "n_surfaces": len(surfaces),
        "n_alphas": len(alphas),
        "n_ky": len(kys),
        "n_samples": len(samples),
        "min_alphas": int(cfg.min_alphas),
        "min_ky": int(cfg.min_ky),
    }
    full_table_gate = {
        "passed": bool(
            full_table.shape[0] == len(samples)
            and full_table.shape[1] >= int(cfg.min_objectives)
            and np.all(np.isfinite(full_table))
        ),
        "shape": list(full_table.shape),
        "finite": bool(np.all(np.isfinite(full_table))),
    }
    reducer_gate = {
        "passed": reducer_matches,
        "reduction": reduction,
        "reduced_base_value": reduced_value,
        "artifact_base_value": artifact_value,
        "abs_error": float(abs(reduced_value - artifact_value)),
        "rtol": float(cfg.value_rtol),
        "atol": float(cfg.value_atol),
        "contract": contract.to_dict(),
    }
    provenance_gate = _artifact_provenance_gate(row_artifact, config=cfg)
    claim_scope_gate = _artifact_claim_scope_gate(row_artifact)
    fd_gate = _artifact_fd_gate(row_artifact)
    ad_fd_gate = _gradient_artifact_gate(list(gradient_artifacts))
    passed = bool(
        provenance_gate["passed"]
        and coverage_gate["passed"]
        and full_table_gate["passed"]
        and objective_name_gate["passed"]
        and reducer_gate["passed"]
        and fd_gate["passed"]
        and ad_fd_gate["passed"]
        and claim_scope_gate["passed"]
    )
    return {
        "kind": "vmec_boozer_reduced_portfolio_artifact_guard",
        "passed": passed,
        "claim_scope": (
            "artifact-level guard for real VMEC/Boozer reduced growth/QL portfolio rows; "
            "not a production nonlinear turbulent-transport optimization claim"
        ),
        "config": cfg.to_dict(),
        "provenance_gate": provenance_gate,
        "coverage_gate": coverage_gate,
        "full_objective_table_gate": full_table_gate,
        "objective_name_gate": objective_name_gate,
        "portfolio_reducer_gate": reducer_gate,
        "finite_difference_gate": fd_gate,
        "ad_fd_gradient_gate": ad_fd_gate,
        "claim_scope_gate": claim_scope_gate,
        "next_action": (
            "Use this guard to admit reduced VMEC/Boozer growth/QL portfolio artifacts. "
            "Production nonlinear claims still require held-out optimized-equilibrium "
            "nonlinear-window ensembles and transport audits."
        ),
    }


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

    ``weighted_mean`` normalizes both sample and objective weights to unit sum,
    making the scalar invariant to the caller's absolute weight scale.  ``mean``
    is the unweighted mean over every table entry.  ``max`` returns the
    worst-case objective-weighted sample and is intended for diagnostics rather
    than smooth gradient-based optimization.
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


__all__ = [
    "PortfolioReduction",
    "ReducedPortfolioArtifactGuardConfig",
    "StellaratorObjectivePortfolioContract",
    "aggregate_objective_portfolio",
    "objective_portfolio_sensitivity_report",
    "portfolio_objective_weight_vector",
    "portfolio_sample_weight_tensor",
    "reduced_portfolio_artifact_guard_report",
    "validate_objective_portfolio_contract",
]
