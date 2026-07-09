"""Finite-difference gates for VMEC/Boozer objectives."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Literal, cast

import numpy as np

from spectraxgk.geometry.backend_discovery import discover_differentiable_geometry_backends
from spectraxgk.objectives.core import (
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    solver_scalar_objective_from_vector,
)
from spectraxgk.objectives.sampling import (
    _aggregate_weights,
    _float_tuple,
    _int_tuple,
    _surface_sample_axis,
    solver_grid_options_from_ky_values,
)
from spectraxgk.objectives.vmec_boozer import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
    vmec_boozer_solver_objective_vector_from_state,
)
from spectraxgk.geometry.vmec_state_controls import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)


@dataclass(frozen=True)
class _FiniteDifferenceSettings:
    step: float
    response_atol: float
    max_curvature_ratio: float


@dataclass(frozen=True)
class _StateParameterContext:
    bundle: dict[str, Any]
    state: Any
    base_coeff: Any
    parameter_family: str
    radial_index: int
    mode_index: int
    parameter_name: str
    base_delta: float


@dataclass(frozen=True)
class _VmecBoozerDependencyFns:
    load_state_bundle_fn: Any
    state_array_fn: Any
    replace_state_coefficient_fn: Any
    parameter_name_fn: Any
    table_with_metadata_fn: Any
    scalar_selector_fn: Any


@dataclass(frozen=True)
class _VmecBoozerScalarDependencyFns:
    load_state_bundle_fn: Any
    state_array_fn: Any
    replace_state_coefficient_fn: Any
    parameter_name_fn: Any
    vector_fn: Any
    scalar_selector_fn: Any


@dataclass(frozen=True)
class _ScalarEvaluation:
    value: float
    vector: list[float]


@dataclass(frozen=True)
class _ScalarFiniteDifferenceTriplet:
    minus: _ScalarEvaluation
    base: _ScalarEvaluation
    plus: _ScalarEvaluation


@dataclass(frozen=True)
class _AggregateSamplePlan:
    surface_samples: tuple[dict[str, float | int | None], ...]
    alpha_values: tuple[float, ...]
    selected_ky_indices: tuple[int, ...]
    normalized_weights: np.ndarray
    n_samples: int


@dataclass(frozen=True)
class _AggregateEvaluation:
    value: float
    sample_values: list[float]
    objective_table: list[list[float]]
    sample_metadata: list[dict[str, object]]


@dataclass(frozen=True)
class _AggregateFiniteDifferenceTriplet:
    minus: _AggregateEvaluation
    base: _AggregateEvaluation
    plus: _AggregateEvaluation


def _report_float(report: dict[str, object], key: str) -> float:
    """Read a numeric finite-difference report field with mypy-safe casting."""

    return float(cast(Any, report[key]))


def _finite_difference_settings(
    perturbation_step: float,
    response_atol: float,
    max_curvature_ratio: float,
) -> _FiniteDifferenceSettings:
    step = float(perturbation_step)
    if step <= 0.0:
        raise ValueError("perturbation_step must be positive")
    curvature_ratio_limit = float(max_curvature_ratio)
    if curvature_ratio_limit < 0.0:
        raise ValueError("max_curvature_ratio must be non-negative")
    return _FiniteDifferenceSettings(
        step=step,
        response_atol=float(response_atol),
        max_curvature_ratio=curvature_ratio_limit,
    )


def _state_parameter_context(
    *,
    case_name: str,
    parameter_family: str,
    radial_index: int | None,
    mode_index: int,
    base_delta: float,
    load_state_bundle_fn: Any,
    state_array_fn: Any,
    parameter_name_fn: Any,
) -> _StateParameterContext:
    bundle = load_state_bundle_fn(str(case_name))
    state = bundle["state"]
    base_coeff = state_array_fn(state, parameter_family)
    default_radial_index = int(base_coeff.shape[0] // 2)
    radial_index_int = (
        default_radial_index if radial_index is None else int(radial_index)
    )
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_coeff.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_coeff.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    return _StateParameterContext(
        bundle=bundle,
        state=state,
        base_coeff=base_coeff,
        parameter_family=str(parameter_family),
        radial_index=radial_index_int,
        mode_index=mode_index_int,
        parameter_name=parameter_name_fn(
            parameter_family,
            radial_index_int,
            mode_index_int,
            default_mid_surface=default_radial_index,
        ),
        base_delta=float(base_delta),
    )


def _perturbed_state(
    ctx: _StateParameterContext, replace_state_coefficient_fn: Any, delta: float
) -> Any:
    return replace_state_coefficient_fn(
        ctx.state,
        ctx.parameter_family,
        ctx.base_coeff,
        ctx.radial_index,
        ctx.mode_index,
        ctx.base_delta + float(delta),
    )


def _fd_diagnostics(
    *,
    minus_value: float,
    base_value: float,
    plus_value: float,
    settings: _FiniteDifferenceSettings,
    extra_values: list[float],
) -> dict[str, object]:
    central_derivative = (plus_value - minus_value) / (2.0 * settings.step)
    response_abs = abs(plus_value - minus_value)
    curvature_abs = abs(plus_value - 2.0 * base_value + minus_value)
    curvature_scale = max(abs(response_abs), settings.response_atol, 1.0e-300)
    curvature_ratio = curvature_abs / curvature_scale
    values = np.asarray(
        [minus_value, base_value, plus_value, central_derivative, *extra_values],
        dtype=float,
    )
    finite = bool(np.all(np.isfinite(values)))
    response_resolved = bool(response_abs >= settings.response_atol)
    finite_difference_consistent = bool(
        curvature_ratio <= settings.max_curvature_ratio
    )
    return {
        "passed": bool(finite and response_resolved and finite_difference_consistent),
        "perturbation_step": settings.step,
        "response_atol": settings.response_atol,
        "max_curvature_ratio": settings.max_curvature_ratio,
        "response_abs": response_abs,
        "curvature_abs": curvature_abs,
        "curvature_ratio": curvature_ratio,
        "finite_values": finite,
        "response_resolved": response_resolved,
        "finite_difference_consistent": finite_difference_consistent,
        "minus_value": minus_value,
        "base_value": base_value,
        "plus_value": plus_value,
        "central_derivative": float(central_derivative),
    }


def _base_report_fields(
    *,
    case_name: str,
    objective: SolverScalarObjective,
    ctx: _StateParameterContext,
    options: dict[str, Any],
) -> dict[str, object]:
    return {
        "case_name": str(case_name),
        "input_path": ctx.bundle["input_path"],
        "wout_path": ctx.bundle["wout_path"],
        "objective": str(objective),
        "parameter_name": ctx.parameter_name,
        "parameter_indices": {
            ctx.parameter_family: [ctx.radial_index, ctx.mode_index]
        },
        "base_delta": ctx.base_delta,
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "options": options,
    }


def _public_options(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _reduce_scalar_values(
    scalar_values: np.ndarray,
    *,
    reduction: Literal["mean", "weighted_mean", "max"],
    weights: np.ndarray,
) -> float:
    if str(reduction) == "mean":
        return float(np.mean(scalar_values))
    if str(reduction) == "weighted_mean":
        return float(np.sum(scalar_values * weights))
    if str(reduction) == "max":
        return float(np.max(scalar_values))
    raise ValueError("reduction must be one of 'mean', 'weighted_mean', or 'max'")


def _aggregate_dependency_fns(kwargs: dict[str, Any]) -> _VmecBoozerDependencyFns:
    return _VmecBoozerDependencyFns(
        load_state_bundle_fn=kwargs.pop(
            "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
        ),
        state_array_fn=kwargs.pop("_state_array_fn", _vmec_boozer_state_array),
        replace_state_coefficient_fn=kwargs.pop(
            "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
        ),
        parameter_name_fn=kwargs.pop(
            "_parameter_name_fn", _vmec_boozer_state_parameter_name
        ),
        table_with_metadata_fn=kwargs.pop(
            "_table_with_metadata_fn",
            vmec_boozer_solver_objective_table_with_metadata_from_state,
        ),
        scalar_selector_fn=kwargs.pop(
            "_scalar_selector_fn", solver_scalar_objective_from_vector
        ),
    )


def _scalar_dependency_fns(kwargs: dict[str, Any]) -> _VmecBoozerScalarDependencyFns:
    """Return injected functions used by scalar VMEC/Boozer FD reports."""

    return _VmecBoozerScalarDependencyFns(
        load_state_bundle_fn=kwargs.pop(
            "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
        ),
        state_array_fn=kwargs.pop("_state_array_fn", _vmec_boozer_state_array),
        replace_state_coefficient_fn=kwargs.pop(
            "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
        ),
        parameter_name_fn=kwargs.pop(
            "_parameter_name_fn", _vmec_boozer_state_parameter_name
        ),
        vector_fn=kwargs.pop(
            "_vector_fn", vmec_boozer_solver_objective_vector_from_state
        ),
        scalar_selector_fn=kwargs.pop(
            "_scalar_selector_fn", solver_scalar_objective_from_vector
        ),
    )


def _evaluate_scalar_fd_point(
    *,
    delta: float,
    ctx: _StateParameterContext,
    fns: _VmecBoozerScalarDependencyFns,
    objective: SolverScalarObjective,
    kwargs: dict[str, Any],
) -> _ScalarEvaluation:
    """Evaluate one scalar objective and full objective vector at a perturbation."""

    traced_state = _perturbed_state(ctx, fns.replace_state_coefficient_fn, delta)
    vector = fns.vector_fn(
        traced_state,
        ctx.bundle["static"],
        ctx.bundle["indata"],
        ctx.bundle["wout"],
        **kwargs,
    )
    scalar = fns.scalar_selector_fn(vector, objective)
    vector_np = np.asarray(vector, dtype=float)
    return _ScalarEvaluation(
        value=float(np.asarray(scalar)),
        vector=vector_np.tolist(),
    )


def _evaluate_scalar_fd_triplet(
    *,
    settings: _FiniteDifferenceSettings,
    ctx: _StateParameterContext,
    fns: _VmecBoozerScalarDependencyFns,
    objective: SolverScalarObjective,
    kwargs: dict[str, Any],
) -> _ScalarFiniteDifferenceTriplet:
    """Evaluate the minus/base/plus scalar FD triplet."""

    return _ScalarFiniteDifferenceTriplet(
        minus=_evaluate_scalar_fd_point(
            delta=-settings.step,
            ctx=ctx,
            fns=fns,
            objective=objective,
            kwargs=kwargs,
        ),
        base=_evaluate_scalar_fd_point(
            delta=0.0,
            ctx=ctx,
            fns=fns,
            objective=objective,
            kwargs=kwargs,
        ),
        plus=_evaluate_scalar_fd_point(
            delta=settings.step,
            ctx=ctx,
            fns=fns,
            objective=objective,
            kwargs=kwargs,
        ),
    )


def _scalar_fd_diagnostics(
    triplet: _ScalarFiniteDifferenceTriplet,
    *,
    settings: _FiniteDifferenceSettings,
) -> dict[str, object]:
    """Return finite-difference diagnostics for one scalar triplet."""

    return _fd_diagnostics(
        minus_value=triplet.minus.value,
        base_value=triplet.base.value,
        plus_value=triplet.plus.value,
        settings=settings,
        extra_values=[
            *triplet.minus.vector,
            *triplet.base.vector,
            *triplet.plus.vector,
        ],
    )


def _scalar_fd_report_payload(
    *,
    case_name: str,
    objective: SolverScalarObjective,
    ctx: _StateParameterContext,
    options: dict[str, Any],
    diagnostics: dict[str, object],
    triplet: _ScalarFiniteDifferenceTriplet,
) -> dict[str, object]:
    """Return the public scalar finite-difference report payload."""

    return {
        "kind": "vmec_boozer_scalar_objective_finite_difference_report",
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "finite-difference sensitivity of one scalar objective through "
            "VMECState -> booz_xform_jax -> SPECTRAX-GK value evaluator; not an AD or nonlinear transport claim"
        ),
        **_base_report_fields(
            case_name=case_name,
            objective=objective,
            ctx=ctx,
            options=options,
        ),
        **diagnostics,
        "minus_objective_vector": triplet.minus.vector,
        "base_objective_vector": triplet.base.vector,
        "plus_objective_vector": triplet.plus.vector,
        "next_action": (
            "Use this finite-difference path to seed real VMEC/Boozer optimizer "
            "drivers, then promote growth objectives with implicit AD/FD gates and "
            "quasilinear objectives with branch-continuity plus finite-difference/SPSA audits."
        ),
    }


def _aggregate_sample_plan(
    *,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None],
    torflux_values: float | tuple[float, ...] | list[float] | None,
    alphas: float | tuple[float, ...] | list[float],
    selected_ky_indices: int | tuple[int, ...] | list[int],
    ky_values: float | tuple[float, ...] | list[float] | None,
    ky_base: float | None,
    weights: tuple[float, ...] | list[float] | np.ndarray | None,
    min_ny: int,
) -> _AggregateSamplePlan:
    surface_samples = _surface_sample_axis(surface_indices, torflux_values)
    alpha_values = _float_tuple(alphas, name="alphas")
    if ky_values is None:
        ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    else:
        ky_grid_options = solver_grid_options_from_ky_values(
            ky_values,
            ky_base=ky_base,
            min_ny=int(min_ny),
        )
        selected_grid = cast(tuple[int, ...], ky_grid_options["selected_ky_indices"])
        ky_indices = tuple(int(item) for item in selected_grid)
    n_samples = len(surface_samples) * len(alpha_values) * len(ky_indices)
    return _AggregateSamplePlan(
        surface_samples=surface_samples,
        alpha_values=alpha_values,
        selected_ky_indices=ky_indices,
        normalized_weights=_aggregate_weights(weights, n_samples),
        n_samples=n_samples,
    )


def _evaluate_aggregate_fd_point(
    *,
    delta: float,
    ctx: _StateParameterContext,
    fns: _VmecBoozerDependencyFns,
    plan: _AggregateSamplePlan,
    objective: SolverScalarObjective,
    reduction: Literal["mean", "weighted_mean", "max"],
    surface_indices: int | None | tuple[int | None, ...] | list[int | None],
    torflux_values: float | tuple[float, ...] | list[float] | None,
    selected_ky_indices: int | tuple[int, ...] | list[int],
    ky_values: float | tuple[float, ...] | list[float] | None,
    ky_base: float | None,
    kwargs: dict[str, Any],
) -> _AggregateEvaluation:
    traced_state = _perturbed_state(ctx, fns.replace_state_coefficient_fn, delta)
    table, sample_metadata = fns.table_with_metadata_fn(
        traced_state,
        ctx.bundle["static"],
        ctx.bundle["indata"],
        ctx.bundle["wout"],
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=plan.alpha_values,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        **kwargs,
    )
    scalar_values = np.asarray(
        [fns.scalar_selector_fn(row, objective) for row in table],
        dtype=float,
    )
    return _AggregateEvaluation(
        value=_reduce_scalar_values(
            scalar_values,
            reduction=reduction,
            weights=plan.normalized_weights,
        ),
        sample_values=scalar_values.tolist(),
        objective_table=np.asarray(table, dtype=float).tolist(),
        sample_metadata=sample_metadata,
    )


def _evaluate_aggregate_fd_triplet(
    *,
    settings: _FiniteDifferenceSettings,
    ctx: _StateParameterContext,
    fns: _VmecBoozerDependencyFns,
    plan: _AggregateSamplePlan,
    objective: SolverScalarObjective,
    reduction: Literal["mean", "weighted_mean", "max"],
    surface_indices: int | None | tuple[int | None, ...] | list[int | None],
    torflux_values: float | tuple[float, ...] | list[float] | None,
    selected_ky_indices: int | tuple[int, ...] | list[int],
    ky_values: float | tuple[float, ...] | list[float] | None,
    ky_base: float | None,
    kwargs: dict[str, Any],
) -> _AggregateFiniteDifferenceTriplet:
    def evaluate(delta: float) -> _AggregateEvaluation:
        return _evaluate_aggregate_fd_point(
            delta=delta,
            ctx=ctx,
            fns=fns,
            plan=plan,
            objective=objective,
            reduction=reduction,
            surface_indices=surface_indices,
            torflux_values=torflux_values,
            selected_ky_indices=selected_ky_indices,
            ky_values=ky_values,
            ky_base=ky_base,
            kwargs=kwargs,
        )

    return _AggregateFiniteDifferenceTriplet(
        minus=evaluate(-settings.step),
        base=evaluate(0.0),
        plus=evaluate(settings.step),
    )


def _aggregate_fd_diagnostics(
    triplet: _AggregateFiniteDifferenceTriplet,
    settings: _FiniteDifferenceSettings,
) -> dict[str, object]:
    return _fd_diagnostics(
        minus_value=triplet.minus.value,
        base_value=triplet.base.value,
        plus_value=triplet.plus.value,
        settings=settings,
        extra_values=[
            *triplet.minus.sample_values,
            *triplet.base.sample_values,
            *triplet.plus.sample_values,
        ],
    )


def _weighted_samples(
    sample_metadata: list[dict[str, object]],
    plan: _AggregateSamplePlan,
) -> list[dict[str, object]]:
    if len(sample_metadata) != int(plan.n_samples):
        raise RuntimeError(
            "VMEC/Boozer aggregate metadata size does not match objective table"
        )
    return [
        dict(row, weight=float(plan.normalized_weights[index]))
        for index, row in enumerate(sample_metadata)
    ]


def _aggregate_axis_report_fields(
    *,
    plan: _AggregateSamplePlan,
    surface_samples: tuple[dict[str, float | int | None], ...],
    torflux_values: float | tuple[float, ...] | list[float] | None,
    ky_values: float | tuple[float, ...] | list[float] | None,
) -> dict[str, object]:
    return {
        "surface_indices": [
            None
            if row.get("surface_index") is None
            else int(cast(int, row["surface_index"]))
            for row in surface_samples
        ],
        "torflux_values": None
        if torflux_values is None
        else list(_float_tuple(torflux_values, name="torflux_values")),
        "alphas": list(plan.alpha_values),
        "selected_ky_indices": list(plan.selected_ky_indices),
        "ky_values": None
        if ky_values is None
        else list(_float_tuple(ky_values, name="ky_values")),
    }


def _aggregate_fd_report_payload(
    *,
    case_name: str,
    objective: SolverScalarObjective,
    reduction: Literal["mean", "weighted_mean", "max"],
    ctx: _StateParameterContext,
    plan: _AggregateSamplePlan,
    diagnostics: dict[str, object],
    options: dict[str, Any],
    torflux_values: float | tuple[float, ...] | list[float] | None,
    ky_values: float | tuple[float, ...] | list[float] | None,
    minus_eval: _AggregateEvaluation,
    base_eval: _AggregateEvaluation,
    plus_eval: _AggregateEvaluation,
) -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "finite-difference sensitivity of an aggregated linear/quasilinear "
            "VMEC/Boozer/SPECTRAX-GK objective over fixed surfaces, field lines, and ky points; "
            "not a nonlinear transport optimization claim"
        ),
        **_base_report_fields(
            case_name=case_name,
            objective=objective,
            ctx=ctx,
            options=options,
        ),
        **diagnostics,
        "reduction": str(reduction),
        "samples": _weighted_samples(base_eval.sample_metadata, plan),
        "n_samples": plan.n_samples,
        **_aggregate_axis_report_fields(
            plan=plan,
            surface_samples=plan.surface_samples,
            torflux_values=torflux_values,
            ky_values=ky_values,
        ),
        "minus_sample_values": minus_eval.sample_values,
        "base_sample_values": base_eval.sample_values,
        "plus_sample_values": plus_eval.sample_values,
        "minus_objective_table": minus_eval.objective_table,
        "base_objective_table": base_eval.objective_table,
        "plus_objective_table": plus_eval.objective_table,
        "next_action": (
            "Use this gate before any multi-surface or multi-ky optimizer loop. "
            "Promote only after branch-continuity and held-out nonlinear-window evidence pass."
        ),
    }


def _load_vmec_jax_example_state_bundle(
    case_name: str,
) -> dict[str, Any]:  # pragma: no cover
    """Load a local ``vmec_jax`` example state bundle for offline gates."""

    discover_differentiable_geometry_backends()
    driver = importlib.import_module("vmec_jax.driver")
    config_mod = importlib.import_module("vmec_jax.config")
    static_mod = importlib.import_module("vmec_jax.static")
    wout_mod = importlib.import_module("vmec_jax.wout")

    input_path, wout_path = driver.example_paths(str(case_name))
    cfg_vmec, indata = config_mod.load_config(str(input_path))
    static = static_mod.build_static(cfg_vmec)
    wout = wout_mod.read_wout(wout_path)
    state = wout_mod.state_from_wout(wout)
    return {
        "case_name": str(case_name),
        "input_path": str(input_path),
        "wout_path": str(wout_path),
        "state": state,
        "static": static,
        "indata": indata,
        "wout": wout,
    }


def vmec_boozer_scalar_objective_finite_difference_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    base_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Finite-difference a scalar objective through a VMEC state coefficient.

    This report is the safe optimization pre-step for full-chain stellarator
    objectives. It perturbs one VMEC state coefficient in a solved
    ``vmec_jax`` state, evaluates the in-memory VMEC/Boozer/SPECTRAX-GK scalar
    objective at ``x0+base_delta-h``, ``x0+base_delta``, and
    ``x0+base_delta+h``, and records the central
    finite-difference sensitivity.

    It is intentionally not an AD claim. Growth-rate objectives can later be
    promoted with implicit eigenpair gates; quasilinear objectives involving
    eigenvectors need this finite-difference/SPSA path or a custom adjoint
    before they are used in production optimization loops.
    """

    fns = _scalar_dependency_fns(kwargs)
    settings = _finite_difference_settings(
        perturbation_step, response_atol, max_curvature_ratio
    )
    ctx = _state_parameter_context(
        case_name=case_name,
        parameter_family=parameter_family,
        radial_index=radial_index,
        mode_index=mode_index,
        base_delta=base_delta,
        load_state_bundle_fn=fns.load_state_bundle_fn,
        state_array_fn=fns.state_array_fn,
        parameter_name_fn=fns.parameter_name_fn,
    )
    triplet = _evaluate_scalar_fd_triplet(
        settings=settings,
        ctx=ctx,
        fns=fns,
        objective=objective,
        kwargs=kwargs,
    )
    return _scalar_fd_report_payload(
        case_name=case_name,
        objective=objective,
        ctx=ctx,
        options=_public_options(kwargs),
        diagnostics=_scalar_fd_diagnostics(triplet, settings=settings),
        triplet=triplet,
    )


def vmec_boozer_aggregate_scalar_objective_finite_difference_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    torflux_values: float | tuple[float, ...] | list[float] | None = None,
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    ky_values: float | tuple[float, ...] | list[float] | None = None,
    ky_base: float | None = None,
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    base_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Finite-difference a multi-surface/multi-``k_y`` aggregate objective."""

    fns = _aggregate_dependency_fns(kwargs)
    settings = _finite_difference_settings(
        perturbation_step, response_atol, max_curvature_ratio
    )
    sample_plan = _aggregate_sample_plan(
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        weights=weights,
        min_ny=int(kwargs.get("ny", 4)),
    )
    ctx = _state_parameter_context(
        case_name=case_name,
        parameter_family=parameter_family,
        radial_index=radial_index,
        mode_index=mode_index,
        base_delta=base_delta,
        load_state_bundle_fn=fns.load_state_bundle_fn,
        state_array_fn=fns.state_array_fn,
        parameter_name_fn=fns.parameter_name_fn,
    )
    triplet = _evaluate_aggregate_fd_triplet(
        settings=settings,
        ctx=ctx,
        fns=fns,
        plan=sample_plan,
        objective=objective,
        reduction=reduction,
        surface_indices=surface_indices,
        torflux_values=torflux_values,
        selected_ky_indices=selected_ky_indices,
        ky_values=ky_values,
        ky_base=ky_base,
        kwargs=kwargs,
    )
    diagnostics = _aggregate_fd_diagnostics(triplet, settings)
    return _aggregate_fd_report_payload(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        ctx=ctx,
        plan=sample_plan,
        diagnostics=diagnostics,
        options=_public_options(kwargs),
        torflux_values=torflux_values,
        ky_values=ky_values,
        minus_eval=triplet.minus,
        base_eval=triplet.base,
        plus_eval=triplet.plus,
    )


__all__ = [
    "_load_vmec_jax_example_state_bundle",
    "_report_float",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
]
