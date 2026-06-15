"""Finite-difference and line-search gates for VMEC/Boozer objectives."""

from __future__ import annotations

import importlib
from typing import Any, Literal, cast

import numpy as np

from spectraxgk.geometry.differentiable import discover_differentiable_geometry_backends
from spectraxgk.solver_objective_core import (
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    solver_scalar_objective_from_vector,
)
from spectraxgk.solver_objective_sampling import (
    _aggregate_weights,
    _float_tuple,
    _int_tuple,
    _surface_sample_axis,
    solver_grid_options_from_ky_values,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_solver_objective_table_with_metadata_from_state,
    vmec_boozer_solver_objective_vector_from_state,
)
from spectraxgk.solver_vmec_state import (
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)


def _report_float(report: dict[str, object], key: str) -> float:
    """Read a numeric finite-difference report field with mypy-safe casting."""

    return float(cast(Any, report[key]))


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

    load_state_bundle_fn = kwargs.pop(
        "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
    )
    state_array_fn = kwargs.pop("_state_array_fn", _vmec_boozer_state_array)
    replace_state_coefficient_fn = kwargs.pop(
        "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
    )
    parameter_name_fn = kwargs.pop(
        "_parameter_name_fn", _vmec_boozer_state_parameter_name
    )
    vector_fn = kwargs.pop("_vector_fn", vmec_boozer_solver_objective_vector_from_state)
    scalar_selector_fn = kwargs.pop(
        "_scalar_selector_fn", solver_scalar_objective_from_vector
    )

    step = float(perturbation_step)
    if step <= 0.0:
        raise ValueError("perturbation_step must be positive")
    curvature_ratio_limit = float(max_curvature_ratio)
    if curvature_ratio_limit < 0.0:
        raise ValueError("max_curvature_ratio must be non-negative")
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
    parameter_name = parameter_name_fn(
        parameter_family,
        radial_index_int,
        mode_index_int,
        default_mid_surface=default_radial_index,
    )

    base_delta_float = float(base_delta)

    def evaluate(delta: float) -> tuple[float, list[float]]:
        traced_state = replace_state_coefficient_fn(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            base_delta_float + float(delta),
        )
        vector = vector_fn(
            traced_state,
            bundle["static"],
            bundle["indata"],
            bundle["wout"],
            **kwargs,
        )
        scalar = scalar_selector_fn(vector, objective)
        vector_np = np.asarray(vector, dtype=float)
        return float(np.asarray(scalar)), vector_np.tolist()

    minus_value, minus_vector = evaluate(-step)
    base_value, base_vector = evaluate(0.0)
    plus_value, plus_vector = evaluate(step)
    central_derivative = (plus_value - minus_value) / (2.0 * step)
    response_abs = abs(plus_value - minus_value)
    curvature_abs = abs(plus_value - 2.0 * base_value + minus_value)
    curvature_scale = max(abs(response_abs), float(response_atol), 1.0e-300)
    curvature_ratio = curvature_abs / curvature_scale
    finite = bool(
        np.all(
            np.isfinite(
                np.asarray(
                    [
                        minus_value,
                        base_value,
                        plus_value,
                        central_derivative,
                        *minus_vector,
                        *base_vector,
                        *plus_vector,
                    ],
                    dtype=float,
                )
            )
        )
    )
    response_resolved = bool(response_abs >= float(response_atol))
    finite_difference_consistent = bool(curvature_ratio <= curvature_ratio_limit)
    return {
        "kind": "vmec_boozer_scalar_objective_finite_difference_report",
        "passed": bool(finite and response_resolved and finite_difference_consistent),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "finite-difference sensitivity of one scalar objective through "
            "VMECState -> booz_xform_jax -> SPECTRAX-GK value evaluator; not an AD or nonlinear transport claim"
        ),
        "case_name": str(case_name),
        "input_path": bundle["input_path"],
        "wout_path": bundle["wout_path"],
        "objective": str(objective),
        "parameter_name": parameter_name,
        "parameter_indices": {
            str(parameter_family): [radial_index_int, mode_index_int]
        },
        "base_delta": base_delta_float,
        "perturbation_step": step,
        "response_atol": float(response_atol),
        "max_curvature_ratio": curvature_ratio_limit,
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
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "minus_objective_vector": minus_vector,
        "base_objective_vector": base_vector,
        "plus_objective_vector": plus_vector,
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "next_action": (
            "Use this finite-difference path to seed real VMEC/Boozer optimizer "
            "drivers, then promote growth objectives with implicit AD/FD gates and "
            "quasilinear objectives with branch-continuity plus finite-difference/SPSA audits."
        ),
    }


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

    load_state_bundle_fn = kwargs.pop(
        "_load_state_bundle_fn", _load_vmec_jax_example_state_bundle
    )
    state_array_fn = kwargs.pop("_state_array_fn", _vmec_boozer_state_array)
    replace_state_coefficient_fn = kwargs.pop(
        "_replace_state_coefficient_fn", _replace_vmec_boozer_state_coefficient
    )
    parameter_name_fn = kwargs.pop(
        "_parameter_name_fn", _vmec_boozer_state_parameter_name
    )
    table_with_metadata_fn = kwargs.pop(
        "_table_with_metadata_fn",
        vmec_boozer_solver_objective_table_with_metadata_from_state,
    )
    scalar_selector_fn = kwargs.pop(
        "_scalar_selector_fn", solver_scalar_objective_from_vector
    )

    step = float(perturbation_step)
    if step <= 0.0:
        raise ValueError("perturbation_step must be positive")
    curvature_ratio_limit = float(max_curvature_ratio)
    if curvature_ratio_limit < 0.0:
        raise ValueError("max_curvature_ratio must be non-negative")
    surface_samples = _surface_sample_axis(surface_indices, torflux_values)
    alpha_values = _float_tuple(alphas, name="alphas")
    if ky_values is None:
        ky_indices = _int_tuple(selected_ky_indices, name="selected_ky_indices")
    else:
        ky_grid_options = solver_grid_options_from_ky_values(
            ky_values,
            ky_base=ky_base,
            min_ny=int(kwargs.get("ny", 4)),
        )
        selected_grid = cast(tuple[int, ...], ky_grid_options["selected_ky_indices"])
        ky_indices = tuple(int(item) for item in selected_grid)
    n_samples = len(surface_samples) * len(alpha_values) * len(ky_indices)
    normalized_weights = _aggregate_weights(weights, n_samples)
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
    parameter_name = parameter_name_fn(
        parameter_family,
        radial_index_int,
        mode_index_int,
        default_mid_surface=default_radial_index,
    )

    base_delta_float = float(base_delta)

    def evaluate(
        delta: float,
    ) -> tuple[float, list[float], list[list[float]], list[dict[str, object]]]:
        traced_state = replace_state_coefficient_fn(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            base_delta_float + float(delta),
        )
        table, sample_metadata = table_with_metadata_fn(
            traced_state,
            bundle["static"],
            bundle["indata"],
            bundle["wout"],
            surface_indices=surface_indices,
            torflux_values=torflux_values,
            alphas=alpha_values,
            selected_ky_indices=selected_ky_indices,
            ky_values=ky_values,
            ky_base=ky_base,
            **kwargs,
        )
        scalar_values = np.asarray(
            [scalar_selector_fn(row, objective) for row in table],
            dtype=float,
        )
        if str(reduction) == "mean":
            scalar = float(np.mean(scalar_values))
        elif str(reduction) == "weighted_mean":
            scalar = float(np.sum(scalar_values * normalized_weights))
        elif str(reduction) == "max":
            scalar = float(np.max(scalar_values))
        else:
            raise ValueError(
                "reduction must be one of 'mean', 'weighted_mean', or 'max'"
            )
        return (
            scalar,
            scalar_values.tolist(),
            np.asarray(table, dtype=float).tolist(),
            sample_metadata,
        )

    minus_value, minus_sample_values, minus_table, _minus_samples = evaluate(-step)
    base_value, base_sample_values, base_table, base_samples = evaluate(0.0)
    plus_value, plus_sample_values, plus_table, _plus_samples = evaluate(step)
    if len(base_samples) != int(n_samples):
        raise RuntimeError(
            "VMEC/Boozer aggregate metadata size does not match objective table"
        )
    samples = [
        dict(row, weight=float(normalized_weights[index]))
        for index, row in enumerate(base_samples)
    ]
    central_derivative = (plus_value - minus_value) / (2.0 * step)
    response_abs = abs(plus_value - minus_value)
    curvature_abs = abs(plus_value - 2.0 * base_value + minus_value)
    curvature_scale = max(abs(response_abs), float(response_atol), 1.0e-300)
    curvature_ratio = curvature_abs / curvature_scale
    finite = bool(
        np.all(
            np.isfinite(
                np.asarray(
                    [
                        minus_value,
                        base_value,
                        plus_value,
                        central_derivative,
                        *minus_sample_values,
                        *base_sample_values,
                        *plus_sample_values,
                    ],
                    dtype=float,
                )
            )
        )
    )
    response_resolved = bool(response_abs >= float(response_atol))
    finite_difference_consistent = bool(curvature_ratio <= curvature_ratio_limit)
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": bool(finite and response_resolved and finite_difference_consistent),
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "finite-difference sensitivity of an aggregated linear/quasilinear "
            "VMEC/Boozer/SPECTRAX-GK objective over fixed surfaces, field lines, and ky points; "
            "not a nonlinear transport optimization claim"
        ),
        "case_name": str(case_name),
        "input_path": bundle["input_path"],
        "wout_path": bundle["wout_path"],
        "objective": str(objective),
        "reduction": str(reduction),
        "samples": samples,
        "n_samples": n_samples,
        "surface_indices": [
            None
            if row.get("surface_index") is None
            else int(cast(int, row["surface_index"]))
            for row in surface_samples
        ],
        "torflux_values": None
        if torflux_values is None
        else list(_float_tuple(torflux_values, name="torflux_values")),
        "alphas": list(alpha_values),
        "selected_ky_indices": list(ky_indices),
        "ky_values": None
        if ky_values is None
        else list(_float_tuple(ky_values, name="ky_values")),
        "parameter_name": parameter_name,
        "parameter_indices": {
            str(parameter_family): [radial_index_int, mode_index_int]
        },
        "base_delta": base_delta_float,
        "perturbation_step": step,
        "response_atol": float(response_atol),
        "max_curvature_ratio": curvature_ratio_limit,
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
        "minus_sample_values": minus_sample_values,
        "base_sample_values": base_sample_values,
        "plus_sample_values": plus_sample_values,
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "minus_objective_table": minus_table,
        "base_objective_table": base_table,
        "plus_objective_table": plus_table,
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "next_action": (
            "Use this gate before any multi-surface or multi-ky optimizer loop. "
            "Promote only after branch-continuity and held-out nonlinear-window evidence pass."
        ),
    }


def vmec_boozer_aggregate_scalar_objective_line_search_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (None,),
    alphas: float | tuple[float, ...] | list[float] = (0.0,),
    selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Run a curvature-gated line search for an aggregate VMEC objective.

    This is the first optimizer-control gate for multi-surface, field-line, or
    ``k_y`` reduced objectives.  It keeps the update one-dimensional so each
    step can be audited against the finite-difference curvature gate and
    rejected when the aggregate objective does not decrease.
    """

    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    )

    max_steps_int = int(max_steps)
    if max_steps_int < 1:
        raise ValueError("max_steps must be >= 1")
    update_step_float = float(update_step)
    if update_step_float <= 0.0:
        raise ValueError("update_step must be positive")
    min_improvement_float = float(min_improvement)
    if min_improvement_float < 0.0:
        raise ValueError("min_improvement must be non-negative")

    delta = float(initial_delta)
    history: list[dict[str, object]] = []
    best_value: float | None = None
    accepted_steps = 0
    stop_reason = "max_steps"
    sample_metadata: list[dict[str, object]] = []
    n_samples = 0
    for step_index in range(max_steps_int):
        report = finite_difference_report_fn(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            weights=weights,
            surface_indices=surface_indices,
            alphas=alphas,
            selected_ky_indices=selected_ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            parameter_family=parameter_family,
            base_delta=delta,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )
        base_value = _report_float(report, "base_value")
        if best_value is None:
            best_value = base_value
        if not sample_metadata and isinstance(report.get("samples"), list):
            sample_metadata = cast(list[dict[str, object]], report["samples"])
        n_samples = int(cast(Any, report.get("n_samples", n_samples)))
        derivative = _report_float(report, "central_derivative")
        row: dict[str, object] = {
            "step": step_index,
            "delta": delta,
            "objective": base_value,
            "central_derivative": derivative,
            "finite_difference_passed": bool(report["passed"]),
            "curvature_ratio": _report_float(report, "curvature_ratio"),
            "accepted": False,
            "candidate_delta": None,
            "candidate_objective": None,
        }
        if not bool(report["passed"]):
            stop_reason = "finite_difference_gate_failed"
            history.append(row)
            break
        if not np.isfinite(derivative) or abs(derivative) == 0.0:
            stop_reason = "zero_or_nonfinite_derivative"
            history.append(row)
            break
        direction = -float(np.sign(derivative))
        candidate_delta = delta + direction * update_step_float
        candidate = finite_difference_report_fn(
            case_name=case_name,
            objective=objective,
            reduction=reduction,
            weights=weights,
            surface_indices=surface_indices,
            alphas=alphas,
            selected_ky_indices=selected_ky_indices,
            radial_index=radial_index,
            mode_index=mode_index,
            parameter_family=parameter_family,
            base_delta=candidate_delta,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )
        candidate_value = _report_float(candidate, "base_value")
        row["candidate_delta"] = candidate_delta
        row["candidate_objective"] = candidate_value
        candidate_ok = bool(candidate["passed"]) and (
            candidate_value < base_value - min_improvement_float
        )
        if not candidate_ok:
            stop_reason = "no_accepted_candidate"
            history.append(row)
            break
        delta = candidate_delta
        best_value = candidate_value
        accepted_steps += 1
        row["accepted"] = True
        history.append(row)

    initial_objective = (
        float(cast(Any, history[0]["objective"])) if history else float("nan")
    )
    final_objective = float(best_value) if best_value is not None else initial_objective
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": bool(accepted_steps > 0 and final_objective < initial_objective),
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "curvature-gated one-parameter line search for an aggregated "
            "VMEC/Boozer/SPECTRAX-GK linear/quasilinear objective; not a "
            "multi-parameter or nonlinear turbulent transport optimization claim"
        ),
        "case_name": str(case_name),
        "objective": str(objective),
        "reduction": str(reduction),
        "samples": sample_metadata,
        "n_samples": n_samples,
        "radial_index": None if radial_index is None else int(radial_index),
        "mode_index": int(mode_index),
        "initial_delta": float(initial_delta),
        "final_delta": delta,
        "perturbation_step": float(perturbation_step),
        "update_step": update_step_float,
        "max_steps": max_steps_int,
        "accepted_steps": accepted_steps,
        "stop_reason": stop_reason,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "relative_reduction": (
            float((initial_objective - final_objective) / abs(initial_objective))
            if np.isfinite(initial_objective) and abs(initial_objective) > 0.0
            else None
        ),
        "history": history,
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }


def vmec_boozer_aggregate_line_search_holdout_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    reduction: Literal["mean", "weighted_mean", "max"] = "mean",
    training_weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    holdout_weights: tuple[float, ...] | list[float] | np.ndarray | None = None,
    training_surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (
        None,
    ),
    training_alphas: float | tuple[float, ...] | list[float] = (0.0,),
    training_selected_ky_indices: int | tuple[int, ...] | list[int] = (1,),
    holdout_surface_indices: int | None | tuple[int | None, ...] | list[int | None] = (
        None,
    ),
    holdout_alphas: float | tuple[float, ...] | list[float] = (0.0,),
    holdout_selected_ky_indices: int | tuple[int, ...] | list[int] = (2,),
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    min_holdout_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Audit a training aggregate update against held-out aggregate samples.

    A report passes only when the training line search accepts at least one
    curvature-gated update and the same final VMEC coefficient offset reduces
    the held-out aggregate objective.  This is a reduced linear/quasilinear
    validation split, not a nonlinear transport optimization claim.
    """

    line_search_report_fn = kwargs.pop(
        "_line_search_report_fn",
        vmec_boozer_aggregate_scalar_objective_line_search_report,
    )
    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    )

    min_holdout_improvement_float = float(min_holdout_improvement)
    if min_holdout_improvement_float < 0.0:
        raise ValueError("min_holdout_improvement must be non-negative")

    training = line_search_report_fn(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        weights=training_weights,
        surface_indices=training_surface_indices,
        alphas=training_alphas,
        selected_ky_indices=training_selected_ky_indices,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        **kwargs,
    )
    final_delta = _report_float(training, "final_delta")

    heldout_initial = finite_difference_report_fn(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        weights=holdout_weights,
        surface_indices=holdout_surface_indices,
        alphas=holdout_alphas,
        selected_ky_indices=holdout_selected_ky_indices,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        base_delta=initial_delta,
        perturbation_step=perturbation_step,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        **kwargs,
    )
    heldout_final = finite_difference_report_fn(
        case_name=case_name,
        objective=objective,
        reduction=reduction,
        weights=holdout_weights,
        surface_indices=holdout_surface_indices,
        alphas=holdout_alphas,
        selected_ky_indices=holdout_selected_ky_indices,
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        base_delta=final_delta,
        perturbation_step=perturbation_step,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
        **kwargs,
    )
    heldout_initial_value = _report_float(heldout_initial, "base_value")
    heldout_final_value = _report_float(heldout_final, "base_value")
    heldout_reduction = heldout_initial_value - heldout_final_value
    heldout_passed = bool(
        bool(heldout_initial["passed"])
        and bool(heldout_final["passed"])
        and heldout_reduction > min_holdout_improvement_float
    )
    training_passed = bool(training["passed"])
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": bool(training_passed and heldout_passed),
        "source_scope": "mode21_vmec_boozer_state_train_holdout",
        "claim_scope": (
            "one-parameter aggregate reduced-objective line search with held-out "
            "surface/field-line/ky validation; not a nonlinear turbulent transport "
            "or broad stellarator optimization claim"
        ),
        "case_name": str(case_name),
        "objective": str(objective),
        "reduction": str(reduction),
        "initial_delta": float(initial_delta),
        "final_delta": final_delta,
        "training_passed": training_passed,
        "heldout_passed": heldout_passed,
        "training_initial_objective": _report_float(training, "initial_objective"),
        "training_final_objective": _report_float(training, "final_objective"),
        "training_relative_reduction": training.get("relative_reduction"),
        "heldout_initial_objective": heldout_initial_value,
        "heldout_final_objective": heldout_final_value,
        "heldout_relative_reduction": (
            float(heldout_reduction / abs(heldout_initial_value))
            if np.isfinite(heldout_initial_value) and abs(heldout_initial_value) > 0.0
            else None
        ),
        "min_holdout_improvement": min_holdout_improvement_float,
        "training_samples": training.get("samples", []),
        "heldout_samples": heldout_initial.get("samples", []),
        "training_report": training,
        "heldout_initial_report": heldout_initial,
        "heldout_final_report": heldout_final,
        "next_action": (
            "Promote only if this split gate passes on multiple held-out surfaces "
            "or field lines and then survives nonlinear-window transport audits."
        ),
    }


def vmec_boozer_scalar_objective_line_search_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    objective: SolverScalarObjective = "growth",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    initial_delta: float = 0.0,
    perturbation_step: float = 1.0e-7,
    update_step: float = 1.0e-8,
    max_steps: int = 3,
    min_improvement: float = 0.0,
    response_atol: float = 0.0,
    max_curvature_ratio: float = 5.0,
    **kwargs: Any,
) -> dict[str, object]:
    """Run a curvature-gated one-parameter VMEC/Boozer objective line search.

    This is the first safe optimizer scaffold for the real in-memory
    VMEC/Boozer/SPECTRAX-GK path. Each accepted update must pass the scalar
    finite-difference curvature gate, and candidate steps are accepted only
    when the scalar objective decreases. It is still a one-coefficient audit,
    not a broad stellarator-optimization claim.
    """

    finite_difference_report_fn = kwargs.pop(
        "_finite_difference_report_fn",
        vmec_boozer_scalar_objective_finite_difference_report,
    )

    max_steps_int = int(max_steps)
    if max_steps_int < 1:
        raise ValueError("max_steps must be >= 1")
    update_step_float = float(update_step)
    if update_step_float <= 0.0:
        raise ValueError("update_step must be positive")
    min_improvement_float = float(min_improvement)
    if min_improvement_float < 0.0:
        raise ValueError("min_improvement must be non-negative")

    delta = float(initial_delta)
    history: list[dict[str, object]] = []
    best_value: float | None = None
    accepted_steps = 0
    stop_reason = "max_steps"
    for step_index in range(max_steps_int):
        report = finite_difference_report_fn(
            case_name=case_name,
            objective=objective,
            radial_index=radial_index,
            mode_index=mode_index,
            parameter_family=parameter_family,
            base_delta=delta,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )
        base_value = _report_float(report, "base_value")
        if best_value is None:
            best_value = base_value
        derivative = _report_float(report, "central_derivative")
        row: dict[str, object] = {
            "step": step_index,
            "delta": delta,
            "objective": base_value,
            "central_derivative": derivative,
            "finite_difference_passed": bool(report["passed"]),
            "curvature_ratio": _report_float(report, "curvature_ratio"),
            "accepted": False,
            "candidate_delta": None,
            "candidate_objective": None,
        }
        if not bool(report["passed"]):
            stop_reason = "finite_difference_gate_failed"
            history.append(row)
            break
        if not np.isfinite(derivative) or abs(derivative) == 0.0:
            stop_reason = "zero_or_nonfinite_derivative"
            history.append(row)
            break
        direction = -float(np.sign(derivative))
        candidate_delta = delta + direction * update_step_float
        candidate = finite_difference_report_fn(
            case_name=case_name,
            objective=objective,
            radial_index=radial_index,
            mode_index=mode_index,
            parameter_family=parameter_family,
            base_delta=candidate_delta,
            perturbation_step=perturbation_step,
            response_atol=response_atol,
            max_curvature_ratio=max_curvature_ratio,
            **kwargs,
        )
        candidate_value = _report_float(candidate, "base_value")
        row["candidate_delta"] = candidate_delta
        row["candidate_objective"] = candidate_value
        candidate_ok = bool(candidate["passed"]) and (
            candidate_value < base_value - min_improvement_float
        )
        if not candidate_ok:
            stop_reason = "no_accepted_candidate"
            history.append(row)
            break
        delta = candidate_delta
        best_value = candidate_value
        accepted_steps += 1
        row["accepted"] = True
        history.append(row)

    initial_objective = (
        float(cast(Any, history[0]["objective"])) if history else float("nan")
    )
    final_objective = float(best_value) if best_value is not None else initial_objective
    return {
        "kind": "vmec_boozer_scalar_objective_line_search_report",
        "passed": bool(accepted_steps > 0 and final_objective < initial_objective),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "curvature-gated one-parameter VMEC/Boozer/SPECTRAX-GK scalar objective "
            "line search; not a multi-parameter stellarator optimization or nonlinear transport claim"
        ),
        "case_name": str(case_name),
        "objective": str(objective),
        "radial_index": None if radial_index is None else int(radial_index),
        "mode_index": int(mode_index),
        "initial_delta": float(initial_delta),
        "final_delta": delta,
        "perturbation_step": float(perturbation_step),
        "update_step": update_step_float,
        "max_steps": max_steps_int,
        "accepted_steps": accepted_steps,
        "stop_reason": stop_reason,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "relative_reduction": (
            float((initial_objective - final_objective) / abs(initial_objective))
            if np.isfinite(initial_objective) and abs(initial_objective) > 0.0
            else None
        ),
        "history": history,
        "options": {
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
    }


__all__ = [
    "_load_vmec_jax_example_state_bundle",
    "_report_float",
    "vmec_boozer_aggregate_line_search_holdout_report",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_aggregate_scalar_objective_line_search_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_line_search_report",
]
