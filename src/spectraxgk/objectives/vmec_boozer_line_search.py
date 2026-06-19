"""Line-search and holdout gates for VMEC/Boozer objectives."""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np

from spectraxgk.objectives.core import SolverScalarObjective
from spectraxgk.objectives.vmec_boozer_fd import (
    _report_float,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    vmec_boozer_scalar_objective_finite_difference_report,
)


def _validate_line_search_controls(
    *,
    max_steps: int,
    update_step: float,
    min_improvement: float,
) -> tuple[int, float, float]:
    max_steps_int = int(max_steps)
    if max_steps_int < 1:
        raise ValueError("max_steps must be >= 1")
    update_step_float = float(update_step)
    if update_step_float <= 0.0:
        raise ValueError("update_step must be positive")
    min_improvement_float = float(min_improvement)
    if min_improvement_float < 0.0:
        raise ValueError("min_improvement must be non-negative")
    return max_steps_int, update_step_float, min_improvement_float


def _serializable_scalar_options(kwargs: dict[str, Any]) -> dict[str, object]:
    return {
        key: value
        for key, value in kwargs.items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }


def _relative_reduction(initial_objective: float, final_objective: float) -> float | None:
    if not np.isfinite(initial_objective) or abs(initial_objective) == 0.0:
        return None
    return float((initial_objective - final_objective) / abs(initial_objective))


def _run_one_parameter_line_search(
    *,
    finite_difference_report_fn: Any,
    probe_kwargs: dict[str, Any],
    initial_delta: float,
    perturbation_step: float,
    update_step: float,
    max_steps: int,
    min_improvement: float,
    response_atol: float,
    max_curvature_ratio: float,
) -> dict[str, object]:
    max_steps_int, update_step_float, min_improvement_float = _validate_line_search_controls(
        max_steps=max_steps,
        update_step=update_step,
        min_improvement=min_improvement,
    )
    base_probe_kwargs = {
        **probe_kwargs,
        "perturbation_step": perturbation_step,
        "response_atol": response_atol,
        "max_curvature_ratio": max_curvature_ratio,
    }
    delta = float(initial_delta)
    history: list[dict[str, object]] = []
    best_value: float | None = None
    accepted_steps = 0
    stop_reason = "max_steps"
    sample_metadata: list[dict[str, object]] = []
    n_samples = 0
    for step_index in range(max_steps_int):
        report = finite_difference_report_fn(base_delta=delta, **base_probe_kwargs)
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
        candidate_delta = delta - float(np.sign(derivative)) * update_step_float
        candidate = finite_difference_report_fn(base_delta=candidate_delta, **base_probe_kwargs)
        candidate_value = _report_float(candidate, "base_value")
        row["candidate_delta"] = candidate_delta
        row["candidate_objective"] = candidate_value
        candidate_ok = bool(candidate["passed"]) and candidate_value < base_value - min_improvement_float
        if not candidate_ok:
            stop_reason = "no_accepted_candidate"
            history.append(row)
            break
        delta = candidate_delta
        best_value = candidate_value
        accepted_steps += 1
        row["accepted"] = True
        history.append(row)

    initial_objective = float(cast(Any, history[0]["objective"])) if history else float("nan")
    final_objective = float(best_value) if best_value is not None else initial_objective
    return {
        "passed": bool(accepted_steps > 0 and final_objective < initial_objective),
        "final_delta": delta,
        "update_step": update_step_float,
        "max_steps": max_steps_int,
        "accepted_steps": accepted_steps,
        "stop_reason": stop_reason,
        "initial_objective": initial_objective,
        "final_objective": final_objective,
        "relative_reduction": _relative_reduction(initial_objective, final_objective),
        "history": history,
        "samples": sample_metadata,
        "n_samples": n_samples,
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

    probe_kwargs = {
        "case_name": case_name,
        "objective": objective,
        "reduction": reduction,
        "weights": weights,
        "surface_indices": surface_indices,
        "alphas": alphas,
        "selected_ky_indices": selected_ky_indices,
        "radial_index": radial_index,
        "mode_index": mode_index,
        "parameter_family": parameter_family,
        **kwargs,
    }
    search = _run_one_parameter_line_search(
        finite_difference_report_fn=finite_difference_report_fn,
        probe_kwargs=probe_kwargs,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
    )
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": bool(search["passed"]),
        "source_scope": "mode21_vmec_boozer_state_multi_point",
        "claim_scope": (
            "curvature-gated one-parameter line search for an aggregated "
            "VMEC/Boozer/SPECTRAX-GK linear/quasilinear objective; not a "
            "multi-parameter or nonlinear turbulent transport optimization claim"
        ),
        "case_name": str(case_name),
        "objective": str(objective),
        "reduction": str(reduction),
        "samples": search["samples"],
        "n_samples": search["n_samples"],
        "radial_index": None if radial_index is None else int(radial_index),
        "mode_index": int(mode_index),
        "initial_delta": float(initial_delta),
        "final_delta": search["final_delta"],
        "perturbation_step": float(perturbation_step),
        "update_step": search["update_step"],
        "max_steps": search["max_steps"],
        "accepted_steps": search["accepted_steps"],
        "stop_reason": search["stop_reason"],
        "initial_objective": search["initial_objective"],
        "final_objective": search["final_objective"],
        "relative_reduction": search["relative_reduction"],
        "history": search["history"],
        "options": _serializable_scalar_options(kwargs),
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

    probe_kwargs = {
        "case_name": case_name,
        "objective": objective,
        "radial_index": radial_index,
        "mode_index": mode_index,
        "parameter_family": parameter_family,
        **kwargs,
    }
    search = _run_one_parameter_line_search(
        finite_difference_report_fn=finite_difference_report_fn,
        probe_kwargs=probe_kwargs,
        initial_delta=initial_delta,
        perturbation_step=perturbation_step,
        update_step=update_step,
        max_steps=max_steps,
        min_improvement=min_improvement,
        response_atol=response_atol,
        max_curvature_ratio=max_curvature_ratio,
    )
    return {
        "kind": "vmec_boozer_scalar_objective_line_search_report",
        "passed": bool(search["passed"]),
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
        "final_delta": search["final_delta"],
        "perturbation_step": float(perturbation_step),
        "update_step": search["update_step"],
        "max_steps": search["max_steps"],
        "accepted_steps": search["accepted_steps"],
        "stop_reason": search["stop_reason"],
        "initial_objective": search["initial_objective"],
        "final_objective": search["final_objective"],
        "relative_reduction": search["relative_reduction"],
        "history": search["history"],
        "options": _serializable_scalar_options(kwargs),
    }


__all__ = [
    "vmec_boozer_aggregate_line_search_holdout_report",
    "vmec_boozer_aggregate_scalar_objective_line_search_report",
    "vmec_boozer_scalar_objective_line_search_report",
]
