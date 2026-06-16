"""Finite-difference gates for VMEC/Boozer objectives."""

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


__all__ = [
    "_load_vmec_jax_example_state_bundle",
    "_report_float",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
]
