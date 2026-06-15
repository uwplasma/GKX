"""Production-adjacent solver-objective geometry-gradient gates.

These helpers validate gradients of actual SPECTRAX-GK linear-RHS observables
with respect to solver-ready geometry arrays.  They are deliberately stricter
than reduced optimization proxies, but still narrower than a full
``vmec_jax -> booz_xform_jax -> solver`` optimization claim.
"""

from __future__ import annotations

import importlib
import time
from typing import Any, Literal, cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.autodiff_validation import (
    explicit_complex_operator_matrix,
    implicit_eigenpair_observable_sensitivity_report,
)
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.diagnostics import (
    heat_flux_species,
    particle_flux_species,
    fieldline_quadrature_weights,
)
from spectraxgk.geometry.differentiable import (
    discover_differentiable_geometry_backends,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_from_vmec_boozer_state,
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
)
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import (
    LinearParams,
    LinearTerms,
    build_linear_cache,
    linear_rhs_cached,
)
from spectraxgk.quasilinear import effective_kperp2, phi_norm2
from spectraxgk.solver_objective_core import (
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
    _default_gradient_linear_params,
    _default_gradient_linear_terms,
    solver_growth_rate_from_geometry,
    solver_linear_operator_matrix_from_geometry,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
)
from spectraxgk.solver_eigen_objectives import (
    dominant_eigenvalue_branch_locality_report,
    dominant_real_eigenvalue,
)
from spectraxgk.solver_geometry_objectives import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    TINY_OBJECTIVE_NAMES,
    _objective_gate_rows,
    default_solver_geometry_design_params,
    solver_ready_geometry_mapping,
    tiny_differentiable_objective_gradient_report,
)
from spectraxgk.solver_nonlinear_window_objective import (
    _reduced_nonlinear_window_metrics_from_linear_observables,
)
from spectraxgk.solver_objective_sampling import (
    _aggregate_sample_metadata,
    _aggregate_weights,
    _float_tuple,
    _int_tuple,
    _ky_sample_axis,
    _surface_index_tuple,
    _surface_sample_axis,
    solver_grid_options_from_ky_values,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    _split_vmec_boozer_objective_kwargs as _split_vmec_boozer_objective_kwargs_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_aggregate_scalar_objective_from_state as _vmec_boozer_aggregate_scalar_objective_from_state_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_scalar_objective_from_state as _vmec_boozer_scalar_objective_from_state_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_solver_objective_table_from_state as _vmec_boozer_solver_objective_table_from_state_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_solver_objective_table_with_metadata_from_state as _vmec_boozer_solver_objective_table_with_metadata_from_state_impl,
)
from spectraxgk.solver_vmec_boozer_objectives import (
    vmec_boozer_solver_objective_vector_from_state as _vmec_boozer_solver_objective_vector_from_state_impl,
)
from spectraxgk.solver_vmec_state import (
    VMEC_BOOZER_STATE_PARAMETER_FAMILIES,
    VMEC_BOOZER_STATE_PARAMETER_NAMES,
    _replace_vmec_boozer_state_coefficient,
    _vmec_boozer_state_array,
    _vmec_boozer_state_parameter_name,
)

VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES = ("gamma", "omega")
VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
)
VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES = (
    "gamma",
    "omega",
    "kperp_eff2",
    "linear_heat_flux_weight",
    "mixing_length_heat_flux_proxy",
    "nonlinear_window_heat_flux_mean",
    "nonlinear_window_heat_flux_cv",
    "nonlinear_window_heat_flux_trend",
)


def _report_float(report: dict[str, object], key: str) -> float:
    """Read a numeric finite-difference report field with mypy-safe casting."""

    return float(cast(Any, report[key]))


def _split_vmec_boozer_objective_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _split_vmec_boozer_objective_kwargs_impl(kwargs)


def vmec_boozer_solver_objective_vector_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives from the in-memory VMEC/Boozer bridge."""

    return _vmec_boozer_solver_objective_vector_from_state_impl(
        state,
        static,
        indata,
        wout,
        geometry_fn=flux_tube_geometry_from_vmec_boozer_state,
        objective_vector_fn=solver_objective_vector_from_geometry,
        **kwargs,
    )


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
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate solver objectives over a surface/field-line/``k_y`` table."""

    return _vmec_boozer_solver_objective_table_from_state_impl(
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
        table_with_metadata_fn=vmec_boozer_solver_objective_table_with_metadata_from_state,
        **kwargs,
    )


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
    **kwargs: Any,
) -> tuple[jnp.ndarray, list[dict[str, object]]]:
    """Evaluate VMEC/Boozer objective rows and return sample metadata."""

    return _vmec_boozer_solver_objective_table_with_metadata_from_state_impl(
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
        geometry_fn=flux_tube_geometry_from_vmec_boozer_state,
        objective_vector_fn=solver_objective_vector_from_geometry,
        **kwargs,
    )


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
    **kwargs: Any,
) -> jnp.ndarray:
    """Reduce a VMEC/Boozer multi-point objective table to one scalar."""

    return _vmec_boozer_aggregate_scalar_objective_from_state_impl(
        state,
        static,
        indata,
        wout,
        objective=objective,
        reduction=reduction,
        weights=weights,
        surface_indices=surface_indices,
        alphas=alphas,
        selected_ky_indices=selected_ky_indices,
        table_fn=vmec_boozer_solver_objective_table_from_state,
        **kwargs,
    )


def vmec_boozer_scalar_objective_from_state(  # pragma: no cover
    state: Any,
    static: Any,
    indata: Any,
    wout: Any,
    *,
    objective: SolverScalarObjective = "growth",
    **kwargs: Any,
) -> jnp.ndarray:
    """Evaluate one scalar optimization objective on the VMEC/Boozer path."""

    return _vmec_boozer_scalar_objective_from_state_impl(
        state,
        static,
        indata,
        wout,
        objective=objective,
        vector_fn=vmec_boozer_solver_objective_vector_from_state,
        **kwargs,
    )


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

    step = float(perturbation_step)
    if step <= 0.0:
        raise ValueError("perturbation_step must be positive")
    curvature_ratio_limit = float(max_curvature_ratio)
    if curvature_ratio_limit < 0.0:
        raise ValueError("max_curvature_ratio must be non-negative")
    bundle = _load_vmec_jax_example_state_bundle(str(case_name))
    state = bundle["state"]
    base_coeff = _vmec_boozer_state_array(state, parameter_family)
    default_radial_index = int(base_coeff.shape[0] // 2)
    radial_index_int = (
        default_radial_index if radial_index is None else int(radial_index)
    )
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_coeff.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_coeff.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_name = _vmec_boozer_state_parameter_name(
        parameter_family,
        radial_index_int,
        mode_index_int,
        default_mid_surface=default_radial_index,
    )

    base_delta_float = float(base_delta)

    def evaluate(delta: float) -> tuple[float, list[float]]:
        traced_state = _replace_vmec_boozer_state_coefficient(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            base_delta_float + float(delta),
        )
        vector = vmec_boozer_solver_objective_vector_from_state(
            traced_state,
            bundle["static"],
            bundle["indata"],
            bundle["wout"],
            **kwargs,
        )
        scalar = solver_scalar_objective_from_vector(vector, objective)
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
    bundle = _load_vmec_jax_example_state_bundle(str(case_name))
    state = bundle["state"]
    base_coeff = _vmec_boozer_state_array(state, parameter_family)
    default_radial_index = int(base_coeff.shape[0] // 2)
    radial_index_int = (
        default_radial_index if radial_index is None else int(radial_index)
    )
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_coeff.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_coeff.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_name = _vmec_boozer_state_parameter_name(
        parameter_family,
        radial_index_int,
        mode_index_int,
        default_mid_surface=default_radial_index,
    )

    base_delta_float = float(base_delta)

    def evaluate(
        delta: float,
    ) -> tuple[float, list[float], list[list[float]], list[dict[str, object]]]:
        traced_state = _replace_vmec_boozer_state_coefficient(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            base_delta_float + float(delta),
        )
        table, sample_metadata = (
            vmec_boozer_solver_objective_table_with_metadata_from_state(
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
        )
        scalar_values = np.asarray(
            [solver_scalar_objective_from_vector(row, objective) for row in table],
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
        report = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
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
        candidate = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
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

    min_holdout_improvement_float = float(min_holdout_improvement)
    if min_holdout_improvement_float < 0.0:
        raise ValueError("min_holdout_improvement must be non-negative")

    training = vmec_boozer_aggregate_scalar_objective_line_search_report(
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

    heldout_initial = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
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
    heldout_final = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
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
        report = vmec_boozer_scalar_objective_finite_difference_report(
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
        candidate = vmec_boozer_scalar_objective_finite_difference_report(
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


def solver_objective_branch_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
    n_laguerre: int = 2,
    n_hermite: int = 1,
) -> dict[str, object]:
    """Validate branch continuity and AD/FD sensitivities for solver objectives.

    This gate is the lightweight counterpart of the VMEC/Boozer offline gates.
    It uses the solver-ready differentiable geometry contract so CI can check
    the objective path without optional geometry backends. The report requires
    the max-growth branch to stay dominant under central finite-difference
    perturbations and validates the objective sensitivities with the implicit
    left/right eigenpair method.
    """

    p = (
        default_solver_geometry_design_params()
        if params is None
        else jnp.asarray(params)
    )
    if p.ndim != 1 or int(p.size) != len(SOLVER_GEOMETRY_PARAMETER_NAMES):
        raise ValueError(
            f"params must be a length-{len(SOLVER_GEOMETRY_PARAMETER_NAMES)} vector"
        )
    n_laguerre_int = int(n_laguerre)
    n_hermite_int = int(n_hermite)
    if n_laguerre_int < 1 or n_hermite_int < 1:
        raise ValueError("n_laguerre and n_hermite must be positive")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    state_shape = (
        n_laguerre_int,
        n_hermite_int,
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    params_linear = _default_gradient_linear_params()
    terms = _default_gradient_linear_terms()
    theta = jnp.asarray(grid.z)

    def geometry_for(x: jnp.ndarray):
        return flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="solver_ready_branch_gradient_gate",
            validate_finite=False,
        )

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(
            grid, geometry_for(x), params_linear, n_laguerre_int, n_hermite_int
        )

    def rhs_phi(state_arr: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            params_linear,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state_arr: rhs_phi(state_arr, cache)[0], state_shape
        )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = (
            _mode21_vmec_boozer_quasilinear_features(
                eigenvalue,
                eigenvector,
                x,
                {
                    "geometry_for": geometry_for,
                    "grid": grid,
                    "params_linear": params_linear,
                    "n_laguerre": n_laguerre_int,
                    "n_hermite": n_hermite_int,
                    "state_shape": state_shape,
                    "rhs_phi": rhs_phi,
                },
            )
        )
        geom = geometry_for(x)
        cache = cache_for(x)
        state_arr = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state_arr, cache)
        zero_field = jnp.zeros_like(phi)
        _vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
        particle_weight = jnp.real(
            jnp.sum(
                particle_flux_species(
                    state_arr,
                    phi,
                    zero_field,
                    zero_field,
                    cache,
                    grid,
                    params_linear,
                    flux_fac,
                )
            )
            / phi_norm2(
                phi, cache, params_linear, fieldline_quadrature_weights(geom, grid)[0]
            )
        )
        return jnp.asarray(
            [gamma, omega, kperp_eff, heat_weight, particle_weight, ql_proxy]
        )

    base_matrix = matrix_fn(p)
    base_eigs = np.asarray(jnp.linalg.eigvals(base_matrix))
    if base_eigs.ndim != 1 or base_eigs.size == 0:
        raise ValueError("matrix_fn must return at least one eigenvalue")
    base_index = int(np.argmax(np.real(base_eigs)))
    base_value = base_eigs[base_index]
    base_gap = (
        float("inf")
        if base_eigs.size == 1
        else float(np.min(np.abs(np.delete(base_eigs, base_index) - base_value)))
    )
    eye = jnp.eye(int(p.size), dtype=p.dtype)
    branch_rows = []
    for i, name in enumerate(SOLVER_GEOMETRY_PARAMETER_NAMES):
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            p_i = p + float(sign) * float(fd_step) * eye[i]
            eigs_i = np.asarray(jnp.linalg.eigvals(matrix_fn(p_i)))
            nearest_index = int(np.argmin(np.abs(eigs_i - base_value)))
            dominant_index = int(np.argmax(np.real(eigs_i)))
            nearest_value = eigs_i[nearest_index]
            nearest_gap = (
                float("inf")
                if eigs_i.size == 1
                else float(
                    np.min(np.abs(np.delete(eigs_i, nearest_index) - nearest_value))
                )
            )
            row_passed = bool(
                nearest_index == dominant_index and nearest_gap >= float(gap_floor)
            )
            branch_rows.append(
                {
                    "parameter": name,
                    "direction": label,
                    "nearest_index": nearest_index,
                    "dominant_index": dominant_index,
                    "nearest_real": float(np.real(nearest_value)),
                    "nearest_imag": float(np.imag(nearest_value)),
                    "nearest_gap": nearest_gap,
                    "passed": row_passed,
                }
            )

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=SOLVER_GEOMETRY_PARAMETER_NAMES,
        objective_names=SOLVER_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    value_vector = solver_objective_vector_from_geometry(
        geometry_for(p),
        selected_ky_index=1,
        n_laguerre=n_laguerre_int,
        n_hermite=n_hermite_int,
        ny=cfg.grid.Ny,
    )
    value_np = np.asarray(value_vector, dtype=float)
    value_finite = bool(np.all(np.isfinite(value_np)))
    branch_passed = bool(
        base_gap >= float(gap_floor) and all(row["passed"] for row in branch_rows)
    )
    ad_fd_passed = bool(gate["passed"] and all(row["passed"] for row in rows))
    return {
        "kind": "solver_objective_branch_gradient_gate",
        "passed": bool(value_finite and branch_passed and ad_fd_passed),
        "source_scope": "solver_ready_geometry_contract",
        "claim_scope": (
            "solver-objective branch-continuity and implicit AD/FD gate on the "
            "solver-ready geometry contract; VMEC/Boozer production gates remain separate"
        ),
        "parameter_names": list(SOLVER_GEOMETRY_PARAMETER_NAMES),
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "params": np.asarray(p, dtype=float).tolist(),
        "grid": {
            "Nx": int(cfg.grid.Nx),
            "Ny": int(cfg.grid.Ny),
            "Nz": int(cfg.grid.Nz),
            "selected_ky_index": 1,
        },
        "n_laguerre": n_laguerre_int,
        "n_hermite": n_hermite_int,
        "state_size": int(np.prod(state_shape)),
        "value_evaluator_finite": value_finite,
        "value_evaluator_objectives": value_np.tolist(),
        "branch_continuity_gate": branch_passed,
        "base_selected_index": base_index,
        "base_eigenvalue_real": float(np.real(base_value)),
        "base_eigenvalue_imag": float(np.imag(base_value)),
        "base_eigenvalue_gap": base_gap,
        "branch_rows": branch_rows,
        "ad_fd_gate": ad_fd_passed,
        "objective_gates": rows,
        "eigenpair_gate": gate,
    }


def _mode21_vmec_boozer_linear_context(  # pragma: no cover
    *,
    case_name: str,
    radial_index: int | None,
    mode_index: int,
    parameter_family: str,
    surface_index: int | None,
    ntheta: int,
    mboz: int,
    nboz: int,
    surface_stencil_width: int | None,
    n_laguerre: int,
    n_hermite: int,
) -> dict[str, Any]:
    """Build shared VMEC/Boozer geometry and linear-RHS closures for gates."""

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
    base_coeff = _vmec_boozer_state_array(state, parameter_family)
    default_radial_index = int(base_coeff.shape[0] // 2)
    radial_index_int = (
        default_radial_index if radial_index is None else int(radial_index)
    )
    mode_index_int = int(mode_index)
    if not (0 <= radial_index_int < int(base_coeff.shape[0])):
        raise ValueError("radial_index is outside the VMEC state radial grid")
    if not (0 <= mode_index_int < int(base_coeff.shape[1])):
        raise ValueError("mode_index is outside the VMEC state mode table")
    parameter_names = (
        _vmec_boozer_state_parameter_name(
            parameter_family,
            radial_index_int,
            mode_index_int,
            default_mid_surface=default_radial_index,
        ),
    )

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=4, Nz=int(ntheta), Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    state_shape = (
        int(n_laguerre),
        int(n_hermite),
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    params_linear = _default_gradient_linear_params()
    terms = _default_gradient_linear_terms()

    def geometry_for(x: jnp.ndarray):
        traced_state = _replace_vmec_boozer_state_coefficient(
            state,
            parameter_family,
            base_coeff,
            radial_index_int,
            mode_index_int,
            x[0],
        )
        mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
            traced_state,
            static,
            indata,
            wout,
            surface_index=surface_index,
            ntheta=int(ntheta),
            mboz=int(mboz),
            nboz=int(nboz),
            surface_stencil_width=surface_stencil_width,
        )
        return flux_tube_geometry_from_mapping(
            mapping,
            source_model="mode21_vmec_boozer_state",
            validate_finite=False,
        )

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(
            grid, geometry_for(x), params_linear, int(n_laguerre), int(n_hermite)
        )

    def rhs_phi(state_arr: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state_arr,
            cache,
            params_linear,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state_arr: rhs_phi(state_arr, cache)[0], state_shape
        )

    return {
        "case_name": str(case_name),
        "cfg": cfg,
        "grid": grid,
        "parameter_names": parameter_names,
        "parameter_indices": {
            str(parameter_family): [radial_index_int, mode_index_int]
        },
        "surface_index": surface_index,
        "mboz": int(mboz),
        "nboz": int(nboz),
        "surface_stencil_width": surface_stencil_width,
        "n_laguerre": int(n_laguerre),
        "n_hermite": int(n_hermite),
        "state_shape": state_shape,
        "params_linear": params_linear,
        "geometry_for": geometry_for,
        "cache_for": cache_for,
        "rhs_phi": rhs_phi,
        "matrix_fn": matrix_fn,
    }


def _mode21_vmec_boozer_quasilinear_features(
    eigenvalue: jnp.ndarray,
    eigenvector: jnp.ndarray,
    x: jnp.ndarray,
    context: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    geom = context["geometry_for"](x)
    grid = context["grid"]
    params_linear = context["params_linear"]
    cache = build_linear_cache(
        grid,
        geom,
        params_linear,
        context["n_laguerre"],
        context["n_hermite"],
    )
    state_arr = jnp.reshape(eigenvector, context["state_shape"])
    _rhs, phi = context["rhs_phi"](state_arr, cache)
    zero_field = jnp.zeros_like(phi)
    vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
    norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
    kperp_eff = effective_kperp2(phi, cache, vol_fac)
    heat_weight = jnp.real(
        jnp.sum(
            heat_flux_species(
                state_arr,
                phi,
                zero_field,
                zero_field,
                cache,
                grid,
                params_linear,
                flux_fac,
            )
        )
        / norm2
    )
    gamma = jnp.real(eigenvalue)
    ql_proxy = (
        gamma
        * heat_weight
        / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
    )
    return gamma, jnp.imag(eigenvalue), kperp_eff, heat_weight, ql_proxy


def linear_solver_geometry_gradient_report(
    params: jnp.ndarray | np.ndarray | None = None,
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 1.0e-1,
    atol: float = 2.0e-3,
    gap_floor: float = 1.0e-6,
) -> dict[str, object]:
    """Validate solver-objective geometry gradients on the actual linear RHS.

    The report differentiates a small electrostatic Cyclone-like linear
    operator with respect to geometry arrays entering the production cache. It
    uses implicit left/right eigenpair sensitivities and compares them with
    nearest-branch central finite differences.
    """

    p = (
        default_solver_geometry_design_params()
        if params is None
        else jnp.asarray(params)
    )
    if p.ndim != 1 or int(p.size) != 2:
        raise ValueError("params must be a length-2 vector")

    cfg = CycloneBaseCase(grid=GridConfig(Nx=1, Ny=6, Nz=4, Lx=6.0, Ly=12.0))
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    n_laguerre = 2
    n_hermite = 1
    state_shape = (n_laguerre, n_hermite, grid.ky.size, grid.kx.size, grid.z.size)
    params_linear = LinearParams(
        R_over_Ln=2.2,
        R_over_LTi=6.9,
        nu=0.0,
        nu_hyper=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
        D_hyper=0.0,
        beta=0.0,
        fapar=0.0,
    )
    terms = LinearTerms(
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    theta = jnp.asarray(grid.z)

    def geometry_for(x: jnp.ndarray):
        return flux_tube_geometry_from_mapping(
            solver_ready_geometry_mapping(x, theta),
            source_model="solver_ready_geometry_gradient_gate",
            validate_finite=False,
        )

    def cache_for(x: jnp.ndarray):
        return build_linear_cache(
            grid, geometry_for(x), params_linear, n_laguerre, n_hermite
        )

    def rhs_phi(state: jnp.ndarray, cache: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        return linear_rhs_cached(
            state,
            cache,
            params_linear,
            terms=terms,
            use_jit=False,
            use_custom_vjp=False,
        )

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        cache = cache_for(x)
        return explicit_complex_operator_matrix(
            lambda state: rhs_phi(state, cache)[0], state_shape
        )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        geom = geometry_for(x)
        cache = cache_for(x)
        state = jnp.reshape(eigenvector, state_shape)
        _rhs, phi = rhs_phi(state, cache)
        zero_field = jnp.zeros_like(phi)
        vol_fac, flux_fac = fieldline_quadrature_weights(geom, grid)
        norm2 = phi_norm2(phi, cache, params_linear, vol_fac)
        kperp_eff = effective_kperp2(phi, cache, vol_fac)
        heat_weight = jnp.real(
            jnp.sum(
                heat_flux_species(
                    state,
                    phi,
                    zero_field,
                    zero_field,
                    cache,
                    grid,
                    params_linear,
                    flux_fac,
                )
            )
            / norm2
        )
        particle_weight = jnp.real(
            jnp.sum(
                particle_flux_species(
                    state,
                    phi,
                    zero_field,
                    zero_field,
                    cache,
                    grid,
                    params_linear,
                    flux_fac,
                )
            )
            / norm2
        )
        gamma = jnp.real(eigenvalue)
        ql_proxy = (
            gamma
            * heat_weight
            / jnp.maximum(kperp_eff, jnp.asarray(1.0e-12, dtype=kperp_eff.dtype))
        )
        return jnp.asarray(
            [
                gamma,
                jnp.imag(eigenvalue),
                kperp_eff,
                heat_weight,
                particle_weight,
                ql_proxy,
            ]
        )

    gate = implicit_eigenpair_observable_sensitivity_report(
        matrix_fn,
        objective_fn,
        p,
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(gate, rtol=rtol, atol=atol)
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in SOLVER_OBJECTIVE_NAMES
    }
    linear_growth_gate = bool(by_objective["gamma"] and by_objective["omega"])
    quasilinear_weight_gate = bool(
        by_objective["linear_heat_flux_weight"]
        and by_objective["mixing_length_heat_flux_proxy"]
    )
    return {
        "kind": "linear_solver_geometry_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "solver_ready_geometry_contract",
        "claim_scope": (
            "actual_linear_rhs_solver_objectives; not yet a full "
            "vmec_jax_to_booz_xform_jax_to_solver gradient claim"
        ),
        "parameter_names": list(SOLVER_GEOMETRY_PARAMETER_NAMES),
        "objective_names": list(SOLVER_OBJECTIVE_NAMES),
        "params": np.asarray(p, dtype=float).tolist(),
        "grid": {
            "Nx": int(cfg.grid.Nx),
            "Ny": int(cfg.grid.Ny),
            "Nz": int(cfg.grid.Nz),
            "selected_ky_index": 1,
        },
        "n_laguerre": n_laguerre,
        "n_hermite": n_hermite,
        "state_size": int(np.prod(state_shape)),
        "linear_growth_gradient_gate": linear_growth_gate,
        "quasilinear_weight_gradient_gate": quasilinear_weight_gate,
        "nonlinear_window_gradient_gate": False,
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Promote this solver-ready geometry-gradient gate to actual mode-21 "
            "VMEC/Boozer state coefficients, then add nonlinear-window objective gradients."
        ),
    }


def mode21_vmec_boozer_linear_frequency_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 5.0e-2,
    atol: float = 2.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    """Validate a full VMEC/Boozer-state gradient of linear frequency.

    This is an offline manuscript artifact gate.  It perturbs one mid-surface
    VMEC Fourier coefficient, maps it through ``vmec_jax`` and
    ``booz_xform_jax`` into the mode-21 equal-arc flux-tube geometry contract,
    builds the SPECTRAX-GK linear RHS, and compares implicit eigenpair
    sensitivities against central finite differences.  Quasilinear flux-weight
    state gradients are intentionally not promoted here because the current
    full-chain diagnostic is substantially heavier and remains an optimization
    campaign lane.
    """

    context = _mode21_vmec_boozer_linear_context(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=1,
        n_hermite=1,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, _eigenvector: jnp.ndarray, _x: jnp.ndarray
    ) -> jnp.ndarray:
        return jnp.asarray([jnp.real(eigenvalue), jnp.imag(eigenvalue)])

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
        objective_names=VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES
    }
    return {
        "kind": "mode21_vmec_boozer_linear_frequency_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc "
            "geometry -> SPECTRAX-GK linear-RHS eigenfrequency gradient"
        ),
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": False,
        "nonlinear_window_gradient_gate": False,
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Promote the full-chain gate from eigenfrequency to quasilinear flux weights after "
            "the heavy Nl>=2 diagnostic is profiled and conditioned below manuscript runtime caps."
        ),
    }


def mode21_vmec_boozer_quasilinear_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 2.0e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
) -> dict[str, object]:
    """Validate full VMEC/Boozer-state gradients of quasilinear observables.

    This offline manuscript gate is the production-gradient companion to
    :func:`mode21_vmec_boozer_linear_frequency_gradient_report`.  It uses a
    richer ``Nl=2, Nm=3`` moment basis so the electrostatic heat-flux weight is
    nonzero, then validates implicit eigenpair sensitivities of ``gamma``,
    ``omega``, ``<k_perp^2>``, the linear heat-flux weight, and the
    mixing-length heat-flux proxy against central finite differences.
    """

    start = time.perf_counter()
    context = _mode21_vmec_boozer_linear_context(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=2,
        n_hermite=3,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = (
            _mode21_vmec_boozer_quasilinear_features(
                eigenvalue,
                eigenvector,
                x,
                context,
            )
        )
        return jnp.asarray([gamma, omega, kperp_eff, heat_weight, ql_proxy])

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
        objective_names=VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES
    }
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc "
            "geometry -> SPECTRAX-GK linear-RHS quasilinear heat-flux-weight gradient"
        ),
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": bool(
            by_objective["linear_heat_flux_weight"]
            and by_objective["mixing_length_heat_flux_proxy"]
        ),
        "nonlinear_window_gradient_gate": False,
        "elapsed_seconds": float(time.perf_counter() - start),
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Use this as the full-chain quasilinear gradient gate for reduced linear/quasilinear "
            "stellarator objectives; keep full nonlinear-window VMEC/Boozer gradients as a separate future lane."
        ),
    }


def mode21_vmec_boozer_nonlinear_window_gradient_report(  # pragma: no cover
    *,
    case_name: str = "nfp4_QH_warm_start",
    radial_index: int | None = None,
    mode_index: int = 1,
    parameter_family: str = "Rcos",
    surface_index: int | None = None,
    fd_step: float = 1.0e-6,
    rtol: float = 7.5e-2,
    atol: float = 5.0e-2,
    gap_floor: float = 1.0e-8,
    ntheta: int = 4,
    mboz: int = 21,
    nboz: int = 21,
    surface_stencil_width: int | None = None,
    nonlinear_dt: float = 0.18,
    nonlinear_steps: int = 96,
    tail_fraction: float = 0.30,
) -> dict[str, object]:
    """Validate VMEC/Boozer-state gradients of a nonlinear-window estimator.

    The gate reuses the full ``vmec_jax`` state to ``booz_xform_jax`` to
    SPECTRAX-GK linear-RHS path from the quasilinear gradient gate, then feeds
    the isolated eigenpair observables into a differentiable late-time
    heat-flux-envelope estimator.  It is a reduced nonlinear-window
    differentiability gate; converged nonlinear turbulence windows and
    optimized-equilibrium nonlinear audits remain separate promotion gates.
    """

    start = time.perf_counter()
    context = _mode21_vmec_boozer_linear_context(
        case_name=str(case_name),
        radial_index=radial_index,
        mode_index=mode_index,
        parameter_family=parameter_family,
        surface_index=surface_index,
        ntheta=ntheta,
        mboz=mboz,
        nboz=nboz,
        surface_stencil_width=surface_stencil_width,
        n_laguerre=2,
        n_hermite=3,
    )

    def objective_fn(
        eigenvalue: jnp.ndarray, eigenvector: jnp.ndarray, x: jnp.ndarray
    ) -> jnp.ndarray:
        gamma, omega, kperp_eff, heat_weight, ql_proxy = (
            _mode21_vmec_boozer_quasilinear_features(
                eigenvalue,
                eigenvector,
                x,
                context,
            )
        )
        nl_mean, nl_cv, nl_trend = (
            _reduced_nonlinear_window_metrics_from_linear_observables(
                gamma,
                kperp_eff,
                heat_weight,
                dt=nonlinear_dt,
                steps=nonlinear_steps,
                tail_fraction=tail_fraction,
            )
        )
        return jnp.asarray(
            [
                gamma,
                omega,
                kperp_eff,
                heat_weight,
                ql_proxy,
                nl_mean,
                nl_cv,
                nl_trend,
            ]
        )

    gate = implicit_eigenpair_observable_sensitivity_report(
        context["matrix_fn"],
        objective_fn,
        jnp.asarray([0.0]),
        step=fd_step,
        rtol=rtol,
        atol=atol,
        gap_floor=gap_floor,
    )
    rows = _objective_gate_rows(
        gate,
        parameter_names=context["parameter_names"],
        objective_names=VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
        rtol=rtol,
        atol=atol,
    )
    by_objective = {
        name: bool(all(row["passed"] for row in rows if row["objective"] == name))
        for name in VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES
    }
    nonlinear_window_gate = bool(
        by_objective["nonlinear_window_heat_flux_mean"]
        and by_objective["nonlinear_window_heat_flux_cv"]
        and by_objective["nonlinear_window_heat_flux_trend"]
    )
    return {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "passed": bool(gate["passed"] and all(row["passed"] for row in rows)),
        "source_scope": "mode21_vmec_boozer_state",
        "claim_scope": (
            "full vmec_jax state coefficient -> booz_xform_jax mode-21 equal-arc geometry "
            "-> SPECTRAX-GK linear-RHS eigenpair -> reduced nonlinear-window estimator gradient"
        ),
        "case_name": context["case_name"],
        "parameter_names": list(context["parameter_names"]),
        "objective_names": list(VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES),
        "parameter_indices": context["parameter_indices"],
        "surface_index": None
        if context["surface_index"] is None
        else int(context["surface_index"]),
        "grid": {
            "Nx": int(context["cfg"].grid.Nx),
            "Ny": int(context["cfg"].grid.Ny),
            "Nz": int(context["cfg"].grid.Nz),
            "selected_ky_index": 1,
        },
        "mboz": context["mboz"],
        "nboz": context["nboz"],
        "surface_stencil_width": (
            None
            if context["surface_stencil_width"] is None
            else int(context["surface_stencil_width"])
        ),
        "n_laguerre": context["n_laguerre"],
        "n_hermite": context["n_hermite"],
        "state_size": int(np.prod(context["state_shape"])),
        "linear_growth_gradient_gate": bool(by_objective["gamma"]),
        "linear_frequency_gradient_gate": bool(by_objective["omega"]),
        "quasilinear_weight_gradient_gate": bool(
            by_objective["linear_heat_flux_weight"]
            and by_objective["mixing_length_heat_flux_proxy"]
        ),
        "nonlinear_window_gradient_gate": nonlinear_window_gate,
        "nonlinear_window_config": {
            "model": "smooth_logistic_heat_flux_envelope_from_linear_observables",
            "dt": float(nonlinear_dt),
            "steps": int(nonlinear_steps),
            "tail_fraction": float(tail_fraction),
        },
        "elapsed_seconds": float(time.perf_counter() - start),
        "objective_gates": rows,
        "eigenpair_gate": gate,
        "next_action": (
            "Use this as a reduced nonlinear-window estimator-gradient gate only. Full stellarator "
            "heat-flux optimization still requires converged nonlinear SPECTRAX-GK window gradients "
            "or robust adjoint/finite-difference audits on optimized equilibria."
        ),
    }


__all__ = [
    "_aggregate_sample_metadata",
    "_aggregate_weights",
    "_float_tuple",
    "_int_tuple",
    "_ky_sample_axis",
    "_surface_index_tuple",
    "_surface_sample_axis",
    "SOLVER_GEOMETRY_PARAMETER_NAMES",
    "SOLVER_OBJECTIVE_NAMES",
    "SolverScalarObjective",
    "TINY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES",
    "VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES",
    "VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES",
    "VMEC_BOOZER_STATE_PARAMETER_FAMILIES",
    "VMEC_BOOZER_STATE_PARAMETER_NAMES",
    "default_solver_geometry_design_params",
    "dominant_eigenvalue_branch_locality_report",
    "dominant_real_eigenvalue",
    "linear_solver_geometry_gradient_report",
    "mode21_vmec_boozer_linear_frequency_gradient_report",
    "mode21_vmec_boozer_nonlinear_window_gradient_report",
    "mode21_vmec_boozer_quasilinear_gradient_report",
    "solver_growth_rate_from_geometry",
    "solver_linear_operator_matrix_from_geometry",
    "solver_objective_branch_gradient_report",
    "solver_objective_vector_from_geometry",
    "solver_scalar_objective_from_vector",
    "solver_grid_options_from_ky_values",
    "solver_ready_geometry_mapping",
    "tiny_differentiable_objective_gradient_report",
    "vmec_boozer_aggregate_line_search_holdout_report",
    "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
    "vmec_boozer_aggregate_scalar_objective_from_state",
    "vmec_boozer_aggregate_scalar_objective_line_search_report",
    "vmec_boozer_scalar_objective_finite_difference_report",
    "vmec_boozer_scalar_objective_from_state",
    "vmec_boozer_scalar_objective_line_search_report",
    "vmec_boozer_solver_objective_table_from_state",
    "vmec_boozer_solver_objective_table_with_metadata_from_state",
    "vmec_boozer_solver_objective_vector_from_state",
]
