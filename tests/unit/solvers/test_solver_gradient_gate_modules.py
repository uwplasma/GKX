from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from gkx.objectives.core import solver_scalar_objective_from_vector
import gkx.objectives.vmec_boozer_context as vmec_gradient_context
import gkx.objectives.vmec_boozer_fd as fd_gates
import gkx.objectives.vmec_boozer_gradients as vmec_gradient_gates
import gkx.objectives.vmec_boozer_line_search as line_search_gates


def _fake_state_bundle(_case_name: str) -> dict[str, object]:
    return {
        "case_name": "fake",
        "input_path": "input.fake",
        "wout_path": "wout.fake.nc",
        "state": object(),
        "runtime": object(),
        "inp": object(),
        "wout": object(),
    }


def _fake_state_array(_state: object, _parameter_family: str) -> np.ndarray:
    return np.zeros((5, 3), dtype=float)


def _fake_replace_state_coefficient(
    _state: object,
    parameter_family: str,
    _base_coeff: np.ndarray,
    radial_index: int,
    mode_index: int,
    delta: float,
) -> dict[str, object]:
    return {
        "family": parameter_family,
        "radial_index": radial_index,
        "mode_index": mode_index,
        "delta": float(delta),
    }


def _fake_parameter_name(
    parameter_family: str,
    radial_index: int,
    mode_index: int,
    *,
    default_mid_surface: int,
) -> str:
    suffix = "mid" if radial_index == default_mid_surface else f"r{radial_index}"
    return f"{parameter_family}_{suffix}_m{mode_index}"


def _fake_objective_vector(
    traced_state: dict[str, object], *_args: Any, **_kwargs: Any
) -> np.ndarray:
    x = float(traced_state["delta"])
    return np.asarray([1.0 + 3.0 * x + x * x, 0.1 + x, 2.0, 3.0, 4.0, 5.0 + x])


def test_vmec_boozer_scalar_fd_gate_uses_injected_value_path() -> None:
    report = fd_gates.vmec_boozer_scalar_objective_finite_difference_report(
        objective="growth",
        perturbation_step=1.0e-2,
        response_atol=1.0e-4,
        max_curvature_ratio=1.0,
        _load_state_bundle_fn=_fake_state_bundle,
        _state_array_fn=_fake_state_array,
        _replace_state_coefficient_fn=_fake_replace_state_coefficient,
        _parameter_name_fn=_fake_parameter_name,
        _vector_fn=_fake_objective_vector,
        _scalar_selector_fn=solver_scalar_objective_from_vector,
    )

    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state"
    assert report["parameter_name"] == "Rcos_mid_m1"
    assert report["central_derivative"] == pytest.approx(3.0)
    assert report["finite_difference_consistent"] is True
    assert np.asarray(report["base_objective_vector"]).shape == (6,)

    with pytest.raises(ValueError, match="perturbation_step"):
        fd_gates.vmec_boozer_scalar_objective_finite_difference_report(
            perturbation_step=0.0,
            _load_state_bundle_fn=_fake_state_bundle,
        )


def _fake_objective_table(
    traced_state: dict[str, object],
    *_args: Any,
    **_kwargs: Any,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x = float(traced_state["delta"])
    table = np.asarray(
        [
            [1.0 + 2.0 * x, 0.0, 1.0, 1.0, 1.0, 2.0],
            [2.0 + 4.0 * x, 0.0, 1.0, 1.0, 1.0, 4.0],
        ],
        dtype=float,
    )
    metadata = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1},
        {"surface_index": 2, "alpha": 0.0, "selected_ky_index": 1},
    ]
    return table, metadata


def test_vmec_boozer_aggregate_fd_gate_tracks_weighted_samples() -> None:
    report = fd_gates.vmec_boozer_aggregate_scalar_objective_finite_difference_report(
        objective="growth",
        reduction="weighted_mean",
        weights=[0.25, 0.75],
        surface_indices=(None, 2),
        perturbation_step=1.0e-2,
        response_atol=1.0e-4,
        max_curvature_ratio=1.0,
        _load_state_bundle_fn=_fake_state_bundle,
        _state_array_fn=_fake_state_array,
        _replace_state_coefficient_fn=_fake_replace_state_coefficient,
        _parameter_name_fn=_fake_parameter_name,
        _table_with_metadata_fn=_fake_objective_table,
        _scalar_selector_fn=solver_scalar_objective_from_vector,
    )

    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state_multi_point"
    assert report["n_samples"] == 2
    assert report["central_derivative"] == pytest.approx(3.5)
    assert [row["weight"] for row in report["samples"]] == pytest.approx([0.25, 0.75])

    with pytest.raises(ValueError, match="max_curvature_ratio"):
        fd_gates.vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            max_curvature_ratio=-1.0,
            _load_state_bundle_fn=_fake_state_bundle,
        )


def _parabola_fd_report(**kwargs: Any) -> dict[str, object]:
    delta = float(kwargs.get("base_delta", 0.0))
    value = 1.0 + (delta - 0.03) ** 2
    derivative = 2.0 * (delta - 0.03)
    return {
        "passed": True,
        "base_value": value,
        "central_derivative": derivative,
        "curvature_ratio": 0.0,
        "n_samples": 1,
        "samples": [{"surface_index": None, "alpha": 0.0, "selected_ky_index": 1}],
    }


def test_line_search_gates_accept_downhill_injected_candidates() -> None:
    scalar = line_search_gates.vmec_boozer_scalar_objective_line_search_report(
        initial_delta=0.0,
        perturbation_step=1.0e-2,
        update_step=1.0e-2,
        max_steps=2,
        _finite_difference_report_fn=_parabola_fd_report,
    )
    aggregate = (
        line_search_gates.vmec_boozer_aggregate_scalar_objective_line_search_report(
            initial_delta=0.0,
            perturbation_step=1.0e-2,
            update_step=1.0e-2,
            max_steps=2,
            _finite_difference_report_fn=_parabola_fd_report,
        )
    )

    assert scalar["passed"] is True
    assert scalar["accepted_steps"] == 2
    assert scalar["final_delta"] == pytest.approx(0.02)
    assert aggregate["passed"] is True
    assert aggregate["source_scope"] == "mode21_vmec_boozer_state_multi_point"

    with pytest.raises(ValueError, match="update_step"):
        line_search_gates.vmec_boozer_scalar_objective_line_search_report(
            update_step=0.0,
            _finite_difference_report_fn=_parabola_fd_report,
        )


def test_aggregate_holdout_gate_reuses_training_delta() -> None:
    calls: list[float] = []

    def fake_line_search(**_kwargs: Any) -> dict[str, object]:
        return {
            "passed": True,
            "final_delta": 0.1,
            "initial_objective": 4.0,
            "final_objective": 3.0,
            "relative_reduction": 0.25,
            "samples": [],
        }

    def fake_fd(**kwargs: Any) -> dict[str, object]:
        delta = float(kwargs.get("base_delta", 0.0))
        calls.append(delta)
        return {
            "passed": True,
            "base_value": 2.0 - delta,
            "samples": [{"selected_ky_index": 2}],
        }

    report = line_search_gates.vmec_boozer_aggregate_line_search_holdout_report(
        initial_delta=0.0,
        _line_search_report_fn=fake_line_search,
        _finite_difference_report_fn=fake_fd,
    )

    assert report["passed"] is True
    assert report["heldout_passed"] is True
    assert report["final_delta"] == pytest.approx(0.1)
    assert calls == pytest.approx([0.0, 0.1])


def _one_mode_context(**kwargs: Any) -> dict[str, object]:
    cfg = SimpleNamespace(grid=SimpleNamespace(Nx=1, Ny=4, Nz=int(kwargs["ntheta"])))

    def matrix_fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([[1.0 + 2.0 * x[0] + 1j * (0.5 - x[0])]])

    return {
        "case_name": kwargs["case_name"],
        "cfg": cfg,
        "parameter_names": ("Rcos_mid_m1",),
        "parameter_indices": {"Rcos": [2, 1]},
        "surface_index": kwargs["surface_index"],
        "mboz": int(kwargs["mboz"]),
        "nboz": int(kwargs["nboz"]),
        "surface_stencil_width": kwargs["surface_stencil_width"],
        "n_laguerre": int(kwargs["n_laguerre"]),
        "n_hermite": int(kwargs["n_hermite"]),
        "state_shape": (1,),
        "matrix_fn": matrix_fn,
    }


def _fake_ql_features(
    eigenvalue: jnp.ndarray,
    _eigenvector: jnp.ndarray,
    x: jnp.ndarray,
    _context: dict[str, Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    gamma = jnp.real(eigenvalue)
    omega = jnp.imag(eigenvalue)
    kperp = 2.0 + x[0]
    heat = 3.0 + 2.0 * x[0]
    proxy = gamma * heat / kperp
    return gamma, omega, kperp, heat, proxy


def _fake_window_metrics(
    gamma: jnp.ndarray,
    kperp: jnp.ndarray,
    heat: jnp.ndarray,
    *,
    dt: float,
    steps: int,
    tail_fraction: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    scale = jnp.asarray(float(dt) * int(steps) * float(tail_fraction))
    return gamma * heat * scale, heat / kperp, gamma - kperp


def test_split_gradient_gate_modules_run_injected_reports() -> None:
    frequency = vmec_gradient_gates.mode21_vmec_boozer_linear_frequency_gradient_report(
        case_name="fake",
        fd_step=1.0e-3,
        rtol=2.0e-4,
        atol=2.0e-4,
        _linear_context_fn=_one_mode_context,
    )
    quasilinear = vmec_gradient_gates.mode21_vmec_boozer_quasilinear_gradient_report(
        case_name="fake",
        fd_step=1.0e-3,
        rtol=2.0e-4,
        atol=2.0e-4,
        _linear_context_fn=_one_mode_context,
        _quasilinear_features_fn=_fake_ql_features,
    )
    nonlinear = vmec_gradient_gates.mode21_vmec_boozer_nonlinear_window_gradient_report(
        case_name="fake",
        fd_step=1.0e-3,
        rtol=2.0e-4,
        atol=2.0e-4,
        _linear_context_fn=_one_mode_context,
        _quasilinear_features_fn=_fake_ql_features,
        _window_metrics_fn=_fake_window_metrics,
    )

    assert frequency["passed"] is True
    assert frequency["source_scope"] == "mode21_vmec_boozer_state"
    assert quasilinear["quasilinear_weight_gradient_gate"] is True
    assert nonlinear["nonlinear_window_gradient_gate"] is True


def test_vmec_boozer_gradient_facade_reexports_context_helpers() -> None:
    assert vmec_gradient_gates._mode21_vmec_boozer_linear_context is (
        vmec_gradient_context._mode21_vmec_boozer_linear_context
    )
    assert vmec_gradient_gates._mode21_vmec_boozer_quasilinear_features is (
        vmec_gradient_context._mode21_vmec_boozer_quasilinear_features
    )
