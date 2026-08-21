from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.geometry.differentiable import observable_gradient_validation_report


def _x64_enabled() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def test_observable_gradient_validation_report_passes_with_conditioning_metadata() -> (
    None
):
    dtype = jnp.float64 if _x64_enabled() else jnp.float32
    fd_step = 2.0e-5 if _x64_enabled() else 1.0e-3
    atol = 1.0e-7 if _x64_enabled() else 5.0e-4
    rtol = 5.0e-5 if _x64_enabled() else 5.0e-3
    params = jnp.asarray([0.24, -0.37], dtype=dtype)

    def observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [
                x[0] * x[0] + 2.0 * x[1],
                jnp.sin(x[0]) + x[1] * x[1],
            ]
        )

    report = observable_gradient_validation_report(
        observables,
        params,
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
        observable_names=("quadratic_drive", "mixed_wave"),
        param_names=("ripple", "shear"),
        tangent=jnp.asarray([0.6, -0.8], dtype=dtype),
    )

    assert report["passed"] is True
    assert report["finite_passed"] is True
    assert report["derivative_tolerance_passed"] is True
    assert report["tangent_tolerance_passed"] is True
    assert report["conditioning_passed"] is True
    assert report["finite_flags"]["autodiff_jacobian"] is True
    assert report["finite_flags"]["finite_difference_jacobian"] is True
    assert report["conditioning"]["jacobian_shape"] == [2, 2]
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert np.isfinite(float(report["conditioning"]["jacobian_condition_number"]))
    assert report["conditioning_gate"]["rank_passed"] is True
    assert report["tangent_direction_norm"] == pytest.approx(
        np.linalg.norm([0.6, -0.8])
    )
    assert np.isfinite(float(report["tangent_ad_norm"]))
    assert len(report["gradient_checks"]) == 4
    assert all(row["passed"] for row in report["gradient_checks"])
    json.dumps(report, allow_nan=False)


def test_observable_gradient_chunking_preserves_jacobian_and_records_policy() -> None:
    dtype = jnp.float64 if _x64_enabled() else jnp.float32
    params = jnp.linspace(0.1, 0.6, 6, dtype=dtype)

    def observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            [jnp.sum(jnp.sin(x)), jnp.dot(x, x), jnp.prod(1.0 + 0.1 * x)]
        )

    common = {
        "fd_step": 2.0e-4 if _x64_enabled() else 2.0e-3,
        "rtol": 2.0e-3 if _x64_enabled() else 2.0e-2,
        "atol": 2.0e-5 if _x64_enabled() else 2.0e-3,
        "condition_number_max": None,
    }
    full = observable_gradient_validation_report(
        observables, params, jacobian_chunk_size=None, **common
    )
    chunked = observable_gradient_validation_report(
        observables, params, jacobian_chunk_size=2, **common
    )

    assert full["jacobian_chunk_size"] is None
    assert chunked["jacobian_chunk_size"] == 2
    assert full["jacobian_mode"] == "reverse"
    assert chunked["jacobian_mode"] == "forward"
    assert full["passed"] is True and chunked["passed"] is True
    np.testing.assert_allclose(
        np.asarray(chunked["jacobian_ad"]),
        np.asarray(full["jacobian_ad"]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

    forward = observable_gradient_validation_report(
        observables, params, jacobian_mode="forward", **common
    )
    reverse = observable_gradient_validation_report(
        observables, params, jacobian_mode="reverse", **common
    )
    assert forward["jacobian_mode"] == "forward"
    assert reverse["jacobian_mode"] == "reverse"
    np.testing.assert_allclose(
        np.asarray(forward["jacobian_ad"]),
        np.asarray(reverse["jacobian_ad"]),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_observable_gradient_validation_rejects_invalid_mode_policy() -> None:
    def fn(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([jnp.sum(x)])

    params = jnp.ones(3)
    with pytest.raises(ValueError, match="jacobian_mode"):
        observable_gradient_validation_report(fn, params, jacobian_mode="adjoint")
    with pytest.raises(ValueError, match="only valid for forward"):
        observable_gradient_validation_report(
            fn, params, jacobian_mode="reverse", jacobian_chunk_size=1
        )


def test_observable_gradient_validation_report_fails_strict_json_on_nonfinite_data() -> (
    None
):
    def nonfinite_observables(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([x[0] / 0.0, x[1] * x[1]])

    report = observable_gradient_validation_report(
        nonfinite_observables,
        jnp.asarray([1.0, 0.2]),
        fd_step=1.0e-3,
        rtol=1.0e-3,
        atol=1.0e-5,
        observable_names=("bad", "finite"),
        param_names=("p0", "p1"),
    )

    assert report["passed"] is False
    assert report["finite_passed"] is False
    assert report["finite_flags"]["observables"] is False
    assert report["finite_flags"]["autodiff_jacobian"] is False
    assert any(
        str(reason).startswith("nonfinite:") for reason in report["failure_reasons"]
    )
    assert report["conditioning"]["finite_ad_jacobian"] is False
    assert report["conditioning_gate"]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_observable_gradient_validation_report_fails_ill_conditioned_synthetic_map() -> (
    None
):
    def rank_deficient_observables(x: jnp.ndarray) -> jnp.ndarray:
        shared = x[0] + x[1]
        return jnp.asarray([shared, 2.0 * shared])

    report = observable_gradient_validation_report(
        rank_deficient_observables,
        jnp.asarray([0.1, -0.2]),
        fd_step=1.0e-3,
        rtol=1.0e-3,
        atol=1.0e-5,
        observable_names=("row0", "row1"),
        param_names=("p0", "p1"),
        min_rank=2,
        condition_number_max=1.0e6,
    )

    assert report["finite_passed"] is True
    assert report["derivative_tolerance_passed"] is True
    assert report["passed"] is False
    assert report["conditioning"]["sensitivity_map_rank"] == 1
    assert report["conditioning_gate"]["rank_passed"] is False
    assert report["conditioning_gate"]["condition_number_passed"] is False
    assert "rank_below_required" in report["failure_reasons"]
    assert "ill_conditioned" in report["failure_reasons"]
    json.dumps(report, allow_nan=False)
