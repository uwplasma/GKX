"""Unit contracts: routing ``[parallel]`` into the nonlinear solver path.

The multi-device cases need more than one JAX device. Run them with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``; the wide-coverage
runner supplies that through ``WIDE_COVERAGE_LOGICAL_CPU_DEVICES``. Without it
they skip rather than silently passing on one device.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gkx.workflows.nonlinear as nonlinear_workflow
from gkx.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from gkx.runtime import run_runtime_nonlinear
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from gkx.workflows.runtime.parallel_nonlinear import (
    NonlinearParallelIdentityError,
    NonlinearParallelPlan,
    NonlinearParallelRoutingError,
    assert_nonlinear_parallel_identity,
    resolve_nonlinear_parallel_plan,
    shard_nonlinear_state,
)

_RUN_KWARGS: dict[str, object] = {
    "ky_target": 0.2,
    "Nl": 3,
    "Nm": 4,
    "dt": 0.01,
    "steps": 3,
    "sample_stride": 1,
}


def _nonlinear_cfg(parallel: RuntimeParallelConfig | None = None) -> RuntimeConfig:
    """Return a small periodic nonlinear case with a ky extent of eight."""

    cfg = RuntimeConfig(
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(
            t_max=0.2, dt=0.01, method="rk2", use_diffrax=False, sample_stride=1
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    return cfg if parallel is None else replace(cfg, parallel=parallel)


def _fake_plan(
    count: int = 2, *, strict_identity: bool = True
) -> NonlinearParallelPlan:
    """Return a plan with placeholder devices for host-only contract checks."""

    return NonlinearParallelPlan(
        axis="ky",
        devices=tuple(object() for _ in range(count)),
        strict_identity=strict_identity,
    )


def _require_devices(count: int) -> None:
    if len(jax.devices()) < count:
        pytest.skip(f"requires {count} logical CPU devices or accelerators")


# ---- plan resolution: nothing is accepted and then ignored ----


def test_serial_strategy_resolves_to_no_plan() -> None:
    assert resolve_nonlinear_parallel_plan(RuntimeParallelConfig()) is None
    assert resolve_nonlinear_parallel_plan(None) is None


@pytest.mark.parametrize(
    "strategy",
    ["batch", "combined_ky", "device_batch", "pmap", "pjit", "state", "velocity"],
)
def test_unsupported_strategy_raises_instead_of_running_serial(strategy: str) -> None:
    """A strategy with no nonlinear route must fail, not quietly run serially."""

    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy=strategy, axis="ky", num_devices=2)
        )
    message = str(excinfo.value)
    assert strategy in message
    assert "shard_map" in message and "axis='ky'" in message


def test_z_axis_is_rejected_with_the_measured_reason() -> None:
    """axis='z' names both blockers rather than pretending to be routable."""

    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy="shard_map", axis="z", num_devices=2)
        )
    message = str(excinfo.value)
    assert "spectral FFT along z" in message
    assert "device_z" in message
    assert "axis='ky'" in message


@pytest.mark.parametrize("axis", ["kx", "species", "hermite", "l", "m"])
def test_unsupported_axis_raises(axis: str) -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="axis='ky'"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy="shard_map", axis=axis, num_devices=2)
        )


def test_independent_worker_options_are_rejected() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="worker pool"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, backend="thread"
            )
        )
    with pytest.raises(NonlinearParallelRoutingError, match="batch_size"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, batch_size=2
            )
        )


def test_unsatisfiable_device_request_raises_a_routing_error() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="num_devices"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=len(jax.devices()) + 8
            )
        )


def test_plan_honours_num_devices() -> None:
    _require_devices(2)
    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
    )
    assert plan is not None
    assert plan.device_count == 2
    assert plan.strict_identity is True
    assert "axis='ky'" in plan.describe()


def test_indivisible_ky_extent_raises() -> None:
    state = jnp.zeros((1, 1, 3, 1, 4), dtype=jnp.complex64)
    with pytest.raises(NonlinearParallelRoutingError, match="not.*divisible"):
        shard_nonlinear_state(state, _fake_plan(2))


# ---- fail-closed identity ----


def test_identity_gate_accepts_an_exact_match() -> None:
    state = jnp.ones((2, 2), dtype=jnp.complex64)
    assert_nonlinear_parallel_identity(
        serial_state=state, sharded_state=state, plan=_fake_plan()
    )


def test_identity_gate_raises_on_a_perturbed_state() -> None:
    serial = jnp.ones((2, 2), dtype=jnp.complex64)
    with pytest.raises(NonlinearParallelIdentityError) as excinfo:
        assert_nonlinear_parallel_identity(
            serial_state=serial,
            sharded_state=serial * 1.5,
            plan=_fake_plan(),
        )
    message = str(excinfo.value)
    assert "final_state" in message
    assert "discarded" in message


def test_identity_gate_raises_on_a_perturbed_diagnostic_trace() -> None:
    """A route that reproduces the state but not the traces still fails."""

    state = jnp.ones((2, 2), dtype=jnp.complex64)
    serial_diag = type("Diag", (), {"Wg_t": np.ones(4), "heat_flux_t": np.ones(4)})()
    sharded_diag = type(
        "Diag", (), {"Wg_t": np.ones(4), "heat_flux_t": np.full(4, 2.0)}
    )()
    with pytest.raises(NonlinearParallelIdentityError, match="heat_flux_t"):
        assert_nonlinear_parallel_identity(
            serial_state=state,
            sharded_state=state,
            serial_diagnostics=serial_diag,
            sharded_diagnostics=sharded_diag,
            plan=_fake_plan(),
        )


def test_identity_gate_reports_a_shape_mismatch_as_a_failure() -> None:
    with pytest.raises(NonlinearParallelIdentityError):
        assert_nonlinear_parallel_identity(
            serial_state=jnp.ones((2, 2), dtype=jnp.complex64),
            sharded_state=jnp.ones((2, 3), dtype=jnp.complex64),
            plan=_fake_plan(),
        )


# ---- end-to-end runtime routing ----


@pytest.mark.parametrize("num_devices", [2, 4])
def test_shard_map_nonlinear_run_matches_serial_diagnostics(num_devices: int) -> None:
    """A sharded nonlinear run reproduces the serial run's diagnostics."""

    _require_devices(num_devices)
    serial = run_runtime_nonlinear(_nonlinear_cfg(), return_state=True, **_RUN_KWARGS)
    sharded = run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=num_devices
            )
        ),
        return_state=True,
        **_RUN_KWARGS,
    )
    assert serial.diagnostics is not None and sharded.diagnostics is not None
    np.testing.assert_allclose(
        np.asarray(sharded.state), np.asarray(serial.state), rtol=0.0, atol=5.0e-6
    )
    for name in ("Wg_t", "Wphi_t", "heat_flux_t", "particle_flux_t"):
        np.testing.assert_allclose(
            np.asarray(getattr(sharded.diagnostics, name)),
            np.asarray(getattr(serial.diagnostics, name)),
            rtol=1.0e-4,
            atol=5.0e-6,
        )


def test_shard_map_nonlinear_run_reports_the_route() -> None:
    _require_devices(2)
    messages: list[str] = []
    run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
        ),
        status_callback=messages.append,
        **_RUN_KWARGS,
    )
    assert any("routing nonlinear run through" in message for message in messages)
    assert any("identity gate passed" in message for message in messages)


def _inject_shard_perturbation(monkeypatch) -> None:
    """Offset the sharded initial state far above the identity tolerance.

    The offset has to beat ``atol`` in absolute terms: the gate is an
    ``abs or rel`` test, so scaling a ``1e-8`` initial amplitude would stay
    inside the absolute tolerance and prove nothing.
    """

    original = nonlinear_workflow.shard_nonlinear_state

    def perturbed(state, plan):
        sharded = original(state, plan)
        return sharded + jnp.asarray(1.0e-3, dtype=sharded.dtype)

    monkeypatch.setattr(nonlinear_workflow, "shard_nonlinear_state", perturbed)


def test_strict_identity_raises_when_the_sharded_answer_drifts(monkeypatch) -> None:
    """Injecting a perturbation into the sharded state must fail the run."""

    _require_devices(2)
    _inject_shard_perturbation(monkeypatch)
    with pytest.raises(NonlinearParallelIdentityError, match="final_state"):
        run_runtime_nonlinear(
            _nonlinear_cfg(
                RuntimeParallelConfig(strategy="shard_map", axis="ky", num_devices=2)
            ),
            **_RUN_KWARGS,
        )


def test_strict_identity_false_skips_the_serial_reference(monkeypatch) -> None:
    """strict_identity is the only thing standing between the two behaviours."""

    _require_devices(2)
    _inject_shard_perturbation(monkeypatch)
    messages: list[str] = []
    result = run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map",
                axis="ky",
                num_devices=2,
                strict_identity=False,
            )
        ),
        status_callback=messages.append,
        **_RUN_KWARGS,
    )
    assert result.diagnostics is not None
    assert any("skipping the serial identity gate" in message for message in messages)


def test_unsupported_strategy_fails_the_runtime_run_before_any_work() -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="shard_map"):
        run_runtime_nonlinear(
            _nonlinear_cfg(RuntimeParallelConfig(strategy="batch", axis="ky")),
            **_RUN_KWARGS,
        )
