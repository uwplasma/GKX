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
from gkx.config import (
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


def _nonlinear_cfg(
    parallel: RuntimeParallelConfig | None = None, *, ky_layout: str = "full"
) -> RuntimeConfig:
    """Return a small periodic nonlinear case with ``Ny = 8``.

    The stored ky extent is 8 on the two-sided axis and ``Nyc = 5`` on the
    ``ky >= 0`` one, which no even device count divides.
    """

    cfg = RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="periodic",
            ky_layout=ky_layout,
        ),
        time=TimeConfig(t_max=0.2, dt=0.01, method="rk2", sample_stride=1),
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


def _live_cfg(
    parallel: RuntimeParallelConfig | None = None, *, ky_layout: str = "full"
) -> RuntimeConfig:
    """Return a deck of the same size whose state and fields actually evolve.

    :func:`_nonlinear_cfg` seeds a single mode that stays put (``phi`` is
    identically zero and the state does not change over the run), so an
    identity on it cannot fail. This one has ``Nx = 4``, a larger box and a
    multimode start at ``1e-2``; over 20 rk2 steps the state moves by about
    ``5e-4`` and ``max |phi|`` is about ``4e-3``.
    """

    cfg = _nonlinear_cfg(parallel, ky_layout=ky_layout)
    return replace(
        cfg,
        grid=replace(cfg.grid, Nx=4, Lx=31.4, Ly=31.4),
        init=replace(cfg.init, init_amp=1.0e-2, init_single=False),
    )


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


@pytest.mark.parametrize("axis", ["kx", "l", "laguerre"])
def test_unsupported_axis_raises(axis: str) -> None:
    with pytest.raises(NonlinearParallelRoutingError, match="axis='ky'"):
        resolve_nonlinear_parallel_plan(
            RuntimeParallelConfig(strategy="shard_map", axis=axis, num_devices=2)
        )


@pytest.mark.parametrize(
    "axis", ["species_hermite", "velocity", "s_m", "species", "m", "hermite"]
)
def test_velocity_axis_aliases_resolve_to_the_production_mesh(axis: str) -> None:
    """Every spelling of the velocity mesh routes to one canonical axis."""

    _require_devices(2)
    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(strategy="shard_map", axis=axis, num_devices=2)
    )
    assert plan is not None
    assert plan.axis == "species_hermite"


def test_auto_selects_the_species_hermite_mesh_and_says_which() -> None:
    """``auto = true`` resolves the mesh from the devices and reports it."""

    _require_devices(2)
    from gkx.workflows.runtime.parallel_nonlinear import resolve_species_hermite_mesh

    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(auto=True, num_devices=2)
    )
    assert plan is not None and plan.axis == "species_hermite" and plan.auto
    resolved = resolve_species_hermite_mesh(
        jnp.zeros((2, 4, 16, 4, 4, 4), dtype=jnp.complex64), plan
    )
    assert resolved.mesh_shape == (2, 1)
    described = resolved.describe()
    assert "2 species x 1 Hermite" in described
    assert "auto-selected" in described
    assert "no halo" in described


def test_auto_conflicting_with_an_explicit_strategy_is_an_error() -> None:
    """``auto`` must not silently overrule an explicit request."""

    with pytest.raises(ValueError, match="conflicts with"):
        RuntimeParallelConfig(auto=True, strategy="batch")
    with pytest.raises(ValueError, match="conflicts with"):
        RuntimeParallelConfig(auto=True, axis="kx")


def test_indivisible_device_count_names_the_counts_that_work() -> None:
    """A rejected mesh reports its alternatives, not only its failure."""

    _require_devices(3)
    from gkx.workflows.runtime.parallel_nonlinear import resolve_species_hermite_mesh

    plan = resolve_nonlinear_parallel_plan(
        RuntimeParallelConfig(auto=True, num_devices=3)
    )
    with pytest.raises(NonlinearParallelRoutingError) as excinfo:
        resolve_species_hermite_mesh(
            jnp.zeros((2, 4, 16, 4, 4, 4), dtype=jnp.complex64), plan
        )
    message = str(excinfo.value)
    assert "supports 1, 2, 4, 8, 16 devices" in message
    assert "divide" in message


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


def test_indivisible_ky_extent_is_placed_replicated_on_the_mesh() -> None:
    """JAX places arrays only in equal shares, so an uneven ky extent -- the
    ``ky >= 0`` axis's odd ``Nyc`` -- enters replicated on the same mesh."""

    _require_devices(2)
    plan = replace(_fake_plan(2), devices=tuple(jax.devices()[:2]))
    state = jnp.arange(12, dtype=jnp.complex64).reshape(1, 1, 3, 1, 4)
    placed = shard_nonlinear_state(state, plan)
    assert placed.sharding.is_fully_replicated
    assert placed.sharding.device_set == set(plan.devices)
    np.testing.assert_array_equal(np.asarray(placed), np.asarray(state))


def test_divisible_ky_extent_is_split_across_the_mesh() -> None:
    _require_devices(2)
    plan = replace(_fake_plan(2), devices=tuple(jax.devices()[:2]))
    state = jnp.zeros((1, 1, 4, 1, 4), dtype=jnp.complex64)
    placed = shard_nonlinear_state(state, plan)
    assert not placed.sharding.is_fully_replicated
    assert {shard.data.shape[-3] for shard in placed.addressable_shards} == {2}


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


@pytest.mark.parametrize("ky_layout", ["full", "half"])
@pytest.mark.parametrize("num_devices", [2, 4])
def test_shard_map_nonlinear_run_matches_serial_diagnostics(
    num_devices: int, ky_layout: str
) -> None:
    """A sharded nonlinear run reproduces the serial run's diagnostics.

    On ``"half"`` the stored extent is ``Nyc = 5``, which neither device count
    divides; the run is accepted rather than refused (SHARD-PAD, plan F.6).
    """

    _require_devices(num_devices)
    serial = run_runtime_nonlinear(
        _live_cfg(ky_layout=ky_layout), return_state=True, **_RUN_KWARGS
    )
    sharded = run_runtime_nonlinear(
        _live_cfg(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=num_devices
            ),
            ky_layout=ky_layout,
        ),
        return_state=True,
        **_RUN_KWARGS,
    )
    assert np.asarray(sharded.state).shape[-3] == (8 if ky_layout == "full" else 5)
    assert serial.diagnostics is not None and sharded.diagnostics is not None
    # The live deck's field energy is nonzero, so the comparison can fail.
    assert np.max(np.abs(np.asarray(serial.diagnostics.Wphi_t))) > 0.0
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


def test_ky_route_scan_runs_replicated(monkeypatch) -> None:
    """Pin the documented limit of the ``ky`` route: its scan is not split.

    ``docs/parallelization.rst`` says the scan's input is replicated on every
    device even when the ky extent divides the device count. If a change makes
    the route partition for real, this fails, and that paragraph -- and the
    claim that an uneven extent loses nothing by entering replicated -- must be
    rewritten with it.
    """

    import gkx.solvers_nonlinear_diagnostics as diagnostics

    _require_devices(2)
    seen: list[object] = []
    original = diagnostics._run_explicit_diagnostic_scan_and_finalize

    def spy(prepared, *args, **kwargs):
        seen.append(prepared.G0.sharding)
        return original(prepared, *args, **kwargs)

    monkeypatch.setattr(diagnostics, "_run_explicit_diagnostic_scan_and_finalize", spy)
    run_runtime_nonlinear(
        _nonlinear_cfg(
            RuntimeParallelConfig(
                strategy="shard_map", axis="ky", num_devices=2, strict_identity=False
            )
        ),
        **_RUN_KWARGS,
    )
    assert len(seen) == 1
    assert seen[0].is_fully_replicated and len(seen[0].device_set) == 2


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


# ---- [time] state_sharding: real placement, results read back ----


def _live_context(monkeypatch, ky_layout: str):
    """Return the runtime context (initial state, grid, geometry, ...) of the live deck."""

    captured: dict = {}
    original = nonlinear_workflow._run_once

    def spy(cfg, ctx, *args, **kwargs):
        captured["ctx"] = ctx
        return original(cfg, ctx, *args, **kwargs)

    monkeypatch.setattr(nonlinear_workflow, "_run_once", spy)
    cfg = _live_cfg(ky_layout=ky_layout)
    run_runtime_nonlinear(cfg, **{**_RUN_KWARGS, "steps": 1})
    monkeypatch.setattr(nonlinear_workflow, "_run_once", original)
    return cfg, captured["ctx"]


@pytest.mark.parametrize(
    ("ky_layout", "spec", "num_devices"),
    [
        ("full", "ky", 2),
        ("full", "ky", 4),
        ("half", "ky", 2),
        ("half", "ky", 4),
        ("full", "kx", 4),
    ],
)
def test_time_state_sharding_matches_serial(
    monkeypatch, ky_layout: str, spec: str, num_devices: int
) -> None:
    """``[time] state_sharding`` runs on real devices and reproduces the serial run.

    Before the fix, the two-sided ky split failed on XLA:CPU with the FFT
    thunk's layout RET_CHECK, and the half layout (``Nyc = 5``) raised
    ``IndivisibleError`` from ``device_put``. The failure only surfaced when
    the result was read, so the test reads both outputs back.
    """

    import gkx.solvers_time_runners as runners
    from gkx.parallel.state import resolve_state_sharding

    _require_devices(num_devices)
    cfg, ctx = _live_context(monkeypatch, ky_layout)
    devices = jax.devices()[:num_devices]
    monkeypatch.setattr(
        runners,
        "resolve_state_sharding",
        lambda G0, name: resolve_state_sharding(G0, name, devices=devices),
    )

    def run(sharding: str | None):
        time_cfg = replace(
            cfg.time, dt=ctx.dt, t_max=20 * ctx.dt, state_sharding=sharding
        )
        return runners.integrate_nonlinear_from_config(
            jnp.array(ctx.G0, copy=True),
            ctx.grid,
            ctx.geom,
            ctx.params,
            time_cfg,
            terms=ctx.terms,
        )

    serial_state, serial_fields = run(None)
    state, fields = run(spec)
    serial_state = np.asarray(serial_state)
    assert np.asarray(state).shape == serial_state.shape
    assert serial_state.shape[-3] == (8 if ky_layout == "full" else 5)
    # The deck must be live, or the comparison below proves nothing.
    assert np.max(np.abs(serial_state - np.asarray(ctx.G0))) > 1.0e-5
    assert np.max(np.abs(np.asarray(serial_fields.phi))) > 1.0e-4
    np.testing.assert_allclose(np.asarray(state), serial_state, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(fields.phi), np.asarray(serial_fields.phi), rtol=0.0, atol=1.0e-12
    )
