from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gkx.diagnostics import SimulationDiagnostics
from gkx.workflows.nonlinear import (
    _DiagnosticPolicy,
    _RunContext,
    _saturation_stop_condition,
)
from gkx.workflows.runtime.config import RuntimeConfig
from gkx.workflows.runtime.chunks import (
    _effective_diagnostics_stride,
    _next_elapsed_time,
    _offset_chunk_diagnostics_time,
    format_duration,
    run_adaptive_runtime_chunk_loop,
)
from gkx.terms.config import FieldState


def _diag(times: list[float]) -> SimulationDiagnostics:
    t = np.asarray(times, dtype=float)
    zeros = np.zeros_like(t)
    ones = np.ones_like(t)
    return SimulationDiagnostics(
        t=t,
        dt_t=np.full_like(t, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=ones,
        Wphi_t=2.0 * ones,
        Wapar_t=zeros,
        heat_flux_t=zeros,
        particle_flux_t=zeros,
        energy_t=3.0 * ones,
    )


def test_format_duration_compacts_minutes_and_hours() -> None:
    assert format_duration(5.0) == "00:05"
    assert format_duration(65.0) == "01:05"
    assert format_duration(3665.0) == "1:01:05"


def test_adaptive_chunk_time_helpers_lock_accumulated_time_axis() -> None:
    shifted = _offset_chunk_diagnostics_time(_diag([0.25, 0.5]), offset=1.0)

    np.testing.assert_allclose(shifted.t, [1.25, 1.5])
    assert _next_elapsed_time(
        shifted, previous_elapsed=1.0, label="test", chunk_index=2
    ) == pytest.approx(1.5)
    assert _effective_diagnostics_stride(-4) == 1
    assert _effective_diagnostics_stride(0) == 1
    assert _effective_diagnostics_stride(3) == 3


def test_adaptive_chunk_time_helper_rejects_empty_or_stalled_chunks() -> None:
    with pytest.raises(RuntimeError, match="chunk 1 produced no time samples"):
        _next_elapsed_time(_diag([]), previous_elapsed=0.0, label="test", chunk_index=1)

    with pytest.raises(RuntimeError, match="made no time-step progress"):
        _next_elapsed_time(
            _diag([1.0]), previous_elapsed=1.0, label="test", chunk_index=1
        )


def test_run_adaptive_runtime_chunk_loop_reports_wall_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 25.0, 25.0])
    monkeypatch.setattr(
        "gkx.workflows.runtime.chunks.time.perf_counter", lambda: next(clock)
    )

    messages: list[str] = []
    chunks = iter(
        [
            (
                np.asarray([0.5, 1.0]),
                _diag([0.5, 1.0]),
                np.asarray([1.0]),
                FieldState(phi=np.asarray([1.0 + 0.0j])),
            ),
            (
                np.asarray([0.25, 0.5]),
                _diag([0.25, 0.5]),
                np.asarray([2.0]),
                FieldState(phi=np.asarray([2.0 + 0.0j])),
            ),
        ]
    )

    result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=lambda _show_progress, _remaining_time: next(chunks),
        t_max=1.5,
        chunk_steps=16,
        label="nonlinear",
        show_progress=True,
        status_callback=messages.append,
    )

    assert (
        messages[0]
        == "starting adaptive nonlinear integration in chunks of 16 steps up to t_max=1.5"
    )
    assert "progress= 66.7%" in messages[1]
    assert "chunk_wall=00:10" in messages[1]
    assert "elapsed=00:10" in messages[1]
    assert "eta=00:05" in messages[1]
    assert "progress=100.0%" in messages[2]
    assert "eta=00:00" in messages[2]
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.t), np.asarray([0.5, 1.0, 1.25, 1.5])
    )
    np.testing.assert_allclose(np.asarray(result.state), np.asarray([2.0]))
    np.testing.assert_allclose(np.asarray(result.fields.phi), np.asarray([2.0 + 0.0j]))


def test_run_adaptive_runtime_chunk_loop_keeps_exact_terminal_sample_with_stride() -> None:
    chunks = iter(
        [
            (
                np.asarray([0.4, 0.8]),
                _diag([0.4, 0.8]),
                np.asarray([1.0]),
                FieldState(phi=np.asarray([1.0 + 0.0j])),
            ),
            (
                np.asarray([0.4]),
                _diag([0.4]),
                np.asarray([2.0]),
                FieldState(phi=np.asarray([2.0 + 0.0j])),
            ),
        ]
    )
    remaining: list[float] = []

    def integrate_chunk(_show_progress, remaining_time):
        remaining.append(remaining_time)
        return next(chunks)

    result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=integrate_chunk,
        t_max=1.2,
        chunk_steps=8,
        label="test",
        diagnostics_stride=2,
    )

    np.testing.assert_allclose(np.asarray(result.diagnostics.t), [0.4, 1.2])
    np.testing.assert_allclose(remaining, [1.2, 0.4])
    np.testing.assert_allclose(np.asarray(result.state), [2.0])
    np.testing.assert_allclose(np.asarray(result.fields.phi), [2.0 + 0.0j])


def test_run_adaptive_runtime_chunk_loop_rejects_horizon_overshoot() -> None:
    with pytest.raises(RuntimeError, match="must honor remaining_time"):
        run_adaptive_runtime_chunk_loop(
            integrate_chunk=lambda _show_progress, _remaining_time: (
                np.asarray([0.8]),
                _diag([0.8]),
                np.asarray([1.0]),
                FieldState(phi=np.asarray([1.0 + 0.0j])),
            ),
            t_max=1.2,
            chunk_steps=8,
            label="test",
        )


def test_run_adaptive_runtime_chunk_loop_rejects_stalled_time_progress() -> None:
    with pytest.raises(RuntimeError, match="made no time-step progress"):
        run_adaptive_runtime_chunk_loop(
            integrate_chunk=lambda _show_progress, _remaining_time: (
                np.asarray([0.0]),
                _diag([0.0]),
                np.asarray([0.0]),
                FieldState(phi=np.asarray([0.0 + 0.0j])),
            ),
            t_max=1.0,
            chunk_steps=8,
            label="test",
        )


def test_run_adaptive_runtime_chunk_loop_rejects_nonfinite_diagnostics() -> None:
    bad = replace(_diag([0.5]), Wphi_t=np.asarray([np.nan]))

    with pytest.raises(
        RuntimeError, match=r"non-finite diagnostics in Wphi_t at sample 0"
    ):
        run_adaptive_runtime_chunk_loop(
            integrate_chunk=lambda _show_progress, _remaining_time: (
                np.asarray([0.5]),
                bad,
                np.asarray([0.0]),
                FieldState(phi=np.asarray([0.0 + 0.0j])),
            ),
            t_max=1.0,
            chunk_steps=8,
            label="test",
        )


def test_run_adaptive_runtime_chunk_loop_stops_early_on_stop_condition() -> None:
    chunks = iter(
        [
            (
                np.asarray([0.5, 1.0]),
                _diag([0.5, 1.0]),
                np.asarray([1.0]),
                FieldState(phi=np.asarray([1.0 + 0.0j])),
            ),
            (
                np.asarray([0.5, 1.0]),
                _diag([0.5, 1.0]),
                np.asarray([2.0]),
                FieldState(phi=np.asarray([2.0 + 0.0j])),
            ),
            (
                np.asarray([0.5, 1.0]),
                _diag([0.5, 1.0]),
                np.asarray([3.0]),
                FieldState(phi=np.asarray([3.0 + 0.0j])),
            ),
        ]
    )
    seen: list[int] = []

    def stop_condition(t, heat_flux, wphi):
        # The check sees the accumulated unstrided traces on the global axis.
        np.testing.assert_allclose(t, 0.5 + 0.5 * np.arange(t.size))
        assert heat_flux.shape == t.shape and wphi.shape == t.shape
        seen.append(int(t.size))
        return {"stop": t.size >= 4, "saturated": t.size >= 4, "mean": 0.0}

    result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=lambda _show_progress, _remaining_time: next(chunks),
        t_max=3.0,
        chunk_steps=8,
        label="test",
        stop_condition=stop_condition,
    )

    # Stopped after the second chunk, well before t_max.
    assert seen == [2, 4]
    np.testing.assert_allclose(np.asarray(result.diagnostics.t), [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(np.asarray(result.state), [2.0])
    assert result.stop_decision is not None
    assert result.stop_decision["saturated"] is True


def test_run_adaptive_runtime_chunk_loop_reports_last_decision_without_stop() -> None:
    def stop_condition(t, heat_flux, wphi):
        return {
            "stop": False,
            "saturated": False,
            "window_tmax": float(t[-1]),
        }

    result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=lambda _show_progress, _remaining_time: (
            np.asarray([0.5, 1.0]),
            _diag([0.5, 1.0]),
            np.asarray([1.0]),
            FieldState(phi=np.asarray([1.0 + 0.0j])),
        ),
        t_max=1.0,
        chunk_steps=8,
        label="test",
        diagnostics_stride=3,
        stop_condition=stop_condition,
    )

    assert result.stop_decision == {
        "stop": False,
        "saturated": False,
        "window_tmax": 1.0,
    }
    assert result.diagnostics.t[-1] == pytest.approx(
        result.stop_decision["window_tmax"]
    )


def _stop_policy_inputs(*, steps: int, diagnostics_on: bool = True):
    cfg = RuntimeConfig()
    ctx = _RunContext(
        geom=None,
        grid=None,
        params=None,
        terms=None,
        G0=None,
        ky_index=0,
        kx_index=0,
        dt=0.05,
        steps=steps,
        adaptive_chunked=False,
    )
    policy = _DiagnosticPolicy(
        diagnostics_on=diagnostics_on,
        sample_stride=1,
        diagnostics_stride=1,
        laguerre_mode="grid",
        fixed_mode_on=False,
        fixed_ky_index=None,
        fixed_kx_index=None,
        external_phi=None,
        resolved_diagnostics=diagnostics_on,
        return_state=False,
        show_progress=False,
    )
    return cfg, ctx, policy


def test_saturation_stop_condition_defaults_on_for_diagnosed_nonlinear_runs() -> None:
    cfg, ctx, policy = _stop_policy_inputs(steps=4000)

    assert cfg.time.run_to == "saturation"
    assert _saturation_stop_condition(cfg, ctx, policy) is not None


@pytest.mark.parametrize(
    "override",
    [
        {"run_to": "t_max"},
        {"run_to": "SATURATION "},  # normalized, still on -- sanity for the parser
    ],
)
def test_saturation_stop_condition_follows_run_to(override: dict) -> None:
    cfg, ctx, policy = _stop_policy_inputs(steps=4000)
    cfg = replace(cfg, time=replace(cfg.time, **override))

    engaged = _saturation_stop_condition(cfg, ctx, policy) is not None
    assert engaged is (override["run_to"].strip().lower() == "saturation")


def test_saturation_stop_condition_rejects_unknown_run_to() -> None:
    cfg, ctx, policy = _stop_policy_inputs(steps=4000)
    cfg = replace(cfg, time=replace(cfg.time, run_to="forever"))

    with pytest.raises(ValueError, match="run_to"):
        _saturation_stop_condition(cfg, ctx, policy)


def test_saturation_stop_condition_off_without_diagnostics_or_enough_steps() -> None:
    # No heat-flux trace to test, so the run keeps the plain t_max horizon.
    cfg, ctx, policy = _stop_policy_inputs(steps=4000, diagnostics_on=False)
    assert _saturation_stop_condition(cfg, ctx, policy) is None

    # Too few steps to ever reach a decision: stay off the chunked route rather
    # than wrap the same integration in a loop that can only run out of steps.
    cfg, ctx, policy = _stop_policy_inputs(steps=3)
    assert _saturation_stop_condition(cfg, ctx, policy) is None
