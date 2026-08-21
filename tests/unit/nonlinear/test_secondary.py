"""Tests for staged secondary-instability helpers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from gkx.config import GridConfig, InitializationConfig, TimeConfig
from gkx.diagnostics import SimulationDiagnostics
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from gkx.workflows.nonlinear import (
    _embed_linear_seed_on_full_grid,
    _leading_finite_prefix,
    _tail_mean_pair,
    build_secondary_stage2_config,
    run_secondary_modes,
    run_secondary_seed,
    write_restart_state,
)


def _base_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        species=(RuntimeSpeciesConfig(name="ion"),),
        grid=GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=125.66370614359172,
            Ly=62.83185307179586,
            boundary="periodic",
            y0=10.0,
        ),
        time=TimeConfig(t_max=2.0, dt=1.0, method="sspx3", use_diffrax=False),
        init=InitializationConfig(
            init_field="density", init_amp=1.0, gaussian_init=False, init_single=True
        ),
    )


def test_build_secondary_stage2_config_sets_restart_controls(tmp_path) -> None:
    cfg = _base_cfg()
    out = build_secondary_stage2_config(cfg, restart_file=tmp_path / "seed.bin")
    assert out.physics == RuntimePhysicsConfig(linear=False, nonlinear=True)
    assert out.terms == RuntimeTermsConfig(nonlinear=1.0)
    assert out.expert == RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0)
    assert out.init.init_file == str(tmp_path / "seed.bin")
    assert out.init.init_file_scale == 500.0
    assert out.init.init_file_mode == "add"
    assert out.init.init_single is False
    assert out.time.method == "sspx3"
    assert out.time.dt == 0.01


def test_write_restart_state_roundtrip(tmp_path) -> None:
    state = (
        np.arange(12, dtype=np.float32) + 1j * np.arange(12, dtype=np.float32)
    ).astype(np.complex64)
    path = write_restart_state(tmp_path / "restart.bin", state)
    restored = np.fromfile(path, dtype=np.complex64)
    assert np.allclose(restored, state)


def test_run_secondary_modes_uses_requested_targets(monkeypatch) -> None:
    captured: list[tuple[float, float]] = []

    class _Result:
        def __init__(self) -> None:
            t = np.array([0.1], dtype=float)
            self.diagnostics = SimulationDiagnostics(
                t=t,
                dt_t=t,
                dt_mean=t[0],
                gamma_t=np.array([1.5]),
                omega_t=np.array([-0.25]),
                Wg_t=t * 0.0,
                Wphi_t=t * 0.0,
                Wapar_t=t * 0.0,
                heat_flux_t=t * 0.0,
                particle_flux_t=t * 0.0,
                energy_t=t * 0.0,
                phi_mode_t=np.array([1.0 + 0.0j]),
            )

    def _fake_runner(*args, **kwargs):
        captured.append((float(kwargs["ky_target"]), float(kwargs["kx_target"])))
        return _Result()

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_nonlinear", _fake_runner
    )
    rows = run_secondary_modes(
        _base_cfg(), modes=((0.0, -0.05), (0.1, 0.05)), Nl=3, Nm=8
    )
    assert captured == [(0.0, -0.05), (0.1, 0.05)]
    assert rows[0].gamma == 1.5
    assert rows[1].omega == -0.25


def test_tail_mean_pair_averages_late_time_window() -> None:
    gamma, omega = _tail_mean_pair(
        np.array([1.0, 2.0, 3.0, 5.0]),
        np.array([0.0, 0.5, 1.0, 1.5]),
        tail_fraction=0.5,
    )
    assert gamma == pytest.approx(4.0)
    assert omega == pytest.approx(1.25)


def test_tail_mean_pair_handles_none_and_last_sample() -> None:
    assert (
        _tail_mean_pair(np.array([np.nan]), np.array([np.nan]), tail_fraction=0.5)
        is None
    )
    gamma, omega = _tail_mean_pair(
        np.array([1.0, 2.0]), np.array([-1.0, -2.0]), tail_fraction=None
    )
    assert gamma == pytest.approx(2.0)
    assert omega == pytest.approx(-2.0)


def test_run_secondary_modes_uses_phi_fit_for_gamma_and_tail_for_omega(
    monkeypatch,
) -> None:
    class _Result:
        def __init__(self) -> None:
            t = np.array([0.0, 1.0, 2.0], dtype=float)
            signal = np.exp((2.0 - 0.25j) * t)
            self.diagnostics = SimulationDiagnostics(
                t=t,
                dt_t=np.full_like(t, 1.0),
                dt_mean=1.0,
                gamma_t=np.array([1.0, 2.0, 3.0]),
                omega_t=np.array([-0.1, -0.2, -0.3]),
                Wg_t=t * 0.0,
                Wphi_t=t * 0.0,
                Wapar_t=t * 0.0,
                heat_flux_t=t * 0.0,
                particle_flux_t=t * 0.0,
                energy_t=t * 0.0,
                phi_mode_t=signal,
            )

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_nonlinear",
        lambda *args, **kwargs: _Result(),
    )
    row = run_secondary_modes(
        _base_cfg(), modes=((0.1, 0.05),), Nl=3, Nm=8, fit_fraction=1.0
    )[0]
    assert row.gamma == pytest.approx(2.0)
    assert row.omega == pytest.approx(-0.2)


def test_run_secondary_modes_fits_mode_trace_when_diagnostics_invalid(
    monkeypatch,
) -> None:
    class _Result:
        def __init__(self) -> None:
            t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
            signal = np.exp((1.5 - 0.125j) * t).astype(np.complex128)
            signal[-1] = np.nan + 1j * np.nan
            self.diagnostics = SimulationDiagnostics(
                t=t,
                dt_t=np.full_like(t, 1.0),
                dt_mean=1.0,
                gamma_t=np.array([np.nan, np.nan, np.nan, np.nan]),
                omega_t=np.array([np.nan, np.nan, np.nan, np.nan]),
                Wg_t=t * 0.0,
                Wphi_t=t * 0.0,
                Wapar_t=t * 0.0,
                heat_flux_t=t * 0.0,
                particle_flux_t=t * 0.0,
                energy_t=t * 0.0,
                phi_mode_t=signal,
            )

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_nonlinear",
        lambda *args, **kwargs: _Result(),
    )
    row = run_secondary_modes(
        _base_cfg(), modes=((0.1, 0.05),), Nl=3, Nm=8, fit_fraction=1.0
    )[0]
    assert row.gamma == pytest.approx(1.5)
    assert row.omega == pytest.approx(0.125)


def test_leading_finite_prefix_stops_before_first_nan() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    sig = np.array(
        [1.0 + 0.0j, 2.0 + 0.0j, np.nan + 0.0j, 4.0 + 0.0j], dtype=np.complex128
    )
    t_out, sig_out = _leading_finite_prefix(t, sig)
    assert np.allclose(t_out, np.array([0.0, 1.0]))
    assert np.allclose(sig_out, np.array([1.0 + 0.0j, 2.0 + 0.0j]))


def test_leading_finite_prefix_all_invalid_returns_empty() -> None:
    t = np.array([0.0, 1.0], dtype=float)
    sig = np.array([np.nan + 0.0j, np.nan + 0.0j], dtype=np.complex128)
    t_out, sig_out = _leading_finite_prefix(t, sig)
    assert t_out.size == 0
    assert sig_out.size == 0


def test_embed_linear_seed_on_full_grid_passthrough_and_invalid_shape() -> None:
    cfg = _base_cfg()
    full_shape = (1, 3, 8, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    full_state = np.ones(full_shape, dtype=np.complex64)
    embedded = _embed_linear_seed_on_full_grid(cfg, full_state, ky_target=0.1)
    assert embedded.shape == full_shape
    assert np.allclose(embedded, full_state)

    with pytest.raises(ValueError):
        _embed_linear_seed_on_full_grid(
            cfg, np.ones((1, 3, 8, 2, cfg.grid.Nz), dtype=np.complex64), ky_target=0.1
        )


def test_run_secondary_seed_requires_final_state(monkeypatch, tmp_path: Path) -> None:
    class _Result:
        state = None

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_linear",
        lambda *args, **kwargs: _Result(),
    )
    with pytest.raises(RuntimeError):
        run_secondary_seed(
            _base_cfg(), restart_path=tmp_path / "seed.bin", ky_target=0.1, Nl=3, Nm=8
        )


def test_run_secondary_modes_requires_diagnostics(monkeypatch) -> None:
    class _Result:
        diagnostics = None

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_nonlinear",
        lambda *args, **kwargs: _Result(),
    )
    with pytest.raises(RuntimeError):
        run_secondary_modes(_base_cfg(), modes=((0.1, 0.05),), Nl=3, Nm=8)


def test_run_secondary_modes_uses_tail_when_phi_mode_missing(monkeypatch) -> None:
    class _Result:
        def __init__(self) -> None:
            t = np.array([0.0, 1.0, 2.0], dtype=float)
            self.diagnostics = SimulationDiagnostics(
                t=t,
                dt_t=np.full_like(t, 1.0),
                dt_mean=1.0,
                gamma_t=np.array([0.5, 0.7, 0.9]),
                omega_t=np.array([-0.1, -0.2, -0.3]),
                Wg_t=t * 0.0,
                Wphi_t=t * 0.0,
                Wapar_t=t * 0.0,
                heat_flux_t=t * 0.0,
                particle_flux_t=t * 0.0,
                energy_t=t * 0.0,
                phi_mode_t=None,
            )

    monkeypatch.setattr(
        "gkx.workflows.nonlinear._run_runtime_nonlinear",
        lambda *args, **kwargs: _Result(),
    )
    row = run_secondary_modes(
        _base_cfg(), modes=((0.1, 0.05),), Nl=3, Nm=8, fit_fraction=None
    )[0]
    assert row.gamma == pytest.approx(0.9)
    assert row.omega == pytest.approx(-0.3)


def test_secondary_stage2_sideband_grows_on_short_window() -> None:
    cfg = _base_cfg()
    with TemporaryDirectory() as tmpdir:
        restart = Path(tmpdir) / "seed.bin"
        run_secondary_seed(cfg, restart_path=restart, ky_target=0.1, Nl=3, Nm=8)
        stage2 = build_secondary_stage2_config(cfg, restart_file=restart, t_max=1.0)
        row = run_secondary_modes(
            stage2, modes=((0.1, 0.05),), Nl=3, Nm=8, steps=100, sample_stride=10
        )[0]

    assert row.gamma > 1.0
