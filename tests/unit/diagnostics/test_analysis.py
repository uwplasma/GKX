"""Analysis helper tests for mode extraction and fit windows."""

import numpy as np
import pytest

from gkx.diagnostics.analysis import (
    ModeSelection,
    _log_amp_phase,
    density_moment,
    extract_mode,
    extract_eigenfunction,
    extract_mode_time_series,
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    fit_growth_rate_uncertainty,
    fit_growth_rate_with_stats,
    instantaneous_growth_rate_from_phi,
    windowed_growth_rate_from_omega_series,
    select_ky_index,
    select_fit_window,
    select_fit_window_loglinear,
    select_fit_window_stationary,
)
import json
from gkx.diagnostics import SimulationDiagnostics
from gkx.diagnostics.analysis import (
    CFL_TERM_NAMES,
    CFL_TERM_UNRESOLVED,
    CFLScales,
    cfl_limiter_report,
    cfl_limiting_term,
    cfl_scales_from_array,
    cfl_term_contributions,
)
from gkx.diagnostics.metadata import CFL_SCALE_LABELS
from gkx.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics,
    slice_runtime_diagnostics,
    stride_runtime_diagnostics,
    timestep_cost_payload,
    timestep_cost_report,
)
from dataclasses import replace
import matplotlib


def test_growth_rate_public_facades_point_to_numerical_owners() -> None:
    """Growth diagnostics keep stable imports with one owner per algorithm."""

    import gkx.diagnostics.analysis as analysis
    import gkx.diagnostics.growth_rates as growth_rates
    import gkx.diagnostics.growth_windows as growth_windows

    assert growth_rates.select_fit_window is growth_windows.select_fit_window
    assert (
        growth_rates.select_fit_window_loglinear
        is growth_windows.select_fit_window_loglinear
    )
    assert growth_rates.instantaneous_growth_rate_from_phi.__module__ == (
        "gkx.diagnostics.growth_rates"
    )
    assert growth_rates.windowed_growth_rate_from_omega_series.__module__ == (
        "gkx.diagnostics.growth_rates"
    )
    assert analysis.fit_growth_rate is growth_rates.fit_growth_rate
    assert analysis.fit_growth_rate_auto is growth_rates.fit_growth_rate_auto


def test_extract_mode_time_series_methods():
    """Mode extraction should work for z_index, max, and svd modes."""
    t = np.linspace(0.0, 1.0, 64)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
    spatial = np.linspace(1.0, 2.0, 4)
    data = ts[:, None] * spatial[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=1)

    z_series = extract_mode_time_series(phi_t, sel, method="z_index")
    assert np.allclose(z_series, ts * spatial[1])

    max_series = extract_mode_time_series(phi_t, sel, method="max")
    assert np.allclose(max_series, ts * spatial[-1])

    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    ratio = svd_series / ts
    ratio_norm = ratio / ratio[0]
    assert np.allclose(ratio_norm, np.ones_like(ratio_norm), atol=1.0e-6)

    project_series = extract_mode_time_series(phi_t, sel, method="project")
    proj_ratio = project_series / ts
    proj_norm = proj_ratio / proj_ratio[0]
    assert np.allclose(proj_norm, np.ones_like(proj_norm), atol=1.0e-6)

    direct = extract_mode(phi_t, sel)
    assert np.allclose(direct, ts * spatial[1])

    try:
        extract_mode_time_series(phi_t, sel, method="bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mode method should raise ValueError")


def test_extract_mode_svd_fallback_nan():
    """SVD mode extraction should fall back when NaNs are present."""
    t = np.linspace(0.0, 1.0, 16)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
    data = ts[:, None] * np.array([1.0, np.nan])[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    proj_series = extract_mode_time_series(phi_t, sel, method="project")
    assert np.allclose(svd_series, proj_series, equal_nan=True)


def test_extract_mode_svd_fallback_linalg(monkeypatch):
    """SVD mode extraction should fall back on LinAlgError."""
    t = np.linspace(0.0, 1.0, 16)
    gamma = 0.1
    omega = 0.2
    ts = np.exp((gamma - 1j * omega) * t)
    data = ts[:, None] * np.linspace(1.0, 2.0, 4)[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    def _bad_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "svd", _bad_svd)
    svd_series = extract_mode_time_series(phi_t, sel, method="svd")
    proj_series = extract_mode_time_series(phi_t, sel, method="project")
    assert np.allclose(svd_series, proj_series)


def test_select_fit_window_and_auto_fit():
    """Auto window should favor the clean exponential region."""
    t = np.linspace(0.0, 10.0, 200)
    gamma = 0.12
    omega = 0.3
    signal = np.exp((gamma - 1j * omega) * t)
    signal = signal.copy()
    signal[:40] *= 1.0 + 0.2 * np.sin(5.0 * t[:40])

    tmin, tmax = select_fit_window(t, signal, window_fraction=0.3, min_points=20)
    assert tmax > tmin
    assert tmin >= t[20]

    tmin2, tmax2 = select_fit_window(
        t,
        signal,
        window_fraction=0.3,
        min_points=20,
        start_fraction=0.5,
        growth_weight=1.0,
        require_positive=True,
    )
    assert tmax2 > tmin2
    assert tmin2 >= t[int(0.5 * t.shape[0])]

    g_fit, w_fit, _tmin, _tmax = fit_growth_rate_auto(
        t, signal, window_fraction=0.3, min_points=20
    )
    assert np.isclose(g_fit, gamma, rtol=1e-2, atol=1e-2)
    assert np.isclose(w_fit, omega, rtol=1e-2, atol=1e-2)

    try:
        select_fit_window(np.array([[0.0, 1.0]]), signal)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid t shape should raise ValueError")

    try:
        select_fit_window(t, np.array([[0.0, 1.0]]))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid signal shape should raise ValueError")

    try:
        select_fit_window(t[:5], signal[:3])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched length should raise ValueError")

    try:
        select_fit_window(t[:1], signal[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("too-short signal should raise ValueError")

    try:
        select_fit_window(t[:2], signal[:2], window_fraction=0.1, min_points=1)
    except ValueError:
        pass
    else:
        raise AssertionError("window too short should raise ValueError")

    flat_signal = np.ones_like(signal)
    _ = select_fit_window(t, flat_signal, window_fraction=0.3, min_points=20)

    decaying = np.exp((-0.2 - 1j * omega) * t)
    tmin3, tmax3 = select_fit_window(
        t,
        decaying,
        window_fraction=0.3,
        min_points=20,
        start_fraction=0.1,
        growth_weight=0.5,
        require_positive=True,
    )
    assert tmax3 > tmin3

    g_fit2, w_fit2, _tmin2, _tmax2 = fit_growth_rate_auto(t, signal, tmin=2.0)
    assert np.isfinite(g_fit2)
    assert np.isfinite(w_fit2)

    try:
        select_fit_window(t, signal, start_fraction=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid start_fraction should raise ValueError")

    try:
        select_fit_window(t, signal, growth_weight=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative growth_weight should raise ValueError")


def test_fit_growth_rate_auto_fallback_respects_start_fraction() -> None:
    t = np.linspace(0.0, 10.0, 200)
    signal = np.exp((-0.08 - 1j * 0.3) * t)

    _g, _w, tmin, tmax = fit_growth_rate_auto(
        t,
        signal,
        min_points=20,
        start_fraction=0.6,
        min_r2=2.0,
        window_method="loglinear",
    )

    assert tmax > tmin
    assert tmin >= t[int(0.6 * t.size)]


def test_select_ky_index_prefers_nonzonal_matching_magnitude() -> None:
    ky = np.array([0.0, -0.01])
    assert select_ky_index(ky, 0.01) == 1
    assert select_ky_index(ky, 0.009) == 1


def test_select_ky_index_prefers_sign_match_when_available() -> None:
    ky = np.array([0.0, -0.01, 0.01])
    assert select_ky_index(ky, 0.01) == 2
    assert select_ky_index(ky, -0.01) == 1


def test_select_ky_index_zero_target_chooses_zonal() -> None:
    ky = np.array([0.02, 0.0, -0.03])
    assert select_ky_index(ky, 0.0) == 1


def test_select_ky_index_keeps_zonal_when_request_is_closer_to_zero() -> None:
    ky = np.array([0.0, -0.01])
    assert select_ky_index(ky, 1.0e-4) == 0


def test_select_ky_index_refuses_target_beyond_grid() -> None:
    # Full FFT grid: +1.5 is the largest positive row, -1.8 the Nyquist row.
    ky = np.fft.fftfreq(12, d=1.0 / (0.3 * 12))
    assert select_ky_index(ky, 1.5) == int(np.argmin(np.abs(ky - 1.5)))
    for target in (1.6, 5.0):
        with pytest.raises(ValueError, match="beyond the grid"):
            select_ky_index(ky, target)


def test_select_ky_index_validates_shape() -> None:
    with pytest.raises(ValueError):
        select_ky_index(np.array([]), 0.1)
    with pytest.raises(ValueError):
        select_ky_index(np.array([[0.0, 0.1]]), 0.1)


def test_extract_eigenfunction_svd_and_snapshot():
    """Eigenfunction extraction should recover the spatial mode."""
    t = np.linspace(0.0, 1.0, 64)
    gamma = 0.2
    omega = 0.4
    ts = np.exp((gamma - 1j * omega) * t)
    mode = np.array([1.0, 2.0, 3.0, 4.0])
    data = ts[:, None] * mode[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snapshot_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode / svd_mode[0], mode / mode[0])
    assert np.allclose(snapshot_mode / snapshot_mode[0], mode / mode[0])


def test_extract_eigenfunction_invalid():
    """Eigenfunction extraction should validate inputs."""
    t = np.linspace(0.0, 1.0, 8)
    phi_t = np.zeros((8, 1, 1, 4))
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t[None, ...], t, sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t[None, :], sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t[:-1], sel)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, method="bad")
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, tmin=2.0, tmax=3.0)
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, z=np.zeros(3))
    with pytest.raises(ValueError):
        extract_eigenfunction(phi_t, t, sel, z=np.zeros((2, 2)))


def test_extract_eigenfunction_z_normalization():
    """Eigenfunction should normalize to theta=0 when z is provided."""
    t = np.linspace(0.0, 1.0, 8)
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    mode = np.array([2.0, 4.0, 6.0, 8.0])
    phi_t = np.ones((t.size, 1, 1, mode.size)) * mode[None, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    out = extract_eigenfunction(phi_t, t, sel, z=z, method="snapshot")
    assert np.isclose(out[1], 1.0)


def test_extract_eigenfunction_z_zero_fallback():
    """If theta=0 value is zero, normalization should fall back to max."""
    t = np.linspace(0.0, 1.0, 8)
    z = np.array([-1.0, 0.0, 1.0, 2.0])
    mode = np.array([1.0, 0.0, 2.0, 3.0])
    phi_t = np.ones((t.size, 1, 1, mode.size)) * mode[None, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    out = extract_eigenfunction(phi_t, t, sel, z=z, method="snapshot")
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_extract_eigenfunction_nan_fallback():
    """SVD eigenfunction extraction should fall back on NaNs."""
    t = np.linspace(0.0, 1.0, 16)
    ts = np.exp((0.1 - 1j * 0.2) * t)
    data = ts[:, None] * np.array([1.0, np.nan])[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snap_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode, snap_mode, equal_nan=True)


def test_extract_eigenfunction_linalg_fallback(monkeypatch):
    """SVD eigenfunction extraction should fall back on LinAlgError."""
    t = np.linspace(0.0, 1.0, 16)
    ts = np.exp((0.1 - 1j * 0.2) * t)
    data = ts[:, None] * np.linspace(1.0, 2.0, 4)[None, :]
    phi_t = data[:, None, None, :]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    def _bad_svd(*_args, **_kwargs):
        raise np.linalg.LinAlgError("forced failure")

    monkeypatch.setattr(np.linalg, "svd", _bad_svd)
    svd_mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    snap_mode = extract_eigenfunction(phi_t, t, sel, method="snapshot")
    assert np.allclose(svd_mode, snap_mode)


def test_extract_eigenfunction_zero_signal():
    """Zero signals should return a finite eigenfunction."""
    t = np.linspace(0.0, 1.0, 8)
    phi_t = np.zeros((8, 1, 1, 4))
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    mode = extract_eigenfunction(phi_t, t, sel, method="svd")
    assert np.all(np.isfinite(mode))


def test_density_moment_supports_5d_and_6d_inputs() -> None:
    jl = np.ones((2, 1, 1, 1), dtype=np.complex128)
    g5 = np.zeros((2, 3, 1, 1, 1), dtype=np.complex128)
    g5[:, 0, ...] = np.array([1.0, 2.0])[:, None, None, None]
    out5 = density_moment(g5, jl)
    assert np.allclose(out5, np.array([3.0]))

    g6 = np.zeros((2, 2, 3, 1, 1, 1), dtype=np.complex128)
    g6[0, :, 0, ...] = np.array([1.0, 2.0])[:, None, None, None]
    g6[1, :, 0, ...] = np.array([3.0, 4.0])[:, None, None, None]
    out6_all = density_moment(g6, jl)
    out6_one = density_moment(g6, jl, species_index=1)
    assert np.allclose(out6_all, np.array([10.0]))
    assert np.allclose(out6_one, np.array([7.0]))

    with pytest.raises(ValueError):
        density_moment(np.zeros((1, 2, 3)), jl)


def test_fit_growth_rate_validates_and_filters_nonfinite() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    signal = np.exp((0.4 - 0.25j) * t)
    signal = signal.astype(np.complex128)
    signal[1] = np.nan + 1j * np.nan
    gamma, omega = fit_growth_rate(t, signal)
    assert np.isclose(gamma, 0.4, atol=1.0e-6)
    assert np.isclose(omega, 0.25, atol=1.0e-6)

    with pytest.raises(ValueError):
        fit_growth_rate(t[None, :], signal)
    with pytest.raises(ValueError):
        fit_growth_rate(t, signal[None, :])
    with pytest.raises(ValueError):
        fit_growth_rate(t[:-1], signal)
    with pytest.raises(ValueError):
        fit_growth_rate(np.array([0.0]), np.array([np.nan + 0.0j]))


def test_fit_growth_rate_with_stats_handles_flat_signal() -> None:
    t = np.linspace(0.0, 1.0, 8)
    signal = np.ones_like(t, dtype=np.complex128)
    gamma, omega, r2_log, r2_phase = fit_growth_rate_with_stats(t, signal)
    assert np.isfinite(gamma)
    assert np.isfinite(omega)
    assert r2_log == -np.inf
    assert r2_phase == -np.inf


def test_select_fit_window_loglinear_validates_arguments() -> None:
    t = np.linspace(0.0, 1.0, 16)
    signal = np.exp((0.1 - 0.2j) * t)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t[None, :], signal)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal[None, :])
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t[:-1], signal)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, min_points=1)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, start_fraction=-0.1)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, max_fraction=0.0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, end_fraction=0.0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, num_windows=0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, growth_weight=-1.0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, min_amp_fraction=1.0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, max_amp_fraction=0.0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, late_penalty=-0.1)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, min_slope_frac=-0.1)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, slope_var_weight=-0.1)


def test_fit_growth_rate_auto_fixed_and_invalid_method() -> None:
    t = np.linspace(0.0, 5.0, 64)
    signal = np.exp((0.2 - 0.1j) * t)
    gamma, omega, tmin, tmax = fit_growth_rate_auto(
        t,
        signal,
        window_method="fixed",
        min_points=8,
        window_fraction=0.5,
    )
    assert np.isfinite(gamma)
    assert np.isfinite(omega)
    assert tmax > tmin

    zeros = np.array([np.nan + 0.0j])
    gamma0, omega0, tmin0, tmax0 = fit_growth_rate_auto(np.array([0.0]), zeros)
    assert (gamma0, omega0, tmin0, tmax0) == (0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError):
        fit_growth_rate_auto(t, signal, window_method="bad")


def test_fit_growth_rate_auto_with_stats_fallback(monkeypatch) -> None:
    t = np.linspace(0.0, 5.0, 64)
    signal = np.exp((0.2 - 0.1j) * t)

    def _boom(*_args, **_kwargs):
        raise ValueError("forced")

    monkeypatch.setattr(
        "gkx.diagnostics.growth_rates.fit_growth_rate_with_stats", _boom
    )
    gamma, omega, tmin, tmax, r2_log, r2_phase = fit_growth_rate_auto_with_stats(
        t, signal
    )
    assert np.isfinite(gamma)
    assert np.isfinite(omega)
    assert tmax > tmin
    assert r2_log == -np.inf
    assert r2_phase == -np.inf


def test_windowed_growth_rate_from_omega_series():
    """Windowed growth/frequency averaging should select the requested (ky, kx) branch."""

    gamma_t = np.array(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.3], [0.4, 0.5]],
            [[0.3, 0.4], [0.5, 0.6]],
            [[0.4, 0.5], [0.6, 0.7]],
        ],
        dtype=float,
    )
    omega_t = -2.0 * gamma_t
    sel = ModeSelection(ky_index=1, kx_index=0, z_index=0)

    g, w, gs, ws = windowed_growth_rate_from_omega_series(
        gamma_t, omega_t, sel, navg_fraction=0.5
    )
    assert np.allclose(gs, np.array([0.3, 0.4, 0.5, 0.6]))
    assert np.allclose(ws, np.array([-0.6, -0.8, -1.0, -1.2]))
    assert np.isclose(g, np.mean([0.5, 0.6]))
    assert np.isclose(w, np.mean([-1.0, -1.2]))

    g_last, w_last, _gs, _ws = windowed_growth_rate_from_omega_series(
        gamma_t, omega_t, sel, use_last=True
    )
    assert np.isclose(g_last, 0.6)
    assert np.isclose(w_last, -1.2)


def test_instantaneous_growth_rate_from_phi_supports_projected_branch_selection():
    """Projected GX growth extraction should recover the dominant full-z branch."""

    t = np.linspace(0.0, 15.0, 256)
    gamma_dom = 0.2
    omega_dom = 1.1
    gamma_mid = 0.05
    omega_mid = -0.3
    dominant = np.exp((gamma_dom - 1j * omega_dom) * t)
    midplane_branch = np.exp((gamma_mid - 1j * omega_mid) * t)

    phi_t = np.zeros((t.size, 1, 1, 4), dtype=np.complex128)
    phi_t[:, 0, 0, :] = (
        dominant[:, None] * np.array([0.0, 1.0, 1.0, 1.0])[None, :]
        + midplane_branch[:, None] * np.array([1.0, 0.0, 0.0, 0.0])[None, :]
    )
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    gamma_z, omega_z, _gz, _oz, _tmid = instantaneous_growth_rate_from_phi(
        phi_t, t, sel, navg_fraction=0.5, mode_method="z_index"
    )
    gamma_proj, omega_proj, _gp, _op, _tmidp = instantaneous_growth_rate_from_phi(
        phi_t, t, sel, navg_fraction=0.5, mode_method="project"
    )

    assert np.isclose(gamma_z, gamma_mid, rtol=5.0e-2, atol=5.0e-3)
    assert np.isclose(omega_z, omega_mid, rtol=5.0e-2, atol=5.0e-3)
    assert np.isclose(gamma_proj, gamma_dom, rtol=5.0e-2, atol=5.0e-3)
    assert np.isclose(omega_proj, omega_dom, rtol=5.0e-2, atol=5.0e-3)


def test_instantaneous_growth_rate_from_phi_validates_inputs_and_handles_last_sample() -> (
    None
):
    t = np.linspace(0.0, 1.0, 8)
    phi_t = np.ones((8, 1, 1, 2), dtype=np.complex128)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    g, w, gamma_t, omega_t, t_mid = instantaneous_growth_rate_from_phi(
        phi_t, t, sel, use_last=True
    )
    assert np.isfinite(g)
    assert np.isfinite(w)
    assert gamma_t.size == omega_t.size == t_mid.size

    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t[None, ...], t, sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t[None, :], sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t[:-1], sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(
            np.ones((1, 1, 1, 2), dtype=np.complex128), np.array([0.0]), sel
        )
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t, sel, mode_method="bad")

    phi_bad = phi_t.copy()
    phi_bad[:-1] = 0.0
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_bad, t, sel)


def test_windowed_growth_rate_from_omega_series_validates_inputs() -> None:
    gamma_t = np.ones((4, 2, 2), dtype=float)
    omega_t = np.ones((4, 2, 2), dtype=float)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    with pytest.raises(ValueError):
        windowed_growth_rate_from_omega_series(gamma_t[0], omega_t, sel)
    with pytest.raises(ValueError):
        windowed_growth_rate_from_omega_series(gamma_t, omega_t[:, :, :1], sel)
    with pytest.raises(ValueError):
        windowed_growth_rate_from_omega_series(
            gamma_t, omega_t, ModeSelection(ky_index=5, kx_index=0, z_index=0)
        )

    gamma_bad = np.full((2, 1, 1), np.nan)
    omega_bad = np.full((2, 1, 1), np.nan)
    with pytest.raises(ValueError):
        windowed_growth_rate_from_omega_series(gamma_bad, omega_bad, sel)


def test_log_amp_phase_handles_empty_and_nonfinite() -> None:
    with pytest.raises(ValueError):
        _log_amp_phase(np.asarray([], dtype=np.complex128))

    log_amp, phase = _log_amp_phase(
        np.array([np.nan + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    )
    assert np.all(np.isfinite(log_amp))
    assert np.all(np.isfinite(phase))


def test_extract_mode_time_series_project_falls_back_to_z_index_when_all_rows_invalid() -> (
    None
):
    phi_t = np.full((4, 1, 1, 2), np.nan + 1j * np.nan)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=1)
    out = extract_mode_time_series(phi_t, sel, method="project")
    assert np.isnan(out).all()


def test_extract_mode_time_series_project_uses_full_history_finite_rows() -> None:
    phi_t = np.full((6, 1, 1, 2), np.nan + 1j * np.nan)
    phi_t[1, 0, 0, :] = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    phi_t[2, 0, 0, :] = np.array([2.0 + 0.0j, 4.0 + 0.0j])
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    out = extract_mode_time_series(phi_t, sel, method="project")
    assert np.all(np.isfinite(out[1:3]))


def test_fit_growth_rate_raises_when_too_few_points_after_mask() -> None:
    t = np.array([0.0, 1.0])
    signal = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError):
        fit_growth_rate(t, signal, tmin=2.0)


def test_fit_growth_rate_with_stats_validation_and_finite_fallbacks() -> None:
    t = np.array([0.0, 1.0, 2.0])
    signal = np.array([1.0 + 0.0j, np.nan + 0.0j, np.nan + 0.0j])
    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t[None, :], np.ones(3))
    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t, np.ones((3, 1)))
    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t[:-1], np.ones(3))
    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t, signal)
    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t, np.ones(3), tmin=3.0)


def test_select_fit_window_extra_validation_and_amp_threshold_path() -> None:
    t = np.linspace(0.0, 4.0, 40)
    signal = np.exp((0.2 - 1j * 0.3) * t)
    with pytest.raises(ValueError):
        select_fit_window(t, signal, min_amp_fraction=1.0)
    bad_signal = np.full_like(signal, np.nan + 0.0j)
    bad_signal[0] = 1.0 + 0.0j
    with pytest.raises(ValueError):
        select_fit_window(t, bad_signal)

    tmin, tmax = select_fit_window(t, signal, min_points=10, min_amp_fraction=0.3)
    assert tmax > tmin


def test_select_fit_window_loglinear_additional_validation_and_fallbacks() -> None:
    t = np.linspace(0.0, 4.0, 40)
    signal = np.exp((0.2 - 1j * 0.3) * t)
    bad_signal = np.full_like(signal, np.nan + 0.0j)
    bad_signal[0] = 1.0 + 0.0j
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, bad_signal)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(np.array([0.0]), np.array([1.0 + 0.0j]))
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, num_windows=0)
    with pytest.raises(ValueError):
        select_fit_window_loglinear(t, signal, max_fraction=0.0)

    flat = np.ones_like(signal)
    tmin, tmax = select_fit_window_loglinear(
        t, flat, min_points=10, min_slope_frac=0.5, growth_weight=0.2
    )
    assert tmax > tmin


def test_select_fit_window_loglinear_prefers_clean_late_growth_branch() -> None:
    t = np.linspace(0.0, 9.0, 181)
    omega = 0.35
    gamma_profile = np.where(t < 2.0, -0.04, np.where(t < 5.0, 0.05, 0.18))
    dt = t[1] - t[0]
    log_amp = np.cumsum(gamma_profile) * dt
    signal = np.exp(log_amp - 1j * omega * t)
    signal[:40] *= 1.0 + 0.12 * np.sin(6.0 * t[:40])

    tmin, tmax = select_fit_window_loglinear(
        t,
        signal,
        min_points=20,
        start_fraction=0.0,
        require_positive=True,
        min_amp_fraction=0.2,
        min_slope_frac=0.8,
        growth_weight=0.5,
        slope_var_weight=0.2,
        late_penalty=0.05,
    )

    assert tmax > tmin
    assert tmin >= 4.5


def test_select_fit_window_loglinear_handles_zero_amplitude_reference() -> None:
    t = np.linspace(0.0, 3.0, 24)
    signal = np.zeros_like(t, dtype=np.complex128)

    tmin, tmax = select_fit_window_loglinear(
        t,
        signal,
        min_points=8,
        min_amp_fraction=0.2,
        max_amp_fraction=0.8,
    )

    assert tmax > tmin


def test_fit_growth_rate_auto_validation_and_nonfinite_paths() -> None:
    with pytest.raises(ValueError):
        fit_growth_rate_auto(np.array([[0.0, 1.0]]), np.array([1.0 + 0.0j, 2.0 + 0.0j]))
    with pytest.raises(ValueError):
        fit_growth_rate_auto(np.array([0.0, 1.0]), np.array([1.0 + 0.0j]))
    t = np.linspace(0.0, 1.0, 8)
    signal = np.full(t.shape, np.nan + 0.0j, dtype=np.complex128)
    signal[0] = 1.0 + 0.0j
    gamma, omega, tmin, tmax = fit_growth_rate_auto(t, signal)
    assert (gamma, omega, tmin, tmax) == (0.0, 0.0, 0.0, 0.0)


def test_fit_growth_rate_auto_invalid_window_method_and_stats_fallback(
    monkeypatch,
) -> None:
    t = np.linspace(0.0, 2.0, 16)
    signal = np.exp((0.1 - 1j * 0.2) * t)
    with pytest.raises(ValueError):
        fit_growth_rate_auto(t, signal, window_method="bad")

    monkeypatch.setattr(
        "gkx.diagnostics.growth_rates.fit_growth_rate_with_stats",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced")),
    )
    gamma, omega, tmin, tmax, r2_log, r2_phase = fit_growth_rate_auto_with_stats(
        t, signal, window_method="fixed"
    )
    assert np.isfinite(gamma)
    assert np.isfinite(omega)
    assert tmax > tmin
    assert r2_log == -np.inf
    assert r2_phase == -np.inf


def test_fit_growth_rate_with_stats_applies_tmax_and_handles_flat_signal() -> None:
    t = np.linspace(0.0, 2.0, 9)
    signal = np.ones_like(t, dtype=np.complex128)

    gamma, omega, r2_log, r2_phase = fit_growth_rate_with_stats(t, signal, tmax=1.0)
    assert np.isfinite(gamma)
    assert np.isfinite(omega)
    assert r2_log == -np.inf
    assert r2_phase == -np.inf

    with pytest.raises(ValueError):
        fit_growth_rate_with_stats(t, signal, tmax=0.0)


def test_instantaneous_growth_rate_from_phi_uses_default_time_axis() -> None:
    phi_t = np.ones((3, 1, 1, 1), dtype=np.complex128)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    gamma_avg, omega_avg, gamma_t, omega_t, t_mid = instantaneous_growth_rate_from_phi(
        phi_t, None, sel
    )
    assert gamma_t.shape == (2,)
    assert omega_t.shape == (2,)
    assert t_mid.shape == (2,)
    assert np.isfinite(gamma_avg)
    assert np.isfinite(omega_avg)


def test_instantaneous_growth_rate_from_phi_branches_and_validation() -> None:
    t = np.array([0.0, 1.0, 2.0])
    phi_t = np.exp((0.2 - 1j * 0.5) * t)[:, None, None, None]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    gamma_avg, omega_avg, gamma_t, omega_t, t_mid = instantaneous_growth_rate_from_phi(
        phi_t,
        t,
        sel,
        use_last=True,
        mode_method="max",
    )
    assert np.isclose(gamma_avg, gamma_t[-1])
    assert np.isclose(omega_avg, omega_t[-1])
    assert t_mid.shape == (2,)

    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t[0], t, sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t[:, None], sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t[:-1], sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t[:1], t[:1], sel)
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(phi_t, t, sel, mode_method="bad")
    with pytest.raises(ValueError):
        instantaneous_growth_rate_from_phi(
            np.array([[[[0.0 + 0.0j]]], [[[np.nan + 0.0j]]]]), np.array([0.0, 1.0]), sel
        )


def test_windowed_growth_rate_from_omega_series_use_last_branch() -> None:
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    gamma_t = np.array([[[0.1]], [[0.2]], [[0.3]]], dtype=float)
    omega_t = np.array([[[0.4]], [[0.5]], [[0.6]]], dtype=float)
    gamma_avg, omega_avg, gamma, omega = windowed_growth_rate_from_omega_series(
        gamma_t,
        omega_t,
        sel,
        use_last=True,
    )
    assert gamma_avg == gamma[-1] == 0.3
    assert omega_avg == omega[-1] == 0.6


def test_log_amp_phase_handles_all_nonfinite_and_zero_scale() -> None:
    log_amp, phase = _log_amp_phase(np.array([np.nan + 0.0j, np.nan + 1.0j]))
    assert log_amp.shape == (2,)
    assert phase.shape == (2,)

    log_amp_zero, phase_zero = _log_amp_phase(np.array([0.0 + 0.0j, 0.0 + 0.0j]))
    assert np.all(np.isfinite(log_amp_zero))
    assert np.all(np.isfinite(phase_zero))


def _transient_then_exponential(
    *,
    gamma: float = 0.25,
    omega: float = 0.4,
    n: int = 800,
    t_max: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A decaying transient branch plus the dominant growing branch."""

    t = np.linspace(0.0, t_max, n)
    dominant = np.exp((gamma - 1j * omega) * t)
    transient = 30.0 * np.exp((-0.9 - 1j * 2.5) * t)
    return t, dominant + transient


def test_select_fit_window_stationary_skips_the_transient() -> None:
    """The stationary window starts after the subdominant branch has decayed."""

    t, signal = _transient_then_exponential()
    window = select_fit_window_stationary(t, signal, min_points=20)
    assert window is not None
    tmin, tmax = window
    # The transient is 30x larger at t=0 and decays at rate 0.9, so it is
    # negligible only after t ~ 8; the window must not reach back into it.
    assert tmin > 6.0
    assert tmax > 0.5 * t[-1]
    assert (tmax - tmin) * 0.25 >= 2.0  # at least two growth times


def test_fit_growth_rate_auto_stationary_beats_loglinear_on_a_transient() -> None:
    """Stationary window selection recovers the dominant branch accurately."""

    gamma_true, omega_true = 0.25, 0.4
    t, signal = _transient_then_exponential(gamma=gamma_true, omega=omega_true)

    g_stat, w_stat, tmin, tmax = fit_growth_rate_auto(
        t, signal, min_points=20, window_method="stationary"
    )
    assert g_stat == pytest.approx(gamma_true, rel=1e-3)
    assert w_stat == pytest.approx(omega_true, rel=1e-3)
    assert tmin < tmax

    g_log, _w_log, _tmin, _tmax = fit_growth_rate_auto(
        t, signal, min_points=20, window_method="loglinear"
    )
    # Both methods must be sane here; the stationary one is at least as close
    # to the true dominant growth rate.
    assert abs(g_stat - gamma_true) <= abs(g_log - gamma_true) + 1e-12


def test_fit_growth_rate_auto_falls_back_when_no_stationary_window() -> None:
    """A purely damped signal has no growth window, so auto falls back."""

    t = np.linspace(0.0, 10.0, 200)
    signal = np.exp((-0.5 - 1j * 0.3) * t)
    assert select_fit_window_stationary(t, signal, min_points=20) is None

    with pytest.warns(RuntimeWarning, match="no stationary growth window"):
        gamma, _omega, _tmin, _tmax = fit_growth_rate_auto(
            t, signal, min_points=20, window_method="stationary"
        )
    assert gamma == pytest.approx(-0.5, abs=1e-6)


def test_fit_growth_rate_auto_rejects_unknown_window_method() -> None:
    t = np.linspace(0.0, 10.0, 64)
    signal = np.exp((0.2 - 1j * 0.1) * t)
    with pytest.raises(ValueError, match="window_method"):
        fit_growth_rate_auto(t, signal, window_method="bogus")


def test_fit_growth_rate_uncertainty_reports_stderr_and_r2() -> None:
    """A clean exponential fits exactly; noise inflates the slope stderr."""

    t = np.linspace(0.0, 20.0, 400)
    clean = np.exp((0.3 - 1j * 0.7) * t)
    stats = fit_growth_rate_uncertainty(t, clean)
    assert stats.gamma == pytest.approx(0.3, rel=1e-9)
    assert stats.omega == pytest.approx(0.7, rel=1e-9)
    assert stats.r2_log == pytest.approx(1.0, abs=1e-9)
    assert stats.gamma_stderr < 1e-6
    assert stats.n_points == 400
    assert (stats.tmin, stats.tmax) == (pytest.approx(0.0), pytest.approx(20.0))

    rng = np.random.default_rng(0)
    noisy = clean * np.exp(rng.normal(scale=0.05, size=t.size))
    noisy_stats = fit_growth_rate_uncertainty(t, noisy)
    assert noisy_stats.gamma_stderr > stats.gamma_stderr
    assert noisy_stats.gamma == pytest.approx(0.3, abs=5.0 * noisy_stats.gamma_stderr)


def test_fit_growth_rate_uncertainty_honours_the_window() -> None:
    t = np.linspace(0.0, 20.0, 400)
    signal = np.exp((0.3 - 1j * 0.7) * t)
    stats = fit_growth_rate_uncertainty(t, signal, tmin=5.0, tmax=15.0)
    assert stats.tmin >= 5.0
    assert stats.tmax <= 15.0
    assert stats.n_points < 400
    assert stats.gamma == pytest.approx(0.3, rel=1e-9)


def test_select_fit_window_stationary_tolerates_sub_percent_drift() -> None:
    """A window drifting well under a percent still counts as stationary.

    gamma(t) is smoothed before the stationarity test, so its residual scatter
    on a clean run is orders of magnitude below any meaningful drift. The
    acceptance band is therefore floored at ``rel_tolerance`` times the mean,
    without which the selector refuses windows that are stationary to 0.5%.
    """

    t = np.linspace(0.0, 120.0, 2400)
    gamma0 = 0.1
    # gamma(t) relaxes onto gamma0 from 0.5% above it with a long time
    # constant: over the late half the drift is well under one percent.
    gamma_t = gamma0 * (1.0 + 0.005 * np.exp(-t / 40.0))
    phase = -0.3 * t
    signal = np.exp(np.cumsum(gamma_t) * (t[1] - t[0]) + 1j * phase)

    window = select_fit_window_stationary(t, signal, min_points=40)
    assert window is not None
    tmin, tmax = window
    assert (tmax - tmin) * gamma0 >= 2.0

    g, _w, _a, _b = fit_growth_rate_auto(
        t, signal, min_points=40, window_method="stationary"
    )
    assert g == pytest.approx(gamma0, rel=0.01)


# ---- from test_timestep_cost.py ----
# Time-step cost and CFL-attribution diagnostics.
#
# Every expected number here is hand-computable from the synthetic series, so a
# regression in the reported cost per unit of simulated time shows up as a
# concrete arithmetic mismatch rather than a shifted baseline.


def _series(dt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, dt)`` with the runtime convention ``t[i] - t[i-1] == dt[i]``."""

    return np.cumsum(dt), dt


def _diag(t: np.ndarray, dt: np.ndarray, **kwargs: object) -> SimulationDiagnostics:
    zeros = np.zeros_like(t)
    return SimulationDiagnostics(
        t=t,
        dt_t=dt,
        dt_mean=np.asarray(np.mean(dt)),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=zeros,
        Wphi_t=zeros,
        Wapar_t=zeros,
        heat_flux_t=zeros,
        particle_flux_t=zeros,
        energy_t=zeros,
        **kwargs,
    )


# --- cost per unit of simulated time --------------------------------------


def test_steps_and_dt_collapse_match_a_hand_computed_series() -> None:
    """Reported cost reproduces arithmetic done by hand on a known series."""

    # dt halves every step: four samples, so three gaps, each of which is
    # exactly one step. t runs 1.0, 1.5, 1.75, 1.875 -> span 0.875.
    t, dt = _series(np.array([1.0, 0.5, 0.25, 0.125]))
    report = timestep_cost_report(t, dt, wall_seconds=8.75)

    assert report.n_samples == 4
    assert report.t_start == pytest.approx(1.0)
    assert report.t_end == pytest.approx(1.875)
    assert report.t_span == pytest.approx(0.875)
    assert report.steps == pytest.approx(3.0)
    assert report.steps_are_exact is True
    assert report.steps_per_unit_time == pytest.approx(3.0 / 0.875)
    assert report.wall_seconds_per_unit_time == pytest.approx(8.75 / 0.875)
    assert report.dt_initial == pytest.approx(1.0)
    assert report.dt_min == pytest.approx(0.125)
    assert report.dt_final == pytest.approx(0.125)
    # The headline collapse number: the first step was eight times the smallest.
    assert report.dt_collapse_ratio == pytest.approx(8.0)


def test_constant_step_run_reports_one_step_per_dt_and_no_collapse() -> None:
    """A run that never adapts costs exactly ``1 / dt`` steps per unit time."""

    t, dt = _series(np.full(101, 0.25))
    report = timestep_cost_report(t, dt)

    assert report.steps == pytest.approx(100.0)
    assert report.steps_per_unit_time == pytest.approx(4.0)
    assert report.dt_collapse_ratio == pytest.approx(1.0)
    assert report.cost_growth_ratio == pytest.approx(1.0)
    assert report.wall_seconds is None
    assert report.wall_seconds_per_unit_time is None


def test_cost_growth_ratio_compares_the_first_and_last_quarter_of_a_run() -> None:
    """Within-run degradation is measured over time, not sample index."""

    # First half at dt=1 (1 step per unit t), second half at dt=0.25 (4 per
    # unit t). The quarters therefore differ by exactly a factor of four.
    dt = np.concatenate([np.full(40, 1.0), np.full(160, 0.25)])
    t, dt = _series(dt)
    report = timestep_cost_report(t, dt)

    assert report.steps_per_unit_time_first_quarter == pytest.approx(1.0)
    assert report.steps_per_unit_time_last_quarter == pytest.approx(4.0)
    assert report.cost_growth_ratio == pytest.approx(4.0)


def test_strided_series_is_reported_as_an_estimate_not_a_count() -> None:
    """Striding makes the step count approximate, and the report says so."""

    t_full, dt_full = _series(np.full(101, 0.25))
    report = timestep_cost_report(t_full[::5], dt_full[::5])

    assert report.steps_are_exact is False
    assert any("strided" in note for note in report.notes)
    # Each retained gap spans five steps, so the estimate recovers the truth
    # here precisely because dt is constant.
    assert report.steps == pytest.approx(100.0)


def test_report_rejects_mismatched_and_unusable_series() -> None:
    """A malformed series raises rather than reporting a meaningless number."""

    with pytest.raises(ValueError, match="same length"):
        timestep_cost_report(np.arange(3.0), np.ones(4))
    with pytest.raises(ValueError, match="usable sample"):
        timestep_cost_report(np.array([np.nan]), np.array([np.nan]))


# --- which CFL term is limiting --------------------------------------------


def test_term_contributions_are_additive_and_reproduce_the_integrator_sum() -> None:
    """Contributions sum to the frequency the integrator actually forms."""

    contributions = cfl_term_contributions(
        magnetic_drift_radial=3.0,
        magnetic_drift_binormal=5.0,
        parallel_streaming=2.0,
        exb_radial=11.0,
        exb_binormal=1.0,
    )

    # max(3, 11) + max(5, 1) + 2 == 18
    assert sum(contributions.values()) == pytest.approx(18.0)
    assert contributions["exb"] == pytest.approx(8.0)
    assert set(contributions) == set(CFL_TERM_NAMES)


@pytest.mark.parametrize(
    ("dominant", "speeds"),
    [
        ("exb", {"exb_binormal": 100.0}),
        ("parallel_streaming", {"parallel_streaming": 100.0}),
        ("magnetic_drift_radial", {"magnetic_drift_radial": 100.0}),
        ("magnetic_drift_binormal", {"magnetic_drift_binormal": 100.0}),
    ],
)
def test_limiting_term_names_the_speed_that_dominates_by_construction(
    dominant: str, speeds: dict[str, float]
) -> None:
    """One speed set 100x the rest is the term the report names."""

    args = {
        "magnetic_drift_radial": 1.0,
        "magnetic_drift_binormal": 1.0,
        "parallel_streaming": 1.0,
        "exb_radial": 1.0,
        "exb_binormal": 1.0,
    }
    args.update(speeds)
    term, share = cfl_limiting_term(cfl_term_contributions(**args))

    assert term == dominant
    assert share > 0.9


def test_matched_exb_does_not_displace_the_drift_it_equals() -> None:
    """An ExB speed that merely matches a drift is not reported as limiting."""

    term, _share = cfl_limiting_term(
        cfl_term_contributions(
            magnetic_drift_radial=4.0,
            magnetic_drift_binormal=1.0,
            parallel_streaming=1.0,
            exb_radial=4.0,
            exb_binormal=1.0,
        )
    )
    assert term == "magnetic_drift_radial"


def test_dt_trajectory_inverts_to_a_growing_exb_share() -> None:
    """A collapsing dt is attributed to the ExB excess over the linear floor."""

    # Linear floor 1 + 2 + 3 = 6, numerator 12, so dt = 12 / omega_total.
    scales = CFLScales(1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0)
    # omega_total = 6 (pure linear floor) then 12, 24, 48.
    report = cfl_limiter_report(np.array([2.0, 1.0, 0.5, 0.25]), scales)

    assert report.omega_linear_floor == pytest.approx(6.0)
    assert report.omega_total_initial == pytest.approx(6.0)
    assert report.omega_total_final == pytest.approx(48.0)
    assert report.exb_share_initial == pytest.approx(0.0)
    # At the end ExB supplies 48 - 6 = 42 of the 48.
    assert report.exb_share_final == pytest.approx(42.0 / 48.0)
    assert report.limiting_term_initial == "parallel_streaming"
    assert report.limiting_term_final == "exb"
    assert report.limiting_term_final_share == pytest.approx(42.0 / 48.0)
    assert report.limiting_term_sample_fractions["exb"] == pytest.approx(0.75)
    assert report.samples_attributed == 4


def test_a_run_pinned_at_the_dt_ceiling_is_reported_as_capped_not_limited() -> None:
    """A capped step only bounds the frequency, so no term is named."""

    scales = CFLScales(1.0, 2.0, 3.0, 12.0, 1.0e-3, 2.0)
    report = cfl_limiter_report(np.full(5, 2.0), scales)

    assert report.samples_at_dt_ceiling == 5
    assert report.samples_attributed == 0
    assert report.limiting_term_final == CFL_TERM_UNRESOLVED
    assert np.isnan(report.omega_total_final)


def test_scales_decode_round_trips_and_rejects_unusable_vectors() -> None:
    """The recorded vector decodes positionally and fails closed."""

    values = [1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0]
    assert len(CFL_SCALE_LABELS) == len(values)
    assert cfl_scales_from_array(np.asarray(values)) == CFLScales(*values)
    assert cfl_scales_from_array(None) is None
    assert cfl_scales_from_array(np.asarray(values[:-1])) is None
    assert cfl_scales_from_array(np.asarray([1.0, 2.0, 3.0, 0.0, 1e-3, 10.0])) is None
    assert cfl_scales_from_array(np.asarray([np.nan, *values[1:]])) is None


# --- what a run's own summary reports --------------------------------------


def test_summary_payload_is_json_safe_and_carries_the_cost_block() -> None:
    """The block a user reads is valid JSON even where ratios are undefined."""

    t, dt = _series(np.array([2.0, 1.0, 0.5, 0.25]))
    scales = np.asarray([1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0])
    payload = timestep_cost_payload(_diag(t, dt, cfl_scales=scales), wall_seconds=42.0)

    assert payload["dt_collapse_ratio"] == pytest.approx(8.0)
    assert payload["wall_seconds_per_unit_time"] == pytest.approx(42.0 / 1.75)
    assert payload["cfl"]["limiting_term_final"] == "exb"
    # exb_growth_ratio is infinite here (the run starts on the linear floor);
    # strict JSON has no Infinity, so it must serialise as null.
    assert payload["cfl"]["exb_growth_ratio"] is None
    json.loads(json.dumps(payload, allow_nan=False))


def test_summary_payload_survives_a_run_without_recorded_cfl_scales() -> None:
    """Fixed-step and reloaded runs still report cost, without attribution."""

    t, dt = _series(np.full(5, 0.5))
    payload = timestep_cost_payload(_diag(t, dt))

    assert payload["cfl"] is None
    assert payload["steps_per_unit_time"] == pytest.approx(2.0)


def test_empty_diagnostics_yield_an_empty_block_rather_than_a_write_failure() -> None:
    """A summary artifact never fails to write because of the cost block."""

    empty = np.asarray([], dtype=float)
    assert timestep_cost_payload(_diag(empty, empty)) == {}


def test_cfl_scales_survive_slicing_striding_and_concatenation() -> None:
    """Run-constant scales are not per-sample, so reshaping must not drop them."""

    t, dt = _series(np.full(8, 0.5))
    scales = np.asarray([1.0, 2.0, 3.0, 12.0, 1.0e-3, 10.0])
    diag = _diag(t, dt, cfl_scales=scales)

    for reshaped in (
        slice_runtime_diagnostics(diag, 4),
        stride_runtime_diagnostics(diag, stride=2),
        concat_runtime_diagnostics([diag, diag]),
    ):
        assert reshaped.cfl_scales is not None
        np.testing.assert_allclose(np.asarray(reshaped.cfl_scales), scales)


# ---- from test_plotting.py ----
# Plotting utilities should generate figures without errors.


matplotlib.use("Agg")

import pathlib
import importlib.util

import numpy as np

from gkx.benchmarking_shared import CycloneReference, CycloneScanResult
import matplotlib.pyplot as plt
import pytest
import gkx.artifacts.plotting as plotting
from gkx.workflows.runtime.results import plot
from scripts.checks._gates.zonal_plots import zonal_flow_response_figure
from gkx.artifacts.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    growth_fit_figure,
    linear_validation_figure,
    LinearValidationPanel,
    linear_runtime_panel_figure,
    nonlinear_runtime_panel_figure,
    plot_saved_output,
    scan_comparison_figure,
)


def test_plotting_facade_dir_and_style():
    names = dir(plotting)
    assert "plot_saved_output" in names
    assert "set_plot_style" in names

    plotting.set_plot_style()
    assert plt.rcParams["axes.grid"] is True


def test_cyclone_reference_figure(tmp_path):
    """The Cyclone reference plot should save successfully."""
    ref = CycloneReference(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        gamma=np.array([0.05, 0.06]),
    )
    fig, _axes = cyclone_reference_figure(ref)
    out = tmp_path / "ref.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_cyclone_comparison_figure(tmp_path):
    """Comparison plot should render with both curves."""
    ref = CycloneReference(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        gamma=np.array([0.05, 0.06]),
    )
    scan = CycloneScanResult(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.25, 0.35]),
        gamma=np.array([0.04, 0.05]),
    )
    fig, _axes = cyclone_comparison_figure(ref, scan)
    out = tmp_path / "comparison.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_validation_figure(tmp_path):
    """Summary panel should render and save."""
    z = np.linspace(-1.0, 1.0, 8)
    panel = LinearValidationPanel(
        name="Cyclone",
        z=z,
        eigenfunction=np.exp(1j * z),
        x=np.array([0.2, 0.3]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        x_label=r"$k_y$",
        x_ref=np.array([0.2, 0.3]),
        gamma_ref=np.array([0.11, 0.21]),
        omega_ref=np.array([0.31, 0.41]),
    )
    fig, _axes = linear_validation_figure([panel])
    out = tmp_path / "summary.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_validation_empty():
    """Empty panel list should raise."""
    try:
        linear_validation_figure([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty panels should raise ValueError")


def test_linear_validation_multiple_panels(tmp_path):
    """Multiple panels should render without errors."""
    z = np.linspace(-1.0, 1.0, 8)
    panels = [
        LinearValidationPanel(
            name="Cyclone",
            z=z,
            eigenfunction=np.exp(1j * z),
            x=np.array([0.2, 0.3]),
            gamma=np.array([0.1, 0.2]),
            omega=np.array([0.3, 0.4]),
            x_label=r"$k_y$",
        ),
        LinearValidationPanel(
            name="ITG",
            z=z,
            eigenfunction=np.exp(1j * 0.5 * z),
            x=np.array([0.2, 0.3]),
            gamma=np.array([0.15, 0.25]),
            omega=np.array([0.35, 0.45]),
            x_label=r"$k_y$",
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    out = tmp_path / "summary_multi.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_runtime_panel_figure(tmp_path):
    t = np.linspace(0.1, 1.0, 8)
    signal = np.exp((0.2 - 0.3j) * t)
    z = np.linspace(-np.pi, np.pi, 16)
    eigen = np.cos(z) + 1j * np.sin(z)
    fig, _axes = linear_runtime_panel_figure(
        t=t,
        signal=signal,
        z=z,
        eigenfunction=eigen,
        gamma=0.2,
        omega=-0.3,
    )
    out = tmp_path / "linear_runtime_panel.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_nonlinear_runtime_panel_figure(tmp_path):
    t = np.linspace(0.1, 1.0, 8)
    fig, _axes = nonlinear_runtime_panel_figure(
        t=t,
        phi2=np.exp(t),
        wphi=np.linspace(1.0, 2.0, 8),
        heat_flux=np.linspace(0.1, 0.8, 8),
        gamma=np.linspace(0.01, 0.08, 8),
        omega=np.linspace(-0.1, -0.8, 8),
    )
    out = tmp_path / "nonlinear_runtime_panel.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_plot_dispatches_promoted_runtime_results_without_side_effects():
    from gkx.diagnostics import SimulationDiagnostics
    from gkx.diagnostics.modes import ModeSelection
    from gkx.workflows.runtime.results import (
        RuntimeLinearResult,
        RuntimeLinearScanResult,
        RuntimeNonlinearResult,
    )

    t = np.linspace(0.1, 1.0, 8)
    z = np.linspace(-np.pi, np.pi, 16)
    linear = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(0, 0),
        t=t,
        signal=np.exp((0.3 - 0.4j) * t),
        z=z,
        eigenfunction=np.cos(z) + 0.2j * np.sin(z),
    )
    fig, axes = plot(linear)
    assert np.asarray(axes).size == 2
    plt.close(fig)
    fig, axes = plot(replace(linear, t=None, signal=None))
    assert np.asarray(axes).size == 3
    plt.close(fig)

    scan = RuntimeLinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.2, 0.3]),
        omega=np.array([-0.4, -0.5]),
    )
    fig, axes = plot(scan)
    assert np.asarray(axes).size == 2
    plt.close(fig)

    ones = np.ones_like(t)
    diagnostics = SimulationDiagnostics(
        t=t,
        dt_t=ones,
        dt_mean=np.asarray(1.0),
        gamma_t=0.1 * ones,
        omega_t=-0.2 * ones,
        Wg_t=ones,
        Wphi_t=2.0 * ones,
        Wapar_t=np.zeros_like(t),
        heat_flux_t=0.3 * ones,
        particle_flux_t=0.1 * ones,
        energy_t=3.0 * ones,
    )
    nonlinear = RuntimeNonlinearResult(t=t, diagnostics=diagnostics)
    fig, axes = plot(nonlinear)
    assert np.asarray(axes).size == 3
    plt.close(fig)

    with pytest.raises(ValueError, match="z and eigenfunction"):
        plot(replace(linear, z=None, eigenfunction=None))
    with pytest.raises(ValueError, match="time diagnostics"):
        plot(replace(nonlinear, diagnostics=None))
    with pytest.raises(TypeError, match="LinearResult"):
        plot(object())


def test_zonal_flow_response_figure(tmp_path):
    t = np.linspace(0.0, 20.0, 2001)
    response = 0.15 + np.exp(-0.08 * t) * np.cos(1.5 * t)

    fig, _axes = zonal_flow_response_figure(t, response, title="ZF response")
    out = tmp_path / "zf_response.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_zonal_flow_response_figure_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        zonal_flow_response_figure(np.array([0.0, 1.0]), np.array([1.0]))


def test_plot_saved_output_linear_bundle(tmp_path):
    base = tmp_path / "linear_case"
    (tmp_path / "linear_case.summary.json").write_text(
        '{"kind":"linear","gamma":0.2,"omega":-0.3}',
        encoding="utf-8",
    )
    (tmp_path / "linear_case.timeseries.csv").write_text(
        "t,signal_real,signal_imag,signal_abs\n0.1,1.0,0.0,1.0\n0.2,1.2,0.1,1.204159\n",
        encoding="utf-8",
    )
    (tmp_path / "linear_case.eigenfunction.csv").write_text(
        "z,eigen_real,eigen_imag,eigen_abs\n-1.0,0.5,-0.2,0.538516\n0.0,1.0,0.0,1.0\n1.0,0.5,0.2,0.538516\n",
        encoding="utf-8",
    )
    out = plot_saved_output(base.with_suffix(".summary.json"))
    assert out.exists()


def test_plot_saved_output_nonlinear_csv_bundle(tmp_path):
    base = tmp_path / "nonlinear_case"
    (tmp_path / "nonlinear_case.summary.json").write_text(
        '{"kind":"nonlinear"}',
        encoding="utf-8",
    )
    (tmp_path / "nonlinear_case.diagnostics.csv").write_text(
        "t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux\n"
        "0.1,0.1,0.01,-0.02,1.0,2.0,0.0,3.0,0.4,0.0\n"
        "0.2,0.1,0.02,-0.03,1.1,2.1,0.0,3.2,0.5,0.0\n",
        encoding="utf-8",
    )
    out = plot_saved_output(base.with_suffix(".summary.json"))
    assert out.exists()


def test_scan_comparison_figure_with_reference_and_log_scale(tmp_path):
    x = np.array([0.1, 0.2, 0.4])
    fig, axes = scan_comparison_figure(
        x,
        np.array([0.2, 0.3, 0.4]),
        np.array([-0.1, -0.2, -0.3]),
        r"$k_y$",
        "Scan",
        x_ref=x,
        gamma_ref=np.array([0.21, 0.31, 0.41]),
        omega_ref=np.array([-0.11, -0.21, -0.31]),
        log_x=True,
    )
    out = tmp_path / "scan_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0].get_xscale() == "log"


def test_growth_fit_figure_with_window(tmp_path):
    t = np.linspace(0.0, 4.0, 32)
    signal = np.exp((0.2 - 0.1j) * t)
    fig, axes = growth_fit_figure(t, signal, tmin=1.0, tmax=3.0)
    fit_x = np.asarray(axes[1].lines[1].get_xdata())
    assert fit_x.min() >= 1.0
    assert fit_x.max() <= 3.0
    out = tmp_path / "growth_fit.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_plot_saved_output_missing_summary_and_bad_kind(tmp_path):
    with np.testing.assert_raises(FileNotFoundError):
        plot_saved_output(tmp_path / "missing.summary.json")

    (tmp_path / "unknown.summary.json").write_text(
        '{"kind":"mystery"}', encoding="utf-8"
    )
    with np.testing.assert_raises(ValueError):
        plot_saved_output(tmp_path / "unknown.summary.json")


def test_plot_saved_output_nonlinear_netcdf_bundle(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    dataset = netcdf4.Dataset
    path = tmp_path / "nonlinear_case.out.nc"
    with dataset(path, "w") as root:
        diag = root.createGroup("Diagnostics")
        diag.createDimension("time", 2)
        diag.createDimension("species", 1)
        t_var = diag.createVariable("t", "f8", ("time",))
        t_var[:] = np.array([0.1, 0.2])
        phi2 = diag.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.array([1.0, 2.0])
        wphi = diag.createVariable("Wphi_st", "f8", ("time", "species"))
        wphi[:] = np.array([[2.0], [3.0]])
        heat = diag.createVariable("HeatFlux_st", "f8", ("time", "species"))
        heat[:] = np.array([[0.4], [0.5]])

    out = plot_saved_output(path)
    assert out.exists()


# ---------------------------------------------------------------------------
# Publication nonlinear-transport figures and real-space snapshots.
# ---------------------------------------------------------------------------

from types import SimpleNamespace

import gkx.artifacts.run_summary as snapshots
from gkx.artifacts.transport_figures import (
    flux_spectra_figure,
    heat_flux_time_figure,
    ky_spectrum_tail_ratio,
    phi2_spectra_figure,
    spectrum_cutoff_warnings,
)


def _memory_nonlinear_diag(*, nt=30, ns=2, nky=5, nkx=7, with_resolved=True):
    from gkx.diagnostics import ResolvedDiagnostics, SimulationDiagnostics

    t = np.linspace(0.0, 30.0, nt)
    heat_species = np.stack(
        [0.6 + 0.1 * np.sin(0.3 * t), 0.3 + 0.05 * np.cos(0.2 * t)][:ns], axis=1
    )
    particle_species = 0.1 * heat_species
    ky = np.linspace(0.0, 1.2, nky)
    kx = np.linspace(-1.5, 1.5, nkx)

    resolved = None
    if with_resolved:
        heat_kyst = (
            np.maximum(
                np.einsum("t,k->tk", 1.0 + 0.05 * np.sin(t), ky * np.exp(-3.0 * ky)),
                0.0,
            )[:, None, :]
            * np.array([1.0, 0.5][:ns])[None, :, None]
        )
        heat_kxst = np.exp(-2.0 * np.abs(kx))[None, None, :] * np.ones((nt, ns, 1))
        phi2_kxky = (
            np.exp(-2.0 * np.abs(kx))[None, None, :]
            * np.exp(-3.0 * ky)[None, :, None]
            * (1.0 + 0.1 * np.cos(0.2 * t))[:, None, None]
        )
        resolved = ResolvedDiagnostics(
            Phi2_kxkyt=phi2_kxky,
            Phi2_zonal_t=phi2_kxky[:, 0, :].sum(axis=-1),
            HeatFlux_kyst=heat_kyst,
            HeatFlux_kxst=heat_kxst,
        )
    zeros = np.zeros_like(t)
    diag = SimulationDiagnostics(
        t=t,
        dt_t=np.full_like(t, 0.05),
        dt_mean=np.asarray(0.05),
        gamma_t=zeros,
        omega_t=zeros,
        Wg_t=1.0 + t,
        Wphi_t=0.5 + 0.1 * t,
        Wapar_t=zeros,
        heat_flux_t=heat_species.sum(axis=1),
        particle_flux_t=particle_species.sum(axis=1),
        energy_t=1.5 + 1.1 * t,
        heat_flux_species_t=heat_species,
        particle_flux_species_t=particle_species,
        resolved=resolved,
    )
    return diag, ky, kx


def _write_synthetic_out_nc(path, *, nt=12, ns=2, nky=5, nkx=7):
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.linspace(0.5, 12.0, nt)
    ky = np.linspace(0.0, 1.2, nky)
    kx = np.linspace(-1.5, 1.5, nkx)
    with netcdf4.Dataset(path, "w") as root:
        for name, size in (("time", nt), ("s", ns), ("ky", nky), ("kx", nkx)):
            root.createDimension(name, size)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("ky", "f4", ("ky",))[:] = ky
        grids.createVariable("kx", "f4", ("kx",))[:] = kx
        diag = root.createGroup("Diagnostics")
        species_history = 0.5 + 0.1 * np.outer(t, np.arange(1, ns + 1))
        for name in (
            "Wg_st",
            "Wphi_st",
            "Wapar_st",
            "HeatFlux_st",
            "ParticleFlux_st",
        ):
            diag.createVariable(name, "f4", ("time", "s"))[:, :] = species_history
        heat_kyst = np.broadcast_to(
            (ky * np.exp(-3.0 * ky))[None, None, :], (nt, ns, nky)
        )
        diag.createVariable("HeatFlux_kyst", "f4", ("time", "s", "ky"))[:, :, :] = (
            heat_kyst
        )
        heat_kxst = np.broadcast_to(
            np.exp(-2.0 * np.abs(kx))[None, None, :], (nt, ns, nkx)
        )
        diag.createVariable("HeatFlux_kxst", "f4", ("time", "s", "kx"))[:, :, :] = (
            heat_kxst
        )
        phi2_kxky = np.broadcast_to(
            (np.exp(-3.0 * ky)[:, None] * np.exp(-2.0 * np.abs(kx))[None, :])[
                None, :, :
            ],
            (nt, nky, nkx),
        )
        diag.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[:, :, :] = (
            phi2_kxky
        )
        diag.createVariable("Phi2_zonal_t", "f4", ("time",))[:] = phi2_kxky[
            :, 0, :
        ].sum(axis=-1)
    return path


def _write_csv_sidecar(tmp_path, name="nl_case"):
    base = tmp_path / name
    (tmp_path / f"{name}.summary.json").write_text(
        '{"kind":"nonlinear"}', encoding="utf-8"
    )
    (tmp_path / f"{name}.diagnostics.csv").write_text(
        "t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux,"
        "heat_flux_s0,heat_flux_s1,particle_flux_s0,particle_flux_s1\n"
        "0.1,0.1,0.01,-0.02,1.0,2.0,0.0,3.0,0.4,0.02,0.3,0.1,0.01,0.01\n"
        "0.2,0.1,0.02,-0.03,1.1,2.1,0.0,3.2,0.5,0.03,0.35,0.15,0.02,0.01\n"
        "0.3,0.1,0.02,-0.03,1.2,2.2,0.0,3.4,0.6,0.04,0.4,0.2,0.02,0.02\n",
        encoding="utf-8",
    )
    return base


def test_heat_flux_time_figure_from_arrays(tmp_path):
    diag, _ky, _kx = _memory_nonlinear_diag()
    out = tmp_path / "heat_flux_time.png"
    fig, axes = heat_flux_time_figure(diag, window=(15.0, 30.0), out=out)
    plt.close(fig)
    assert out.exists()
    assert axes[0].get_ylabel()
    assert axes[1].get_xlabel()


def test_heat_flux_time_figure_from_csv_sidecar(tmp_path):
    base = _write_csv_sidecar(tmp_path)
    out = tmp_path / "heat_flux_csv.png"
    fig, _axes = heat_flux_time_figure(
        base.with_suffix(".diagnostics.csv"), window=(0.1, 0.3), out=out
    )
    plt.close(fig)
    assert out.exists()


def test_flux_spectra_figure_from_netcdf(tmp_path):
    path = _write_synthetic_out_nc(tmp_path / "case.out.nc")
    out = tmp_path / "flux_spectra.png"
    fig, axes = flux_spectra_figure(path, out=out)
    plt.close(fig)
    assert out.exists()
    assert len(axes) == 2


def test_ky_spectrum_tail_ratio_distinguishes_cutoff_decay():
    ky = np.arange(9, dtype=float)
    resolved = np.array([0.0, 1.0, 0.7, 0.4, 0.2, 0.08, 0.03, 0.01, 0.005])
    unresolved = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])

    assert ky_spectrum_tail_ratio(ky, resolved) == pytest.approx(0.005)
    assert ky_spectrum_tail_ratio(ky, unresolved) == pytest.approx(1.0)
    assert ky_spectrum_tail_ratio(ky[:4], resolved[:4]) is None


def test_spectrum_cutoff_warning_names_resolution_action(tmp_path):
    path = _write_synthetic_out_nc(tmp_path / "case.out.nc", nky=9)
    netcdf4 = pytest.importorskip("netCDF4")
    with netcdf4.Dataset(path, "a") as root:
        heat = root.groups["Diagnostics"].variables["HeatFlux_kyst"]
        heat[:, :, :] = np.arange(9, dtype=float)[None, None, :]

    warnings = spectrum_cutoff_warnings(path)

    assert len(warnings) == 1
    assert "heat-flux ky cutoff is unresolved" in warnings[0]
    assert "Increase Ny at fixed Ly" in warnings[0]
    assert "Nx/Ny convergence" in warnings[0]


def test_flux_spectrum_marks_unresolved_ky_cutoff():
    diag, ky, kx = _memory_nonlinear_diag(nky=9)
    heat = np.broadcast_to(
        np.arange(9, dtype=float)[None, None, :],
        (diag.t.size, 2, 9),
    )
    diag = replace(diag, resolved=replace(diag.resolved, HeatFlux_kyst=heat))

    fig, axes = flux_spectra_figure(diag, ky=ky, kx=kx)

    assert any("cutoff unresolved" in text.get_text() for text in axes[0].texts)
    plt.close(fig)


def test_phi2_spectra_figure_from_netcdf(tmp_path):
    path = _write_synthetic_out_nc(tmp_path / "case.out.nc")
    out = tmp_path / "phi2_spectra.png"
    fig, axes = phi2_spectra_figure(path, out=out)
    plt.close(fig)
    assert out.exists()
    assert axes.shape == (2, 2)


def test_flux_spectra_figure_from_memory_arrays(tmp_path):
    diag, ky, kx = _memory_nonlinear_diag()
    with pytest.raises(ValueError, match="ky"):
        flux_spectra_figure(diag)
    out = tmp_path / "flux_spectra_memory.png"
    fig, _axes = flux_spectra_figure(diag, ky=ky, kx=kx, window=(10.0, 30.0), out=out)
    plt.close(fig)
    assert out.exists()


def test_phi2_spectra_figure_from_memory_arrays(tmp_path):
    diag, ky, kx = _memory_nonlinear_diag()
    out = tmp_path / "phi2_spectra_memory.png"
    fig, _axes = phi2_spectra_figure(diag, ky=ky, kx=kx, out=out)
    plt.close(fig)
    assert out.exists()


def test_spectra_figures_reject_csv_sidecar(tmp_path):
    base = _write_csv_sidecar(tmp_path)
    for figure in (flux_spectra_figure, phi2_spectra_figure):
        with pytest.raises(ValueError, match=r"out\.nc"):
            figure(base.with_suffix(".diagnostics.csv"))


def test_spectra_figures_reject_missing_resolved():
    diag, ky, kx = _memory_nonlinear_diag(with_resolved=False)
    for figure in (flux_spectra_figure, phi2_spectra_figure):
        with pytest.raises(ValueError, match="resolved"):
            figure(diag, ky=ky, kx=kx)


def _synthetic_spectral_phi(*, nx=12, ny=10, nz=6, seed=3):
    rng = np.random.default_rng(seed)
    real_xyz = rng.normal(size=(nx, ny, nz))
    real_yxz = np.transpose(real_xyz, (1, 0, 2))
    spectral = np.fft.rfft2(real_yxz, axes=(-2, -3)) / float(nx * ny)
    return real_xyz, spectral


def test_potential_real_space_round_trip():
    real_xyz, spectral = _synthetic_spectral_phi()
    ny = real_xyz.shape[1]
    recovered = snapshots.potential_real_space(spectral, ny_full=ny)
    np.testing.assert_allclose(recovered, real_xyz, atol=1e-12)

    full = np.fft.fft2(np.transpose(real_xyz, (1, 0, 2)), axes=(0, 1)) / float(
        real_xyz.shape[0] * ny
    )
    recovered_full = snapshots.potential_real_space(full)
    np.testing.assert_allclose(recovered_full, real_xyz, atol=1e-12)

    with pytest.raises(ValueError):
        snapshots.potential_real_space(spectral[:2], ny_full=ny)


def _movie_tool(name):
    root = pathlib.Path(__file__).parents[3]
    script = root / "scripts" / "artifacts" / "build_turbulence_movie.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_movie_snapshot_replays_lightweight_physical_cuts(tmp_path, monkeypatch):
    module = _movie_tool("gkx_movie_test")

    snapshot = tmp_path / "cuts.npz"
    np.savez_compressed(
        snapshot,
        phi_xy=np.zeros((2, 6, 8)),
        phi_yz=np.zeros((2, 8, 10)),
        times=np.array([1.0, 2.0]),
        label="QHS",
        q=1.2,
        epsilon=0.1,
        major_radius=10.0,
        nfp=6,
        extent=np.array([20.0, 30.0]),
        cylindrical_R_profile=np.array([10.0, 10.1]),
        cylindrical_Z_profile=np.array([0.0, 0.1]),
        toroidal_angle_profile=np.array([0.0, 0.2]),
    )
    rendered = []
    monkeypatch.setattr(
        module, "render_frame", lambda *args, **kwargs: rendered.append((args, kwargs))
    )
    monkeypatch.setattr(module, "_encode", lambda *args, **kwargs: 0)

    assert (
        module.render_snapshots(
            snapshot, tmp_path / "movie.mp4", fps=2, frames_only=True, keep_frames=True
        )
        == 0
    )
    assert len(rendered) == 2
    assert rendered[0][0][0].shape == (6, 8)
    assert rendered[0][0][1].shape == (8, 10)
    assert rendered[0][0][2].nfp == 6
    np.testing.assert_allclose(rendered[0][0][2].cylindrical_R_profile, [10.0, 10.1])
    assert rendered[0][1]["extent"] == (20.0, 30.0)


def test_movie_encode_caps_lightweight_width(tmp_path, monkeypatch):
    module = _movie_tool("gkx_movie_encode_test")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return module.subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert (
        module._encode(
            tmp_path,
            [tmp_path / "frame_0000.png"],
            tmp_path / "movie.mp4",
            5,
            False,
            True,
        )
        == 0
    )
    assert commands[0][commands[0].index("-vf") + 1] == "scale=900:-2"


def test_movie_restores_campaign_state_and_absolute_time(tmp_path):
    module = _movie_tool("gkx_movie_state_test")

    shape = (1, 2, 2, 4, 4, 4)
    state = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).astype(complex)
    archive = tmp_path / "state.npz"
    np.savez_compressed(archive, state=state, t_end=250.0, saturated=True)
    restored, t_start, saturated, label = module._movie_initial_state(
        archive, shape=shape, dtype=np.complex64, seed=0, amplitude=1e-3
    )

    np.testing.assert_allclose(restored, state)
    assert (t_start, saturated, label) == (250.0, True, str(archive))
    with pytest.raises(ValueError, match="movie grid shape"):
        module._movie_initial_state(
            archive,
            shape=shape[:-1] + (5,),
            dtype=np.complex64,
            seed=0,
            amplitude=1e-3,
        )


def test_movie_uses_deck_moment_resolution_unless_overridden():
    module = _movie_tool("gkx_movie_moments_test")

    raw = {"run": {"Nl": 6, "Nm": 10}}
    assert module._movie_moment_dims(raw, None, None) == (6, 10)
    assert module._movie_moment_dims(raw, 3, 5) == (3, 5)
    assert module._movie_moment_dims({}, None, None) == (4, 8)


def test_movie_imported_geometry_requires_physical_profiles():
    module = _movie_tool("gkx_movie_geometry_test")

    with pytest.raises(RuntimeError, match="require physical R, Z"):
        module._require_movie_geometry_profiles(object(), model="vmec")
    geometry = type(
        "Geometry",
        (),
        {
            "cylindrical_R_profile": np.array([1.0, 1.1]),
            "cylindrical_Z_profile": np.array([0.0, 0.1]),
            "toroidal_angle_profile": np.array([0.0, 0.2]),
        },
    )()
    module._require_movie_geometry_profiles(geometry, model="vmec")
    module._require_movie_geometry_profiles(object(), model="s-alpha")


def test_phi_xy_snapshot_figure_renders(tmp_path):
    _real, spectral = _synthetic_spectral_phi()
    grid = SimpleNamespace(x0=1.6, y0=2.4)
    out = tmp_path / "phi_xy_snapshot.png"
    fig, ax = snapshots.phi_xy_snapshot_figure(
        spectral, grid, ny_full=10, time=12.5, out=out
    )
    plt.close(fig)
    assert out.exists()
    assert ax.get_xlabel() == r"$x/\rho_i$"


def test_flux_tube_3d_figure_renders(tmp_path):
    real_xyz, _spectral = _synthetic_spectral_phi()
    geom = SimpleNamespace(q=1.4, epsilon=0.18, R0=2.78, nfp=1)
    out = tmp_path / "flux_tube_3d.png"
    fig, _ax3d = snapshots.flux_tube_3d_figure(real_xyz, geom, time=12.5, out=out)
    plt.close(fig)
    assert out.exists()


def test_flux_tube_uses_imported_cylindrical_field_line() -> None:
    R = np.asarray([4.0, 5.0, 4.5])
    Z = np.asarray([-0.2, 0.1, 0.4])
    zeta = np.asarray([-0.5, 0.0, 0.7])
    geom = SimpleNamespace(
        q=1.4,
        epsilon=0.18,
        R0=2.78,
        cylindrical_R_profile=R,
        cylindrical_Z_profile=Z,
        toroidal_angle_profile=zeta,
    )

    centre, *_ = snapshots._field_line_tube(geom, samples=3)

    np.testing.assert_allclose(centre[:, 0], R * np.cos(zeta))
    np.testing.assert_allclose(centre[:, 1], R * np.sin(zeta))
    np.testing.assert_allclose(centre[:, 2], Z)


def _colorbar_axes(fig, main_ax):
    extra = [ax for ax in fig.axes if ax is not main_ax]
    assert extra, "expected a colorbar axes"
    return extra[0]


def test_small_amplitude_colorbar_states_its_decade_in_the_label():
    """A 1e-5 field must not park offset text where the title ends up.

    matplotlib would otherwise print a floating "1e-5" above the colorbar, in
    the same corner the left-aligned title reaches once a time stamp is
    appended -- the two overprinted each other on real saturated output.
    """

    _real, spectral = _synthetic_spectral_phi()
    fig, ax = snapshots.phi_xy_snapshot_figure(1.0e-5 * spectral, ny_full=10, time=12.5)
    fig.canvas.draw()  # offset text is only populated at draw time
    bar_ax = _colorbar_axes(fig, ax)
    assert r"\times 10^{-5}" in bar_ax.get_ylabel()
    assert bar_ax.yaxis.get_offset_text().get_text() == ""
    plt.close(fig)


def test_order_one_colorbar_keeps_a_plain_label():
    _real, spectral = _synthetic_spectral_phi()
    fig, ax = snapshots.phi_xy_snapshot_figure(spectral, ny_full=10)
    fig.canvas.draw()
    bar_ax = _colorbar_axes(fig, ax)
    assert r"\times" not in bar_ax.get_ylabel()
    assert bar_ax.yaxis.get_offset_text().get_text() == ""
    plt.close(fig)


def test_flux_spectra_annotation_gets_clear_headroom():
    """The averaging-window box must not be drawn over the spectrum itself."""

    diag, ky, kx = _memory_nonlinear_diag()
    fig, axes = flux_spectra_figure(diag, ky=ky, kx=kx, window=(10.0, 30.0))
    ky_ax = axes[0]
    drawn = np.concatenate(
        [np.asarray(line.get_ydata(), dtype=float) for line in ky_ax.get_lines()]
    )
    assert ky_ax.get_ylim()[1] > float(np.nanmax(drawn))
    plt.close(fig)


# ---------------------------------------------------------------------------
# The one-page run summary: every panel drawn by the function that owns it.
# ---------------------------------------------------------------------------


def _write_final_field_bundle(base, *, nx=8, ny=6, nz=4, t_last=12.0):
    """Write the ``*.big.nc`` companion that carries a run's final fields."""

    netcdf4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(5)
    phi_yxz = rng.normal(size=(ny, nx, nz)) * 1e-3
    path = pathlib.Path(f"{base}.big.nc")
    with netcdf4.Dataset(path, "w") as root:
        for name, size in (("time", 1), ("x", nx), ("y", ny), ("theta", nz)):
            root.createDimension(name, size)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([t_last])
        grids.createVariable("x", "f4", ("x",))[:] = np.linspace(
            0.0, 40.0, nx, endpoint=False
        )
        grids.createVariable("y", "f4", ("y",))[:] = np.linspace(
            0.0, 30.0, ny, endpoint=False
        )
        geom = root.createGroup("Geometry")
        for name, value in (("q", 1.4), ("rmaj", 3.0), ("aminor", 0.5), ("nfp", 3)):
            geom.createVariable(name, "f4", ())[:] = np.float32(value)
        diag = root.createGroup("Diagnostics")
        diag.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[0, ...] = (
            phi_yxz
        )
    return phi_yxz, path


def test_run_summary_figure_draws_every_panel_from_a_bundle(tmp_path):
    """The summary page carries the traces, both spectra, the map, and the text."""

    from gkx.artifacts.run_summary import nonlinear_summary_figure

    base = tmp_path / "case"
    _write_synthetic_out_nc(tmp_path / "case.out.nc")
    _write_final_field_bundle(base)
    (tmp_path / "case.toml").write_text(
        '[geometry]\nmodel = "vmec"\nvmec_file = "wout_demo.nc"\ntorflux = 0.64\n'
        "[grid]\nNx = 8\nNy = 6\nNz = 4\n"
        "[run]\nNl = 2\nNm = 3\n",
        encoding="utf-8",
    )
    out = tmp_path / "case.summary.png"

    fig, axes = nonlinear_summary_figure(
        tmp_path / "case.out.nc", window=(6.0, 12.0), saturated=True, out=out
    )
    try:
        assert out.exists()
        assert set(axes) == {
            "heat_flux",
            "particle_flux",
            "metadata",
            "flux_spectrum",
            "phi2_spectrum",
            "potential",
        }
        # Every data panel carries a labelled axis, and the potential a colorbar.
        assert axes["heat_flux"].get_ylabel()
        assert axes["particle_flux"].get_xlabel()
        assert axes["flux_spectrum"].get_xlabel()
        assert axes["phi2_spectrum"].get_ylabel()
        assert axes["potential"].images
        assert axes["potential"].get_xlabel() and axes["potential"].get_ylabel()
        extras = [ax for ax in fig.axes if ax not in axes.values()]
        assert extras and extras[0].get_ylabel(), "potential needs a labelled colorbar"
        # The window is shaded on both traces rather than only annotated.
        assert axes["heat_flux"].patches and axes["particle_flux"].patches
        # The metadata panel is text-only and names the deck and equilibrium.
        assert not axes["metadata"].axison
        text = " ".join(item.get_text() for item in axes["metadata"].texts)
        assert "wout_demo.nc" in text
        assert "case.toml" in text
        assert "8 x 6 x 4" in text
        assert "measured saturation" in text
    finally:
        plt.close(fig)


def test_run_summary_does_not_plot_a_rejected_window(tmp_path):
    """Keep a failed stop-policy window in metadata, not in plotted averages."""

    from gkx.artifacts.run_summary import nonlinear_summary_figure

    _write_synthetic_out_nc(tmp_path / "capped.out.nc")
    _write_final_field_bundle(tmp_path / "capped")

    fig, axes = nonlinear_summary_figure(
        tmp_path / "capped.out.nc", window=(6.0, 12.0), saturated=False
    )
    try:
        assert not axes["heat_flux"].patches
        assert not axes["particle_flux"].patches
        text = " ".join(item.get_text() for item in axes["metadata"].texts)
        assert "NOT saturated" in text
        assert "[6, 12]" in text
    finally:
        plt.close(fig)


def test_run_summary_figure_survives_a_bundle_without_spectra(tmp_path):
    """A CSV sidecar still gets a page; the panels it cannot draw say why."""

    from gkx.artifacts.run_summary import nonlinear_summary_figure

    base = _write_csv_sidecar(tmp_path, name="thin_case")
    out = tmp_path / "thin.summary.png"

    fig, axes = nonlinear_summary_figure(base.with_suffix(".diagnostics.csv"), out=out)
    try:
        assert out.exists()
        assert axes["heat_flux"].get_lines()
        for key in ("flux_spectrum", "phi2_spectrum", "potential"):
            note = " ".join(item.get_text() for item in axes[key].texts)
            assert "out.nc" in note or "final-field" in note
        text = " ".join(item.get_text() for item in axes["metadata"].texts)
        assert "second half" in text
    finally:
        plt.close(fig)


def test_run_summary_final_field_undoes_the_writer_normalization(tmp_path):
    """``PhiXY`` is stored through ifft2, so the amplitude needs its Ny*Nx back."""

    from gkx.artifacts.run_summary import load_final_field

    base = tmp_path / "amp"
    phi_yxz, _path = _write_final_field_bundle(base, nx=8, ny=6)
    field = load_final_field(tmp_path / "amp.out.nc")

    expected = np.transpose(phi_yxz, (1, 0, 2)) * (6 * 8)
    assert np.allclose(field.phi_xyz, expected, rtol=1e-5, atol=1e-9)
    # The stored axes drop the repeated endpoint, so the box length is the last
    # sample plus one spacing -- the extent the writer was handed.
    assert field.extent == pytest.approx((40.0, 30.0))
    assert field.geometry.nfp == 3
    assert field.geometry.epsilon == pytest.approx(0.5 / 3.0)
    assert field.time == pytest.approx(12.0)


def test_flux_tube_figure_renders_from_a_saved_bundle(tmp_path):
    """The 3-D tube reads the same companion and labels its own amplitude."""

    from gkx.artifacts.run_summary import flux_tube_figure

    _write_final_field_bundle(tmp_path / "tube")
    out = tmp_path / "tube.flux_tube_3d.png"

    fig, ax3d = flux_tube_figure(tmp_path / "tube.out.nc", out=out)
    try:
        assert out.exists()
        assert ax3d.collections
        assert r"c_s/a" in ax3d.get_title()
    finally:
        plt.close(fig)


def test_embedded_spectra_panels_leave_the_host_figure_alone(tmp_path):
    """``axes=``/``panels=`` draw into a caller's figure without owning it."""

    path = _write_synthetic_out_nc(tmp_path / "embed.out.nc")
    fig, (left, right) = plt.subplots(1, 2)
    try:
        flux_spectra_figure(path, panels=("ky",), axes=(left,), out=tmp_path / "no.png")
        phi2_spectra_figure(path, panels=("ky",), axes=(right,))
        # No suptitle, and nothing saved: layout and output stay the caller's.
        assert not (tmp_path / "no.png").exists()
        assert fig._suptitle is None
        assert left.get_lines() and right.get_lines()
        with pytest.raises(ValueError, match="axes="):
            flux_spectra_figure(path, panels=("ky", "kx"), axes=(left,))
        with pytest.raises(ValueError, match="unknown panels"):
            phi2_spectra_figure(path, panels=("nope",), axes=(right,))
    finally:
        plt.close(fig)


def test_summary_does_not_call_an_unsaturated_window_a_measurement() -> None:
    """A run that hit its cap must not read as converged.

    The stop policy records the window it evaluated whether or not the trace
    settled in it, so the window alone cannot distinguish the two. Labelling a
    capped run "measured saturation" tells a reader the flux converged while it
    was still climbing.
    """

    from gkx.artifacts.run_figures import measured_window_is_saturated

    assert measured_window_is_saturated({"saturation": {"saturated": False}}) is False
    assert measured_window_is_saturated({"saturation": {"saturated": True}}) is True
    assert measured_window_is_saturated({}) is None


def test_loglinear_window_does_not_depend_on_a_constant_phase() -> None:
    """A purely growing mode (constant phase) and the same mode rotating at a
    fixed real frequency have identical log-amplitudes and exactly linear
    phases, so the log-linear search must pick the same window for both."""

    from gkx.diagnostics.analysis import select_fit_window_loglinear

    t = np.linspace(0.0, 20.0, 200)
    envelope = np.exp(0.3 * t) * (1.0 + 0.5 * np.exp(-t))
    rotating = select_fit_window_loglinear(t, envelope * np.exp(-0.7j * t))
    growing = select_fit_window_loglinear(t, envelope.astype(complex))
    assert growing == rotating
    assert growing[1] - growing[0] > 5.0
