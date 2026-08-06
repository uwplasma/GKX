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
    fit_growth_rate_with_stats,
    instantaneous_growth_rate_from_phi,
    windowed_growth_rate_from_omega_series,
    select_ky_index,
    select_fit_window,
    select_fit_window_loglinear,
)


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
        "gkx.diagnostics.analysis.fit_growth_rate_with_stats", _boom
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
        "gkx.diagnostics.analysis.fit_growth_rate_with_stats",
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
