"""Plotting utilities should generate figures without errors."""

import matplotlib

matplotlib.use("Agg")

import pathlib

import numpy as np

from gkx.benchmarking.shared import CycloneReference, CycloneScanResult
import matplotlib.pyplot as plt
import pytest
import gkx.artifacts.plotting as plotting
from gkx.artifacts.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    eigenfunction_reference_overlay_figure,
    eigenfunction_overlap_summary_figure,
    etg_trend_figure,
    growth_rate_heatmap,
    growth_fit_figure,
    linear_validation_figure,
    linear_validation_multi_reference_figure,
    LinearValidationPanel,
    MultiReferenceValidationPanel,
    ReferenceSeries,
    linear_runtime_panel_figure,
    nonlinear_runtime_panel_figure,
    plot_saved_output,
    scan_comparison_figure,
    scan_multi_reference_figure,
    zonal_flow_response_figure,
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


def test_etg_trend_figure(tmp_path):
    """ETG trend plot should render and save."""
    R = np.array([4.0, 6.0, 8.0])
    gamma = np.array([0.1, 0.2, 0.3])
    omega = np.array([-0.4, -0.5, -0.6])
    fig, _axes = etg_trend_figure(R, gamma, omega, ky_target=3.0)
    out = tmp_path / "etg_trend.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_growth_rate_heatmap(tmp_path):
    """Heatmap plot should render and save."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    gamma = np.random.random((y.size, x.size))
    fig, _ax = growth_rate_heatmap(x, y, gamma, "Test", r"$R/L_n$", r"$R/L_T$")
    out = tmp_path / "heatmap.png"
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


def test_eigenfunction_reference_overlay_figure(tmp_path):
    theta = np.linspace(-np.pi, np.pi, 32)
    reference = np.cos(theta) + 1j * 0.25 * np.sin(theta)
    trial = reference * np.exp(1j * 0.41)

    fig, _axes = eigenfunction_reference_overlay_figure(
        theta,
        trial,
        theta,
        reference,
        title="KBM overlay",
    )
    out = tmp_path / "eigenfunction_overlay.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_eigenfunction_reference_overlay_figure_rejects_shape_mismatch():
    theta = np.linspace(-1.0, 1.0, 8)
    with pytest.raises(ValueError):
        eigenfunction_reference_overlay_figure(
            theta,
            np.ones(8, dtype=np.complex128),
            theta[:-1],
            np.ones(7, dtype=np.complex128),
        )


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


def test_linear_validation_multi_reference_figure(tmp_path):
    z = np.linspace(-1.0, 1.0, 8)
    panel = MultiReferenceValidationPanel(
        name="Cyclone",
        z=z,
        eigenfunction=np.exp(1j * z),
        x=np.array([0.2, 0.3]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        x_label=r"$k_y$",
        references=[
            ReferenceSeries(
                label="RefA",
                x=np.array([0.2, 0.3]),
                gamma=np.array([0.11, 0.21]),
                omega=np.array([0.31, 0.41]),
                color="#1f77b4",
            )
        ],
        log_x=True,
    )
    fig, axes = linear_validation_multi_reference_figure([panel])
    out = tmp_path / "linear_validation_multi.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0, 1].get_xscale() == "log"


def test_linear_validation_multi_reference_figure_empty():
    with np.testing.assert_raises(ValueError):
        linear_validation_multi_reference_figure([])


def test_scan_multi_reference_figure(tmp_path):
    x = np.array([0.1, 0.2, 0.4])
    refs = [
        ReferenceSeries(
            label="GX",
            x=x,
            gamma=np.array([0.21, 0.31, 0.41]),
            omega=np.array([-0.11, -0.21, -0.31]),
            color="#1f77b4",
        )
    ]
    fig, axes = scan_multi_reference_figure(
        x,
        np.array([0.2, 0.3, 0.4]),
        np.array([-0.1, -0.2, -0.3]),
        r"$k_y$",
        "Multi-ref",
        refs,
        log_x=True,
    )
    out = tmp_path / "scan_multi_reference.png"
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


def test_eigenfunction_overlap_summary_figure(tmp_path):
    ky = np.array([0.1, 0.2, 0.4])
    fig, axes = eigenfunction_overlap_summary_figure(
        ky,
        np.array([0.98, 0.95, 0.93]),
        np.array([0.05, 0.08, 0.10]),
        title="KBM overlap audit",
    )
    out = tmp_path / "eig_overlap.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0].get_xscale() == "log"


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

import gkx.artifacts.snapshots as snapshots
from gkx.artifacts.transport_figures import (
    flux_spectra_figure,
    heat_flux_time_figure,
    phi2_spectra_figure,
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


def test_run_summary_figure_survives_a_bundle_without_spectra(tmp_path):
    """A CSV sidecar still gets a page; the panels it cannot draw say why."""

    from gkx.artifacts.run_summary import nonlinear_summary_figure

    base = _write_csv_sidecar(tmp_path, name="thin_case")
    out = tmp_path / "thin.summary.png"

    fig, axes = nonlinear_summary_figure(
        base.with_suffix(".diagnostics.csv"), out=out
    )
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
