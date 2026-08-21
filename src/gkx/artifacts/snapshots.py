"""Real-space electrostatic-potential snapshots without solver dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np

from gkx.artifacts.figure_style import figure_style, save_figure

#: Shipped decks use ``diagnostic_norm = "rho_star"``; this is the
#: rho-star-normalized potential, not ``ephi/T_i``.
PHI_LABEL: str = r"$(e\phi/T_i)\,/\,\rho_*$"


def potential_real_space(
    phi_spectral: np.ndarray, *, ny_full: int | None = None
) -> np.ndarray:
    """Transform a spectral potential ``phi(ky, kx, z)`` to real ``phi(x, y, z)``.

    ``ky`` uses the nonlinear bracket's compressed real-FFT layout; ``kx`` is
    a full complex axis. Accepts either the full Hermitian ``ky`` array or its
    non-negative block; pass ``ny_full`` with the compressed form.
    """

    phi = np.asarray(phi_spectral)
    if phi.ndim != 3:
        raise ValueError("phi_spectral must have shape (ky, kx, z)")
    if ny_full is None:
        ny_full = int(phi.shape[0])
    ny_full = int(ny_full)
    nyc = ny_full // 2 + 1
    if phi.shape[0] == ny_full:
        phi = phi[:nyc]
    elif phi.shape[0] != nyc:
        raise ValueError(
            f"phi_spectral has {phi.shape[0]} ky rows; expected the full axis "
            f"({ny_full}) or the compressed real-FFT block ({nyc})"
        )
    nkx = phi.shape[1]
    scale = float(ny_full * nkx)
    real = np.fft.irfft2(phi, s=(nkx, ny_full), axes=(-2, -3)) * scale
    return np.transpose(real, (1, 0, 2))  # (y, x, z) -> (x, y, z)


def _as_real_space(phi: np.ndarray, *, ny_full: int | None = None) -> np.ndarray:
    """Return ``phi(x, y, z)``, transforming complex spectral input if needed."""

    arr = np.asarray(phi)
    if np.iscomplexobj(arr):
        return potential_real_space(arr, ny_full=ny_full)
    if arr.ndim != 3:
        raise ValueError("phi must have shape (x, y, z) in real space")
    return arr.astype(float, copy=False)


def _grid_extent(grid: Any | None) -> tuple[float, float] | None:
    """Perpendicular box size ``(Lx, Ly)`` in rho_i from a spectral grid."""

    if grid is None:
        return None
    x0 = getattr(grid, "x0", None)
    y0 = getattr(grid, "y0", None)
    if x0 is None or y0 is None:
        return None
    return 2.0 * np.pi * float(x0), 2.0 * np.pi * float(y0)


def _field_line_tube(geometry: Any, samples: int, *, turns: float = 1.5):
    """Cartesian field line and local frame; prefer imported physical coordinates."""

    q = float(getattr(geometry, "q", 1.4) or 1.4)
    epsilon = float(getattr(geometry, "epsilon", 0.18) or 0.18)
    major = float(getattr(geometry, "R0", 3.0) or 3.0)
    minor = epsilon * major
    R = getattr(geometry, "cylindrical_R_profile", None)
    Z = getattr(geometry, "cylindrical_Z_profile", None)
    zeta = getattr(geometry, "toroidal_angle_profile", None)
    if R is not None and Z is not None and zeta is not None:
        source = np.linspace(0.0, 1.0, len(R))
        target = np.linspace(0.0, 1.0, samples)
        R_line = np.interp(target, source, np.asarray(R, dtype=float))
        Z_line = np.interp(target, source, np.asarray(Z, dtype=float))
        zeta_line = np.interp(
            target, source, np.unwrap(np.asarray(zeta, dtype=float))
        )
        centre = np.stack(
            [R_line * np.cos(zeta_line), R_line * np.sin(zeta_line), Z_line],
            axis=-1,
        )
        major = float(np.mean(R_line))
        Z_mean = float(np.mean(Z_line))
        minor = max(float(np.max(np.hypot(R_line - major, Z_line - Z_mean))), 1.0e-6)
        radial_R = R_line - major
        radial_Z = Z_line - Z_mean
    else:
        theta = np.linspace(-turns * np.pi, turns * np.pi, samples)
        zeta_line = q * theta
        radius = major + minor * np.cos(theta)
        centre = np.stack([radius * np.cos(zeta_line), radius * np.sin(zeta_line),
                           minor * np.sin(theta)], axis=-1)
        radial_R = minor * np.cos(theta)
        radial_Z = minor * np.sin(theta)
    tangent = np.gradient(centre, axis=0)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True) + 1e-30
    outward = np.stack(
        [np.cos(zeta_line) * radial_R, np.sin(zeta_line) * radial_R, radial_Z],
        axis=-1,
    )
    outward -= tangent * np.sum(outward * tangent, axis=-1, keepdims=True)
    outward /= np.linalg.norm(outward, axis=-1, keepdims=True) + 1e-30
    binormal = np.cross(tangent, outward)
    return centre, outward, binormal, minor, major


def _torus_wireframe(major: float, minor: float, n_major: int = 60, n_minor: int = 18):
    """A faint torus for spatial context behind the tube."""

    u = np.linspace(0.0, 2.0 * np.pi, n_major)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor)
    uu, vv = np.meshgrid(u, v, indexing="ij")  # type: np.ndarray, np.ndarray
    r = major + minor * np.cos(vv)
    return r * np.cos(uu), r * np.sin(uu), minor * np.sin(vv)


def _decade_factor(scale: float) -> tuple[float, str]:
    """Power of ten to fold into a colorbar label, as ``(factor, label)``."""

    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0, ""
    exponent = int(np.floor(np.log10(float(scale))))
    # Inside this band the plain tick labels are short enough to read as they
    # are ("0.04", "137.7"); outside it they would need an exponent.
    if -2 <= exponent <= 3:
        return 1.0, ""
    return 10.0**exponent, rf"$\times 10^{{{exponent}}}$"


def label_amplitude_colorbar(bar: Any, scale: float, phi_label: str) -> None:
    """Label a ``phi`` colorbar, folding any power of ten into the label text.

    Left to itself matplotlib renders the exponent of a small-amplitude field
    as *offset text* floating above the colorbar -- and that corner is also
    where a left-aligned axes title ends up once a time stamp is appended to
    it. On a saturated run whose amplitude is order ``1e-5`` the two collide
    outright: the ``1e-5`` prints on top of the ``t c_s/a = 12.0``.

    Stating the decade in the colorbar label instead is collision-proof
    regardless of how long the title runs, and is what a published figure does
    anyway -- the reader gets the units in the one place they are looking for
    them rather than as a superscript hovering elsewhere.
    """

    factor, factor_label = _decade_factor(scale)
    axis = bar.ax.yaxis if bar.orientation == "vertical" else bar.ax.xaxis
    if factor_label:
        axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / factor:g}"))
        bar.set_label(f"{phi_label}  [{factor_label}]")
        return
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.set_major_formatter(formatter)
    bar.set_label(phi_label)


def draw_phi_xy_cut(
    ax: plt.Axes,
    phi_xy: np.ndarray,
    *,
    scale: float,
    extent: tuple[float, float] | None = None,
    phi_label: str = PHI_LABEL,
    cmap: str = "RdBu_r",
    colorbar: bool = True,
):
    """Render one perpendicular ``phi(x, y)`` cut onto an existing axes.

    ``phi_xy`` is indexed ``(x, y)``. With ``extent = (Lx, Ly)`` the axes carry
    physical ``rho_i`` units; without it they fall back to grid indices and are
    labelled as such rather than pretending to a unit they do not have.
    """

    phi_arr = np.asarray(phi_xy, dtype=float)
    if phi_arr.ndim != 2:
        raise ValueError("phi_xy must be a 2D (x, y) cut")
    imshow_extent = None if extent is None else (0.0, extent[0], 0.0, extent[1])
    mesh = ax.imshow(
        phi_arr.T,
        origin="lower",
        cmap=cmap,
        vmin=-scale,
        vmax=scale,
        aspect="auto",
        interpolation="bilinear",
        extent=imshow_extent,
    )
    if extent is None:
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    else:
        ax.set_xlabel(r"$x/\rho_i$")
        ax.set_ylabel(r"$y/\rho_i$")
    ax.grid(False)
    if colorbar:
        bar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        label_amplitude_colorbar(bar, scale, phi_label)
    return mesh


def draw_flux_tube_3d(
    ax3d: Any,
    phi_xyz: np.ndarray,
    geometry: Any,
    *,
    scale: float,
    turns: float = 1.5,
    elev: float = 32.0,
    azim: float = -60.0,
    radius_fraction: float = 0.85,
    cmap: str = "RdBu_r",
    show_torus: bool = True,
) -> None:
    """Render physical imported coordinates, with an analytic fallback."""

    phi_arr = np.asarray(phi_xyz, dtype=float)
    if phi_arr.ndim != 3:
        raise ValueError("phi_xyz must have shape (x, y, z)")
    nx, ny, nz = phi_arr.shape

    # Resample along z so the tube is smooth even when the parallel grid is
    # coarse; nz is a physics resolution, not a rendering one.
    samples = max(4 * nz, 160)
    centre, outward, binormal, minor, major = _field_line_tube(
        geometry, samples, turns=turns
    )

    source_z = np.linspace(0.0, 1.0, nz)
    target_z = np.linspace(0.0, 1.0, samples)
    slab = phi_arr[nx // 2]  # (y, z)
    resampled = np.stack(
        [np.interp(target_z, source_z, slab[row]) for row in range(ny)], axis=0
    )

    angle = np.linspace(0.0, 2.0 * np.pi, ny, endpoint=False)
    physical = getattr(geometry, "cylindrical_R_profile", None) is not None
    radius = radius_fraction * minor * (0.3 if physical else 1.0)
    surface = (
        centre[None, :, :]
        + radius * np.cos(angle)[:, None, None] * outward[None, :, :]
        + radius * np.sin(angle)[:, None, None] * binormal[None, :, :]
    )
    normed = 0.5 + 0.5 * np.clip(resampled / (scale + 1e-30), -1.0, 1.0)

    if show_torus and not physical:
        wire_x, wire_y, wire_z = _torus_wireframe(major, minor)
        ax3d.plot_wireframe(
            wire_x,
            wire_y,
            wire_z,
            rstride=6,
            cstride=3,
            color="#B8B8B8",
            linewidth=0.35,
            alpha=0.5,
        )
    ax3d.plot_surface(
        surface[..., 0],
        surface[..., 1],
        surface[..., 2],
        facecolors=plt.get_cmap(cmap)(normed),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    span = (major + minor) * 1.02
    ax3d.set_xlim(-span, span)
    ax3d.set_ylim(-span, span)
    ax3d.set_zlim(-span, span)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_axis_off()
    ax3d.view_init(elev=elev, azim=azim)


def phi_xy_snapshot_figure(
    phi_spectral: np.ndarray,
    grid: Any | None = None,
    geom: Any | None = None,
    *,
    z_index: int | None = None,
    scale: float | None = None,
    time: float | None = None,
    ny_full: int | None = None,
    extent: tuple[float, float] | None = None,
    phi_label: str = PHI_LABEL,
    # Short enough that the appended time stamp clears the colorbar's
    # power-of-ten offset text at the top-right of the figure.
    title: str = r"Outboard-midplane cut of $\phi$",
    out: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Publication x-y cut of the potential at the outboard midplane.

    ``phi_spectral`` is either the spectral field ``(ky, kx, z)`` (complex) or
    an already-transformed real-space field ``(x, y, z)``. ``grid`` supplies
    the perpendicular box size for physical ``rho_i`` axes; ``geom`` is
    accepted for API symmetry with :func:`flux_tube_3d_figure` and reserved
    for annotations. ``out`` optionally saves the figure via
    :func:`gkx.artifacts.figure_style.save_figure` (the figure stays open).
    """

    del geom  # geometry does not enter the perpendicular cut
    phi_xyz = _as_real_space(phi_spectral, ny_full=ny_full)
    nz = phi_xyz.shape[2]
    cut_index = nz // 2 if z_index is None else int(z_index)
    midplane = phi_xyz[:, :, cut_index]
    vmax = float(np.abs(midplane).max()) if scale is None else float(scale)
    vmax = max(vmax, 1e-30)
    if extent is None:
        extent = _grid_extent(grid)

    with figure_style():
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        draw_phi_xy_cut(ax, midplane, scale=vmax, extent=extent, phi_label=phi_label)
        label = title
        if time is not None:
            label = f"{title}    $t\\,c_s/a = {float(time):.1f}$"
        ax.set_title(label)
        fig.tight_layout()
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, ax


def flux_tube_3d_figure(
    phi_spectral: np.ndarray,
    geom: Any,
    *,
    grid: Any | None = None,
    scale: float | None = None,
    time: float | None = None,
    ny_full: int | None = None,
    turns: float = 1.5,
    elev: float = 32.0,
    azim: float = -60.0,
    phi_label: str = PHI_LABEL,
    title: str = r"Flux tube along $\mathbf{B}$",
    out: str | Path | None = None,
) -> Tuple[plt.Figure, Any]:
    """Render the field-aligned potential, using host-only physical coordinates."""

    del grid  # perpendicular box size does not enter the 3D rendering
    phi_xyz = _as_real_space(phi_spectral, ny_full=ny_full)
    vmax = float(np.abs(phi_xyz).max()) if scale is None else float(scale)
    vmax = max(vmax, 1e-30)

    with figure_style():
        fig = plt.figure(figsize=(7.2, 6.0))
        ax3d = fig.add_subplot(1, 1, 1, projection="3d")
        draw_flux_tube_3d(
            ax3d,
            phi_xyz,
            geom,
            scale=vmax,
            turns=turns,
            elev=elev,
            azim=azim,
        )
        label = title
        if time is not None:
            label = f"{title}    $t\\,c_s/a = {float(time):.1f}$"
        ax3d.set_title(label, y=0.97)
        mappable = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
        )
        mappable.set_array(np.array([]))
        bar = fig.colorbar(mappable, ax=ax3d, fraction=0.04, pad=0.02, shrink=0.8)
        label_amplitude_colorbar(bar, vmax, phi_label)
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, ax3d


__all__ = [
    "PHI_LABEL",
    "draw_flux_tube_3d",
    "draw_phi_xy_cut",
    "flux_tube_3d_figure",
    "label_amplitude_colorbar",
    "phi_xy_snapshot_figure",
    "potential_real_space",
]
