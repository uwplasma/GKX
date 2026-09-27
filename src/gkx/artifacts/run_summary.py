"""The one figure a finished nonlinear run is read from.

The individual figures a run writes each answer one question. A person opening
a result directory for the first time has a different question -- *what
happened?* -- and answering it from six PNGs means knowing which to open in
which order. This module composes the single page that answers it: the flux
traces with the window the run actually measured, the ``ky`` spectra that say
which scales carry the transport, the real-space potential, and a text panel
naming the equilibrium, the resolution, the deck, and the saturated
``<Q> +/- SEM``.

Nothing here re-derives a figure. The panels are drawn by the same functions
that draw them standalone (:mod:`gkx.artifacts.transport_figures` and
:mod:`gkx.artifacts.run_summary`), through their ``axes=``/``panels=``
arguments, so the summary cannot drift away from the figures it summarizes.

The one thing this module does own is reading the *final field* sidecar. The
``*.out.nc`` history carries spectra but no potential; the potential lives in
the ``*.big.nc`` companion, stored through ``numpy.fft.ifft2`` and therefore
scaled by ``1/(Ny Nx)`` relative to the convention the solver and
:func:`gkx.artifacts.run_summary.potential_real_space` use. Undoing that here is
what keeps the amplitude on the colorbar the same number the rest of the run
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap
from types import SimpleNamespace
from typing import Any, Callable, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from gkx.artifacts.figure_style import (
    figure_style,
    panel_label,
    save_figure,
)
from gkx.artifacts.plotting import _artifact_base
from gkx.artifacts.transport_figures import (
    flux_spectra_figure,
    heat_flux_time_figure,
    phi2_spectra_figure,
)
from gkx.core_ky_layout import nyc_from_ny


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
    nyc = nyc_from_ny(ny_full)
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
        zeta_line = np.interp(target, source, np.unwrap(np.asarray(zeta, dtype=float)))
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
        centre = np.stack(
            [
                radius * np.cos(zeta_line),
                radius * np.sin(zeta_line),
                minor * np.sin(theta),
            ],
            axis=-1,
        )
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
    """Put scientific notation in the label, away from the figure title."""

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
    if physical:
        lower = np.min(surface, axis=(0, 1))
        upper = np.max(surface, axis=(0, 1))
        width = np.maximum(upper - lower, 1.0e-6)
        pad = 0.04 * width
        ax3d.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
        ax3d.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
        ax3d.set_zlim(lower[2] - pad[2], upper[2] + pad[2])
        ax3d.set_box_aspect(width)
    else:
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


#: Companion file holding the final real-space fields of a nonlinear run.
FINAL_FIELD_SUFFIX = ".big.nc"

_METADATA_LABEL_WIDTH = 12
# Wide enough for a VMEC ``wout_*.nc`` file name on one line: wrapping a
# filename mid-token is exactly the row a reader needs to read whole.
_METADATA_VALUE_WIDTH = 38


@dataclass(frozen=True)
class FinalField:
    """Final-time potential of a nonlinear run, in solver amplitude units."""

    phi_xyz: np.ndarray
    extent: tuple[float, float] | None
    geometry: Any
    time: float | None


def run_label(source: str | Path) -> str:
    """Return a title for one output bundle that identifies the run.

    The grouped run directory the equilibrium shorthand writes names every
    bundle inside it ``gkx``, so the file's own stem says nothing. Prefixing
    the directory turns the title back into the identity of the run.
    """

    base = _artifact_base(Path(source))
    parent = base.parent.name
    return f"{parent}/{base.name}" if parent not in ("", ".", "..") else base.name


def final_field_path(source: str | Path) -> Path:
    """Return the ``*.big.nc`` companion of any file in an output bundle."""

    return Path(f"{_artifact_base(Path(source))}{FINAL_FIELD_SUFFIX}")


def has_final_field(source: str | Path) -> bool:
    """Return whether the bundle carries the final fields the snapshots need."""

    return final_field_path(source).is_file()


def _axis_extent(values: np.ndarray | None) -> float | None:
    """Box length of a periodic axis sampled without its repeated endpoint."""

    if values is None or values.size < 2:
        return None
    return float(values[-1] + (values[1] - values[0]))


def load_final_field(source: str | Path) -> FinalField:
    """Read the final potential, box size, and geometry from ``*.big.nc``."""

    import netCDF4

    path = final_field_path(source)
    with netCDF4.Dataset(path) as root:
        diag = root.groups["Diagnostics"]
        # (time, y, x, theta) with a single stored time.
        phi_yxz = np.asarray(diag.variables["PhiXY"][0, ...], dtype=float)
        grids = root.groups.get("Grids")
        x_vals = y_vals = None
        time = None
        if grids is not None:
            if "x" in grids.variables:
                x_vals = np.asarray(grids.variables["x"][:], dtype=float)
            if "y" in grids.variables:
                y_vals = np.asarray(grids.variables["y"][:], dtype=float)
            if "time" in grids.variables:
                stamps = np.asarray(grids.variables["time"][:], dtype=float)
                time = float(stamps[-1]) if stamps.size else None
        geometry = _bundle_geometry(root.groups.get("Geometry"))

    # Undo the 1/(Ny Nx) of the writer's ifft2 so the amplitude matches the
    # convention every other GKX potential figure is drawn in.
    phi_xyz = np.transpose(phi_yxz, (1, 0, 2)) * float(
        phi_yxz.shape[0] * phi_yxz.shape[1]
    )
    lx, ly = _axis_extent(x_vals), _axis_extent(y_vals)
    extent = None if lx is None or ly is None else (lx, ly)
    return FinalField(phi_xyz=phi_xyz, extent=extent, geometry=geometry, time=time)


def _scalar(group: Any, name: str, default: float) -> float:
    if group is None or name not in group.variables:
        return float(default)
    try:
        return float(np.asarray(group.variables[name][...]).reshape(-1)[0])
    except (IndexError, ValueError):  # pragma: no cover - malformed bundle
        return float(default)


def _profile(group: Any, name: str) -> np.ndarray | None:
    if group is None or name not in group.variables:
        return None
    values = np.asarray(group.variables[name][...], dtype=float).reshape(-1)
    return values if values.size >= 2 and np.isfinite(values).all() else None


def _bundle_geometry(group: Any) -> Any:
    """Read plotting geometry from a nonlinear bundle."""

    major = _scalar(group, "rmaj", 3.0)
    minor = _scalar(group, "aminor", 0.18 * major)
    return SimpleNamespace(
        q=_scalar(group, "q", 1.4),
        epsilon=(minor / major) if major else 0.18,
        R0=major,
        nfp=max(int(_scalar(group, "nfp", 1.0)), 1),
        cylindrical_R_profile=_profile(group, "Rplot"),
        cylindrical_Z_profile=_profile(group, "Zplot"),
        toroidal_angle_profile=_profile(group, "zeta_plot"),
    )


def _read_deck(source: str | Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the resolved deck written beside the output, when there is one."""

    path = Path(f"{_artifact_base(Path(source))}.toml")
    if not path.is_file():
        return None, None
    import tomllib

    try:
        return tomllib.loads(path.read_text(encoding="utf-8")), path
    except (OSError, tomllib.TOMLDecodeError):
        return None, path


def _root_resolution(source: str | Path) -> dict[str, int]:
    """Grid sizes recorded in the NetCDF root, used when no deck is present."""

    path = Path(source)
    if path.suffix.lower() != ".nc":
        return {}
    import netCDF4

    names = ("nx", "ny", "ntheta", "nlaguerre", "nhermite", "nspecies")
    try:
        with netCDF4.Dataset(path) as root:
            return {
                name: int(np.asarray(root.variables[name][...]).reshape(-1)[0])
                for name in names
                if name in root.variables
            }
    except OSError:  # pragma: no cover - unreadable bundle
        return {}


def _table(deck: dict[str, Any] | None, name: str) -> dict[str, Any]:
    value = (deck or {}).get(name)
    return value if isinstance(value, dict) else {}


def _resolution_lines(
    deck: dict[str, Any] | None, root: dict[str, int]
) -> list[tuple[str, str]]:
    """Perpendicular/parallel grid and velocity-space resolution, when known."""

    grid, run = _table(deck, "grid"), _table(deck, "run")
    nx = grid.get("Nx", root.get("nx"))
    ny = grid.get("Ny", root.get("ny"))
    nz = grid.get("Nz", root.get("ntheta"))
    nl = run.get("Nl", root.get("nlaguerre"))
    nm = run.get("Nm", root.get("nhermite"))
    lines: list[tuple[str, str]] = []
    if None not in (nx, ny, nz):
        lines.append(("grid", f"Nx x Ny x Nz = {nx} x {ny} x {nz}"))
    if None not in (nl, nm):
        lines.append(("moments", f"Nl x Nm = {nl} x {nm}"))
    return lines


def _flux_lines(
    diag: Any,
    window: tuple[float, float] | None,
    measured: bool,
    saturated: bool | None = None,
) -> list[tuple[str, str]]:
    """Stop time, averaging window, and the windowed mean +/- SEM of Q and Gamma."""

    from gkx.artifacts.transport_figures import _mean_sem, _time_window

    t = np.asarray(diag.t, dtype=float)
    lines: list[tuple[str, str]] = []
    if t.size:
        lines.append(("stop time", f"t cs/a = {float(t[-1]):.4g}  ({t.size} samples)"))
    try:
        mask, tmin, tmax = _time_window(t, window)
    except ValueError:  # pragma: no cover - empty or unusable trace
        return lines
    lines.append(("window", f"t in [{tmin:.4g}, {tmax:.4g}]"))
    if not measured:
        origin = "second half (none recorded)"
    elif saturated is False:
        origin = "stop policy, NOT saturated (cap reached)"
    elif saturated is True:
        origin = "measured saturation"
    else:
        origin = "stop policy window"
    lines.append(("window from", origin))
    for label, values in (
        ("<Q>/Q_gB", np.asarray(diag.heat_flux_t, dtype=float)),
        ("<G>/G_gB", np.asarray(diag.particle_flux_t, dtype=float)),
    ):
        mean, sem = _mean_sem(values[mask])
        lines.append((label, f"{mean:.4g} +/- {sem:.2g}"))
    return lines


def summary_metadata_lines(
    source: str | Path,
    diag: Any,
    *,
    window: tuple[float, float] | None,
    measured_window: bool,
    saturated: bool | None = None,
) -> list[tuple[str, str]]:
    """Assemble the ``(label, value)`` rows of the metadata panel.

    Every row is optional: the deck, the NetCDF root, and the diagnostics are
    read for what they happen to carry, because this figure has to render for a
    bundle produced by any of the output paths, not only the equilibrium
    shorthand that writes all three.
    """

    deck, deck_path = _read_deck(source)
    geometry = _table(deck, "geometry")
    lines: list[tuple[str, str]] = []
    equilibrium = geometry.get("vmec_file") or geometry.get("geometry_file")
    if equilibrium:
        lines.append(("equilibrium", Path(str(equilibrium)).name))
    if geometry.get("model"):
        detail = f"model = {geometry['model']}"
        if geometry.get("torflux") is not None:
            detail += f", torflux = {geometry['torflux']}"
        lines.append(("geometry", detail))
    lines += _resolution_lines(deck, _root_resolution(source))
    species = (deck or {}).get("species")
    if isinstance(species, list) and species:
        kinetic = sum(1 for entry in species if entry.get("kinetic", True))
        lines.append(("species", f"{len(species)} ({kinetic} kinetic)"))
    lines += _flux_lines(diag, window, measured_window, saturated)
    if deck_path is not None:
        lines.append(("input deck", deck_path.name))
    lines.append(("output", Path(source).name))
    return lines


def _wrapped_metadata_text(lines: Sequence[tuple[str, str]]) -> str:
    """Render ``(label, value)`` rows as a fixed-width two-column block."""

    rendered: list[str] = []
    pad = " " * _METADATA_LABEL_WIDTH
    for label, value in lines:
        chunks = textwrap.wrap(str(value), width=_METADATA_VALUE_WIDTH) or [""]
        rendered.append(f"{label:<{_METADATA_LABEL_WIDTH}}{chunks[0]}")
        rendered.extend(f"{pad}{chunk}" for chunk in chunks[1:])
    return "\n".join(rendered)


def draw_metadata_panel(
    ax: plt.Axes, lines: Sequence[tuple[str, str]], *, heading: str = "run summary"
) -> None:
    """Render the text panel of the summary figure onto ``ax``."""

    ax.set_axis_off()
    ax.text(
        0.0,
        1.0,
        heading,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.0,
        0.97,
        _wrapped_metadata_text(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        linespacing=1.6,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#F6F6F6",
            "edgecolor": "#CCCCCC",
            "linewidth": 0.8,
        },
    )


def _panel_or_note(draw: Callable[[], Any], ax: plt.Axes, *rest: plt.Axes) -> None:
    """Draw one panel, replacing it with its own reason when it cannot be drawn.

    A summary page is a report on a run, so a bundle that does not carry one of
    the inputs -- a CSV sidecar has no spectra, a run without saved fields has
    no potential -- should say which panel is missing and why, in the place the
    panel would have been, rather than cost the whole page.
    """

    try:
        draw()
    except Exception as exc:
        for blank in (ax, *rest):
            blank.clear()
            blank.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "\n".join(textwrap.wrap(str(exc), width=46)[:6]),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#555555",
        )


def _draw_potential_panel(ax: plt.Axes, source: str | Path) -> None:
    """Outboard-midplane ``phi(x, y)`` from the bundle's final-field companion."""

    if not has_final_field(source):
        raise FileNotFoundError(
            f"no final-field companion at {final_field_path(source).name}; "
            "the potential map needs a NetCDF run that saved its fields"
        )
    field = load_final_field(source)
    midplane = field.phi_xyz[:, :, field.phi_xyz.shape[2] // 2]
    scale = max(float(np.abs(midplane).max()), 1e-30)
    draw_phi_xy_cut(ax, midplane, scale=scale, extent=field.extent)
    # The saved field is the final state, not the windowed average, so the
    # panel says which time it is showing.
    stamp = "" if field.time is None else rf"   $t\,c_s/a = {field.time:.1f}$"
    ax.set_title(rf"$\phi$ at the outboard midplane{stamp}")


def nonlinear_summary_figure(
    source: str | Path,
    *,
    window: tuple[float, float] | None = None,
    saturated: bool | None = None,
    title: str | None = None,
    out: str | Path | None = None,
) -> Tuple[plt.Figure, dict[str, plt.Axes]]:
    """Compose the whole-run summary page for one nonlinear output bundle.

    ``source`` names any file of the bundle. ``window`` is the averaging window
    the stop policy evaluated. A rejected window remains in the metadata but is
    not shaded or reported as an average on the time trace; spectra retain their
    labelled second-half diagnostic average.
    """

    from gkx.artifacts.transport_figures import _coerce_nonlinear_source

    diag, _ky, _kx, _kind = _coerce_nonlinear_source(str(source))
    heading = title if title is not None else f"GKX nonlinear run: {run_label(source)}"
    plot_window = None if saturated is False else window

    with figure_style():
        # Two rows of three: the traces stacked in one column so they share a
        # time axis, the two ky spectra in the next, and the potential over the
        # metadata in the last. Every cell carries a panel, which is what keeps
        # the page dense enough to read at README width.
        fig = plt.figure(figsize=(14.4, 8.8), layout="constrained")
        grid = fig.add_gridspec(2, 3)
        ax_q = fig.add_subplot(grid[0, 0])
        ax_g = fig.add_subplot(grid[1, 0], sharex=ax_q)
        ax_qky = fig.add_subplot(grid[0, 1])
        ax_phiky = fig.add_subplot(grid[1, 1])
        ax_xy = fig.add_subplot(grid[0, 2])
        ax_meta = fig.add_subplot(grid[1, 2])

        _panel_or_note(
            lambda: heat_flux_time_figure(
                str(source), window=plot_window, title="", axes=(ax_q, ax_g)
            ),
            ax_q,
            ax_g,
        )
        plt.setp(ax_q.get_xticklabels(), visible=False)
        _panel_or_note(
            lambda: flux_spectra_figure(
                str(source), window=plot_window, panels=("ky",), axes=(ax_qky,)
            ),
            ax_qky,
        )
        _panel_or_note(
            lambda: phi2_spectra_figure(
                str(source), window=plot_window, panels=("ky",), axes=(ax_phiky,)
            ),
            ax_phiky,
        )
        _panel_or_note(lambda: _draw_potential_panel(ax_xy, source), ax_xy)
        draw_metadata_panel(
            ax_meta,
            summary_metadata_lines(
                source,
                diag,
                window=window,
                measured_window=window is not None,
                saturated=saturated,
            ),
        )

        axes = {
            "heat_flux": ax_q,
            "particle_flux": ax_g,
            "metadata": ax_meta,
            "flux_spectrum": ax_qky,
            "phi2_spectrum": ax_phiky,
            "potential": ax_xy,
        }
        # The metadata panel is titled rather than lettered: it is the caption,
        # not one of the results.
        for letter, key in zip(
            "abcde",
            (
                "heat_flux",
                "particle_flux",
                "flux_spectrum",
                "phi2_spectrum",
                "potential",
            ),
        ):
            panel_label(axes[key], f"({letter})")
        fig.suptitle(heading)
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, axes


def flux_tube_figure(
    source: str | Path,
    *,
    title: str | None = None,
    out: str | Path | None = None,
) -> Tuple[plt.Figure, Any]:
    """3-D rendering of the final potential on the flux tube the run integrated.

    This is the figure that shows what the geometry actually was, which no
    other output of a run does: everything else is a reduction over the
    field-aligned coordinate.
    """

    field = load_final_field(source)
    scale = max(float(np.abs(field.phi_xyz).max()), 1e-30)
    label = title if title is not None else r"Flux tube along $\mathbf{B}$"
    if field.time is not None:
        label = f"{label}    $t\\,c_s/a = {field.time:.1f}$"

    with figure_style():
        fig = plt.figure(figsize=(7.2, 6.0))
        ax3d = fig.add_subplot(1, 1, 1, projection="3d")
        draw_flux_tube_3d(ax3d, field.phi_xyz, field.geometry, scale=scale)
        ax3d.set_title(label, y=0.97)
        mappable = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(vmin=-scale, vmax=scale)
        )
        mappable.set_array(np.array([]))
        bar = fig.colorbar(mappable, ax=ax3d, fraction=0.04, pad=0.02, shrink=0.8)
        label_amplitude_colorbar(bar, scale, PHI_LABEL)
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, ax3d


def phi_xy_figure(
    source: str | Path,
    *,
    title: str | None = None,
    out: str | Path | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Full-size outboard-midplane cut of the final potential from a bundle."""

    field = load_final_field(source)

    return phi_xy_snapshot_figure(
        field.phi_xyz,
        extent=field.extent,
        time=field.time,
        title=r"Outboard-midplane cut of $\phi$" if title is None else title,
        out=out,
    )


__all__ = [
    "FINAL_FIELD_SUFFIX",
    "FinalField",
    "draw_metadata_panel",
    "final_field_path",
    "flux_tube_figure",
    "has_final_field",
    "load_final_field",
    "nonlinear_summary_figure",
    "phi_xy_figure",
    "run_label",
    "summary_metadata_lines",
    "PHI_LABEL",
    "draw_flux_tube_3d",
    "draw_phi_xy_cut",
    "flux_tube_3d_figure",
    "label_amplitude_colorbar",
    "phi_xy_snapshot_figure",
    "potential_real_space",
]
