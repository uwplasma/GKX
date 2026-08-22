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
:mod:`gkx.artifacts.snapshots`), through their ``axes=``/``panels=``
arguments, so the summary cannot drift away from the figures it summarizes.

The one thing this module does own is reading the *final field* sidecar. The
``*.out.nc`` history carries spectra but no potential; the potential lives in
the ``*.big.nc`` companion, stored through ``numpy.fft.ifft2`` and therefore
scaled by ``1/(Ny Nx)`` relative to the convention the solver and
:func:`gkx.artifacts.snapshots.potential_real_space` use. Undoing that here is
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

from gkx.artifacts.figure_style import figure_style, panel_label, save_figure
from gkx.artifacts.plotting import _artifact_base
from gkx.artifacts.snapshots import (
    PHI_LABEL,
    draw_flux_tube_3d,
    draw_phi_xy_cut,
    label_amplitude_colorbar,
)
from gkx.artifacts.transport_figures import (
    flux_spectra_figure,
    heat_flux_time_figure,
    phi2_spectra_figure,
)

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


def _bundle_geometry(group: Any) -> Any:
    """Duck-typed geometry the 3-D flux tube needs: ``q``, ``epsilon``, ``R0``, ``nfp``."""

    major = _scalar(group, "rmaj", 3.0)
    minor = _scalar(group, "aminor", 0.18 * major)
    return SimpleNamespace(
        q=_scalar(group, "q", 1.4),
        epsilon=(minor / major) if major else 0.18,
        R0=major,
        nfp=max(int(_scalar(group, "nfp", 1.0)), 1),
    )


def _read_deck(source: str | Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Load the resolved deck written beside the output, when there is one."""

    path = Path(f"{_artifact_base(Path(source))}.toml")
    if not path.is_file():
        return None, None
    from gkx.utils import tomlcompat as tomllib

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
    from gkx.artifacts.snapshots import phi_xy_snapshot_figure

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
]
