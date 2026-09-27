"""Dispatch to readers for other gyrokinetic codes' output bundles.

``gkx --plot`` accepts a bundle written by another code so that a cross-code
comparison is one command rather than a bespoke script. The readers themselves
live in their own modules, one per code; this indirection keeps their names out
of the plotting path, so the core figure code never learns which codes exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import numpy as np


_GX_TITLE_MARKER = "gx simulation data"
_GKX_CODE_INFO_VALUE = "gkx"
_GX_BUILD_ATTRS = ("Hash", "BuildDate", "BuildUser", "BuildHost")

_TIME_LABEL = r"$t \, c_s/a$"
_KY_LABEL = r"$k_y \rho_i$"


def _root_title(root: Any) -> str:
    try:
        return str(root.getncattr("Title"))
    except Exception:
        return ""


def _code_info_value(root: Any) -> str:
    variable = root.variables.get("code_info")
    if variable is None:
        return ""
    try:
        return str(variable.getncattr("value"))
    except Exception:
        return ""


def _code_info_is_gx_build(root: Any) -> bool:
    variable = root.variables.get("code_info")
    if variable is None:
        return False
    attrs = set(variable.ncattrs())
    return any(name in attrs for name in _GX_BUILD_ATTRS)


def is_gx_output(path: str | Path) -> bool:
    """Return whether ``path`` is a GX output bundle rather than a GKX one.

    A file GKX wrote always says so on ``code_info``; anything else that
    carries GX's title or GX's build provenance is treated as GX. Anything
    unreadable is not claimed either way.
    """

    try:
        import netCDF4

        with netCDF4.Dataset(Path(path)) as root:
            if _code_info_value(root).strip().lower() == _GKX_CODE_INFO_VALUE:
                return False
            if _GX_TITLE_MARKER in _root_title(root).strip().lower():
                return True
            return _code_info_is_gx_build(root)
    except Exception:
        return False


def _optional(group: Any, name: str) -> np.ndarray | None:
    if group is None or name not in group.variables:
        return None
    return np.asarray(group.variables[name][:], dtype=float)


def read_gx_output(path: str | Path) -> dict[str, np.ndarray]:
    """Return every plottable series a GX bundle carries, skipping absentees.

    Keys mirror the NetCDF variable names, except that the per-species
    ``*_st`` fluxes also gain summed ``heat_flux``/``particle_flux`` totals.
    A key is present only when the file carried it.
    """

    import netCDF4

    found: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(Path(path)) as root:
        grids = root.groups.get("Grids")
        diag = root.groups.get("Diagnostics")
        for key, group, name in (
            ("t", grids, "time"),
            ("ky", grids, "ky"),
            ("kx", grids, "kx"),
            ("Phi2_t", diag, "Phi2_t"),
            ("Phi2_zonal_t", diag, "Phi2_zonal_t"),
            ("Phi2_kyt", diag, "Phi2_kyt"),
            ("HeatFlux_st", diag, "HeatFlux_st"),
            ("HeatFlux_kyst", diag, "HeatFlux_kyst"),
            ("ParticleFlux_st", diag, "ParticleFlux_st"),
        ):
            values = _optional(group, name)
            if values is not None:
                found[key] = values
    for total, per_species in (
        ("heat_flux", "HeatFlux_st"),
        ("particle_flux", "ParticleFlux_st"),
    ):
        if per_species in found:
            found[total] = np.sum(found[per_species], axis=1)
    return found


def _second_half_mask(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    return t >= float(t[0] + 0.5 * (t[-1] - t[0]))


def _time_panels(series: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray, bool]]:
    """Return ``(ylabel, values, log_scale)`` for each available time trace."""

    panels: list[tuple[str, np.ndarray, bool]] = []
    if "Phi2_t" in series:
        panels.append((r"$\langle |\Phi|^2 \rangle$", series["Phi2_t"], True))
    if "heat_flux" in series:
        panels.append((r"$Q/Q_{\mathrm{gB}}$", series["heat_flux"], False))
    if "particle_flux" in series:
        panels.append(
            (r"$\Gamma/\Gamma_{\mathrm{gB}}$", series["particle_flux"], False)
        )
    return panels


def gx_summary_figure(
    path: str | Path,
    *,
    title: str | None = None,
    out: str | Path | None = None,
) -> tuple[Any, Any]:
    """Plot whatever a GX bundle carries: time traces plus spectra when present.

    Panels are chosen by what the file actually holds, so a GX linear run
    yields the ``Phi^2(t)`` trace alone while a nonlinear run adds ``Q(t)``,
    ``Gamma(t)``, and the time-averaged ``ky`` spectrum over the second half
    of the run. The title names GX explicitly.
    """

    import matplotlib.pyplot as plt

    from gkx.artifacts.figure_style import GKX_COLORS, figure_style, save_figure

    series = read_gx_output(path)
    t = series.get("t")
    if t is None or t.size == 0:
        raise ValueError(f"{Path(path).name} carries no Grids/time axis to plot")

    panels = _time_panels(series)
    spectrum = _ky_spectrum(series)
    if not panels and spectrum is None:
        raise ValueError(
            f"{Path(path).name} carries no Phi2, flux, or spectral diagnostics to plot"
        )

    colors = (GKX_COLORS["blue"], GKX_COLORS["vermillion"], GKX_COLORS["green"])
    count = len(panels) + (1 if spectrum is not None else 0)
    with figure_style():
        fig, axes = plt.subplots(count, 1, figsize=(6.8, 2.2 * count + 1.0))
        axes = np.atleast_1d(axes)
        for index, (label, values, log_scale) in enumerate(panels):
            ax = axes[index]
            ax.plot(t, values, color=colors[index % len(colors)], linewidth=2.0)
            ax.set_ylabel(label)
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel(_TIME_LABEL)
        if spectrum is not None:
            from gkx.artifacts.figure_style import _label_log_ky_axis

            axis, values, label = spectrum
            ax = axes[-1]
            ax.plot(axis, values, marker="o", color=GKX_COLORS["blue"], linewidth=1.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            # A ky range narrower than a decade otherwise draws one tick.
            _label_log_ky_axis(ax)
            ax.set_xlabel(_KY_LABEL)
            ax.set_ylabel(label)
        axes[0].set_title(title or f"GX data: {Path(path).name}")
        fig.tight_layout()
        if out is not None:
            save_figure(fig, out, close=False)
    return fig, axes


def _ky_spectrum(
    series: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Second-half-averaged ``ky`` spectrum, preferring heat flux over ``Phi^2``.

    The zonal ``ky = 0`` channel is dropped: it carries no flux and cannot be
    drawn on the logarithmic axis the rest of the spectrum needs.
    """

    ky = series.get("ky")
    t = series.get("t")
    if ky is None or t is None:
        return None
    mask = _second_half_mask(t)
    for key, label, per_species in (
        ("HeatFlux_kyst", r"$\langle Q(k_y) \rangle$", True),
        ("Phi2_kyt", r"$\langle |\Phi(k_y)|^2 \rangle$", False),
    ):
        values = series.get(key)
        if values is None:
            continue
        averaged = np.mean(values[mask], axis=0)
        if per_species:
            averaged = np.sum(averaged, axis=0)
        averaged = np.atleast_1d(averaged)
        if averaged.size != ky.size:
            continue
        keep = ky > 0.0
        if not np.any(keep):
            continue
        return ky[keep], np.abs(averaged[keep]), label
    return None


def plot_gx_output(path: str | Path, *, out: str | Path | None = None) -> Path:
    """Render a GX bundle to ``out`` (default ``<stem>.gx_plot.png``)."""

    import matplotlib.pyplot as plt

    source = Path(path)
    stem = source.name
    for suffix in (".out.nc", ".nc"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    out_path = Path(out) if out is not None else source.with_name(f"{stem}.gx_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _axes = gx_summary_figure(source, out=out_path)
    plt.close(fig)
    return out_path


def foreign_output_plotter(path: str | Path) -> Callable[..., Any] | None:
    """Return a plotter for ``path`` when another code (GX) wrote it, else ``None``."""

    return plot_gx_output if is_gx_output(Path(path)) else None
