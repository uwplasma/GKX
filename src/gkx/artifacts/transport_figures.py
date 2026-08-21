"""Figures for saved or in-memory nonlinear transport diagnostics.

CSV sidecars carry time traces; resolved spectra require a NetCDF bundle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

from gkx.artifacts.figure_style import (
    GKX_COLORS,
    SERIES,
    annotate_reference,
    figure_style,
    panel_label,
    save_figure,
)
from gkx.artifacts.plotting import _artifact_base

_TIME_LABEL = r"$t \, c_s/a$"
_HEAT_FLUX_LABEL = r"$Q/Q_{\mathrm{gB}}$"
_PARTICLE_FLUX_LABEL = r"$\Gamma/\Gamma_{\mathrm{gB}}$"
_KY_LABEL = r"$k_y \rho_i$"
_KX_LABEL = r"$k_x \rho_i$"

_KY_TAIL_FRACTION = 0.10
_KY_TAIL_RATIO_LIMIT = 0.10

_CSV_HAS_NO_SPECTRA = (
    "{name} needs k-resolved spectra, which only the NetCDF output bundle "
    "(*.out.nc) carries; a *.diagnostics.csv sidecar stores time traces only. "
    "Re-run with `gkx run-runtime-nonlinear ... --out <case>.out.nc` and point "
    "this figure at the .out.nc file."
)


def _is_netcdf_bundle_path(path: Path) -> bool:
    return path.suffix.lower() == ".nc" or path.name.lower().endswith(".out.nc")


def _structured_column(table: Any, name: str) -> np.ndarray | None:
    names = set(table.dtype.names or ())
    if name not in names:
        return None
    return np.atleast_1d(np.asarray(table[name], dtype=float))


def _structured_species_matrix(table: Any, stem: str) -> np.ndarray | None:
    """Collect ``{stem}_s0, {stem}_s1, ...`` columns into ``(time, species)``."""

    names = tuple(table.dtype.names or ())
    indexed: list[tuple[int, str]] = []
    for name in names:
        prefix = f"{stem}_s"
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            indexed.append((int(name[len(prefix) :]), name))
    if not indexed:
        return None
    cols = [
        np.atleast_1d(np.asarray(table[name], dtype=float))
        for _idx, name in sorted(indexed)
    ]
    return np.stack(cols, axis=1)


def _diagnostics_from_csv_sidecar(base: Path):
    """Rebuild a ``SimulationDiagnostics`` from the CSV diagnostics sidecar."""

    from gkx.diagnostics import SimulationDiagnostics

    csv_path = Path(f"{base}.diagnostics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"no diagnostics sidecar found at {csv_path}")
    table = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float)
    t = _structured_column(table, "t")
    if t is None:
        raise ValueError(f"{csv_path} has no 't' column")
    zeros = np.zeros_like(t)

    def col(name: str) -> np.ndarray:
        found = _structured_column(table, name)
        return zeros if found is None else found

    dt_t = col("dt")
    return SimulationDiagnostics(
        t=t,
        dt_t=dt_t,
        dt_mean=np.asarray(np.mean(dt_t[dt_t > 0.0]) if np.any(dt_t > 0.0) else 0.0),
        gamma_t=col("gamma"),
        omega_t=col("omega"),
        Wg_t=col("Wg"),
        Wphi_t=col("Wphi"),
        Wapar_t=col("Wapar"),
        heat_flux_t=col("heat_flux"),
        particle_flux_t=col("particle_flux"),
        energy_t=col("energy"),
        heat_flux_species_t=_structured_species_matrix(table, "heat_flux"),
        particle_flux_species_t=_structured_species_matrix(table, "particle_flux"),
        turbulent_heating_t=_structured_column(table, "turbulent_heating"),
        turbulent_heating_species_t=_structured_species_matrix(
            table, "turbulent_heating"
        ),
        resolved=None,
    )


def _netcdf_wavenumbers(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read the dealiased ``ky``/``kx`` axes from a NetCDF output bundle."""

    import netCDF4

    ky = kx = None
    with netCDF4.Dataset(path) as root:
        grids = root.groups.get("Grids")
        if grids is not None:
            if "ky" in grids.variables:
                ky = np.asarray(grids.variables["ky"][:], dtype=float)
            if "kx" in grids.variables:
                kx = np.asarray(grids.variables["kx"][:], dtype=float)
    return ky, kx


def _coerce_nonlinear_source(
    source: Any,
) -> tuple[Any, np.ndarray | None, np.ndarray | None, str]:
    """Return ``(diagnostics, ky, kx, source_kind)`` for any accepted source."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if _is_netcdf_bundle_path(path):
            from gkx.artifacts.io import load_nonlinear_netcdf_diagnostics

            ky, kx = _netcdf_wavenumbers(path)
            return load_nonlinear_netcdf_diagnostics(path), ky, kx, "netcdf"
        base = _artifact_base(path)
        return _diagnostics_from_csv_sidecar(base), None, None, "csv"
    return source, None, None, "memory"


def _time_window(
    t: np.ndarray, window: tuple[float, float] | None
) -> tuple[np.ndarray, float, float]:
    """Return ``(mask, tmin, tmax)``; default window is the second half."""

    t_arr = np.asarray(t, dtype=float)
    if t_arr.size == 0:
        raise ValueError("diagnostics carry no time samples")
    if window is None:
        tmin = float(t_arr[0] + 0.5 * (t_arr[-1] - t_arr[0]))
        tmax = float(t_arr[-1])
    else:
        tmin, tmax = float(window[0]), float(window[1])
    mask = (t_arr >= tmin) & (t_arr <= tmax)
    if not np.any(mask):
        raise ValueError(
            f"averaging window [{tmin}, {tmax}] contains no samples "
            f"(t spans [{t_arr[0]:.4g}, {t_arr[-1]:.4g}])"
        )
    return mask, tmin, tmax


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = (
        float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        if arr.size >= 2
        else float("nan")
    )
    return mean, sem


def _species_series(diag: Any, attr: str) -> np.ndarray | None:
    """Per-species ``(time, species)`` history, only when >= 2 species exist."""

    values = getattr(diag, attr, None)
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr


def _species_label(species_labels: Any, index: int) -> str:
    if species_labels is not None and index < len(species_labels):
        return str(species_labels[index])
    return f"species {index}"


def _host_axes(
    axes: Any,
    *,
    rows: int,
    cols: int,
    figsize: tuple[float, float],
    sharex: bool = False,
) -> tuple[plt.Figure, np.ndarray, bool]:
    """Return figure/axes and whether this call created them."""

    if axes is None:
        fig, created = plt.subplots(
            rows, cols, figsize=figsize, sharex=sharex, squeeze=False
        )
        return fig, created.ravel(), True
    given = np.asarray(list(axes) if not isinstance(axes, plt.Axes) else [axes])
    expected = rows * cols
    if given.size != expected:
        raise ValueError(f"axes= needs {expected} axes, got {given.size}")
    return given.flat[0].figure, given, False


def heat_flux_time_figure(
    source: Any,
    *,
    window: tuple[float, float] | None = None,
    species_labels: list[str] | None = None,
    title: str = "Turbulent fluxes",
    out: str | Path | None = None,
    axes: Any = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ``Q(t)`` and ``Gamma(t)`` with an optional measured window.

    Sources may be diagnostics, NetCDF, or CSV. Figure SEM treats samples as
    independent; published estimates need correlation-corrected uncertainty.
    ``axes=`` embeds the two panels in another figure.
    """

    diag, _ky, _kx, _kind = _coerce_nonlinear_source(source)
    t = np.asarray(diag.t, dtype=float)
    heat = np.asarray(diag.heat_flux_t, dtype=float)
    particle = np.asarray(diag.particle_flux_t, dtype=float)
    heat_species = _species_series(diag, "heat_flux_species_t")
    particle_species = _species_series(diag, "particle_flux_species_t")

    with figure_style():
        fig, panel_axes, owns_figure = _host_axes(
            axes, rows=2, cols=1, figsize=(6.8, 5.8), sharex=True
        )
        ax_q, ax_g = panel_axes

        for ax, total, species, total_label, y_label, color in (
            (ax_q, heat, heat_species, "total", _HEAT_FLUX_LABEL, GKX_COLORS["blue"]),
            (
                ax_g,
                particle,
                particle_species,
                "total",
                _PARTICLE_FLUX_LABEL,
                GKX_COLORS["vermillion"],
            ),
        ):
            if species is not None:
                for index in range(species.shape[1]):
                    ax.plot(
                        t,
                        species[:, index],
                        linewidth=1.3,
                        alpha=0.85,
                        color=SERIES[(index + 2) % len(SERIES)],
                        label=_species_label(species_labels, index),
                    )
            ax.plot(t, total, color=color, linewidth=2.1, label=total_label)
            ax.set_ylabel(y_label)
            if species is not None:
                ax.legend(loc="best", ncols=2)

        if window is not None:
            mask, tmin, tmax = _time_window(t, window)
            q_mean, q_sem = _mean_sem(heat[mask])
            g_mean, g_sem = _mean_sem(particle[mask])
            for ax in panel_axes:
                ax.axvspan(tmin, tmax, color=GKX_COLORS["grey"], alpha=0.16, zorder=0)
            annotate_reference(
                ax_q,
                rf"$\langle Q \rangle = {q_mean:.3g} \pm {q_sem:.2g}$"
                rf"  over $t \in [{tmin:.4g}, {tmax:.4g}]$",
                loc="lower right",
            )
            annotate_reference(
                ax_g,
                rf"$\langle \Gamma \rangle = {g_mean:.3g} \pm {g_sem:.2g}$"
                rf"  over $t \in [{tmin:.4g}, {tmax:.4g}]$",
                loc="lower right",
            )

        if title:
            ax_q.set_title(title)
        ax_g.set_xlabel(_TIME_LABEL)
        if owns_figure:
            fig.tight_layout()
            if out is not None:
                save_figure(fig, out, close=False)
    return fig, panel_axes


def _resolved_spectra_or_error(diag: Any, *, name: str, kind: str) -> Any:
    if kind == "csv":
        raise ValueError(_CSV_HAS_NO_SPECTRA.format(name=name))
    resolved = getattr(diag, "resolved", None)
    if resolved is None:
        raise ValueError(
            f"{name} needs resolved spectra but the diagnostics carry none; "
            "use a NetCDF output bundle (*.out.nc) written with "
            "[output] resolved_diagnostics = true"
        )
    return resolved


def _spectral_axis_or_error(
    axis: np.ndarray | None, *, length: int, axis_name: str, name: str
) -> np.ndarray:
    if axis is None:
        raise ValueError(
            f"{name}: pass {axis_name}= explicitly when the source is not a "
            f"NetCDF output bundle carrying its Grids/{axis_name} axis"
        )
    arr = np.asarray(axis, dtype=float)
    if arr.size != length:
        raise ValueError(
            f"{name}: {axis_name} axis has {arr.size} points but the spectra "
            f"resolve {length}"
        )
    return arr


def _window_average(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(values, dtype=float)[mask], axis=0)


def ky_spectrum_tail_ratio(ky: np.ndarray, spectrum: np.ndarray) -> float | None:
    r"""Return the high-:math:`k_y` amplitude relative to the spectral peak.

    The last 10% of positive ``ky`` modes must decay below the peak for the
    cutoff to be credible. This ratio is only a necessary resolution check;
    passing it does not replace an ``Nx``/``Ny`` convergence scan.
    """

    axis = np.asarray(ky, dtype=float).reshape(-1)
    values = np.abs(np.asarray(spectrum, dtype=float).reshape(-1))
    if axis.size != values.size:
        raise ValueError("ky and spectrum must have the same length")
    keep = (axis > 0.0) & np.isfinite(axis) & np.isfinite(values)
    if np.count_nonzero(keep) < 4:
        return None
    order = np.argsort(axis[keep])
    amplitudes = values[keep][order]
    peak = float(np.max(amplitudes))
    if peak <= 0.0:
        return None
    tail_count = max(1, int(np.ceil(_KY_TAIL_FRACTION * amplitudes.size)))
    return float(np.max(amplitudes[-tail_count:]) / peak)


def _annotate_ky_cutoff(
    ax: plt.Axes, ky: np.ndarray, spectrum: np.ndarray
) -> float | None:
    """Mark a spectrum whose retained high-``ky`` tail has not decayed."""

    ratio = ky_spectrum_tail_ratio(ky, spectrum)
    if ratio is None or ratio < _KY_TAIL_RATIO_LIMIT:
        return ratio
    ax.text(
        0.98,
        0.97,
        rf"$k_y$ cutoff unresolved" "\n" rf"tail / peak = {ratio:.2f}",
        transform=ax.transAxes,
        fontsize=8.5,
        color=GKX_COLORS["vermillion"],
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": GKX_COLORS["vermillion"],
            "linewidth": 0.8,
            "alpha": 0.95,
        },
    )
    return ratio


def spectrum_cutoff_warnings(
    source: Any, *, window: tuple[float, float] | None = None
) -> tuple[str, ...]:
    """Explain unresolved ``ky`` cutoffs in a saved nonlinear result."""

    diag, ky, _kx, kind = _coerce_nonlinear_source(source)
    resolved = _resolved_spectra_or_error(
        diag, name="spectrum cutoff check", kind=kind
    )
    if ky is None:
        return ()
    mask, _tmin, _tmax = _time_window(np.asarray(diag.t, dtype=float), window)
    candidates = (
        ("heat-flux", getattr(resolved, "HeatFlux_kyst", None)),
        ("potential", getattr(resolved, "Phi2_kyt", None)),
    )
    messages: list[str] = []
    for label, history in candidates:
        if history is None:
            continue
        averaged = _window_average(history, mask)
        spectrum = averaged.sum(axis=0) if averaged.ndim == 2 else averaged
        ratio = ky_spectrum_tail_ratio(ky, spectrum)
        if ratio is not None and ratio >= _KY_TAIL_RATIO_LIMIT:
            messages.append(
                f"{label} ky cutoff is unresolved: the highest 10% of retained "
                f"positive-ky modes reach {ratio:.0%} of the spectral peak "
                f"(warning threshold {_KY_TAIL_RATIO_LIMIT:.0%}). Increase Ny at "
                "fixed Ly, then repeat matched Nx/Ny convergence; this warning is "
                "necessary, not sufficient, for resolution."
            )
    return tuple(messages)


def _annotate_with_headroom(
    ax: plt.Axes, text: str, *, loc: str = "upper left", fraction: float = 0.22
) -> None:
    """Clear a strip, then place an annotation without covering data."""

    lo, hi = ax.get_ylim()
    upper = loc.startswith("upper")
    if ax.get_yscale() == "log":
        if lo > 0.0 and hi > lo:
            span = np.log10(hi) - np.log10(lo)
            if upper:
                ax.set_ylim(lo, 10.0 ** (np.log10(hi) + fraction * span))
            else:
                ax.set_ylim(10.0 ** (np.log10(lo) - fraction * span), hi)
    else:
        span = hi - lo
        if span > 0.0:
            if upper:
                ax.set_ylim(lo, hi + fraction * span)
            else:
                ax.set_ylim(lo - fraction * span, hi)
    annotate_reference(ax, text, loc=loc)


def _label_log_ky_axis(ax: plt.Axes) -> None:
    """Label major/minor ticks on a narrow log-``ky`` interval."""

    if ax.get_xscale() != "log":
        return
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0), numticks=12))
    plain = FuncFormatter(lambda value, _pos: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(plain)


def _plot_ky_spectrum(
    ax: plt.Axes,
    ky: np.ndarray,
    per_species: np.ndarray,
    *,
    species_labels: list[str] | None,
    y_label: str,
) -> None:
    """Plot a per-species ``(species, ky)`` spectrum against positive ``ky``."""

    positive = ky > 0.0
    ky_plot = ky[positive]
    total = per_species.sum(axis=0)
    if per_species.shape[0] >= 2:
        for index in range(per_species.shape[0]):
            ax.plot(
                ky_plot,
                per_species[index][positive],
                linewidth=1.3,
                alpha=0.85,
                color=SERIES[(index + 2) % len(SERIES)],
                label=_species_label(species_labels, index),
            )
        ax.plot(ky_plot, total[positive], color=GKX_COLORS["blue"], label="total")
        ax.legend(loc="best")
    else:
        ax.plot(ky_plot, total[positive], color=GKX_COLORS["blue"])
    if ky_plot.size >= 2:
        ax.set_xscale("log")
        _label_log_ky_axis(ax)
    ax.axhline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_KY_LABEL)
    ax.set_ylabel(y_label)
    _annotate_ky_cutoff(ax, ky, total)


def _plot_kx_spectrum(
    ax: plt.Axes,
    kx: np.ndarray,
    per_species: np.ndarray,
    *,
    species_labels: list[str] | None,
    y_label: str,
) -> None:
    """Plot a per-species ``(species, kx)`` spectrum on the signed ``kx`` axis."""

    order = np.argsort(kx)
    kx_plot = kx[order]
    total = per_species.sum(axis=0)
    if per_species.shape[0] >= 2:
        for index in range(per_species.shape[0]):
            ax.plot(
                kx_plot,
                per_species[index][order],
                linewidth=1.3,
                alpha=0.85,
                color=SERIES[(index + 2) % len(SERIES)],
                label=_species_label(species_labels, index),
            )
        ax.plot(kx_plot, total[order], color=GKX_COLORS["blue"], label="total")
        ax.legend(loc="best")
    else:
        ax.plot(kx_plot, total[order], color=GKX_COLORS["blue"])
    ax.axhline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_KX_LABEL)
    ax.set_ylabel(y_label)


def _selected_panels(
    panels: list[tuple[str, np.ndarray, np.ndarray]], wanted: Any
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Keep the requested panels, in the requested order, or all of them."""

    if wanted is None:
        return panels
    available = {panel[0]: panel for panel in panels}
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(
            f"panels={tuple(wanted)!r} requested {missing!r}, which this source "
            f"does not resolve; available: {tuple(available)!r}"
        )
    return [available[name] for name in wanted]


def flux_spectra_figure(
    source: Any,
    *,
    ky: np.ndarray | None = None,
    kx: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
    species_labels: list[str] | None = None,
    title: str = "Heat-flux spectra",
    out: str | Path | None = None,
    axes: Any = None,
    panels: Any = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot window-averaged ``Q(ky)`` and ``Q(kx)``.

    NetCDF or resolved in-memory diagnostics are accepted; CSV is not.
    ``panels``/``axes`` support embedding selected panels.
    """

    diag, ky_file, kx_file, kind = _coerce_nonlinear_source(source)
    resolved = _resolved_spectra_or_error(diag, name="flux_spectra_figure", kind=kind)
    heat_ky = getattr(resolved, "HeatFlux_kyst", None)
    heat_kx = getattr(resolved, "HeatFlux_kxst", None)
    if heat_ky is None and heat_kx is None:
        raise ValueError(
            "flux_spectra_figure: the source has resolved diagnostics but no "
            "HeatFlux_kyst/HeatFlux_kxst spectra; write the NetCDF output "
            "bundle (*.out.nc) with [output] resolved_diagnostics = true"
        )
    mask, tmin, tmax = _time_window(np.asarray(diag.t, dtype=float), window)

    found: list[tuple[str, np.ndarray, np.ndarray]] = []
    if heat_ky is not None:
        avg = _window_average(heat_ky, mask)  # (species, ky)
        axis = _spectral_axis_or_error(
            ky if ky is not None else ky_file,
            length=avg.shape[-1],
            axis_name="ky",
            name="flux_spectra_figure",
        )
        found.append(("ky", axis, avg))
    if heat_kx is not None:
        avg = _window_average(heat_kx, mask)  # (species, kx)
        axis = _spectral_axis_or_error(
            kx if kx is not None else kx_file,
            length=avg.shape[-1],
            axis_name="kx",
            name="flux_spectra_figure",
        )
        found.append(("kx", axis, avg))
    drawn = _selected_panels(found, panels)

    with figure_style():
        fig, panel_axes, owns_figure = _host_axes(
            axes, rows=1, cols=len(drawn), figsize=(5.6 * len(drawn), 4.4)
        )
        for index, (which, axis, avg) in enumerate(drawn):
            ax = panel_axes[index]
            if which == "ky":
                _plot_ky_spectrum(
                    ax,
                    axis,
                    avg,
                    species_labels=species_labels,
                    y_label=r"$Q(k_y)/Q_{\mathrm{gB}}$",
                )
            else:
                _plot_kx_spectrum(
                    ax,
                    axis,
                    avg,
                    species_labels=species_labels,
                    y_label=r"$Q(k_x)/Q_{\mathrm{gB}}$",
                )
            if owns_figure and len(drawn) > 1:
                panel_label(ax, f"({'ab'[index]})")
        _annotate_with_headroom(
            panel_axes[0],
            rf"time average over $t \in [{tmin:.4g}, {tmax:.4g}]$",
            loc="upper left",
        )
        if owns_figure:
            fig.suptitle(title)
            fig.tight_layout()
            if out is not None:
                save_figure(fig, out, close=False)
    return fig, panel_axes


def _zonal_split(
    resolved: Any, ky: np.ndarray | None, phi2_kxky: np.ndarray | None
) -> np.ndarray | None:
    """Zonal ``Phi^2(t)`` from the dedicated trace or the 2D spectrum."""

    zonal = getattr(resolved, "Phi2_zonal_t", None)
    if zonal is not None:
        return np.asarray(zonal, dtype=float)
    if phi2_kxky is not None and ky is not None and ky.size and ky[0] == 0.0:
        return np.asarray(phi2_kxky, dtype=float)[:, 0, :].sum(axis=-1)
    return None


def _draw_phi2_ky_panel(
    ax: plt.Axes,
    ky_axis: np.ndarray,
    ky_avg: np.ndarray,
    *,
    tmin: float,
    tmax: float,
) -> None:
    """Time-averaged ``Phi^2(ky)``; the zonal ``ky = 0`` row lives elsewhere."""

    positive = ky_axis > 0.0
    ax.plot(ky_axis[positive], ky_avg[positive], color=GKX_COLORS["blue"])
    if np.count_nonzero(positive) >= 2:
        ax.set_xscale("log")
        _label_log_ky_axis(ax)
    if np.all(ky_avg[positive] > 0.0) and np.any(positive):
        ax.set_yscale("log")
    ax.set_xlabel(_KY_LABEL)
    ax.set_ylabel(r"$\Phi^2(k_y)$")
    _annotate_ky_cutoff(ax, ky_axis, ky_avg)
    _annotate_with_headroom(
        ax, rf"time average over $t \in [{tmin:.4g}, {tmax:.4g}]$", loc="upper left"
    )


def _draw_phi2_kx_panel(
    ax: plt.Axes, kx_axis: np.ndarray, kx_avg: np.ndarray, order: np.ndarray
) -> None:
    """Time-averaged ``Phi^2(kx)`` on the signed ``kx`` axis."""

    ax.plot(kx_axis[order], kx_avg[order], color=GKX_COLORS["blue"])
    if np.all(kx_avg > 0.0):
        ax.set_yscale("log")
    ax.axvline(0.0, color=GKX_COLORS["grey"], linewidth=0.8, alpha=0.6)
    ax.set_xlabel(_KX_LABEL)
    ax.set_ylabel(r"$\Phi^2(k_x)$")


def _draw_phi2_map_panel(
    ax: plt.Axes,
    kx_sorted: np.ndarray,
    ky_axis: np.ndarray,
    avg_2d: np.ndarray | None,
) -> None:
    """``Phi^2(kx, ky)`` heatmap, on a log colour scale when the data allow it."""

    if avg_2d is None:
        ax.text(
            0.5,
            0.5,
            "no Phi2_kxkyt in source",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#555555",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return
    from matplotlib.colors import LogNorm

    positive_2d = avg_2d[avg_2d > 0.0]
    norm = None
    if positive_2d.size:
        norm = LogNorm(vmin=float(positive_2d.min()), vmax=float(positive_2d.max()))
        avg_2d = np.where(avg_2d > 0.0, avg_2d, float(positive_2d.min()))
    mesh = ax.pcolormesh(
        kx_sorted, ky_axis, avg_2d, cmap="magma", norm=norm, shading="auto"
    )
    ax.grid(False)
    bar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(r"$\Phi^2(k_x, k_y)$")
    ax.set_xlabel(_KX_LABEL)
    ax.set_ylabel(_KY_LABEL)


def _draw_phi2_zonal_panel(
    ax: plt.Axes,
    t: np.ndarray,
    total_t: np.ndarray,
    zonal_t: np.ndarray | None,
    *,
    tmin: float,
    tmax: float,
) -> None:
    """Zonal versus nonzonal ``Phi^2`` over time, with the window shaded."""

    ax.plot(t, total_t, color=GKX_COLORS["black"], label="total")
    if zonal_t is not None:
        # Dashed, so the nonzonal trace stays readable where it coincides with
        # the total (whenever the zonal share is negligible).
        nonzonal_t = np.clip(total_t - zonal_t, 0.0, None)
        ax.plot(t, zonal_t, color=GKX_COLORS["green"], label=r"zonal ($k_y=0$)")
        ax.plot(
            t,
            nonzonal_t,
            color=GKX_COLORS["orange"],
            linestyle="--",
            label=r"nonzonal ($k_y\neq 0$)",
        )
    finite_total = total_t[np.isfinite(total_t)]
    if finite_total.size and np.all(finite_total > 0.0):
        ax.set_yscale("log")
    ax.axvspan(tmin, tmax, color=GKX_COLORS["grey"], alpha=0.16, zorder=0)
    ax.set_xlabel(_TIME_LABEL)
    ax.set_ylabel(r"$\Phi^2$")
    ax.legend(loc="best")


PHI2_PANEL_NAMES = ("ky", "kx", "map", "zonal")


def phi2_spectra_figure(
    source: Any,
    *,
    ky: np.ndarray | None = None,
    kx: np.ndarray | None = None,
    window: tuple[float, float] | None = None,
    title: str = "Potential spectra",
    out: str | Path | None = None,
    axes: Any = None,
    panels: Any = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot ``Phi^2`` by ``ky``, ``kx``, 2-D map, and zonal share.

    Sources and embedding follow :func:`flux_spectra_figure`.
    """

    diag, ky_file, kx_file, kind = _coerce_nonlinear_source(source)
    resolved = _resolved_spectra_or_error(diag, name="phi2_spectra_figure", kind=kind)
    phi2_kxky = getattr(resolved, "Phi2_kxkyt", None)
    phi2_ky = getattr(resolved, "Phi2_kyt", None)
    phi2_kx = getattr(resolved, "Phi2_kxt", None)
    if phi2_kxky is not None:
        phi2_kxky = np.asarray(phi2_kxky, dtype=float)  # (time, ky, kx)
        if phi2_ky is None:
            phi2_ky = phi2_kxky.sum(axis=2)
        if phi2_kx is None:
            phi2_kx = phi2_kxky.sum(axis=1)
    if phi2_ky is None or phi2_kx is None:
        raise ValueError(
            "phi2_spectra_figure needs Phi2_kxkyt (or Phi2_kyt and Phi2_kxt) "
            "resolved spectra, which only the NetCDF output bundle (*.out.nc) "
            "written with [output] resolved_diagnostics = true carries"
        )
    phi2_ky = np.asarray(phi2_ky, dtype=float)
    phi2_kx = np.asarray(phi2_kx, dtype=float)

    t = np.asarray(diag.t, dtype=float)
    mask, tmin, tmax = _time_window(t, window)
    ky_axis = _spectral_axis_or_error(
        ky if ky is not None else ky_file,
        length=phi2_ky.shape[-1],
        axis_name="ky",
        name="phi2_spectra_figure",
    )
    kx_axis = _spectral_axis_or_error(
        kx if kx is not None else kx_file,
        length=phi2_kx.shape[-1],
        axis_name="kx",
        name="phi2_spectra_figure",
    )

    ky_avg = _window_average(phi2_ky, mask)
    kx_avg = _window_average(phi2_kx, mask)
    total_t = phi2_ky.sum(axis=1)
    zonal_t = _zonal_split(resolved, ky_axis, phi2_kxky)

    order = np.argsort(kx_axis)
    map_2d = None if phi2_kxky is None else _window_average(phi2_kxky, mask)[:, order]
    drawers = {
        "ky": lambda ax: _draw_phi2_ky_panel(ax, ky_axis, ky_avg, tmin=tmin, tmax=tmax),
        "kx": lambda ax: _draw_phi2_kx_panel(ax, kx_axis, kx_avg, order),
        "map": lambda ax: _draw_phi2_map_panel(ax, kx_axis[order], ky_axis, map_2d),
        "zonal": lambda ax: _draw_phi2_zonal_panel(
            ax, t, total_t, zonal_t, tmin=tmin, tmax=tmax
        ),
    }
    names = PHI2_PANEL_NAMES if panels is None else tuple(panels)
    unknown = [name for name in names if name not in drawers]
    if unknown:
        raise ValueError(
            f"phi2_spectra_figure: unknown panels {unknown!r}; "
            f"choose from {PHI2_PANEL_NAMES!r}"
        )
    rows, cols = (2, 2) if len(names) == 4 else (1, len(names))

    with figure_style():
        fig, panel_axes, owns_figure = _host_axes(
            axes, rows=rows, cols=cols, figsize=(5.6 * cols, 4.3 * rows)
        )
        for index, name in enumerate(names):
            drawers[name](panel_axes[index])
            if owns_figure and len(names) > 1:
                panel_label(panel_axes[index], f"({'abcd'[index]})")
        if owns_figure:
            fig.suptitle(title)
            fig.tight_layout()
            if out is not None:
                save_figure(fig, out, close=False)
    # A standalone four-panel figure keeps the 2x2 shape plt.subplots returned.
    if owns_figure and rows > 1:
        return fig, panel_axes.reshape(rows, cols)
    return fig, panel_axes


__all__ = [
    "PHI2_PANEL_NAMES",
    "flux_spectra_figure",
    "heat_flux_time_figure",
    "phi2_spectra_figure",
]
