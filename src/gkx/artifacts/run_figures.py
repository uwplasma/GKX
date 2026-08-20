"""The figure set a completed run writes beside its own output.

A run that finishes and leaves only a NetCDF bundle behind is a run whose
result nobody has looked at. This module is what the executable calls once the
artifacts are on disk, so the default outcome of ``gkx run ...`` is a directory
containing both the data and the pictures of it. Nothing here decides physics;
it decides which of the figure functions in
:mod:`gkx.artifacts.transport_figures`, :mod:`gkx.artifacts.run_summary` and
:mod:`gkx.artifacts.plotting` a given run kind can support, and what to name
the files. A nonlinear NetCDF run gets the whole set: the flux traces, both
spectra, the real-space potential and the flux tube it was integrated on, and
the one-page summary that a reader opens first.

Two rules shape the code. First, plotting runs *after* the science is already
saved, so a failure here is never allowed to propagate: a headless machine, a
broken matplotlib backend, or a figure function meeting an edge case it does
not like must cost a picture, not a simulation. Every render is therefore
wrapped and reported on stderr. Second, the CSV diagnostics sidecar carries
time traces only -- the k-resolved spectra exist solely in the NetCDF bundle --
so the spectra figures are skipped rather than attempted when the run wrote
the sidecar form, which is a plain fact about the output format and not a
failure worth warning about.
"""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping, Sequence as ABCSequence
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

TIME_TRACE_SUFFIX = "flux_time"
FLUX_SPECTRA_SUFFIX = "flux_spectra"
PHI2_SPECTRA_SUFFIX = "phi2_spectra"
SNAPSHOT_SUFFIX = "snapshot_xy"
FLUX_TUBE_SUFFIX = "flux_tube_3d"
SUMMARY_SUFFIX = "summary"
PANEL_SUFFIX = "plot"

# Ordered per run kind: the first artifact key that is present names the file
# the figures are rendered from. Nonlinear prefers the NetCDF bundle because
# it is the only form carrying spectra.
_SOURCE_KEYS: dict[str, tuple[str, ...]] = {
    "nonlinear": ("out", "diagnostics", "summary"),
    "linear": ("summary",),
    "linear_scan": ("summary",),
}

_WINDOW_PAIR_KEYS = (
    ("average_window_tmin", "average_window_tmax"),
    ("saturation_window_tmin", "saturation_window_tmax"),
    ("fit_window_tmin", "fit_window_tmax"),
)
_WINDOW_NESTED_KEYS = (
    "average_window",
    "averaging_window",
    "saturation_window",
    # What the stop policy actually writes: gkx.diagnostics.saturation names the
    # bounds window_tmin/window_tmax inside a "saturation" table. Without this
    # entry no run in the repository ever shaded the window it measured.
    "saturation",
)


def _warn(message: str) -> None:
    """Report a plotting problem without touching the run's exit status."""

    print(f"warning: {message}", file=sys.stderr, flush=True)


def _is_netcdf_source(path: Path) -> bool:
    return path.suffix.lower() == ".nc" or path.name.lower().endswith(".out.nc")


def _pair_from(value: Any) -> tuple[float, float] | None:
    """Coerce a mapping or two-element sequence into ``(tmin, tmax)``."""

    if isinstance(value, ABCMapping):
        lo = value.get("tmin", value.get("window_tmin"))
        hi = value.get("tmax", value.get("window_tmax"))
    elif isinstance(value, ABCSequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            return None
        lo, hi = value[0], value[1]
    else:
        return None
    if lo is None or hi is None:
        return None
    try:
        window = (float(lo), float(hi))
    except (TypeError, ValueError):
        return None
    return window if window[1] > window[0] else None


def measured_average_window(summary: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Return the averaging window a run measured, when its summary records one.

    Shading a window the run did not measure would invent a claim about where
    the trace saturated, so the absence of any of these keys means no shading
    rather than a guessed default.
    """

    if not summary:
        return None
    for key in _WINDOW_NESTED_KEYS:
        window = _pair_from(summary.get(key))
        if window is not None:
            return window
    for lo_key, hi_key in _WINDOW_PAIR_KEYS:
        window = _pair_from((summary.get(lo_key), summary.get(hi_key)))
        if window is not None:
            return window
    return None


def measured_window_is_saturated(summary: Mapping[str, Any] | None) -> bool | None:
    """Return whether the run that produced ``summary`` actually saturated.

    The stop policy records the window it evaluated whether or not the trace
    converged in it, so the window alone cannot say. A run that reached its
    cap still carries a window, and calling that a measured saturation would
    tell a reader the flux converged when it was still climbing.
    """

    if not summary:
        return None
    block = summary.get("saturation")
    if isinstance(block, Mapping) and "saturated" in block:
        return bool(block["saturated"])
    return None


def _read_summary(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _render(out_path: Path, render: Callable[[Path], Any]) -> Path | None:
    """Draw one figure, containing every failure mode it can have.

    The matplotlib import is inside the guard on purpose: an unusable backend
    fails at import time, and that is exactly the case this must survive.
    """

    try:
        import matplotlib.pyplot as plt

        fig, _axes = render(out_path)
        plt.close(fig)
    except Exception as exc:
        _warn(f"could not write {out_path.name}: {exc}")
        return None
    return out_path


def write_nonlinear_run_figures(
    source: str | Path,
    *,
    base: str | Path | None = None,
    window: tuple[float, float] | None = None,
    saturated: bool | None = None,
    label: str | None = None,
) -> list[Path]:
    """Write ``Q(t)``/``Gamma(t)`` plus the spectra a NetCDF bundle supports."""

    from gkx.artifacts.plotting import _artifact_base
    from gkx.artifacts.run_summary import (
        flux_tube_figure,
        has_final_field,
        nonlinear_summary_figure,
        phi_xy_figure,
        run_label,
    )
    from gkx.artifacts.transport_figures import (
        flux_spectra_figure,
        heat_flux_time_figure,
        phi2_spectra_figure,
    )

    source_path = Path(source)
    stem = Path(base) if base is not None else _artifact_base(source_path)
    name = label if label is not None else run_label(stem)
    spectra_available = _is_netcdf_source(source_path)

    def _draw(figure: Callable[..., Any], title: str) -> Callable[[Path], Any]:
        return lambda out: figure(
            str(source_path), window=window, title=title, out=out
        )

    plan: list[tuple[str, Callable[[Path], Any]]] = [
        (
            TIME_TRACE_SUFFIX,
            _draw(heat_flux_time_figure, f"GKX nonlinear fluxes: {name}"),
        )
    ]
    if spectra_available:
        plan += [
            (
                FLUX_SPECTRA_SUFFIX,
                _draw(flux_spectra_figure, f"GKX heat-flux spectra: {name}"),
            ),
            (
                PHI2_SPECTRA_SUFFIX,
                _draw(phi2_spectra_figure, f"GKX potential spectra: {name}"),
            ),
        ]
    # The final fields live in the *.big.nc companion, which only a NetCDF run
    # that saved its state writes. Its absence is a fact about the output form,
    # like the missing spectra above, so the panels are skipped rather than
    # attempted and warned about.
    if has_final_field(source_path):
        plan += [
            (
                SNAPSHOT_SUFFIX,
                lambda out: phi_xy_figure(str(source_path), out=out),
            ),
            (
                FLUX_TUBE_SUFFIX,
                lambda out: flux_tube_figure(
                    str(source_path),
                    title=rf"GKX flux tube: {name}",
                    out=out,
                ),
            ),
        ]
    plan.append(
        (
            SUMMARY_SUFFIX,
            lambda out: nonlinear_summary_figure(
                str(source_path), window=window, saturated=saturated, out=out
            ),
        )
    )

    written: list[Path] = []
    for suffix, render in plan:
        rendered = _render(Path(f"{stem}.{suffix}.png"), render)
        if rendered is not None:
            written.append(rendered)
    return written


def write_panel_run_figure(source: str | Path) -> list[Path]:
    """Write the single saved-output panel used by linear points and scans."""

    from gkx.artifacts.plotting import plot_saved_output

    try:
        return [Path(plot_saved_output(source))]
    except Exception as exc:
        _warn(f"could not plot {Path(source).name}: {exc}")
        return []


def is_gkx_nonlinear_bundle(source: str | Path) -> bool:
    """Return whether ``source`` is a GKX nonlinear NetCDF history bundle."""

    path = Path(source)
    if not _is_netcdf_source(path) or not path.is_file():
        return False
    try:
        from gkx.artifacts.foreign_output import foreign_output_plotter

        if foreign_output_plotter(path) is not None:
            return False
        import netCDF4

        with netCDF4.Dataset(path) as root:
            group = root.groups.get("Diagnostics")
            return group is not None and "HeatFlux_st" in group.variables
    except Exception:
        return False


def replot_nonlinear_bundle(source: str | Path) -> list[Path]:
    """Re-render the standard figure set for an already-saved nonlinear bundle.

    ``gkx --plot`` renders one panel from whatever it is given, including other
    codes' output. When the file is GKX's own nonlinear bundle there is a whole
    set to reproduce, and the window to reproduce it over is in the summary
    sidecar the run wrote next to it -- so a re-plot shows the same measured
    window the run reported rather than a fresh guess.
    """

    try:
        from gkx.artifacts.plotting import _artifact_base

        path = Path(source)
        if not is_gkx_nonlinear_bundle(path):
            return []
        base = _artifact_base(path)
        summary = _read_summary(Path(f"{base}.summary.json"))
        window = measured_average_window(summary)
        saturated = measured_window_is_saturated(summary)
        return write_nonlinear_run_figures(
            path, base=base, window=window, saturated=saturated
        )
    except Exception as exc:  # pragma: no cover - guards a broken plotting stack
        _warn(f"could not replot {Path(source).name}: {exc}")
        return []


def auto_plot_saved_run(
    kind: str,
    paths: Mapping[str, str],
    *,
    window: tuple[float, float] | None = None,
) -> list[str]:
    """Render the figure set for one completed run of ``kind``.

    ``paths`` is the artifact mapping the run's writer returned. Returns the
    figure paths that were actually written, which is empty -- never an
    exception -- when the run produced nothing plottable.
    """

    source = None
    for key in _SOURCE_KEYS.get(kind, ()):
        candidate = paths.get(key)
        if candidate:
            source = candidate
            break
    if source is None:
        return []
    if kind != "nonlinear":
        return [str(path) for path in write_panel_run_figure(source)]
    resolved = window
    if resolved is None:
        resolved = measured_average_window(_read_summary(paths.get("summary")))
    return [
        str(path) for path in write_nonlinear_run_figures(source, window=resolved)
    ]


__all__ = [
    "FLUX_SPECTRA_SUFFIX",
    "FLUX_TUBE_SUFFIX",
    "PANEL_SUFFIX",
    "PHI2_SPECTRA_SUFFIX",
    "SNAPSHOT_SUFFIX",
    "SUMMARY_SUFFIX",
    "TIME_TRACE_SUFFIX",
    "auto_plot_saved_run",
    "is_gkx_nonlinear_bundle",
    "measured_average_window",
    "replot_nonlinear_bundle",
    "write_nonlinear_run_figures",
    "write_panel_run_figure",
]
