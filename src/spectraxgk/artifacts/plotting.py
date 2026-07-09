"""Public plotting facade for publication-ready figures.

The plotting implementation is split by figure family while this module keeps
one stable import path for examples, tools, and external users.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import matplotlib.pyplot as plt


def set_plot_style() -> None:
    """Apply the shared publication style used by generated figures."""

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 120,
        }
    )


_EXPORTS = {
    "LinearValidationPanel": ("spectraxgk.artifacts.benchmark_plots", "LinearValidationPanel"),
    "MultiReferenceValidationPanel": ("spectraxgk.artifacts.benchmark_plots", "MultiReferenceValidationPanel"),
    "ReferenceSeries": ("spectraxgk.artifacts.benchmark_plots", "ReferenceSeries"),
    "cyclone_comparison_figure": ("spectraxgk.artifacts.benchmark_plots", "cyclone_comparison_figure"),
    "cyclone_reference_figure": ("spectraxgk.artifacts.benchmark_plots", "cyclone_reference_figure"),
    "eigenfunction_overlap_summary_figure": (
        "spectraxgk.artifacts.diagnostic_plots",
        "eigenfunction_overlap_summary_figure",
    ),
    "eigenfunction_reference_overlay_figure": (
        "spectraxgk.artifacts.diagnostic_plots",
        "eigenfunction_reference_overlay_figure",
    ),
    "etg_trend_figure": ("spectraxgk.artifacts.benchmark_plots", "etg_trend_figure"),
    "growth_fit_figure": ("spectraxgk.artifacts.diagnostic_plots", "growth_fit_figure"),
    "growth_rate_heatmap": ("spectraxgk.artifacts.benchmark_plots", "growth_rate_heatmap"),
    "linear_runtime_panel_figure": ("spectraxgk.artifacts.runtime_plots", "linear_runtime_panel_figure"),
    "linear_validation_figure": ("spectraxgk.artifacts.benchmark_plots", "linear_validation_figure"),
    "linear_validation_multi_reference_figure": (
        "spectraxgk.artifacts.benchmark_plots",
        "linear_validation_multi_reference_figure",
    ),
    "nonlinear_runtime_panel_figure": ("spectraxgk.artifacts.runtime_plots", "nonlinear_runtime_panel_figure"),
    "plot_saved_output": ("spectraxgk.artifacts.runtime_plots", "plot_saved_output"),
    "scan_comparison_figure": ("spectraxgk.artifacts.benchmark_plots", "scan_comparison_figure"),
    "scan_multi_reference_figure": ("spectraxgk.artifacts.benchmark_plots", "scan_multi_reference_figure"),
    "zonal_flow_response_figure": ("spectraxgk.artifacts.zonal_plots", "zonal_flow_response_figure"),
}


def __getattr__(name: str) -> Any:
    """Load plotting family helpers only when the public facade needs them."""

    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = ["set_plot_style", *_EXPORTS]
