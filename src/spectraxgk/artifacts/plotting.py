"""Public plotting facade for publication-ready figures.

The plotting implementation is split by figure family while this module keeps
one stable import path for examples, tools, and external users.
"""

from __future__ import annotations

from spectraxgk.artifacts.benchmark_plots import (
    LinearValidationPanel,
    MultiReferenceValidationPanel,
    ReferenceSeries,
    cyclone_comparison_figure,
    cyclone_reference_figure,
    etg_trend_figure,
    growth_rate_heatmap,
    linear_validation_figure,
    linear_validation_multi_reference_figure,
    scan_comparison_figure,
    scan_multi_reference_figure,
)
from spectraxgk.artifacts.diagnostic_plots import (
    eigenfunction_overlap_summary_figure,
    eigenfunction_reference_overlay_figure,
    growth_fit_figure,
)
from spectraxgk.artifacts.plot_style import set_plot_style
from spectraxgk.artifacts.runtime_plots import (
    linear_runtime_panel_figure,
    nonlinear_runtime_panel_figure,
    plot_saved_output,
)
from spectraxgk.artifacts.zonal_plots import zonal_flow_response_figure

__all__ = [
    "LinearValidationPanel",
    "MultiReferenceValidationPanel",
    "ReferenceSeries",
    "cyclone_comparison_figure",
    "cyclone_reference_figure",
    "eigenfunction_overlap_summary_figure",
    "eigenfunction_reference_overlay_figure",
    "etg_trend_figure",
    "growth_fit_figure",
    "growth_rate_heatmap",
    "linear_runtime_panel_figure",
    "linear_validation_figure",
    "linear_validation_multi_reference_figure",
    "nonlinear_runtime_panel_figure",
    "plot_saved_output",
    "scan_comparison_figure",
    "scan_multi_reference_figure",
    "set_plot_style",
    "zonal_flow_response_figure",
]
