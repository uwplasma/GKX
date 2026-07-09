"""Zonal-response plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.artifacts.plotting import set_plot_style


@dataclass(frozen=True)
class _ZonalPlotData:
    t: np.ndarray
    response_norm: np.ndarray
    residual: float
    residual_std: float
    env_t: np.ndarray
    env_y: np.ndarray
    fit_count: int
    fit_tmin: float
    fit_tmax: float
    damping_method: str
    frequency_method: str
    max_peak_t: np.ndarray
    max_peak_y: np.ndarray
    min_peak_t: np.ndarray
    min_peak_y: np.ndarray
    metrics: Any


def _validated_zonal_trace(t: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError("t and response must be one-dimensional arrays of equal length")
    return t_arr, resp


def _array_metric(metrics: Any, name: str) -> np.ndarray:
    value = getattr(metrics, name, np.asarray([], dtype=float))
    return np.asarray(value, dtype=float)


def _zonal_plot_data(t: np.ndarray, response: np.ndarray, metrics: Any) -> _ZonalPlotData:
    response_norm = response / float(metrics.initial_level)
    return _ZonalPlotData(
        t=t,
        response_norm=response_norm,
        residual=float(metrics.residual_level),
        residual_std=float(metrics.residual_std),
        env_t=np.asarray(metrics.peak_times, dtype=float),
        env_y=np.asarray(metrics.peak_envelope, dtype=float),
        fit_count=int(getattr(metrics, "peak_fit_count", len(metrics.peak_times))),
        fit_tmin=float(getattr(metrics, "fit_tmin", t[0])),
        fit_tmax=float(getattr(metrics, "fit_tmax", t[-1])),
        damping_method=str(getattr(metrics, "damping_method", "combined_envelope")),
        frequency_method=str(getattr(metrics, "frequency_method", "peak_spacing")),
        max_peak_t=_array_metric(metrics, "max_peak_times"),
        max_peak_y=_array_metric(metrics, "max_peak_values"),
        min_peak_t=_array_metric(metrics, "min_peak_times"),
        min_peak_y=_array_metric(metrics, "min_peak_values"),
        metrics=metrics,
    )


def _fit_window_mask(values: np.ndarray, data: _ZonalPlotData) -> np.ndarray:
    return (values >= data.fit_tmin) & (values <= data.fit_tmax)


def _plot_branchwise_points(
    ax: plt.Axes,
    *,
    data: _ZonalPlotData,
) -> None:
    if data.damping_method != "branchwise_extrema":
        return
    if data.max_peak_t.size:
        keep = _fit_window_mask(data.max_peak_t, data)
        ax.plot(
            data.max_peak_t[keep],
            data.max_peak_y[keep],
            linestyle="none",
            marker="o",
            color="#2a9d8f",
            markersize=5.2,
            label="maxima fit points",
        )
    if data.min_peak_t.size:
        keep = _fit_window_mask(data.min_peak_t, data)
        ax.plot(
            data.min_peak_t[keep],
            data.min_peak_y[keep],
            linestyle="none",
            marker="o",
            color="#7b2cbf",
            markersize=5.2,
            label="minima fit points",
        )


def _plot_normalized_response_panel(
    ax: plt.Axes,
    *,
    data: _ZonalPlotData,
    y_label: str,
) -> None:
    ax.plot(data.t, data.response_norm, color="#0f4c81", linewidth=2.2, label="response")
    ax.axhline(
        data.residual,
        color="#c44e52",
        linestyle="--",
        linewidth=2.0,
        label="residual",
    )
    ax.axvspan(data.fit_tmin, data.fit_tmax, color="#d9ead3", alpha=0.22, linewidth=0.0)
    _plot_branchwise_points(ax, data=data)
    ax.fill_between(
        data.t,
        data.residual - data.residual_std,
        data.residual + data.residual_std,
        color="#c44e52",
        alpha=0.15,
        linewidth=0.0,
    )
    ax.set_xlabel("t")
    ax.set_ylabel(y_label)
    ax.set_title("Normalized response")
    ax.legend(loc="best", frameon=False)


def _combined_envelope_fit(data: _ZonalPlotData) -> tuple[np.ndarray, np.ndarray, str] | None:
    fit_mask = _fit_window_mask(data.env_t, data)
    fit_env_t = data.env_t[fit_mask]
    fit_env_y = data.env_y[fit_mask]
    finite_damping = np.isfinite(float(data.metrics.gam_damping_rate))
    if data.damping_method != "combined_envelope" or data.fit_count < 2:
        return None
    if not finite_damping or not fit_env_t.size:
        return None

    fit_n = min(data.fit_count, fit_env_t.size)
    fit_t = fit_env_t[:fit_n]
    fit = fit_env_y[0] * np.exp(-float(data.metrics.gam_damping_rate) * (fit_t - fit_t[0]))
    label = "envelope fit" if fit_n == fit_env_t.size else f"envelope fit (first {fit_n} peaks)"
    return fit_t, fit, label


def _plot_envelope_panel(ax: plt.Axes, *, data: _ZonalPlotData) -> None:
    ax.plot(
        data.t,
        np.maximum(np.abs(data.response_norm - data.residual), 1.0e-14),
        color="#4c956c",
        linewidth=2.0,
        alpha=0.5,
    )
    if data.env_t.size:
        ax.plot(
            data.env_t,
            data.env_y,
            color="#c44e52",
            marker="o",
            linewidth=1.8,
            label="envelope peaks",
        )
    fit = _combined_envelope_fit(data)
    if fit is not None:
        fit_t, fit_y, label = fit
        ax.plot(fit_t, fit_y, color="#2a9d8f", linestyle="--", linewidth=2.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("envelope")
    ax.set_title("GAM envelope")
    if data.env_t.size:
        ax.legend(loc="best", frameon=False)


def _metrics_annotation(data: _ZonalPlotData) -> str:
    metrics = data.metrics
    norm_policy = getattr(metrics, "initial_policy", "window_abs_mean")
    return (
        f"residual = {metrics.residual_level:.4f}\n"
        f"std = {metrics.residual_std:.4f}\n"
        f"ω_GAM = {metrics.gam_frequency:.4f}\n"
        f"γ_damp = {metrics.gam_damping_rate:.4f}\n"
        f"fit_peaks = {data.fit_count}\n"
        f"norm = {norm_policy}\n"
        f"damp = {data.damping_method}\n"
        f"freq = {data.frequency_method}\n"
        f"fit_t = [{data.fit_tmin:.1f}, {data.fit_tmax:.1f}]"
    )


def _annotate_envelope_panel(ax: plt.Axes, *, data: _ZonalPlotData) -> None:
    ax.text(
        0.03,
        0.97,
        _metrics_annotation(data),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "#cccccc",
        },
    )


def _style_zonal_axes(axes: np.ndarray) -> None:
    for axis in axes:
        axis.grid(True, alpha=0.25)


def zonal_flow_response_figure(
    t: np.ndarray,
    response: np.ndarray,
    *,
    metrics: Any = None,
    title: str = "Zonal-flow response",
    y_label: str = "normalized response",
) -> tuple[plt.Figure, np.ndarray]:
    """Render a zonal-flow response trace and its envelope summary."""

    from spectraxgk.benchmarks import zonal_flow_response_metrics

    set_plot_style()
    t_arr, resp = _validated_zonal_trace(t, response)
    if metrics is None:
        metrics = zonal_flow_response_metrics(t_arr, resp)
    data = _zonal_plot_data(t_arr, resp, metrics)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    _plot_normalized_response_panel(axes[0], data=data, y_label=y_label)
    _plot_envelope_panel(axes[1], data=data)
    _annotate_envelope_panel(axes[1], data=data)
    _style_zonal_axes(axes)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes


__all__ = ["zonal_flow_response_figure"]
