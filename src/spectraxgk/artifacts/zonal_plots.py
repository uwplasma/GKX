"""Zonal-response plotting helpers."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.artifacts.plot_style import set_plot_style

def zonal_flow_response_figure(
    t: np.ndarray,
    response: np.ndarray,
    *,
    metrics=None,
    title: str = "Zonal-flow response",
    y_label: str = "normalized response",
) -> Tuple[plt.Figure, np.ndarray]:
    """Render a zonal-flow response trace and its envelope summary."""

    from spectraxgk.validation.benchmarks.harness import zonal_flow_response_metrics

    set_plot_style()
    t_arr = np.asarray(t, dtype=float)
    resp = np.asarray(response, dtype=float)
    if t_arr.ndim != 1 or resp.ndim != 1 or t_arr.size != resp.size:
        raise ValueError("t and response must be one-dimensional arrays of equal length")
    if metrics is None:
        metrics = zonal_flow_response_metrics(t_arr, resp)

    response_norm = resp / float(metrics.initial_level)
    residual = float(metrics.residual_level)
    env_t = np.asarray(metrics.peak_times, dtype=float)
    env_y = np.asarray(metrics.peak_envelope, dtype=float)
    fit_count = int(getattr(metrics, "peak_fit_count", env_t.size))
    fit_tmin = float(getattr(metrics, "fit_tmin", t_arr[0]))
    fit_tmax = float(getattr(metrics, "fit_tmax", t_arr[-1]))
    damping_method = str(getattr(metrics, "damping_method", "combined_envelope"))
    frequency_method = str(getattr(metrics, "frequency_method", "peak_spacing"))
    max_peak_t = np.asarray(getattr(metrics, "max_peak_times", np.asarray([], dtype=float)), dtype=float)
    max_peak_y = np.asarray(getattr(metrics, "max_peak_values", np.asarray([], dtype=float)), dtype=float)
    min_peak_t = np.asarray(getattr(metrics, "min_peak_times", np.asarray([], dtype=float)), dtype=float)
    min_peak_y = np.asarray(getattr(metrics, "min_peak_values", np.asarray([], dtype=float)), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax0, ax1 = axes

    ax0.plot(t_arr, response_norm, color="#0f4c81", linewidth=2.2, label="response")
    ax0.axhline(residual, color="#c44e52", linestyle="--", linewidth=2.0, label="residual")
    ax0.axvspan(fit_tmin, fit_tmax, color="#d9ead3", alpha=0.22, linewidth=0.0)
    if damping_method == "branchwise_extrema":
        if max_peak_t.size:
            keep = (max_peak_t >= fit_tmin) & (max_peak_t <= fit_tmax)
            ax0.plot(max_peak_t[keep], max_peak_y[keep], linestyle="none", marker="o", color="#2a9d8f", markersize=5.2, label="maxima fit points")
        if min_peak_t.size:
            keep = (min_peak_t >= fit_tmin) & (min_peak_t <= fit_tmax)
            ax0.plot(min_peak_t[keep], min_peak_y[keep], linestyle="none", marker="o", color="#7b2cbf", markersize=5.2, label="minima fit points")
    ax0.fill_between(
        t_arr,
        residual - float(metrics.residual_std),
        residual + float(metrics.residual_std),
        color="#c44e52",
        alpha=0.15,
        linewidth=0.0,
    )
    ax0.set_xlabel("t")
    ax0.set_ylabel(y_label)
    ax0.set_title("Normalized response")
    ax0.legend(loc="best", frameon=False)

    ax1.plot(t_arr, np.maximum(np.abs(response_norm - residual), 1.0e-14), color="#4c956c", linewidth=2.0, alpha=0.5)
    if env_t.size:
        ax1.plot(env_t, env_y, color="#c44e52", marker="o", linewidth=1.8, label="envelope peaks")
    fit_env_t = env_t[(env_t >= fit_tmin) & (env_t <= fit_tmax)]
    fit_env_y = env_y[(env_t >= fit_tmin) & (env_t <= fit_tmax)]
    if damping_method == "combined_envelope" and fit_count >= 2 and np.isfinite(float(metrics.gam_damping_rate)) and fit_env_t.size:
        fit_n = min(fit_count, fit_env_t.size)
        fit_t = fit_env_t[:fit_n]
        fit = fit_env_y[0] * np.exp(-float(metrics.gam_damping_rate) * (fit_t - fit_t[0]))
        label = "envelope fit" if fit_n == fit_env_t.size else f"envelope fit (first {fit_n} peaks)"
        ax1.plot(fit_t, fit, color="#2a9d8f", linestyle="--", linewidth=2.0, label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("t")
    ax1.set_ylabel("envelope")
    ax1.set_title("GAM envelope")
    if env_t.size:
        ax1.legend(loc="best", frameon=False)
    ax1.text(
        0.03,
        0.97,
        (
            f"residual = {metrics.residual_level:.4f}\n"
            f"std = {metrics.residual_std:.4f}\n"
            f"ω_GAM = {metrics.gam_frequency:.4f}\n"
            f"γ_damp = {metrics.gam_damping_rate:.4f}\n"
            f"fit_peaks = {fit_count}\n"
            f"norm = {getattr(metrics, 'initial_policy', 'window_abs_mean')}\n"
            f"damp = {damping_method}\n"
            f"freq = {frequency_method}\n"
            f"fit_t = [{fit_tmin:.1f}, {fit_tmax:.1f}]"
        ),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig, axes

__all__ = ["zonal_flow_response_figure"]
