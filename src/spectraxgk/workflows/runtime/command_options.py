"""Executable runtime command option resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spectraxgk.workflows.runtime.config import RuntimeConfig

RUNTIME_COMMAND_FIT_KEYS = {
    "auto_window",
    "tmin",
    "tmax",
    "window_fraction",
    "min_points",
    "start_fraction",
    "growth_weight",
    "require_positive",
    "min_amp_fraction",
    "mode_method",
}


@dataclass(frozen=True)
class RuntimeLinearCommandOptions:
    """Resolved executable options for one linear runtime command."""

    ky: float
    Nl: int
    Nm: int
    solver: str
    fit_signal: str
    method: str | None
    dt: float | None
    steps: int | None
    sample_stride: int
    method_for_header: str
    dt_for_header: float
    steps_for_header: int
    show_progress: bool


@dataclass(frozen=True)
class RuntimeScanCommandOptions:
    """Resolved executable options for one linear scan command."""

    ky_values: tuple[float, ...]
    Nl: int
    Nm: int
    solver: str
    fit_signal: str
    method: str | None
    dt: float | None
    steps: int | None
    sample_stride: int
    batch_ky: bool
    show_progress: bool
    workers: int
    parallel_executor: str


@dataclass(frozen=True)
class RuntimeNonlinearCommandOptions:
    """Resolved executable options for one nonlinear runtime command."""

    ky: float
    Nl: int
    Nm: int
    method: str
    dt: float
    steps: int | None
    sample_stride: int
    diagnostics_stride: int | None
    diagnostics: bool
    laguerre_mode: str | None
    show_progress: bool


def _arg_or_section(args: Any, section: dict[str, Any], name: str, default: Any) -> Any:
    """Return an executable flag override or a TOML section value."""

    value = getattr(args, name, None)
    if value is not None:
        return value
    return section.get(name, default)


def _resolve_linear_fit_options(
    args: Any, section: dict[str, Any]
) -> tuple[str, str]:
    """Resolve the linear eigensignal solver and fit signal."""

    return (
        str(_arg_or_section(args, section, "solver", "auto")),
        str(_arg_or_section(args, section, "fit_signal", "auto")),
    )


def _resolve_grid_time_options(
    args: Any, section: dict[str, Any], cfg: RuntimeConfig
) -> tuple[int, int, str | None, float | None, int | None, int]:
    """Resolve resolution, optional time controls, and output cadence."""

    method = _arg_or_section(args, section, "method", None)
    dt = _arg_or_section(args, section, "dt", None)
    steps = _arg_or_section(args, section, "steps", None)
    return (
        int(_arg_or_section(args, section, "Nl", 24)),
        int(_arg_or_section(args, section, "Nm", 12)),
        None if method is None else str(method),
        None if dt is None else float(dt),
        None if steps is None else int(steps),
        int(_arg_or_section(args, section, "sample_stride", cfg.time.sample_stride)),
    )


def _runtime_fit_config(data: dict[str, Any]) -> dict[str, Any]:
    """Return fit options supported by runtime executable commands."""

    return {
        k: v for k, v in data.get("fit", {}).items() if k in RUNTIME_COMMAND_FIT_KEYS
    }


def should_show_progress(args: Any, configured: bool) -> bool:
    """Resolve progress output from executable flags, TOML config, and TTY state."""

    import sys

    if getattr(args, "progress", False):
        return True
    if getattr(args, "no_progress", False):
        return False
    return bool(configured or sys.stdout.isatty())


def _resolve_linear_command_options(
    args: Any,
    cfg: RuntimeConfig,
    run_cfg: dict[str, Any],
) -> RuntimeLinearCommandOptions:
    """Resolve linear command options from flags, TOML, and config defaults."""

    Nl, Nm, method, dt, steps, sample_stride = _resolve_grid_time_options(
        args, run_cfg, cfg
    )
    solver, fit_signal = _resolve_linear_fit_options(args, run_cfg)
    dt_for_header = dt if dt is not None else float(cfg.time.dt)
    return RuntimeLinearCommandOptions(
        ky=float(_arg_or_section(args, run_cfg, "ky", 0.3)),
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        fit_signal=fit_signal,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        method_for_header=str(method if method is not None else cfg.time.method),
        dt_for_header=dt_for_header,
        steps_for_header=(
            steps
            if steps is not None
            else int(round(float(cfg.time.t_max) / dt_for_header))
        ),
        show_progress=should_show_progress(args, bool(cfg.time.progress_bar)),
    )


def _parse_ky_values(args: Any, scan_cfg: dict[str, Any]) -> tuple[float, ...]:
    """Resolve scan ky values from CLI or TOML, failing closed on empty scans."""

    raw = getattr(args, "ky_values", None)
    if raw is not None:
        values = tuple(float(x) for x in str(raw).split(",") if x.strip())
    else:
        values = tuple(float(x) for x in scan_cfg.get("ky", ()))
    if not values:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")
    return values


def _resolve_scan_command_options(
    args: Any,
    cfg: RuntimeConfig,
    scan_cfg: dict[str, Any],
) -> RuntimeScanCommandOptions:
    """Resolve linear-scan command options from flags and TOML defaults."""

    Nl, Nm, method, dt, steps, sample_stride = _resolve_grid_time_options(
        args, scan_cfg, cfg
    )
    solver, fit_signal = _resolve_linear_fit_options(args, scan_cfg)
    return RuntimeScanCommandOptions(
        ky_values=_parse_ky_values(args, scan_cfg),
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        fit_signal=fit_signal,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        batch_ky=bool(getattr(args, "batch_ky", False)),
        show_progress=should_show_progress(args, bool(cfg.time.progress_bar)),
        workers=int(getattr(args, "workers", 1)),
        parallel_executor=str(getattr(args, "parallel_executor", "thread")),
    )


def _resolve_nonlinear_command_options(
    args: Any,
    cfg: RuntimeConfig,
    run_cfg: dict[str, Any],
) -> RuntimeNonlinearCommandOptions:
    """Resolve nonlinear command options from flags, TOML, and config defaults."""

    Nl, Nm, method, dt, steps, sample_stride = _resolve_grid_time_options(
        args, run_cfg, cfg
    )
    if steps is not None:
        nonlinear_steps: int | None = steps
    elif bool(cfg.time.fixed_dt):
        nonlinear_steps = int(round(cfg.time.t_max / cfg.time.dt))
    else:
        nonlinear_steps = None

    if getattr(args, "no_diagnostics", False):
        diagnostics = False
    elif getattr(args, "diagnostics", False):
        diagnostics = True
    else:
        diagnostics = bool(run_cfg.get("diagnostics", cfg.time.diagnostics))

    diagnostics_stride = getattr(args, "diagnostics_stride", None)
    laguerre_mode = _arg_or_section(args, run_cfg, "laguerre_mode", None)
    return RuntimeNonlinearCommandOptions(
        ky=float(_arg_or_section(args, run_cfg, "ky", 0.3)),
        Nl=Nl,
        Nm=Nm,
        dt=dt if dt is not None else float(cfg.time.dt),
        steps=nonlinear_steps,
        method=method if method is not None else str(cfg.time.method),
        sample_stride=sample_stride,
        diagnostics_stride=(
            None if diagnostics_stride is None else int(diagnostics_stride)
        ),
        diagnostics=diagnostics,
        laguerre_mode=None if laguerre_mode is None else str(laguerre_mode),
        show_progress=should_show_progress(args, bool(cfg.time.progress_bar)),
    )


__all__ = [
    "RUNTIME_COMMAND_FIT_KEYS",
    "RuntimeLinearCommandOptions",
    "RuntimeNonlinearCommandOptions",
    "RuntimeScanCommandOptions",
    "_arg_or_section",
    "_parse_ky_values",
    "_resolve_grid_time_options",
    "_resolve_linear_command_options",
    "_resolve_linear_fit_options",
    "_resolve_nonlinear_command_options",
    "_resolve_scan_command_options",
    "_runtime_fit_config",
    "should_show_progress",
]
