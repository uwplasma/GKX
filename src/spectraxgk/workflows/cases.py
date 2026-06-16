"""Runtime TOML case workflows for executable entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from spectraxgk.runtime_config import RuntimeConfig
from spectraxgk.runtime_results import RuntimeLinearResult, RuntimeNonlinearResult

RUNTIME_CASE_FIT_KEYS = {
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
    "fit_signal",
}


@dataclass(frozen=True)
class RuntimeCaseDeps:
    """Patchable dependencies for runtime TOML case workflows."""

    load_runtime_from_toml: Callable[[str | Path], tuple[RuntimeConfig, dict[str, Any]]]
    run_runtime_linear: Callable[..., RuntimeLinearResult]
    run_runtime_nonlinear: Callable[..., RuntimeNonlinearResult]
    write_runtime_linear_artifacts: Callable[[str | Path, Any], dict[str, str]]
    run_runtime_nonlinear_with_artifacts: Callable[..., tuple[RuntimeNonlinearResult, dict[str, str]]]


__all__ = [
    "RUNTIME_CASE_FIT_KEYS",
    "RuntimeCaseDeps",
    "run_linear_case",
    "run_nonlinear_case",
]


def default_runtime_case_deps() -> RuntimeCaseDeps:
    """Build default executable workflow dependencies."""

    from spectraxgk.io import load_runtime_from_toml
    from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
    from spectraxgk.runtime_artifacts import (
        run_runtime_nonlinear_with_artifacts,
        write_runtime_linear_artifacts,
    )

    return RuntimeCaseDeps(
        load_runtime_from_toml=load_runtime_from_toml,
        run_runtime_linear=run_runtime_linear,
        run_runtime_nonlinear=run_runtime_nonlinear,
        write_runtime_linear_artifacts=write_runtime_linear_artifacts,
        run_runtime_nonlinear_with_artifacts=run_runtime_nonlinear_with_artifacts,
    )


def run_linear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    solver: str | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    show_progress: bool = True,
    deps: RuntimeCaseDeps | None = None,
) -> int:
    """Run a linear case from a runtime TOML with optional overrides."""

    case_deps = default_runtime_case_deps() if deps is None else deps
    cfg, raw = case_deps.load_runtime_from_toml(config_path)
    run_cfg = dict(raw.get("run", {}))
    fit_cfg = {
        k: v for k, v in raw.get("fit", {}).items() if k in RUNTIME_CASE_FIT_KEYS
    }

    result = case_deps.run_runtime_linear(
        cfg,
        ky_target=float(ky if ky is not None else run_cfg.get("ky", 0.3)),
        Nl=int(Nl if Nl is not None else run_cfg.get("Nl", 24)),
        Nm=int(Nm if Nm is not None else run_cfg.get("Nm", 12)),
        solver=str(solver if solver is not None else run_cfg.get("solver", "auto")),
        method=method if method is not None else run_cfg.get("method", None),
        dt=dt if dt is not None else run_cfg.get("dt", None),
        steps=steps if steps is not None else run_cfg.get("steps", None),
        sample_stride=sample_stride
        if sample_stride is not None
        else raw.get("time", {}).get("sample_stride", None),
        show_progress=show_progress,
        **fit_cfg,
    )
    if cfg.output.path:
        paths = case_deps.write_runtime_linear_artifacts(cfg.output.path, result)
        if "summary" in paths:
            print(f"saved {paths['summary']}")
    print(f"ky={result.ky:.6f} gamma={result.gamma:.8f} omega={result.omega:.8f}")
    return 0


def run_nonlinear_case(
    config_path: str | Path,
    *,
    ky: float | None = None,
    Nl: int | None = None,
    Nm: int | None = None,
    method: str | None = None,
    dt: float | None = None,
    steps: int | None = None,
    sample_stride: int | None = None,
    diagnostics_stride: int | None = None,
    show_progress: bool = True,
    deps: RuntimeCaseDeps | None = None,
) -> int:
    """Run a nonlinear case from a runtime TOML with optional overrides."""

    case_deps = default_runtime_case_deps() if deps is None else deps
    cfg, raw = case_deps.load_runtime_from_toml(config_path)
    run_cfg = dict(raw.get("run", {}))
    time_cfg = dict(raw.get("time", {}))

    def _status(message: str) -> None:
        print(f"runtime: {message}")

    ky_target = float(ky if ky is not None else run_cfg.get("ky", 0.3))
    Nl_use = int(Nl if Nl is not None else run_cfg.get("Nl", 4))
    Nm_use = int(Nm if Nm is not None else run_cfg.get("Nm", 8))
    method_use = method if method is not None else run_cfg.get("method", None)
    dt_use = dt if dt is not None else time_cfg.get("dt", None)
    steps_use = steps if steps is not None else run_cfg.get("steps", None)
    sample_stride_use = (
        sample_stride
        if sample_stride is not None
        else time_cfg.get("sample_stride", None)
    )
    diagnostics_stride_use = (
        diagnostics_stride
        if diagnostics_stride is not None
        else time_cfg.get("diagnostics_stride", None)
    )

    if cfg.output.path:
        result, paths = case_deps.run_runtime_nonlinear_with_artifacts(
            cfg,
            out=cfg.output.path,
            ky_target=ky_target,
            Nl=Nl_use,
            Nm=Nm_use,
            dt=dt_use,
            steps=steps_use,
            method=method_use,
            sample_stride=sample_stride_use,
            diagnostics_stride=diagnostics_stride_use,
            diagnostics=True,
            show_progress=show_progress,
            status_callback=_status,
        )
        if "summary" in paths:
            print(f"saved {paths['summary']}")
    else:
        result = case_deps.run_runtime_nonlinear(
            cfg,
            ky_target=ky_target,
            Nl=Nl_use,
            Nm=Nm_use,
            method=method_use,
            dt=dt_use,
            steps=steps_use,
            sample_stride=sample_stride_use,
            diagnostics_stride=diagnostics_stride_use,
            diagnostics=True,
            resolved_diagnostics=False,
            show_progress=show_progress,
            status_callback=_status,
        )
    if result.diagnostics is None or result.ky_selected is None:
        print("completed without streamed diagnostics")
        return 0
    diag = result.diagnostics
    print(
        "ky={:.6f} Wg={:.8e} Wphi={:.8e} heat={:.8e} pflux={:.8e}".format(
            float(result.ky_selected),
            float(diag.Wg_t[-1]),
            float(diag.Wphi_t[-1]),
            float(diag.heat_flux_t[-1]),
            float(diag.particle_flux_t[-1]),
        )
    )
    return 0
