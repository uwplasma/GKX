"""Runtime TOML case workflows for executable entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeLinearResult, RuntimeNonlinearResult

from spectraxgk.workflows.runtime.commands import (
    RUNTIME_COMMAND_FIT_KEYS,
    RuntimeCommandDeps,
    apply_quasilinear_overrides,
    apply_runtime_path_overrides,
    print_linear_run_header,
    run_runtime_linear_command,
    run_runtime_nonlinear_command,
    runtime_output_path,
    scan_runtime_linear_command,
    should_show_progress,
)

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

_CASE_LINEAR_SPECS = (
    ("ky_target", "run", "ky", 0.3, float),
    ("Nl", "run", "Nl", 24, int),
    ("Nm", "run", "Nm", 12, int),
    ("solver", "run", "solver", "auto", str),
    ("method", "run", "method", None, None),
    ("dt", "run", "dt", None, None),
    ("steps", "run", "steps", None, None),
    ("sample_stride", "time", "sample_stride", None, None),
)
_CASE_NONLINEAR_SPECS = (
    ("ky_target", "run", "ky", 0.3, float),
    ("Nl", "run", "Nl", 4, int),
    ("Nm", "run", "Nm", 8, int),
    ("method", "run", "method", None, None),
    ("dt", "time", "dt", None, None),
    ("steps", "run", "steps", None, None),
    ("sample_stride", "time", "sample_stride", None, None),
    ("diagnostics_stride", "time", "diagnostics_stride", None, None),
)


@dataclass(frozen=True)
class RuntimeCaseDeps:
    """Patchable dependencies for runtime TOML case workflows."""

    load_runtime_from_toml: Callable[[str | Path], tuple[RuntimeConfig, dict[str, Any]]]
    run_runtime_linear: Callable[..., RuntimeLinearResult]
    run_runtime_nonlinear: Callable[..., RuntimeNonlinearResult]
    write_runtime_linear_artifacts: Callable[[str | Path, Any], dict[str, str]]
    run_runtime_nonlinear_with_artifacts: Callable[
        ..., tuple[RuntimeNonlinearResult, dict[str, str]]
    ]


__all__ = [
    "RUNTIME_CASE_FIT_KEYS",
    "RUNTIME_COMMAND_FIT_KEYS",
    "RuntimeCaseDeps",
    "RuntimeCommandDeps",
    "apply_quasilinear_overrides",
    "apply_runtime_path_overrides",
    "print_linear_run_header",
    "run_linear_case",
    "run_nonlinear_case",
    "run_runtime_linear_command",
    "run_runtime_nonlinear_command",
    "runtime_output_path",
    "scan_runtime_linear_command",
    "should_show_progress",
]


def default_runtime_case_deps() -> RuntimeCaseDeps:
    """Build default executable workflow dependencies."""

    from spectraxgk.workflows.runtime.toml import load_runtime_from_toml
    from spectraxgk.runtime import run_runtime_linear, run_runtime_nonlinear
    from spectraxgk.workflows.runtime.artifacts import (
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


def _runtime_case_fit_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return fit options accepted by programmatic runtime-case helpers."""

    return {k: v for k, v in raw.get("fit", {}).items() if k in RUNTIME_CASE_FIT_KEYS}


def _case_run_kwargs(
    raw: dict[str, Any],
    overrides: Mapping[str, Any],
    specs: tuple[tuple[str, str, str, Any, Callable[[Any], Any] | None], ...],
) -> dict[str, Any]:
    """Resolve explicit Python arguments before falling back to TOML/defaults."""

    run_cfg = dict(raw.get("run", {}))
    time_cfg = dict(raw.get("time", {}))
    sections = {"run": run_cfg, "time": time_cfg}
    resolved: dict[str, Any] = {}
    for output_name, section_name, input_name, default, converter in specs:
        value = overrides.get(input_name)
        if value is None:
            value = sections[section_name].get(input_name, default)
        resolved[output_name] = converter(value) if converter is not None else value
    return resolved


def _nonlinear_case_run_kwargs(
    raw: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve keyword arguments for one programmatic nonlinear TOML case."""

    resolved = _case_run_kwargs(raw, overrides, _CASE_NONLINEAR_SPECS)
    resolved["diagnostics"] = True
    return resolved


def _linear_case_run_kwargs(
    raw: dict[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve keyword arguments for one programmatic linear TOML case."""

    return _case_run_kwargs(raw, overrides, _CASE_LINEAR_SPECS)


def _status_printer(message: str) -> None:
    """Print programmatic runtime-case progress messages."""

    print(f"runtime: {message}")


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
    run_kwargs = _linear_case_run_kwargs(
        raw,
        {
            "ky": ky,
            "Nl": Nl,
            "Nm": Nm,
            "solver": solver,
            "method": method,
            "dt": dt,
            "steps": steps,
            "sample_stride": sample_stride,
        },
    )
    run_kwargs["show_progress"] = show_progress

    result = case_deps.run_runtime_linear(
        cfg,
        **run_kwargs,
        **_runtime_case_fit_config(raw),
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
    run_kwargs = _nonlinear_case_run_kwargs(
        raw,
        {
            "ky": ky,
            "Nl": Nl,
            "Nm": Nm,
            "method": method,
            "dt": dt,
            "steps": steps,
            "sample_stride": sample_stride,
            "diagnostics_stride": diagnostics_stride,
        },
    )
    run_kwargs["show_progress"] = show_progress

    if cfg.output.path:
        result, paths = case_deps.run_runtime_nonlinear_with_artifacts(
            cfg,
            out=cfg.output.path,
            **run_kwargs,
            status_callback=_status_printer,
        )
        if "summary" in paths:
            print(f"saved {paths['summary']}")
    else:
        result = case_deps.run_runtime_nonlinear(
            cfg,
            resolved_diagnostics=False,
            **run_kwargs,
            status_callback=_status_printer,
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
