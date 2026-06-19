"""Runtime executable command workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from spectraxgk.workflows.runtime.config import RuntimeConfig
from spectraxgk.workflows.runtime.results import RuntimeLinearResult, RuntimeNonlinearResult

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

_PRELOADED_RUNTIME_CONFIG_ATTR = "_spectraxgk_preloaded_runtime_config"
_PRELOADED_RUNTIME_DATA_ATTR = "_spectraxgk_preloaded_runtime_data"
_LINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "timeseries",
    "eigenfunction",
    "state",
    "quasilinear_summary",
    "quasilinear_species",
)
_SCAN_ARTIFACT_DISPLAY_KEYS = ("summary", "scan", "quasilinear_spectrum")
_QUASILINEAR_ARTIFACT_DISPLAY_KEYS = ("quasilinear_summary", "quasilinear_species")
_NONLINEAR_ARTIFACT_DISPLAY_KEYS = (
    "summary",
    "diagnostics",
    "state",
    "out",
    "big",
    "restart",
)


@dataclass(frozen=True)
class RuntimeCommandDeps:
    """Patchable dependencies for executable runtime subcommands."""

    load_runtime_from_toml: Callable[[str | Path], tuple[RuntimeConfig, dict[str, Any]]]
    run_runtime_linear: Callable[..., RuntimeLinearResult]
    run_runtime_scan: Callable[..., Any]
    run_runtime_nonlinear_with_artifacts: Callable[
        ..., tuple[RuntimeNonlinearResult, dict[str, str]]
    ]
    write_runtime_linear_artifacts: Callable[
        [str | Path, RuntimeLinearResult], dict[str, str]
    ]
    write_runtime_linear_scan_artifacts: Callable[[str | Path, Any], dict[str, str]]
    write_quasilinear_artifacts: Callable[[str | Path, dict[str, Any]], dict[str, str]]
    resolve_runtime_path: Callable[..., str | None]


def build_runtime_command_deps(facade: Any) -> RuntimeCommandDeps:
    """Build runtime command dependencies from a patchable executable facade."""

    return RuntimeCommandDeps(
        load_runtime_from_toml=facade.load_runtime_from_toml,
        run_runtime_linear=facade.run_runtime_linear,
        run_runtime_scan=facade.run_runtime_scan,
        run_runtime_nonlinear_with_artifacts=facade.run_runtime_nonlinear_with_artifacts,
        write_runtime_linear_artifacts=facade.write_runtime_linear_artifacts,
        write_runtime_linear_scan_artifacts=facade.write_runtime_linear_scan_artifacts,
        write_quasilinear_artifacts=facade.write_quasilinear_artifacts,
        resolve_runtime_path=facade.resolve_runtime_path,
    )


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


def attach_preloaded_runtime_config(
    args: Any,
    cfg: RuntimeConfig,
    data: dict[str, Any],
) -> None:
    """Attach already-loaded runtime TOML data to a parser namespace.

    The generic ``run`` dispatcher inspects a config to choose linear or
    nonlinear execution. Passing the loaded object forward avoids a second TOML
    parse while direct subcommands remain self-contained.
    """

    setattr(args, _PRELOADED_RUNTIME_CONFIG_ATTR, cfg)
    setattr(args, _PRELOADED_RUNTIME_DATA_ATTR, data)


def load_runtime_command_config(
    args: Any,
    *,
    deps: RuntimeCommandDeps,
) -> tuple[RuntimeConfig, dict[str, Any]]:
    """Load runtime TOML data, reusing the generic-dispatch preload if present."""

    cfg = getattr(args, _PRELOADED_RUNTIME_CONFIG_ATTR, None)
    data = getattr(args, _PRELOADED_RUNTIME_DATA_ATTR, None)
    if cfg is not None and data is not None:
        return cast(RuntimeConfig, cfg), cast(dict[str, Any], data)
    return deps.load_runtime_from_toml(args.config)


def _prepare_runtime_command_config(
    args: Any,
    *,
    deps: RuntimeCommandDeps,
    path_overrides: bool,
    quasilinear_overrides: bool,
) -> tuple[RuntimeConfig, dict[str, Any]]:
    """Load runtime command config and apply the command-specific overrides."""

    cfg, data = load_runtime_command_config(args, deps=deps)
    if path_overrides:
        cfg = apply_runtime_path_overrides(
            cfg,
            args,
            resolve_runtime_path=deps.resolve_runtime_path,
        )
    if quasilinear_overrides:
        cfg = apply_quasilinear_overrides(cfg, args)
    return cfg, data


def _arg_or_section(args: Any, section: dict[str, Any], name: str, default: Any) -> Any:
    """Return a CLI override or a TOML section value."""

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


def _resolve_linear_command_options(
    args: Any,
    cfg: RuntimeConfig,
    run_cfg: dict[str, Any],
) -> RuntimeLinearCommandOptions:
    """Resolve linear command options from CLI flags, TOML, and config defaults."""

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
    """Resolve linear-scan command options from CLI flags and TOML defaults."""

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
    """Resolve nonlinear command options from CLI flags, TOML, and config defaults."""

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


def runtime_output_path(args: Any, cfg: RuntimeConfig) -> str | None:
    """Return the executable output path override or TOML output path."""

    if getattr(args, "out", None) is not None:
        return str(args.out)
    return cfg.output.path


def should_show_progress(args: Any, configured: bool) -> bool:
    """Resolve progress output from CLI flags, TOML config, and TTY state."""

    import sys

    if getattr(args, "progress", False):
        return True
    if getattr(args, "no_progress", False):
        return False
    return bool(configured or sys.stdout.isatty())


def print_linear_run_header(
    *,
    label: str,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    solver: str,
    method: str,
    dt: float,
    steps: int,
    grid_shape: tuple[int, int, int],
    show_progress: bool,
    extra: str | None = None,
) -> None:
    """Print the standard executable header for linear initial-value runs."""

    print(f"starting {label}")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"solver={solver} method={method} dt={dt:.6g} steps={steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"progress={'on' if show_progress else 'off'}"
    )
    if extra is not None:
        print(extra)


def print_nonlinear_run_header(
    *,
    config_path: str,
    ky: float,
    Nl: int,
    Nm: int,
    method: str,
    dt: float,
    steps: int | None,
    grid_shape: tuple[int, int, int],
    diagnostics: bool,
    show_progress: bool,
) -> None:
    """Print the standard executable header for nonlinear initial-value runs."""

    print("starting runtime nonlinear run")
    print(
        f"config={config_path} ky={ky:.4f} Nl={Nl} Nm={Nm} "
        f"method={method} dt={dt:.6g} "
        f"steps={'auto' if steps is None else steps}"
    )
    print(
        f"grid=Nx{grid_shape[0]} Ny{grid_shape[1]} Nz{grid_shape[2]} "
        f"diagnostics={'on' if diagnostics else 'off'} "
        f"progress={'on' if show_progress else 'off'}"
    )


def print_nonlinear_run_summary(result: RuntimeNonlinearResult) -> bool:
    """Print final nonlinear diagnostics and return whether diagnostics exist."""

    import numpy as np

    diag = result.diagnostics
    if diag is None:
        print("nonlinear run completed")
        return False
    t_values = np.asarray(diag.t)
    t_last = float(t_values[-1]) if t_values.size else 0.0
    print(
        "nonlinear: "
        f"t={t_last:.6g} "
        f"ky_sel={result.ky_selected:.6g} "
        f"kx_sel={result.kx_selected:.6g} "
        f"dt_mean={float(diag.dt_mean):.6g} "
        f"Wg={float(diag.Wg_t[-1]):.6g} "
        f"Wphi={float(diag.Wphi_t[-1]):.6g} "
        f"Wapar={float(diag.Wapar_t[-1]):.6g}"
    )
    return True


def apply_runtime_path_overrides(
    cfg: RuntimeConfig,
    args: Any,
    *,
    resolve_runtime_path: Callable[..., str | None],
) -> RuntimeConfig:
    """Apply cwd-resolved executable path overrides for geometry and init files."""

    from dataclasses import replace

    cwd = Path.cwd()
    geometry = cfg.geometry
    vmec_cli = getattr(args, "vmec_file", None)
    geom_cli = getattr(args, "geometry_file", None)
    if vmec_cli is not None:
        geometry = replace(
            geometry, vmec_file=resolve_runtime_path(str(vmec_cli), base_dir=cwd)
        )
    if geom_cli is not None:
        geometry = replace(
            geometry, geometry_file=resolve_runtime_path(str(geom_cli), base_dir=cwd)
        )

    init = cfg.init
    init_cli = getattr(args, "init_file", None)
    if init_cli is not None:
        init = replace(
            init, init_file=resolve_runtime_path(str(init_cli), base_dir=cwd)
        )

    return replace(cfg, geometry=geometry, init=init)


def apply_quasilinear_overrides(cfg: RuntimeConfig, args: Any) -> RuntimeConfig:
    """Apply executable quasilinear diagnostic overrides."""

    from dataclasses import replace

    ql = cfg.quasilinear
    updates: dict[str, object] = {}
    if getattr(args, "quasilinear", False):
        updates["enabled"] = True
    mapping = {
        "ql_mode": "mode",
        "ql_saturation_rule": "saturation_rule",
        "ql_csat": "csat",
        "ql_normalization": "amplitude_normalization",
        "ql_output": "output_path",
    }
    for arg_name, field_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            updates[field_name] = value
    if not updates:
        return cfg
    return replace(cfg, quasilinear=replace(ql, **cast(Any, updates)))


def _status_printer(prefix: str) -> Callable[[str], None]:
    def _emit(message: str) -> None:
        print(f"{prefix}: {message}", flush=True)

    return _emit


def _print_saved_paths(paths: Mapping[str, str], keys: Sequence[str]) -> None:
    """Print saved artifact paths in the command-defined display order."""

    for key in keys:
        if key in paths:
            print(f"saved {paths[key]}")


def _write_command_outputs(
    out_path: str | None,
    payload: Any | None,
    *,
    writer: Callable[[str | Path, Any], dict[str, str]],
    display_keys: Sequence[str],
) -> dict[str, str]:
    """Write command artifacts when both destination and payload exist."""

    if out_path is None or payload is None:
        return {}
    paths = writer(out_path, payload)
    _print_saved_paths(paths, display_keys)
    return paths


def _write_linear_runtime_command_outputs(
    args: Any,
    cfg: RuntimeConfig,
    result: RuntimeLinearResult,
    *,
    deps: RuntimeCommandDeps,
) -> dict[str, dict[str, str]]:
    """Write all optional artifacts produced by one linear runtime command."""

    linear_paths = _write_command_outputs(
        runtime_output_path(args, cfg),
        result,
        writer=deps.write_runtime_linear_artifacts,
        display_keys=_LINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    ql_paths = _write_command_outputs(
        getattr(args, "ql_output", None) or cfg.quasilinear.output_path,
        result.quasilinear,
        writer=deps.write_quasilinear_artifacts,
        display_keys=_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
    )
    return {"linear": linear_paths, "quasilinear": ql_paths}


def _write_scan_runtime_command_outputs(
    args: Any,
    cfg: RuntimeConfig,
    scan: Any,
    *,
    deps: RuntimeCommandDeps,
) -> dict[str, str]:
    """Write optional artifacts produced by one linear-scan runtime command."""

    return _write_command_outputs(
        runtime_output_path(args, cfg) or cfg.quasilinear.output_path,
        scan,
        writer=deps.write_runtime_linear_scan_artifacts,
        display_keys=_SCAN_ARTIFACT_DISPLAY_KEYS,
    )


def print_nonlinear_command_outputs(paths: Mapping[str, str], *, enabled: bool) -> None:
    """Print nonlinear artifact paths after diagnostics confirm a saved run."""

    if enabled:
        _print_saved_paths(paths, _NONLINEAR_ARTIFACT_DISPLAY_KEYS)


def plot_saved_output_command(
    argv: Sequence[str],
    *,
    plot_saved_output: Callable[..., Path],
) -> int:
    """Render a saved runtime artifact from the top-level ``--plot`` command."""

    if len(argv) < 2:
        print("usage: spectraxgk --plot OUTPUT_FILE [--out FIGURE.png]")
        return 1
    input_path = argv[1]
    out_path = None
    if len(argv) > 2:
        if len(argv) == 4 and argv[2] == "--out":
            out_path = argv[3]
        else:
            print("usage: spectraxgk --plot OUTPUT_FILE [--out FIGURE.png]")
            return 1
    rendered = plot_saved_output(input_path, out=out_path)
    print(f"saved {rendered}")
    return 0


def run_runtime_linear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-linear subcommand after parser dispatch."""

    cfg, data = _prepare_runtime_command_config(
        args,
        deps=deps,
        path_overrides=True,
        quasilinear_overrides=True,
    )
    run_cfg = data.get("run", {})
    fit_cfg = _runtime_fit_config(data)
    opts = _resolve_linear_command_options(args, cfg, run_cfg)

    print_linear_run_header(
        label="runtime linear run",
        config_path=str(args.config),
        ky=opts.ky,
        Nl=opts.Nl,
        Nm=opts.Nm,
        solver=opts.solver,
        method=opts.method_for_header,
        dt=opts.dt_for_header,
        steps=opts.steps_for_header,
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        show_progress=opts.show_progress,
        extra=(
            f"model={cfg.physics.reduced_model} electrostatic={cfg.physics.electrostatic} "
            f"electromagnetic={cfg.physics.electromagnetic} fit_signal={opts.fit_signal}"
        ),
    )

    res = deps.run_runtime_linear(
        cfg,
        ky_target=opts.ky,
        Nl=opts.Nl,
        Nm=opts.Nm,
        solver=opts.solver,
        method=opts.method,
        dt=opts.dt,
        steps=opts.steps,
        sample_stride=opts.sample_stride,
        fit_signal=opts.fit_signal,
        show_progress=opts.show_progress,
        status_callback=_status_printer("runtime"),
        **fit_cfg,
    )
    print(f"ky={res.ky:.4f} gamma={res.gamma:.6f} omega={res.omega:.6f}")
    _write_linear_runtime_command_outputs(args, cfg, res, deps=deps)
    return 0


def scan_runtime_linear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-linear ky-scan subcommand after parser dispatch."""

    cfg, data = _prepare_runtime_command_config(
        args,
        deps=deps,
        path_overrides=False,
        quasilinear_overrides=True,
    )
    scan_cfg = data.get("scan", {})
    fit_cfg = _runtime_fit_config(data)
    opts = _resolve_scan_command_options(args, cfg, scan_cfg)

    scan = deps.run_runtime_scan(
        cfg,
        list(opts.ky_values),
        Nl=opts.Nl,
        Nm=opts.Nm,
        solver=opts.solver,
        method=opts.method,
        dt=opts.dt,
        steps=opts.steps,
        sample_stride=opts.sample_stride,
        batch_ky=opts.batch_ky,
        fit_signal=opts.fit_signal,
        show_progress=opts.show_progress,
        workers=opts.workers,
        parallel_executor=opts.parallel_executor,
        **fit_cfg,
    )
    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
    _write_scan_runtime_command_outputs(args, cfg, scan, deps=deps)
    return 0


def run_runtime_nonlinear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-nonlinear subcommand after parser dispatch."""

    cfg, data = _prepare_runtime_command_config(
        args,
        deps=deps,
        path_overrides=True,
        quasilinear_overrides=False,
    )
    run_cfg = data.get("run", {})
    opts = _resolve_nonlinear_command_options(args, cfg, run_cfg)

    print_nonlinear_run_header(
        config_path=str(args.config),
        ky=opts.ky,
        Nl=opts.Nl,
        Nm=opts.Nm,
        method=opts.method,
        dt=opts.dt,
        steps=opts.steps,
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        diagnostics=opts.diagnostics,
        show_progress=opts.show_progress,
    )

    out_path = runtime_output_path(args, cfg)
    result, paths = deps.run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=opts.ky,
        Nl=opts.Nl,
        Nm=opts.Nm,
        dt=opts.dt,
        steps=opts.steps,
        method=opts.method,
        sample_stride=opts.sample_stride,
        diagnostics_stride=opts.diagnostics_stride,
        laguerre_mode=opts.laguerre_mode,
        diagnostics=opts.diagnostics,
        show_progress=opts.show_progress,
        status_callback=_status_printer("runtime"),
    )
    if not print_nonlinear_run_summary(result):
        return 0
    print_nonlinear_command_outputs(paths, enabled=out_path is not None)
    return 0
