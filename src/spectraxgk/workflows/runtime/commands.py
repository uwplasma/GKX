"""Runtime executable command workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

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


def run_runtime_linear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-linear subcommand after parser dispatch."""

    cfg, data = deps.load_runtime_from_toml(args.config)
    cfg = apply_runtime_path_overrides(
        cfg, args, resolve_runtime_path=deps.resolve_runtime_path
    )
    cfg = apply_quasilinear_overrides(cfg, args)
    run_cfg = data.get("run", {})
    fit_cfg = {
        k: v for k, v in data.get("fit", {}).items() if k in RUNTIME_COMMAND_FIT_KEYS
    }

    ky = float(args.ky if args.ky is not None else run_cfg.get("ky", 0.3))
    Nl = int(args.Nl if args.Nl is not None else run_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else run_cfg.get("Nm", 12))
    solver = str(
        args.solver if args.solver is not None else run_cfg.get("solver", "auto")
    )
    fit_signal = str(
        args.fit_signal
        if args.fit_signal is not None
        else run_cfg.get("fit_signal", "auto")
    )
    method = args.method if args.method is not None else run_cfg.get("method", None)
    dt = args.dt if args.dt is not None else run_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else run_cfg.get("steps", None)
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    dt_use = float(dt if dt is not None else cfg.time.dt)
    steps_use = (
        int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_use))
    )
    method_use = str(method if method is not None else cfg.time.method)
    show_progress = should_show_progress(args, bool(cfg.time.progress_bar))

    print_linear_run_header(
        label="runtime linear run",
        config_path=str(args.config),
        ky=ky,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method_use,
        dt=dt_use,
        steps=steps_use,
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        show_progress=show_progress,
        extra=(
            f"model={cfg.physics.reduced_model} electrostatic={cfg.physics.electrostatic} "
            f"electromagnetic={cfg.physics.electromagnetic} fit_signal={fit_signal}"
        ),
    )

    res = deps.run_runtime_linear(
        cfg,
        ky_target=ky,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        fit_signal=fit_signal,
        show_progress=show_progress,
        status_callback=_status_printer("runtime"),
        **fit_cfg,
    )
    print(f"ky={res.ky:.4f} gamma={res.gamma:.6f} omega={res.omega:.6f}")
    out_path = runtime_output_path(args, cfg)
    if out_path is not None:
        paths = deps.write_runtime_linear_artifacts(out_path, res)
        print(f"saved {paths['summary']}")
        for key in (
            "timeseries",
            "eigenfunction",
            "state",
            "quasilinear_summary",
            "quasilinear_species",
        ):
            if key in paths:
                print(f"saved {paths[key]}")
    ql_output = getattr(args, "ql_output", None) or cfg.quasilinear.output_path
    if ql_output is not None and res.quasilinear is not None:
        paths = deps.write_quasilinear_artifacts(str(ql_output), res.quasilinear)
        print(f"saved {paths['quasilinear_summary']}")
        if "quasilinear_species" in paths:
            print(f"saved {paths['quasilinear_species']}")
    return 0


def scan_runtime_linear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-linear ky-scan subcommand after parser dispatch."""

    import numpy as np

    cfg, data = deps.load_runtime_from_toml(args.config)
    cfg = apply_quasilinear_overrides(cfg, args)
    scan_cfg = data.get("scan", {})
    fit_cfg = {
        k: v for k, v in data.get("fit", {}).items() if k in RUNTIME_COMMAND_FIT_KEYS
    }

    if args.ky_values is not None:
        ky_values = np.asarray(
            [float(x) for x in args.ky_values.split(",") if x.strip()], dtype=float
        )
    else:
        ky_values = np.asarray(scan_cfg.get("ky", []), dtype=float)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = int(args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12))
    solver = str(
        args.solver if args.solver is not None else scan_cfg.get("solver", "auto")
    )
    fit_signal = str(
        args.fit_signal
        if args.fit_signal is not None
        else scan_cfg.get("fit_signal", "auto")
    )
    method = args.method if args.method is not None else scan_cfg.get("method", None)
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else scan_cfg.get("steps", None)
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else scan_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    show_progress = should_show_progress(args, bool(cfg.time.progress_bar))

    scan = deps.run_runtime_scan(
        cfg,
        list(ky_values.tolist()),
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        batch_ky=bool(args.batch_ky),
        fit_signal=fit_signal,
        show_progress=show_progress,
        workers=int(getattr(args, "workers", 1)),
        parallel_executor=str(getattr(args, "parallel_executor", "thread")),
        **fit_cfg,
    )
    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
    out_path = runtime_output_path(args, cfg) or cfg.quasilinear.output_path
    if out_path is not None:
        paths = deps.write_runtime_linear_scan_artifacts(out_path, scan)
        print(f"saved {paths['summary']}")
        print(f"saved {paths['scan']}")
        if "quasilinear_spectrum" in paths:
            print(f"saved {paths['quasilinear_spectrum']}")
    return 0


def run_runtime_nonlinear_command(args: Any, *, deps: RuntimeCommandDeps) -> int:
    """Execute the runtime-nonlinear subcommand after parser dispatch."""

    import numpy as np

    cfg, data = deps.load_runtime_from_toml(args.config)
    cfg = apply_runtime_path_overrides(
        cfg, args, resolve_runtime_path=deps.resolve_runtime_path
    )
    run_cfg = data.get("run", {})

    ky = float(args.ky if args.ky is not None else run_cfg.get("ky", 0.3))
    Nl = int(args.Nl if args.Nl is not None else run_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else run_cfg.get("Nm", 12))
    dt = float(args.dt if args.dt is not None else run_cfg.get("dt", cfg.time.dt))
    if args.steps is not None:
        steps: int | None = int(args.steps)
    elif run_cfg.get("steps", None) is not None:
        steps = int(run_cfg["steps"])
    elif bool(cfg.time.fixed_dt):
        steps = int(round(cfg.time.t_max / cfg.time.dt))
    else:
        steps = None
    method = str(
        args.method
        if args.method is not None
        else run_cfg.get("method", cfg.time.method)
    )
    sample_stride = int(
        args.sample_stride
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    diagnostics_stride = (
        None if args.diagnostics_stride is None else int(args.diagnostics_stride)
    )
    if args.no_diagnostics:
        diagnostics = False
    elif args.diagnostics:
        diagnostics = True
    else:
        diagnostics = run_cfg.get("diagnostics", cfg.time.diagnostics)
    laguerre_mode = (
        args.laguerre_mode
        if args.laguerre_mode is not None
        else run_cfg.get("laguerre_mode")
    )
    show_progress = should_show_progress(args, bool(cfg.time.progress_bar))

    print("starting runtime nonlinear run")
    print(
        f"config={args.config} ky={ky:.4f} Nl={Nl} Nm={Nm} method={method} dt={dt:.6g} "
        f"steps={'auto' if steps is None else steps}"
    )
    print(
        f"grid=Nx{int(cfg.grid.Nx)} Ny{int(cfg.grid.Ny)} Nz{int(cfg.grid.Nz)} "
        f"diagnostics={'on' if diagnostics else 'off'} progress={'on' if show_progress else 'off'}"
    )

    out_path = runtime_output_path(args, cfg)
    result, paths = deps.run_runtime_nonlinear_with_artifacts(
        cfg,
        out=out_path,
        ky_target=ky,
        Nl=Nl,
        Nm=Nm,
        dt=dt,
        steps=steps,
        method=method,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        laguerre_mode=laguerre_mode,
        diagnostics=diagnostics,
        show_progress=show_progress,
        status_callback=_status_printer("runtime"),
    )
    diag = result.diagnostics
    if diag is None:
        print("nonlinear run completed")
        return 0
    t_last = float(np.asarray(diag.t)[-1]) if np.asarray(diag.t).size else 0.0

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
    if out_path is not None:
        for key in ("summary", "diagnostics", "state", "out", "big", "restart"):
            if key in paths:
                print(f"saved {paths[key]}")
    return 0
