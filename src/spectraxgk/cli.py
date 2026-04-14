"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import extract_eigenfunction, extract_mode_time_series
from spectraxgk.benchmarking import normalize_eigenfunction, run_linear_scan
from spectraxgk.benchmarks import (
    ETGBaseCase,
    CycloneBaseCase,
    load_cyclone_reference,
    load_etg_reference,
    run_cyclone_linear,
    run_etg_linear,
)
from spectraxgk.geometry import SAlphaGeometry
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_case_from_toml, load_krylov_from_toml, load_linear_terms_from_toml, load_runtime_from_toml, load_toml
from spectraxgk.plotting import growth_fit_figure, scan_comparison_figure, set_plot_style
from spectraxgk.runtime_artifacts import (
    run_runtime_nonlinear_with_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_nonlinear_artifacts,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_scan, run_runtime_nonlinear


def _runtime_output_path(args: argparse.Namespace, cfg) -> str | None:
    if getattr(args, "out", None) is not None:
        return str(args.out)
    return cfg.output.path


_RUNTIME_TOP_LEVEL_KEYS = {
    "species",
    "physics",
    "collisions",
    "normalization",
    "expert",
    "output",
}
_LEGACY_CASE_TOP_LEVEL_KEYS = {"case", "model", "gx_reference"}


def _is_runtime_toml(data: dict) -> bool:
    if any(key in data for key in _LEGACY_CASE_TOP_LEVEL_KEYS):
        return False
    if any(key in data for key in _RUNTIME_TOP_LEVEL_KEYS):
        return True
    return True


def _should_show_progress(args: argparse.Namespace, configured: bool) -> bool:
    if getattr(args, "progress", False):
        return True
    if getattr(args, "no_progress", False):
        return False
    return bool(configured or sys.stdout.isatty())


def _print_linear_run_header(
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


def _status_printer(prefix: str):
    def _emit(message: str) -> None:
        print(f"{prefix}: {message}")

    return _emit


def _cmd_cyclone_info(_: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    print("Cyclone base case")
    print(cfg.to_dict())
    return 0


def _cmd_cyclone_kperp(args: argparse.Namespace) -> int:
    cfg = CycloneBaseCase()
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    theta = grid.z
    kx0 = jnp.array(args.kx0)
    ky = jnp.array(args.ky)
    kperp2 = geom.k_perp2(kx0, ky, theta)
    print(f"k_perp^2(theta) min={kperp2.min():.6g} max={kperp2.max():.6g}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg, _raw = load_runtime_from_toml(args.config)
    except Exception as exc:
        print(f"Error loading {args.config}: {exc}")
        return 1

    if cfg.physics.nonlinear:
        return _cmd_run_runtime_nonlinear(args)
    return _cmd_run_runtime_linear(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectrax-gk")
    parser.add_argument(
        "config_pos",
        nargs="?",
        help="Path to TOML config (shorthand for 'run --config ...')",
    )
    sub = parser.add_subparsers(dest="cmd")

    generic_run = sub.add_parser(
        "run",
        help="Run a simulation from a TOML config (auto-detect linear/nonlinear)",
    )
    generic_run.add_argument("--config", required=True, help="Path to TOML config")
    generic_run.add_argument("--ky", type=float, default=None)
    generic_run.add_argument("--Nl", type=int, default=None)
    generic_run.add_argument("--Nm", type=int, default=None)
    generic_run.add_argument("--steps", type=int, default=None)
    generic_run.add_argument("--dt", type=float, default=None)
    generic_run.add_argument("--solver", type=str, default=None)
    generic_run.add_argument("--method", type=str, default=None)
    generic_run.add_argument("--fit-signal", type=str, default=None)
    generic_run.add_argument("--sample-stride", type=int, default=None)
    generic_run.add_argument("--diagnostics-stride", type=int, default=None)
    diag_group = generic_run.add_mutually_exclusive_group()
    diag_group.add_argument("--diagnostics", action="store_true", help="Enable diagnostics output")
    diag_group.add_argument("--no-diagnostics", action="store_true", help="Disable diagnostics output")
    generic_run.add_argument("--laguerre-mode", type=str, default=None, help="grid or spectral (nonlinear only)")
    generic_run.add_argument("--init-file", type=str, default=None, help="Optional init file for nonlinear runs")
    generic_run.add_argument("--out", type=str, default=None, help="Optional artifact path/prefix")
    generic_progress = generic_run.add_mutually_exclusive_group()
    generic_progress.add_argument("--progress", action="store_true", help="Enable progress output")
    generic_progress.add_argument("--no-progress", action="store_true", help="Disable progress output")
    generic_run.set_defaults(func=_cmd_run)

    info = sub.add_parser("cyclone-info", help="Print Cyclone base case defaults")
    info.set_defaults(func=_cmd_cyclone_info)

    kperp = sub.add_parser("cyclone-kperp", help="Compute k_perp^2(theta)")
    kperp.add_argument("--kx0", type=float, default=0.0)
    kperp.add_argument("--ky", type=float, default=0.3)
    kperp.set_defaults(func=_cmd_cyclone_kperp)

    run_linear = sub.add_parser("run-linear", help="Run a single linear case from a TOML config")
    run_linear.add_argument("--config", required=True, help="Path to TOML config")
    run_linear.add_argument("--case", default=None, help="Case name (cyclone, etg, ...)")
    run_linear.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_linear.add_argument("--Nl", type=int, default=None)
    run_linear.add_argument("--Nm", type=int, default=None)
    run_linear.add_argument("--solver", type=str, default=None, help="auto, time, or krylov")
    run_linear.add_argument("--method", type=str, default=None, help="time integrator method")
    run_linear.add_argument("--dt", type=float, default=None)
    run_linear.add_argument("--steps", type=int, default=None)
    run_linear.add_argument("--fit-signal", type=str, default=None, help="auto, phi, or density")
    run_linear.add_argument("--plot", action="store_true", help="Save fit/eigenfunction plots")
    run_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    run_linear_progress = run_linear.add_mutually_exclusive_group()
    run_linear_progress.add_argument("--progress", action="store_true", help="Enable progress output")
    run_linear_progress.add_argument("--no-progress", action="store_true", help="Disable progress output")
    run_linear.set_defaults(func=_cmd_run_linear)

    scan_linear = sub.add_parser("scan-linear", help="Run a ky scan from a TOML config")
    scan_linear.add_argument("--config", required=True, help="Path to TOML config")
    scan_linear.add_argument("--case", default=None, help="Case name (cyclone, etg, ...)")
    scan_linear.add_argument("--ky-values", type=str, default=None, help="Comma-separated ky list")
    scan_linear.add_argument("--Nl", type=int, default=None)
    scan_linear.add_argument("--Nm", type=int, default=None)
    scan_linear.add_argument("--solver", type=str, default=None, help="auto, time, or krylov")
    scan_linear.add_argument("--method", type=str, default=None, help="time integrator method")
    scan_linear.add_argument("--dt", type=float, default=None)
    scan_linear.add_argument("--steps", type=int, default=None)
    scan_linear.add_argument("--fit-signal", type=str, default=None, help="auto, phi, or density")
    scan_linear.add_argument("--plot", action="store_true", help="Save comparison plot if reference exists")
    scan_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    scan_linear.set_defaults(func=_cmd_scan_linear)

    run_runtime = sub.add_parser(
        "run-runtime-linear",
        help="Run one linear point from unified runtime TOML config",
    )
    run_runtime.add_argument("--config", required=True, help="Path to TOML config")
    run_runtime.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_runtime.add_argument("--Nl", type=int, default=None)
    run_runtime.add_argument("--Nm", type=int, default=None)
    run_runtime.add_argument("--solver", type=str, default=None, help="auto, time, or krylov")
    run_runtime.add_argument("--method", type=str, default=None, help="time integrator method")
    run_runtime.add_argument("--dt", type=float, default=None)
    run_runtime.add_argument("--steps", type=int, default=None)
    run_runtime.add_argument("--sample-stride", type=int, default=None)
    run_runtime.add_argument("--fit-signal", type=str, default=None, help="auto, phi, or density")
    run_runtime.add_argument("--out", type=str, default=None, help="Optional artifact path/prefix")
    run_runtime_progress = run_runtime.add_mutually_exclusive_group()
    run_runtime_progress.add_argument("--progress", action="store_true", help="Enable progress output")
    run_runtime_progress.add_argument("--no-progress", action="store_true", help="Disable progress output")
    run_runtime.set_defaults(func=_cmd_run_runtime_linear)

    scan_runtime = sub.add_parser(
        "scan-runtime-linear",
        help="Run a ky scan from unified runtime TOML config",
    )
    scan_runtime.add_argument("--config", required=True, help="Path to TOML config")
    scan_runtime.add_argument("--ky-values", type=str, default=None, help="Comma-separated ky list")
    scan_runtime.add_argument("--Nl", type=int, default=None)
    scan_runtime.add_argument("--Nm", type=int, default=None)
    scan_runtime.add_argument("--solver", type=str, default=None, help="auto, time, or krylov")
    scan_runtime.add_argument("--method", type=str, default=None, help="time integrator method")
    scan_runtime.add_argument("--dt", type=float, default=None)
    scan_runtime.add_argument("--steps", type=int, default=None)
    scan_runtime.add_argument("--sample-stride", type=int, default=None)
    scan_runtime.add_argument("--batch-ky", action="store_true", help="Integrate all ky in one batch")
    scan_runtime.add_argument("--fit-signal", type=str, default=None, help="auto, phi, or density")
    scan_runtime_progress = scan_runtime.add_mutually_exclusive_group()
    scan_runtime_progress.add_argument("--progress", action="store_true", help="Enable progress output")
    scan_runtime_progress.add_argument("--no-progress", action="store_true", help="Disable progress output")
    scan_runtime.set_defaults(func=_cmd_scan_runtime_linear)

    run_runtime_nl = sub.add_parser(
        "run-runtime-nonlinear",
        help="Run one nonlinear point from unified runtime TOML config",
    )
    run_runtime_nl.add_argument("--config", required=True, help="Path to TOML config")
    run_runtime_nl.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_runtime_nl.add_argument("--Nl", type=int, default=None)
    run_runtime_nl.add_argument("--Nm", type=int, default=None)
    run_runtime_nl.add_argument("--dt", type=float, default=None)
    run_runtime_nl.add_argument("--steps", type=int, default=None)
    run_runtime_nl.add_argument("--method", type=str, default=None)
    run_runtime_nl.add_argument("--sample-stride", type=int, default=None)
    run_runtime_nl.add_argument("--diagnostics-stride", type=int, default=None)
    diag_group = run_runtime_nl.add_mutually_exclusive_group()
    diag_group.add_argument("--diagnostics", action="store_true", help="Enable diagnostics output")
    diag_group.add_argument("--no-diagnostics", action="store_true", help="Disable diagnostics output")
    run_runtime_nl.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        help="grid or spectral (nonlinear Laguerre handling)",
    )
    run_runtime_nl.add_argument("--init-file", type=str, default=None, help="Optional init file (GX g_state)")
    run_runtime_nl.add_argument("--out", type=str, default=None, help="Optional artifact path/prefix")
    run_runtime_nl_progress = run_runtime_nl.add_mutually_exclusive_group()
    run_runtime_nl_progress.add_argument("--progress", action="store_true", help="Enable progress output")
    run_runtime_nl_progress.add_argument("--no-progress", action="store_true", help="Disable progress output")
    run_runtime_nl.set_defaults(func=_cmd_run_runtime_nonlinear)

    return parser


def main() -> int:
    known_cmds = {
        "run",
        "cyclone-info",
        "cyclone-kperp",
        "run-linear",
        "scan-linear",
        "run-runtime-linear",
        "scan-runtime-linear",
        "run-runtime-nonlinear",
    }
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] not in known_cmds:
        if Path(sys.argv[1]).exists():
            data = load_toml(sys.argv[1])
            command = "run" if _is_runtime_toml(data) else "run-linear"
            parser = build_parser()
            args = parser.parse_args([command, "--config", sys.argv[1], *sys.argv[2:]])
            return args.func(args)

    parser = build_parser()
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return 0
    return args.func(args)


def _resolve_case(case_name: str):
    name = case_name.lower()
    if name == "cyclone":
        return CycloneBaseCase, run_cyclone_linear
    if name == "etg":
        return ETGBaseCase, run_etg_linear
    raise ValueError(f"Unsupported case '{case_name}' in executable dispatcher (supported: cyclone, etg)")


def _load_scan_ky(data: dict) -> np.ndarray:
    scan = data.get("scan", {})
    ky_vals = scan.get("ky")
    if ky_vals is None:
        return np.asarray([])
    return np.asarray(ky_vals, dtype=float)


def _cmd_run_linear(args: argparse.Namespace) -> int:
    case_name, cfg, data = load_case_from_toml(args.config, args.case)
    case_cls, run_fn = _resolve_case(case_name)
    _ = case_cls  # keep type checkers happy
    run_cfg = data.get("run", {})
    fit_cfg = dict(data.get("fit", {}))

    ky = args.ky if args.ky is not None else run_cfg.get("ky", 0.3)
    Nl = args.Nl if args.Nl is not None else run_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else run_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else run_cfg.get("solver", "auto")
    fit_signal = args.fit_signal if args.fit_signal is not None else fit_cfg.pop("fit_signal", "auto")
    method = args.method if args.method is not None else run_cfg.get("method", cfg.time.method)
    dt = args.dt if args.dt is not None else run_cfg.get("dt", cfg.time.dt)
    steps = args.steps if args.steps is not None else run_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))
    show_progress = _should_show_progress(args, bool(cfg.time.progress_bar))

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)

    _print_linear_run_header(
        label=f"legacy linear {case_name} run",
        config_path=str(args.config),
        ky=float(ky),
        Nl=int(Nl),
        Nm=int(Nm),
        solver=str(solver),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        grid_shape=(int(cfg.grid.Nx), int(cfg.grid.Ny), int(cfg.grid.Nz)),
        show_progress=show_progress,
        extra="detected legacy case TOML; using run-linear path",
    )

    result = run_fn(
        ky_target=float(ky),
        cfg=cfg,
        Nl=int(Nl),
        Nm=int(Nm),
        solver=str(solver),
        method=str(method),
        dt=float(dt),
        steps=int(steps),
        krylov_cfg=krylov_cfg,
        terms=terms,
        fit_signal=str(fit_signal),
        show_progress=show_progress,
        status_callback=_status_printer(case_name),
        **fit_cfg,
    )
    print(f"ky={result.ky:.4f} gamma={result.gamma:.6f} omega={result.omega:.6f}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        grid = build_spectral_grid(cfg.grid)
        if result.t.size > 1:
            signal = extract_mode_time_series(result.phi_t, result.selection, method="project")
            fig, _ax = growth_fit_figure(result.t, signal)
            fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_fit.png")
        z_np = np.asarray(grid.z)
        eigen = extract_eigenfunction(result.phi_t, result.t, result.selection, z=z_np)
        eigen = normalize_eigenfunction(eigen, z_np)
        set_plot_style()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(grid.z, eigen.real, label="Re")
        ax.plot(grid.z, eigen.imag, linestyle="--", label="Im")
        ax.set_xlabel(r"$\\theta$")
        ax.set_ylabel(r"$\\phi$")
        ax.set_title(f"{case_name} ky={result.ky:.3f}")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"{case_name}_ky{result.ky:.3f}_eig.png")
        plt.close(fig)
    return 0


def _cmd_scan_linear(args: argparse.Namespace) -> int:
    case_name, cfg, data = load_case_from_toml(args.config, args.case)
    _case_cls, run_fn = _resolve_case(case_name)
    scan_cfg = data.get("scan", {})
    fit_cfg = dict(data.get("fit", {}))

    if args.ky_values is not None:
        ky_values = np.asarray([float(x) for x in args.ky_values.split(",") if x.strip() != ""])
    else:
        ky_values = _load_scan_ky(data)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else scan_cfg.get("solver", "auto")
    fit_signal = args.fit_signal if args.fit_signal is not None else fit_cfg.pop("fit_signal", "auto")
    method = args.method if args.method is not None else scan_cfg.get("method", cfg.time.method)
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", cfg.time.dt)
    steps = args.steps if args.steps is not None else scan_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)

    scan = run_linear_scan(
        ky_values=ky_values,
        run_linear_fn=run_fn,
        cfg=cfg,
        Nl=int(Nl),
        Nm=int(Nm),
        dt=float(dt),
        steps=int(steps),
        method=str(method),
        solver=str(solver),
        krylov_cfg=krylov_cfg,
        window_kw=fit_cfg,
        run_kwargs={"terms": terms} if terms is not None else None,
    )

    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")

    if args.plot:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        ref = None
        if case_name == "cyclone":
            ref = load_cyclone_reference()
        elif case_name == "etg":
            ref = load_etg_reference()
        if ref is not None:
            fig, _ax = scan_comparison_figure(
                scan.ky,
                scan.gamma,
                scan.omega,
                r"$k_y \rho_i$",
                f"{case_name.upper()} scan",
                x_ref=ref.ky,
                gamma_ref=ref.gamma,
                omega_ref=ref.omega,
                log_x=True,
            )
            fig.savefig(outdir / f"{case_name}_scan_comparison.png")
        else:
            print("No reference available for this case; skipping comparison plot.")
    return 0


_RUNTIME_FIT_KEYS = {
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


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    cfg, data = load_runtime_from_toml(args.config)
    run_cfg = data.get("run", {})
    fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in _RUNTIME_FIT_KEYS}

    ky = float(args.ky if args.ky is not None else run_cfg.get("ky", 0.3))
    Nl = int(args.Nl if args.Nl is not None else run_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else run_cfg.get("Nm", 12))
    solver = str(args.solver if args.solver is not None else run_cfg.get("solver", "auto"))
    fit_signal = str(args.fit_signal if args.fit_signal is not None else run_cfg.get("fit_signal", "auto"))
    method = args.method if args.method is not None else run_cfg.get("method", None)
    dt = args.dt if args.dt is not None else run_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else run_cfg.get("steps", None)
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    dt_use = float(dt if dt is not None else cfg.time.dt)
    steps_use = int(steps) if steps is not None else int(round(float(cfg.time.t_max) / dt_use))
    method_use = str(method if method is not None else cfg.time.method)
    show_progress = _should_show_progress(args, bool(cfg.time.progress_bar))

    _print_linear_run_header(
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

    res = run_runtime_linear(
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
    out_path = _runtime_output_path(args, cfg)
    if out_path is not None:
        paths = write_runtime_linear_artifacts(out_path, res)
        print(f"saved {paths['summary']}")
        if "timeseries" in paths:
            print(f"saved {paths['timeseries']}")
        if "state" in paths:
            print(f"saved {paths['state']}")
    return 0


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    cfg, data = load_runtime_from_toml(args.config)
    scan_cfg = data.get("scan", {})
    fit_cfg = {k: v for k, v in data.get("fit", {}).items() if k in _RUNTIME_FIT_KEYS}

    if args.ky_values is not None:
        ky_values = np.asarray([float(x) for x in args.ky_values.split(",") if x.strip()], dtype=float)
    else:
        ky_raw = scan_cfg.get("ky", [])
        ky_values = np.asarray(ky_raw, dtype=float)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = int(args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24))
    Nm = int(args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12))
    solver = str(args.solver if args.solver is not None else scan_cfg.get("solver", "auto"))
    fit_signal = str(args.fit_signal if args.fit_signal is not None else scan_cfg.get("fit_signal", "auto"))
    method = args.method if args.method is not None else scan_cfg.get("method", None)
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", None)
    steps = args.steps if args.steps is not None else scan_cfg.get("steps", None)
    sample_stride = (
        int(args.sample_stride)
        if args.sample_stride is not None
        else scan_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    batch_ky = bool(args.batch_ky)
    show_progress = (
        True
        if getattr(args, "progress", False)
        else False
        if getattr(args, "no_progress", False)
        else bool(cfg.time.progress_bar)
    )

    ky_sequence = cast(list[float], ky_values.tolist())
    scan = run_runtime_scan(
        cfg,
        ky_sequence,
        Nl=Nl,
        Nm=Nm,
        solver=solver,
        method=method,
        dt=dt,
        steps=steps,
        sample_stride=sample_stride,
        batch_ky=batch_ky,
        fit_signal=fit_signal,
        show_progress=show_progress,
        **fit_cfg,
    )
    for ky, g, w in zip(scan.ky, scan.gamma, scan.omega):
        print(f"ky={ky:.4f} gamma={g:.6f} omega={w:.6f}")
    return 0


def _cmd_run_runtime_nonlinear(args: argparse.Namespace) -> int:
    cfg, data = load_runtime_from_toml(args.config)
    run_cfg = data.get("run", {})

    if args.init_file is not None:
        cfg = replace(cfg, init=replace(cfg.init, init_file=str(args.init_file)))

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
    method = str(args.method if args.method is not None else run_cfg.get("method", cfg.time.method))
    sample_stride = int(
        args.sample_stride
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", cfg.time.sample_stride)
    )
    diagnostics_stride = (
        None
        if args.diagnostics_stride is None
        else int(args.diagnostics_stride)
    )
    if args.no_diagnostics:
        diagnostics = False
    elif args.diagnostics:
        diagnostics = True
    else:
        diagnostics = run_cfg.get("diagnostics", cfg.time.diagnostics)
    laguerre_mode = args.laguerre_mode if args.laguerre_mode is not None else run_cfg.get(
        "laguerre_mode"
    )
    show_progress = _should_show_progress(args, bool(cfg.time.progress_bar))

    print("starting runtime nonlinear run")
    print(
        f"config={args.config} ky={ky:.4f} Nl={Nl} Nm={Nm} method={method} dt={dt:.6g} "
        f"steps={'auto' if steps is None else steps}"
    )
    print(
        f"grid=Nx{int(cfg.grid.Nx)} Ny{int(cfg.grid.Ny)} Nz{int(cfg.grid.Nz)} "
        f"diagnostics={'on' if diagnostics else 'off'} progress={'on' if show_progress else 'off'}"
    )

    out_path = _runtime_output_path(args, cfg)

    result, paths = run_runtime_nonlinear_with_artifacts(
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
