"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from spectraxgk.io import (
    load_case_from_toml,
    load_krylov_from_toml,
    load_linear_terms_from_toml,
    load_runtime_from_toml,
    load_toml,
    resolve_runtime_path,
)
from spectraxgk._version import __version__
from spectraxgk.plotting import (
    growth_fit_figure,
    linear_runtime_panel_figure,
    plot_saved_output,
    scan_comparison_figure,
    set_plot_style,
)
from spectraxgk.runtime_artifacts import (
    run_runtime_nonlinear_with_artifacts,
    write_quasilinear_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_linear_scan_artifacts,
)
from spectraxgk.workflows.demo import (
    DefaultDemoDeps,
    default_example_config_path as _default_example_config_path,
    run_default_linear_demo,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_scan
from spectraxgk.workflows.cases import (
    RuntimeCommandDeps,
    apply_quasilinear_overrides as _apply_quasilinear_overrides_impl,
    apply_runtime_path_overrides as _apply_runtime_path_overrides_impl,
    print_linear_run_header as _print_linear_run_header,
    run_runtime_linear_command,
    run_runtime_nonlinear_command,
    runtime_output_path as _runtime_output_path_impl,
    scan_runtime_linear_command,
    should_show_progress as _should_show_progress,
)


_RUNTIME_TOP_LEVEL_KEYS = {
    "species",
    "physics",
    "collisions",
    "normalization",
    "expert",
    "output",
    "quasilinear",
}
_LEGACY_CASE_TOP_LEVEL_KEYS = {
    "case",
    "model",
    "reference_alignment",
}


def _runtime_output_path(args: argparse.Namespace, cfg) -> str | None:
    return _runtime_output_path_impl(args, cfg)


def _is_runtime_toml(data: dict) -> bool:
    if any(key in data for key in _LEGACY_CASE_TOP_LEVEL_KEYS):
        return False
    if any(key in data for key in _RUNTIME_TOP_LEVEL_KEYS):
        return True
    return True


def _status_printer(prefix: str):
    def _emit(message: str) -> None:
        print(f"{prefix}: {message}", flush=True)

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


def _cmd_default_demo() -> int:
    deps = DefaultDemoDeps(
        load_case_from_toml=load_case_from_toml,
        run_cyclone_linear=run_cyclone_linear,
        cyclone_base_case=CycloneBaseCase,
        build_spectral_grid=build_spectral_grid,
        extract_mode_time_series=extract_mode_time_series,
        extract_eigenfunction=extract_eigenfunction,
        normalize_eigenfunction=normalize_eigenfunction,
        linear_runtime_panel_figure=linear_runtime_panel_figure,
        write_runtime_linear_artifacts=write_runtime_linear_artifacts,
    )
    return run_default_linear_demo(
        deps=deps,
        example_path=_default_example_config_path(),
    )

def _cmd_plot_saved_output(argv: list[str]) -> int:
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


def _add_quasilinear_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--quasilinear",
        action="store_true",
        help="Compute quasilinear transport diagnostics",
    )
    cmd.add_argument("--ql-mode", type=str, default=None, help="weights or saturated")
    cmd.add_argument(
        "--ql-saturation-rule",
        type=str,
        default=None,
        help="none, mixing_length, or lapillonne_2011",
    )
    cmd.add_argument(
        "--ql-csat",
        type=float,
        default=None,
        help="Saturation-rule calibration constant",
    )
    cmd.add_argument(
        "--ql-normalization",
        type=str,
        default=None,
        help="phi_rms, phi_midplane, or field_energy",
    )
    cmd.add_argument(
        "--ql-output",
        type=str,
        default=None,
        help="Optional quasilinear output path",
    )


# Path-valued CLI flags (--vmec-file, --geometry-file, --init-file) follow
# shell conventions: relative paths resolve against cwd, ~ expands to $HOME,
# and $VAR is expanded from the environment. See _apply_runtime_path_overrides.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name if sys.argv else "spectraxgk"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
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
    diag_group.add_argument(
        "--diagnostics", action="store_true", help="Enable diagnostics output"
    )
    diag_group.add_argument(
        "--no-diagnostics", action="store_true", help="Disable diagnostics output"
    )
    generic_run.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        help="grid or spectral (nonlinear only)",
    )
    generic_run.add_argument(
        "--init-file",
        type=str,
        default=None,
        help="Optional init file for nonlinear runs",
    )
    generic_run.add_argument(
        "--vmec-file", type=str, default=None, help="Override [geometry].vmec_file"
    )
    generic_run.add_argument(
        "--geometry-file",
        type=str,
        default=None,
        help="Override [geometry].geometry_file",
    )
    generic_run.add_argument(
        "--out", type=str, default=None, help="Optional output path/prefix"
    )
    _add_quasilinear_flags(generic_run)
    generic_progress = generic_run.add_mutually_exclusive_group()
    generic_progress.add_argument(
        "--progress", action="store_true", help="Enable progress output"
    )
    generic_progress.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    generic_run.set_defaults(func=_cmd_run)

    info = sub.add_parser("cyclone-info", help="Print Cyclone base case defaults")
    info.set_defaults(func=_cmd_cyclone_info)

    kperp = sub.add_parser("cyclone-kperp", help="Compute k_perp^2(theta)")
    kperp.add_argument("--kx0", type=float, default=0.0)
    kperp.add_argument("--ky", type=float, default=0.3)
    kperp.set_defaults(func=_cmd_cyclone_kperp)

    run_linear = sub.add_parser(
        "run-linear", help="Run a single linear case from a TOML config"
    )
    run_linear.add_argument("--config", required=True, help="Path to TOML config")
    run_linear.add_argument(
        "--case", default=None, help="Case name (cyclone, etg, ...)"
    )
    run_linear.add_argument("--ky", type=float, default=None, help="Single ky value")
    run_linear.add_argument("--Nl", type=int, default=None)
    run_linear.add_argument("--Nm", type=int, default=None)
    run_linear.add_argument(
        "--solver", type=str, default=None, help="auto, time, or krylov"
    )
    run_linear.add_argument(
        "--method", type=str, default=None, help="time integrator method"
    )
    run_linear.add_argument("--dt", type=float, default=None)
    run_linear.add_argument("--steps", type=int, default=None)
    run_linear.add_argument("--sample-stride", type=int, default=None)
    run_linear.add_argument(
        "--fit-signal", type=str, default=None, help="auto, phi, or density"
    )
    run_linear.add_argument(
        "--plot", action="store_true", help="Save fit/eigenfunction plots"
    )
    run_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    run_linear_progress = run_linear.add_mutually_exclusive_group()
    run_linear_progress.add_argument(
        "--progress", action="store_true", help="Enable progress output"
    )
    run_linear_progress.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    run_linear.set_defaults(func=_cmd_run_linear)

    scan_linear = sub.add_parser("scan-linear", help="Run a ky scan from a TOML config")
    scan_linear.add_argument("--config", required=True, help="Path to TOML config")
    scan_linear.add_argument(
        "--case", default=None, help="Case name (cyclone, etg, ...)"
    )
    scan_linear.add_argument(
        "--ky-values", type=str, default=None, help="Comma-separated ky list"
    )
    scan_linear.add_argument("--Nl", type=int, default=None)
    scan_linear.add_argument("--Nm", type=int, default=None)
    scan_linear.add_argument(
        "--solver", type=str, default=None, help="auto, time, or krylov"
    )
    scan_linear.add_argument(
        "--method", type=str, default=None, help="time integrator method"
    )
    scan_linear.add_argument("--dt", type=float, default=None)
    scan_linear.add_argument("--steps", type=int, default=None)
    scan_linear.add_argument(
        "--fit-signal", type=str, default=None, help="auto, phi, or density"
    )
    scan_linear.add_argument(
        "--plot", action="store_true", help="Save comparison plot if reference exists"
    )
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
    run_runtime.add_argument(
        "--solver", type=str, default=None, help="auto, time, or krylov"
    )
    run_runtime.add_argument(
        "--method", type=str, default=None, help="time integrator method"
    )
    run_runtime.add_argument("--dt", type=float, default=None)
    run_runtime.add_argument("--steps", type=int, default=None)
    run_runtime.add_argument("--sample-stride", type=int, default=None)
    run_runtime.add_argument(
        "--fit-signal", type=str, default=None, help="auto, phi, or density"
    )
    run_runtime.add_argument(
        "--vmec-file", type=str, default=None, help="Override [geometry].vmec_file"
    )
    run_runtime.add_argument(
        "--geometry-file",
        type=str,
        default=None,
        help="Override [geometry].geometry_file",
    )
    run_runtime.add_argument(
        "--out", type=str, default=None, help="Optional output path/prefix"
    )
    _add_quasilinear_flags(run_runtime)
    run_runtime_progress = run_runtime.add_mutually_exclusive_group()
    run_runtime_progress.add_argument(
        "--progress", action="store_true", help="Enable progress output"
    )
    run_runtime_progress.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    run_runtime.set_defaults(func=_cmd_run_runtime_linear)

    scan_runtime = sub.add_parser(
        "scan-runtime-linear",
        help="Run a ky scan from unified runtime TOML config",
    )
    scan_runtime.add_argument("--config", required=True, help="Path to TOML config")
    scan_runtime.add_argument(
        "--ky-values", type=str, default=None, help="Comma-separated ky list"
    )
    scan_runtime.add_argument("--Nl", type=int, default=None)
    scan_runtime.add_argument("--Nm", type=int, default=None)
    scan_runtime.add_argument(
        "--solver", type=str, default=None, help="auto, time, or krylov"
    )
    scan_runtime.add_argument(
        "--method", type=str, default=None, help="time integrator method"
    )
    scan_runtime.add_argument("--dt", type=float, default=None)
    scan_runtime.add_argument("--steps", type=int, default=None)
    scan_runtime.add_argument("--sample-stride", type=int, default=None)
    scan_runtime.add_argument(
        "--batch-ky", action="store_true", help="Integrate all ky in one batch"
    )
    scan_runtime.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent ky workers for the serial scan path, including quasilinear spectra.",
    )
    scan_runtime.add_argument(
        "--parallel-executor",
        choices=("thread", "process"),
        default="thread",
        help="Executor for independent ky workers.",
    )
    scan_runtime.add_argument(
        "--fit-signal", type=str, default=None, help="auto, phi, or density"
    )
    scan_runtime.add_argument(
        "--out", type=str, default=None, help="Optional scan output path/prefix"
    )
    _add_quasilinear_flags(scan_runtime)
    scan_runtime_progress = scan_runtime.add_mutually_exclusive_group()
    scan_runtime_progress.add_argument(
        "--progress", action="store_true", help="Enable progress output"
    )
    scan_runtime_progress.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    scan_runtime.set_defaults(func=_cmd_scan_runtime_linear)

    run_runtime_nl = sub.add_parser(
        "run-runtime-nonlinear",
        help="Run one nonlinear point from unified runtime TOML config",
    )
    run_runtime_nl.add_argument("--config", required=True, help="Path to TOML config")
    run_runtime_nl.add_argument(
        "--ky", type=float, default=None, help="Single ky value"
    )
    run_runtime_nl.add_argument("--Nl", type=int, default=None)
    run_runtime_nl.add_argument("--Nm", type=int, default=None)
    run_runtime_nl.add_argument("--dt", type=float, default=None)
    run_runtime_nl.add_argument("--steps", type=int, default=None)
    run_runtime_nl.add_argument("--method", type=str, default=None)
    run_runtime_nl.add_argument("--sample-stride", type=int, default=None)
    run_runtime_nl.add_argument("--diagnostics-stride", type=int, default=None)
    diag_group = run_runtime_nl.add_mutually_exclusive_group()
    diag_group.add_argument(
        "--diagnostics", action="store_true", help="Enable diagnostics output"
    )
    diag_group.add_argument(
        "--no-diagnostics", action="store_true", help="Disable diagnostics output"
    )
    run_runtime_nl.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        help="grid or spectral (nonlinear Laguerre handling)",
    )
    run_runtime_nl.add_argument(
        "--init-file",
        type=str,
        default=None,
        help="Optional restart/init-state file containing a compatible distribution state",
    )
    run_runtime_nl.add_argument(
        "--vmec-file", type=str, default=None, help="Override [geometry].vmec_file"
    )
    run_runtime_nl.add_argument(
        "--geometry-file",
        type=str,
        default=None,
        help="Override [geometry].geometry_file",
    )
    run_runtime_nl.add_argument(
        "--out", type=str, default=None, help="Optional output path/prefix"
    )
    run_runtime_nl_progress = run_runtime_nl.add_mutually_exclusive_group()
    run_runtime_nl_progress.add_argument(
        "--progress", action="store_true", help="Enable progress output"
    )
    run_runtime_nl_progress.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )
    run_runtime_nl.set_defaults(func=_cmd_run_runtime_nonlinear)

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        return _cmd_default_demo()
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        return _cmd_plot_saved_output(sys.argv[1:])

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
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in known_cmds
    ):
        if Path(sys.argv[1]).exists():
            data = load_toml(sys.argv[1])
            command = "run" if _is_runtime_toml(data) else "run-linear"
            parser = build_parser()
            args = parser.parse_args([command, "--config", sys.argv[1], *sys.argv[2:]])
            return args.func(args)

    parser = build_parser()
    args = parser.parse_args()
    if args.cmd is None:
        return _cmd_default_demo()
    return args.func(args)


def _resolve_case(case_name: str):
    name = case_name.lower()
    if name == "cyclone":
        return CycloneBaseCase, run_cyclone_linear
    if name == "etg":
        return ETGBaseCase, run_etg_linear
    raise ValueError(
        f"Unsupported case '{case_name}' in executable dispatcher (supported: cyclone, etg)"
    )


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
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    method = (
        args.method
        if args.method is not None
        else run_cfg.get("method", cfg.time.method)
    )
    dt = args.dt if args.dt is not None else run_cfg.get("dt", cfg.time.dt)
    steps = (
        args.steps
        if args.steps is not None
        else run_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))
    )
    sample_stride = (
        args.sample_stride
        if args.sample_stride is not None
        else run_cfg.get("sample_stride", getattr(cfg.time, "sample_stride", None))
    )
    show_progress = _should_show_progress(
        args, bool(getattr(cfg.time, "progress_bar", False))
    )

    terms = load_linear_terms_from_toml(data)
    krylov_cfg = load_krylov_from_toml(data)

    _print_linear_run_header(
        label=f"named linear {case_name} run",
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
        extra="detected named case TOML; using run-linear path",
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
        sample_stride=None if sample_stride is None else int(sample_stride),
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
            signal = extract_mode_time_series(
                result.phi_t, result.selection, method="project"
            )
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
        ky_values = np.asarray(
            [float(x) for x in args.ky_values.split(",") if x.strip() != ""]
        )
    else:
        ky_values = _load_scan_ky(data)
    if ky_values.size == 0:
        raise ValueError("No ky values provided. Use --ky-values or [scan].ky in TOML.")

    Nl = args.Nl if args.Nl is not None else scan_cfg.get("Nl", 24)
    Nm = args.Nm if args.Nm is not None else scan_cfg.get("Nm", 12)
    solver = args.solver if args.solver is not None else scan_cfg.get("solver", "auto")
    fit_signal = (
        args.fit_signal
        if args.fit_signal is not None
        else fit_cfg.pop("fit_signal", "auto")
    )
    auto_window = bool(fit_cfg.pop("auto_window", True))
    method = (
        args.method
        if args.method is not None
        else scan_cfg.get("method", cfg.time.method)
    )
    dt = args.dt if args.dt is not None else scan_cfg.get("dt", cfg.time.dt)
    steps = (
        args.steps
        if args.steps is not None
        else scan_cfg.get("steps", int(round(cfg.time.t_max / cfg.time.dt)))
    )

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
        auto_window=auto_window,
        window_kw=fit_cfg,
        run_kwargs={"terms": terms, "fit_signal": str(fit_signal)}
        if terms is not None
        else {"fit_signal": str(fit_signal)},
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


def _runtime_command_deps() -> RuntimeCommandDeps:
    return RuntimeCommandDeps(
        load_runtime_from_toml=load_runtime_from_toml,
        run_runtime_linear=run_runtime_linear,
        run_runtime_scan=run_runtime_scan,
        run_runtime_nonlinear_with_artifacts=run_runtime_nonlinear_with_artifacts,
        write_runtime_linear_artifacts=write_runtime_linear_artifacts,
        write_runtime_linear_scan_artifacts=write_runtime_linear_scan_artifacts,
        write_quasilinear_artifacts=write_quasilinear_artifacts,
        resolve_runtime_path=resolve_runtime_path,
    )


def _apply_runtime_path_overrides(cfg, args: argparse.Namespace):
    return _apply_runtime_path_overrides_impl(
        cfg, args, resolve_runtime_path=resolve_runtime_path
    )


def _apply_quasilinear_overrides(cfg, args: argparse.Namespace):
    return _apply_quasilinear_overrides_impl(cfg, args)


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    return run_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    return scan_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_run_runtime_nonlinear(args: argparse.Namespace) -> int:
    return run_runtime_nonlinear_command(args, deps=_runtime_command_deps())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
