"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from spectraxgk.analysis import extract_eigenfunction, extract_mode_time_series
from spectraxgk.validation.benchmarks.harness import normalize_eigenfunction, run_linear_scan
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
from spectraxgk.workflows.runtime.toml import (
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
from spectraxgk.workflows.runtime.artifacts import (
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
from spectraxgk.workflows.named_cases import (
    NamedLinearCommandDeps,
    load_scan_ky as _load_scan_ky_impl,
    run_named_linear_command,
    scan_named_linear_command,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_scan
from spectraxgk.workflows.cases import (
    RuntimeCommandDeps,
    print_linear_run_header as _print_linear_run_header,
    run_runtime_linear_command,
    run_runtime_nonlinear_command,
    runtime_output_path as _runtime_output_path_impl,
    scan_runtime_linear_command,
    should_show_progress as _should_show_progress,
)


_RUNTIME_TOP_LEVEL_KEYS = {
    "species", "physics", "collisions", "normalization", "expert", "output",
    "quasilinear",
}
_LEGACY_CASE_TOP_LEVEL_KEYS = {"case", "model", "reference_alignment"}


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
    return run_default_linear_demo(deps=deps, example_path=_default_example_config_path())

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
        "--quasilinear", action="store_true",
        help="Compute quasilinear transport diagnostics",
    )
    for flag, kwargs in (
        ("--ql-mode", {"help": "weights or saturated"}),
        ("--ql-saturation-rule", {"help": "none, mixing_length, or lapillonne_2011"}),
        ("--ql-csat", {"type": float, "help": "Saturation-rule calibration constant"}),
        ("--ql-normalization", {"help": "phi_rms, phi_midplane, or field_energy"}),
        ("--ql-output", {"help": "Optional quasilinear output path"}),
    ):
        options: dict[str, Any] = {"type": str, "default": None, **kwargs}
        cmd.add_argument(flag, **options)


def _add_progress_flags(cmd: argparse.ArgumentParser) -> None:
    group = cmd.add_mutually_exclusive_group()
    group.add_argument("--progress", action="store_true", help="Enable progress output")
    group.add_argument(
        "--no-progress", action="store_true", help="Disable progress output"
    )


def _add_diagnostics_flags(cmd: argparse.ArgumentParser) -> None:
    group = cmd.add_mutually_exclusive_group()
    group.add_argument(
        "--diagnostics", action="store_true", help="Enable diagnostics output"
    )
    group.add_argument(
        "--no-diagnostics", action="store_true", help="Disable diagnostics output"
    )


def _add_resolution_flags(
    cmd: argparse.ArgumentParser,
    *,
    ky_help: str | None = None,
) -> None:
    cmd.add_argument("--ky", type=float, default=None, help=ky_help)
    cmd.add_argument("--Nl", type=int, default=None)
    cmd.add_argument("--Nm", type=int, default=None)


def _add_time_solver_flags(
    cmd: argparse.ArgumentParser,
    *,
    solver: bool = False,
    sample_stride: bool = False,
    fit_signal: bool = False,
) -> None:
    if solver:
        cmd.add_argument("--solver", type=str, default=None, help="auto, time, or krylov")
    cmd.add_argument("--method", type=str, default=None, help="time integrator method")
    cmd.add_argument("--dt", type=float, default=None)
    cmd.add_argument("--steps", type=int, default=None)
    if sample_stride:
        cmd.add_argument("--sample-stride", type=int, default=None)
    if fit_signal:
        cmd.add_argument("--fit-signal", type=str, default=None, help="auto, phi, or density")


def _add_runtime_paths(
    cmd: argparse.ArgumentParser,
    *,
    init_help: str | None = None,
    out_help: str = "Optional output path/prefix",
) -> None:
    if init_help is not None:
        cmd.add_argument("--init-file", type=str, default=None, help=init_help)
    cmd.add_argument(
        "--vmec-file", type=str, default=None, help="Override [geometry].vmec_file"
    )
    cmd.add_argument(
        "--geometry-file",
        type=str,
        default=None,
        help="Override [geometry].geometry_file",
    )
    cmd.add_argument("--out", type=str, default=None, help=out_help)


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
    _add_resolution_flags(generic_run)
    generic_run.add_argument("--solver", type=str, default=None)
    _add_time_solver_flags(generic_run, sample_stride=True, fit_signal=True)
    generic_run.add_argument("--diagnostics-stride", type=int, default=None)
    _add_diagnostics_flags(generic_run)
    generic_run.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        help="grid or spectral (nonlinear only)",
    )
    _add_runtime_paths(generic_run, init_help="Optional init file for nonlinear runs")
    _add_quasilinear_flags(generic_run)
    _add_progress_flags(generic_run)
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
    _add_resolution_flags(run_linear, ky_help="Single ky value")
    _add_time_solver_flags(
        run_linear, solver=True, sample_stride=True, fit_signal=True
    )
    run_linear.add_argument(
        "--plot", action="store_true", help="Save fit/eigenfunction plots"
    )
    run_linear.add_argument("--outdir", default=".", help="Output directory for plots")
    _add_progress_flags(run_linear)
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
    _add_time_solver_flags(scan_linear, solver=True, fit_signal=True)
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
    _add_resolution_flags(run_runtime, ky_help="Single ky value")
    _add_time_solver_flags(
        run_runtime, solver=True, sample_stride=True, fit_signal=True
    )
    _add_runtime_paths(run_runtime)
    _add_quasilinear_flags(run_runtime)
    _add_progress_flags(run_runtime)
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
    _add_time_solver_flags(
        scan_runtime, solver=True, sample_stride=True, fit_signal=True
    )
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
        "--out", type=str, default=None, help="Optional scan output path/prefix"
    )
    _add_quasilinear_flags(scan_runtime)
    _add_progress_flags(scan_runtime)
    scan_runtime.set_defaults(func=_cmd_scan_runtime_linear)

    run_runtime_nl = sub.add_parser(
        "run-runtime-nonlinear",
        help="Run one nonlinear point from unified runtime TOML config",
    )
    run_runtime_nl.add_argument("--config", required=True, help="Path to TOML config")
    _add_resolution_flags(run_runtime_nl, ky_help="Single ky value")
    _add_time_solver_flags(run_runtime_nl)
    run_runtime_nl.add_argument("--sample-stride", type=int, default=None)
    run_runtime_nl.add_argument("--diagnostics-stride", type=int, default=None)
    _add_diagnostics_flags(run_runtime_nl)
    run_runtime_nl.add_argument(
        "--laguerre-mode",
        type=str,
        default=None,
        help="grid or spectral (nonlinear Laguerre handling)",
    )
    _add_runtime_paths(
        run_runtime_nl,
        init_help="Optional restart/init-state file containing a compatible distribution state",
    )
    _add_progress_flags(run_runtime_nl)
    run_runtime_nl.set_defaults(func=_cmd_run_runtime_nonlinear)

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        return _cmd_default_demo()
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        return _cmd_plot_saved_output(sys.argv[1:])

    known_cmds = {
        "run", "cyclone-info", "cyclone-kperp", "run-linear", "scan-linear",
        "run-runtime-linear", "scan-runtime-linear", "run-runtime-nonlinear",
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
    return _load_scan_ky_impl(data)


def _cmd_run_linear(args: argparse.Namespace) -> int:
    return run_named_linear_command(args, deps=_named_linear_command_deps())


def _cmd_scan_linear(args: argparse.Namespace) -> int:
    return scan_named_linear_command(args, deps=_named_linear_command_deps())


def _named_linear_command_deps() -> NamedLinearCommandDeps:
    return NamedLinearCommandDeps(
        load_case_from_toml=load_case_from_toml,
        resolve_case=_resolve_case,
        load_linear_terms_from_toml=load_linear_terms_from_toml,
        load_krylov_from_toml=load_krylov_from_toml,
        should_show_progress=_should_show_progress,
        print_linear_run_header=_print_linear_run_header,
        status_printer=_status_printer,
        run_linear_scan=run_linear_scan,
        build_spectral_grid=build_spectral_grid,
        extract_mode_time_series=extract_mode_time_series,
        growth_fit_figure=growth_fit_figure,
        extract_eigenfunction=extract_eigenfunction,
        normalize_eigenfunction=normalize_eigenfunction,
        set_plot_style=set_plot_style,
        load_cyclone_reference=load_cyclone_reference,
        load_etg_reference=load_etg_reference,
        scan_comparison_figure=scan_comparison_figure,
    )


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


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    return run_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    return scan_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_run_runtime_nonlinear(args: argparse.Namespace) -> int:
    return run_runtime_nonlinear_command(args, deps=_runtime_command_deps())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
