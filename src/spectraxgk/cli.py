"""Command line interface for SPECTRAX-GK."""

from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path
from typing import Any, Sequence

from spectraxgk.workflows.runtime import toml as runtime_toml
from spectraxgk.workflows.runtime.toml import (
    load_runtime_from_toml,
    load_toml,
    resolve_runtime_path,
)
from spectraxgk._version import __version__
from spectraxgk.artifacts.plotting import (
    linear_runtime_panel_figure,
    plot_saved_output,
)
from spectraxgk.geometry.miller_eik import generate_runtime_miller_eik
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik
from spectraxgk.workflows.runtime.artifacts import (
    run_runtime_nonlinear_with_artifacts,
    write_quasilinear_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_linear_scan_artifacts,
)
from spectraxgk.workflows.demo import (
    DefaultDemoDeps,
    run_default_linear_demo,
)
from spectraxgk.runtime import run_runtime_linear, run_runtime_scan
from spectraxgk.workflows.runtime.commands import (
    RuntimeCommandDeps,
    attach_preloaded_runtime_config,
    build_runtime_command_deps,
    plot_saved_output_command,
    run_runtime_linear_command,
    run_runtime_nonlinear_command,
    scan_runtime_linear_command,
)


# These imports remain on the executable facade so tests and downstream callers
# can patch command dependencies without reaching into workflow internals.
_PATCHABLE_RUNTIME_COMMAND_GLOBALS = (
    load_runtime_from_toml,
    resolve_runtime_path,
    run_runtime_linear,
    run_runtime_scan,
    run_runtime_nonlinear_with_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_linear_scan_artifacts,
    write_quasilinear_artifacts,
)


def _is_runtime_toml(data: dict[str, Any]) -> bool:
    """Return whether TOML data should use the runtime executable path."""

    return runtime_toml.is_runtime_toml(data)


def _toml_shorthand_command(data: dict[str, Any]) -> str:
    """Return the executable command used for direct TOML path shorthand."""

    return runtime_toml.toml_shorthand_command(data)


def _direct_config_shorthand_args(argv: Sequence[str]) -> list[str] | None:
    """Return parser arguments for ``spectraxgk case.toml`` shorthand."""

    return runtime_toml.direct_config_shorthand_args(argv, load_toml_func=load_toml)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        cfg, data = load_runtime_from_toml(args.config)
    except Exception as exc:
        print(f"Error loading {args.config}: {exc}")
        return 1

    attach_preloaded_runtime_config(args, cfg, data)
    if cfg.physics.nonlinear:
        return _cmd_run_runtime_nonlinear(args)
    return _cmd_run_runtime_linear(args)


def _cmd_default_demo() -> int:
    deps = DefaultDemoDeps(
        load_runtime_from_toml=load_runtime_from_toml,
        run_runtime_linear=run_runtime_linear,
        linear_runtime_panel_figure=linear_runtime_panel_figure,
        write_runtime_linear_artifacts=write_runtime_linear_artifacts,
    )
    return run_default_linear_demo(deps=deps)


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


def _add_config_flag(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--config", required=True, help="Path to TOML config")


def _add_ky_values_flag(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument("--ky-values", type=str, default=None, help="Comma-separated ky list")


def _add_scan_worker_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--batch-ky", action="store_true", help="Integrate all ky in one batch"
    )
    cmd.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent ky workers for the serial scan path, including quasilinear spectra.",
    )
    cmd.add_argument(
        "--parallel-executor",
        choices=("thread", "process"),
        default="thread",
        help="Executor for independent ky workers.",
    )


def _add_laguerre_mode_flag(cmd: argparse.ArgumentParser, *, help_text: str) -> None:
    cmd.add_argument("--laguerre-mode", type=str, default=None, help=help_text)


def _with_miller_helper_overrides(
    cfg: Any, args: argparse.Namespace
) -> Any:
    if args.geometry_helper_python is None and args.geometry_helper_repo is None:
        return cfg
    return replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            geometry_helper_python=args.geometry_helper_python,
            geometry_helper_repo=(
                None
                if args.geometry_helper_repo is None
                else str(args.geometry_helper_repo)
            ),
        ),
    )


def _cmd_generate_geometry(args: argparse.Namespace) -> int:
    cfg, _ = load_runtime_from_toml(args.config)
    if args.geometry == "vmec":
        output = generate_runtime_vmec_eik(
            cfg, output_path=args.out, force=bool(args.force)
        )
    elif args.geometry == "miller":
        output = generate_runtime_miller_eik(
            _with_miller_helper_overrides(cfg, args),
            output_path=args.out,
            force=bool(args.force),
        )
    else:  # pragma: no cover - argparse owns the allowed values.
        raise ValueError(f"unknown geometry backend: {args.geometry}")
    print(output)
    return 0


def _add_geometry_parser(sub: argparse._SubParsersAction) -> None:
    geometry = sub.add_parser(
        "geometry", help="Generate solver geometry from a runtime TOML configuration"
    )
    backends = geometry.add_subparsers(dest="geometry", required=True)
    for name, help_text in (
        ("vmec", "Generate a VMEC-derived EIK file"),
        ("miller", "Generate a Miller EIK file"),
    ):
        backend = backends.add_parser(name, help=help_text)
        backend.add_argument("--config", required=True, type=Path)
        backend.add_argument("--out", type=Path, default=None)
        backend.add_argument("--force", action="store_true")
        if name == "miller":
            backend.add_argument("--geometry-helper-repo", type=Path, default=None)
            backend.add_argument("--geometry-helper-python", default=None)
        backend.set_defaults(func=_cmd_generate_geometry)


def _add_generic_run_parser(sub: argparse._SubParsersAction) -> None:
    generic_run = sub.add_parser(
        "run",
        help="Run a simulation from a TOML config (auto-detect linear/nonlinear)",
    )
    _add_config_flag(generic_run)
    _add_resolution_flags(generic_run)
    generic_run.add_argument("--solver", type=str, default=None)
    _add_time_solver_flags(generic_run, sample_stride=True, fit_signal=True)
    generic_run.add_argument("--diagnostics-stride", type=int, default=None)
    _add_diagnostics_flags(generic_run)
    _add_laguerre_mode_flag(generic_run, help_text="grid or spectral (nonlinear only)")
    _add_runtime_paths(generic_run, init_help="Optional init file for nonlinear runs")
    _add_quasilinear_flags(generic_run)
    _add_progress_flags(generic_run)
    generic_run.set_defaults(func=_cmd_run)


def _add_runtime_parsers(sub: argparse._SubParsersAction) -> None:
    run_runtime = sub.add_parser(
        "run-runtime-linear",
        help="Run one linear point from unified runtime TOML config",
    )
    _add_config_flag(run_runtime)
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
    _add_config_flag(scan_runtime)
    _add_ky_values_flag(scan_runtime)
    scan_runtime.add_argument("--Nl", type=int, default=None)
    scan_runtime.add_argument("--Nm", type=int, default=None)
    _add_time_solver_flags(
        scan_runtime, solver=True, sample_stride=True, fit_signal=True
    )
    _add_scan_worker_flags(scan_runtime)
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
    _add_config_flag(run_runtime_nl)
    _add_resolution_flags(run_runtime_nl, ky_help="Single ky value")
    _add_time_solver_flags(run_runtime_nl)
    run_runtime_nl.add_argument("--sample-stride", type=int, default=None)
    run_runtime_nl.add_argument("--diagnostics-stride", type=int, default=None)
    _add_diagnostics_flags(run_runtime_nl)
    _add_laguerre_mode_flag(
        run_runtime_nl, help_text="grid or spectral (nonlinear Laguerre handling)"
    )
    _add_runtime_paths(
        run_runtime_nl,
        init_help="Optional restart/init-state file containing a matching distribution state",
    )
    _add_progress_flags(run_runtime_nl)
    run_runtime_nl.set_defaults(func=_cmd_run_runtime_nonlinear)


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
    sub = parser.add_subparsers(dest="cmd")
    _add_generic_run_parser(sub)
    _add_runtime_parsers(sub)
    _add_geometry_parser(sub)

    return parser


def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        return _cmd_default_demo()
    if argv[0] == "--plot":
        return plot_saved_output_command(argv, plot_saved_output=plot_saved_output)

    shorthand_args = _direct_config_shorthand_args(argv)
    if shorthand_args is not None:
        parser = build_parser()
        args = parser.parse_args(shorthand_args)
        return args.func(args)

    parser = build_parser()
    args = parser.parse_args()
    if args.cmd is None:
        return _cmd_default_demo()
    return args.func(args)


def _runtime_command_deps() -> RuntimeCommandDeps:
    return build_runtime_command_deps(sys.modules[__name__])


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    return run_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    return scan_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_run_runtime_nonlinear(args: argparse.Namespace) -> int:
    return run_runtime_nonlinear_command(args, deps=_runtime_command_deps())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
