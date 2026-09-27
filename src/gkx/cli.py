"""Command line interface for GKX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence
from dataclasses import dataclass
from typing import Callable

from gkx.workflows.runtime import (
    wout as runtime_wout,
)
from gkx.workflows.runtime.toml import (
    load_runtime_from_toml,
    load_toml,
    resolve_runtime_path,
)
from gkx._version import __version__
from gkx.artifacts.plotting import (
    linear_runtime_panel_figure,
    plot_saved_output,
)
from gkx.geometry.miller_eik import generate_runtime_miller_eik
from gkx.geometry.vmec_eik import generate_runtime_vmec_eik
from gkx.workflows.runtime.artifacts import (
    run_runtime_nonlinear_with_artifacts,
    write_quasilinear_artifacts,
    write_runtime_linear_artifacts,
    write_runtime_linear_scan_artifacts,
)
from gkx.runtime import run_runtime_linear, run_runtime_scan
from gkx.compilation_cache import enable_persistent_compilation_cache
from gkx.workflows.runtime.commands import (
    RuntimeCommandDeps,
    attach_preloaded_runtime_config,
    build_runtime_command_deps,
    plot_saved_output_command,
    run_runtime_linear_command,
    run_runtime_nonlinear_command,
    scan_runtime_linear_command,
)


# The demo is the first thing most people run, so it reports a number that can
# be checked rather than a fast one that cannot. At dt = 0.03 over 500 steps it
# printed gamma = 0.089982 against a certified Krylov eigenvalue of 0.103263 at
# these same settings -- 12.9 percent low -- and emitted four warnings saying so:
# the step exceeded the estimated CFL bound of 0.02197, and 15 time units gave
# only 1.34 e-foldings to fit. Both are fixed by construction here. dt sits
# under the CFL bound, and 4000 steps reach t = 80, which is past the
# gamma * t_max >= 7 the fitter asks for. The run costs about 7 s instead of
# 3 s and lands within 0.02 percent of the eigenvalue with no warnings.
# tests/validation/physics_gates/test_collision_physics.py pins that agreement.
DEFAULT_DEMO_SETTINGS: dict[str, float | int | str] = {
    "ky": 0.3,
    "Nl": 7,
    "Nm": 14,
    "solver": "time",
    "method": "rk4",
    "dt": 0.02,
    "steps": 4000,
    "sample_stride": 5,
    "fit_signal": "phi",
}


@dataclass(frozen=True)
class DefaultDemoDeps:
    """Patchable runtime and output dependencies for the default demo."""

    load_runtime_from_toml: Callable[..., tuple[Any, dict[str, Any]]]
    run_runtime_linear: Callable[..., Any]
    linear_runtime_panel_figure: Callable[..., tuple[Any, Any]]
    write_runtime_linear_artifacts: Callable[[str | Path, Any], dict[str, str]]


def default_demo_plot_path() -> Path:
    """Return the figure path produced by the no-input executable."""

    return Path("gkx_default_linear.png")


def default_demo_artifact_base() -> Path:
    """Return the artifact stem produced by the no-input executable."""

    return Path("gkx_default_linear")


def default_demo_toml_path() -> Path:
    """Return the reproducer path produced by the no-input executable."""

    return Path("gkx_default_linear.toml")


def default_demo_toml_text() -> str:
    """Build the complete runtime TOML used by the educational demo."""

    settings = DEFAULT_DEMO_SETTINGS
    return f"""# Reproducer for the no-input `gkx` demo.
# Run with: gkx gkx_default_linear.toml --progress

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
nu = 0.0
kinetic = true

[grid]
Nx = 1
Ny = 24
Nz = 96
Lx = 62.8
Ly = 62.8
boundary = "linked"
y0 = 20.0
ntheta = 32
nperiod = 2

[time]
t_max = {float(settings["dt"]) * int(settings["steps"]):.6g}
dt = {settings["dt"]}
method = "{settings["method"]}"
sample_stride = {settings["sample_stride"]}
progress_bar = true

[geometry]
model = "s-alpha"
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1.0e-10
gaussian_init = true
gaussian_width = 0.5

[physics]
linear = true
nonlinear = false
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
collisions = true
hypercollisions = true

[collisions]
nu_hermite = 1.0
nu_laguerre = 2.0
nu_hyper = 0.0
p_hyper = 4.0
hypercollisions_const = 0.0
hypercollisions_kz = 1.0
damp_ends_amp = 0.1
damp_ends_widthfrac = 0.125

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[terms]
streaming = 1.0
mirror = 1.0
curvature = 1.0
gradb = 1.0
diamagnetic = 1.0
collisions = 1.0
hypercollisions = 1.0
end_damping = 1.0
apar = 1.0
bpar = 0.0
nonlinear = 0.0

[run]
ky = {settings["ky"]}
Nl = {settings["Nl"]}
Nm = {settings["Nm"]}
solver = "{settings["solver"]}"
method = "{settings["method"]}"
dt = {settings["dt"]}
steps = {settings["steps"]}
sample_stride = {settings["sample_stride"]}

[fit]
fit_signal = "{settings["fit_signal"]}"
auto_window = true
window_fraction = 0.4
start_fraction = 0.2
min_points = 25
"""


def _status(message: str) -> None:
    print(f"demo: {message}", flush=True)


def _print_intro(toml_path: Path) -> None:
    settings = DEFAULT_DEMO_SETTINGS
    print(
        "No input specified; running the default Cyclone initial-value demo.",
        flush=True,
    )
    print(
        "The first run includes JAX compilation; progress reports elapsed time and ETA.",
        flush=True,
    )
    print(
        f"ky={settings['ky']} Nl={settings['Nl']} Nm={settings['Nm']} "
        f"method={settings['method']} dt={settings['dt']} steps={settings['steps']}",
        flush=True,
    )
    print(f"wrote reproducible input: {toml_path}", flush=True)


def _write_plot(deps: DefaultDemoDeps, result: Any) -> Path:
    path = default_demo_plot_path()
    fig, _axes = deps.linear_runtime_panel_figure(
        t=result.t,
        signal=result.signal,
        z=result.z,
        eigenfunction=result.eigenfunction,
        gamma=float(result.gamma),
        omega=float(result.omega),
        title="GKX default Cyclone initial-value demo",
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def run_default_linear_demo(*, deps: DefaultDemoDeps) -> int:
    """Run one small runtime case and write its TOML, data, and figure locally."""

    settings = DEFAULT_DEMO_SETTINGS
    toml_path = default_demo_toml_path()
    toml_path.write_text(default_demo_toml_text(), encoding="utf-8")
    _print_intro(toml_path)
    cfg, raw = deps.load_runtime_from_toml(toml_path)
    fit = dict(raw.get("fit", {}))
    result = deps.run_runtime_linear(
        cfg,
        ky_target=float(settings["ky"]),
        Nl=int(settings["Nl"]),
        Nm=int(settings["Nm"]),
        solver=str(settings["solver"]),
        method=str(settings["method"]),
        dt=float(settings["dt"]),
        steps=int(settings["steps"]),
        sample_stride=int(settings["sample_stride"]),
        fit_signal=str(fit.pop("fit_signal", settings["fit_signal"])),
        show_progress=True,
        status_callback=_status,
        **fit,
    )
    paths = deps.write_runtime_linear_artifacts(default_demo_artifact_base(), result)
    plot_path = _write_plot(deps, result)
    print(
        f"gamma={float(result.gamma):.6f} omega={float(result.omega):.6f}", flush=True
    )
    for path in paths.values():
        print(f"saved {path}", flush=True)
    print(f"saved {plot_path}", flush=True)
    print(f"rerun with: gkx {toml_path} --progress", flush=True)
    return 0


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


def _direct_config_shorthand_args(argv: Sequence[str]) -> list[str] | None:
    """Return parser arguments for ``gkx case.toml`` shorthand."""

    return runtime_wout.direct_config_shorthand_args(argv, load_toml_func=load_toml)


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
        "--quasilinear",
        action="store_true",
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


def _add_plot_flags(cmd: argparse.ArgumentParser) -> None:
    cmd.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip the figures a completed run writes beside its output",
    )


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


def _add_saturation_flags(cmd: argparse.ArgumentParser) -> None:
    group = cmd.add_mutually_exclusive_group()
    group.add_argument(
        "--until-saturated",
        action="store_true",
        help="Stop once the heat-flux window saturates ([time] run_to = 'saturation')",
    )
    group.add_argument(
        "--no-until-saturated",
        action="store_true",
        help="Always integrate to t_max ([time] run_to = 't_max')",
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
        cmd.add_argument(
            "--solver", type=str, default=None, help="auto, time, or krylov"
        )
    cmd.add_argument("--method", type=str, default=None, help="time integrator method")
    cmd.add_argument("--dt", type=float, default=None)
    cmd.add_argument("--steps", type=int, default=None)
    if sample_stride:
        cmd.add_argument("--sample-stride", type=int, default=None)
    if fit_signal:
        cmd.add_argument(
            "--fit-signal", type=str, default=None, help="auto, phi, or density"
        )


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
    cmd.add_argument(
        "--ky-values", type=str, default=None, help="Comma-separated ky list"
    )


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
    cmd.add_argument(
        "--warm-start",
        dest="warm_start",
        action="store_true",
        default=None,
        help="Seed each ky from its neighbour's converged state (default off).",
    )
    cmd.add_argument(
        "--no-warm-start",
        dest="warm_start",
        action="store_false",
        help="Start every ky from the cold initial condition (the default).",
    )


def _add_laguerre_mode_flag(cmd: argparse.ArgumentParser, *, help_text: str) -> None:
    cmd.add_argument("--laguerre-mode", type=str, default=None, help=help_text)


def _cmd_generate_geometry(args: argparse.Namespace) -> int:
    cfg, _ = load_runtime_from_toml(args.config)
    if args.geometry == "vmec":
        output = generate_runtime_vmec_eik(
            cfg, output_path=args.out, force=bool(args.force)
        )
    elif args.geometry == "miller":
        output = generate_runtime_miller_eik(
            cfg,
            output_path=args.out,
            force=bool(args.force),
        )
    else:  # pragma: no cover - argparse owns the allowed values.
        raise ValueError(f"unknown geometry backend: {args.geometry}")
    print(output)
    return 0


# The plan's six commands are run, scan, estimate, plot, inspect and validate.
# estimate, inspect and validate were previously reachable only as flags on the
# equilibrium shorthand or not at all, which meant a user could not discover
# them from `gkx --help`.

_DEPRECATED_COMMANDS = {
    "run-runtime-linear": "run",
    "scan-runtime-linear": "scan",
    "run-runtime-nonlinear": "run",
}


def _warn_deprecated_command(name: str) -> None:
    """Name the replacement for a command kept for one release."""

    replacement = _DEPRECATED_COMMANDS[name]
    print(
        f"gkx: '{name}' is deprecated and will be removed in the next release; "
        f"use 'gkx {replacement}' instead.",
        file=sys.stderr,
    )


def _cmd_estimate(args: argparse.Namespace) -> int:
    """Print the deterministic minimum-grid estimate for an equilibrium."""

    from gkx.workflows.runtime.wout import (
        estimate_resolution,
        format_estimate_table,
    )

    estimate = estimate_resolution(
        args.equilibrium, torflux=args.torflux, target_error=args.tier
    )
    print(f"equilibrium: {args.equilibrium}")
    print(format_estimate_table(estimate))
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Describe a case or a saved result without running anything."""

    import json

    target = Path(args.target)
    if target.suffix.lower() == ".toml":
        from gkx.workflows.runtime.toml import load

        payload = load(target).summary()
    else:
        from gkx.artifacts.plotting import _artifact_base, _sidecar

        # The writers append their suffix to the output path, so a run whose
        # [output] path is foo.out.nc writes foo.out.nc.summary.json. Try that
        # first; fall back to the stripped base for a path already carrying a
        # sidecar suffix.
        candidates = (
            _sidecar(target, ".summary.json"),
            _sidecar(_artifact_base(target), ".summary.json"),
        )
        summary_path = next((c for c in candidates if c.exists()), None)
        if summary_path is None:
            print(f"gkx: no summary beside {target}", file=sys.stderr)
            return 1
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a case file and report the first problem in plain language."""

    from gkx.workflows.runtime.toml import load

    try:
        case = load(Path(args.config))
        case.validate()
    except (ValueError, OSError) as exc:
        print(f"gkx: {args.config} is not runnable: {exc}", file=sys.stderr)
        return 1
    print(f"gkx: {args.config} is a valid case")
    return 0


def _add_product_parsers(sub: argparse._SubParsersAction) -> None:
    """Register estimate, inspect and validate."""

    estimate = sub.add_parser(
        "estimate", help="Print the minimum-grid estimate for an equilibrium"
    )
    estimate.add_argument("equilibrium", type=str)
    estimate.add_argument(
        "--tier",
        choices=("preview", "standard", "cautious"),
        default="standard",
    )
    estimate.add_argument("--torflux", type=float, default=None)
    estimate.set_defaults(func=_cmd_estimate)

    inspect_p = sub.add_parser(
        "inspect", help="Describe a case TOML or a saved result without running it"
    )
    inspect_p.add_argument("target", type=str)
    inspect_p.set_defaults(func=_cmd_inspect)

    validate = sub.add_parser("validate", help="Check that a case TOML is runnable")
    validate.add_argument("config", type=str)
    validate.set_defaults(func=_cmd_validate)


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
    _add_saturation_flags(generic_run)
    _add_laguerre_mode_flag(generic_run, help_text="grid or spectral (nonlinear only)")
    _add_runtime_paths(generic_run, init_help="Optional init file for nonlinear runs")
    _add_quasilinear_flags(generic_run)
    _add_progress_flags(generic_run)
    _add_plot_flags(generic_run)
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
    _add_plot_flags(run_runtime)
    run_runtime.set_defaults(func=_cmd_run_runtime_linear)

    scan_runtime = sub.add_parser(
        "scan",
        aliases=("scan-runtime-linear",),
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
    _add_plot_flags(scan_runtime)
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
    _add_saturation_flags(run_runtime_nl)
    _add_laguerre_mode_flag(
        run_runtime_nl, help_text="grid or spectral (nonlinear Laguerre handling)"
    )
    _add_runtime_paths(
        run_runtime_nl,
        init_help="Optional restart/init-state file containing a matching distribution state",
    )
    _add_progress_flags(run_runtime_nl)
    _add_plot_flags(run_runtime_nl)
    run_runtime_nl.set_defaults(func=_cmd_run_runtime_nonlinear)


# Path-valued CLI flags (--vmec-file, --geometry-file, --init-file) follow
# shell conventions: relative paths resolve against cwd, ~ expands to $HOME,
# and $VAR is expanded from the environment. See _apply_runtime_path_overrides.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name if sys.argv else "gkx",
        description="Run, diagnose, and plot GKX simulations.",
        epilog=(
            "Run without arguments for the self-contained linear demo. "
            "Run a VMEC/VMEX equilibrium with: %(prog)s wout_XXX.nc [deck.toml] "
            "[--linear] (aliases: --vmec/--vmex FILE); add --estimate to print "
            "the minimum-grid estimate and exit without running. "
            "Plot a saved result with: %(prog)s plot OUTPUT_FILE [--out FIGURE.png] (legacy: %(prog)s --plot OUTPUT_FILE)."
        ),
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
    _add_product_parsers(sub)

    return parser


def main() -> int:
    argv = sys.argv[1:]
    # Every executable path that can compile gets the persistent cache, before
    # any solver is built. Re-running an unchanged case is the common case
    # while a user edits a TOML, and it is the case that was paying a full
    # cold compile every time. See gkx.compilation_cache.
    enable_persistent_compilation_cache()
    if not argv:
        return _cmd_default_demo()
    if argv[0] in {"plot", "--plot"}:
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


def _warn_if_invoked_deprecated(args: argparse.Namespace) -> None:
    # ``gkx run`` and ``gkx scan`` dispatch here too; only the old spellings warn.
    if getattr(args, "cmd", None) in _DEPRECATED_COMMANDS:
        _warn_deprecated_command(args.cmd)


def _cmd_run_runtime_linear(args: argparse.Namespace) -> int:
    _warn_if_invoked_deprecated(args)
    return run_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_scan_runtime_linear(args: argparse.Namespace) -> int:
    _warn_if_invoked_deprecated(args)
    return scan_runtime_linear_command(args, deps=_runtime_command_deps())


def _cmd_run_runtime_nonlinear(args: argparse.Namespace) -> int:
    _warn_if_invoked_deprecated(args)
    return run_runtime_nonlinear_command(args, deps=_runtime_command_deps())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
