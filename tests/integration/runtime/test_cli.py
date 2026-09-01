"""Runtime user-facing surface: the gkx CLI commands and the runtime command/orchestration helpers they dispatch into."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from gkx import __version__
from gkx.cli import (
    _cmd_run,
    _cmd_run_runtime_linear,
    _cmd_run_runtime_nonlinear,
    _cmd_scan_runtime_linear,
    _direct_config_shorthand_args,
    _is_runtime_toml,
    _toml_shorthand_command,
    main,
)
from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.core.grid import build_spectral_grid
from gkx.diagnostics import ResolvedDiagnostics, SimulationDiagnostics
from gkx.diagnostics.analysis import ModeSelection
from gkx.diagnostics.growth_rates import (
    fit_growth_rate,
    fit_growth_rate_auto,
    fit_growth_rate_auto_with_stats,
    fit_growth_rate_with_stats,
)
from gkx.diagnostics.modes import extract_mode_time_series
from gkx.runtime import (
    _build_initial_condition,
    _build_gaussian_profile,
    _concat_runtime_diagnostics,
    _enforce_full_ky_hermitian,
    _expand_ky,
    _centered_glibc_random_pairs,
    _default_hermite_hypercollision_exponent,
    _dealiased_initial_mode_pairs,
    _periodic_zp_from_grid,
    _infer_runtime_nonlinear_steps,
    _load_initial_state_from_file,
    _midplane_index,
    _normalize_linear_solver_name,
    _require_full_gk_runtime_model,
    _resolve_runtime_hl_dims,
    _reshape_netcdf_state,
    _runtime_external_phi,
    _runtime_default_krylov_config,
    _runtime_model_key,
    _select_nonlinear_mode_indices,
    _slice_runtime_diagnostics,
    _species_to_linear,
    _stride_runtime_diagnostics,
    _zero_kx_index,
    _run_runtime_scan_batch,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    build_runtime_term_config,
    run_runtime_nonlinear,
    run_runtime_scan,
)
from gkx.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from gkx.terms.config import FieldState
from gkx.workflows.runtime.chunks import (
    build_runtime_progress_message,
    format_duration,
)
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimeOutputConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
)
from gkx.workflows.runtime.diagnostics import (
    RuntimeQuasilinearFinalizationDeps,
    finalize_runtime_linear_quasilinear,
)
from gkx.workflows.runtime.diagnostics import (
    half_horizon_settled_probe,
    warn_if_growth_unresolved,
)
from gkx.workflows.runtime.diagnostics import _prepare_runtime_linear_fit_inputs
from gkx.workflows.runtime.diagnostics import fit_runtime_linear_diagnostics
from gkx.workflows.runtime.orchestration_scan import (
    _BatchDiagnostics,
    _fit_batch_scan_point,
    _RuntimeScanOptions,
    run_runtime_scan_ky_task,
)
from pathlib import Path
from support.paths import REPO_ROOT
from types import SimpleNamespace
import argparse
import gkx.cli as cli
import gkx.runtime as runtime
import gkx.workflows.runtime.commands as runtime_cases
import gkx.workflows.runtime.commands as runtime_commands
import gkx.workflows.runtime.orchestration_artifacts as runtime_artifacts
import gkx.workflows.runtime.policies as runtime_policies
import gkx.workflows.runtime.warm_start as warm_start
import json
import numpy as np
import os
import pytest
import sys
import tomllib
import warnings


def _project_version() -> str:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]


def _runtime_linear_result() -> RuntimeLinearResult:
    """Return the canonical tiny result used by executable output tests."""

    return RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0, 2.0]),
    )


def test_version_exposed():
    """Version string should be exported from the package."""
    assert __version__ == _project_version()


def test_cli_version_flag(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gkx", "--version"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert f"gkx {__version__}" in out


def test_cli_without_args_runs_default_demo(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    class _FakeFigure:
        def savefig(self, path, **_kwargs):
            Path(path).write_bytes(b"fake-image")

    fake_result = RuntimeLinearResult(
        ky=0.3,
        gamma=0.12,
        omega=-0.34,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        t=np.asarray([0.1, 0.2]),
        signal=np.asarray([1.0 + 0.0j, 1.2 + 0.1j]),
        z=np.asarray([-1.0, 1.0]),
        eigenfunction=np.asarray([1.0 + 0.0j, 0.5 + 0.2j]),
    )

    monkeypatch.chdir(tmp_path)
    run_kwargs: dict[str, object] = {}

    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (RuntimeConfig(), {"fit": {"fit_signal": "phi"}}),
    )

    def _fake_run(_cfg, **kwargs):
        run_kwargs.update(kwargs)
        return fake_result

    monkeypatch.setattr("gkx.cli.run_runtime_linear", _fake_run)
    monkeypatch.setattr(
        "gkx.cli.linear_runtime_panel_figure",
        lambda **_kwargs: (_FakeFigure(), None),
    )
    monkeypatch.setattr("matplotlib.pyplot.close", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sys, "argv", ["gkx"])

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "No input specified" in out
    assert run_kwargs["solver"] == "time"
    assert run_kwargs["sample_stride"] == 5
    assert run_kwargs["show_progress"] is True
    assert (tmp_path / "gkx_default_linear.png").exists()
    assert (tmp_path / "gkx_default_linear.toml").exists()
    assert (tmp_path / "gkx_default_linear.summary.json").exists()


def test_cli_help_advertises_default_demo_and_plot_route() -> None:
    help_text = cli.build_parser().format_help()

    assert "without arguments for the self-contained linear demo" in help_text
    assert "plot OUTPUT_FILE" in help_text
    assert "--plot OUTPUT_FILE" in help_text
    assert "scan-runtime-linear" in help_text


@pytest.mark.parametrize("plot_command", ["plot", "--plot"])
def test_cli_global_plot_uses_saved_output_renderer(
    capsys, monkeypatch, tmp_path: Path, plot_command: str
) -> None:
    rendered = tmp_path / "rendered.png"
    monkeypatch.setattr("gkx.cli.plot_saved_output", lambda path, out=None: rendered)
    monkeypatch.setattr(
        sys, "argv", ["gkx", plot_command, "tools_out/linear_case.summary.json"]
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {rendered}" in out


def test_cli_global_plot_accepts_out_argument(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    rendered = tmp_path / "rendered-out.png"
    captured: dict[str, str | None] = {}

    def _plot(path, out=None):
        captured["path"] = path
        captured["out"] = out
        return rendered

    monkeypatch.setattr("gkx.cli.plot_saved_output", _plot)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", "--plot", "case.summary.json", "--out", "figure.png"],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert captured == {"path": "case.summary.json", "out": "figure.png"}
    assert f"saved {rendered}" in out


def test_cli_global_plot_renders_linear_scan_bundle(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """--plot should render the saved ky-scan bundle written by scan-runtime-linear."""

    scan = type(
        "Scan",
        (),
        {
            "ky": np.array([0.2, 0.3]),
            "gamma": np.array([0.05, 0.08]),
            "omega": np.array([0.25, 0.31]),
        },
    )()
    paths = cli.write_runtime_linear_scan_artifacts(tmp_path / "scan", scan)
    assert (tmp_path / "scan.scan.csv").exists()

    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", paths["scan"]])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {tmp_path / 'scan.plot.png'}" in out
    assert (tmp_path / "scan.plot.png").exists()


def test_cli_plot_usage_errors(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gkx", "plot"])
    assert main() == 1
    assert "usage: gkx plot" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", "a", "--bad"])
    assert main() == 1
    assert "usage: gkx plot" in capsys.readouterr().out


def test_runtime_command_deps_are_built_from_patchable_cli_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object()
    writer = object()

    monkeypatch.setattr(cli, "run_runtime_scan", runner)
    monkeypatch.setattr(cli, "write_runtime_linear_artifacts", writer)

    deps = cli._runtime_command_deps()

    assert deps.run_runtime_scan is runner
    assert deps.write_runtime_linear_artifacts is writer


def test_cli_runtime_toml_dispatch_is_uniform() -> None:
    assert _is_runtime_toml({"physics": {}}) is True
    assert _is_runtime_toml({"case": "cyclone"}) is True
    assert _is_runtime_toml({}) is True
    assert _toml_shorthand_command({"physics": {}}) == "run"
    assert _toml_shorthand_command({"case": "cyclone"}) == "run"

    parser = cli.build_parser()
    promoted = parser.parse_args(["scan", "--config", "case.toml"])
    legacy = parser.parse_args(["scan-runtime-linear", "--config", "case.toml"])
    assert promoted.func is legacy.func is cli._cmd_scan_runtime_linear


def test_cli_geometry_routes_vmec_and_miller_backends(
    capsys, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    @dataclass(frozen=True)
    class _GeometryConfig:
        geometry_helper_python: str | None = None
        geometry_helper_repo: str | None = None

    @dataclass(frozen=True)
    class _RuntimeConfig:
        geometry: _GeometryConfig

    cfg = _RuntimeConfig(geometry=_GeometryConfig())
    loaded_configs: list[Path] = []
    calls: list[tuple[str, object, Path | None, bool]] = []

    def _load_runtime(path):
        loaded_configs.append(Path(path))
        return cfg, {"source": str(path)}

    def _vmec(runtime_cfg, *, output_path, force):
        calls.append(("vmec", runtime_cfg, output_path, force))
        return tmp_path / "vmec.eik.nc"

    def _miller(runtime_cfg, *, output_path, force):
        calls.append(("miller", runtime_cfg, output_path, force))
        return tmp_path / "miller.eiknc.nc"

    monkeypatch.setattr(cli, "load_runtime_from_toml", _load_runtime)
    monkeypatch.setattr(cli, "generate_runtime_vmec_eik", _vmec)
    monkeypatch.setattr(cli, "generate_runtime_miller_eik", _miller)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "geometry",
            "vmec",
            "--config",
            str(tmp_path / "vmec.toml"),
            "--out",
            str(tmp_path / "vmec.nc"),
            "--force",
        ],
    )
    assert main() == 0
    assert calls[-1] == ("vmec", cfg, tmp_path / "vmec.nc", True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "geometry",
            "miller",
            "--config",
            str(tmp_path / "miller.toml"),
            "--out",
            str(tmp_path / "miller.nc"),
        ],
    )
    assert main() == 0
    kind, runtime_cfg, output_path, force = calls[-1]
    assert kind == "miller"
    assert output_path == tmp_path / "miller.nc"
    assert force is False
    assert runtime_cfg is cfg
    assert loaded_configs == [tmp_path / "vmec.toml", tmp_path / "miller.toml"]
    assert "vmec.eik.nc" in capsys.readouterr().out


def test_direct_config_shorthand_args_resolve_command_and_guards(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "case.toml"
    cfg_path.write_text("[physics]\n", encoding="utf-8")

    monkeypatch.setattr("gkx.cli.load_toml", lambda _path: {"physics": {}})
    assert _direct_config_shorthand_args([str(cfg_path), "--no-progress"]) == [
        "run",
        "--config",
        str(cfg_path),
        "--no-progress",
    ]

    monkeypatch.setattr("gkx.cli.load_toml", lambda _path: {"case": "cyclone"})
    assert _direct_config_shorthand_args([str(cfg_path), "--plot"]) == [
        "run",
        "--config",
        str(cfg_path),
        "--plot",
    ]

    assert _direct_config_shorthand_args([]) is None
    assert _direct_config_shorthand_args(["--version"]) is None
    assert _direct_config_shorthand_args(["run", "--config", str(cfg_path)]) is None
    assert _direct_config_shorthand_args([str(tmp_path / "missing.toml")]) is None


def test_cmd_run_handles_load_error_and_dispatches(monkeypatch, capsys) -> None:
    args = argparse.Namespace(config="bad.toml")

    def _boom(_path):
        raise RuntimeError("forced")

    monkeypatch.setattr("gkx.cli.load_runtime_from_toml", _boom)
    assert _cmd_run(args) == 1
    assert "Error loading bad.toml" in capsys.readouterr().out

    nonlinear_cfg = type(
        "Cfg", (), {"physics": type("Phys", (), {"nonlinear": True})()}
    )()
    linear_cfg = type(
        "Cfg", (), {"physics": type("Phys", (), {"nonlinear": False})()}
    )()
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (nonlinear_cfg, {})
    )
    monkeypatch.setattr("gkx.cli._cmd_run_runtime_nonlinear", lambda _args: 7)
    assert _cmd_run(args) == 7

    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (linear_cfg, {})
    )
    monkeypatch.setattr("gkx.cli._cmd_run_runtime_linear", lambda _args: 9)
    assert _cmd_run(args) == 9


def test_cmd_run_reuses_loaded_runtime_config_for_linear_dispatch(
    monkeypatch,
) -> None:
    cfg = RuntimeConfig()
    load_calls: list[str] = []
    captured: dict[str, object] = {}

    def _load_runtime(path):
        load_calls.append(str(path))
        return cfg, {"run": {"ky": 0.2}}

    def _run_runtime_linear(cfg_in, **kwargs):
        captured["cfg"] = cfg_in
        captured["kwargs"] = kwargs
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            t=np.asarray([0.0, 1.0]),
            signal=np.asarray([1.0, 1.2]),
        )

    monkeypatch.setattr("gkx.cli.load_runtime_from_toml", _load_runtime)
    monkeypatch.setattr("gkx.cli.run_runtime_linear", _run_runtime_linear)
    args = argparse.Namespace(
        config="case.toml",
        ky=None,
        Nl=None,
        Nm=None,
        solver=None,
        fit_signal=None,
        method=None,
        dt=None,
        steps=None,
        sample_stride=None,
        progress=False,
        no_progress=True,
        out=None,
        vmec_file=None,
        geometry_file=None,
        quasilinear=False,
        ql_mode=None,
        ql_saturation_rule=None,
        ql_csat=None,
        ql_normalization=None,
        ql_output=None,
    )

    assert _cmd_run(args) == 0
    assert load_calls == ["case.toml"]
    assert captured["kwargs"]["ky_target"] == pytest.approx(0.2)


def test_main_shorthand_dispatches_all_toml_through_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    cfg_path = tmp_path / "case.toml"
    cfg_path.write_text("[physics]\n", encoding="utf-8")
    captured: list[list[str]] = []

    class _Parser:
        def parse_args(self, argv):
            captured.append(list(argv))
            return argparse.Namespace(func=lambda _args: 11)

    monkeypatch.setattr("gkx.cli.load_toml", lambda _path: {"physics": {}})
    monkeypatch.setattr("gkx.cli.build_parser", lambda: _Parser())
    monkeypatch.setattr(sys, "argv", ["gkx", str(cfg_path)])
    assert main() == 11
    assert captured[-1][:2] == ["run", "--config"]

    monkeypatch.setattr("gkx.cli.load_toml", lambda _path: {"case": "cyclone"})
    monkeypatch.setattr(sys, "argv", ["gkx", str(cfg_path)])
    assert main() == 11
    assert captured[-1][:2] == ["run", "--config"]


def test_cli_run_runtime_linear(capsys, monkeypatch, tmp_path: Path):
    """The unified runtime command should run a tiny linear configuration."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 4
Nm = 6
solver = "krylov"
"""
    path = tmp_path / "runtime_cli.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", "run-runtime-linear", "--config", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "starting runtime linear run" in out
    assert "gamma=" in out


def test_cli_run_runtime_linear_writes_artifacts(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""
    path = tmp_path / "runtime_cli_linear_out.toml"
    path.write_text(cfg, encoding="utf-8")
    out_base = tmp_path / "linear_bundle"

    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear", lambda _cfg, **_kwargs: _runtime_linear_result()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-linear",
            "--config",
            str(path),
            "--out",
            str(out_base),
        ],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "saved" in out
    assert (tmp_path / "linear_bundle.summary.json").exists()
    assert (tmp_path / "linear_bundle.timeseries.csv").exists()


def test_cmd_scan_runtime_linear_branches(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    scan = type(
        "Scan",
        (),
        {
            "ky": np.array([0.1]),
            "gamma": np.array([0.2]),
            "omega": np.array([-0.3]),
        },
    )()
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg, {"scan": {"ky": [0.1]}, "fit": {}}),
    )
    monkeypatch.setattr("gkx.cli.run_runtime_scan", lambda *args, **kwargs: scan)
    args = argparse.Namespace(
        config="case.toml",
        ky_values="0.1",
        Nl=None,
        Nm=None,
        solver=None,
        fit_signal=None,
        method=None,
        dt=None,
        steps=None,
        sample_stride=None,
        batch_ky=True,
        progress=False,
        no_progress=True,
    )
    assert _cmd_scan_runtime_linear(args) == 0
    assert "ky=0.1000 gamma=0.200000 omega=-0.300000" in capsys.readouterr().out

    args.ky_values = None
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg, {"scan": {}, "fit": {}}),
    )
    with pytest.raises(ValueError):
        _cmd_scan_runtime_linear(args)


def test_cmd_scan_runtime_linear_writes_quasilinear_spectrum(
    monkeypatch, capsys
) -> None:
    cfg = RuntimeConfig()
    scan = type(
        "Scan",
        (),
        {
            "ky": np.array([0.1]),
            "gamma": np.array([0.2]),
            "omega": np.array([-0.3]),
            "quasilinear": ({"ky": 0.1, "heat_flux_weight_total": 1.0},),
        },
    )()
    captured: dict[str, object] = {}

    def _fake_run_runtime_scan(cfg_in, *_args, **kwargs):
        captured["quasilinear"] = cfg_in.quasilinear
        captured["kwargs"] = kwargs
        return scan

    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg, {"scan": {"ky": [0.1]}, "fit": {}}),
    )
    monkeypatch.setattr("gkx.cli.run_runtime_scan", _fake_run_runtime_scan)
    monkeypatch.setattr(
        "gkx.cli.write_runtime_linear_scan_artifacts",
        lambda *_args, **_kwargs: {
            "summary": "scan.summary.json",
            "scan": "scan.csv",
            "quasilinear_spectrum": "scan.ql.csv",
        },
    )
    args = argparse.Namespace(
        config="case.toml",
        ky_values="0.1",
        Nl=None,
        Nm=None,
        solver=None,
        fit_signal=None,
        method=None,
        dt=None,
        steps=None,
        sample_stride=None,
        batch_ky=False,
        workers=3,
        parallel_executor="thread",
        progress=False,
        no_progress=True,
        out="scan_bundle",
        quasilinear=True,
        ql_mode="weights",
        ql_saturation_rule=None,
        ql_csat=None,
        ql_normalization=None,
        ql_output=None,
    )
    assert _cmd_scan_runtime_linear(args) == 0
    assert captured["quasilinear"].enabled is True
    assert captured["kwargs"]["workers"] == 3
    assert captured["kwargs"]["parallel_executor"] == "thread"
    assert "saved scan.ql.csv" in capsys.readouterr().out


def test_cmd_run_runtime_nonlinear_branches(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    cfg = RuntimeConfig()
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0, 0.0]),
        omega_t=np.asarray([0.0, 0.0]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.0, 0.0]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.0, 3.2]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        ky_selected=0.2,
        kx_selected=0.0,
    )
    paths = {
        "summary": "a",
        "diagnostics": "b",
        "state": "c",
        "out": "d",
        "big": "e",
        "restart": "f",
    }
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg, {"run": {"steps": 5}}),
    )
    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        lambda *args, **kwargs: (result, paths),
    )
    args = argparse.Namespace(
        config="case.toml",
        init_file=str(tmp_path / "seed.bin"),
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        diagnostics_stride=None,
        diagnostics=False,
        no_diagnostics=False,
        laguerre_mode=None,
        progress=False,
        no_progress=False,
        out="out.nc",
    )
    assert _cmd_run_runtime_nonlinear(args) == 0
    out = capsys.readouterr().out
    assert "starting runtime nonlinear run" in out
    assert "saved a" in out and "saved f" in out

    no_diag_result = RuntimeNonlinearResult(
        t=np.asarray([0.1]), diagnostics=None, ky_selected=0.2, kx_selected=0.0
    )
    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        lambda *args, **kwargs: (no_diag_result, {}),
    )
    args.no_diagnostics = True
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert "nonlinear run completed" in capsys.readouterr().out


def test_cmd_run_runtime_linear_prints_optional_artifact_paths(
    monkeypatch, capsys
) -> None:
    cfg = RuntimeConfig()
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {}})
    )
    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear",
        lambda *_args, **_kwargs: RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        ),
    )
    monkeypatch.setattr(
        "gkx.cli.write_runtime_linear_artifacts",
        lambda *_args, **_kwargs: {
            "summary": "sum.json",
            "timeseries": "diag.csv",
            "eigenfunction": "eig.csv",
            "state": "state.npy",
        },
    )
    args = argparse.Namespace(
        config="case.toml",
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        solver=None,
        fit_signal=None,
        progress=False,
        no_progress=False,
        out="bundle",
    )
    assert _cmd_run_runtime_linear(args) == 0
    out = capsys.readouterr().out
    assert "saved eig.csv" in out
    assert "saved state.npy" in out


def test_cmd_run_runtime_linear_applies_quasilinear_flags(monkeypatch, capsys) -> None:
    cfg = RuntimeConfig()
    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg_in, **kwargs):
        captured["quasilinear"] = cfg_in.quasilinear
        captured["kwargs"] = kwargs
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            quasilinear={
                "species": ["ion"],
                "heat_flux_weight_species": [1.0],
                "particle_flux_weight_species": [0.0],
            },
        )

    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {}})
    )
    monkeypatch.setattr("gkx.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        "gkx.cli.write_quasilinear_artifacts",
        lambda *_args, **_kwargs: {
            "quasilinear_summary": "ql.json",
            "quasilinear_species": "ql.csv",
        },
    )

    args = argparse.Namespace(
        config="case.toml",
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        solver=None,
        fit_signal=None,
        progress=False,
        no_progress=False,
        out=None,
        vmec_file=None,
        geometry_file=None,
        quasilinear=True,
        ql_mode="saturated",
        ql_saturation_rule="mixing_length",
        ql_csat=0.5,
        ql_normalization="field_energy",
        ql_output="ql_out",
    )
    assert _cmd_run_runtime_linear(args) == 0
    ql_cfg = captured["quasilinear"]
    assert ql_cfg.enabled is True
    assert ql_cfg.mode == "saturated"
    assert ql_cfg.saturation_rule == "mixing_length"
    assert ql_cfg.csat == pytest.approx(0.5)
    assert ql_cfg.amplitude_normalization == "field_energy"
    assert "saved ql.json" in capsys.readouterr().out


def test_cmd_run_runtime_nonlinear_fixed_dt_and_explicit_diagnostics(
    monkeypatch, capsys
) -> None:
    cfg = RuntimeConfig()
    cfg = RuntimeConfig(
        time=type(cfg.time)(
            **{**cfg.time.__dict__, "fixed_dt": True, "t_max": 1.0, "dt": 0.2}
        )
    )
    captured: dict[str, object] = {}
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1]), diagnostics=None, ky_selected=0.2, kx_selected=0.0
    )

    def _runner(*_args, **kwargs):
        captured.update(kwargs)
        return result, {}

    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (cfg, {"run": {}})
    )
    monkeypatch.setattr("gkx.cli.run_runtime_nonlinear_with_artifacts", _runner)
    args = argparse.Namespace(
        config="case.toml",
        init_file=None,
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        diagnostics_stride=None,
        diagnostics=True,
        no_diagnostics=False,
        laguerre_mode=None,
        progress=False,
        no_progress=False,
        out=None,
    )
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert captured["steps"] == 5
    assert captured["diagnostics"] is True


def test_cli_run_runtime_linear_uses_toml_output_path(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/from_toml"
"""
    path = tmp_path / "runtime_cli_linear_toml_out.toml"
    path.write_text(cfg, encoding="utf-8")

    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear", lambda _cfg, **_kwargs: _runtime_linear_result()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", "run-runtime-linear", "--config", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {tmp_path / 'artifacts' / 'from_toml.summary.json'}" in out
    assert (tmp_path / "artifacts" / "from_toml.summary.json").exists()
    assert (tmp_path / "artifacts" / "from_toml.timeseries.csv").exists()


def test_cli_run_runtime_linear_cli_out_overrides_toml_output_path(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/from_toml"
"""
    path = tmp_path / "runtime_cli_linear_toml_override.toml"
    path.write_text(cfg, encoding="utf-8")
    out_base = tmp_path / "cli_override"

    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear", lambda _cfg, **_kwargs: _runtime_linear_result()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-linear",
            "--config",
            str(path),
            "--out",
            str(out_base),
        ],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert f"saved {out_base}.summary.json" in out
    assert (tmp_path / "cli_override.summary.json").exists()
    assert not (tmp_path / "artifacts" / "from_toml.summary.json").exists()


def test_cli_direct_config_shorthand_uses_toml_output_path(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[output]
path = "artifacts/direct_shorthand"
"""
    path = tmp_path / "runtime_cli_direct.toml"
    path.write_text(cfg, encoding="utf-8")

    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear", lambda _cfg, **_kwargs: _runtime_linear_result()
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", str(path)],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "starting runtime linear run" in out
    assert f"saved {tmp_path / 'artifacts' / 'direct_shorthand.summary.json'}" in out
    assert (tmp_path / "artifacts" / "direct_shorthand.summary.json").exists()


def test_cli_run_runtime_nonlinear(capsys, monkeypatch, tmp_path: Path):
    """The unified runtime nonlinear command should run a tiny configuration."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", "run-runtime-nonlinear", "--config", str(path), "--steps", "3"],
    )
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear" in out


def test_cli_direct_config_shorthand_runs_nonlinear(
    capsys, monkeypatch, tmp_path: Path
):
    """Direct ``gkx path/to/config.toml`` should dispatch nonlinear configs."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""
    path = tmp_path / "runtime_cli_shorthand.toml"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gkx", str(path), "--steps", "3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear" in out


def test_cli_direct_config_shorthand_accepts_no_progress(
    monkeypatch, tmp_path: Path
) -> None:
    """Direct config shorthand should forward progress flags to nonlinear runtime runs."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0
"""
    path = tmp_path / "runtime_cli_shorthand_progress.toml"
    path.write_text(cfg, encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(_cfg, **kwargs):
        captured.update(kwargs)
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.1]),
            dt_mean=np.asarray(0.1),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([2.0]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([3.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([3.0]),
            heat_flux_species_t=None,
            particle_flux_species_t=None,
            phi_mode_t=None,
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        _fake_run_runtime_nonlinear_with_artifacts,
    )
    monkeypatch.setattr(
        sys, "argv", ["gkx", str(path), "--steps", "3", "--no-progress"]
    )
    code = main()
    assert code == 0
    assert captured["show_progress"] is False


def test_cli_run_runtime_nonlinear_outputs_species_flux_columns(
    capsys, monkeypatch, tmp_path: Path
):
    """Nonlinear CSV output should include per-species flux diagnostics when available."""
    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.1
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear_species.toml"
    out_path = tmp_path / "diag.csv"
    path.write_text(cfg, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-nonlinear",
            "--config",
            str(path),
            "--steps",
            "3",
            "--out",
            str(out_path),
        ],
    )
    code = main()
    _out = capsys.readouterr().out
    assert code == 0
    header = out_path.read_text(encoding="utf-8").splitlines()[0]
    assert "heat_flux_s0" in header
    assert "particle_flux_s0" in header


def test_cli_run_runtime_nonlinear_keeps_adaptive_steps_none(
    capsys, monkeypatch, tmp_path: Path
):
    """Adaptive nonlinear executable runs should keep ``steps=None`` unless explicitly set."""

    cfg = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"
fixed_dt = false

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
"""
    path = tmp_path / "runtime_cli_nonlinear_adaptive.toml"
    path.write_text(cfg, encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(cfg, **kwargs):
        captured["steps"] = kwargs.get("steps")
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.01]),
            dt_mean=np.asarray(0.01),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([0.5]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([0.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([1.5]),
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        _fake_run_runtime_nonlinear_with_artifacts,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", "run-runtime-nonlinear", "--config", str(path)],
    )

    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear:" in out
    assert "t=0.1" in out
    assert captured["steps"] is None


_RUNTIME_LINEAR_TOML_WITH_VMEC = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778
vmec_file = "toml_wout.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_LINEAR_TOML_IMPORTED_GEOMETRY = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
model = "vmec-eik"
geometry_file = "from_config.eik.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_LINEAR_TOML_VMEC_MODEL = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
model = "vmec"
vmec_file = "wout_from_config.nc"
geometry_file = "generated_from_config.eik.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"
"""


_RUNTIME_NONLINEAR_TOML_MIN = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 6
Nz = 16
Lx = 62.8
Ly = 62.8
boundary = "periodic"

[time]
t_max = 0.2
dt = 0.01
method = "rk2"

[geometry]
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778
vmec_file = "toml_wout.nc"

[init]
init_field = "density"
init_amp = 1e-8
gaussian_init = false

[physics]
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
tau_e = 1.0
nonlinear = true

[terms]
nonlinear = 1.0

[normalization]
contract = "cyclone"
diagnostic_norm = "none"

[run]
ky = 0.2
Nl = 3
Nm = 4
steps = 1
"""


def test_cli_run_runtime_linear_cli_vmec_file_resolves_against_cwd(
    monkeypatch, tmp_path: Path
) -> None:
    """--vmec-file override lands on cfg.geometry.vmec_file, resolved relative to shell cwd."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_WITH_VMEC, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "sub").mkdir(parents=True)
    (shell_cwd / "sub" / "cli_wout.nc").write_text("", encoding="utf-8")
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["vmec_file"] = cfg.geometry.vmec_file
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("gkx.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--vmec-file",
            "sub/cli_wout.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str((shell_cwd / "sub" / "cli_wout.nc").resolve())
    assert captured["vmec_file"] == expected_cwd_resolved
    # Regression guard: must NOT resolve against the config file's parent directory.
    assert captured["vmec_file"] != str((config_dir / "sub" / "cli_wout.nc"))


def test_cli_run_runtime_nonlinear_init_file_expands_home(
    monkeypatch, tmp_path: Path
) -> None:
    """--init-file with leading ~ should expand to $HOME (regression guard for prior bypass)."""
    config_path = tmp_path / "runtime.toml"
    config_path.write_text(_RUNTIME_NONLINEAR_TOML_MIN, encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(cfg, **_kwargs):
        captured["init_file"] = cfg.init.init_file
        diag = SimulationDiagnostics(
            t=np.asarray([0.1]),
            dt_t=np.asarray([0.01]),
            dt_mean=np.asarray(0.01),
            gamma_t=np.asarray([0.0]),
            omega_t=np.asarray([0.0]),
            Wg_t=np.asarray([1.0]),
            Wphi_t=np.asarray([0.5]),
            Wapar_t=np.asarray([0.0]),
            heat_flux_t=np.asarray([0.0]),
            particle_flux_t=np.asarray([0.0]),
            energy_t=np.asarray([1.5]),
        )
        return (
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {},
        )

    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        _fake_run_runtime_nonlinear_with_artifacts,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-nonlinear",
            "--config",
            str(config_path),
            "--init-file",
            "~/gkx_test/g_state.h5",
            "--steps",
            "1",
        ],
    )
    assert main() == 0
    init_file = captured["init_file"]
    assert init_file is not None
    assert init_file.startswith(os.path.expanduser("~"))
    assert init_file.endswith("g_state.h5")
    assert "~" not in init_file


def test_cli_run_runtime_linear_cli_geometry_file_resolves_against_cwd_for_imported_geometry(
    monkeypatch, tmp_path: Path
) -> None:
    """--geometry-file overrides [geometry].geometry_file (cwd-resolved) without touching [geometry].model."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_IMPORTED_GEOMETRY, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "sub").mkdir(parents=True)
    (shell_cwd / "sub" / "cli.eik.nc").write_text("", encoding="utf-8")
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["geometry_file"] = cfg.geometry.geometry_file
        captured["model"] = cfg.geometry.model
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("gkx.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--geometry-file",
            "sub/cli.eik.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str((shell_cwd / "sub" / "cli.eik.nc").resolve())
    assert captured["geometry_file"] == expected_cwd_resolved
    # Override resolves against shell cwd, not the config file's parent directory.
    assert captured["geometry_file"] != str(config_dir / "sub" / "cli.eik.nc")
    # --geometry-file must not change [geometry].model: imported-geometry stays imported.
    assert captured["model"] == "vmec-eik"


def test_cli_run_runtime_linear_cli_geometry_file_does_not_change_vmec_model(
    monkeypatch, tmp_path: Path
) -> None:
    """--geometry-file on a model="vmec" TOML must not flip the model to imported-EIK."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "runtime.toml"
    config_path.write_text(_RUNTIME_LINEAR_TOML_VMEC_MODEL, encoding="utf-8")

    shell_cwd = tmp_path / "shellwd"
    (shell_cwd / "cache").mkdir(parents=True)
    monkeypatch.chdir(shell_cwd)

    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(cfg, **_kwargs):
        captured["geometry_file"] = cfg.geometry.geometry_file
        captured["vmec_file"] = cfg.geometry.vmec_file
        captured["model"] = cfg.geometry.model
        return RuntimeLinearResult(
            ky=0.2,
            gamma=0.3,
            omega=-0.4,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=1),
            t=np.asarray([0.1, 0.2]),
            signal=np.asarray([1.0, 2.0]),
        )

    monkeypatch.setattr("gkx.cli.run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gkx",
            "run-runtime-linear",
            "--config",
            str(config_path),
            "--geometry-file",
            "cache/generated_cli.eik.nc",
        ],
    )
    assert main() == 0
    expected_cwd_resolved = str(
        (shell_cwd / "cache" / "generated_cli.eik.nc").resolve()
    )
    assert captured["geometry_file"] == expected_cwd_resolved
    # --geometry-file must not promote a VMEC-backed run into imported-geometry mode.
    assert captured["model"] == "vmec"


def _record_auto_plot(monkeypatch) -> list[tuple[str, dict[str, str]]]:
    """Capture what the executable asks the figure writer to draw."""

    calls: list[tuple[str, dict[str, str]]] = []

    def _fake(kind, paths, **_kwargs):
        calls.append((kind, dict(paths)))
        return [f"{kind}.png"]

    monkeypatch.setattr("gkx.artifacts.run_figures.auto_plot_saved_run", _fake)
    return calls


def _nonlinear_namespace(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        config="case.toml",
        init_file=None,
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        diagnostics_stride=None,
        diagnostics=False,
        no_diagnostics=False,
        laguerre_mode=None,
        progress=False,
        no_progress=False,
        no_plots=False,
        out="case.out.nc",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _linear_namespace(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        config="case.toml",
        ky=None,
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        solver=None,
        fit_signal=None,
        progress=False,
        no_progress=False,
        no_plots=False,
        out="bundle",
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _stub_nonlinear_run(monkeypatch, paths: dict[str, str], cfg=None) -> None:
    diag = SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]),
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0, 0.0]),
        omega_t=np.asarray([0.0, 0.0]),
        Wg_t=np.asarray([1.0, 1.1]),
        Wphi_t=np.asarray([2.0, 2.1]),
        Wapar_t=np.asarray([0.0, 0.0]),
        heat_flux_t=np.asarray([3.0, 3.1]),
        particle_flux_t=np.asarray([4.0, 4.1]),
        energy_t=np.asarray([3.0, 3.2]),
        phi_mode_t=None,
    )
    result = RuntimeNonlinearResult(
        t=np.asarray([0.1, 0.2]),
        diagnostics=diag,
        ky_selected=0.2,
        kx_selected=0.0,
    )
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg if cfg is not None else RuntimeConfig(), {"run": {}}),
    )
    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        lambda *_args, **_kwargs: (result, paths),
    )


def _stub_linear_run(monkeypatch, paths: dict[str, str], cfg=None) -> None:
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg if cfg is not None else RuntimeConfig(), {"run": {}}),
    )
    monkeypatch.setattr(
        "gkx.cli.run_runtime_linear",
        lambda *_args, **_kwargs: _runtime_linear_result(),
    )
    monkeypatch.setattr(
        "gkx.cli.write_runtime_linear_artifacts", lambda *_a, **_k: paths
    )


def test_cli_nonlinear_run_auto_plots_its_saved_bundle(monkeypatch, capsys) -> None:
    """A finished nonlinear run draws its figures without being asked."""

    calls = _record_auto_plot(monkeypatch)
    paths = {"summary": "case.summary.json", "out": "case.out.nc"}
    _stub_nonlinear_run(monkeypatch, paths)

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace()) == 0

    assert calls == [("nonlinear", paths)]
    assert "saved nonlinear.png" in capsys.readouterr().out


def test_cli_linear_run_auto_plots_its_saved_bundle(monkeypatch, capsys) -> None:
    calls = _record_auto_plot(monkeypatch)
    paths = {"summary": "bundle.summary.json", "timeseries": "bundle.timeseries.csv"}
    _stub_linear_run(monkeypatch, paths)

    assert _cmd_run_runtime_linear(_linear_namespace()) == 0

    assert calls == [("linear", paths)]
    assert "saved linear.png" in capsys.readouterr().out


def test_cli_linear_scan_auto_plots_its_saved_scan(monkeypatch, capsys) -> None:
    calls = _record_auto_plot(monkeypatch)
    paths = {"summary": "scan.summary.json", "scan": "scan.scan.csv"}
    scan = type(
        "Scan",
        (),
        {
            "ky": np.array([0.2, 0.3]),
            "gamma": np.array([0.05, 0.08]),
            "omega": np.array([0.25, 0.31]),
        },
    )()
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml", lambda _path: (RuntimeConfig(), {"scan": {}})
    )
    monkeypatch.setattr("gkx.cli.run_runtime_scan", lambda *_a, **_k: scan)
    monkeypatch.setattr(
        "gkx.cli.write_runtime_linear_scan_artifacts", lambda *_a, **_k: paths
    )
    args = argparse.Namespace(
        config="case.toml",
        ky_values="0.2,0.3",
        Nl=None,
        Nm=None,
        dt=None,
        steps=None,
        method=None,
        sample_stride=None,
        solver=None,
        fit_signal=None,
        batch_ky=False,
        workers=1,
        parallel_executor="thread",
        progress=False,
        no_progress=False,
        quasilinear=False,
        ql_mode=None,
        ql_saturation_rule=None,
        ql_csat=None,
        ql_normalization=None,
        ql_output=None,
        no_plots=False,
        out="scan",
    )

    assert _cmd_scan_runtime_linear(args) == 0

    assert calls == [("linear_scan", paths)]
    assert "saved linear_scan.png" in capsys.readouterr().out


def test_cli_no_plots_flag_suppresses_the_figure_set(monkeypatch) -> None:
    calls = _record_auto_plot(monkeypatch)
    _stub_nonlinear_run(monkeypatch, {"summary": "s.json", "out": "case.out.nc"})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(no_plots=True)) == 0

    assert calls == []


def test_cli_output_plots_false_suppresses_the_figure_set(monkeypatch) -> None:
    """``[output] plots = false`` is the configuration form of ``--no-plots``."""

    from dataclasses import replace

    cfg = RuntimeConfig()
    cfg = replace(cfg, output=replace(cfg.output, plots=False))
    calls = _record_auto_plot(monkeypatch)
    _stub_nonlinear_run(monkeypatch, {"summary": "s.json", "out": "c.out.nc"}, cfg=cfg)

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace()) == 0

    assert calls == []


def test_cli_output_plots_default_is_on() -> None:
    assert RuntimeConfig().output.plots is True


def test_cli_plot_failure_never_fails_a_completed_run(monkeypatch, capsys) -> None:
    """A broken plotting stack costs the figures, never the simulation."""

    def _explode(*_args, **_kwargs):
        raise RuntimeError("no usable matplotlib backend")

    monkeypatch.setattr("gkx.artifacts.run_figures.auto_plot_saved_run", _explode)
    _stub_nonlinear_run(monkeypatch, {"summary": "s.json", "out": "case.out.nc"})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace()) == 0

    out = capsys.readouterr().out
    assert "no usable matplotlib backend" in out
    assert "saved case.out.nc" in out


def _write_grouped_out_nc(path: Path, *, code: str = "gkx", nt: int = 6) -> Path:
    """Write a minimal bundle in the grouped layout GKX and GX both use."""

    netcdf4 = pytest.importorskip("netCDF4")
    nky, nkx, ns = 4, 3, 1
    t = np.linspace(0.5, 6.0, nt)
    ky = np.linspace(0.0, 0.6, nky)
    kx = np.linspace(-0.4, 0.4, nkx)
    with netcdf4.Dataset(path, "w") as root:
        for name, size in (("time", nt), ("s", ns), ("ky", nky), ("kx", nkx)):
            root.createDimension(name, size)
        if code == "gkx":
            info = root.createVariable("code_info", "i4", ())
            info[:] = np.int32(1)
            info.setncattr("value", "gkx")
        else:
            root.setncattr("Title", "GX simulation data")
            info = root.createVariable("code_info", "i4", ())
            info[:] = np.int32(1)
            info.setncattr("Hash", "deadbeef")
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("ky", "f4", ("ky",))[:] = ky
        grids.createVariable("kx", "f4", ("kx",))[:] = kx
        diag = root.createGroup("Diagnostics")
        history = 0.5 + 0.1 * np.outer(t, np.arange(1, ns + 1))
        for name in ("Wg_st", "Wphi_st", "Wapar_st", "HeatFlux_st", "ParticleFlux_st"):
            diag.createVariable(name, "f4", ("time", "s"))[:, :] = history
        diag.createVariable("Phi2_t", "f4", ("time",))[:] = np.exp(0.1 * t)
        diag.createVariable("HeatFlux_kyst", "f4", ("time", "s", "ky"))[:, :, :] = (
            np.broadcast_to((ky * np.exp(-2.0 * ky))[None, None, :], (nt, ns, nky))
        )
        diag.createVariable("HeatFlux_kxst", "f4", ("time", "s", "kx"))[:, :, :] = (
            np.broadcast_to(np.exp(-np.abs(kx))[None, None, :], (nt, ns, nkx))
        )
        phi2_kxky = np.broadcast_to(
            (np.exp(-2.0 * ky)[:, None] * np.exp(-np.abs(kx))[None, :])[None, :, :],
            (nt, nky, nkx),
        )
        diag.createVariable("Phi2_kxkyt", "f4", ("time", "ky", "kx"))[:, :, :] = (
            phi2_kxky
        )
    return path


def test_cli_nonlinear_run_writes_the_real_figure_set(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """The default figure set really lands on disk beside a NetCDF bundle."""

    bundle = _write_grouped_out_nc(tmp_path / "case.out.nc")
    _stub_nonlinear_run(monkeypatch, {"out": str(bundle)})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    out = capsys.readouterr().out
    for suffix in ("flux_time", "flux_spectra", "phi2_spectra", "summary"):
        figure = tmp_path / f"case.{suffix}.png"
        assert figure.exists(), suffix
        assert f"saved {figure}" in out


def _write_final_field_companion(base: Path, *, nx: int = 8, ny: int = 6, nz: int = 4):
    """Write the ``*.big.nc`` companion carrying a run's final real-space fields."""

    netcdf4 = pytest.importorskip("netCDF4")
    rng = np.random.default_rng(11)
    phi_yxz = rng.normal(size=(ny, nx, nz)) * 1e-3
    path = Path(f"{base}.big.nc")
    with netcdf4.Dataset(path, "w") as root:
        for name, size in (("time", 1), ("x", nx), ("y", ny), ("theta", nz)):
            root.createDimension(name, size)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = np.asarray([6.0])
        grids.createVariable("x", "f4", ("x",))[:] = np.linspace(
            0.0, 40.0, nx, endpoint=False
        )
        grids.createVariable("y", "f4", ("y",))[:] = np.linspace(
            0.0, 30.0, ny, endpoint=False
        )
        geom = root.createGroup("Geometry")
        for name, value in (("q", 1.4), ("rmaj", 3.0), ("aminor", 0.5), ("nfp", 3)):
            geom.createVariable(name, "f4", ())[:] = np.float32(value)
        diag = root.createGroup("Diagnostics")
        diag.createVariable("PhiXY", "f4", ("time", "y", "x", "theta"))[0, ...] = (
            phi_yxz
        )
    return path


def test_cli_nonlinear_run_includes_the_flux_tube_in_the_standard_set(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A nonlinear run that saved its fields also draws the geometry it ran on."""

    bundle = _write_grouped_out_nc(tmp_path / "tube.out.nc")
    _write_final_field_companion(tmp_path / "tube")
    _stub_nonlinear_run(monkeypatch, {"out": str(bundle)})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    out = capsys.readouterr().out
    for suffix in ("flux_tube_3d", "snapshot_xy", "summary"):
        figure = tmp_path / f"tube.{suffix}.png"
        assert figure.exists(), suffix
        assert f"saved {figure}" in out


def test_cli_nonlinear_run_without_saved_fields_skips_the_flux_tube(
    monkeypatch, tmp_path: Path
) -> None:
    """No ``*.big.nc`` is a fact about the output form, not a failure to warn about."""

    bundle = _write_grouped_out_nc(tmp_path / "nofields.out.nc")
    _stub_nonlinear_run(monkeypatch, {"out": str(bundle)})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    assert not (tmp_path / "nofields.flux_tube_3d.png").exists()
    assert not (tmp_path / "nofields.snapshot_xy.png").exists()
    assert (tmp_path / "nofields.summary.png").exists()


def test_cli_global_plot_reproduces_the_whole_gkx_figure_set(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """--plot on a saved GKX bundle rebuilds the set, over the recorded window."""

    bundle = _write_grouped_out_nc(tmp_path / "again.out.nc")
    _write_final_field_companion(tmp_path / "again")
    (tmp_path / "again.summary.json").write_text(
        '{"kind":"nonlinear","saturation":{"window_tmin":3.0,"window_tmax":6.0}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", str(bundle)])
    assert main() == 0

    out = capsys.readouterr().out
    for suffix in ("plot", "flux_time", "flux_spectra", "flux_tube_3d", "summary"):
        assert (tmp_path / f"again.{suffix}.png").exists(), suffix
        assert f"saved {tmp_path / f'again.{suffix}.png'}" in out


def test_cli_nonlinear_csv_sidecar_degrades_to_the_time_traces(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A CSV sidecar has no spectra, so it gets Q(t)/Gamma(t) and no crash."""

    base = tmp_path / "nl_case"
    (tmp_path / "nl_case.summary.json").write_text(
        '{"kind":"nonlinear"}', encoding="utf-8"
    )
    (tmp_path / "nl_case.diagnostics.csv").write_text(
        "t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux\n"
        "0.1,0.1,0.01,-0.02,1.0,2.0,0.0,3.0,0.4,0.02\n"
        "0.2,0.1,0.02,-0.03,1.1,2.1,0.0,3.2,0.5,0.03\n"
        "0.3,0.1,0.02,-0.03,1.2,2.2,0.0,3.4,0.6,0.04\n",
        encoding="utf-8",
    )
    _stub_nonlinear_run(
        monkeypatch,
        {
            "summary": str(tmp_path / "nl_case.summary.json"),
            "diagnostics": str(tmp_path / "nl_case.diagnostics.csv"),
        },
    )

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(base))) == 0

    assert (tmp_path / "nl_case.flux_time.png").exists()
    assert not (tmp_path / "nl_case.flux_spectra.png").exists()
    assert not (tmp_path / "nl_case.phi2_spectra.png").exists()
    assert "saved" in capsys.readouterr().out


def test_cli_nonlinear_run_shades_a_measured_average_window(
    monkeypatch, tmp_path: Path
) -> None:
    """A summary that records where it averaged gets that window shaded."""

    from gkx.artifacts.run_figures import measured_average_window

    bundle = _write_grouped_out_nc(tmp_path / "win.out.nc")
    summary = tmp_path / "win.summary.json"
    summary.write_text('{"kind":"nonlinear","average_window":[3.0,6.0]}', "utf-8")
    captured: dict[str, object] = {}

    def _spy(source, *, window=None, title="", out=None):
        captured["window"] = window
        import matplotlib.pyplot as plt

        return plt.subplots(1, 1)

    monkeypatch.setattr(
        "gkx.artifacts.transport_figures.heat_flux_time_figure", _spy, raising=True
    )
    _stub_nonlinear_run(monkeypatch, {"out": str(bundle), "summary": str(summary)})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    assert captured["window"] == (3.0, 6.0)
    assert measured_average_window(
        {"fit_window_tmin": 1.0, "fit_window_tmax": 2.0}
    ) == (
        1.0,
        2.0,
    )
    assert measured_average_window({}) is None
    assert measured_average_window({"average_window": [4.0, 1.0]}) is None
    # The shape gkx.diagnostics.saturation actually writes into the summary.
    assert measured_average_window(
        {"saturation": {"window_tmin": 3.05, "window_tmax": 6.0}}
    ) == (3.05, 6.0)


def test_cli_nonlinear_run_does_not_shade_a_rejected_window(
    monkeypatch, tmp_path: Path
) -> None:
    """A capped run gets an explicit diagnostic view, not a saturation claim."""

    bundle = _write_grouped_out_nc(tmp_path / "capped.out.nc")
    summary = tmp_path / "capped.summary.json"
    summary.write_text(
        '{"kind":"nonlinear","saturation":'
        '{"window_tmin":3.0,"window_tmax":6.0,"saturated":false}}',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _spy(source, *, window=None, title="", out=None):
        captured.update(window=window, title=title)
        import matplotlib.pyplot as plt

        return plt.subplots(1, 1)

    def _warnings(source, *, window=None):
        captured["warning_window"] = window
        return ()

    monkeypatch.setattr(
        "gkx.artifacts.transport_figures.heat_flux_time_figure", _spy, raising=True
    )
    monkeypatch.setattr(
        "gkx.artifacts.transport_figures.spectrum_cutoff_warnings",
        _warnings,
        raising=True,
    )
    _stub_nonlinear_run(monkeypatch, {"out": str(bundle), "summary": str(summary)})

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    assert captured["window"] is None
    assert captured["warning_window"] is None
    assert "not saturated; diagnostic only" in str(captured["title"])


def test_cli_global_plot_renders_a_gx_netcdf_bundle(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """--plot accepts GX output so a cross-code check is one command."""

    from gkx.artifacts.gx_output import is_gx_output

    gx_bundle = _write_grouped_out_nc(tmp_path / "gx_case.out.nc", code="gx")
    gkx_bundle = _write_grouped_out_nc(tmp_path / "gkx_case.out.nc", code="gkx")
    assert is_gx_output(gx_bundle) is True
    assert is_gx_output(gkx_bundle) is False

    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", str(gx_bundle)])
    assert main() == 0

    rendered = tmp_path / "gx_case.plot.png"
    assert rendered.exists()
    assert f"saved {rendered}" in capsys.readouterr().out


def test_cli_global_plot_titles_gx_data_as_gx(tmp_path: Path) -> None:
    """The figure must say GX so a lifted panel cannot be read as GKX."""

    import matplotlib.pyplot as plt

    from gkx.artifacts.gx_output import gx_summary_figure

    gx_bundle = _write_grouped_out_nc(tmp_path / "gx_titled.out.nc", code="gx")
    fig, axes = gx_summary_figure(gx_bundle)
    try:
        assert "GX" in axes[0].get_title(loc="left")
    finally:
        plt.close(fig)


def test_compilation_cache_directory_honours_the_environment(tmp_path: Path) -> None:
    from gkx.utils import compilation_cache as cache

    override = tmp_path / "elsewhere"
    assert (
        cache.compilation_cache_directory({cache.DIRECTORY_ENV_VAR: str(override)})
        == override
    )
    # No override: the repository-local .cache/gkx tree, matching where
    # generated *.eik.nc geometry already lives.
    assert cache.compilation_cache_directory({}).parts[-3:] == (
        ".cache",
        "gkx",
        "jax",
    )


def test_compilation_cache_can_be_switched_off(tmp_path: Path) -> None:
    from gkx.utils import compilation_cache as cache

    assert cache.compilation_cache_enabled({}) is True
    for value in ("0", "off", "false", "no"):
        assert cache.compilation_cache_enabled({cache.ENABLE_ENV_VAR: value}) is False
    assert (
        cache.enable_persistent_compilation_cache(
            {cache.ENABLE_ENV_VAR: "0", cache.DIRECTORY_ENV_VAR: str(tmp_path)}
        )
        is None
    )


def test_compilation_cache_is_namespaced_by_jax_version(tmp_path: Path) -> None:
    """A JAX upgrade must start a fresh directory, never reuse old executables."""

    import jax

    from gkx.utils import compilation_cache as cache

    previous = jax.config.values.get("jax_compilation_cache_dir")
    try:
        directory = cache.enable_persistent_compilation_cache(
            {cache.DIRECTORY_ENV_VAR: str(tmp_path)}
        )
        assert directory == tmp_path / f"jax-{jax.__version__}"
        assert directory.is_dir()
        assert jax.config.values["jax_compilation_cache_dir"] == str(directory)
    finally:
        jax.config.update("jax_compilation_cache_dir", previous)


def test_cli_main_installs_the_compilation_cache(monkeypatch, tmp_path: Path) -> None:
    installed: list[object] = []
    monkeypatch.setattr(
        cli, "enable_persistent_compilation_cache", lambda: installed.append(True)
    )
    rendered = tmp_path / "rendered.png"
    monkeypatch.setattr("gkx.cli.plot_saved_output", lambda path, out=None: rendered)
    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", "case.summary.json"])

    assert main() == 0
    assert installed == [True]


def _write_tiny_wout(path: Path) -> Path:
    """Write a minimal NetCDF file bearing the VMEC/VMEX wout signature."""

    from netCDF4 import Dataset

    with Dataset(path, "w") as ds:
        ds.createDimension("mn_mode", 4)
        ds.createDimension("radius", 3)
        for name in ("xm", "xn"):
            ds.createVariable(name, "f8", ("mn_mode",))[:] = 0.0
        for name in ("rmnc", "zmns"):
            ds.createVariable(name, "f8", ("radius", "mn_mode"))[:] = 0.0
    return path


def _fake_nonlinear_result() -> RuntimeNonlinearResult:
    diag = SimulationDiagnostics(
        t=np.asarray([0.1]),
        dt_t=np.asarray([0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([1.0]),
        Wphi_t=np.asarray([0.5]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([0.0]),
        particle_flux_t=np.asarray([0.0]),
        energy_t=np.asarray([1.5]),
    )
    return RuntimeNonlinearResult(
        t=np.asarray([0.1]), diagnostics=diag, ky_selected=0.2, kx_selected=0.0
    )


def test_wout_sniffing_detects_signature_and_rejects_others(tmp_path: Path) -> None:
    from gkx.workflows.runtime import wout as runtime_wout

    wout = _write_tiny_wout(tmp_path / "wout_tiny.nc")
    assert runtime_wout.is_wout_file(wout) is True

    from netCDF4 import Dataset

    plain = tmp_path / "plain.nc"
    with Dataset(plain, "w") as ds:
        ds.createDimension("x", 2)
        ds.createVariable("data", "f8", ("x",))[:] = 0.0
    assert runtime_wout.is_wout_file(plain) is False

    toml_file = tmp_path / "case.toml"
    toml_file.write_text("[physics]\n", encoding="utf-8")
    assert runtime_wout.is_wout_file(toml_file) is False
    assert runtime_wout.is_wout_file(tmp_path / "missing.nc") is False


def test_wout_default_deck_is_single_sourced_from_examples() -> None:
    from gkx.workflows.runtime import wout as runtime_wout

    deck = runtime_wout.default_wout_deck_path()
    assert deck.is_file()
    assert deck.resolve() == (REPO_ROOT / "examples" / "common_input.toml").resolve()
    data = tomllib.loads(deck.read_text(encoding="utf-8"))
    assert data["geometry"]["model"] == "vmec"
    assert data["physics"]["nonlinear"] is True
    assert data["terms"]["nonlinear"] == 1.0
    assert data["species"][0]["tprim"] == 3.0
    assert data["species"][0]["fprim"] == 1.0
    assert data["grid"]["boundary"] == "fix aspect"
    assert data["geometry"]["torflux"] == 0.64


def test_direct_config_shorthand_wout_positional_uses_default_deck(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_tiny.nc")

    args = _direct_config_shorthand_args([str(wout), "--steps", "3"])

    resolved = tmp_path / "wout_tiny" / "gkx.toml"
    assert args == ["run", "--config", str(resolved), "--steps", "3"]
    data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert data["geometry"]["model"] == "vmec"
    assert data["geometry"]["vmec_file"] == str(wout.resolve())
    # The default target is the NetCDF bundle, not a bare prefix: the spectra,
    # the potential map, and the restart file exist only in that form.
    assert data["output"]["path"] == str(tmp_path / "wout_tiny" / "gkx.out.nc")
    assert data["physics"]["nonlinear"] is True
    assert data["species"][0]["tprim"] == 3.0


def test_direct_config_shorthand_wout_header_names_the_shipped_deck(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """The header says which deck the defaults came from, and how to replace it."""

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_hdr.nc")

    _direct_config_shorthand_args([str(wout)])

    out = capsys.readouterr().out
    shipped = (REPO_ROOT / "examples" / "common_input.toml").resolve()
    assert f"default deck: {shipped}" in out
    assert "gkx my_input.toml wout_hdr.nc" in out
    assert f"wrote resolved input: {tmp_path / 'wout_hdr' / 'gkx.toml'}" in out


def test_direct_config_shorthand_wout_header_names_a_supplied_deck(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """A user's own deck is named as theirs, without the copy-and-edit advice."""

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_own.nc")
    deck = tmp_path / "mine.toml"
    deck.write_text(_RUNTIME_NONLINEAR_TOML_MIN, encoding="utf-8")

    _direct_config_shorthand_args([str(deck), str(wout)])

    out = capsys.readouterr().out
    assert f"input deck: {deck}" in out
    assert "default deck" not in out


def test_direct_config_shorthand_toml_plus_wout_both_orders_force_vmec_model(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_case.nc")
    deck = tmp_path / "deck.toml"
    deck.write_text(_RUNTIME_NONLINEAR_TOML_MIN, encoding="utf-8")

    for argv in ([str(deck), str(wout)], [str(wout), str(deck)]):
        args = _direct_config_shorthand_args(argv)
        resolved = tmp_path / "wout_case" / "gkx.toml"
        assert args == ["run", "--config", str(resolved)]
        data = tomllib.loads(resolved.read_text(encoding="utf-8"))
        # Non-VMEC deck geometry is forced onto the supplied equilibrium.
        assert data["geometry"]["model"] == "vmec"
        assert data["geometry"]["vmec_file"] == str(wout.resolve())
        # The rest of the user deck is preserved.
        assert data["grid"]["Ny"] == 6
        assert data["run"]["steps"] == 1


def test_direct_config_shorthand_wout_flag_aliases_match_positional(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_alias.nc")
    deck = tmp_path / "deck.toml"
    deck.write_text(_RUNTIME_NONLINEAR_TOML_MIN, encoding="utf-8")
    resolved = str(tmp_path / "wout_alias" / "gkx.toml")

    assert _direct_config_shorthand_args(["--vmec", str(wout)]) == [
        "run",
        "--config",
        resolved,
    ]
    assert _direct_config_shorthand_args([f"--vmex={wout}"]) == [
        "run",
        "--config",
        resolved,
    ]
    assert _direct_config_shorthand_args([str(deck), "--vmex", str(wout)]) == [
        "run",
        "--config",
        resolved,
    ]
    data = tomllib.loads(Path(resolved).read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["geometry"]["model"] == "vmec"
    assert data["grid"]["Ny"] == 6


def test_cli_wout_run_groups_outputs_and_writes_resolved_deck(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_tiny.nc")
    captured: dict[str, object] = {}

    def _fake_run_runtime_nonlinear_with_artifacts(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["out"] = kwargs.get("out")
        return _fake_nonlinear_result(), {}

    monkeypatch.setattr(
        "gkx.cli.run_runtime_nonlinear_with_artifacts",
        _fake_run_runtime_nonlinear_with_artifacts,
    )
    monkeypatch.setattr(sys, "argv", ["gkx", str(wout), "--steps", "2"])

    assert main() == 0
    out = capsys.readouterr().out
    assert "wrote resolved input" in out
    cfg = captured["cfg"]
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file == str(wout.resolve())
    assert captured["out"] == str(tmp_path / "wout_tiny" / "gkx.out.nc")
    assert (tmp_path / "wout_tiny" / "gkx.toml").exists()


def test_cli_wout_linear_flag_runs_default_ky_scan(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    from gkx.workflows.runtime.wout import DEFAULT_LINEAR_KY_VALUES

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_lin.nc")
    captured: dict[str, object] = {}
    scan = type(
        "Scan",
        (),
        {
            "ky": np.array([0.1]),
            "gamma": np.array([0.2]),
            "omega": np.array([-0.3]),
        },
    )()

    def _fake_run_runtime_scan(cfg, ky_values, **kwargs):
        captured["cfg"] = cfg
        captured["ky_values"] = ky_values
        return scan

    def _fake_write_scan(base, scan_in, **_kwargs):
        captured["out_base"] = str(base)
        return {"summary": "scan.summary.json", "scan": "scan.csv"}

    monkeypatch.setattr("gkx.cli.run_runtime_scan", _fake_run_runtime_scan)
    monkeypatch.setattr("gkx.cli.write_runtime_linear_scan_artifacts", _fake_write_scan)
    monkeypatch.setattr(sys, "argv", ["gkx", str(wout), "--linear", "--no-progress"])

    assert main() == 0
    out = capsys.readouterr().out
    assert "saved scan.csv" in out
    cfg = captured["cfg"]
    assert cfg.physics.linear is True
    assert cfg.physics.nonlinear is False
    assert cfg.geometry.model == "vmec"
    assert captured["ky_values"] == list(DEFAULT_LINEAR_KY_VALUES)
    assert captured["out_base"] == str(tmp_path / "wout_lin" / "gkx")
    assert (tmp_path / "wout_lin" / "gkx.toml").exists()


_WOUT_LINEAR_E2E_DECK = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 3.0
fprim = 1.0
nu = 0.0
kinetic = true

[grid]
Nx = 4
Ny = 8
Nz = 16
Lx = 12.56
Ly = 12.56
boundary = "linked"
y0 = 2.0

[time]
t_max = 2.0
dt = 0.5
method = "rk3"
sample_stride = 1
fixed_dt = false
cfl = 0.9

[geometry]
model = "vmec"
vmec_file = "replaced-by-the-shorthand.nc"
torflux = 0.64
alpha = 0.0
npol = 1.0

[init]
init_field = "density"
init_amp = 0.001
gaussian_init = false
init_single = false

[physics]
linear = false
nonlinear = true
electrostatic = true
adiabatic_electrons = true
tau_e = 1.0
collisions = false
hypercollisions = true

[terms]
nonlinear = 1.0

[run]
Nl = 2
Nm = 4

[scan]
ky = [0.5]
Nl = 2
Nm = 4
"""


def _write_closed_interval_vmec_eik(path: Path) -> Path:
    """Write the `*.eik.nc` a VMEC import produces for a circular flux tube.

    Written by the shipped writer and read back by the shipped loader, so the
    geometry under test is a real imported flux tube: sampled on the closed
    theta interval, one point longer than the grid built from it.
    """

    from types import SimpleNamespace

    from gkx.geometry.imported_vmec import write_vmec_eik_netcdf

    nz, shat, q, eps, r0 = 16, 0.8, 1.4, 0.18, 3.0
    theta = np.linspace(-np.pi, np.pi, nz + 1)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    shear = shat * theta
    drift = 2.0 * (cos_t + shear * sin_t) / r0
    drift0 = -2.0 * shat * sin_t / r0
    write_vmec_eik_netcdf(
        path,
        {
            "theta": theta,
            "theta_PEST": theta,
            "bmag": 1.0 / (1.0 + eps * cos_t),
            "gradpar": np.full(theta.size, 1.0 / (q * r0)),
            "grho": np.ones(theta.size),
            "gds2": 1.0 + shear * shear,
            "gds21": -shat * shear,
            "gds22": np.full(theta.size, shat * shat),
            "gbdrift": drift,
            "cvdrift": drift,
            "gbdrift0": drift0,
            "cvdrift0": drift0,
            "Rplot": r0 + eps * cos_t,
            "Zplot": eps * sin_t,
            "grad_x": np.zeros((3, theta.size)),
            "grad_y": np.zeros((3, theta.size)),
            "b_vec": np.zeros((3, theta.size)),
            "dpsidrho": 1.0,
            "kxfac": 1.0,
            "Rmaj": r0,
            "q": q,
            "shat": shat,
            "scale": 1.0,
            "alpha": 0.0,
            "zeta_center": 0.0,
            "nfp": 1,
        },
        request=SimpleNamespace(),
    )
    return path


def test_cli_wout_linear_shorthand_scans_imported_vmec_geometry_end_to_end(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """`gkx wout_XXX.nc --linear` solves, with nothing about the run stubbed.

    Three defects each masked the next, so nothing short of running the whole
    command catches them: the CFL hint aborted on the imported geometry's
    closed theta interval, the multimode seed left a one-k_y grid at zero, and
    the inherited nonlinear step overflowed. Two of the three fail silently or
    as a downstream symptom, which is why the assertions here are that the scan
    exits clean *and* reports a mode that actually grew.
    """

    from gkx.geometry import load_imported_geometry_netcdf
    from gkx.workflows.runtime.wout import LINEAR_SCAN_DT

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_e2e.nc")
    eik = _write_closed_interval_vmec_eik(tmp_path / "geom.eik.nc")
    imported = load_imported_geometry_netcdf(eik)
    assert imported.theta_closed_interval is True
    monkeypatch.setattr("gkx.runtime.generate_runtime_vmec_eik", lambda _cfg: eik)

    deck = tmp_path / "deck.toml"
    deck.write_text(_WOUT_LINEAR_E2E_DECK, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["gkx", str(wout), str(deck), "--linear", "--no-progress", "--no-plots"],
    )

    assert main() == 0

    scanned = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("ky=")
    ]
    assert len(scanned) == 1
    gamma = float(scanned[0].split("gamma=")[1].split()[0])
    omega = float(scanned[0].split("omega=")[1].split()[0])
    assert np.isfinite(gamma) and np.isfinite(omega)
    # An all-zero seed fits to |gamma| ~ 1e-16 and exits 0, so the magnitude is
    # what separates a solved mode from a silently empty one.
    assert gamma > 1.0e-6

    resolved = tmp_path / "wout_e2e" / "gkx.toml"
    data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert data["geometry"]["model"] == "vmec"
    assert data["geometry"]["vmec_file"] == str(wout.resolve())
    assert data["physics"]["linear"] is True
    assert data["physics"]["nonlinear"] is False
    assert data["terms"]["nonlinear"] == 0.0
    # The deck's nonlinear step is reduced for the scan; its ky list survives.
    assert data["time"]["dt"] == LINEAR_SCAN_DT
    assert data["scan"]["ky"] == [0.5]
    for name in ("gkx.scan.csv", "gkx.summary.json"):
        assert (tmp_path / "wout_e2e" / name).is_file()


def test_wout_linear_leaves_a_step_already_under_the_scan_bound_alone(
    monkeypatch, tmp_path: Path
) -> None:
    """The scan step is a ceiling, not an assignment.

    A deck that already asked for a finer step than the shorthand's ceiling
    keeps it, and its own `[scan]` table is filled in rather than replaced.
    """

    from gkx.workflows.runtime.wout import LINEAR_SCAN_DT

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_fine_dt.nc")
    deck = tmp_path / "deck.toml"
    deck.write_text(
        _RUNTIME_NONLINEAR_TOML_MIN.replace("dt = 0.01", "dt = 0.0005")
        + "\n[scan]\nky = [0.25]\n",
        encoding="utf-8",
    )

    args = _direct_config_shorthand_args([str(deck), str(wout), "--linear"])

    resolved = tmp_path / "wout_fine_dt" / "gkx.toml"
    assert args == ["scan-runtime-linear", "--config", str(resolved)]
    data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert data["time"]["dt"] == 0.0005 < LINEAR_SCAN_DT
    assert data["scan"]["ky"] == [0.25]


def test_direct_config_shorthand_wout_honors_out_flag(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_out.nc")

    args = _direct_config_shorthand_args([str(wout), "--out", "custom/prefix"])

    resolved = tmp_path / "custom" / "prefix.toml"
    assert args == ["run", "--config", str(resolved), "--out", "custom/prefix"]
    data = tomllib.loads(resolved.read_text(encoding="utf-8"))
    assert data["output"]["path"] == str(tmp_path / "custom" / "prefix")


def _scan_stub():
    return type(
        "Scan",
        (),
        {
            "ky": np.array([0.1, 0.2]),
            "gamma": np.array([0.2, 0.3]),
            "omega": np.array([-0.3, -0.4]),
            "quasilinear": None,
            "parallel": None,
            "warm_start": None,
        },
    )()


@pytest.mark.parametrize(
    ("argv_flag", "toml_scan", "expected"),
    [
        (None, {}, None),
        ("--no-warm-start", {}, False),
        ("--warm-start", {"warm_start": False}, True),
        (None, {"warm_start": False}, False),
    ],
)
def test_scan_runtime_linear_resolves_warm_start(
    monkeypatch, capsys, argv_flag, toml_scan, expected
) -> None:
    """The flag wins over [scan] warm_start, which wins over the config default."""

    cfg = RuntimeConfig()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "gkx.cli.load_runtime_from_toml",
        lambda _path: (cfg, {"scan": {"ky": [0.1, 0.2], **toml_scan}, "fit": {}}),
    )

    def _fake_scan(*_args, **kwargs):
        captured.update(kwargs)
        return _scan_stub()

    monkeypatch.setattr("gkx.cli.run_runtime_scan", _fake_scan)
    argv = ["gkx", "scan-runtime-linear", "--config", "case.toml", "--no-plots"]
    if argv_flag is not None:
        argv.append(argv_flag)
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    capsys.readouterr()
    assert captured["warm_start"] is expected


def _valid_case_toml(tmp_path) -> Path:
    """Write a case that validates: analytic geometry needs no wout file."""

    source = Path(__file__).resolve().parents[3] / "examples" / "common_input.toml"
    text = source.read_text(encoding="utf-8").replace(
        'model = "vmec"', 'model = "s-alpha"'
    )
    target = tmp_path / "case.toml"
    target.write_text(text, encoding="utf-8")
    return target


def test_cli_validate_accepts_a_runnable_case(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["gkx", "validate", str(_valid_case_toml(tmp_path))]
    )
    assert main() == 0
    assert "is a valid case" in capsys.readouterr().out


def test_cli_validate_names_the_problem_and_exits_nonzero(
    tmp_path, capsys, monkeypatch
) -> None:
    """The shipped template declares VMEC geometry with no wout file."""

    source = Path(__file__).resolve().parents[3] / "examples" / "common_input.toml"
    target = tmp_path / "template.toml"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["gkx", "validate", str(target)])
    assert main() == 1
    assert "requires geometry.vmec_file" in capsys.readouterr().err


def test_cli_validate_reports_a_missing_file_rather_than_raising(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["gkx", "validate", str(tmp_path / "absent.toml")])
    assert main() == 1
    assert "not runnable" in capsys.readouterr().err


def test_cli_inspect_describes_a_case_without_running_it(
    tmp_path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["gkx", "inspect", str(_valid_case_toml(tmp_path))]
    )
    assert main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["geometry_model"] == "s-alpha"
    assert payload["n_species"] >= 1


def test_cli_inspect_reads_a_saved_result_summary(
    tmp_path, capsys, monkeypatch
) -> None:
    base = tmp_path / "run.out.nc"
    base.with_name(base.name + ".summary.json").write_text(
        json.dumps({"kind": "linear", "gamma": 0.14}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", ["gkx", "inspect", str(base)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["gamma"] == 0.14


def test_cli_inspect_reports_a_missing_summary(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["gkx", "inspect", str(tmp_path / "none.out.nc")])
    assert main() == 1
    assert "no summary beside" in capsys.readouterr().err


def test_cli_help_lists_the_product_commands() -> None:
    from gkx.cli import build_parser

    text = build_parser().format_help()
    for command in ("run", "scan", "estimate", "inspect", "validate"):
        assert command in text


def test_deprecated_commands_name_their_replacement(capsys, monkeypatch) -> None:
    """They keep working for one release and say what to use instead."""

    from gkx.cli import _warn_deprecated_command

    for name, replacement in (
        ("run-runtime-linear", "run"),
        ("scan-runtime-linear", "scan"),
        ("run-runtime-nonlinear", "run"),
    ):
        _warn_deprecated_command(name)
        err = capsys.readouterr().err
        assert name in err and f"gkx {replacement}" in err


# ---- from test_runtime_helpers.py ----


def _base_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=4, Ny=6, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(t_max=0.4, dt=0.1, method="rk2", sample_stride=1),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )


def _diag(offset: float = 0.0, *, resolved: bool = True) -> SimulationDiagnostics:
    res = None
    if resolved:
        res = ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float) + offset,
            Wg_kxst=np.ones((2, 1, 4), dtype=float) + offset,
        )
    return SimulationDiagnostics(
        t=np.asarray([0.1, 0.2]) + offset,
        dt_t=np.asarray([0.1, 0.1]),
        dt_mean=np.asarray(0.1),
        gamma_t=np.asarray([0.01, 0.02]) + offset,
        omega_t=np.asarray([0.03, 0.04]) + offset,
        Wg_t=np.asarray([1.0, 1.1]) + offset,
        Wphi_t=np.asarray([2.0, 2.1]) + offset,
        Wapar_t=np.asarray([0.5, 0.6]) + offset,
        heat_flux_t=np.asarray([3.0, 3.1]) + offset,
        particle_flux_t=np.asarray([4.0, 4.1]) + offset,
        energy_t=np.asarray([3.5, 3.8]) + offset,
        heat_flux_species_t=np.asarray([[3.0], [3.1]]) + offset,
        particle_flux_species_t=np.asarray([[4.0], [4.1]]) + offset,
        turbulent_heating_t=np.asarray([5.0, 5.1]) + offset,
        turbulent_heating_species_t=np.asarray([[5.0], [5.1]]) + offset,
        phi_mode_t=np.asarray([1.0 + 0.0j, 1.1 + 0.1j]),
        resolved=res,
    )


def test_runtime_linear_terms_disable_zero_collision_frequency() -> None:
    cfg_zero = replace(
        _base_cfg(),
        species=(RuntimeSpeciesConfig(name="ion", nu=0.0),),
        physics=RuntimePhysicsConfig(collisions=True, hypercollisions=False),
    )
    cfg_nonzero = replace(
        cfg_zero,
        species=(RuntimeSpeciesConfig(name="ion", nu=0.05),),
    )

    assert build_runtime_linear_terms(cfg_zero).collisions == 0.0
    assert build_runtime_linear_terms(cfg_nonzero).collisions == 1.0


def test_runtime_small_helper_functions() -> None:
    cfg = _base_cfg()
    grid = build_spectral_grid(cfg.grid)

    assert _normalize_linear_solver_name(" explicit_time ") == "explicit_time"
    assert _normalize_linear_solver_name("krylov") == "krylov"
    assert _midplane_index(grid) == min(grid.z.size // 2 + 1, grid.z.size - 1)
    assert _midplane_index(type("Grid", (), {"z": np.asarray([0.0])})()) == 0
    assert _zero_kx_index(grid) == int(np.argmin(np.abs(np.asarray(grid.kx))))
    assert _dealiased_initial_mode_pairs(grid)[0] == (0, 1)
    assert _periodic_zp_from_grid(np.asarray([0.0])) == 1.0
    assert _periodic_zp_from_grid(np.asarray([0.0, 0.0])) == 1.0
    assert _default_hermite_hypercollision_exponent(None) == 20.0
    assert _default_hermite_hypercollision_exponent(3) == 1.0
    assert _default_hermite_hypercollision_exponent(40) == 20.0
    assert _runtime_model_key(cfg) == "gyrokinetic"


def test_runtime_policy_helpers_preserve_public_runtime_facade_exports() -> None:
    for name in runtime_policies.__all__:
        assert getattr(runtime, name) is getattr(runtime_policies, name)


def test_runtime_facade_module_is_patchable_public_surface() -> None:
    assert runtime._runtime_facade_module() is runtime


def test_runtime_command_helpers_have_single_canonical_owner() -> None:
    command_names = [
        "RuntimeCommandDeps",
        "apply_quasilinear_overrides",
        "apply_runtime_path_overrides",
        "run_runtime_linear_command",
        "run_runtime_nonlinear_command",
        "runtime_output_path",
        "scan_runtime_linear_command",
        "should_show_progress",
    ]

    for name in command_names:
        assert getattr(runtime_cases, name) is getattr(runtime_commands, name)


def test_runtime_command_artifact_helpers_live_with_artifact_orchestration() -> None:
    assert runtime_artifacts.COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS == (
        "summary",
        "timeseries",
        "eigenfunction",
        "state",
        "quasilinear_summary",
        "quasilinear_species",
    )
    assert runtime_artifacts.COMMAND_NONLINEAR_ARTIFACT_DISPLAY_KEYS[-3:] == (
        "out",
        "big",
        "restart",
    )


def test_prepare_runtime_command_config_applies_explicit_override_policy() -> None:
    cfg = _base_cfg()
    args = SimpleNamespace(
        config="case.toml",
        vmec_file="wout_case.nc",
        geometry_file=None,
        init_file=None,
        quasilinear=True,
        ql_mode="saturated",
        ql_saturation_rule=None,
        ql_csat=None,
        ql_normalization=None,
        ql_output="ql_out",
    )
    deps = runtime_commands.build_runtime_command_deps(
        SimpleNamespace(
            load_runtime_from_toml=lambda _path: (cfg, {"run": {"ky": 0.3}}),
            run_runtime_linear=lambda *_args, **_kwargs: None,
            run_runtime_scan=lambda *_args, **_kwargs: None,
            run_runtime_nonlinear_with_artifacts=lambda *_args, **_kwargs: None,
            write_runtime_linear_artifacts=lambda *_args, **_kwargs: {},
            write_runtime_linear_scan_artifacts=lambda *_args, **_kwargs: {},
            write_quasilinear_artifacts=lambda *_args, **_kwargs: {},
            resolve_runtime_path=lambda value, **_kwargs: f"resolved::{value}",
        )
    )

    prepared, data = runtime_commands._prepare_runtime_command_config(
        args,
        deps=deps,
        path_overrides=True,
        quasilinear_overrides=True,
    )

    assert data == {"run": {"ky": 0.3}}
    assert prepared.geometry.vmec_file == "resolved::wout_case.nc"
    assert prepared.quasilinear.enabled is True
    assert prepared.quasilinear.mode == "saturated"
    assert prepared.quasilinear.output_path == "ql_out"

    untouched, _ = runtime_commands._prepare_runtime_command_config(
        args,
        deps=deps,
        path_overrides=False,
        quasilinear_overrides=False,
    )
    assert untouched.geometry.vmec_file is None
    assert untouched.quasilinear.enabled is False


def test_runtime_command_option_helpers_normalize_cli_and_toml_values() -> None:
    cfg = _base_cfg()
    section = {
        "Nm": "7",
        "solver": "explicit_time",
        "method": "rk4",
        "steps": "16",
        "sample_stride": "3",
    }
    args = SimpleNamespace(
        Nl="9",
        Nm=None,
        solver=None,
        fit_signal="phi",
        method=None,
        dt="0.05",
        steps=None,
        sample_stride=None,
    )

    assert runtime_commands._resolve_grid_time_options(args, section, cfg) == (
        9,
        7,
        "rk4",
        0.05,
        16,
        3,
    )
    assert runtime_commands._resolve_linear_fit_options(args, section) == (
        "explicit_time",
        "phi",
    )

    empty_args = SimpleNamespace(Nl=None, Nm=None, method=None, dt=None, steps=None)
    assert runtime_commands._resolve_grid_time_options(empty_args, {}, cfg) == (
        24,
        12,
        None,
        None,
        None,
        cfg.time.sample_stride,
    )


def test_runtime_linear_command_rejects_invalid_sampling_before_setup() -> None:
    args = SimpleNamespace(
        ky=None,
        Nl=None,
        Nm=None,
        solver="time",
        fit_signal=None,
        method="rk4",
        dt=0.1,
        steps=17,
        sample_stride=4,
        progress=False,
        no_progress=True,
    )

    with pytest.raises(ValueError, match="must be divisible by sample_stride"):
        runtime_commands._resolve_linear_command_options(args, _base_cfg(), {})


def test_runtime_case_option_helpers_resolve_python_overrides() -> None:
    raw = {
        "run": {
            "ky": "0.2",
            "Nl": "4",
            "Nm": "6",
            "solver": "time",
            "method": "rk2",
            "dt": "0.05",
            "steps": "12",
        },
        "time": {"sample_stride": "3", "diagnostics_stride": "5"},
        "fit": {"fit_signal": "phi", "ignored": "value"},
    }

    assert runtime_cases._runtime_case_fit_config(raw) == {"fit_signal": "phi"}
    assert runtime_cases._linear_case_run_kwargs(
        raw,
        {
            "ky": 0.4,
            "Nl": None,
            "Nm": 7,
            "solver": None,
            "method": "rk4",
            "dt": None,
            "steps": 20,
            "sample_stride": None,
        },
    ) == {
        "ky_target": 0.4,
        "Nl": 4,
        "Nm": 7,
        "solver": "time",
        "method": "rk4",
        "dt": "0.05",
        "steps": 20,
        "sample_stride": "3",
    }
    assert runtime_cases._nonlinear_case_run_kwargs(
        raw,
        {
            "ky": None,
            "Nl": 8,
            "Nm": None,
            "method": None,
            "dt": 0.02,
            "steps": None,
            "sample_stride": 2,
            "diagnostics_stride": None,
        },
    ) == {
        "ky_target": 0.2,
        "Nl": 8,
        "Nm": 6,
        "method": "rk2",
        "dt": 0.02,
        "steps": "12",
        "sample_stride": 2,
        "diagnostics_stride": "5",
        "diagnostics": True,
    }


def test_plot_saved_output_command_routes_renderer_and_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _renderer(path: str, *, out: str | None = None) -> Path:
        captured["path"] = path
        captured["out"] = out
        return Path("rendered.png")

    assert (
        runtime_commands.plot_saved_output_command(
            ["--plot", "case.summary.json", "--out", "figure.png"],
            plot_saved_output=_renderer,
        )
        == 0
    )
    assert captured == {"path": "case.summary.json", "out": "figure.png"}
    assert "saved rendered.png" in capsys.readouterr().out

    assert (
        runtime_commands.plot_saved_output_command(
            ["--plot"], plot_saved_output=_renderer
        )
        == 1
    )
    assert "usage: gkx --plot" in capsys.readouterr().out


def test_runtime_nonlinear_command_print_helpers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime_commands.print_nonlinear_run_header(
        config_path="case.toml",
        runtime_config=_base_cfg(),
        ky=0.2,
        Nl=4,
        Nm=6,
        method="rk2",
        dt=0.05,
        steps=None,
        grid_shape=(8, 10, 12),
        diagnostics=True,
        show_progress=False,
    )

    diag = _diag()
    assert (
        runtime_commands.print_nonlinear_run_summary(
            RuntimeNonlinearResult(
                t=diag.t,
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=-0.1,
            )
        )
        is True
    )
    assert (
        runtime_commands.print_nonlinear_run_summary(
            RuntimeNonlinearResult(
                t=np.asarray([0.1]),
                diagnostics=None,
                ky_selected=0.2,
                kx_selected=0.0,
            )
        )
        is False
    )

    out = capsys.readouterr().out
    assert "starting runtime nonlinear run" in out
    assert "steps=auto" in out
    assert "diagnostics=on progress=off" in out
    assert "physics=electrostatic kinetic=ion(a/L_T=2.49,a/L_n=0.8)" in out
    assert "adiabatic=electrons" in out
    assert "a/L_T=-a d(ln T)/dr" in out
    assert "gamma=d ln|phi_k|/dt" in out
    assert "selected-mode diagnostics" in out
    assert "Wg=distribution free energy; Q=radial heat flux/Q_gB" in out
    assert "saturation uses Q/Wphi/Wg" in out
    assert "nonlinear: t=0.2" in out
    assert "ky_sel=0.2" in out
    assert "Wphi=2.1" in out
    assert "nonlinear run completed" in out


def test_runtime_command_artifact_output_helpers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str]] = []
    deps = SimpleNamespace(
        write_runtime_linear_artifacts=lambda path, _result: (
            calls.append(("linear", str(path)))
            or {
                "state": "linear.state.nc",
                "summary": "linear.summary.json",
                "timeseries": "linear.timeseries.csv",
            }
        ),
        write_runtime_linear_scan_artifacts=lambda path, _scan: (
            calls.append(("scan", str(path)))
            or {
                "quasilinear_spectrum": "scan.ql.csv",
                "summary": "scan.summary.json",
                "scan": "scan.csv",
            }
        ),
        write_quasilinear_artifacts=lambda path, _ql: (
            calls.append(("quasilinear", str(path)))
            or {
                "quasilinear_species": "ql.species.csv",
                "quasilinear_summary": "ql.summary.json",
            }
        ),
    )
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.3,
        omega=-0.4,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        quasilinear={"model": "test"},
    )

    assert (
        runtime_artifacts.write_command_outputs(
            None,
            result,
            writer=deps.write_runtime_linear_artifacts,
            display_keys=runtime_artifacts.COMMAND_LINEAR_ARTIFACT_DISPLAY_KEYS,
        )
        == {}
    )
    assert (
        runtime_artifacts.write_command_outputs(
            None,
            result.quasilinear,
            writer=deps.write_quasilinear_artifacts,
            display_keys=runtime_artifacts.COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
        )
        == {}
    )
    no_ql = replace(result, quasilinear=None)
    assert (
        runtime_artifacts.write_command_outputs(
            "ql.json",
            no_ql.quasilinear,
            writer=deps.write_quasilinear_artifacts,
            display_keys=runtime_artifacts.COMMAND_QUASILINEAR_ARTIFACT_DISPLAY_KEYS,
        )
        == {}
    )
    assert calls == []

    cfg = replace(
        _base_cfg(),
        output=RuntimeOutputConfig(path="linear.json"),
        quasilinear=RuntimeQuasilinearConfig(output_path="ql.json"),
    )
    args = SimpleNamespace(out=None, ql_output=None)
    assert runtime_commands._write_linear_runtime_command_outputs(
        args,
        cfg,
        result,
        deps=deps,  # type: ignore[arg-type]
    ) == {
        "linear": {
            "state": "linear.state.nc",
            "summary": "linear.summary.json",
            "timeseries": "linear.timeseries.csv",
        },
        "quasilinear": {
            "quasilinear_species": "ql.species.csv",
            "quasilinear_summary": "ql.summary.json",
        },
    }
    scan_cfg = replace(
        _base_cfg(),
        quasilinear=RuntimeQuasilinearConfig(output_path="scan.json"),
    )
    assert runtime_commands._write_scan_runtime_command_outputs(
        args,
        scan_cfg,
        SimpleNamespace(),
        deps=deps,  # type: ignore[arg-type]
    ) == {
        "quasilinear_spectrum": "scan.ql.csv",
        "summary": "scan.summary.json",
        "scan": "scan.csv",
    }
    runtime_artifacts.print_nonlinear_command_outputs(
        {
            "restart": "restart.nc",
            "summary": "nonlinear.summary.json",
            "diagnostics": "nonlinear.diag.nc",
        },
        enabled=True,
    )
    runtime_artifacts.print_nonlinear_command_outputs(
        {"summary": "not-printed.json"}, enabled=False
    )

    assert calls == [
        ("linear", "linear.json"),
        ("quasilinear", "ql.json"),
        ("scan", "scan.json"),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "saved linear.summary.json",
        "saved linear.timeseries.csv",
        "saved linear.state.nc",
        "saved ql.summary.json",
        "saved ql.species.csv",
        "saved scan.summary.json",
        "saved scan.csv",
        "saved scan.ql.csv",
        "saved nonlinear.summary.json",
        "saved nonlinear.diag.nc",
        "saved restart.nc",
    ]


def test_runtime_dispatch_deps_are_built_from_patchable_runtime_scope() -> None:
    linear_deps = runtime._runtime_linear_dispatch_deps()
    nonlinear_deps = runtime._runtime_nonlinear_dispatch_deps()

    assert (
        linear_deps.full_deps.build_runtime_geometry is runtime.build_runtime_geometry
    )
    assert linear_deps.full_deps.build_linear_cache is runtime.build_linear_cache
    assert (
        nonlinear_deps.full_deps.build_runtime_geometry
        is runtime.build_runtime_geometry
    )
    assert (
        nonlinear_deps.full_deps.integrate_nonlinear_from_config
        is runtime.integrate_nonlinear_from_config
    )


def test_runtime_scan_deps_are_built_from_patchable_runtime_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scan_plan = object()
    geometry_builder = object()

    monkeypatch.setattr(runtime, "_runtime_independent_parallel_plan", scan_plan)
    monkeypatch.setattr(runtime, "build_runtime_geometry", geometry_builder)

    orchestration_deps = runtime._runtime_scan_orchestration_deps()
    batch_deps = runtime._runtime_scan_batch_deps()

    assert orchestration_deps.runtime_independent_parallel_plan is scan_plan
    assert orchestration_deps.run_runtime_scan_batch is runtime._run_runtime_scan_batch
    assert batch_deps.build_runtime_geometry is geometry_builder
    assert (
        batch_deps.integrate_linear_diagnostics is runtime.integrate_linear_diagnostics
    )


def test_runtime_scan_ky_task_forwards_linear_options() -> None:
    cfg = _base_cfg()
    sentinel = object()
    calls: list[dict[str, object]] = []
    task = {
        "cfg": cfg,
        "ky": "0.4",
        "Nl": "5",
        "Nm": "3",
        "solver": "time",
        "method": "rk2",
        "dt": 0.02,
        "steps": 7,
        "sample_stride": 2,
        "auto_window": False,
        "tmin": 1.0,
        "tmax": 2.0,
        "window_fraction": 0.25,
        "min_points": 6,
        "start_fraction": 0.1,
        "growth_weight": 0.3,
        "require_positive": False,
        "min_amp_fraction": 0.05,
        "window_method": "loglinear",
        "krylov_cfg": None,
        "mode_method": "project",
        "fit_signal": "phi",
        "show_progress": True,
    }

    def _runner(cfg_arg, **kwargs):
        calls.append({"cfg": cfg_arg, **kwargs})
        return sentinel

    assert run_runtime_scan_ky_task(task, run_runtime_linear=_runner) is sentinel
    assert calls == [
        {
            "cfg": cfg,
            "ky_target": 0.4,
            "Nl": 5,
            "Nm": 3,
            "solver": "time",
            "method": "rk2",
            "dt": 0.02,
            "steps": 7,
            "sample_stride": 2,
            "auto_window": False,
            "tmin": 1.0,
            "tmax": 2.0,
            "window_fraction": 0.25,
            "min_points": 6,
            "start_fraction": 0.1,
            "growth_weight": 0.3,
            "require_positive": False,
            "min_amp_fraction": 0.05,
            "window_method": "loglinear",
            "krylov_cfg": None,
            "mode_method": "project",
            "fit_signal": "phi",
            "show_progress": True,
            # A task that says nothing about warm start forwards the cold
            # defaults, so the independent-worker path is unchanged.
            "initial_state": None,
            "return_state": False,
        }
    ]


def test_runtime_independent_parallel_plan_resolves_config_and_arguments() -> None:
    cfg = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=4, backend="process"
        ),
    )

    plan = runtime_policies._runtime_independent_parallel_plan(
        cfg, problem_size=3, workers=1, executor="thread"
    )

    assert plan.requested_workers == 4
    assert plan.effective_workers == 3
    assert plan.executor == "process"
    assert plan.source == "runtime_config"
    assert plan.enabled is True
    assert plan.to_dict()["enabled"] is True

    explicit = runtime_policies._runtime_independent_parallel_plan(
        cfg, problem_size=3, workers=2, executor="threads"
    )

    assert explicit.requested_workers == 2
    assert explicit.executor == "thread"
    assert explicit.source == "arguments"


def test_runtime_independent_parallel_plan_rejects_invalid_policy() -> None:
    cfg_bad_backend = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=2, backend="mpi"
        ),
    )
    cfg_bad_axis = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(strategy="batch", axis="kx", num_devices=2),
    )

    with pytest.raises(ValueError, match="workers"):
        runtime_policies._runtime_independent_parallel_plan(
            _base_cfg(), problem_size=1, workers=0, executor="thread"
        )
    with pytest.raises(ValueError, match="backend"):
        runtime_policies._runtime_independent_parallel_plan(
            cfg_bad_backend, problem_size=2, workers=1, executor="thread"
        )
    with pytest.raises(ValueError, match="axis='ky'"):
        runtime_policies._runtime_independent_parallel_plan(
            cfg_bad_axis, problem_size=2, workers=1, executor="thread"
        )


def test_runtime_chunk_progress_policy() -> None:
    message, snapshot = build_runtime_progress_message(
        label="nonlinear",
        chunk_index=3,
        t_elapsed=2.0,
        t_max=4.0,
        chunk_wall_seconds=61.0,
        elapsed_seconds=180.0,
    )

    assert format_duration(3661.0) == "1:01:01"
    assert snapshot.progress == pytest.approx(0.5)
    assert snapshot.eta_seconds == pytest.approx(180.0)
    assert "completed nonlinear chunk 3" in message
    assert "progress= 50.0%" in message
    assert "chunk_wall=01:01" in message
    assert "elapsed=03:00" in message
    assert "eta=03:00" in message


def test_runtime_random_pair_edge_cases() -> None:
    empty = _centered_glibc_random_pairs(3, 0)
    assert empty.shape == (0, 2)

    seed_zero = _centered_glibc_random_pairs(0, 3)
    seed_one = _centered_glibc_random_pairs(1, 3)
    np.testing.assert_allclose(seed_zero, seed_one)


def test_runtime_mode_index_selection_and_step_inference() -> None:
    cfg = _base_cfg()
    grid = build_spectral_grid(cfg.grid)
    ky_idx, kx_idx = _select_nonlinear_mode_indices(
        grid, ky_target=0.2, kx_target=None, use_dealias_mask=False
    )
    assert 0 <= ky_idx < grid.ky.size
    assert 0 <= kx_idx < grid.kx.size

    empty_mask_grid = type(
        "Grid",
        (),
        {
            "ky": np.asarray([0.0, 0.2, 0.4]),
            "kx": np.asarray([-0.5, 0.0, 0.5]),
            "dealias_mask": np.zeros((3, 3), dtype=bool),
        },
    )()
    ky_idx2, kx_idx2 = _select_nonlinear_mode_indices(
        empty_mask_grid, ky_target=0.4, kx_target=0.5, use_dealias_mask=True
    )
    assert (ky_idx2, kx_idx2) == (2, 2)

    dealiased_grid = type(
        "Grid",
        (),
        {
            "ky": np.asarray([0.0, 0.2, 0.4]),
            "kx": np.asarray([0.0, 0.5, 1.0]),
            "dealias_mask": np.asarray(
                [
                    [True, True, True],
                    [True, False, True],
                    [False, False, False],
                ],
                dtype=bool,
            ),
        },
    )()
    ky_idx3, kx_idx3 = _select_nonlinear_mode_indices(
        dealiased_grid, ky_target=0.39, kx_target=0.9, use_dealias_mask=True
    )
    assert (ky_idx3, kx_idx3) == (1, 2)

    assert _infer_runtime_nonlinear_steps(cfg, dt=0.1, steps=7) == 7
    assert (
        _infer_runtime_nonlinear_steps(
            replace(cfg, time=replace(cfg.time, fixed_dt=True)), dt=0.05, steps=None
        )
        == 4
    )
    adaptive_cfg = replace(cfg, time=replace(cfg.time, fixed_dt=False, dt_max=None))
    assert _infer_runtime_nonlinear_steps(adaptive_cfg, dt=0.2, steps=None) == 2
    with pytest.raises(ValueError):
        _infer_runtime_nonlinear_steps(
            replace(cfg, time=replace(cfg.time, t_max=0.0)), dt=0.1, steps=0
        )


def test_runtime_nonlinear_diagnostics_kwargs_policy() -> None:
    base = _base_cfg()
    cfg = replace(
        base,
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            base.time,
            method="rk4",
            nonlinear_dealias=True,
            collision_split=True,
            collision_scheme="exp",
            implicit_restart=7,
            implicit_preconditioner="jacobi",
            cfl_fac=None,
        ),
    )

    kwargs = runtime_policies.build_runtime_nonlinear_diagnostics_kwargs(
        cfg,
        dt=0.05,
        steps=9,
        method=None,
        term_config="terms",
        sample_stride=2,
        diagnostics_stride=3,
        laguerre_mode="grid",
        ky_index=1,
        kx_index=2,
        fixed_dt=False,
        fixed_mode_ky_index=4,
        fixed_mode_kx_index=5,
        external_phi=0.25,
        resolved_diagnostics=False,
        show_progress=True,
    )

    assert kwargs["dt"] == pytest.approx(0.05)
    assert kwargs["steps"] == 9
    assert kwargs["method"] == "rk4"
    assert kwargs["terms"] == "terms"
    assert kwargs["sample_stride"] == 2
    assert kwargs["diagnostics_stride"] == 3
    assert kwargs["use_dealias_mask"] is True
    assert kwargs["laguerre_mode"] == "grid"
    assert kwargs["omega_ky_index"] == 1
    assert kwargs["omega_kx_index"] == 2
    assert kwargs["fixed_dt"] is False
    assert kwargs["collision_split"] is True
    assert kwargs["collision_scheme"] == "exp"
    assert kwargs["implicit_restart"] == 7
    assert kwargs["implicit_preconditioner"] == "jacobi"
    assert kwargs["fixed_mode_ky_index"] == 4
    assert kwargs["fixed_mode_kx_index"] == 5
    assert kwargs["external_phi"] == pytest.approx(0.25)
    assert kwargs["resolved_diagnostics"] is False
    assert kwargs["show_progress"] is True


def test_runtime_diagnostic_slice_stride_concat() -> None:
    diag = _diag()
    sliced = _slice_runtime_diagnostics(diag, 1)
    assert sliced.t.shape == (1,)
    assert sliced.resolved is not None and sliced.resolved.Phi2_kxt.shape[0] == 1
    zero = _slice_runtime_diagnostics(diag, 0)
    assert float(zero.dt_mean) == 0.0
    with pytest.raises(ValueError):
        _slice_runtime_diagnostics(diag, -1)

    strided = _stride_runtime_diagnostics(diag, stride=2)
    assert strided.t.shape == (1,)
    assert _stride_runtime_diagnostics(diag, stride=1) is diag

    concat = _concat_runtime_diagnostics([diag, _diag(offset=1.0)])
    assert concat.t.shape == (4,)
    assert concat.resolved is not None and concat.resolved.Phi2_kxt.shape[0] == 4
    concat_none = _concat_runtime_diagnostics(
        [replace(diag, resolved=None), replace(_diag(offset=1.0), resolved=None)]
    )
    assert concat_none.resolved is None
    with pytest.raises(ValueError):
        _concat_runtime_diagnostics([])


def test_fit_runtime_linear_diagnostics_density_fit_contract() -> None:
    t = np.asarray([0.1, 0.2, 0.3, 0.4])
    phi = np.ones((4, 1, 1, 2), dtype=np.complex128)
    density = np.asarray([1.0, 1.5, 2.25, 3.375], dtype=np.complex128)[
        :, None, None, None
    ] * np.ones((1, 1, 1, 2), dtype=np.complex128)
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    out = fit_runtime_linear_diagnostics(
        t=t,
        phi_t=phi,
        density_t=density,
        selection=sel,
        z=np.asarray([-1.0, 1.0]),
        fit_signal="density",
        mode_method="z_index",
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=1.0,
        min_points=3,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=True,
        min_amp_fraction=0.0,
    )

    assert out.fit_signal_used == "density"
    assert out.gamma > 0.0
    assert out.fit_window_tmin is not None
    assert out.fit_window_tmax is not None
    np.testing.assert_allclose(out.signal, density[:, 0, 0, 0])


def test_fit_runtime_linear_diagnostics_auto_selects_best_scored_channel() -> None:
    t = np.asarray([0.0, 1.0, 2.0])
    phi = np.ones((3, 1, 1, 1), dtype=np.complex128)
    density = np.asarray([1.0, 2.0, 4.0], dtype=np.complex128)[:, None, None, None]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    def _fake_auto_stats(t_arr, signal, **_kwargs):  # type: ignore[no-untyped-def]
        score = 10.0 if float(np.real(signal[-1])) > 1.5 else 1.0
        return 0.1 * score, -0.2, float(t_arr[0]), float(t_arr[-1]), score, 0.0

    out = fit_runtime_linear_diagnostics(
        t=t,
        phi_t=phi,
        density_t=density,
        selection=sel,
        z=np.asarray([0.0]),
        fit_signal="auto",
        mode_method="z_index",
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=1.0,
        min_points=3,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=True,
        min_amp_fraction=0.0,
        fit_growth_rate_auto_with_stats_fn=_fake_auto_stats,
        extract_eigenfunction_fn=lambda *_args, **_kwargs: np.asarray([1.0 + 0.0j]),
    )

    assert out.fit_signal_used == "density"
    assert out.gamma == pytest.approx(1.0)
    np.testing.assert_allclose(out.signal, density[:, 0, 0, 0])


def _two_phase_channel(
    t: np.ndarray,
    *,
    early_gamma: float,
    early_omega: float,
    late_gamma: float,
    late_omega: float,
    t_break: float,
) -> np.ndarray:
    """Build a channel whose growth rate changes at ``t_break``.

    The requested window sees ``late_gamma``/``late_omega`` only; any window the
    automatic selector picks for itself includes the faster early phase, so a
    discarded window shows up as a different gamma rather than a rounding
    difference.
    """

    early = np.exp((early_gamma - 1j * early_omega) * t)
    late = np.exp((early_gamma - 1j * early_omega) * t_break) * np.exp(
        (late_gamma - 1j * late_omega) * (t - t_break)
    )
    return np.where(t < t_break, early, late).astype(np.complex128)


def _fixed_window_fit_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Time base, phi/density histories, and the pinned-window fit options."""

    t = np.linspace(0.0, 4.0, 401)
    phi = _two_phase_channel(
        t,
        early_gamma=1.0,
        early_omega=0.3,
        late_gamma=0.2,
        late_omega=0.5,
        t_break=2.0,
    )
    density = _two_phase_channel(
        t,
        early_gamma=0.8,
        early_omega=0.1,
        late_gamma=0.35,
        late_omega=0.9,
        t_break=2.0,
    )
    options = {
        "mode_method": "z_index",
        "auto_window": False,
        "tmin": 3.0,
        "tmax": 4.0,
        "window_fraction": 0.3,
        "min_points": 20,
        "start_fraction": 0.0,
        "growth_weight": 1.0,
        "require_positive": True,
        "min_amp_fraction": 0.0,
        "window_method": "stationary",
    }
    return t, phi[:, None, None, None], density[:, None, None, None], options


def test_auto_fit_signal_keeps_the_requested_runtime_window() -> None:
    """'auto' selects the channel; an explicit window still decides the fit."""

    t, phi, density, options = _fixed_window_fit_case()
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    z = np.asarray([0.0])

    auto = fit_runtime_linear_diagnostics(
        t=t,
        phi_t=phi,
        density_t=density,
        selection=sel,
        z=z,
        fit_signal="auto",
        **options,
    )
    explicit = fit_runtime_linear_diagnostics(
        t=t,
        phi_t=phi,
        density_t=density,
        selection=sel,
        z=z,
        fit_signal=auto.fit_signal_used,
        **options,
    )

    assert auto.gamma == explicit.gamma
    assert auto.omega == explicit.omega
    assert auto.fit_window_tmin == pytest.approx(3.0)
    assert auto.fit_window_tmax == pytest.approx(4.0)
    # growth_weight=1.0 scores the faster late-phase channel higher, and the
    # window it is scored over is the requested one, not an auto-selected one.
    assert auto.fit_signal_used == "density"
    assert auto.gamma == pytest.approx(0.35, rel=1e-6)
    assert auto.omega == pytest.approx(0.9, rel=1e-6)


def test_auto_fit_signal_keeps_the_requested_window_in_a_batched_scan() -> None:
    """The combined-ky scan path selects a channel, not a window, too."""

    t, phi, density, options = _fixed_window_fit_case()
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)
    diagnostics = _BatchDiagnostics(phi_t=phi, density_t=density, time=t)
    deps = SimpleNamespace(
        extract_mode_time_series=extract_mode_time_series,
        fit_growth_rate_auto_with_stats=fit_growth_rate_auto_with_stats,
        fit_growth_rate_auto=fit_growth_rate_auto,
        fit_growth_rate=fit_growth_rate,
        fit_growth_rate_with_stats=fit_growth_rate_with_stats,
    )
    scan_options = _RuntimeScanOptions(
        method=None,
        dt=None,
        steps=None,
        sample_stride=None,
        fit_signal="auto",
        show_progress=False,
        **options,
    )
    assert scan_options.fit_fields() == options

    gamma_auto, omega_auto = _fit_batch_scan_point(
        diagnostics,
        sel,
        fit_key="auto",
        options=scan_options,
        deps=deps,
    )
    gamma_density, omega_density = _fit_batch_scan_point(
        diagnostics,
        sel,
        fit_key="density",
        options=scan_options,
        deps=deps,
    )

    assert gamma_auto == gamma_density
    assert omega_auto == omega_density
    assert gamma_auto == pytest.approx(0.35, rel=1e-6)
    assert omega_auto == pytest.approx(0.9, rel=1e-6)


def test_finalize_runtime_linear_quasilinear_contract() -> None:
    cfg = replace(
        _base_cfg(),
        quasilinear=RuntimeQuasilinearConfig(enabled=True, csat=0.7),
    )
    result = RuntimeLinearResult(
        ky=0.2,
        gamma=0.1,
        omega=-0.3,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        state=np.ones((1, 1, 1, 1, 2), dtype=np.complex128),
    )
    statuses: list[str] = []

    class _Payload:
        def to_dict(self) -> dict[str, object]:
            return {"heat_flux_weight_total": 1.5, "mode": "saturated"}

    calls: dict[str, object] = {}

    def _compute(state, **kwargs):
        calls["state_shape"] = np.asarray(state).shape
        calls["metadata"] = kwargs["metadata"]
        calls["csat"] = kwargs["csat"]
        return _Payload()

    out = finalize_runtime_linear_quasilinear(
        result,
        enabled=True,
        cfg=cfg,
        grid=object(),
        geom=object(),
        params=SimpleNamespace(),
        terms=object(),
        Nl=2,
        Nm=3,
        solver_name="krylov",
        species_names=("ion",),
        return_state_requested=False,
        deps=RuntimeQuasilinearFinalizationDeps(
            build_linear_cache=lambda *_args: "cache",
            compute_quasilinear_from_linear_state=_compute,
            linear_terms_to_term_config=lambda terms: terms,
        ),
        status_callback=statuses.append,
    )

    assert out.state is None
    assert out.quasilinear == {"heat_flux_weight_total": 1.5, "mode": "saturated"}
    assert calls["state_shape"] == (1, 1, 1, 1, 2)
    assert calls["csat"] == pytest.approx(0.7)
    assert calls["metadata"] == {
        "runtime_config_enabled": True,
        "solver": "krylov",
        "delta_ky": cfg.quasilinear.delta_ky,
        "species_selection": cfg.quasilinear.species,
        "write_spectrum": cfg.quasilinear.write_spectrum,
    }
    assert statuses == [
        "computing quasilinear transport weights",
        "quasilinear transport weights complete",
    ]


def test_runtime_diagnostic_concat_rejects_misaligned_optional_channels() -> None:
    """Optional species channels must stay aligned with the common time axis."""

    diag0 = replace(_diag(resolved=False), heat_flux_species_t=None)
    diag1 = _diag(offset=1.0, resolved=False)

    with pytest.raises(ValueError, match="optional diagnostic heat_flux_species_t"):
        _concat_runtime_diagnostics([diag0, diag1])

    with pytest.raises(ValueError, match="resolved diagnostics"):
        _concat_runtime_diagnostics(
            [replace(_diag(), resolved=None), _diag(offset=1.0)]
        )

    partial0 = replace(
        _diag(),
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float),
            Wg_kxst=None,
        ),
    )
    partial1 = replace(
        _diag(offset=1.0),
        resolved=ResolvedDiagnostics(
            Phi2_kxt=np.ones((2, 4), dtype=float),
            Wg_kxst=np.ones((2, 1, 4), dtype=float),
        ),
    )
    with pytest.raises(ValueError, match="resolved diagnostic Wg_kxst"):
        _concat_runtime_diagnostics([partial0, partial1])


def test_runtime_species_and_model_helpers() -> None:
    cfg = _base_cfg()
    species = _species_to_linear(cfg.species)
    assert len(species) == 1
    with pytest.raises(ValueError):
        _species_to_linear((RuntimeSpeciesConfig(name="adiabatic", kinetic=False),))

    etg_cfg = replace(
        cfg,
        species=(RuntimeSpeciesConfig(name="electron", charge=-1.0, kinetic=True),),
        normalization=RuntimeNormalizationConfig(contract="etg"),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )
    krylov = _runtime_default_krylov_config(etg_cfg)
    assert krylov.method == "shift_invert"
    assert krylov.mode_family == "etg"
    assert _runtime_default_krylov_config(cfg).method != "shift_invert"

    kbm_cfg = replace(
        cfg,
        normalization=RuntimeNormalizationConfig(contract="kbm"),
    )
    kbm_krylov = _runtime_default_krylov_config(kbm_cfg)
    assert kbm_krylov.method == "shift_invert"
    assert kbm_krylov.mode_family == "kbm"

    assert _resolve_runtime_hl_dims(cfg, Nl=None, Nm=None) == (24, 12)
    unsupported_cfg = replace(
        cfg, physics=replace(cfg.physics, reduced_model="mystery")
    )
    with pytest.raises(ValueError, match="Unknown physics.reduced_model"):
        _resolve_runtime_hl_dims(
            unsupported_cfg,
            Nl=2,
            Nm=1,
        )

    _require_full_gk_runtime_model(cfg)
    with pytest.raises(ValueError, match="Unknown physics.reduced_model"):
        _require_full_gk_runtime_model(unsupported_cfg)


@pytest.mark.parametrize(
    ("name", "t", "phi", "density"),
    [
        ("time", np.array([0.0, np.nan]), np.ones(2), None),
        ("field", np.arange(2.0), np.array([1.0, np.inf]), None),
        ("density", np.arange(2.0), np.ones(2), np.array([1.0, np.nan])),
    ],
)
def test_linear_history_rejects_nonfinite_trajectories(
    name: str,
    t: np.ndarray,
    phi: np.ndarray,
    density: np.ndarray | None,
) -> None:
    """Unstable histories must fail before a finite growth fit can hide them."""

    with pytest.raises(FloatingPointError, match=rf"non-finite {name} history"):
        _prepare_runtime_linear_fit_inputs(
            t=t,
            phi_t=phi,
            density_t=density,
            z=np.ones(1),
            fit_signal="auto",
        )

    _prepare_runtime_linear_fit_inputs(
        t=np.arange(2.0),
        phi_t=np.ones(2),
        density_t=np.ones(2),
        z=np.ones(1),
        fit_signal="auto",
    )


def test_linear_fit_rejects_overflowing_trajectory() -> None:
    """Finite-but-overflowed histories must fail loudly, naming the timestep."""

    t = np.linspace(0.0, 10.0, 64)
    # Synthetic unstable run: still finite everywhere but overflow-scale
    # (~1e102 at the end), the regime a plain finite mask fits confidently.
    phi = np.exp((23.5 + 0.3j) * t)[:, None, None, None] * np.ones(
        (1, 1, 1, 2), dtype=np.complex128
    )
    assert np.all(np.isfinite(phi))

    with pytest.raises(
        FloatingPointError, match=r"overflowing field history.*timestep instability"
    ):
        fit_runtime_linear_diagnostics(
            t=t,
            phi_t=phi,
            density_t=None,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            z=np.asarray([-1.0, 1.0]),
            fit_signal="phi",
            mode_method="z_index",
            auto_window=True,
            tmin=None,
            tmax=None,
            window_fraction=1.0,
            min_points=3,
            start_fraction=0.0,
            growth_weight=0.0,
            require_positive=True,
            min_amp_fraction=0.0,
        )


def test_runtime_wrapper_patch_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_cfg()
    captured: dict[str, object] = {}
    geom = object()

    def _fake_build_geom(_cfg):
        captured["geom_called"] = True
        return geom

    def _fake_build_params(_cfg, *, Nm, geom):
        captured["params"] = {"Nm": Nm, "geom": geom}
        return "params"

    def _fake_build_terms(_cfg):
        captured["terms"] = _cfg
        return "terms"

    def _fake_build_term_config(_cfg):
        captured["term_cfg"] = _cfg
        return "term_cfg"

    monkeypatch.setattr("gkx.runtime.build_runtime_geometry", _fake_build_geom)
    monkeypatch.setattr(
        "gkx.workflows.runtime.startup.build_runtime_linear_params",
        _fake_build_params,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.startup.build_runtime_linear_terms",
        _fake_build_terms,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.startup.build_runtime_term_config",
        _fake_build_term_config,
    )

    assert build_runtime_linear_params(cfg, Nm=7) == "params"
    assert captured["geom_called"] is True
    assert captured["params"] == {"Nm": 7, "geom": geom}

    captured.clear()
    explicit_geom = object()
    assert build_runtime_linear_params(cfg, Nm=5, geom=explicit_geom) == "params"
    assert "geom_called" not in captured
    assert captured["params"] == {"Nm": 5, "geom": explicit_geom}

    assert build_runtime_linear_terms(cfg) == "terms"
    assert captured["terms"] is cfg
    assert build_runtime_term_config(cfg) == "term_cfg"
    assert captured["term_cfg"] is cfg


def test_runtime_external_phi_helper() -> None:
    cfg = _base_cfg()

    assert _runtime_external_phi(cfg) is None
    assert (
        _runtime_external_phi(
            replace(cfg, expert=RuntimeExpertConfig(source=" default "))
        )
        is None
    )
    assert _runtime_external_phi(
        replace(cfg, expert=RuntimeExpertConfig(source="phiext_full", phi_ext=0.375))
    ) == pytest.approx(0.375)

    with pytest.raises(ValueError, match="unsupported expert.source"):
        _runtime_external_phi(
            replace(cfg, expert=RuntimeExpertConfig(source="bad_source", phi_ext=1.0))
        )


def test_runtime_build_geometry_vmec_and_miller_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _base_cfg()
    captured: list[tuple[str, str | None]] = []

    def _fake_build(geom_cfg):
        captured.append((geom_cfg.model, geom_cfg.geometry_file))
        return geom_cfg

    vmec_path = tmp_path / "vmec.eik.nc"
    miller_path = tmp_path / "miller.eik.nc"
    vmec_path.write_bytes(b"x")
    miller_path.write_bytes(b"x")

    monkeypatch.setattr("gkx.runtime.build_flux_tube_geometry", _fake_build)
    monkeypatch.setattr("gkx.runtime.generate_runtime_vmec_eik", lambda _cfg: vmec_path)
    monkeypatch.setattr(
        "gkx.runtime.generate_runtime_miller_eik", lambda _cfg: miller_path
    )

    vmec_geom = runtime._runtime_geometry_config_for_builder(
        replace(cfg, geometry=GeometryConfig(model="vmec"))
    )
    miller_geom = runtime._runtime_geometry_config_for_builder(
        replace(cfg, geometry=GeometryConfig(model="miller"))
    )
    default_geom = runtime._runtime_geometry_config_for_builder(cfg)

    assert (vmec_geom.model, vmec_geom.geometry_file) == ("vmec-eik", str(vmec_path))
    assert (miller_geom.model, miller_geom.geometry_file) == (
        "imported-eik",
        str(miller_path),
    )
    assert default_geom is cfg.geometry

    build_runtime_geometry(replace(cfg, geometry=GeometryConfig(model="vmec")))
    build_runtime_geometry(replace(cfg, geometry=GeometryConfig(model="miller")))
    build_runtime_geometry(cfg)
    assert captured[0] == ("vmec-eik", str(vmec_path))
    assert captured[1] == ("imported-eik", str(miller_path))
    assert captured[2][0] == cfg.geometry.model


def test_runtime_initial_state_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    z = np.linspace(-1.0, 1.0, 5)
    profile = _build_gaussian_profile(
        z,
        kx=0.2,
        ky=0.1,
        s_hat=0.8,
        width=0.5,
        envelope_constant=1.0,
        envelope_sine=0.2,
    )
    assert profile.shape == z.shape
    assert np.allclose(
        _build_gaussian_profile(
            z,
            kx=0.2,
            ky=0.0,
            s_hat=0.8,
            width=0.5,
            envelope_constant=1.0,
            envelope_sine=0.2,
        ),
        np.zeros_like(z),
    )

    raw = np.arange(2 * 3 * 2 * 4 * 5, dtype=np.float32).astype(np.complex64)
    reshaped = _reshape_netcdf_state(raw, nspec=1, nl=2, nm=3, nyc=2, nx=4, nz=5)
    assert reshaped.shape == (1, 2, 3, 2, 4, 5)

    expanded = _expand_ky(np.ones((1, 2, 3, 4, 5), dtype=np.complex64), nyc=3)
    assert expanded.shape[-3] == 4
    assert (
        _expand_ky(np.ones((1, 2, 3, 4, 5), dtype=np.complex64), nyc=2).shape[-3] == 3
    )


def test_runtime_single_mode_init_populates_zonal_ky0_branch() -> None:
    cfg = replace(
        _base_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=True,
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.max(np.abs(g0)) > 0.0
    assert np.max(np.abs(g0[0, 0, 0, 0, 1, :])) > 0.0


def test_runtime_gaussian_single_mode_init_populates_zonal_ky0_branch() -> None:
    cfg = replace(
        _base_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            init_single=True,
            gaussian_width=0.35,
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.max(np.abs(g0)) > 0.0
    assert np.max(np.abs(g0[0, 0, 0, 0, 1, :])) > 0.0


def test_runtime_initial_state_loading_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = np.zeros((1, 1, 1, 4, 4, 2), dtype=np.complex64)
    full[..., 1, :, :] = 1.0 + 2.0j
    herm = _enforce_full_ky_hermitian(full)
    assert herm.shape == full.shape
    assert np.allclose(
        _enforce_full_ky_hermitian(np.ones((1, 1, 1, 1, 2), dtype=np.complex64)),
        np.ones((1, 1, 1, 1, 2), dtype=np.complex64),
    )

    nc_path = tmp_path / "restart.nc"
    monkeypatch.setattr(
        "gkx.runtime.load_netcdf_restart_state",
        lambda *_args, **_kwargs: np.ones((1, 2, 3, 4, 4, 5), dtype=np.complex64),
    )
    assert _load_initial_state_from_file(
        nc_path, nspecies=1, Nl=2, Nm=3, ny=4, nx=4, nz=5
    ).shape == (1, 2, 3, 4, 4, 5)

    ny = 4
    nx = 4
    nz = 5
    nyc = ny // 2 + 1
    nyc_raw = np.ones(1 * 2 * 3 * nyc * nx * nz, dtype=np.complex64)
    nyc_path = tmp_path / "restart.bin"
    nyc_raw.tofile(nyc_path)
    assert _load_initial_state_from_file(
        nyc_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
    ).shape == (1, 2, 3, 4, 4, 5)

    full_raw = np.ones(1 * 2 * 3 * ny * nx * nz, dtype=np.complex64)
    full_path = tmp_path / "restart_full.bin"
    full_raw.tofile(full_path)
    assert _load_initial_state_from_file(
        full_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
    ).shape == (1, 2, 3, 4, 4, 5)

    bad_path = tmp_path / "restart_bad.bin"
    np.ones(7, dtype=np.complex64).tofile(bad_path)
    with pytest.raises(ValueError):
        _load_initial_state_from_file(
            bad_path, nspecies=1, Nl=2, Nm=3, ny=ny, nx=nx, nz=nz
        )


def test_runtime_initial_condition_validation_branches() -> None:
    cfg = _base_cfg()
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, gaussian_width=0.0)),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, init_file_mode="bad")),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, init_field="bad")),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )


def test_run_runtime_scan_batch_validation_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg,
            np.asarray([], dtype=float),
            Nl=2,
            Nm=3,
            method="rk2",
            dt=0.1,
            steps=2,
            sample_stride=1,
            auto_window=True,
            tmin=None,
            tmax=None,
            window_fraction=0.4,
            min_points=2,
            start_fraction=0.0,
            growth_weight=0.0,
            require_positive=False,
            min_amp_fraction=0.0,
            mode_method="project",
            fit_signal="phi",
            show_progress=False,
        )

    grid = build_spectral_grid(cfg.grid)
    geom = object()
    params = type("Params", (), {"rho_star": np.asarray(1.0)})()
    monkeypatch.setattr("gkx.runtime.build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        "gkx.runtime.apply_geometry_grid_defaults",
        lambda _geom, grid_cfg: grid_cfg,
    )
    monkeypatch.setattr("gkx.runtime.build_spectral_grid", lambda _cfg: grid)
    monkeypatch.setattr(
        "gkx.runtime.build_runtime_linear_params",
        lambda *_args, **_kwargs: params,
    )
    monkeypatch.setattr("gkx.runtime.build_runtime_linear_terms", lambda _cfg: object())
    monkeypatch.setattr(
        "gkx.runtime._build_initial_condition",
        lambda *_args, **_kwargs: np.ones(
            (1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        "gkx.runtime.integrate_linear_diagnostics",
        lambda *_args, **_kwargs: (
            None,
            np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
            2.0
            * np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "gkx.runtime.extract_mode_time_series",
        lambda arr, sel, method="project": np.asarray(
            arr[:, sel.ky_index, sel.kx_index, 0]
        ),
    )
    monkeypatch.setattr(
        "gkx.runtime.fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            0.2,
            0.3,
            0.0,
            0.2,
            2.0 if np.max(np.abs(signal)) < 1.5 else 1.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        "gkx.runtime.fit_growth_rate_auto",
        lambda *args, **kwargs: (0.4, 0.5, 0.0, 0.2),
    )
    monkeypatch.setattr(
        "gkx.runtime.fit_growth_rate", lambda *args, **kwargs: (0.6, 0.7)
    )
    monkeypatch.setattr(
        "gkx.runtime.apply_diagnostic_normalization",
        lambda g, o, **kwargs: (g, o),
    )

    scan_auto = _run_runtime_scan_batch(
        cfg,
        np.asarray([0.1, 0.2], dtype=float),
        Nl=2,
        Nm=3,
        method="rk2",
        dt=0.1,
        steps=2,
        sample_stride=1,
        auto_window=True,
        tmin=None,
        tmax=None,
        window_fraction=0.4,
        min_points=2,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=False,
        min_amp_fraction=0.0,
        mode_method="project",
        fit_signal="auto",
        show_progress=False,
    )
    assert scan_auto.gamma.shape == (2,)

    scan_density = _run_runtime_scan_batch(
        cfg,
        np.asarray([0.1], dtype=float),
        Nl=2,
        Nm=3,
        method="rk2",
        dt=0.1,
        steps=2,
        sample_stride=1,
        auto_window=False,
        tmin=0.0,
        tmax=0.2,
        window_fraction=0.4,
        min_points=2,
        start_fraction=0.0,
        growth_weight=0.0,
        require_positive=False,
        min_amp_fraction=0.0,
        mode_method="project",
        fit_signal="density",
        show_progress=False,
    )
    assert np.allclose(scan_density.gamma, np.array([0.6]))

    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg,
            np.asarray([0.1], dtype=float),
            Nl=2,
            Nm=3,
            method="rk2",
            dt=0.1,
            steps=2,
            sample_stride=1,
            auto_window=True,
            tmin=None,
            tmax=None,
            window_fraction=0.4,
            min_points=2,
            start_fraction=0.0,
            growth_weight=0.0,
            require_positive=False,
            min_amp_fraction=0.0,
            mode_method="project",
            fit_signal="invalid",
            show_progress=False,
        )


def test_run_runtime_scan_default_parallel_config_keeps_serial_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = _base_cfg()
    calls: list[float] = []

    def _unexpected_batch(*_args, **_kwargs):
        raise AssertionError(
            "default runtime parallel config should not use the combined-ky batch path"
        )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(gamma=10.0 + ky, omega=-(20.0 + ky), quasilinear=None)

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _unexpected_batch)
    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    result = run_runtime_scan(
        cfg,
        [0.3, 0.1],
        solver="time",
        workers=1,
        parallel_executor="thread",
        show_progress=False,
    )

    np.testing.assert_allclose(calls, [0.3, 0.1])
    np.testing.assert_allclose(result.ky, [0.3, 0.1])
    np.testing.assert_allclose(result.gamma, [10.3, 10.1])
    np.testing.assert_allclose(result.omega, [-20.3, -20.1])
    assert result.parallel is not None
    assert result.parallel["requested_workers"] == 1
    assert result.parallel["effective_workers"] == 1
    assert result.parallel["executor"] == "thread"
    assert "serial ky ordering" in result.parallel["identity_contract"]
    assert result.parallel["quasilinear_state_extraction"] is False


def test_run_runtime_scan_collects_quasilinear_payloads_and_worker_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = _base_cfg()
    calls: list[float] = []

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(
            gamma=1.0 + ky,
            omega=-(2.0 + ky),
            quasilinear={
                "ky": ky,
                "heat_flux_weight_total": 10.0 * ky,
                "claim_level": "bounded_unit_contract",
            },
        )

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    result = run_runtime_scan(
        cfg,
        [0.4, 0.2, 0.1],
        solver="time",
        workers=8,
        parallel_executor="thread",
        show_progress=False,
    )

    # Parallel (thread) executor: worker invocation order is nondeterministic,
    # so assert every ky was dispatched exactly once rather than a fixed order.
    # The input-order contract is verified below on the reassembled results.
    np.testing.assert_allclose(sorted(calls), [0.1, 0.2, 0.4])
    np.testing.assert_allclose(result.gamma, [1.4, 1.2, 1.1])
    assert result.quasilinear is not None
    assert [payload["ky"] for payload in result.quasilinear] == [0.4, 0.2, 0.1]
    assert result.parallel is not None
    assert result.parallel["requested_workers"] == 8
    assert result.parallel["effective_workers"] == 3
    assert result.parallel["quasilinear_state_extraction"] is True


def test_run_runtime_scan_parallel_config_requests_combined_ky_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        parallel=RuntimeParallelConfig(strategy="combined_ky", axis="ky"),
    )
    captured: dict[str, object] = {}
    sentinel = SimpleNamespace(
        ky=np.asarray([0.2, 0.4]),
        gamma=np.asarray([0.1, 0.2]),
        omega=np.asarray([-0.3, -0.4]),
        quasilinear=None,
    )

    def _fake_batch(_cfg, ky_arr, **kwargs):
        captured["ky_arr"] = np.asarray(ky_arr)
        captured["solverless_kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _fake_batch)
    monkeypatch.setattr(
        runtime,
        "run_runtime_linear",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("combined-ky scan must not dispatch per-ky workers")
        ),
    )

    result = run_runtime_scan(
        cfg,
        [0.2, 0.4],
        solver="time",
        method="rk2",
        sample_stride=2,
        show_progress=True,
    )

    assert result is sentinel
    np.testing.assert_allclose(captured["ky_arr"], [0.2, 0.4])
    assert captured["solverless_kwargs"]["method"] == "rk2"
    assert captured["solverless_kwargs"]["sample_stride"] == 2
    assert captured["solverless_kwargs"]["show_progress"] is True


def test_run_runtime_nonlinear_final_state_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_final_state(*args, **kwargs):
        captured["show_progress"] = kwargs.get("show_progress")
        return (
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(runtime, "integrate_nonlinear_from_config", _fake_final_state)

    out = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        diagnostics=False,
        show_progress=True,
    )

    assert captured["show_progress"] is True
    assert out.diagnostics is None
    assert out.phi2 is not None
    assert out.state is None


def test_run_runtime_nonlinear_final_state_without_progress_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_final_state(*args, **kwargs):
        captured["n_args"] = len(args)
        captured["show_progress"] = kwargs.get("show_progress", None)
        return (
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(runtime, "integrate_nonlinear_from_config", _fake_final_state)

    out = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        diagnostics=False,
        show_progress=False,
    )

    assert captured["n_args"] == 5
    assert captured["show_progress"] is None
    assert out.diagnostics is None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_return_state_uses_diagnostics_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        t = np.asarray([0.1, 0.2], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[-1]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        diagnostics=False,
        return_state=True,
    )

    assert out.diagnostics is None
    assert out.state is not None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_fixed_mode_requires_indices() -> None:
    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=None, ikx_fixed=None),
    )
    with pytest.raises(ValueError, match="expert.iky_fixed and expert.ikx_fixed"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4)


def test_run_runtime_nonlinear_rejects_unknown_external_source() -> None:
    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(source="bad_source"),
    )
    with pytest.raises(ValueError, match="unsupported expert.source"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4)


def test_run_runtime_nonlinear_adaptive_chunk_requires_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            _base_cfg().time, fixed_dt=False, t_max=0.3, dt=0.1, diagnostics=True
        ),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        t = np.asarray([0.0], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=np.asarray([0.1], dtype=float),
            dt_mean=float(0.1),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.zeros(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    with pytest.raises(RuntimeError, match="made no time-step progress"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=True)


def test_run_runtime_nonlinear_phiext_source_uses_diagnostics_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        expert=RuntimeExpertConfig(source="phiext_full", phi_ext=0.25),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        captured.update(kwargs)
        t = np.asarray([0.1, 0.2], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[-1]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.asarray([1.0, 1.1]),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=False)

    assert captured["external_phi"] == pytest.approx(0.25)
    assert out.diagnostics is None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_adaptive_chunk_forwards_fixed_mode_and_collision_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    base = _base_cfg()
    cfg = replace(
        base,
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        time=replace(
            base.time,
            fixed_dt=False,
            t_max=0.35,
            dt=0.1,
            diagnostics=True,
            collision_split=True,
            collision_scheme="exp",
        ),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0),
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: type("P", (), {"rho_star": np.asarray(1.0)})(),
    )
    monkeypatch.setattr(runtime, "build_runtime_term_config", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "_select_nonlinear_mode_indices", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_diag_integrator(*args, **kwargs):
        captured.append(dict(kwargs))
        t = np.asarray([0.1, 0.2], dtype=float)
        if kwargs.get("time_horizon") is not None:
            t = np.minimum(t, float(kwargs["time_horizon"]))
        dt_t = np.diff(np.insert(t, 0, 0.0))
        diag = SimulationDiagnostics(
            t=t,
            dt_t=dt_t,
            dt_mean=float(np.mean(dt_t)),
            gamma_t=np.asarray([0.0, 0.0], dtype=float),
            omega_t=np.asarray([0.0, 0.0], dtype=float),
            Wg_t=np.asarray([0.0, 0.0], dtype=float),
            Wphi_t=np.asarray([1.0, 1.2], dtype=float),
            Wapar_t=np.asarray([0.0, 0.0], dtype=float),
            heat_flux_t=np.asarray([0.0, 0.0], dtype=float),
            particle_flux_t=np.asarray([0.0, 0.0], dtype=float),
            energy_t=np.asarray([0.0, 0.0], dtype=float),
        )
        return (
            t,
            diag,
            np.ones(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=np.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
                ),
                apar=None,
                bpar=None,
            ),
        )

    monkeypatch.setattr(
        runtime, "integrate_nonlinear_explicit_diagnostics_state", _fake_diag_integrator
    )

    out = run_runtime_nonlinear(
        cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=True, show_progress=True
    )

    assert len(captured) == 2
    assert captured[0]["fixed_dt"] is False
    assert captured[0]["collision_split"] is True
    assert captured[0]["collision_scheme"] == "exp"
    assert captured[0]["fixed_mode_ky_index"] == 1
    assert captured[0]["fixed_mode_kx_index"] == 0
    assert captured[0]["show_progress"] is True
    assert out.diagnostics is not None
    assert out.state is None
    assert out.fields is not None
    np.testing.assert_allclose(out.diagnostics.t, [0.1, 0.2, 0.3, 0.35])


def test_runtime_nonlinear_result_summary_contracts() -> None:
    from gkx.workflows.runtime.results import (
        build_runtime_nonlinear_result,
        nonlinear_field_phi2,
    )

    fields = FieldState(
        phi=np.asarray(
            [
                [1.0 + 0.0j, 0.0 + 2.0j],
                [3.0 + 4.0j, 0.0 + 0.0j],
            ],
            dtype=np.complex64,
        )
    )
    np.testing.assert_allclose(nonlinear_field_phi2(fields), np.asarray(7.5))

    summary_fields = FieldState(phi=np.asarray([1.0 + 1.0j, 2.0 + 0.0j]))
    state = np.asarray([3.0])
    summarized = build_runtime_nonlinear_result(
        t=np.asarray([0.1, 0.2]),
        diagnostics=None,
        fields=summary_fields,
        state=state,
        ky_selected=0.3,
        kx_selected=-0.5,
        summarize_fields=True,
    )
    assert summarized.t.size == 0
    assert summarized.diagnostics is None
    assert summarized.fields is summary_fields
    assert summarized.state is state
    assert summarized.ky_selected == pytest.approx(0.3)
    assert summarized.kx_selected == pytest.approx(-0.5)
    np.testing.assert_allclose(summarized.phi2, np.asarray(3.0))

    t = np.asarray([0.1, 0.2])
    preserved = build_runtime_nonlinear_result(
        t=t,
        diagnostics=None,
        fields=None,
        state=None,
        ky_selected=None,
        kx_selected=None,
        summarize_fields=False,
    )
    np.testing.assert_allclose(preserved.t, t)
    assert preserved.diagnostics is None
    assert preserved.phi2 is None
    assert preserved.fields is None

    with pytest.raises(RuntimeError, match="final fields are required"):
        build_runtime_nonlinear_result(
            t=np.asarray([0.1]),
            diagnostics=None,
            fields=None,
            state=None,
            ky_selected=None,
            kx_selected=None,
            summarize_fields=True,
        )


def test_warn_if_growth_unresolved_flags_short_runs() -> None:
    """A run shorter than five e-foldings is reported as under-resolved."""

    t = np.linspace(0.0, 10.0, 101)

    with pytest.warns(RuntimeWarning, match="growth under-resolved"):
        message = warn_if_growth_unresolved(
            gamma=0.2, t=t, fit_window_tmin=0.0, fit_window_tmax=10.0
        )
    # The message reports the achieved e-folding count and the horizon that
    # would reach the gamma*t_max >= 7 threshold (7 / 0.2 = 35).
    assert message is not None
    assert "2.00 e-foldings" in message
    assert "t_max >~ 35" in message

    # Long enough overall, but the selected window covers under two growth
    # times, so the fit itself is only marginally constrained.
    with pytest.warns(RuntimeWarning, match="marginally resolved"):
        message = warn_if_growth_unresolved(
            gamma=1.0, t=t, fit_window_tmin=9.0, fit_window_tmax=10.0
        )
    assert message is not None and "1.00 growth times" in message


def test_warn_if_growth_unresolved_stays_quiet_when_resolved() -> None:
    """Resolved growth and decaying modes produce no warning."""

    t = np.linspace(0.0, 100.0, 1001)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # gamma*t_max = 20 >= 7, and the window spans 10 growth times.
        assert (
            warn_if_growth_unresolved(
                gamma=0.2, t=t, fit_window_tmin=50.0, fit_window_tmax=100.0
            )
            is None
        )
        # A damped mode has no growth to resolve.
        assert (
            warn_if_growth_unresolved(
                gamma=-0.2, t=t, fit_window_tmin=0.0, fit_window_tmax=1.0
            )
            is None
        )


def test_half_horizon_settled_probe_detects_an_unsettled_fit() -> None:
    """Halving the window must not move gamma once the mode dominates."""

    t = np.linspace(0.0, 40.0, 800)

    settled_signal = np.exp((0.2 - 1j * 0.3) * t)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        drift, settled = half_horizon_settled_probe(
            t, settled_signal, gamma=0.2, tmin=10.0, tmax=40.0
        )
    assert settled is True
    assert drift == pytest.approx(0.0, abs=1e-6)

    # A dominant branch still emerging from a decaying transient: the second
    # half of the window fits a visibly larger gamma than the whole window.
    ramping = np.exp((0.2 - 1j * 0.3) * t) + 50.0 * np.exp((-0.15 - 1j * 1.0) * t)
    with pytest.warns(RuntimeWarning, match="fit not settled"):
        drift, settled = half_horizon_settled_probe(
            t, ramping, gamma=0.10, tmin=0.0, tmax=40.0
        )
    assert settled is False
    assert drift > 0.05


def test_half_horizon_settled_probe_declines_degenerate_input() -> None:
    t = np.linspace(0.0, 10.0, 50)
    signal = np.exp(0.1 * t)
    assert half_horizon_settled_probe(t, signal, gamma=0.0, tmin=None, tmax=None) == (
        None,
        None,
    )
    assert half_horizon_settled_probe(t, signal, gamma=0.1, tmin=5.0, tmax=1.0) == (
        None,
        None,
    )


def test_warm_start_carry_state_rescales_without_losing_a_bit() -> None:
    """A carried state differs from the converged one by an exact power of two."""

    converged = np.asarray(
        [[1.0 + 2.0j, -3.0 + 0.5j], [0.25 - 4.0j, 7.0 + 0.0j]], dtype=np.complex64
    ) * np.float32(1.0e18)

    carried = warm_start.carry_state(converged)

    assert carried is not None
    assert carried.dtype == converged.dtype
    assert float(np.linalg.norm(carried)) == pytest.approx(1.0, rel=0.5)
    ratio = converged.reshape(-1)[0] / carried.reshape(-1)[0]
    # Exact, not approximate: every entry is scaled by the same binary exponent.
    np.testing.assert_array_equal(carried * ratio, converged)


def test_warm_start_refuses_states_that_cannot_seed_a_solve() -> None:
    assert warm_start.carry_state(None) is None
    assert warm_start.carry_state(np.zeros((2, 2), dtype=np.complex64)) is None
    assert warm_start.carry_state(np.asarray([], dtype=np.complex64)) is None
    assert warm_start.carry_state(np.asarray([np.nan, 1.0])) is None
    assert warm_start.carry_state(np.asarray([np.inf, 1.0])) is None
    with pytest.raises(ValueError, match="amplitude must be positive"):
        warm_start.carry_state(np.asarray([1.0]), amplitude=0.0)


def test_warm_start_visit_order_puts_neighbours_next_to_each_other() -> None:
    order = warm_start.scan_visit_order([0.5, 0.1, 0.4, 0.2])
    np.testing.assert_array_equal(order, [1, 3, 2, 0])
    values = np.asarray([0.5, 0.1, 0.4, 0.2])[order]
    assert np.all(np.diff(values) > 0.0)


def test_warm_start_refusal_names_the_contract_it_protects() -> None:
    enabled = warm_start.WarmStartPolicy(enabled=True)
    assert (
        warm_start.linear_scan_warm_start_refusal(
            policy=enabled, solver_key="krylov", workers=1
        )
        is None
    )
    assert "disabled" in warm_start.linear_scan_warm_start_refusal(
        policy=warm_start.WarmStartPolicy(enabled=False), solver_key="krylov", workers=1
    )
    assert "final state" in warm_start.linear_scan_warm_start_refusal(
        policy=enabled, solver_key="explicit_time", workers=1
    )
    assert "independent" in warm_start.linear_scan_warm_start_refusal(
        policy=enabled, solver_key="krylov", workers=4
    )


def test_warm_start_policy_reads_output_section_and_honors_override() -> None:
    cfg = RuntimeConfig()
    # Opt-in: nothing warm starts until something asks for it.
    assert warm_start.WarmStartPolicy.from_config(cfg).enabled is False
    assert warm_start.WarmStartPolicy.from_config(cfg, override=True).enabled is True
    on = replace(cfg, output=replace(cfg.output, warm_start=True))
    assert warm_start.WarmStartPolicy.from_config(on).enabled is True
    assert warm_start.WarmStartPolicy.from_config(on, override=False).enabled is False


def test_resolve_scan_warm_start_prefers_flag_then_scan_section() -> None:
    flag_on = SimpleNamespace(warm_start=True)
    assert warm_start.resolve_scan_warm_start(flag_on, {"warm_start": False}) is True
    flag_off = SimpleNamespace(warm_start=False)
    assert warm_start.resolve_scan_warm_start(flag_off, {"warm_start": True}) is False
    unset = SimpleNamespace(warm_start=None)
    assert warm_start.resolve_scan_warm_start(unset, {"warm_start": True}) is True
    assert warm_start.resolve_scan_warm_start(unset, {}) is None
    assert warm_start.resolve_scan_warm_start(SimpleNamespace(), {}) is None


def test_saturation_refresh_policy_bounds_reuse_and_geometry_drift() -> None:
    cache = warm_start.SaturationWarmStart(
        policy=warm_start.SaturationRefreshPolicy(
            max_reuse=2, geometry_tolerance=0.05, warm_step_fraction=0.25
        )
    )
    base = np.asarray([1.0, 2.0, 3.0])
    state = np.ones((2, 2), dtype=np.complex64)

    first = cache.plan(base, cold_steps=8000)
    assert (first.warm, first.steps, first.seed) == (False, 8000, None)
    cache.record(state, base, warm=first.warm)

    # Geometry barely moved: reuse the saturated state on a reduced budget.
    near = base * 1.001
    second = cache.plan(near, cold_steps=8000)
    assert second.warm is True
    assert second.steps == 2000
    np.testing.assert_array_equal(second.seed, state)
    cache.record(state, near, warm=second.warm)

    # A large geometry step invalidates the attractor and restores cold cost.
    far = base + 10.0
    third = cache.plan(far, cold_steps=8000)
    assert (third.warm, third.steps) == (False, 8000)
    assert "geometry moved" in third.reason

    # The reuse budget alone forces a cold spin-up even when nothing moved.
    cache.record(state, near, warm=True)
    cache.record(state, near, warm=True)
    exhausted = cache.plan(near, cold_steps=8000)
    assert exhausted.warm is False
    assert exhausted.reason == "reuse budget exhausted"


def test_saturation_refresh_policy_rejects_unusable_state_and_bad_bounds() -> None:
    cache = warm_start.SaturationWarmStart()
    cache.record(np.zeros((2, 2), dtype=np.complex64), np.asarray([1.0]), warm=False)
    plan = cache.plan(np.asarray([1.0]), cold_steps=100)
    assert plan.warm is False
    assert plan.reason == "no usable saved state"
    with pytest.raises(ValueError, match="cold_steps"):
        cache.plan(np.asarray([1.0]), cold_steps=0)
    with pytest.raises(ValueError, match="warm_step_fraction"):
        warm_start.SaturationRefreshPolicy(warm_step_fraction=0.0)
    with pytest.raises(ValueError, match="max_reuse"):
        warm_start.SaturationRefreshPolicy(max_reuse=-1)
    with pytest.raises(ValueError, match="geometry_tolerance"):
        warm_start.SaturationRefreshPolicy(geometry_tolerance=-1.0)


def test_warm_start_geometry_signature_tracks_the_metric_profiles() -> None:
    geom = SimpleNamespace(
        gradpar_value=0.25,
        bmag_profile=np.asarray([1.0, 1.1]),
        gds2_profile=np.asarray([2.0, 2.2]),
        unrelated=np.asarray([9.0]),
    )
    signature = warm_start.flux_tube_signature(geom)
    np.testing.assert_allclose(signature, [0.25, 1.0, 1.1, 2.0, 2.2])
    assert warm_start.relative_change(signature, signature) == 0.0
    assert warm_start.relative_change(signature, signature[:-1]) == float("inf")
    assert warm_start.relative_change([1.0], [np.nan]) == float("inf")
    with pytest.raises(ValueError, match="signature profiles"):
        warm_start.flux_tube_signature(SimpleNamespace(other=1.0))
    with pytest.raises(ValueError, match="at least one array"):
        warm_start.signature_from_arrays([])
