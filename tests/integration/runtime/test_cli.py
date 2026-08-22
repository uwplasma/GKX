"""Executable tests for basic command execution."""

import argparse
from dataclasses import dataclass
import os
import sys
from gkx.utils import tomlcompat as tomllib
from pathlib import Path

from support.paths import REPO_ROOT

import numpy as np
import pytest

from gkx import __version__
import gkx.cli as cli
from gkx.diagnostics.analysis import ModeSelection
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
from gkx.diagnostics import SimulationDiagnostics
from gkx.runtime import RuntimeLinearResult, RuntimeNonlinearResult
from gkx.workflows.runtime.config import RuntimeConfig


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
    assert "--plot OUTPUT_FILE" in help_text


def test_cli_global_plot_uses_saved_output_renderer(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    rendered = tmp_path / "rendered.png"
    monkeypatch.setattr(
        "gkx.cli.plot_saved_output", lambda path, out=None: rendered
    )
    monkeypatch.setattr(
        sys, "argv", ["gkx", "--plot", "tools_out/linear_case.summary.json"]
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
    monkeypatch.setattr(sys, "argv", ["gkx", "--plot"])
    assert main() == 1
    assert "usage: gkx --plot" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["gkx", "--plot", "a", "--bad"])
    assert main() == 1
    assert "usage: gkx --plot" in capsys.readouterr().out


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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false
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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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
use_diffrax = false

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


# ---------------------------------------------------------------------------
# Automatic figures: a completed run draws its own output.
# ---------------------------------------------------------------------------


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
    _stub_nonlinear_run(
        monkeypatch, {"out": str(bundle), "summary": str(summary)}
    )

    assert _cmd_run_runtime_nonlinear(_nonlinear_namespace(out=str(bundle))) == 0

    assert captured["window"] == (3.0, 6.0)
    assert measured_average_window({"fit_window_tmin": 1.0, "fit_window_tmax": 2.0}) == (
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
    _stub_nonlinear_run(
        monkeypatch, {"out": str(bundle), "summary": str(summary)}
    )

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


# ---------------------------------------------------------------------------
# Persistent compilation cache: cold start is the cost users actually pay.
# ---------------------------------------------------------------------------


def test_compilation_cache_directory_honours_the_environment(tmp_path: Path) -> None:
    from gkx.utils import compilation_cache as cache

    override = tmp_path / "elsewhere"
    assert cache.compilation_cache_directory(
        {cache.DIRECTORY_ENV_VAR: str(override)}
    ) == override
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




# ---- wout equilibrium shorthand ------------------------------------------


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
    monkeypatch.setattr(
        "gkx.cli.write_runtime_linear_scan_artifacts", _fake_write_scan
    )
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
