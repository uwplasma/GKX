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
