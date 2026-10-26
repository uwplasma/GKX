"""Runtime user-facing surface: the gkx CLI commands and the runtime command/orchestration helpers they dispatch into."""

from __future__ import annotations

from support.helpers import patch_runtime
from support.helpers import (
    linear_result,
    nonlinear_result,
    patch_nonlinear_setup,
    runtime_deck,
    scan_stub,
    simple_diag,
    state_and_fields,
    zero_field_diag,
)

from dataclasses import dataclass
from dataclasses import replace
from gkx import __version__
from gkx.cli import (
    _cmd_run,
    _cmd_run_runtime_linear,
    _cmd_run_runtime_nonlinear,
    _cmd_scan_runtime_linear,
    _direct_config_shorthand_args,
    main,
)
from gkx.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
)
from gkx.core_grid import build_spectral_grid
from gkx.diagnostics import ResolvedDiagnostics, SimulationDiagnostics
from gkx.diagnostics.analysis import ModeSelection
from gkx.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_terms,
    run_runtime_nonlinear,
    run_runtime_scan,
    RuntimeLinearResult,
)
from gkx.workflows.runtime.startup import (
    _centered_glibc_random_pairs,
    _default_hermite_hypercollision_exponent,
    _dealiased_initial_mode_pairs,
    _periodic_zp_from_grid,
    _runtime_model_key,
)
from gkx.workflows.runtime.diagnostic_arrays import (
    concat_runtime_diagnostics as _concat_runtime_diagnostics,
    slice_runtime_diagnostics as _slice_runtime_diagnostics,
    stride_runtime_diagnostics as _stride_runtime_diagnostics,
)
from gkx.workflows.nonlinear import _infer_runtime_nonlinear_steps
from gkx.workflows.linear import (
    _midplane_index,
    _normalize_linear_solver_name,
)
from gkx.workflows.runtime.orchestration_scan import (
    run_runtime_scan_batch as _run_runtime_scan_batch,
)
from gkx.terms.config import FieldState
from gkx.workflows.runtime.diagnostics import (
    finalize_runtime_linear_quasilinear,
    half_horizon_settled_probe,
    warn_if_growth_unresolved,
    _prepare_runtime_linear_fit_inputs,
    fit_runtime_linear_diagnostics,
)
from pathlib import Path
from support.paths import REPO_ROOT
from types import SimpleNamespace
import argparse
import gkx.workflows.runtime.commands as runtime_commands
import gkx.workflows.runtime.artifacts as runtime_artifacts
import gkx.workflows.runtime.orchestration_scan as runtime_policies
import gkx.workflows.runtime.warm_start as warm_start
import json
import numpy as np
import pytest
import sys
import tomllib
import warnings


def _project_version() -> str:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]


_RUN_NL = "[run]\nky = 0.2\nNl = 3\nNm = 4\n"
_GEOM_VMEC_FILE = (
    'q = 1.4\ns_hat = 0.8\nepsilon = 0.18\nR0 = 2.77778\nvmec_file = "toml_wout.nc"'
)


def test_version_exposed():
    """Version string should be exported from the package."""
    assert __version__ == _project_version()


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

    patch_runtime(
        monkeypatch,
        "load_runtime_from_toml",
        lambda _path: (RuntimeConfig(), {"fit": {"fit_signal": "phi"}}),
    )

    def _fake_run(_cfg, **kwargs):
        run_kwargs.update(kwargs)
        return fake_result

    patch_runtime(monkeypatch, "run_runtime_linear", _fake_run)
    patch_runtime(
        monkeypatch,
        "linear_runtime_panel_figure",
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


def test_cli_global_plot_renders_linear_scan_bundle(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    """--plot should render the saved ky-scan bundle written by scan-runtime-linear."""

    scan = scan_stub([0.2, 0.3], [0.05, 0.08], [0.25, 0.31])
    paths = runtime_artifacts.write_runtime_linear_scan_artifacts(
        tmp_path / "scan", scan
    )
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

    patch_runtime(monkeypatch, "load_runtime_from_toml", _load_runtime)
    patch_runtime(monkeypatch, "generate_runtime_vmec_eik", _vmec)
    patch_runtime(monkeypatch, "generate_runtime_miller_eik", _miller)

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

    patch_runtime(monkeypatch, "load_toml", lambda _path: {"physics": {}})
    assert _direct_config_shorthand_args([str(cfg_path), "--no-progress"]) == [
        "run",
        "--config",
        str(cfg_path),
        "--no-progress",
    ]

    patch_runtime(monkeypatch, "load_toml", lambda _path: {"case": "cyclone"})
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

    patch_runtime(monkeypatch, "load_runtime_from_toml", _boom)
    assert _cmd_run(args) == 1
    assert "Error loading bad.toml" in capsys.readouterr().out

    nonlinear_cfg = type(
        "Cfg", (), {"physics": type("Phys", (), {"nonlinear": True})()}
    )()
    linear_cfg = type(
        "Cfg", (), {"physics": type("Phys", (), {"nonlinear": False})()}
    )()
    patch_runtime(
        monkeypatch, "load_runtime_from_toml", lambda _path: (nonlinear_cfg, {})
    )
    patch_runtime(monkeypatch, "_cmd_run_runtime_nonlinear", lambda _args: 7)
    assert _cmd_run(args) == 7

    patch_runtime(monkeypatch, "load_runtime_from_toml", lambda _path: (linear_cfg, {}))
    patch_runtime(monkeypatch, "_cmd_run_runtime_linear", lambda _args: 9)
    assert _cmd_run(args) == 9


def test_cmd_run_honours_fit_signal_from_the_fit_table(monkeypatch) -> None:
    """``[fit] fit_signal`` is honoured by ``gkx run``, as by the case helpers.

    The command path used to filter it out and fit ``auto`` instead, while
    ``run_runtime_case`` read the same deck key; unknown ``[fit]`` keys were
    dropped the same silent way.
    """

    captured: dict[str, object] = {}
    decks: list[dict[str, object]] = []

    def _load_runtime(_path):
        return RuntimeConfig(), decks[-1]

    def _run_runtime_linear(_cfg, **kwargs):
        captured.update(kwargs)
        return linear_result(
            z_index=0, t=np.asarray([0.0, 1.0]), signal=np.asarray([1.0, 1.2])
        )

    patch_runtime(monkeypatch, "load_runtime_from_toml", _load_runtime)
    patch_runtime(monkeypatch, "run_runtime_linear", _run_runtime_linear)
    args = argparse.Namespace(config="case.toml", no_progress=True)

    decks.append({"run": {"ky": 0.2}, "fit": {"fit_signal": "phi", "tmin": 1.0}})
    assert _cmd_run(args) == 0
    assert captured["fit_signal"] == "phi"
    assert captured["tmin"] == pytest.approx(1.0)

    decks.append({"run": {"ky": 0.2, "fit_signal": "density"}, "fit": {}})
    assert _cmd_run(args) == 0
    assert captured["fit_signal"] == "density"

    decks.append(
        {"run": {"ky": 0.2, "fit_signal": "density"}, "fit": {"fit_signal": "phi"}}
    )
    with pytest.raises(ValueError, match="fit_signal"):
        _cmd_run(args)

    decks.append({"run": {"ky": 0.2}, "fit": {"window": 0.3}})
    with pytest.raises(ValueError, match=r"\[fit\].*'window'"):
        _cmd_run(args)


def test_cmd_scan_runtime_linear_writes_quasilinear_spectrum(
    monkeypatch, capsys
) -> None:
    cfg = RuntimeConfig()
    scan = scan_stub(
        [0.1], [0.2], [-0.3], quasilinear=({"ky": 0.1, "heat_flux_weight_total": 1.0},)
    )
    captured: dict[str, object] = {}

    def _fake_run_runtime_scan(cfg_in, *_args, **kwargs):
        captured["quasilinear"] = cfg_in.quasilinear
        captured["kwargs"] = kwargs
        return scan

    patch_runtime(
        monkeypatch,
        "load_runtime_from_toml",
        lambda _path: (cfg, {"scan": {"ky": [0.1]}, "fit": {}}),
    )
    patch_runtime(monkeypatch, "run_runtime_scan", _fake_run_runtime_scan)
    patch_runtime(
        monkeypatch,
        "write_runtime_linear_scan_artifacts",
        lambda *_args, **_kwargs: {
            "summary": "scan.summary.json",
            "scan": "scan.csv",
            "quasilinear_spectrum": "scan.ql.csv",
        },
    )
    args = argparse.Namespace(
        config="case.toml",
        ky_values="0.1",
        workers=3,
        parallel_executor="thread",
        no_progress=True,
        out="scan_bundle",
        quasilinear=True,
        ql_mode="weights",
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
    diag = simple_diag(
        t=(0.1, 0.2),
        Wg=(1.0, 1.1),
        Wphi=(2.0, 2.1),
        heat=(3.0, 3.1),
        particle=(4.0, 4.1),
        energy=(3.0, 3.2),
        phi_mode_t=None,
    )
    result = nonlinear_result(diag)
    paths = {
        "summary": "a",
        "diagnostics": "b",
        "state": "c",
        "out": "d",
        "big": "e",
        "restart": "f",
    }
    patch_runtime(
        monkeypatch,
        "load_runtime_from_toml",
        lambda _path: (cfg, {"run": {"steps": 5}}),
    )
    patch_runtime(
        monkeypatch,
        "run_runtime_nonlinear_with_artifacts",
        lambda *args, **kwargs: (result, paths),
    )
    args = argparse.Namespace(
        config="case.toml", init_file=str(tmp_path / "seed.bin"), out="out.nc"
    )
    assert _cmd_run_runtime_nonlinear(args) == 0
    out = capsys.readouterr().out
    assert "starting runtime nonlinear run" in out
    assert "saved a" in out and "saved f" in out

    no_diag_result = nonlinear_result()
    patch_runtime(
        monkeypatch,
        "run_runtime_nonlinear_with_artifacts",
        lambda *args, **kwargs: (no_diag_result, {}),
    )
    args.no_diagnostics = True
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert "nonlinear run completed" in capsys.readouterr().out


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

    patch_runtime(
        monkeypatch, "load_runtime_from_toml", lambda _path: (cfg, {"run": {}})
    )
    patch_runtime(monkeypatch, "run_runtime_linear", _fake_run_runtime_linear)
    patch_runtime(
        monkeypatch,
        "write_quasilinear_artifacts",
        lambda *_args, **_kwargs: {
            "quasilinear_summary": "ql.json",
            "quasilinear_species": "ql.csv",
        },
    )

    args = argparse.Namespace(
        config="case.toml",
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
    result = nonlinear_result()

    def _runner(*_args, **kwargs):
        captured.update(kwargs)
        return result, {}

    patch_runtime(
        monkeypatch, "load_runtime_from_toml", lambda _path: (cfg, {"run": {}})
    )
    patch_runtime(monkeypatch, "run_runtime_nonlinear_with_artifacts", _runner)
    args = argparse.Namespace(config="case.toml", diagnostics=True)
    assert _cmd_run_runtime_nonlinear(args) == 0
    assert captured["steps"] == 5
    assert captured["diagnostics"] is True


@pytest.mark.parametrize(
    ("tail", "command"),
    [
        # The unified runtime nonlinear command runs a tiny configuration.
        pytest.param(_RUN_NL, ["run-runtime-nonlinear", "--config"], id="command"),
        # Direct ``gkx path/to/config.toml`` dispatches nonlinear configs.
        pytest.param("", [], id="shorthand"),
    ],
)
def test_cli_run_runtime_nonlinear(capsys, monkeypatch, tmp_path: Path, tail, command):
    path = tmp_path / "runtime_cli_nonlinear.toml"
    path.write_text(runtime_deck(t_max=0.1, nonlinear=True, tail=tail), "utf-8")
    monkeypatch.setattr(sys, "argv", ["gkx", *command, str(path), "--steps", "3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "nonlinear" in out


def test_cli_run_runtime_nonlinear_outputs_species_flux_columns(
    capsys, monkeypatch, tmp_path: Path
):
    """Nonlinear CSV output should include per-species flux diagnostics when available."""
    cfg = runtime_deck(t_max=0.1, nonlinear=True, tail=_RUN_NL)
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


_RUNTIME_NONLINEAR_TOML_MIN = runtime_deck(
    geometry=_GEOM_VMEC_FILE, nonlinear=True, tail=_RUN_NL + "steps = 1\n"
)


def _record_auto_plot(monkeypatch) -> list[tuple[str, dict[str, str]]]:
    """Capture what the executable asks the figure writer to draw."""

    calls: list[tuple[str, dict[str, str]]] = []

    def _fake(kind, paths, **_kwargs):
        calls.append((kind, dict(paths)))
        return [f"{kind}.png"]

    monkeypatch.setattr("gkx.artifacts.run_figures.auto_plot_saved_run", _fake)
    return calls


def _nonlinear_namespace(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(config="case.toml", out="case.out.nc")
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def _stub_nonlinear_run(monkeypatch, paths: dict[str, str], cfg=None) -> None:
    diag = simple_diag(
        t=(0.1, 0.2),
        Wg=(1.0, 1.1),
        Wphi=(2.0, 2.1),
        heat=(3.0, 3.1),
        particle=(4.0, 4.1),
        energy=(3.0, 3.2),
        phi_mode_t=None,
    )
    result = nonlinear_result(diag)
    patch_runtime(
        monkeypatch,
        "load_runtime_from_toml",
        lambda _path: (cfg if cfg is not None else RuntimeConfig(), {"run": {}}),
    )
    patch_runtime(
        monkeypatch,
        "run_runtime_nonlinear_with_artifacts",
        lambda *_args, **_kwargs: (result, paths),
    )


@pytest.mark.parametrize("via_config", [False, True], ids=["flag", "config"])
def test_cli_no_plots_suppresses_the_figure_set(monkeypatch, via_config) -> None:
    """``[output] plots = false`` is the configuration form of ``--no-plots``."""

    cfg = RuntimeConfig()
    cfg = replace(cfg, output=replace(cfg.output, plots=not via_config))
    calls = _record_auto_plot(monkeypatch)
    _stub_nonlinear_run(monkeypatch, {"summary": "s.json", "out": "c.out.nc"}, cfg=cfg)

    args = _nonlinear_namespace(no_plots=not via_config)
    assert _cmd_run_runtime_nonlinear(args) == 0

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

    from gkx.artifacts.foreign_output import is_gx_output

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

    from gkx.artifacts.foreign_output import gx_summary_figure

    gx_bundle = _write_grouped_out_nc(tmp_path / "gx_titled.out.nc", code="gx")
    fig, axes = gx_summary_figure(gx_bundle)
    try:
        assert "GX" in axes[0].get_title(loc="left")
    finally:
        plt.close(fig)


def test_compilation_cache_directory_honours_the_environment(tmp_path: Path) -> None:
    import gkx.compilation_cache as cache

    override = tmp_path / "elsewhere"
    assert (
        cache.compilation_cache_directory({cache.DIRECTORY_ENV_VAR: str(override)})
        == override
    )
    # No override: the repository-local .cache/gkx tree, matching where
    # generated *.eik.nc geometry already lives.
    assert cache.compilation_cache_directory({}).parts[-3:] == (".cache", "gkx", "jax")


def test_compilation_cache_can_be_switched_off(tmp_path: Path) -> None:
    import gkx.compilation_cache as cache

    assert cache.compilation_cache_enabled({}) is True
    for value in ("0", "off", "false", "no"):
        assert cache.compilation_cache_enabled({cache.ENABLE_ENV_VAR: value}) is False
    assert (
        cache.enable_persistent_compilation_cache(
            {cache.ENABLE_ENV_VAR: "0", cache.DIRECTORY_ENV_VAR: str(tmp_path)}
        )
        is None
    )


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


def test_cli_wout_linear_flag_runs_default_ky_scan(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    from gkx.workflows.runtime.wout import DEFAULT_LINEAR_KY_VALUES

    monkeypatch.chdir(tmp_path)
    wout = _write_tiny_wout(tmp_path / "wout_lin.nc")
    captured: dict[str, object] = {}
    scan = scan_stub([0.1], [0.2], [-0.3])

    def _fake_run_runtime_scan(cfg, ky_values, **kwargs):
        captured["cfg"] = cfg
        captured["ky_values"] = ky_values
        return scan

    def _fake_write_scan(base, scan_in, **_kwargs):
        captured["out_base"] = str(base)
        return {"summary": "scan.summary.json", "scan": "scan.csv"}

    patch_runtime(monkeypatch, "run_runtime_scan", _fake_run_runtime_scan)
    patch_runtime(monkeypatch, "write_runtime_linear_scan_artifacts", _fake_write_scan)
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


def test_only_the_deprecated_spellings_warn(capsys, monkeypatch) -> None:
    """``gkx run``/``gkx scan`` (and the deck shorthand) print no deprecation."""

    from gkx.cli import build_parser

    for target in (
        "run_runtime_linear_command",
        "scan_runtime_linear_command",
        "run_runtime_nonlinear_command",
    ):
        monkeypatch.setattr(f"gkx.cli.{target}", lambda _args: 0)
    for nonlinear in (False, True):
        cfg = replace(
            RuntimeConfig(),
            physics=replace(RuntimeConfig().physics, nonlinear=nonlinear),
        )
        patch_runtime(monkeypatch, "load_runtime_from_toml", lambda _p: (cfg, {}))
        for argv, warned in (
            (["run", "--config", "case.toml"], False),
            (["scan", "--config", "case.toml"], False),
            (["scan-runtime-linear", "--config", "case.toml"], True),
            (["run-runtime-linear", "--config", "case.toml"], True),
            (["run-runtime-nonlinear", "--config", "case.toml"], True),
        ):
            args = build_parser().parse_args(argv)
            assert args.func(args) == 0
            err = capsys.readouterr().err
            assert ("deprecated" in err) is warned, (argv, err)


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
        cfg_zero, species=(RuntimeSpeciesConfig(name="ion", nu=0.05),)
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
    assert _dealiased_initial_mode_pairs(grid)[0] == (0, 1)
    assert _periodic_zp_from_grid(np.asarray([0.0])) == 1.0
    assert _periodic_zp_from_grid(np.asarray([0.0, 0.0])) == 1.0
    assert _default_hermite_hypercollision_exponent(None) == 20.0
    assert _default_hermite_hypercollision_exponent(3) == 1.0
    assert _default_hermite_hypercollision_exponent(40) == 20.0
    assert _runtime_model_key(cfg) == "gyrokinetic"


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
                [[True, True, True], [True, False, True], [False, False, False]],
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
        == 8  # t_max / the overriding dt, not / the deck dt
    )
    adaptive_cfg = replace(cfg, time=replace(cfg.time, fixed_dt=False, dt_max=None))
    assert _infer_runtime_nonlinear_steps(adaptive_cfg, dt=0.2, steps=None) == 2
    with pytest.raises(ValueError):
        _infer_runtime_nonlinear_steps(
            replace(cfg, time=replace(cfg.time, t_max=0.0)), dt=0.1, steps=0
        )


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


_AUTO_FIT_OPTIONS = dict(
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
        **_AUTO_FIT_OPTIONS,
    )

    assert out.fit_signal_used == "density"
    assert out.gamma > 0.0
    assert out.fit_window_tmin is not None
    assert out.fit_window_tmax is not None
    np.testing.assert_allclose(out.signal, density[:, 0, 0, 0])


def test_finalize_runtime_linear_quasilinear_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_cfg(), quasilinear=RuntimeQuasilinearConfig(enabled=True, csat=0.7)
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

    patch_runtime(monkeypatch, "build_linear_cache", lambda *_args: "cache")
    patch_runtime(monkeypatch, "compute_quasilinear_from_linear_state", _compute)
    patch_runtime(monkeypatch, "linear_terms_to_term_config", lambda terms: terms)
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
            Phi2_kxt=np.ones((2, 4), dtype=float), Wg_kxst=None
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


@pytest.mark.parametrize(
    ("name", "t", "phi", "density"),
    [
        ("time", np.array([0.0, np.nan]), np.ones(2), None),
        ("field", np.arange(2.0), np.array([1.0, np.inf]), None),
        ("density", np.arange(2.0), np.ones(2), np.array([1.0, np.nan])),
    ],
)
def test_linear_history_rejects_nonfinite_trajectories(
    name: str, t: np.ndarray, phi: np.ndarray, density: np.ndarray | None
) -> None:
    """Unstable histories must fail before a finite growth fit can hide them."""

    with pytest.raises(FloatingPointError, match=rf"non-finite {name} history"):
        _prepare_runtime_linear_fit_inputs(
            t=t, phi_t=phi, density_t=density, z=np.ones(1), fit_signal="auto"
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
            **_AUTO_FIT_OPTIONS,
        )


@pytest.mark.parametrize(
    "gaussian",
    [{"gaussian_init": False}, {"gaussian_init": True, "gaussian_width": 0.35}],
)
def test_runtime_single_mode_init_populates_zonal_ky0_branch(gaussian) -> None:
    cfg = replace(
        _base_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density", init_amp=1.0, init_single=True, **gaussian
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


@pytest.mark.parametrize(
    "init_override",
    [{"gaussian_width": 0.0}, {"init_file_mode": "bad"}, {"init_field": "bad"}],
)
def test_runtime_initial_condition_validation_branches(init_override) -> None:
    cfg = _base_cfg()
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)

    with pytest.raises(ValueError):
        _build_initial_condition(
            grid,
            geom,
            replace(cfg, init=replace(cfg.init, **init_override)),
            ky_index=1,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )


_SCAN_BATCH_AUTO = dict(
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
    show_progress=False,
)
_SCAN_BATCH_FIT = {**_SCAN_BATCH_AUTO, "auto_window": False, "tmin": 0.0, "tmax": 0.2}


def test_run_runtime_scan_batch_validation_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_cfg()
    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg, np.asarray([], dtype=float), **_SCAN_BATCH_AUTO, fit_signal="phi"
        )

    grid = build_spectral_grid(cfg.grid)
    geom = object()
    params = type("Params", (), {"rho_star": np.asarray(1.0)})()
    patch_runtime(monkeypatch, "build_runtime_geometry", lambda _cfg: geom)
    patch_runtime(
        monkeypatch, "apply_geometry_grid_defaults", lambda _geom, grid_cfg: grid_cfg
    )
    patch_runtime(monkeypatch, "build_spectral_grid", lambda _cfg: grid)
    patch_runtime(
        monkeypatch, "build_runtime_linear_params", lambda *_args, **_kwargs: params
    )
    patch_runtime(monkeypatch, "build_runtime_linear_terms", lambda _cfg: object())
    patch_runtime(
        monkeypatch,
        "_build_initial_condition",
        lambda *_args, **_kwargs: np.ones(
            (1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
        ),
    )
    patch_runtime(
        monkeypatch,
        "integrate_linear_diagnostics",
        lambda *_args, **_kwargs: (
            None,
            np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
            2.0
            * np.ones((3, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
        ),
    )
    patch_runtime(
        monkeypatch,
        "extract_mode_time_series",
        lambda arr, sel, method="project": np.asarray(
            arr[:, sel.ky_index, sel.kx_index, 0]
        ),
    )
    patch_runtime(
        monkeypatch,
        "fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            0.2,
            0.3,
            0.0,
            0.2,
            2.0 if np.max(np.abs(signal)) < 1.5 else 1.0,
            0.0,
        ),
    )
    patch_runtime(
        monkeypatch,
        "fit_growth_rate_auto",
        lambda *args, **kwargs: (0.4, 0.5, 0.0, 0.2),
    )
    patch_runtime(monkeypatch, "fit_growth_rate", lambda *args, **kwargs: (0.6, 0.7))
    patch_runtime(
        monkeypatch, "apply_diagnostic_normalization", lambda g, o, **kwargs: (g, o)
    )

    scan_auto = _run_runtime_scan_batch(
        cfg, np.asarray([0.1, 0.2], dtype=float), **_SCAN_BATCH_AUTO, fit_signal="auto"
    )
    assert scan_auto.gamma.shape == (2,)

    scan_density = _run_runtime_scan_batch(
        cfg, np.asarray([0.1], dtype=float), **_SCAN_BATCH_FIT, fit_signal="density"
    )
    assert np.allclose(scan_density.gamma, np.array([0.6]))

    with pytest.raises(ValueError):
        _run_runtime_scan_batch(
            cfg,
            np.asarray([0.1], dtype=float),
            **_SCAN_BATCH_AUTO,
            fit_signal="invalid",
        )


def test_run_runtime_scan_default_parallel_config_keeps_serial_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

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

    patch_runtime(monkeypatch, "run_runtime_scan_batch", _unexpected_batch)
    patch_runtime(monkeypatch, "run_runtime_linear", _fake_run_runtime_linear)

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

    patch_runtime(monkeypatch, "run_runtime_linear", _fake_run_runtime_linear)

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

    cfg = replace(
        _base_cfg(), parallel=RuntimeParallelConfig(strategy="combined_ky", axis="ky")
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

    patch_runtime(monkeypatch, "run_runtime_scan_batch", _fake_batch)
    patch_runtime(
        monkeypatch,
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


def _nonlinear_cfg(**overrides) -> RuntimeConfig:
    return replace(
        _base_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        **overrides,
    )


def _stub_nonlinear_startup(monkeypatch, cfg):
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(cfg.grid)
    patch_nonlinear_setup(monkeypatch, patch_runtime, geom, grid)
    return grid


@pytest.mark.parametrize("show_progress", [True, False])
def test_run_runtime_nonlinear_final_state_contract(
    monkeypatch: pytest.MonkeyPatch, show_progress: bool
) -> None:
    cfg = _nonlinear_cfg()
    grid = _stub_nonlinear_startup(monkeypatch, cfg)
    captured: dict[str, object] = {}

    def _fake_final_state(*args, **kwargs):
        captured["n_args"] = len(args)
        captured["show_progress"] = kwargs.get("show_progress", None)
        return state_and_fields(grid)

    patch_runtime(monkeypatch, "integrate_nonlinear_from_config", _fake_final_state)

    out = run_runtime_nonlinear(
        cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=False, show_progress=show_progress
    )

    if show_progress:
        assert captured["show_progress"] is True
        assert out.state is None
    else:
        assert captured["n_args"] == 5
        assert captured["show_progress"] is None
    assert out.diagnostics is None
    assert out.phi2 is not None


@pytest.mark.parametrize(
    ("expert", "match"),
    [
        (
            RuntimeExpertConfig(fixed_mode=True, iky_fixed=None, ikx_fixed=None),
            "expert.iky_fixed and expert.ikx_fixed",
        ),
        (RuntimeExpertConfig(source="bad_source"), "unsupported expert.source"),
    ],
)
def test_run_runtime_nonlinear_rejects_invalid_expert_config(expert, match) -> None:
    with pytest.raises(ValueError, match=match):
        run_runtime_nonlinear(_nonlinear_cfg(expert=expert), ky_target=0.2, Nl=3, Nm=4)


def test_run_runtime_nonlinear_phiext_source_uses_diagnostics_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _nonlinear_cfg(expert=RuntimeExpertConfig(source="phiext_full", phi_ext=0.25))
    grid = _stub_nonlinear_startup(monkeypatch, cfg)
    captured: dict[str, object] = {}

    def _fake_diag_integrator(*args, **kwargs):
        captured.update(kwargs)
        t = np.asarray([0.1, 0.2], dtype=float)
        return (t, zero_field_diag(t, Wphi=[1.0, 1.1]), *state_and_fields(grid))

    patch_runtime(
        monkeypatch,
        "integrate_nonlinear_explicit_diagnostics_state",
        _fake_diag_integrator,
    )

    out = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=False)

    assert captured["external_phi"] == pytest.approx(0.25)
    assert out.diagnostics is None
    assert out.phi2 is not None


def test_run_runtime_nonlinear_adaptive_chunk_forwards_fixed_mode_and_collision_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _nonlinear_cfg(
        time=replace(
            _base_cfg().time,
            fixed_dt=False,
            t_max=0.35,
            dt=0.1,
            diagnostics=True,
            collision_split=True,
            collision_scheme="exp",
        ),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0),
    )
    grid = _stub_nonlinear_startup(monkeypatch, cfg)
    captured: list[dict[str, object]] = []

    def _fake_diag_integrator(*args, **kwargs):
        captured.append(dict(kwargs))
        t = np.asarray([0.1, 0.2], dtype=float)
        if kwargs.get("time_horizon") is not None:
            t = np.minimum(t, float(kwargs["time_horizon"]))
        dt_t = np.diff(np.insert(t, 0, 0.0))
        diag = zero_field_diag(t, dt_t=dt_t, dt_mean=np.mean(dt_t), Wphi=[1.0, 1.2])
        return (t, diag, *state_and_fields(grid))

    patch_runtime(
        monkeypatch,
        "integrate_nonlinear_explicit_diagnostics_state",
        _fake_diag_integrator,
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
            [[1.0 + 0.0j, 0.0 + 2.0j], [3.0 + 4.0j, 0.0 + 0.0j]], dtype=np.complex64
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


@pytest.mark.parametrize("tmax,settled", [(0.5154, False), (0.7, True)])
def test_fit_settled_requires_two_growth_times_in_the_window(tmax, settled) -> None:
    """The beta 0.010 KBM fit spanned 0.37 growth times yet reported settled."""

    t = np.linspace(0.0, 1.0, 40001)
    phi = np.exp((24.0 - 22.6j) * t)[:, None, None, None] * np.ones((1, 1, 1, 2))
    options = {**_AUTO_FIT_OPTIONS, "auto_window": False, "tmin": 0.5, "tmax": tmax}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = fit_runtime_linear_diagnostics(
            t=t,
            phi_t=phi,
            density_t=None,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            z=np.asarray([-1.0, 1.0]),
            fit_signal="phi",
            **options,
        )
    assert out.gamma == pytest.approx(24.0, rel=1e-6)
    assert out.fit_settled is settled
