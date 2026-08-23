"""Tests for unified runtime config and TOML loading."""

from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_repo_script

import pytest

from gkx.workflows.runtime.toml import (
    direct_config_shorthand_args,
    is_runtime_toml,
    load_toml,
    load_runtime_from_toml,
    toml_shorthand_command,
)
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeParallelConfig,
    RuntimeQuasilinearConfig,
)


def _load_module_from_path(name: str, path: Path):
    return load_repo_script(path.relative_to(REPO_ROOT), module_name=name)


def test_runtime_config_to_dict_contains_sections() -> None:
    cfg = RuntimeConfig()
    d = cfg.to_dict()
    assert set(d) == {
        "grid",
        "time",
        "geometry",
        "init",
        "species",
        "physics",
        "collisions",
        "normalization",
        "terms",
        "expert",
        "output",
        "quasilinear",
        "parallel",
    }
    assert len(d["species"]) == 1


def test_runtime_defaults_match_reference_contract() -> None:
    cfg = RuntimeConfig()
    assert cfg.geometry.drift_scale == 1.0
    assert cfg.normalization.diagnostic_norm == "rho_star"
    assert cfg.normalization.flux_scale == 1.0
    assert cfg.collisions.p_hyper_m is None
    assert cfg.collisions.damp_ends_amp == pytest.approx(0.1)
    assert cfg.collisions.damp_ends_widthfrac == pytest.approx(0.125)
    assert cfg.parallel.strategy == "serial"
    assert cfg.parallel.axis == "ky"


def test_runtime_config_to_dict_is_json_roundtrippable_with_serial_aliases() -> None:
    cfg = RuntimeConfig(
        quasilinear=RuntimeQuasilinearConfig(channels="em"),
        parallel=RuntimeParallelConfig(strategy=" off ", axis=" KY "),
    )

    payload = cfg.to_dict()
    restored = json.loads(json.dumps(payload))

    assert payload["quasilinear"]["channels"] == ("em",)
    assert restored["quasilinear"]["channels"] == ["em"]
    assert restored["parallel"]["strategy"] == "serial"
    assert restored["parallel"]["axis"] == "ky"
    assert restored["parallel"]["strict_identity"] is True
    assert restored["species"][0]["name"] == "ion"


def test_load_runtime_from_toml_handles_path_and_species_edge_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GKX_TEST_ROOT", str(tmp_path))
    toml = """
species = []

[geometry]
model = "vmec"
vmec_file = "$GKX_TEST_ROOT/vmec.nc"
geometry_file = "$GKX_TEST_MISSING/geom.nc"

[quasilinear]
channels = "em"
output_path = "$GKX_TEST_ROOT/ql"

[output]
restart_to_file = "$GKX_TEST_MISSING/restart.nc"

[parallel]
strategy = "OFF"
axis = " KY "
"""
    path = tmp_path / "runtime_edges.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert len(cfg.species) == 1
    assert cfg.species[0].name == "ion"
    assert cfg.geometry.vmec_file == str((tmp_path / "vmec.nc").resolve())
    assert cfg.geometry.geometry_file == "$GKX_TEST_MISSING/geom.nc"
    assert cfg.quasilinear.channels == ("em",)
    assert cfg.quasilinear.output_path == str((tmp_path / "ql").resolve())
    assert cfg.output.restart_to_file == "$GKX_TEST_MISSING/restart.nc"
    assert cfg.parallel.strategy == "serial"
    assert cfg.parallel.axis == "ky"


def test_toml_shorthand_policy_uses_one_runtime_command(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "case.toml"
    cfg_path.write_text("[physics]\n", encoding="utf-8")

    assert is_runtime_toml({"physics": {}}) is True
    assert is_runtime_toml({"case": "cyclone"}) is True
    assert is_runtime_toml({}) is True
    assert toml_shorthand_command({"physics": {}}) == "run"
    assert toml_shorthand_command({"case": "cyclone"}) == "run"
    assert direct_config_shorthand_args(
        [str(cfg_path), "--no-progress"],
        load_toml_func=lambda _path: {"physics": {}},
    ) == ["run", "--config", str(cfg_path), "--no-progress"]
    assert direct_config_shorthand_args(
        [str(cfg_path), "--plot"],
        load_toml_func=lambda _path: {"case": "cyclone"},
    ) == ["run", "--config", str(cfg_path), "--plot"]
    assert direct_config_shorthand_args([]) is None
    assert direct_config_shorthand_args(["--version"]) is None
    assert direct_config_shorthand_args(["run", "--config", str(cfg_path)]) is None
    assert direct_config_shorthand_args([str(tmp_path / "missing.toml")]) is None


def test_load_runtime_from_toml_rejects_single_species_table(tmp_path: Path) -> None:
    path = tmp_path / "runtime_bad_species.toml"
    path.write_text(
        """
[species]
name = "ion"
""",
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match=r"\[\[species\]\] entries"):
        load_runtime_from_toml(path)


def test_load_runtime_from_toml_roundtrip(tmp_path: Path) -> None:
    toml = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[[species]]
name = "electron"
charge = -1.0
mass = 0.00027248
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[grid]
Nx = 1
Ny = 8
Nz = 16

[physics]
electromagnetic = true
use_apar = true
adiabatic_electrons = false
beta = 0.2

[expert]
fixed_mode = true
iky_fixed = 1
ikx_fixed = 0

[init]
init_file = "/tmp/restart.bin"
init_file_scale = 5.0
init_file_mode = "add"

[normalization]
contract = "kbm"
omega_star_scale = 0.7

[output]
path = "tools_out/runtime_case"

[quasilinear]
enabled = true
mode = "saturated"
saturation_rule = "mixing_length"
amplitude_normalization = "phi_rms"
csat = 0.7
channels = ["es"]
output_path = "tools_out/ql_case"

[parallel]
strategy = "batch-ky"
axis = "ky"
batch_size = 3
num_devices = 2
strict_identity = true
profile = true
backend = "auto"
"""
    path = tmp_path / "runtime.toml"
    path.write_text(toml, encoding="utf-8")
    cfg, data = load_runtime_from_toml(path)
    assert isinstance(data, dict)
    assert cfg.grid.Ny == 8
    assert cfg.physics.electromagnetic
    assert cfg.physics.use_apar
    assert not cfg.physics.adiabatic_electrons
    assert cfg.physics.beta == pytest.approx(0.2)
    assert cfg.normalization.contract == "kbm"
    assert cfg.normalization.omega_star_scale == pytest.approx(0.7)
    assert cfg.expert.fixed_mode is True
    assert cfg.expert.iky_fixed == 1
    assert cfg.expert.ikx_fixed == 0
    assert cfg.init.init_file == str(Path("/tmp/restart.bin").resolve())
    assert cfg.init.init_file_scale == pytest.approx(5.0)
    assert cfg.init.init_file_mode == "add"
    assert cfg.output.path == str((tmp_path / "tools_out" / "runtime_case").resolve())
    assert cfg.quasilinear.enabled is True
    assert cfg.quasilinear.mode == "saturated"
    assert cfg.quasilinear.saturation_rule == "mixing_length"
    assert cfg.quasilinear.csat == pytest.approx(0.7)
    assert cfg.quasilinear.channels == ("es",)
    assert cfg.quasilinear.output_path == str(
        (tmp_path / "tools_out" / "ql_case").resolve()
    )
    assert cfg.parallel.strategy == "combined_ky"
    assert cfg.parallel.axis == "ky"
    assert cfg.parallel.batch_size == 3
    assert cfg.parallel.num_devices == 2
    assert cfg.parallel.strict_identity is True
    assert cfg.parallel.profile is True
    assert len(cfg.species) == 2
    assert cfg.species[1].charge == pytest.approx(-1.0)


def test_runtime_parallel_config_validates_values() -> None:
    assert RuntimeParallelConfig(strategy="batch-ky").strategy == "combined_ky"
    with pytest.raises(ValueError):
        RuntimeParallelConfig(strategy="unknown")
    with pytest.raises(ValueError):
        RuntimeParallelConfig(batch_size=0)
    with pytest.raises(ValueError):
        RuntimeParallelConfig(num_devices=0)


def test_gx_aligned_kbm_runtime_examples_keep_end_damping_enabled() -> None:
    cfg_dir = REPO_ROOT / "examples" / "nonlinear" / "axisymmetric"
    paths = [
        cfg_dir / "runtime_kbm_nonlinear.toml",
        cfg_dir / "runtime_kbm_nonlinear_seed.toml",
        cfg_dir / "runtime_kbm_nonlinear_short.toml",
        cfg_dir / "runtime_kbm_nonlinear_short_lockin.toml",
        cfg_dir / "runtime_kbm_nonlinear_t100.toml",
        cfg_dir / "runtime_kbm_nonlinear_t100_nx4ny8_dt9e4.toml",
    ]
    for path in paths:
        cfg, _ = load_runtime_from_toml(path)
        assert cfg.terms.end_damping == pytest.approx(1.0), path.name


def test_linear_axisymmetric_runtime_examples_keep_parity_collision_contract() -> None:
    cfg_dir = REPO_ROOT / "examples" / "linear" / "axisymmetric"
    expected = {
        "cyclone.toml": (1.0, 2.0, 0.0, 1.0),
        "etg.toml": (1.0, 2.0, 0.0, 1.0),
        "runtime_etg.toml": (1.0, 2.0, 0.0, 1.0),
        "runtime_kaw.toml": (1.0, 2.0, 0.0, 1.0),
        "runtime_kbm.toml": (1.0, 2.0, 0.0, 1.0),
    }
    for name, (nu_h, nu_l, hyper_const, hyper_kz) in expected.items():
        cfg, _ = load_runtime_from_toml(cfg_dir / name)
        assert cfg.collisions.nu_hermite == pytest.approx(nu_h), name
        assert cfg.collisions.nu_laguerre == pytest.approx(nu_l), name
        assert cfg.collisions.hypercollisions_const == pytest.approx(hyper_const), name
        assert cfg.collisions.hypercollisions_kz == pytest.approx(hyper_kz), name

    _cfg, cyclone_raw = load_runtime_from_toml(cfg_dir / "cyclone.toml")
    assert cyclone_raw["fit"]["mode_method"] == "z_index"

    for name in ("etg.toml", "runtime_etg.toml"):
        cfg, raw = load_runtime_from_toml(cfg_dir / name)
        assert cfg.time.method == "rk4", name
        assert cfg.time.dt == pytest.approx(1.6e-4), name
        assert cfg.time.t_max == pytest.approx(2.0), name
        assert raw["run"]["solver"] == "time", name
        assert raw["scan"]["solver"] == "time", name


def test_nonaxisymmetric_quasilinear_examples_keep_electrostatic_contract() -> None:
    cfg_dir = REPO_ROOT / "examples" / "linear" / "non-axisymmetric"
    for name in (
        "runtime_hsx_linear_quasilinear.toml",
        "runtime_w7x_linear_quasilinear_vmec.toml",
    ):
        cfg, _ = load_runtime_from_toml(cfg_dir / name)
        assert cfg.quasilinear.enabled is True, name
        assert cfg.quasilinear.channels == ("es",), name
        assert cfg.physics.electrostatic is True, name
        assert cfg.physics.electromagnetic is False, name
        assert cfg.terms.apar == pytest.approx(0.0), name
        assert cfg.terms.bpar == pytest.approx(0.0), name


def test_etg_nonlinear_pilot_example_keeps_two_species_full_gk_contract() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "axisymmetric"
        / "runtime_etg_nonlinear.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert len(cfg.species) == 2
    assert cfg.physics.linear is False
    assert cfg.physics.nonlinear is True
    assert cfg.physics.electrostatic is True
    assert cfg.physics.electromagnetic is False
    assert cfg.physics.adiabatic_ions is False
    assert cfg.physics.adiabatic_electrons is False
    assert cfg.grid.Lx == pytest.approx(1.25)
    assert cfg.init.gaussian_init is True
    assert cfg.init.init_single is False
    assert cfg.collisions.hypercollisions_const == pytest.approx(0.0)
    assert cfg.collisions.hypercollisions_kz == pytest.approx(1.0)
    assert data["run"]["ky"] == pytest.approx(5.0)
    assert cfg.output.path == str(
        (path.parents[3] / "tools_out" / "etg_nonlinear_runtime").resolve()
    )


def test_load_runtime_from_toml_keeps_imported_geometry_fields(tmp_path: Path) -> None:
    toml = """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 3.0
fprim = 1.0
kinetic = true

[grid]
Nx = 1
Ny = 12
Nz = 32

[geometry]
model = "imported-netcdf"
geometry_file = "/tmp/w7x.eik.nc"

[physics]
adiabatic_electrons = true
electromagnetic = false

[run]
ky = 0.3
Nl = 8
Nm = 12
solver = "explicit_time"
"""
    path = tmp_path / "runtime_w7x.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "imported-netcdf"
    assert cfg.geometry.geometry_file == str(Path("/tmp/w7x.eik.nc").resolve())
    assert cfg.physics.adiabatic_electrons is True


def test_w7x_imported_geometry_example_toml_loads() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "linear"
        / "non-axisymmetric"
        / "runtime_w7x_linear_imported_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.geometry_file is None
    assert cfg.geometry.vmec_file == str(
        (path.parents[2] / "vmec" / "wout_nfp3_QI_fixed_resolution_final.nc").resolve()
    )
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.init.init_field == "density"
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.normalization.diagnostic_norm == "rho_star"


def test_w7x_nonlinear_imported_geometry_example_toml_loads() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_w7x_nonlinear_imported_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.geometry_file is None
    assert cfg.geometry.vmec_file == str(
        (path.parents[2] / "vmec" / "wout_nfp3_QI_fixed_resolution_final.nc").resolve()
    )
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.nonlinear == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str(
        (path.parents[3] / "tools_out" / "w7x_nonlinear_imported_runtime").resolve()
    )


def test_w7x_nonlinear_imported_geometry_builder_keeps_collision_contract() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "w7x_nonlinear_imported_geometry.py"
    )
    mod = _load_module_from_path("w7x_nonlinear_imported_geometry", path)
    cfg = mod.build_w7x_nonlinear_cfg("/tmp/w7x.eik.nc", dt=0.1, t_max=200.0)
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.hypercollisions == pytest.approx(1.0)
    assert cfg.collisions.D_hyper == pytest.approx(0.05)


def test_hsx_nonlinear_vmec_geometry_example_toml_loads() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_hsx_nonlinear_vmec_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file is not None
    assert cfg.geometry.vmec_file == str(
        (path.parents[2] / "vmec" / "wout_NuhrenbergZille_1988_QHS.nc").resolve()
    )
    assert cfg.geometry.geometry_helper_python is None
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.nonlinear == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str(
        (path.parents[3] / "tools_out" / "hsx_nonlinear_vmec_runtime").resolve()
    )


def test_hsx_nonlinear_vmec_geometry_builder_keeps_collision_contract() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "hsx_nonlinear_vmec_geometry.py"
    )
    mod = _load_module_from_path("hsx_nonlinear_vmec_geometry", path)
    cfg = mod.build_hsx_nonlinear_cfg(
        "/tmp/hsx.nc",
        geometry_file=None,
        geometry_helper_repo=None,
        geometry_helper_python=None,
        torflux=0.64,
        alpha=0.0,
        npol=1.0,
        dt=0.1,
        t_max=200.0,
    )
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert cfg.terms.hypercollisions == pytest.approx(1.0)
    assert cfg.collisions.D_hyper == pytest.approx(0.05)


def test_hsx_nonlinear_vmec_wrapper_defaults_to_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "hsx_nonlinear_vmec_geometry.py"
    )
    mod = _load_module_from_path("hsx_nonlinear_vmec_geometry_main", path)

    captured: dict[str, object] = {}

    def fake_run_nonlinear_case(config_path, **kwargs):
        captured["config_path"] = Path(config_path)
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(mod, "run_nonlinear_case", fake_run_nonlinear_case)
    monkeypatch.setattr(mod, "STEPS", 200)

    rc = mod.main()

    assert rc == 0
    assert captured["config_path"] == mod.CONFIG
    assert captured["kwargs"]["steps"] == 200


def test_w7x_nonlinear_vmec_geometry_example_toml_loads() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "non-axisymmetric"
        / "runtime_w7x_nonlinear_vmec_geometry.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file is not None
    assert cfg.geometry.vmec_file == str(
        (path.parents[2] / "vmec" / "wout_nfp3_QI_fixed_resolution_final.nc").resolve()
    )
    assert cfg.geometry.geometry_helper_python is None
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.collisions is True
    assert cfg.terms.collisions == pytest.approx(1.0)
    assert "steps" not in data.get("run", {})
    assert cfg.output.path == str(
        (path.parents[3] / "tools_out" / "w7x_nonlinear_vmec_runtime").resolve()
    )


def test_load_runtime_from_toml_resolves_relative_runtime_paths_against_config_dir(
    tmp_path: Path,
) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    toml = """
[geometry]
model = "vmec"
vmec_file = "../vmec/wout.nc"
geometry_file = "../geom/run.eik.nc"
torflux = 0.64

[init]
init_file = "../restart/state.bin"

[output]
path = "../out/run.out.nc"
restart_to_file = "../out/run.restart.nc"
restart_from_file = "../out/run.resume.nc"
"""
    path = cfg_dir / "runtime.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.vmec_file == str((tmp_path / "vmec" / "wout.nc").resolve())
    assert cfg.geometry.geometry_file == str(
        (tmp_path / "geom" / "run.eik.nc").resolve()
    )
    assert cfg.init.init_file == str((tmp_path / "restart" / "state.bin").resolve())
    assert cfg.output.path == str((tmp_path / "out" / "run.out.nc").resolve())
    assert cfg.output.restart_to_file == str(
        (tmp_path / "out" / "run.restart.nc").resolve()
    )
    assert cfg.output.restart_from_file == str(
        (tmp_path / "out" / "run.resume.nc").resolve()
    )


def test_secondary_slab_example_toml_loads() -> None:
    path = REPO_ROOT / "benchmarks" / "runtime_secondary_slab.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "slab"
    assert cfg.geometry.s_hat == pytest.approx(1.0e-8)
    assert cfg.physics.linear is True
    assert cfg.physics.nonlinear is False
    assert cfg.physics.adiabatic_electrons is True


def test_load_runtime_from_toml_accepts_desc_eik_geometry_alias(tmp_path: Path) -> None:
    toml = """
[geometry]
model = "desc-eik"
geometry_file = "/tmp/w7x-desc.eik.nc"
"""
    path = tmp_path / "runtime_desc.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "desc-eik"
    assert cfg.geometry.geometry_file == str(Path("/tmp/w7x-desc.eik.nc").resolve())


def test_load_runtime_from_toml_accepts_vmec_geometry_helper_python(
    tmp_path: Path,
) -> None:
    toml = """
[geometry]
model = "vmec"
vmec_file = "/tmp/wout_test.nc"
torflux = 0.64
geometry_helper_python = "python3"
"""
    path = tmp_path / "runtime_vmec.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file == str(Path("/tmp/wout_test.nc").resolve())
    assert cfg.geometry.geometry_helper_python == "python3"


def test_load_runtime_from_toml_accepts_geometry_helper_fields(
    tmp_path: Path,
) -> None:
    toml = """
[geometry]
model = "vmec"
vmec_file = "/tmp/wout_test.nc"
torflux = 0.64
geometry_helper_python = "python3"
geometry_helper_repo = "/tmp/helper"
"""
    path = tmp_path / "runtime_vmec_helper.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.geometry_helper_python == "python3"
    assert cfg.geometry.geometry_helper_repo == str(Path("/tmp/helper"))


def test_load_runtime_from_toml_accepts_miller_geometry_fields(tmp_path: Path) -> None:
    toml = """
[geometry]
model = "miller"
rhoc = 0.5
q = 1.4
s_hat = 0.8
R0 = 2.77778
R_geo = 2.77778
shift = 0.0
akappa = 1.0
akappri = 0.0
tri = 0.0
tripri = 0.0
betaprim = 0.0
geometry_helper_python = "python3"
"""
    path = tmp_path / "runtime_miller.toml"
    path.write_text(toml, encoding="utf-8")

    cfg, _ = load_runtime_from_toml(path)

    assert cfg.geometry.model == "miller"
    assert cfg.geometry.rhoc == pytest.approx(0.5)
    assert cfg.geometry.R_geo == pytest.approx(2.77778)
    assert cfg.geometry.akappa == pytest.approx(1.0)
    assert cfg.geometry.tripri == pytest.approx(0.0)
    assert cfg.geometry.geometry_helper_python == "python3"


def test_cyclone_nonlinear_gx_miller_example_toml_loads() -> None:
    path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "axisymmetric"
        / "runtime_cyclone_nonlinear_miller.toml"
    )

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "miller"
    assert cfg.geometry.q == pytest.approx(1.4)
    assert cfg.geometry.s_hat == pytest.approx(0.8)
    assert cfg.geometry.rhoc == pytest.approx(0.5)
    assert cfg.physics.nonlinear is True
    assert cfg.physics.adiabatic_electrons is True


def test_miller_zonal_response_example_uses_merlo_case_iii_contract() -> None:
    path = REPO_ROOT / "benchmarks" / "runtime_miller_zonal_response.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.expert.source == "default"
    assert cfg.expert.phi_ext == pytest.approx(0.0)
    assert cfg.init.init_field == "density"
    assert cfg.init.init_amp == pytest.approx(1.0e-6)
    assert cfg.output.save_for_restart is True
    assert cfg.geometry.q == pytest.approx(1.389)
    assert cfg.geometry.s_hat == pytest.approx(0.751)
    assert cfg.geometry.akappa == pytest.approx(1.4723)
    assert cfg.geometry.tri == pytest.approx(-0.0070)
    assert cfg.geometry.shift == pytest.approx(-0.1569)
    assert cfg.grid.Nz == 32
    assert data["run"]["Nl"] == 4
    assert data["run"]["kx"] == pytest.approx(0.05)
    assert data["run"]["ky"] == pytest.approx(0.0)
    # Converged Hermite baseline, not the retired Nm=24 one. Nm >= 120 is what
    # puts the whole analysis window before the recurrence onset
    # t_quiet ~ 5.5 sqrt(Nm), and dt <= 0.0025 is what keeps Nm=144 stable --
    # the Hermite streaming CFL scales as sqrt(Nm) and dt=0.005 goes non-finite
    # at t=46.5 there.
    assert data["run"]["Nm"] >= 120
    assert data["run"]["dt"] <= 0.0025
    assert data["run"]["steps"] * data["run"]["dt"] == pytest.approx(60.0)
    # A zero-gradient relaxation run has no saturation to stop at: the default
    # run_to = "saturation" declares the ~0 heat flux converged inside the first
    # chunk and truncates the trace at t ~ 6 without raising.
    assert cfg.time.run_to == "t_max"


def test_w7x_zonal_response_vmec_example_uses_test4_contract() -> None:
    path = REPO_ROOT / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml"

    cfg, data = load_runtime_from_toml(path)

    assert isinstance(data, dict)
    assert cfg.geometry.model == "vmec"
    assert cfg.geometry.vmec_file == str(
        (
            REPO_ROOT / "examples" / "vmec" / "wout_nfp3_QI_fixed_resolution_final.nc"
        ).resolve()
    )
    assert cfg.geometry.torflux == pytest.approx(0.64)
    assert cfg.geometry.alpha == pytest.approx(0.0)
    assert cfg.geometry.R0 == pytest.approx(5.485)
    assert cfg.grid.boundary == "linked"
    assert cfg.grid.nperiod == 4
    assert cfg.grid.Nz == 256
    assert cfg.init.gaussian_init is True
    assert cfg.init.gaussian_width == pytest.approx(1.0)
    assert cfg.init.init_field == "phi"
    assert cfg.physics.adiabatic_electrons is True
    assert cfg.physics.nonlinear is False
    assert cfg.physics.collisions is False
    assert cfg.physics.hypercollisions is False
    assert cfg.species[0].tprim == pytest.approx(0.0)
    assert cfg.species[0].fprim == pytest.approx(0.0)
    assert data["run"]["ky"] == pytest.approx(0.0)
    assert data["run"]["kx"] == pytest.approx(0.05)
    assert data["run"]["Nl"] == 8
    assert data["run"]["Nm"] == 32
    assert data["run"]["dt"] == pytest.approx(0.05)


def test_output_warm_start_is_opt_in_and_round_trips(tmp_path: Path) -> None:
    """Warm start is an [output] restart control and is opt-in from TOML."""

    assert RuntimeConfig().output.warm_start is False
    assert RuntimeConfig().to_dict()["output"]["warm_start"] is False

    path = tmp_path / "warm_on.toml"
    path.write_text("[output]\nwarm_start = true\n", encoding="utf-8")
    cfg, _data = load_runtime_from_toml(path)

    assert cfg.output.warm_start is True
    # Nothing else about the restart contract moved.
    assert cfg.output.restart is False
    assert cfg.output.save_for_restart is True


def test_load_toml_names_a_netcdf_handed_to_it_instead_of_a_decode_error(
    tmp_path: Path,
) -> None:
    """A wout that missed the equilibrium sniff must not surface as byte 55.

    tomllib reports a binary file as a UnicodeDecodeError against an offset,
    which tells a user nothing about what they actually passed.
    """

    masquerading = tmp_path / "looks_like_a_config.toml"
    masquerading.write_bytes(b"CDF\x02\x00\x00\x00\x00" + b"\xc8" * 64)

    with pytest.raises(ValueError, match="NetCDF.*not a TOML input file"):
        load_toml(masquerading)


def test_load_toml_reports_invalid_toml_with_the_file_name(tmp_path: Path) -> None:
    config = tmp_path / "broken.toml"
    config.write_text("this = = not toml", encoding="utf-8")

    with pytest.raises(ValueError, match="broken.toml is not valid TOML"):
        load_toml(config)
