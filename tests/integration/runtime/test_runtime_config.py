"""Runtime configuration layer: TOML/deck loading into RuntimeConfig, the resolution estimator, and the pure selection policies applied before a run starts."""

from __future__ import annotations

from gkx.workflows.runtime import wout as runtime_wout
from gkx.workflows.runtime.config import (
    RuntimeConfig,
    RuntimeParallelConfig,
    RuntimeQuasilinearConfig,
)
from gkx.workflows.runtime.policies import (
    RuntimeIndependentParallelPlan,
    _active_kx_indices,
    _active_ky_indices,
    _infer_runtime_nonlinear_steps,
    _midplane_index,
    _nearest_index_from_candidates,
    _normalize_linear_solver_name,
    _parallel_requests_combined_ky_scan,
    _runtime_external_phi,
    _runtime_independent_parallel_plan,
    _select_nonlinear_mode_indices,
    _validate_dealias_mask_shape,
    _zero_kx_index,
)
from gkx.workflows.runtime.resolution import (
    PERP_LADDER,
    GeometryFeatures,
    geometry_class,
    ky_max_target,
    perp_points_for,
    resolution_from_features,
)
from gkx.workflows.runtime.toml import (
    direct_config_shorthand_args,
    is_runtime_toml,
    load_toml,
    load_runtime_from_toml,
    toml_shorthand_command,
)
from pathlib import Path
from support.paths import REPO_ROOT, load_repo_script
from types import SimpleNamespace
import json
import numpy as np
import os
import pytest


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


def test_runtime_toml_schema_accepts_v1_and_legacy_but_rejects_other_versions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "case.toml"
    path.write_text("schema_version = 1\n[physics]\n", encoding="utf-8")
    _cfg, raw = load_runtime_from_toml(path)
    assert raw["schema_version"] == 1

    path.write_text("[physics]\n", encoding="utf-8")
    _cfg, raw = load_runtime_from_toml(path)
    assert "schema_version" not in raw

    for invalid in ('"1"', "true"):
        path.write_text(f"schema_version = {invalid}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be the integer 1"):
            load_runtime_from_toml(path)
    path.write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported GKX TOML schema_version 2"):
        load_runtime_from_toml(path)


def test_runtime_toml_rejects_removed_diffrax_time_keys(tmp_path: Path) -> None:
    """A deck that still selects Diffrax must fail loudly, not be ignored."""

    path = tmp_path / "case.toml"

    path.write_text(
        "schema_version = 1\n[time]\nt_max = 1.0\nuse_diffrax = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="'use_diffrax'") as excinfo:
        load_runtime_from_toml(path)
    message = str(excinfo.value)
    assert "no longer ships" in message
    # The message has to name the native replacement, not just the removal.
    assert "method" in message

    # A deck that only carries the tuning keys must fail the same way.
    path.write_text(
        "schema_version = 1\n[time]\ndiffrax_solver = \"Tsit5\"\n"
        "diffrax_max_steps = 20000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="diffrax_solver") as excinfo:
        load_runtime_from_toml(path)
    assert "diffrax_max_steps" in str(excinfo.value)

    # Selecting the native owner explicitly still loads.
    path.write_text(
        "schema_version = 1\n[time]\nt_max = 1.0\nmethod = \"rk4\"\n",
        encoding="utf-8",
    )
    cfg, _raw = load_runtime_from_toml(path)
    assert cfg.time.method == "rk4"


def test_maintained_runtime_decks_carry_no_removed_time_keys() -> None:
    """Every shipped deck must load under the current schema."""

    paths = sorted((REPO_ROOT / "examples").rglob("*.toml"))
    paths += sorted((REPO_ROOT / "benchmarks").rglob("*.toml"))
    paths += sorted((REPO_ROOT / "tools" / "comparison").rglob("*.toml"))
    assert paths
    for path in paths:
        time_section = load_toml(path).get("time", {})
        assert not [key for key in time_section if "diffrax" in key], path


def test_maintained_runtime_decks_declare_schema_v1() -> None:
    paths = sorted((REPO_ROOT / "examples").rglob("*.toml"))
    paths += [
        REPO_ROOT / "benchmarks" / name
        for name in (
            "collisional_zonal_response.toml",
            "runtime_miller_zonal_response.toml",
            "runtime_secondary_slab.toml",
            "runtime_w7x_zonal_response_vmec.toml",
        )
    ]
    assert paths
    for path in paths:
        assert load_toml(path).get("schema_version") == 1, path


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
    # dt is set by the parallel-streaming CFL of the equilibrium the deck loads,
    # not by taste: at Nm = 32 the runtime's own bound is 0.0311, and the 0.05
    # this deck used to ship went non-finite at t = 5.65 of a requested 60.
    # Raising it, or raising Nm, needs the bound re-derived first.
    assert data["run"]["dt"] == pytest.approx(0.02)
    assert data["run"]["steps"] == 3000
    assert cfg.time.dt == pytest.approx(data["run"]["dt"])
    assert data["run"]["steps"] * data["run"]["dt"] == pytest.approx(cfg.time.t_max)


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


# ---- from test_resolution_estimator.py ----
# Resolution-estimator contract against the 2026-08 y0=14 ladder.


LADDER_DKY = 1.0 / 14.0


SCAN_CASES = {
    "tok_diiid": ("wout_DIII-D_lasym_false.nc", 0.1381, 1, "tokamak"),
    "qhs": ("wout_NuhrenbergZille_1988_QHS.nc", 0.5522, 3, "stellarator"),
    "qi": ("wout_QI_stel_seed_3127.nc", 0.5909, 19, "stellarator"),
    "qa_b0p5": (
        "wout_LandremanPaul2021_QA_beta0p5_bootstrap.nc",
        0.6193,
        1,
        "stellarator",
    ),
    "qh": (
        "wout_LandremanPaul2021_QH_reactorScale_lowres.nc",
        0.6744,
        2,
        "stellarator",
    ),
    "qa_b2p5": (
        "wout_LandremanPaul2021_QA_beta2p5_bootstrap.nc",
        0.6931,
        1,
        "stellarator",
    ),
    "qa_vac": ("wout_LandremanPaul2021_QA_lowres.nc", 0.7964, 1, "stellarator"),
}


TIER_RUNGS = {
    "tokamak": {"preview": 64, "standard": 96, "cautious": 128},
    "stellarator": {"preview": 96, "standard": 128, "cautious": 192},
}


def _features(
    anisotropy: float, wells: int = 1, shat: float = 1.0, nfp: int = 2
) -> GeometryFeatures:
    return GeometryFeatures(
        anisotropy=anisotropy, shat=shat, q=2.0, nfp=nfp, bmag_wells=wells, zp=1.0
    )


def test_class_split_is_nfp_first_anisotropy_second() -> None:
    assert geometry_class(_features(0.14, nfp=1)) == "tokamak"
    # nfp > 1 is a stellarator no matter how tokamak-like the metric looks.
    assert geometry_class(_features(0.14, nfp=2)) == "stellarator"
    # An nfp=1 equilibrium with a stellarator-band metric rounds up in cost.
    assert geometry_class(_features(0.55, nfp=1)) == "stellarator"


def test_tiers_land_on_the_calibrated_rungs() -> None:
    for klass, nfp in (("tokamak", 1), ("stellarator", 2)):
        anisotropy = 0.14 if klass == "tokamak" else 0.62
        for tier, rung in TIER_RUNGS[klass].items():
            est = resolution_from_features(
                _features(anisotropy, nfp=nfp), dky=LADDER_DKY, target_error=tier
            )
            assert (est["nx"], est["ny"]) == (rung, rung), (klass, tier)
            assert est["geometry_class"] == klass


def test_target_error_tiers_are_monotone() -> None:
    for nfp in (1, 2):
        for anisotropy in (0.14, 0.55, 0.80):
            rungs = [
                PERP_LADDER.index(
                    int(
                        resolution_from_features(
                            _features(anisotropy, nfp=nfp),
                            dky=LADDER_DKY,
                            target_error=t,
                        )["nx"]
                    )
                )
                for t in ("preview", "standard", "cautious")
            ]
            assert rungs[0] <= rungs[1] <= rungs[2]


def test_invalid_target_error_raises() -> None:
    with pytest.raises(ValueError, match="target_error"):
        ky_max_target(_features(0.5), "fast")


def test_perp_points_reproduce_ladder_reaches() -> None:
    # At the ladder's box the class tiers land exactly on the ladder rungs.
    for target, rung in ((1.5, 64), (2.2, 96), (2.9, 128), (4.4, 192)):
        assert perp_points_for(LADDER_DKY, target) == rung


def test_velocity_floors() -> None:
    base = resolution_from_features(_features(0.5), dky=LADDER_DKY)
    assert (base["nl"], base["nm"]) == (4, 8)
    no_hyper = resolution_from_features(
        _features(0.5), dky=LADDER_DKY, hypercollisions=False
    )
    assert (no_hyper["nl"], no_hyper["nm"]) == (6, 12)
    kin_e = resolution_from_features(
        _features(0.5), dky=LADDER_DKY, kinetic_electrons=True
    )
    assert kin_e["nm"] >= 16


def test_parallel_floor_notes_and_cautious_nz() -> None:
    # QI-like tube: 19 deep |B| wells ask Nz >= 114 -> 120 after rounding.
    feats = _features(0.59, wells=19, shat=-0.34)
    std = resolution_from_features(feats, dky=LADDER_DKY)
    assert std["nz"] == 48  # ladder default kept; the floor is advisory
    assert any("Nz >= 120" in note for note in std["notes"])
    caut = resolution_from_features(feats, dky=LADDER_DKY, target_error="cautious")
    assert caut["nz"] == 120


def test_stellarator_upper_estimate_note_and_cheaper_box() -> None:
    est = resolution_from_features(_features(0.7964, shat=0.02), dky=1.0 / 21.0)
    # A wider box than the calibrated one earns the cheaper-box pointer.
    assert any("y0 = 14" in note for note in est["notes"])
    assert any("upper estimate" in note for note in est["notes"])
    assert any("low-shear" in note for note in est["notes"])
    at_ladder = resolution_from_features(_features(0.62), dky=LADDER_DKY)
    assert not any("cheaper box" in note for note in at_ladder["notes"])
    assert not any(
        "upper estimate" in note
        for note in resolution_from_features(
            _features(0.14, nfp=1), dky=LADDER_DKY
        )["notes"]
    )


def test_estimate_flag_short_circuits_before_any_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str | None, str]] = []

    def _fake_print(wout_path: Path, config_arg: str | None, *, target_error: str) -> None:
        calls.append((wout_path, config_arg, target_error))

    monkeypatch.setattr(runtime_wout, "_print_resolution_estimate", _fake_print)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        runtime_wout.wout_shorthand_args("wout_case.nc", None, ["--estimate=preview"])
    assert excinfo.value.code == 0
    assert calls and calls[0][2] == "preview"
    assert list(tmp_path.iterdir()) == []  # no resolved deck, no output dir


def test_estimate_flag_defaults_to_standard() -> None:
    args = ["--out", "x", "--estimate", "--progress"]
    assert runtime_wout._pop_estimate_flag(args) == "standard"
    assert args == ["--out", "x", "--progress"]
    assert runtime_wout._pop_estimate_flag(args) is None


def _scan_wouts_dir() -> Path | None:
    candidates = [os.environ.get("GKX_RESOLUTION_SCAN_WOUTS")]
    candidates.append(str(Path.home() / "gkx-runs/resolution_scan/wouts"))
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand)
    return None


@pytest.mark.integration
def test_estimator_end_to_end_on_scan_equilibria() -> None:
    """Full wout -> geometry -> estimate path against the recorded features."""

    wouts = _scan_wouts_dir()
    if wouts is None:
        pytest.skip("resolution-scan wout files not present on this machine")
    from gkx.workflows.runtime.resolution import estimate_resolution

    checked = 0
    for _case, (fname, anisotropy, wells, klass) in SCAN_CASES.items():
        path = wouts / fname
        if not path.is_file():
            continue
        checked += 1
        est = estimate_resolution(path, torflux=0.64)
        feats = est["features"]
        assert feats.anisotropy == pytest.approx(anisotropy, abs=0.02)
        assert feats.bmag_wells == wells
        assert est["geometry_class"] == klass
        assert 0.0 < est["dt"] <= 0.1
    if checked == 0:
        pytest.skip("no scan wout files found in the scan directory")


# ---- from test_runtime_policies.py ----


def test_nearest_index_from_candidates_locks_retained_mode_tie_order() -> None:
    values = np.asarray([0.0, 0.5, 1.0, 1.5])

    assert _nearest_index_from_candidates(values, 0.75, np.asarray([1, 2])) == 1
    assert _nearest_index_from_candidates(values, 1.4, np.asarray([0, 2])) == 2

    with pytest.raises(ValueError, match="values must be non-empty"):
        _nearest_index_from_candidates(np.asarray([]), 1.0, np.asarray([0]))
    with pytest.raises(ValueError, match="candidate indices"):
        _nearest_index_from_candidates(values, 1.0, np.asarray([], dtype=int))


def test_active_dealias_indices_fall_back_to_full_axis_for_empty_masks() -> None:
    empty_mask = np.zeros((3, 4), dtype=bool)

    np.testing.assert_array_equal(_active_ky_indices(empty_mask, 3), [0, 1, 2])
    np.testing.assert_array_equal(_active_kx_indices(empty_mask, 1, 4), [0, 1, 2, 3])

    mixed_mask = np.asarray(
        [
            [False, False, False, True],
            [False, False, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(_active_ky_indices(mixed_mask, 3), [0, 2])
    np.testing.assert_array_equal(_active_kx_indices(mixed_mask, 2, 4), [0])


def test_select_nonlinear_mode_indices_uses_nearest_retained_dealiased_mode() -> None:
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4, 0.8]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.asarray(
            [
                [False, False, True],
                [False, False, False],
                [True, False, False],
            ],
            dtype=bool,
        ),
    )

    ky_idx, kx_idx = _select_nonlinear_mode_indices(
        grid,
        ky_target=0.39,
        kx_target=0.2,
        use_dealias_mask=True,
    )

    assert (ky_idx, kx_idx) == (0, 2)


def test_select_nonlinear_mode_indices_validates_dealias_mask_shape() -> None:
    bad_grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.ones((2, 2), dtype=bool),
    )

    with pytest.raises(ValueError, match="dealias_mask shape"):
        _select_nonlinear_mode_indices(
            bad_grid,
            ky_target=0.4,
            kx_target=0.0,
            use_dealias_mask=True,
        )


def test_validate_dealias_mask_shape_returns_boolean_view() -> None:
    mask = _validate_dealias_mask_shape(
        np.asarray([[1, 0], [0, 1]], dtype=int),
        ky_size=2,
        kx_size=2,
    )

    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_runtime_independent_parallel_plan_serializes_argument_policy() -> None:
    cfg = SimpleNamespace(parallel=None)

    plan = _runtime_independent_parallel_plan(
        cfg,
        problem_size=3,
        workers=8,
        executor="threads",
    )
    empty = _runtime_independent_parallel_plan(
        cfg,
        problem_size=0,
        workers=2,
        executor="process",
    )

    assert isinstance(plan, RuntimeIndependentParallelPlan)
    assert plan.requested_workers == 8
    assert plan.effective_workers == 3
    assert plan.executor == "thread"
    assert plan.source == "arguments"
    assert plan.enabled is True
    assert plan.to_dict()["enabled"] is True
    assert empty.effective_workers == 0
    assert empty.enabled is False


def test_runtime_independent_parallel_plan_honors_batch_config_and_guards() -> None:
    cfg = SimpleNamespace(
        parallel=SimpleNamespace(
            strategy="batch",
            axis="ky",
            num_devices=4,
            batch_size=None,
            backend="processes",
        )
    )

    plan = _runtime_independent_parallel_plan(
        cfg,
        problem_size=2,
        workers=1,
        executor="thread",
    )

    assert plan.requested_workers == 4
    assert plan.effective_workers == 2
    assert plan.executor == "process"
    assert plan.strategy == "batch"
    assert plan.axis == "ky"
    assert plan.source == "runtime_config"

    with pytest.raises(ValueError, match="problem_size"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=-1,
            workers=1,
            executor="thread",
        )
    with pytest.raises(ValueError, match="workers"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=1,
            workers=0,
            executor="thread",
        )
    with pytest.raises(ValueError, match="parallel_executor"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(parallel=None),
            problem_size=1,
            workers=1,
            executor="gpu",
        )
    with pytest.raises(ValueError, match="axis='ky'"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="batch", axis="kx", backend="auto")
            ),
            problem_size=2,
            workers=1,
            executor="thread",
        )
    with pytest.raises(ValueError, match="independent scans"):
        _runtime_independent_parallel_plan(
            SimpleNamespace(
                parallel=SimpleNamespace(strategy="batch", axis="ky", backend="mpi")
            ),
            problem_size=2,
            workers=1,
            executor="thread",
        )


def test_runtime_solver_and_combined_ky_policy_helpers_normalize_inputs() -> None:
    assert _normalize_linear_solver_name(" explicit_time ") == "explicit_time"
    assert _normalize_linear_solver_name(" Krylov ") == "krylov"

    assert _parallel_requests_combined_ky_scan(SimpleNamespace(parallel=None)) is False
    assert (
        _parallel_requests_combined_ky_scan(
            SimpleNamespace(parallel=SimpleNamespace(strategy="Combined_KY", axis="KY"))
        )
        is True
    )
    assert (
        _parallel_requests_combined_ky_scan(
            SimpleNamespace(parallel=SimpleNamespace(strategy="combined_ky", axis="kx"))
        )
        is False
    )


def test_runtime_mode_and_axis_helpers_cover_unmasked_selection() -> None:
    single_z = SimpleNamespace(z=np.asarray([0.0]), kx=np.asarray([1.0]))
    centered = SimpleNamespace(
        z=np.linspace(-1.0, 1.0, 5),
        kx=np.asarray([2.0, -0.05, 0.2]),
    )
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.5, 1.0]),
        kx=np.asarray([-1.0, 0.2, 2.0]),
        dealias_mask=np.zeros((3, 3), dtype=bool),
    )

    assert _midplane_index(single_z) == 0
    assert _midplane_index(centered) == 3
    assert _zero_kx_index(centered) == 1
    assert _select_nonlinear_mode_indices(
        grid,
        ky_target=0.6,
        kx_target=None,
        use_dealias_mask=False,
    ) == (1, 1)


def test_runtime_step_and_external_phi_policies_are_fail_closed() -> None:
    fixed = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=True, t_max=1.0, dt=0.25, dt_max=None)
    )
    adaptive_capped = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=False, t_max=1.0, dt=0.2, dt_max=0.3)
    )
    adaptive_uncapped = SimpleNamespace(
        time=SimpleNamespace(fixed_dt=False, t_max=1.0, dt=0.2, dt_max=None)
    )

    assert _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=None) == 4
    assert _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=7) == 7
    assert _infer_runtime_nonlinear_steps(adaptive_capped, dt=0.2, steps=None) == 4
    assert _infer_runtime_nonlinear_steps(adaptive_uncapped, dt=0.2, steps=None) == 5
    with pytest.raises(ValueError, match="steps"):
        _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=0)

    assert (
        _runtime_external_phi(
            SimpleNamespace(expert=SimpleNamespace(source=" default ", phi_ext=3.0))
        )
        is None
    )
    assert _runtime_external_phi(
        SimpleNamespace(expert=SimpleNamespace(source="phiext_full", phi_ext=2.5))
    ) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="unsupported expert.source"):
        _runtime_external_phi(
            SimpleNamespace(expert=SimpleNamespace(source="external_phi", phi_ext=1.0))
        )
