"""Runtime configuration layer: TOML/deck loading into RuntimeConfig, the resolution estimator, and the pure selection policies applied before a run starts."""

from __future__ import annotations

from dataclasses import replace

from gkx.workflows.runtime import wout as runtime_wout
from gkx.config import (
    RuntimeConfig,
    RuntimeParallelConfig,
    RuntimeQuasilinearConfig,
)
from gkx.runtime import (
    RuntimeIndependentParallelPlan,
    _runtime_external_phi,
    _select_nonlinear_mode_indices,
)
from gkx.workflows.nonlinear import (
    _active_kx_indices,
    _active_ky_indices,
    _infer_runtime_nonlinear_steps,
    _nearest_index_from_candidates,
    _validate_dealias_mask_shape,
)
from gkx.workflows.linear import (
    _midplane_index,
    _normalize_linear_solver_name,
    _zero_kx_index,
)
from gkx.workflows.runtime.orchestration_scan import (
    _parallel_requests_combined_ky_scan,
    _runtime_independent_parallel_plan,
)
from gkx.workflows.runtime.wout import (
    PERP_LADDER,
    GeometryFeatures,
    geometry_class,
    ky_max_target,
    perp_points_for,
    resolution_from_features,
    direct_config_shorthand_args,
)
from gkx.workflows.runtime.toml import (
    is_runtime_toml,
    load_toml,
    load_runtime_from_toml,
    toml_shorthand_command,
)
from pathlib import Path
from support.paths import REPO_ROOT
from support.helpers import assert_fields
from types import SimpleNamespace
import json
import numpy as np
import os
import pytest


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
    assert_fields(
        RuntimeConfig(),
        {
            "geometry.drift_scale": 1.0,
            "normalization.diagnostic_norm": "rho_star",
            "normalization.flux_scale": 1.0,
            "collisions.p_hyper_m": None,
            "collisions.damp_ends_amp": 0.1,
            "collisions.damp_ends_widthfrac": 0.125,
            "parallel.strategy": "serial",
            "parallel.axis": "ky",
        },
    )


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
        'schema_version = 1\n[time]\ndiffrax_solver = "Tsit5"\n'
        "diffrax_max_steps = 20000\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="diffrax_solver") as excinfo:
        load_runtime_from_toml(path)
    assert "diffrax_max_steps" in str(excinfo.value)

    # Selecting the native owner explicitly still loads.
    path.write_text(
        'schema_version = 1\n[time]\nt_max = 1.0\nmethod = "rk4"\n',
        encoding="utf-8",
    )
    cfg, _raw = load_runtime_from_toml(path)
    assert cfg.time.method == "rk4"


def test_runtime_toml_rejects_misspelled_keys_and_sections(tmp_path: Path) -> None:
    """A misspelled deck key must fail, not silently run the default physics.

    Before this check ``[geometry] shat = 0.3`` loaded as ``s_hat = 0.8``,
    ``[grid] nz = 32`` as ``Nz = 64`` and a ``[collision]`` table vanished.
    """

    path = tmp_path / "case.toml"
    cases = (
        ("[geometry]\nshat = 0.3\n", "geometry.*'shat'.*s_hat"),
        ("[grid]\nnz = 32\n", "grid.*'nz'.*Nz"),
        ("[time]\ntmax = 5.0\n", "time.*'tmax'.*t_max"),
        ("[init]\ninit_amplitude = 1.0\n", "init.*'init_amplitude'.*init_amp"),
        ("[collision]\nnu_hyper = 5.0\n", "'collision'.*'collisions'"),
    )
    for body, pattern in cases:
        path.write_text("schema_version = 1\n" + body, encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            load_runtime_from_toml(path)

    for body, pattern in (
        ("[run]\nnl = 8\n", r"\[run\].*'nl'.*'Nl'"),
        ("[scan]\nky_values = [0.1]\n", r"\[scan\].*'ky_values'"),
    ):
        path.write_text("schema_version = 1\n" + body, encoding="utf-8")
        with pytest.raises(ValueError, match=pattern):
            load_runtime_from_toml(path)

    path.write_text(
        "schema_version = 1\n[geometry]\ns_hat = 0.3\n[grid]\nNz = 32\n",
        encoding="utf-8",
    )
    cfg, _raw = load_runtime_from_toml(path)
    assert (cfg.geometry.s_hat, cfg.grid.Nz) == (0.3, 32)


def test_zonal_deck_kx_selects_the_fitted_mode() -> None:
    """``[run] kx`` picks the fitted kx; it used to be ignored for kx = 0.

    On the Merlo Case III deck (ky = 0, kx = 0.05) the ignored key left the
    run fitting the (kx, ky) = (0, 0) mode, whose potential is identically zero
    with Boltzmann electrons.
    """

    from gkx.runtime import (
        run_runtime_linear,
    )

    path = REPO_ROOT / "benchmarks" / "cases" / "miller_zonal_response.toml"
    cfg, data = load_runtime_from_toml(path)
    cfg = replace(cfg, grid=replace(cfg.grid, Nz=16))
    res = run_runtime_linear(
        cfg,
        ky_target=0.0,
        kx_target=data["run"]["kx"],
        Nl=2,
        Nm=4,
        solver="time",
        dt=0.01,
        steps=40,
        sample_stride=4,
        fit_signal="phi",
        require_positive=False,
        min_points=4,
    )
    assert res.selection.kx_index != 0
    assert np.max(np.abs(res.signal)) > 0.0


def _maintained_decks(*, include_comparison: bool = True) -> list[Path]:
    paths = sorted((REPO_ROOT / "examples").rglob("*.toml"))
    paths += sorted((REPO_ROOT / "benchmarks").rglob("*.toml"))
    if include_comparison:
        paths += sorted((REPO_ROOT / "tools" / "comparison").rglob("*.toml"))
    assert paths
    return paths


def test_maintained_runtime_decks_carry_no_removed_time_keys() -> None:
    """Every shipped deck must load under the current schema."""

    for path in _maintained_decks():
        time_section = load_toml(path).get("time", {})
        assert not [key for key in time_section if "diffrax" in key], path


def test_maintained_runtime_decks_declare_schema_v1() -> None:
    paths = sorted((REPO_ROOT / "examples").rglob("*.toml"))
    paths += sorted((REPO_ROOT / "benchmarks" / "cases").glob("*.toml"))
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


def _minimal_deck(time_table: str) -> str:
    return f"""
schema_version = 1

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
kinetic = true

[grid]
Nx = 1
Ny = 8
Nz = 16
{time_table}
"""


@pytest.mark.parametrize(
    ("time_table", "expected"),
    [
        # A step nobody chose must not be applied as a fixed one.
        # ``TimeConfig.dt`` defaults to 0.1, which is ~7.8x the CFL-stable step
        # of the shipped Cyclone deck: applied fixed, it overflows to a
        # FloatingPointError whatever the scheme. With the controller on, the
        # same deck integrates and rk4 covers the horizon in 29.1% fewer
        # right-hand-side evaluations than rk2. So the integrator and the step
        # policy are defaulted as a pair, and only where the deck chose no step.
        pytest.param("", (None, False, "rk4"), id="no-time-table"),
        pytest.param("\n[time]\nt_max = 4.0\n", (None, False, "rk4"), id="no-step"),
        # Every shipped deck sets ``dt``, so none of them may move. Fourteen
        # shipped decks and parity fixtures omit ``fixed_dt`` and depend on it
        # being ``True`` at their own ``dt``; several back evidence-ledger rows.
        # A chosen step keeps rk2 too, because a fixed step gives rk4 no
        # step-size compensation for its four stages -- it would simply be
        # twice the cost.
        pytest.param(
            "\n[time]\nt_max = 4.0\ndt = 0.002\n",
            (0.002, True, "rk2"),
            id="chosen-step",
        ),
        # The coupling supplies defaults; it never overrides what the deck said.
        pytest.param(
            '\n[time]\nt_max = 4.0\nmethod = "rk3"\nfixed_dt = true\n',
            (None, True, "rk3"),
            id="explicit-overrides",
        ),
    ],
)
def test_step_choice_selects_the_integrator_and_step_pairing(
    tmp_path: Path, time_table: str, expected: tuple
) -> None:
    dt, fixed_dt, method = expected
    path = tmp_path / "deck.toml"
    path.write_text(_minimal_deck(time_table), encoding="utf-8")
    cfg, _ = load_runtime_from_toml(path)
    if dt is not None:
        assert cfg.time.dt == dt
    assert cfg.time.fixed_dt is fixed_dt
    assert cfg.time.method == method


def test_every_shipped_deck_chooses_its_own_step() -> None:
    """The guard that makes the pairing change inert for shipped decks.

    If a deck ever ships without ``dt``, it silently moves onto the CFL
    controller and its recorded numbers move with it. This test is what makes
    that a deliberate act rather than an accident.
    """

    import tomllib

    missing = []
    for deck in _maintained_decks():
        data = tomllib.loads(deck.read_text(encoding="utf-8"))
        if "schema_version" not in data and "grid" not in data:
            continue  # not a GKX runtime deck (e.g. a GX-schema reference file)
        table = data.get("time")
        if isinstance(table, dict) and "dt" not in table:
            missing.append(str(deck.relative_to(REPO_ROOT)))
    assert missing == []


def test_load_runtime_from_toml_roundtrip(tmp_path: Path) -> None:
    species = "\n".join(
        f"""
[[species]]
name = "{name}"
charge = {charge}
mass = {mass}
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true
"""
        for name, charge, mass in (("ion", 1.0, 1.0), ("electron", -1.0, 0.00027248))
    )
    toml = (
        species
        + """
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
    )
    path = tmp_path / "runtime.toml"
    path.write_text(toml, encoding="utf-8")
    cfg, data = load_runtime_from_toml(path)
    assert isinstance(data, dict)
    assert_fields(
        cfg,
        {
            "grid.Ny": 8,
            "physics.electromagnetic": True,
            "physics.use_apar": True,
            "physics.adiabatic_electrons": False,
            "physics.beta": 0.2,
            "normalization.contract": "kbm",
            "normalization.omega_star_scale": 0.7,
            "expert.fixed_mode": True,
            "expert.iky_fixed": 1,
            "expert.ikx_fixed": 0,
            "init.init_file": str(Path("/tmp/restart.bin").resolve()),
            "init.init_file_scale": 5.0,
            "init.init_file_mode": "add",
            "output.path": str((tmp_path / "tools_out" / "runtime_case").resolve()),
            "quasilinear.enabled": True,
            "quasilinear.mode": "saturated",
            "quasilinear.saturation_rule": "mixing_length",
            "quasilinear.csat": 0.7,
            "quasilinear.channels": ("es",),
            "quasilinear.output_path": str(
                (tmp_path / "tools_out" / "ql_case").resolve()
            ),
            "parallel.strategy": "combined_ky",
            "parallel.axis": "ky",
            "parallel.batch_size": 3,
            "parallel.num_devices": 2,
            "parallel.strict_identity": True,
            "parallel.profile": True,
            "species[1].charge": -1.0,
        },
    )
    assert len(cfg.species) == 2


def test_runtime_parallel_config_validates_values() -> None:
    assert RuntimeParallelConfig(strategy="batch-ky").strategy == "combined_ky"
    for bad in ({"strategy": "unknown"}, {"batch_size": 0}, {"num_devices": 0}):
        with pytest.raises(ValueError):
            RuntimeParallelConfig(**bad)


def test_gx_aligned_kbm_runtime_examples_keep_end_damping_enabled() -> None:
    cfg_dir = REPO_ROOT / "benchmarks" / "cases"
    paths = [
        cfg_dir / "kbm_nonlinear.toml",
        cfg_dir / "kbm_nonlinear_seed.toml",
        cfg_dir / "kbm_nonlinear_short.toml",
        cfg_dir / "kbm_nonlinear_short_lockin.toml",
        cfg_dir / "kbm_nonlinear_t100.toml",
        cfg_dir / "kbm_nonlinear_t100_nx4ny8_dt9e4.toml",
    ]
    for path in paths:
        cfg, _ = load_runtime_from_toml(path)
        assert cfg.terms.end_damping == pytest.approx(1.0), path.name


def test_linear_axisymmetric_runtime_examples_keep_parity_collision_contract() -> None:
    cfg_dir = REPO_ROOT / "benchmarks" / "cases"
    cyclone = REPO_ROOT / "examples" / "01_linear_tokamak" / "case_full.toml"
    kbm = REPO_ROOT / "examples" / "06_electromagnetic" / "case_full.toml"
    expected = {
        cyclone: (1.0, 2.0, 0.0, 1.0),
        cfg_dir / "etg_linear.toml": (1.0, 2.0, 0.0, 1.0),
        cfg_dir / "etg_linear_scan.toml": (1.0, 2.0, 0.0, 1.0),
        cfg_dir / "kaw_linear.toml": (1.0, 2.0, 0.0, 1.0),
        kbm: (1.0, 2.0, 0.0, 1.0),
    }
    for name, (nu_h, nu_l, hyper_const, hyper_kz) in expected.items():
        cfg, _ = load_runtime_from_toml(name)
        assert cfg.collisions.nu_hermite == pytest.approx(nu_h), name
        assert cfg.collisions.nu_laguerre == pytest.approx(nu_l), name
        assert cfg.collisions.hypercollisions_const == pytest.approx(hyper_const), name
        assert cfg.collisions.hypercollisions_kz == pytest.approx(hyper_kz), name

    _cfg, cyclone_raw = load_runtime_from_toml(cyclone)
    assert cyclone_raw["fit"]["mode_method"] == "z_index"

    for name in ("etg_linear.toml", "etg_linear_scan.toml"):
        cfg, raw = load_runtime_from_toml(cfg_dir / name)
        assert cfg.time.method == "rk4", name
        assert cfg.time.dt == pytest.approx(1.6e-4), name
        assert cfg.time.t_max == pytest.approx(2.0), name
        assert raw["run"]["solver"] == "time", name
        assert raw["scan"]["solver"] == "time", name


def test_nonaxisymmetric_quasilinear_examples_keep_electrostatic_contract() -> None:
    for name in (
        REPO_ROOT / "examples" / "02_linear_stellarator" / "case_full.toml",
        REPO_ROOT / "benchmarks" / "cases" / "w7x_linear_quasilinear_vmec.toml",
    ):
        cfg, _ = load_runtime_from_toml(name)
        assert cfg.quasilinear.enabled is True, name
        assert cfg.quasilinear.channels == ("es",), name
        assert cfg.physics.electrostatic is True, name
        assert cfg.physics.electromagnetic is False, name
        assert cfg.terms.apar == pytest.approx(0.0), name
        assert cfg.terms.bpar == pytest.approx(0.0), name


_QI_WOUT = REPO_ROOT / "examples" / "vmec" / "wout_nfp3_QI_fixed_resolution_final.nc"
_TOOLS_OUT = REPO_ROOT / "tools_out"

# Each shipped deck's pinned contract, as dotted paths into (cfg, raw data).
_SHIPPED_DECK_CONTRACTS = {
    "benchmarks/cases/etg_nonlinear.toml": {
        "cfg.physics.linear": False,
        "cfg.physics.nonlinear": True,
        "cfg.physics.electrostatic": True,
        "cfg.physics.electromagnetic": False,
        "cfg.physics.adiabatic_ions": False,
        "cfg.physics.adiabatic_electrons": False,
        "cfg.grid.Lx": 1.25,
        "cfg.init.gaussian_init": True,
        "cfg.init.init_single": False,
        "cfg.collisions.hypercollisions_const": 0.0,
        "cfg.collisions.hypercollisions_kz": 1.0,
        "data.run.ky": 5.0,
        "cfg.output.path": str((_TOOLS_OUT / "etg_nonlinear_runtime").resolve()),
        "n_species": 2,
    },
    "benchmarks/cases/w7x_linear_imported_geometry.toml": {
        "cfg.geometry.model": "vmec",
        "cfg.geometry.geometry_file": None,
        "cfg.geometry.vmec_file": str(_QI_WOUT.resolve()),
        "cfg.geometry.torflux": 0.64,
        "cfg.init.init_field": "density",
        "cfg.physics.adiabatic_electrons": True,
        "cfg.normalization.diagnostic_norm": "rho_star",
    },
    "benchmarks/cases/w7x_nonlinear_imported_geometry.toml": {
        "cfg.geometry.model": "vmec",
        "cfg.geometry.geometry_file": None,
        "cfg.geometry.vmec_file": str(_QI_WOUT.resolve()),
        "cfg.geometry.torflux": 0.64,
        "cfg.physics.nonlinear": True,
        "cfg.physics.adiabatic_electrons": True,
        "cfg.physics.collisions": True,
        "cfg.terms.collisions": 1.0,
        "cfg.terms.nonlinear": 1.0,
        "run_has_steps": False,
        "cfg.output.path": str(
            (_TOOLS_OUT / "w7x_nonlinear_imported_runtime").resolve()
        ),
    },
    "examples/04_nonlinear_stellarator/case_full.toml": {
        "cfg.geometry.model": "vmec",
        "cfg.geometry.vmec_file": str(
            (
                REPO_ROOT / "examples" / "vmec" / "wout_NuhrenbergZille_1988_QHS.nc"
            ).resolve()
        ),
        "cfg.geometry.geometry_helper_python": None,
        "cfg.geometry.torflux": 0.64,
        "cfg.physics.nonlinear": True,
        "cfg.physics.adiabatic_electrons": True,
        "cfg.physics.collisions": True,
        "cfg.terms.collisions": 1.0,
        "cfg.terms.nonlinear": 1.0,
        "run_has_steps": False,
        "cfg.output.path": str((_TOOLS_OUT / "hsx_nonlinear_vmec_runtime").resolve()),
    },
    "benchmarks/cases/w7x_nonlinear_vmec_geometry.toml": {
        "cfg.geometry.model": "vmec",
        "cfg.geometry.vmec_file": str(_QI_WOUT.resolve()),
        "cfg.geometry.geometry_helper_python": None,
        "cfg.geometry.torflux": 0.64,
        "cfg.physics.nonlinear": True,
        "cfg.physics.adiabatic_electrons": True,
        "cfg.physics.collisions": True,
        "cfg.terms.collisions": 1.0,
        "run_has_steps": False,
        "cfg.output.path": str((_TOOLS_OUT / "w7x_nonlinear_vmec_runtime").resolve()),
    },
    "benchmarks/cases/secondary_slab.toml": {
        "cfg.geometry.model": "slab",
        "cfg.geometry.s_hat": 1.0e-8,
        "cfg.physics.linear": True,
        "cfg.physics.nonlinear": False,
        "cfg.physics.adiabatic_electrons": True,
    },
    "benchmarks/cases/cyclone_nonlinear_miller.toml": {
        "cfg.geometry.model": "miller",
        "cfg.geometry.q": 1.4,
        "cfg.geometry.s_hat": 0.8,
        "cfg.geometry.rhoc": 0.5,
        "cfg.physics.nonlinear": True,
        "cfg.physics.adiabatic_electrons": True,
    },
    # Merlo Case III contract.
    "benchmarks/cases/miller_zonal_response.toml": {
        "cfg.expert.source": "default",
        "cfg.expert.phi_ext": 0.0,
        "cfg.init.init_field": "density",
        "cfg.init.init_amp": 1.0e-6,
        "cfg.output.save_for_restart": True,
        "cfg.geometry.q": 1.389,
        "cfg.geometry.s_hat": 0.751,
        "cfg.geometry.akappa": 1.4723,
        "cfg.geometry.tri": -0.0070,
        "cfg.geometry.shift": -0.1569,
        "cfg.grid.Nz": 32,
        "data.run.Nl": 4,
        "data.run.kx": 0.05,
        "data.run.ky": 0.0,
        # A zero-gradient relaxation run has no saturation to stop at: the
        # default run_to = "saturation" declares the ~0 heat flux converged
        # inside the first chunk and truncates the trace at t ~ 6 without raising.
        "cfg.time.run_to": "t_max",
    },
    # W7-X test 4 contract.
    "benchmarks/cases/w7x_zonal_response_vmec.toml": {
        "cfg.geometry.model": "vmec",
        "cfg.geometry.vmec_file": str(_QI_WOUT.resolve()),
        "cfg.geometry.torflux": 0.64,
        "cfg.geometry.alpha": 0.0,
        "cfg.geometry.R0": 5.485,
        "cfg.grid.boundary": "linked",
        "cfg.grid.nperiod": 4,
        "cfg.grid.Nz": 256,
        "cfg.init.gaussian_init": True,
        "cfg.init.gaussian_width": 1.0,
        "cfg.init.init_field": "phi",
        "cfg.physics.adiabatic_electrons": True,
        "cfg.physics.nonlinear": False,
        "cfg.physics.collisions": False,
        "cfg.physics.hypercollisions": False,
        "cfg.species[0].tprim": 0.0,
        "cfg.species[0].fprim": 0.0,
        "data.run.ky": 0.0,
        "data.run.kx": 0.05,
        "data.run.Nl": 8,
        "data.run.Nm": 32,
        # dt is set by the parallel-streaming CFL of the equilibrium the deck
        # loads, not by taste: at Nm = 32 the runtime's own bound is 0.0311, and
        # the 0.05 this deck used to ship went non-finite at t = 5.65 of a
        # requested 60. Raising it, or raising Nm, needs the bound re-derived.
        "data.run.dt": 0.02,
        "data.run.steps": 3000,
        "cfg.time.dt": 0.02,
    },
}


@pytest.mark.parametrize("relative", sorted(_SHIPPED_DECK_CONTRACTS))
def test_shipped_deck_keeps_its_contract(relative: str) -> None:
    cfg, data = load_runtime_from_toml(REPO_ROOT / relative)
    assert isinstance(data, dict)
    view = {
        "cfg": cfg,
        "data": data,
        "n_species": len(cfg.species),
        "run_has_steps": "steps" in data.get("run", {}),
    }
    assert_fields(view, _SHIPPED_DECK_CONTRACTS[relative], label=relative)


def test_zonal_response_decks_cover_their_horizon() -> None:
    _cfg, data = load_runtime_from_toml(
        REPO_ROOT / "benchmarks" / "cases" / "miller_zonal_response.toml"
    )
    # Converged Hermite baseline, not the retired Nm=24 one. Nm >= 120 is what
    # puts the whole analysis window before the recurrence onset
    # t_quiet ~ 5.5 sqrt(Nm), and dt <= 0.0025 is what keeps Nm=144 stable --
    # the Hermite streaming CFL scales as sqrt(Nm) and dt=0.005 goes non-finite
    # at t=46.5 there.
    assert data["run"]["Nm"] >= 120
    assert data["run"]["dt"] <= 0.0025
    assert data["run"]["steps"] * data["run"]["dt"] == pytest.approx(60.0)

    cfg, data = load_runtime_from_toml(
        REPO_ROOT / "benchmarks" / "cases" / "w7x_zonal_response_vmec.toml"
    )
    assert data["run"]["steps"] * data["run"]["dt"] == pytest.approx(cfg.time.t_max)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            '[[species]]\nname = "ion"\ncharge = 1.0\nmass = 1.0\ndensity = 1.0\n'
            "temperature = 1.0\ntprim = 3.0\nfprim = 1.0\nkinetic = true\n"
            "[grid]\nNx = 1\nNy = 12\nNz = 32\n"
            '[geometry]\nmodel = "imported-netcdf"\n'
            'geometry_file = "/tmp/w7x.eik.nc"\n'
            "[physics]\nadiabatic_electrons = true\nelectromagnetic = false\n"
            '[run]\nky = 0.3\nNl = 8\nNm = 12\nsolver = "explicit_time"\n',
            {
                "geometry.model": "imported-netcdf",
                "geometry.geometry_file": str(Path("/tmp/w7x.eik.nc").resolve()),
                "physics.adiabatic_electrons": True,
            },
            id="imported-netcdf",
        ),
        pytest.param(
            '[geometry]\nmodel = "desc-eik"\ngeometry_file = "/tmp/w7x-desc.eik.nc"\n',
            {
                "geometry.model": "desc-eik",
                "geometry.geometry_file": str(Path("/tmp/w7x-desc.eik.nc").resolve()),
            },
            id="desc-eik-alias",
        ),
        pytest.param(
            '[geometry]\nmodel = "vmec"\nvmec_file = "/tmp/wout_test.nc"\n'
            'torflux = 0.64\ngeometry_helper_python = "python3"\n',
            {
                "geometry.model": "vmec",
                "geometry.vmec_file": str(Path("/tmp/wout_test.nc").resolve()),
                "geometry.geometry_helper_python": "python3",
            },
            id="vmec-helper-python",
        ),
        pytest.param(
            '[geometry]\nmodel = "vmec"\nvmec_file = "/tmp/wout_test.nc"\n'
            'torflux = 0.64\ngeometry_helper_python = "python3"\n'
            'geometry_helper_repo = "/tmp/helper"\n',
            {
                "geometry.geometry_helper_python": "python3",
                "geometry.geometry_helper_repo": str(Path("/tmp/helper")),
            },
            id="helper-fields",
        ),
        pytest.param(
            '[geometry]\nmodel = "miller"\nrhoc = 0.5\nq = 1.4\ns_hat = 0.8\n'
            "R0 = 2.77778\nR_geo = 2.77778\nshift = 0.0\nakappa = 1.0\n"
            "akappri = 0.0\ntri = 0.0\ntripri = 0.0\nbetaprim = 0.0\n"
            'geometry_helper_python = "python3"\n',
            {
                "geometry.model": "miller",
                "geometry.rhoc": 0.5,
                "geometry.R_geo": 2.77778,
                "geometry.akappa": 1.0,
                "geometry.tripri": 0.0,
                "geometry.geometry_helper_python": "python3",
            },
            id="miller",
        ),
    ],
)
def test_load_runtime_from_toml_keeps_geometry_fields(
    tmp_path: Path, body: str, expected: dict
) -> None:
    path = tmp_path / "runtime_geometry.toml"
    path.write_text(body, encoding="utf-8")
    cfg, data = load_runtime_from_toml(path)
    assert isinstance(data, dict)
    assert_fields(cfg, expected)


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

    expected = {
        "geometry.vmec_file": "vmec/wout.nc",
        "geometry.geometry_file": "geom/run.eik.nc",
        "init.init_file": "restart/state.bin",
        "output.path": "out/run.out.nc",
        "output.restart_to_file": "out/run.restart.nc",
        "output.restart_from_file": "out/run.resume.nc",
    }
    assert_fields(cfg, {k: str((tmp_path / v).resolve()) for k, v in expected.items()})


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
        for note in resolution_from_features(_features(0.14, nfp=1), dky=LADDER_DKY)[
            "notes"
        ]
    )


def test_estimate_flag_short_circuits_before_any_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, str | None, str]] = []

    def _fake_print(
        wout_path: Path, config_arg: str | None, *, target_error: str
    ) -> None:
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
    from gkx.workflows.runtime.wout import estimate_resolution

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
    mask = np.asarray(
        [[False, False, True], [False, False, False], [True, False, False]], dtype=bool
    )
    grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4, 0.8]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=mask,
    )
    assert _select_nonlinear_mode_indices(
        grid, ky_target=0.39, kx_target=0.2, use_dealias_mask=True
    ) == (0, 2)

    bad_grid = SimpleNamespace(
        ky=np.asarray([0.0, 0.4]),
        kx=np.asarray([-1.0, 0.0, 1.0]),
        dealias_mask=np.ones((2, 2), dtype=bool),
    )
    with pytest.raises(ValueError, match="dealias_mask shape"):
        _select_nonlinear_mode_indices(
            bad_grid, ky_target=0.4, kx_target=0.0, use_dealias_mask=True
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
        cfg, problem_size=3, workers=8, executor="threads"
    )
    empty = _runtime_independent_parallel_plan(
        cfg, problem_size=0, workers=2, executor="process"
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
        cfg, problem_size=2, workers=1, executor="thread"
    )

    assert plan.requested_workers == 4
    assert plan.effective_workers == 2
    assert plan.executor == "process"
    assert plan.strategy == "batch"
    assert plan.axis == "ky"
    assert plan.source == "runtime_config"

    def batch(axis: str, backend: str) -> SimpleNamespace:
        return SimpleNamespace(
            parallel=SimpleNamespace(strategy="batch", axis=axis, backend=backend)
        )

    serial = SimpleNamespace(parallel=None)
    for config, problem_size, workers, executor, match in (
        (serial, -1, 1, "thread", "problem_size"),
        (serial, 1, 0, "thread", "workers"),
        (serial, 1, 1, "gpu", "parallel_executor"),
        (batch("kx", "auto"), 2, 1, "thread", "axis='ky'"),
        (batch("ky", "mpi"), 2, 1, "thread", "independent scans"),
    ):
        with pytest.raises(ValueError, match=match):
            _runtime_independent_parallel_plan(
                config, problem_size=problem_size, workers=workers, executor=executor
            )


def test_runtime_solver_and_combined_ky_policy_helpers_normalize_inputs() -> None:
    assert _normalize_linear_solver_name(" explicit_time ") == "explicit_time"
    assert _normalize_linear_solver_name(" Krylov ") == "krylov"

    assert _parallel_requests_combined_ky_scan(SimpleNamespace(parallel=None)) is False
    for strategy, axis, expected in (
        ("Combined_KY", "KY", True),
        ("combined_ky", "kx", False),
    ):
        cfg = SimpleNamespace(parallel=SimpleNamespace(strategy=strategy, axis=axis))
        assert _parallel_requests_combined_ky_scan(cfg) is expected


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
    def time_cfg(fixed_dt: bool, dt: float, dt_max) -> SimpleNamespace:
        return SimpleNamespace(
            time=SimpleNamespace(fixed_dt=fixed_dt, t_max=1.0, dt=dt, dt_max=dt_max)
        )

    fixed = time_cfg(True, 0.25, None)
    for cfg, dt, steps, expected in (
        (fixed, 0.125, None, 4),
        (fixed, 0.125, 7, 7),
        (time_cfg(False, 0.2, 0.3), 0.2, None, 4),
        (time_cfg(False, 0.2, None), 0.2, None, 5),
    ):
        assert _infer_runtime_nonlinear_steps(cfg, dt=dt, steps=steps) == expected
    with pytest.raises(ValueError, match="steps"):
        _infer_runtime_nonlinear_steps(fixed, dt=0.125, steps=0)

    def expert(source: str, phi_ext: float) -> SimpleNamespace:
        return SimpleNamespace(expert=SimpleNamespace(source=source, phi_ext=phi_ext))

    assert _runtime_external_phi(expert(" default ", 3.0)) is None
    assert _runtime_external_phi(expert("phiext_full", 2.5)) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="unsupported expert.source"):
        _runtime_external_phi(expert("external_phi", 1.0))
