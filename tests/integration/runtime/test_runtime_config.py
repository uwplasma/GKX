"""Runtime configuration layer: TOML/deck loading into RuntimeConfig, the resolution estimator, and the pure selection policies applied before a run starts."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from gkx.workflows.runtime import wout as runtime_wout
from gkx import config as gkx_config
from gkx.config import (
    RuntimeConfig,
    RuntimeParallelConfig,
    RuntimeQuasilinearConfig,
)
from gkx.runtime import (
    RuntimeIndependentParallelPlan,
    _runtime_external_phi,
)
from gkx.workflows.nonlinear import (
    _infer_runtime_nonlinear_steps,
    _nearest_index_from_candidates,
)
from gkx.workflows.linear import (
    _normalize_linear_solver_name,
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
)
from gkx.workflows.runtime.toml import (
    load_toml,
    load_runtime_from_toml,
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
            "collisions.damp_ends_amp": None,
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


def test_runtime_toml_removed_keys_load_at_their_old_default_and_refuse_others(
    tmp_path: Path,
) -> None:
    """Resolved decks of older versions carry removed keys at their old value."""

    path = tmp_path / "case.toml"
    path.write_text(
        "schema_version = 1\n[physics]\nz_ion = 1.0\n[time]\nsave_state = false\n",
        encoding="utf-8",
    )
    load_runtime_from_toml(path)
    path.write_text("schema_version = 1\n[physics]\ntau_fac = 2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[physics\] tau_fac"):
        load_runtime_from_toml(path)


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


# Decks that leave the step to the CFL controller and the stepper to the default
# (``method = "auto"`` for adiabatic nonlinear decks, ``imex-ark3`` for low-beta
# kinetic electrons). Every other shipped deck pins ``dt``: linear fits and parity
# runs reproduce a published step. A deck joins or leaves this set on purpose.
_DEFAULT_STEPPER_DECKS = {
    "benchmarks/cases/circular_vmec_nonlinear.toml",
    "benchmarks/cases/cyclone_nonlinear_miller.toml",
    "benchmarks/cases/cyclone_nonlinear_t400.toml",
    "benchmarks/cases/w7x_nonlinear_imported_geometry.toml",
    "benchmarks/cases/w7x_nonlinear_vmec_geometry.toml",
    "examples/03_nonlinear_tokamak/case.toml",
    "examples/03_nonlinear_tokamak/case_full.toml",
    "examples/04_nonlinear_stellarator/case.toml",
    "examples/04_nonlinear_stellarator/case_full.toml",
    "examples/05_kinetic_electrons/case.toml",
    "examples/05_kinetic_electrons/case_stellarator.toml",
    "examples/common_input.toml",
}


def test_only_the_default_stepper_decks_leave_the_step_to_the_controller() -> None:
    """A deck that ships without ``dt`` moves onto the CFL controller and the default
    stepper; this test makes that a deliberate act rather than an accident."""

    import tomllib

    missing = set()
    for deck in _maintained_decks():
        data = tomllib.loads(deck.read_text(encoding="utf-8"))
        if "schema_version" not in data and "grid" not in data:
            continue  # not a GKX runtime deck (e.g. a GX-schema reference file)
        table = data.get("time")
        if isinstance(table, dict) and "dt" not in table:
            missing.add(str(deck.relative_to(REPO_ROOT)))
    assert missing == _DEFAULT_STEPPER_DECKS


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
        "cfg.grid.nperiod": None,
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


def _electron_ion_case(**changes) -> RuntimeConfig:
    from gkx.config import (
        RuntimeCollisionConfig,
        RuntimePhysicsConfig,
        RuntimeSpeciesConfig,
        RuntimeTermsConfig,
        TimeConfig,
    )

    species = (
        RuntimeSpeciesConfig(charge=1.0, mass=1.0, density=1.0, nu=0.1),
        RuntimeSpeciesConfig(charge=-1.0, mass=0.01, density=1.0, nu=0.2),
    )
    base = dict(
        species=species,
        physics=RuntimePhysicsConfig(adiabatic_electrons=False),
        collisions=RuntimeCollisionConfig(electron_ion=True),
        terms=RuntimeTermsConfig(),
        time=TimeConfig(collision_operator="none"),
    )
    return RuntimeConfig(**{**base, **changes})


def test_electron_ion_rates_scope_and_refusals() -> None:
    from gkx.config import (
        RuntimeCollisionConfig,
        RuntimePhysicsConfig,
        RuntimeRunConfig,
    )

    assert RuntimeConfig()._electron_ion_rates() is None
    assert _electron_ion_case()._electron_ion_rates() == (0.0, 0.2)
    ion, electron = _electron_ion_case().species
    physics = RuntimePhysicsConfig(adiabatic_electrons=False)
    cases = {
        "one kinetic ion": dict(species=(ion,)),
        "finite positive": dict(
            species=(ion, replace(electron, mass=0.0)),
        ),
        "kinetic, quasineutral": dict(species=(ion, replace(electron, density=2.0))),
        "beta=0 only": dict(physics=replace(physics, beta=1.0e-3)),
        "standard Dougherty": dict(
            collisions=RuntimeCollisionConfig(nu_hermite=2.0, electron_ion=True)
        ),
        "unsharded": dict(run=RuntimeRunConfig(mu_vperp_max=4.0)),
        "uniform": dict(species=(ion, replace(electron, Nl=2))),
    }
    for match, changes in cases.items():
        with pytest.raises(ValueError, match=match):
            _electron_ion_case(**changes)._electron_ion_rates()


@pytest.mark.parametrize(
    ("cls", "kwargs", "match"),
    [
        ("GridConfig", dict(ky_layout="sideways"), "ky_layout"),
        ("TimeConfig", dict(chunk_steps=0), "chunk_steps"),
        ("TimeConfig", dict(chunk_steps=True), "chunk_steps"),
        ("TimeConfig", dict(chunk_steps=2.5), "chunk_steps"),
        ("TimeConfig", dict(diagnostics_interval=0.0), "diagnostics_interval"),
        ("TimeConfig", dict(diagnostics_interval=float("nan")), "diagnostics_interval"),
        ("TimeConfig", dict(damp_ends_rate=-1.0), "damp_ends_rate"),
        ("TimeConfig", dict(damp_ends_rate=float("inf")), "damp_ends_rate"),
        ("RuntimeSpeciesConfig", dict(Nm=0), "positive integer"),
        ("RuntimeSpeciesConfig", dict(hermite_closure="open"), "hermite_closure"),
        ("RuntimeOutputConfig", dict(time_interval=-1.0), "time_interval"),
        ("RuntimeRunConfig", dict(steps=0), r"\[run\] steps"),
        ("RuntimeParallelConfig", dict(strategy="mpi"), "Unknown parallel strategy"),
        ("RuntimeParallelConfig", dict(axis=" "), "nonempty"),
        ("RuntimeParallelConfig", dict(auto=True, strategy="batch"), "conflicts"),
        ("RuntimeParallelConfig", dict(auto=True, axis="kx"), "conflicts"),
        ("RuntimeParallelConfig", dict(batch_size=0), "batch_size"),
        ("RuntimeParallelConfig", dict(num_devices=0), "num_devices"),
        ("RuntimeCollisionConfig", dict(nu_hyper_l=0.5), "Laguerre hypercollision"),
    ],
)
def test_config_sections_refuse_bad_inputs(cls, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        getattr(gkx_config, cls)(**kwargs)


def test_config_normalises_spelling_and_derives_the_step_policy() -> None:
    assert gkx_config.GridConfig(ky_layout=" FULL ").ky_layout == "full"
    # No dt: the CFL controller picks the step with rk4. A chosen dt is fixed, rk2.
    assert (gkx_config.TimeConfig().method, gkx_config.TimeConfig().fixed_dt) == (
        "rk4",
        False,
    )
    chosen = gkx_config.TimeConfig(dt=0.05)
    assert (chosen.method, chosen.fixed_dt) == ("rk2", True)
    assert gkx_config.TimeConfig(dt=0.05, fixed_dt=False).method == "rk4"
    assert gkx_config.RuntimeQuasilinearConfig(channels="es").channels == ("es",)
    parallel = gkx_config.RuntimeParallelConfig(
        auto=True, strategy="serial", axis="s_m"
    )
    assert (parallel.strategy, parallel.axis) == ("shard_map", "species_hermite")
    run = gkx_config.RuntimeRunConfig(Nl="4", ky=1, steps=None)
    assert (run.Nl, run.ky, run.is_empty()) == (4, 1.0, False)
    assert gkx_config.RuntimeRunConfig().is_empty()


def test_cfl_prefactors_follow_the_stepper_and_fall_with_the_moment_count() -> None:
    for method, fac in (
        ("rk3", 1.73),
        ("lawson-rk4", 2.82),
        ("RK4-PI", 2.82),
        ("rk2", 1.0),
    ):
        assert gkx_config.explicit_method_default_cfl_fac(method) == fac
    assert gkx_config.resolve_cfl_fac("rk4", 0.3) == 0.3
    assert gkx_config.resolve_nonlinear_cfl_fac("rk4", 0.3, 999) == 0.3
    assert (
        gkx_config.resolve_nonlinear_cfl_fac("rk2", None, 999) == 1.0
    )  # no measured edge
    rk4 = [
        gkx_config.resolve_nonlinear_cfl_fac("rk4", None, m) for m in (1, 32, 40, 10**6)
    ]
    assert rk4[0] == rk4[1] > rk4[2] > rk4[3] > 0.0  # moment scaling, floored
    assert rk4[3] == pytest.approx(rk4[1] * 1.73 / 2.1)
    assert "retired" in str(gkx_config.method_error("Euler", ("rk2",)))
    assert "must be one of" in str(gkx_config.method_error("nope", ("rk2",)))


@pytest.mark.parametrize(
    "changes",
    [
        dict(source=""),
        dict(sha256="A" * 64),
        dict(route="adaptive"),
        dict(scale_by_dt="yes"),
        dict(amplitude=-1.0),
        dict(dt_input=0.0),
        dict(route="timestep_free", dt_step=0.5),
        dict(amplitude=1.0e308, dt_input=1.0e-10),
    ],
)
def test_end_damping_reference_validates_its_provenance(changes) -> None:
    ok = dict(
        source="ref",
        sha256="0" * 64,
        route="fixed_linear",
        amplitude=1.0,
        dt_input=0.5,
        dt_step=0.25,
        scale_by_dt=True,
    )
    assert gkx_config.EndDampingReference(**ok).rate == pytest.approx(8.0)
    with pytest.raises(ValueError):
        gkx_config.EndDampingReference(**{**ok, **changes})
    free = gkx_config.EndDampingReference(
        **{**ok, "route": "timestep_free", "dt_step": 0.0}
    )
    assert free.rate == pytest.approx(2.0)


def test_case_validation_collects_every_disagreement() -> None:
    from gkx.config import (
        GeometryConfig,
        RuntimeCollisionConfig,
        RuntimePhysicsConfig,
        RuntimeTermsConfig,
        TimeConfig,
    )

    bad = RuntimeConfig(
        species=(),
        physics=RuntimePhysicsConfig(
            linear=True, nonlinear=True, hypercollisions=False
        ),
        collisions=RuntimeCollisionConfig(nu_hyper_l=0.5, hypercollisions_const=1.0),
        terms=RuntimeTermsConfig(hypercollisions=0.0),
        time=TimeConfig(dt=-1.0, t_max=0.0),
        geometry=GeometryConfig(model="vmec", vmec_file=None),
    )
    with pytest.raises(ValueError, match="invalid case") as err:
        bad.validate()
    for fragment in (
        "at least one species",
        "at least one kinetic",
        "cannot both be true",
        "physics.hypercollisions = true",
        "nonzero terms.hypercollisions",
        "time.dt must be positive",
        "time.t_max must be positive",
        "vmec_file",
    ):
        assert fragment in str(err.value)


def test_deck_writer_refuses_unserialisable_values_and_skips_nulls() -> None:
    text = gkx_config.deck_text({"a": 1, "skip": None, "tab": {"x": (1, 2), "y": None}})
    assert text == "a = 1\n\n[tab]\nx = [1, 2]\n"
    with pytest.raises(TypeError, match="unsupported"):
        gkx_config.deck_text({"a": object()})


_THETA = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)


@pytest.mark.parametrize(
    ("bmag", "wells"),
    [
        (np.cos(2 * _THETA), 2),  # two full-depth wells
        (np.cos(_THETA) + 0.02 * np.cos(9 * _THETA), 1),  # ripple below 10% of range
        (np.ones(64), 1),  # no |B| range
        (np.arange(3.0), 1),  # too few points to resolve a well
        (np.array([]), 1),
        (np.arange(16.0), 1),  # monotone: the only extremum pair wraps around
    ],
)
def test_bmag_well_count_ignores_shallow_ripple_and_degenerate_traces(bmag, wells):
    assert runtime_wout._count_bmag_wells(bmag) == wells


@pytest.mark.parametrize("closed", [True, False])
def test_geometry_features_drop_the_closed_interval_endpoint(closed):
    n = 8
    geom = SimpleNamespace(
        theta_closed_interval=closed,
        gds2_profile=np.full(n, 4.0),
        gds22_profile=np.full(n, 9.0),
        bmag_profile=np.linspace(1.0, 2.0, n),
        s_hat=-1.5,
        q=1.4,
        nfp=3,
    )
    feats = runtime_wout.geometry_features(geom, zp=3.0)
    # rms |grad x| = sqrt(9)/|shat|, max |grad y| = sqrt(4)
    assert feats.anisotropy == pytest.approx(3.0 / 1.5 / 2.0)
    assert (feats.shat, feats.q, feats.nfp, feats.zp) == (-1.5, 1.4, 3, 3.0)
    assert feats.bmag_wells == 1
    assert runtime_wout.geometry_class(feats) == "stellarator"


def test_estimator_rejects_unknown_tier_and_saturates_the_ladder():
    feats = GeometryFeatures(
        anisotropy=0.1, shat=0.8, q=1.4, nfp=1, bmag_wells=1, zp=1.0
    )
    with pytest.raises(ValueError, match="target_error"):
        ky_max_target(feats, "fast")
    assert perp_points_for(1.0e-3, 4.4) == PERP_LADDER[-1]


def test_estimate_resolution_from_an_analytic_flux_tube(monkeypatch) -> None:
    from gkx.config import CycloneBaseCase
    from gkx.geometry import SAlphaGeometry

    from gkx.geometry.core import ensure_flux_tube_geometry_data

    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(CycloneBaseCase().geometry),
        np.linspace(-np.pi, np.pi, 32, endpoint=False),
    )
    monkeypatch.setattr(runtime_wout, "build_runtime_geometry", lambda cfg: geom)
    est = runtime_wout.estimate_resolution("wout_case.nc", torflux=0.5)
    assert est["torflux"] == 0.5
    assert 0.0 < est["dt"] <= 0.1
    assert est["nx"] == est["ny"] in PERP_LADDER
    table = runtime_wout.format_estimate_table(est)
    assert table.splitlines()[0].startswith("geometry: shat=")
    assert all(f"{key:>6} =" in table for key in ("nx", "nz", "nm", "dt", "t_max"))


def test_wout_flag_parsing_and_output_prefix_rules(tmp_path, monkeypatch) -> None:
    args = ["--vmex=a.nc", "run", "--vmec", "b.nc"]
    assert runtime_wout.extract_wout_flag_value(args) == "b.nc"
    assert args == ["run"]
    with pytest.raises(SystemExit, match="requires an equilibrium FILE"):
        runtime_wout.extract_wout_flag_value(["--vmec"])

    assert runtime_wout._flag_value(["--out=x", "--n"], "--out") == "x"
    assert runtime_wout._flag_value(["--out"], "--out") is None
    monkeypatch.chdir(tmp_path)
    wout = tmp_path / "wout_case.nc"
    assert runtime_wout._resolved_output_prefix({}, [], wout) == (
        tmp_path / "wout_case" / "gkx",
        False,
    )
    assert runtime_wout._resolved_output_prefix(
        {"output": {"path": "deck_out"}}, [], wout
    ) == (Path("deck_out"), True)
    prefix = Path("p")
    assert runtime_wout._resolved_output_target(
        prefix, explicit=False, linear=False
    ) == Path("p.out.nc")
    assert (
        runtime_wout._resolved_output_target(prefix, explicit=True, linear=False)
        == prefix
    )
    assert (
        runtime_wout._resolved_output_target(prefix, explicit=False, linear=True)
        == prefix
    )


def test_linear_scan_defaults_keep_a_small_step_and_a_user_scan() -> None:
    data = {"time": {"dt": 0.005}, "scan": {"ky": [0.3]}}
    runtime_wout._apply_linear_scan_defaults(data)
    assert data["time"]["dt"] == 0.005 and data["scan"]["ky"] == [0.3]
    assert data["physics"] == {"linear": True, "nonlinear": False}
    data = {"time": {"dt": 0.1}}
    runtime_wout._apply_linear_scan_defaults(data)
    assert data["time"]["dt"] == runtime_wout.LINEAR_SCAN_DT


def test_wout_file_detection_and_class_defaults_on_unreadable_files(
    tmp_path, capsys
) -> None:
    junk = tmp_path / "wout_junk.nc"
    junk.write_bytes(b"not netcdf")
    assert runtime_wout.is_wout_file(junk) is False
    assert runtime_wout.is_wout_file(tmp_path / "x.txt") is False
    assert runtime_wout._wout_nfp(junk) is None
    data: dict = {}
    runtime_wout._apply_class_resolution_defaults(data, junk)
    assert data == {} and capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("nfp", "grid", "word"), [(1, 64, "tokamak"), (5, 96, "stellarator")]
)
def test_class_defaults_follow_the_field_period_count(
    monkeypatch, capsys, nfp, grid, word
):
    monkeypatch.setattr(runtime_wout, "_wout_nfp", lambda path: nfp)
    data = {"grid": {"Nx": 96, "Ny": 96}}
    runtime_wout._apply_class_resolution_defaults(data, Path("w.nc"))
    assert data["grid"] == {"Nx": grid, "Ny": grid}
    assert word in capsys.readouterr().out


def test_default_deck_falls_back_to_examples_and_then_fails(monkeypatch) -> None:
    real = Path.is_file
    packaged = runtime_wout.resources.files("gkx").joinpath("data/common_input.toml")
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: False if str(self) == str(packaged) else real(self),
    )
    assert runtime_wout.default_wout_deck_path().parent.name == "examples"
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError, match="common_input.toml"):
        runtime_wout.default_wout_deck_path()


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


def test_nonlinear_mode_selection_respects_the_dealias_mask() -> None:
    from gkx.workflows import nonlinear as nl

    ky, kx = np.array([0.0, 0.5, 1.0]), np.array([-1.0, 0.0, 1.0])
    mask = np.array([[True, True, False], [False, False, False], [True, True, True]])
    grid = SimpleNamespace(ky=ky, kx=kx, dealias_mask=mask)
    # Row 1 keeps no kx: it is not a candidate ky, and an empty row falls back to all kx.
    np.testing.assert_array_equal(nl._active_ky_indices(mask, 3), [0, 2])
    np.testing.assert_array_equal(
        nl._active_ky_indices(np.zeros((3, 3), bool), 3), [0, 1, 2]
    )
    np.testing.assert_array_equal(nl._active_kx_indices(mask, 0, 3), [0, 1])
    np.testing.assert_array_equal(nl._active_kx_indices(mask, 1, 3), [0, 1, 2])
    pick = partial(nl._select_nonlinear_mode_indices, grid)
    assert pick(ky_target=0.5, kx_target=None, use_dealias_mask=False) == (1, 1)
    assert pick(ky_target=0.5, kx_target=None, use_dealias_mask=True) == (0, 1)
    assert pick(ky_target=1.0, kx_target=0.9, use_dealias_mask=True) == (2, 2)
    with pytest.raises(ValueError, match="dealias_mask shape"):
        nl._validate_dealias_mask_shape(mask, ky_size=3, kx_size=2)


def test_nonlinear_runaway_detector_and_run_to_validation() -> None:
    from gkx.config import TimeConfig
    from gkx.workflows import nonlinear as nl

    assert nl._chunk_runaway(np.array([1.0, np.nan]))
    assert nl._chunk_runaway(np.array([1.0, 5.0e4]))
    assert not nl._chunk_runaway(np.array([1.0, 9.0e2]))  # below the energy floor
    assert not nl._chunk_runaway(np.array([0.0, 5.0e4]))  # zero start is no baseline
    assert not nl._chunk_runaway(np.array([]))
    cfg = RuntimeConfig(time=TimeConfig(run_to="forever"))
    with pytest.raises(ValueError, match="unknown .time. run_to"):
        nl._saturation_stop_condition(cfg, None, None)
    with pytest.raises(ValueError, match="nonlinear physics"):
        nl.prepare(RuntimeConfig())  # the default case is linear


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


def test_runtime_step_and_external_phi_policies_are_fail_closed() -> None:
    def time_cfg(fixed_dt: bool, dt: float, dt_max) -> SimpleNamespace:
        return SimpleNamespace(
            time=SimpleNamespace(fixed_dt=fixed_dt, t_max=1.0, dt=dt, dt_max=dt_max)
        )

    fixed = time_cfg(True, 0.25, None)
    for cfg, dt, steps, expected in (
        (fixed, 0.125, None, 8),  # t_max over the dt actually used
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


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_runtime_chunk_steps_requires_positive_integer(value):
    from gkx.config import TimeConfig

    with pytest.raises(ValueError, match="chunk_steps"):
        TimeConfig(chunk_steps=value)


@pytest.mark.parametrize("value", [True, 0.0, -0.1, float("nan"), float("inf")])
def test_runtime_physical_diagnostic_interval_requires_positive_finite(value):
    from gkx.config import TimeConfig

    with pytest.raises(ValueError, match="diagnostics_interval"):
        TimeConfig(diagnostics_interval=value)


def test_runtime_time_cadence_loads_through_existing_time_section(tmp_path):
    path = tmp_path / "case.toml"
    path.write_text(
        "schema_version = 1\n[time]\nchunk_steps = 7\ndiagnostics_interval = 0.25\n"
    )
    cfg, _ = load_runtime_from_toml(path)
    assert cfg.time.chunk_steps == 7
    assert cfg.time.diagnostics_interval == 0.25
    assert cfg.to_dict()["time"]["diagnostics_interval"] == 0.25


@pytest.mark.parametrize("value", [True, 0.0, -0.1, float("nan"), float("inf")])
def test_output_physical_interval_requires_positive_finite(value):
    from gkx.config import RuntimeOutputConfig

    with pytest.raises(ValueError, match="time_interval"):
        RuntimeOutputConfig(time_interval=value)


@pytest.mark.parametrize(("method", "fixed_dt"), [("rk2", True), ("rk4-pi", False)])
def test_runtime_chunk_policy_preserves_state_and_fresh_physical_samples(
    method, fixed_dt
):
    from gkx.workflows.nonlinear import run_runtime_nonlinear, prepare

    cfg, _ = load_runtime_from_toml(
        REPO_ROOT / "examples/03_nonlinear_tokamak/case.toml"
    )
    cfg = replace(
        cfg,
        grid=replace(cfg.grid, Nx=4, Ny=4, Nz=4),
        time=replace(
            cfg.time,
            dt=0.001,
            dt_max=0.001,
            t_max=0.008,
            method=method,
            fixed_dt=fixed_dt,
            run_to="t_max",
            sample_stride=1,
            diagnostics_stride=1,
        ),
    )
    options = dict(Nl=2, Nm=4, steps=8, resolved_diagnostics=False, return_state=True)
    dense = run_runtime_nonlinear(cfg, **options)
    for chunk_steps in (1, 3):
        case = replace(
            cfg,
            time=replace(
                cfg.time, chunk_steps=chunk_steps, diagnostics_interval=0.0025
            ),
        )
        selected = run_runtime_nonlinear(case, **options)
        np.testing.assert_allclose(selected.state, dense.state, rtol=1e-12, atol=1e-15)
        indices = np.searchsorted(dense.t, selected.t - 1e-14)
        np.testing.assert_allclose(
            selected.diagnostics.heat_flux_t,
            dense.diagnostics.heat_flux_t[indices],
            rtol=1e-12,
            atol=1e-15,
        )
        np.testing.assert_allclose(selected.t, [0.003, 0.005, 0.008], atol=1e-14)
    case = replace(cfg, time=replace(cfg.time, diagnostics_interval=0.0025))
    backend = prepare(case, Nl=2, Nm=4, steps=8, resolved_diagnostics=False)
    np.testing.assert_allclose(backend.run()[0], [0.003, 0.005, 0.008], atol=1e-14)
    if fixed_dt:
        arrays = backend.run_arrays()
        np.testing.assert_allclose(arrays[1][1], [0.003, 0.005, 0.008], atol=1e-14)
        import jax
        import jax.numpy as jnp

        def loss(state):
            final = backend.run_arrays(state)[0]
            return jnp.real(jnp.vdot(final, final))

        grad = jax.jit(jax.grad(loss))(backend.initial_state)
        dense_backend = replace(backend, _retention_interval=None)

        def dense_loss(state):
            final = dense_backend.run_arrays(state)[0]
            return jnp.real(jnp.vdot(final, final))

        reference_grad = jax.jit(jax.grad(dense_loss))(backend.initial_state)
        np.testing.assert_allclose(grad, reference_grad, rtol=1e-12, atol=1e-15)
    else:
        with pytest.raises(ValueError, match="fixed-shape"):
            backend.run_arrays()
    import jax

    with pytest.raises(ValueError, match="host run"):
        jax.jit(lambda state: backend.run(initial_state=state)[2])(
            backend.initial_state
        )
    with pytest.raises(ValueError, match="chunk_steps requires host"):
        prepare(
            replace(case, time=replace(case.time, chunk_steps=3)), Nl=2, Nm=4, steps=8
        )


def _vmec_deck(tmp_path, grid_extra="", geometry_extra="npol = 1.0"):
    deck = tmp_path / "deck" / "case.toml"
    deck.parent.mkdir(parents=True, exist_ok=True)
    deck.write_text(
        f'[grid]\n{grid_extra}\n[geometry]\nmodel = "vmec"\nvmec_file = "w.nc"\n'
        f"torflux = 0.5\n{geometry_extra}\n"
    )
    return deck


def test_relative_vmec_file_override_resolves_against_deck_directory(tmp_path):
    from gkx.config import GeometryConfig
    from gkx.workflows.runtime.toml import load

    case = load(_vmec_deck(tmp_path))
    deck_dir = (tmp_path / "deck").resolve()
    assert case.geometry.vmec_file == str(deck_dir / "w.nc")
    other = case.replace(geometry=replace(case.geometry, vmec_file="sub/x.nc"))
    assert other.geometry.vmec_file == str(deck_dir / "sub" / "x.nc")
    absolute = str((tmp_path / "abs.nc").resolve())
    kept = case.replace(geometry=replace(case.geometry, vmec_file=absolute))
    assert kept.geometry.vmec_file == absolute
    assert isinstance(case.geometry, GeometryConfig)


def test_vmec_nperiod_conflicting_with_npol_is_rejected(tmp_path):
    from gkx.workflows.runtime.toml import load

    case = load(_vmec_deck(tmp_path, grid_extra="nperiod = 4"))
    with pytest.raises(ValueError, match="geometry.npol"):
        case.validate()
    ok = load(_vmec_deck(tmp_path / "x", grid_extra="nperiod = 1"))
    with pytest.warns(RuntimeWarning, match="short field line"):
        ok.validate()


def test_short_field_line_warns_only_below_threshold():
    import warnings

    from gkx.config import GridConfig, RuntimeConfig

    with pytest.warns(RuntimeWarning, match="nperiod = 1"):
        RuntimeConfig(grid=GridConfig(nperiod=1)).validate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RuntimeConfig(grid=GridConfig(nperiod=2)).validate()


def _lenard_bernstein_runtime(nus, nu_hermite, nu_laguerre):
    from gkx.workflows.runtime.startup import (
        build_runtime_geometry,
        build_runtime_linear_params,
    )

    cfg, _ = load_runtime_from_toml(REPO_ROOT / "examples/07_collisions/case.toml")
    ion = cfg.species[0]
    species = tuple(replace(ion, nu=nu) for nu in nus)
    cfg = replace(
        cfg,
        species=species,
        time=replace(cfg.time, collision_operator="lenard_bernstein"),
        collisions=replace(
            cfg.collisions, nu_hermite=nu_hermite, nu_laguerre=nu_laguerre
        ),
    )
    return build_runtime_linear_params(cfg, Nm=4, geom=build_runtime_geometry(cfg))


@pytest.mark.parametrize(
    ("nu_hermite", "nu_laguerre"),
    [(0.0, 0.0), (-1.0, 2.0), (1.0, -2.0), (float("nan"), 2.0), (1.0, float("inf"))],
)
def test_lenard_bernstein_refuses_missing_or_invalid_damping(nu_hermite, nu_laguerre):
    with pytest.raises(ValueError, match="Lenard-Bernstein"):
        _lenard_bernstein_runtime((0.05,), nu_hermite, nu_laguerre)


def test_lenard_bernstein_refusal_covers_every_species():
    # The first species has nu = 0; the second one still collides.
    with pytest.raises(ValueError, match="Lenard-Bernstein"):
        _lenard_bernstein_runtime((0.0, 0.05), 0.0, 0.0)
    # No species collides, so there is nothing to refuse.
    params = _lenard_bernstein_runtime((0.0, 0.0), 0.0, 0.0)
    assert params.collision_model == "dougherty"
    # Valid damping passes.
    assert _lenard_bernstein_runtime((0.0, 0.05), 1.0, 2.0).nu_hermite == 1.0
