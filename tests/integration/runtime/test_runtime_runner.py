"""Tests for unified runtime-configured linear runner."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from support.paths import REPO_ROOT
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import gkx.workflows.runtime.startup as startup
from gkx.diagnostics.analysis import late_time_linear_metrics
from gkx.config import (
    CycloneBaseCase,
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from gkx.diagnostics import SimulationDiagnostics, ResolvedDiagnostics
from gkx.diagnostics.modes import ModeSelection
from gkx.geometry import (
    SAlphaGeometry,
    apply_geometry_grid_defaults,
    sample_flux_tube_geometry,
)
from gkx.core.grid import build_spectral_grid
from gkx.workflows.runtime.toml import load_runtime_from_toml
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.runtime import (
    _build_initial_condition,
    _centered_glibc_random_pairs,
    _dealiased_initial_mode_pairs,
    _periodic_zp_from_grid,
    _infer_runtime_nonlinear_steps,
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_linear_terms,
    run_runtime_linear,
    run_runtime_nonlinear,
    run_runtime_scan,
)
from gkx.workflows.runtime.commands import run_linear_case, run_nonlinear_case
from gkx.workflows.runtime.config import (
    RuntimeCollisionConfig,
    RuntimeConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimeParallelConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)
from gkx.workflows.runtime.orchestration_scan import run_runtime_parameter_scan
from gkx.workflows.runtime.diagnostics import refit_runtime_linear_trajectory
from gkx.workflows.runtime.results import RuntimeParameterScanResult
from gkx.terms.assembly import compute_fields_cached
from gkx.terms.config import FieldState
from support.paired_solvax import requires_paired_solvax

pytestmark = pytest.mark.integration


def _base_runtime_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        time=TimeConfig(
            t_max=0.2, dt=0.01, method="rk2", use_diffrax=False, sample_stride=1
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
    )


def _write_root_eik_geometry(
    path: Path,
    sampled,
    dataset_type,
    *,
    drift_scale: float = 1.0,
    jacobian: float | None = None,
    drhodpsi: float | None = None,
    nfp: float | None = None,
) -> None:
    theta = np.asarray(sampled.theta)
    profiles = {
        "bmag": sampled.bmag_profile,
        "gds2": sampled.gds2_profile,
        "gds21": sampled.gds21_profile,
        "gds22": sampled.gds22_profile,
        "cvdrift": drift_scale * np.asarray(sampled.cv_profile),
        "gbdrift": drift_scale * np.asarray(sampled.gb_profile),
        "cvdrift0": drift_scale * np.asarray(sampled.cv0_profile),
        "gbdrift0": drift_scale * np.asarray(sampled.gb0_profile),
        "jacob": (
            sampled.jacobian_profile
            if jacobian is None
            else np.full(theta.size, jacobian)
        ),
        "grho": sampled.grho_profile,
        "gradpar": np.full(theta.size, sampled.gradpar_value),
    }
    scalars = {
        "q": sampled.q,
        "shat": sampled.s_hat,
        "Rmaj": sampled.R0,
        "kxfac": sampled.kxfac,
        "scale": sampled.theta_scale,
        "nfp": sampled.nfp if nfp is None else nfp,
        "alpha": sampled.alpha,
    }
    if drhodpsi is not None:
        scalars["drhodpsi"] = drhodpsi

    with dataset_type(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        for name, values in profiles.items():
            root.createVariable(name, "f8", ("z",))[:] = np.asarray(values)
        for name, value in scalars.items():
            root.createVariable(name, "f8", ())[:] = value


def _glibc_random_pairs_reference(seed: int, count: int) -> np.ndarray:
    rand_max = float((1 << 31) - 1)
    seed_use = 1 if int(seed) == 0 else int(seed)
    state = np.zeros(344 + 2 * count, dtype=np.uint64)
    state[0] = np.uint64(seed_use)
    for i in range(1, 31):
        state[i] = np.uint64((16807 * int(state[i - 1])) % int(rand_max))
    for i in range(31, 34):
        state[i] = state[i - 31]
    for i in range(34, state.size):
        state[i] = (state[i - 31] + state[i - 3]) & np.uint64(0xFFFFFFFF)
    rand_vals = (state[344:] >> np.uint64(1)).astype(float, copy=False)
    half = 0.5 * rand_max
    inv = 1.0 / rand_max
    out = np.empty((count, 2), dtype=float)
    for i in range(count):
        out[i, 0] = (rand_vals[2 * i] - half) * inv
        out[i, 1] = (rand_vals[2 * i + 1] - half) * inv
    return out


def test_runtime_linear_cyclone_etg_kbm_time_smoke() -> None:
    ion = RuntimeSpeciesConfig(
        name="ion",
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )
    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )

    cyclone = replace(
        _base_runtime_cfg(),
        species=(ion,),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    etg = replace(
        _base_runtime_cfg(),
        species=(electron,),
        normalization=RuntimeNormalizationConfig(
            contract="etg", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )
    kbm = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=62.8, Ly=62.8, boundary="periodic"),
        init=InitializationConfig(
            init_field="all", init_amp=1.0e-8, gaussian_init=False
        ),
        species=(ion, electron),
        normalization=RuntimeNormalizationConfig(
            contract="kbm", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            beta=0.2,
            hypercollisions=False,
        ),
    )

    for cfg, ky in ((cyclone, 0.2), (etg, 2.0), (kbm, 0.2)):
        res = run_runtime_linear(
            cfg,
            ky_target=ky,
            Nl=4,
            Nm=6,
            solver="time",
        )
        assert np.isfinite(res.gamma)
        assert np.isfinite(res.omega)


def test_runtime_linear_etg_defaults_to_frequency_targeted_krylov(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=2.49,
        fprim=0.8,
    )
    cfg = replace(
        _base_runtime_cfg(),
        species=(electron,),
        normalization=RuntimeNormalizationConfig(
            contract="etg", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            adiabatic_ions=True,
            electrostatic=True,
            electromagnetic=False,
        ),
    )

    captured: dict[str, object] = {}

    def fake_dominant_eigenpair(_g0, _cache, _params, terms=None, **kwargs):
        captured["terms"] = terms
        captured.update(kwargs)
        return np.asarray(0.2 + 0.3j, dtype=np.complex64), np.zeros_like(
            np.asarray(_g0)
        )

    monkeypatch.setattr(runtime, "dominant_eigenpair", fake_dominant_eigenpair)

    out = run_runtime_linear(cfg, ky_target=2.0, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert captured["method"] == "shift_invert"
    assert captured["omega_target_factor"] == pytest.approx(0.4)
    assert captured["omega_sign"] == -1
    assert captured["shift_source"] == "target"
    assert captured["shift_selection"] == "targeted"
    assert captured["mode_family"] == "etg"


def test_runtime_terms_and_params_follow_toggles() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="rho_star"
        ),
        physics=RuntimePhysicsConfig(
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            use_bpar=False,
            adiabatic_electrons=True,
            tau_e=1.0,
            beta=0.1,
            collisions=False,
            hypercollisions=False,
        ),
    )
    params = build_runtime_linear_params(cfg, Nm=6)
    terms = build_runtime_linear_terms(cfg)
    assert float(params.beta) == 0.1
    assert float(params.fapar) == 1.0
    assert terms.apar == 1.0
    assert terms.bpar == 0.0
    assert terms.collisions == 0.0
    assert terms.hypercollisions == 0.0


def test_runtime_terms_disable_collisions_when_species_nu_is_zero() -> None:
    cfg_zero = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion", nu=0.0),),
        physics=RuntimePhysicsConfig(collisions=True, hypercollisions=False),
    )
    cfg_nonzero = replace(
        cfg_zero,
        species=(RuntimeSpeciesConfig(name="ion", nu=0.05),),
    )

    assert build_runtime_linear_terms(cfg_zero).collisions == 0.0
    assert build_runtime_linear_terms(cfg_nonzero).collisions == 1.0


def test_runtime_hypercollision_default_tracks_hermite_count() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        collisions=RuntimeCollisionConfig(),
    )
    params_nm8 = build_runtime_linear_params(cfg, Nm=8)
    params_nm16 = build_runtime_linear_params(cfg, Nm=16)
    assert float(params_nm8.p_hyper_m) == 4.0
    assert float(params_nm16.p_hyper_m) == 8.0


def test_runtime_end_damping_defaults_use_unscaled_reference_rate() -> None:
    base = _base_runtime_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.2),
        collisions=RuntimeCollisionConfig(),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.damp_ends_amp) == pytest.approx(0.1)
    assert float(params.damp_ends_widthfrac) == pytest.approx(0.125)


def test_runtime_end_damping_can_explicitly_scale_by_dt() -> None:
    base = _base_runtime_cfg()
    cfg = replace(
        base,
        time=replace(base.time, dt=0.2),
        collisions=RuntimeCollisionConfig(damp_ends_scale_by_dt=True),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.damp_ends_amp) == pytest.approx(0.5)
    assert float(params.damp_ends_widthfrac) == pytest.approx(0.125)


def test_runtime_startup_model_and_geometry_branches(
    monkeypatch, tmp_path: Path
) -> None:
    base = _base_runtime_cfg()
    captured = []

    def fake_build_flux_tube_geometry(geom_cfg):
        captured.append(geom_cfg)
        return SimpleNamespace(s_hat=0.8, gradpar=lambda: 1.0)

    monkeypatch.setattr(
        startup, "build_flux_tube_geometry", fake_build_flux_tube_geometry
    )
    monkeypatch.setattr(
        startup, "generate_runtime_vmec_eik", lambda _cfg: tmp_path / "vmec.eik.nc"
    )
    monkeypatch.setattr(
        startup, "generate_runtime_miller_eik", lambda _cfg: tmp_path / "miller.eik.nc"
    )

    vmec_cfg = replace(base, geometry=replace(base.geometry, model="vmec"))
    miller_cfg = replace(base, geometry=replace(base.geometry, model="miller"))
    default_cfg = replace(base, geometry=replace(base.geometry, model="s-alpha"))

    vmec_geom_cfg = startup.runtime_geometry_config_for_builder(
        vmec_cfg,
        vmec_eik_builder=startup.generate_runtime_vmec_eik,
        miller_eik_builder=startup.generate_runtime_miller_eik,
    )
    miller_geom_cfg = startup.runtime_geometry_config_for_builder(
        miller_cfg,
        vmec_eik_builder=startup.generate_runtime_vmec_eik,
        miller_eik_builder=startup.generate_runtime_miller_eik,
    )
    default_geom_cfg = startup.runtime_geometry_config_for_builder(
        default_cfg,
        vmec_eik_builder=startup.generate_runtime_vmec_eik,
        miller_eik_builder=startup.generate_runtime_miller_eik,
    )

    assert vmec_geom_cfg.model == "vmec-eik"
    assert vmec_geom_cfg.geometry_file.endswith("vmec.eik.nc")
    assert miller_geom_cfg.model == "imported-eik"
    assert miller_geom_cfg.geometry_file.endswith("miller.eik.nc")
    assert default_geom_cfg is default_cfg.geometry

    assert startup.build_runtime_geometry(vmec_cfg).s_hat == pytest.approx(0.8)
    assert captured[-1].model == "vmec-eik"
    assert captured[-1].geometry_file.endswith("vmec.eik.nc")
    assert startup.build_runtime_geometry(miller_cfg).s_hat == pytest.approx(0.8)
    assert captured[-1].model == "imported-eik"
    assert captured[-1].geometry_file.endswith("miller.eik.nc")
    assert startup.build_runtime_geometry(default_cfg).s_hat == pytest.approx(0.8)
    assert captured[-1].model == "s-alpha"


def test_runtime_startup_reduced_model_and_species_validation(monkeypatch) -> None:
    no_kinetic = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="adiabatic", kinetic=False),),
    )
    with pytest.raises(ValueError, match="kinetic species"):
        startup._species_to_linear(no_kinetic.species)

    unknown = replace(
        _base_runtime_cfg(), physics=RuntimePhysicsConfig(reduced_model="not-a-model")
    )
    with pytest.raises(ValueError, match="Unknown physics.reduced_model"):
        startup._resolve_runtime_hl_dims(unknown, Nl=None, Nm=None)
    with pytest.raises(ValueError, match="Unknown physics.reduced_model"):
        startup._require_full_gk_runtime_model(unknown)

    fake_geom = SimpleNamespace(s_hat=0.8, gradpar=lambda: 1.25)
    monkeypatch.setattr(startup, "build_runtime_geometry", lambda _cfg: fake_geom)
    ion_cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion", charge=1.0, kinetic=True),),
    )
    params = startup.build_runtime_linear_params(ion_cfg, Nm=2)
    assert float(params.kpar_scale) == pytest.approx(1.25)

    conflict = replace(
        _base_runtime_cfg(),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        species=(
            RuntimeSpeciesConfig(name="ion", charge=1.0, kinetic=True),
            RuntimeSpeciesConfig(name="electron", charge=-1.0, kinetic=True),
        ),
    )
    with pytest.raises(ValueError, match="adiabatic_electrons"):
        startup.build_runtime_linear_params(conflict, Nm=2, geom=fake_geom)

    electron_cfg = replace(
        _base_runtime_cfg(),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False, adiabatic_ions=True, tau_e=1.25
        ),
        species=(RuntimeSpeciesConfig(name="electron", charge=-1.0, kinetic=True),),
    )
    electron_params = startup.build_runtime_linear_params(
        electron_cfg, Nm=2, geom=fake_geom
    )
    assert float(electron_params.tau_e) == pytest.approx(1.25)

    kinetic_ion_conflict = replace(
        electron_cfg,
        species=(RuntimeSpeciesConfig(name="ion", charge=1.0, kinetic=True),),
    )
    with pytest.raises(ValueError, match="adiabatic_ions"):
        startup.build_runtime_linear_params(kinetic_ion_conflict, Nm=2, geom=fake_geom)

    double_adiabatic = replace(
        electron_cfg,
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=True, adiabatic_ions=True, tau_e=1.0
        ),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        startup.build_runtime_linear_params(double_adiabatic, Nm=2, geom=fake_geom)


def test_runtime_hypercollision_explicit_override_is_preserved() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        collisions=RuntimeCollisionConfig(p_hyper_m=11.0),
    )
    params = build_runtime_linear_params(cfg, Nm=8)
    assert float(params.p_hyper_m) == 11.0


def test_runtime_scan_returns_arrays() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    scan = run_runtime_scan(
        cfg,
        ky_values=[0.1, 0.2],
        Nl=4,
        Nm=6,
        solver="krylov",
    )
    assert scan.ky.shape == (2,)
    assert scan.gamma.shape == (2,)
    assert scan.omega.shape == (2,)


@requires_paired_solvax("adaptive_eigenpair", "estimate_rk4_timestep")
def test_runtime_linear_cyclone_krylov_matches_time_solver_growth() -> None:
    """solver="krylov" must recover the ITG branch the time solver finds.

    Physics-anchored regression gate for the collisional cyclone example: the
    generic runtime Krylov default used to run an ungated propagator Arnoldi
    whose returned pair was unconverged (relative residual ~1) and silently
    reported damped, unphysical growth rates while ``solver="time"`` found the
    unstable ITG mode. The certified adaptive default must recover the exact
    dominant eigenvalue of the linear operator (dense ground truth) and agree
    with the time-solver branch, or raise from its residual gate -- a silently
    wrong eigenvalue is the defect.
    """

    from gkx.operators.linear.params import linear_terms_to_term_config
    from gkx.solvers.linear.krylov_algorithms import _apply_operator
    from gkx.core.grid import select_ky_grid
    from gkx.diagnostics.modes import select_ky_index

    runtime, _raw = load_runtime_from_toml(
        REPO_ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    runtime = replace(
        runtime,
        grid=replace(runtime.grid, Ny=8, ntheta=16, nperiod=1, Nz=16),
    )
    reference = run_runtime_linear(
        runtime, ky_target=0.3, Nl=8, Nm=8, solver="time", steps=20_000
    )
    krylov = run_runtime_linear(runtime, ky_target=0.3, Nl=8, Nm=8, solver="krylov")
    assert reference.ky == pytest.approx(krylov.ky)

    # Operator ground truth: the certified pair must be the exact dominant
    # eigenvalue of the discretized linear operator at this resolution.
    geom = build_runtime_geometry(runtime)
    grid_full = build_spectral_grid(apply_geometry_grid_defaults(geom, runtime.grid))
    grid = select_ky_grid(
        grid_full, select_ky_index(np.asarray(grid_full.ky), 0.3)
    )
    params = build_runtime_linear_params(runtime, Nm=8, geom=geom)
    term_cfg = linear_terms_to_term_config(build_runtime_linear_terms(runtime))
    cache = build_linear_cache(grid, geom, params, 8, 8)
    shape = (1, 8, 8, grid.ky.size, grid.kx.size, grid.z.size)
    size = int(np.prod(shape))
    identity = np.eye(size, dtype=np.complex64)
    columns = [
        np.asarray(
            _apply_operator(
                jnp.asarray(identity[j].reshape(shape)),
                cache,
                params,
                term_cfg,
            )
        ).reshape(-1)
        for j in range(size)
    ]
    eigenvalues = np.linalg.eigvals(np.stack(columns, axis=1).astype(np.complex128))
    dominant = eigenvalues[np.argmax(eigenvalues.real)]
    assert krylov.gamma == pytest.approx(float(dominant.real), abs=2.0e-3)
    assert krylov.omega == pytest.approx(float(-dominant.imag), abs=2.0e-3)

    # Same unstable branch as the time solver: positive growth, matching
    # propagation direction and frequency. The time-solver growth rate carries
    # a known fit/integration bias at this resolution, so gate it loosely; the
    # defect regressed here reported gamma < 0 and an unrelated frequency.
    assert reference.gamma > 0.0
    assert krylov.gamma > 0.0
    assert np.sign(krylov.omega) == np.sign(reference.omega)
    assert krylov.omega == pytest.approx(reference.omega, rel=0.15)
    assert krylov.gamma == pytest.approx(reference.gamma, rel=0.75)


def test_runtime_scan_batch_matches_serial() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    full_grid = build_spectral_grid(cfg.grid)
    # The second mode lies outside the nonlinear two-thirds mask. Linear batch
    # scans must retain it and match independent single-mode integrations.
    ky_vals = [float(full_grid.ky[1]), float(full_grid.ky[2])]
    serial = run_runtime_scan(
        cfg,
        ky_values=ky_vals,
        Nl=4,
        Nm=6,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
    )
    batched = run_runtime_scan(
        cfg,
        ky_values=ky_vals,
        Nl=4,
        Nm=6,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
        batch_ky=True,
    )
    assert np.allclose(serial.gamma, batched.gamma, rtol=5.0e-2, atol=1.0e-8)
    assert np.allclose(serial.omega, batched.omega, rtol=5.0e-2, atol=1.0e-8)


def test_runtime_linear_time_solver_can_return_state() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="sspx3",
        dt=0.01,
        steps=2,
        return_state=True,
    )
    assert res.state is not None
    assert res.state.ndim == 6
    assert res.state.shape[:3] == (1, 3, 4)
    assert res.state.shape[-1] == cfg.grid.Nz


def test_runtime_linear_records_fit_window_metadata() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        fit_signal="phi",
        auto_window=False,
        tmin=0.02,
        tmax=0.08,
    )
    assert res.fit_signal_used == "phi"
    assert res.fit_window_tmin == pytest.approx(0.02)
    assert res.fit_window_tmax == pytest.approx(0.08)


def test_runtime_linear_progress_with_sample_stride_gt_one() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        time=replace(_base_runtime_cfg().time, sample_stride=2),
    )
    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        method="rk2",
        dt=0.01,
        steps=10,
        sample_stride=2,
        show_progress=True,
    )
    assert np.isfinite(res.gamma)
    assert np.isfinite(res.omega)


def test_runtime_linear_explicit_time_rejects_return_state() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    with pytest.raises(ValueError, match="return_state"):
        run_runtime_linear(
            cfg,
            ky_target=0.1,
            Nl=3,
            Nm=4,
            solver="explicit_time",
            method="rk4",
            dt=0.01,
            steps=2,
            return_state=True,
        )


def test_runtime_linear_rejects_invalid_fit_signal() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    with pytest.raises(ValueError, match="fit_signal"):
        run_runtime_linear(
            cfg,
            ky_target=0.1,
            Nl=3,
            Nm=4,
            solver="time",
            fit_signal="bad-signal",
            dt=0.01,
            steps=2,
        )


def test_runtime_linear_auto_fit_prefers_density_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        runtime,
        "integrate_linear_diagnostics",
        lambda *args, **kwargs: (
            np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64),
            np.ones((3, 1, 1, grid.z.size), dtype=np.complex64),
            3.0 * np.ones((3, 1, 1, grid.z.size), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda arr, sel, method="project": (
            np.asarray([1.0, 1.1, 1.2], dtype=np.complex128)
            if np.max(np.abs(arr)) < 2.0
            else np.asarray([1.0, 2.0, 4.0], dtype=np.complex128)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            (0.05, -0.02, 0.01, 0.03, 1.0, 0.0)
            if np.max(np.abs(signal)) < 2.0
            else (0.2, -0.08, 0.01, 0.03, 2.0, 0.0)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_eigenfunction",
        lambda *args, **kwargs: np.ones(grid.z.size, dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="auto",
        dt=0.01,
        steps=3,
    )

    assert res.fit_signal_used == "density"
    assert np.allclose(res.signal, np.array([1.0, 2.0, 4.0]))
    assert res.fit_window_tmin == pytest.approx(0.01)
    assert res.fit_window_tmax == pytest.approx(0.03)


def test_runtime_linear_diffrax_project_mode_keeps_full_field_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg0 = _base_runtime_cfg()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=True, sample_stride=1, dt=0.01, t_max=0.03),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_integrate_linear_from_config(*args, **kwargs):
        assert kwargs["save_mode"] is None
        phi_t = np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
        return np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64), phi_t

    monkeypatch.setattr(
        runtime, "integrate_linear_from_config", _fake_integrate_linear_from_config
    )
    gamma_ref = 0.25
    omega_ref = -0.12
    t_saved = np.asarray([0.01, 0.02, 0.03], dtype=float)
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda phi_t, sel, method="project": np.exp(
            (gamma_ref - 1j * omega_ref) * t_saved
        ).astype(np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "extract_eigenfunction",
        lambda *args, **kwargs: np.ones(grid.z.size, dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="phi",
        mode_method="project",
    )

    assert res.fit_signal_used == "phi"
    metrics = late_time_linear_metrics(res, tail_fraction=2.0 / 3.0)
    assert res.gamma == pytest.approx(gamma_ref, rel=1.0e-3)
    assert res.omega == pytest.approx(omega_ref, rel=1.0e-3)
    assert metrics.gamma_fit == pytest.approx(gamma_ref, rel=1.0e-3)
    assert metrics.omega_fit == pytest.approx(omega_ref, rel=1.0e-3)


def test_runtime_linear_forwards_velocity_parallel_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg0 = _base_runtime_cfg()
    cfg = replace(
        cfg0,
        time=replace(
            cfg0.time, use_diffrax=False, sample_stride=1, dt=0.01, t_max=0.03
        ),
        parallel=RuntimeParallelConfig(
            strategy="velocity", axis="hermite", backend="auto", num_devices=1
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_integrate_linear_from_config(*args, **kwargs):
        captured["parallel"] = kwargs["parallel"]
        phi_t = np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
        return np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64), phi_t

    monkeypatch.setattr(
        runtime, "integrate_linear_from_config", _fake_integrate_linear_from_config
    )
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda *args, **kwargs: np.asarray([1.0, 1.1, 1.2], dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "fit_growth_rate_auto",
        lambda *args, **kwargs: (0.05, -0.02, 0.01, 0.03),
    )
    monkeypatch.setattr(
        runtime,
        "extract_eigenfunction",
        lambda *args, **kwargs: np.ones(grid.z.size, dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="phi",
        mode_method="project",
    )

    assert res.fit_signal_used == "phi"
    assert captured["parallel"] is cfg.parallel

    with pytest.raises(NotImplementedError, match="fit_signal='phi'"):
        run_runtime_linear(
            cfg,
            ky_target=0.1,
            Nl=3,
            Nm=4,
            solver="time",
            fit_signal="auto",
        )


def test_runtime_linear_diffrax_auto_fit_with_density_keeps_full_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg0 = _base_runtime_cfg()
    cfg = replace(
        cfg0,
        time=replace(cfg0.time, use_diffrax=True, sample_stride=1, dt=0.01, t_max=0.03),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )

    def _fake_integrate_linear_from_config(*args, **kwargs):
        assert kwargs["save_mode"] is None
        assert kwargs["save_field"] == "phi+density"
        phi_t = np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
        density_t = 3.0 * np.ones((3, 1, 1, grid.z.size), dtype=np.complex64)
        return np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64), (
            phi_t,
            density_t,
        )

    monkeypatch.setattr(
        runtime, "integrate_linear_from_config", _fake_integrate_linear_from_config
    )
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda arr, sel, method="project": (
            np.asarray([1.0, 1.1, 1.2], dtype=np.complex128)
            if np.max(np.abs(arr)) < 2.0
            else np.asarray([1.0, 2.0, 4.0], dtype=np.complex128)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "fit_growth_rate_auto_with_stats",
        lambda t, signal, **kwargs: (
            (0.05, -0.02, 0.01, 0.03, 1.0, 0.0)
            if np.max(np.abs(signal)) < 2.0
            else (0.2, -0.08, 0.01, 0.03, 2.0, 0.0)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_eigenfunction",
        lambda *args, **kwargs: np.ones(grid.z.size, dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="time",
        fit_signal="auto",
        mode_method="project",
    )

    assert res.fit_signal_used == "density"
    assert np.allclose(res.signal, np.array([1.0, 2.0, 4.0]))


def test_runtime_linear_auto_solver_falls_back_to_krylov(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    status: list[str] = []

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros(
            (1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64
        ),
    )
    monkeypatch.setattr(
        runtime,
        "integrate_linear_from_config",
        lambda *args, **kwargs: (
            np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64),
            np.ones((3, 1, 1, grid.z.size), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "extract_mode_time_series",
        lambda *args, **kwargs: np.asarray([1.0, 0.9, 0.8], dtype=np.complex128),
    )
    monkeypatch.setattr(
        runtime, "fit_growth_rate_auto", lambda *args, **kwargs: (-0.1, 0.2, 0.01, 0.03)
    )
    monkeypatch.setattr(
        runtime, "build_linear_cache", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        runtime,
        "dominant_eigenpair",
        lambda *args, **kwargs: (
            np.asarray(0.3 + 0.4j, dtype=np.complex64),
            np.zeros((1, 3, 4, 1, 1, grid.z.size), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "apply_diagnostic_normalization",
        lambda gamma, omega, **kwargs: (gamma, omega),
    )

    res = run_runtime_linear(
        cfg,
        ky_target=0.1,
        Nl=3,
        Nm=4,
        solver="auto",
        fit_signal="phi",
        dt=0.01,
        steps=3,
        status_callback=status.append,
    )

    assert res.gamma == pytest.approx(0.3)
    assert res.omega == pytest.approx(-0.4)
    assert any("falling back to Krylov solve" in msg for msg in status)


def test_runtime_nonlinear_smoke() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(
        cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.01, steps=3, sample_stride=1
    )
    assert res.diagnostics is not None
    assert res.diagnostics.t.size == 3


def test_runtime_nonlinear_diagnostics_stride() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=5,
        sample_stride=1,
        diagnostics_stride=2,
    )
    assert res.diagnostics is not None
    assert res.diagnostics.t.size == 3


def test_runtime_nonlinear_sampled_diagnostics_match_full_scan() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    full = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=2,
        Nm=3,
        dt=0.01,
        steps=5,
        sample_stride=1,
        diagnostics_stride=1,
    )
    sampled = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=2,
        Nm=3,
        dt=0.01,
        steps=5,
        sample_stride=2,
        diagnostics_stride=2,
    )

    assert full.diagnostics is not None
    assert sampled.diagnostics is not None
    sample_idx = np.asarray([0, 2, 4], dtype=int)
    for name in ("t", "dt_t", "gamma_t", "omega_t", "Wg_t", "Wphi_t", "heat_flux_t"):
        np.testing.assert_allclose(
            np.asarray(getattr(sampled.diagnostics, name)),
            np.asarray(getattr(full.diagnostics, name))[sample_idx],
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    assert full.diagnostics.resolved is not None
    assert sampled.diagnostics.resolved is not None
    np.testing.assert_allclose(
        np.asarray(sampled.diagnostics.resolved.Wphi_kyst),
        np.asarray(full.diagnostics.resolved.Wphi_kyst)[sample_idx],
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_runtime_nonlinear_compact_diagnostics_match_scalar_channels() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    full = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=2,
        Nm=3,
        dt=0.01,
        steps=4,
        sample_stride=2,
        diagnostics_stride=2,
        resolved_diagnostics=True,
    )
    compact = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=2,
        Nm=3,
        dt=0.01,
        steps=4,
        sample_stride=2,
        diagnostics_stride=2,
        resolved_diagnostics=False,
    )

    assert full.diagnostics is not None
    assert compact.diagnostics is not None
    assert full.diagnostics.resolved is not None
    assert compact.diagnostics.resolved is None
    for name in (
        "t",
        "dt_t",
        "gamma_t",
        "omega_t",
        "Wg_t",
        "Wphi_t",
        "heat_flux_t",
        "particle_flux_t",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(compact.diagnostics, name)),
            np.asarray(getattr(full.diagnostics, name)),
            rtol=1.0e-6,
            atol=1.0e-8,
        )


def test_runtime_nonlinear_em_flux_channels_sum_to_total() -> None:
    ion = RuntimeSpeciesConfig(
        name="ion",
        charge=1.0,
        mass=1.0,
        density=1.0,
        temperature=1.0,
        tprim=1.0,
        fprim=1.0,
    )
    electron = RuntimeSpeciesConfig(
        name="electron",
        charge=-1.0,
        mass=1.0 / 3670.0,
        density=1.0,
        temperature=1.0,
        tprim=1.0,
        fprim=1.0,
    )
    cfg = replace(
        _base_runtime_cfg(),
        species=(ion, electron),
        normalization=RuntimeNormalizationConfig(
            contract="kbm", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(
            adiabatic_electrons=False,
            electrostatic=False,
            electromagnetic=True,
            use_apar=True,
            use_bpar=True,
            nonlinear=True,
            beta=0.02,
            collisions=False,
            hypercollisions=False,
        ),
        init=InitializationConfig(
            init_field="all", init_amp=1.0e-8, gaussian_init=False
        ),
        terms=RuntimeTermsConfig(
            nonlinear=1.0, hypercollisions=0.0, end_damping=0.0, apar=1.0, bpar=1.0
        ),
    )

    res = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=2,
        sample_stride=1,
        return_state=True,
    )

    assert res.diagnostics is not None
    assert isinstance(res.diagnostics.resolved, ResolvedDiagnostics)
    resolved = res.diagnostics.resolved
    assert resolved is not None
    for total, es, apar, bpar in (
        (
            resolved.HeatFlux_kxst,
            resolved.HeatFluxES_kxst,
            resolved.HeatFluxApar_kxst,
            resolved.HeatFluxBpar_kxst,
        ),
        (
            resolved.HeatFlux_kyst,
            resolved.HeatFluxES_kyst,
            resolved.HeatFluxApar_kyst,
            resolved.HeatFluxBpar_kyst,
        ),
        (
            resolved.HeatFlux_kxkyst,
            resolved.HeatFluxES_kxkyst,
            resolved.HeatFluxApar_kxkyst,
            resolved.HeatFluxBpar_kxkyst,
        ),
        (
            resolved.HeatFlux_zst,
            resolved.HeatFluxES_zst,
            resolved.HeatFluxApar_zst,
            resolved.HeatFluxBpar_zst,
        ),
        (
            resolved.ParticleFlux_kxst,
            resolved.ParticleFluxES_kxst,
            resolved.ParticleFluxApar_kxst,
            resolved.ParticleFluxBpar_kxst,
        ),
        (
            resolved.ParticleFlux_kyst,
            resolved.ParticleFluxES_kyst,
            resolved.ParticleFluxApar_kyst,
            resolved.ParticleFluxBpar_kyst,
        ),
        (
            resolved.ParticleFlux_kxkyst,
            resolved.ParticleFluxES_kxkyst,
            resolved.ParticleFluxApar_kxkyst,
            resolved.ParticleFluxBpar_kxkyst,
        ),
        (
            resolved.ParticleFlux_zst,
            resolved.ParticleFluxES_zst,
            resolved.ParticleFluxApar_zst,
            resolved.ParticleFluxBpar_zst,
        ),
    ):
        assert total is not None
        assert es is not None
        assert apar is not None
        assert bpar is not None
        np.testing.assert_allclose(
            np.asarray(total),
            np.asarray(es) + np.asarray(apar) + np.asarray(bpar),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    assert res.diagnostics.turbulent_heating_species_t is not None
    turb_heat_s = np.asarray(res.diagnostics.turbulent_heating_species_t)
    assert resolved.TurbulentHeating_kxst is not None
    assert resolved.TurbulentHeating_kyst is not None
    assert resolved.TurbulentHeating_kxkyst is not None
    assert resolved.TurbulentHeating_zst is not None
    np.testing.assert_allclose(
        np.asarray(resolved.TurbulentHeating_kxst).sum(axis=2),
        turb_heat_s,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(resolved.TurbulentHeating_kyst).sum(axis=2),
        turb_heat_s,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(resolved.TurbulentHeating_kxkyst).sum(axis=(2, 3)),
        turb_heat_s,
        rtol=1.0e-5,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(resolved.TurbulentHeating_zst).sum(axis=2),
        turb_heat_s,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_runtime_nonlinear_disable_diagnostics() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=0.2,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=3,
        diagnostics=False,
    )
    assert res.diagnostics is None
    assert res.phi2 is not None


def test_runtime_nonlinear_disable_diagnostics_uses_final_state_integrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    captured: dict[str, object] = {}

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: SimpleNamespace(),
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


def test_runtime_nonlinear_validates_dt_steps_and_fixed_mode_contract() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    with pytest.raises(ValueError, match="dt must be > 0"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.0, steps=2)

    with pytest.raises(ValueError, match="steps must be >= 1"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.1, steps=0)

    fixed_cfg = replace(
        cfg,
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=None, ikx_fixed=None),
    )
    with pytest.raises(ValueError, match="expert.iky_fixed and expert.ikx_fixed"):
        run_runtime_nonlinear(fixed_cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.1, steps=2)


def test_runtime_nonlinear_adaptive_chunk_no_progress_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gkx.runtime as runtime

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
        time=replace(_base_runtime_cfg().time, fixed_dt=False, t_max=0.2, dt=0.1),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: geom)
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda *args, **kwargs: SimpleNamespace(),
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

    zero_diag = SimulationDiagnostics(
        t=np.asarray([0.0]),
        dt_t=np.asarray([0.0]),
        dt_mean=0.0,
        gamma_t=np.asarray([0.0]),
        omega_t=np.asarray([0.0]),
        Wg_t=np.asarray([0.0]),
        Wphi_t=np.asarray([0.0]),
        Wapar_t=np.asarray([0.0]),
        heat_flux_t=np.asarray([0.0]),
        particle_flux_t=np.asarray([0.0]),
        energy_t=np.asarray([0.0]),
    )

    monkeypatch.setattr(
        runtime,
        "integrate_nonlinear_explicit_diagnostics_state",
        lambda *args, **kwargs: (
            np.asarray([0.0]),
            zero_diag,
            np.zeros(
                (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
            ),
            FieldState(
                phi=jnp.ones(
                    (grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
                ),
                apar=None,
                bpar=None,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="made no time-step progress"):
        run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, diagnostics=True)


def test_runtime_init_file_replace_mode_scales_loaded_state(tmp_path) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_file=str(tmp_path / "restart.bin"),
            init_file_scale=2.5,
            init_file_mode="replace",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    raw = (1.0 + 2.0j) * np.ones(
        (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    raw.tofile(cfg.init.init_file)

    g0 = _build_initial_condition(
        grid, geom, cfg, ky_index=1, kx_index=0, Nl=3, Nm=4, nspecies=1
    )

    assert np.allclose(np.asarray(g0), 2.5 * raw)


def test_runtime_init_file_add_mode_adds_seed_perturbation(tmp_path) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
            init_file=str(tmp_path / "restart.bin"),
            init_file_scale=3.0,
            init_file_mode="add",
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    base = np.ones(
        (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64
    )
    base.tofile(cfg.init.init_file)

    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=1, kx_index=0, Nl=3, Nm=4, nspecies=1
        )
    )

    assert np.allclose(g0[0, 1:, :, :, :, :], 3.0)
    assert np.allclose(g0[0, 0, 1:, :, :, :], 3.0)
    assert np.allclose(g0[0, 0, 0, 0, :, :], 3.0)
    assert np.allclose(g0[0, 0, 0, 1, 1:, :], 3.0)
    expected = 3.0 + 0.25
    assert np.allclose(g0[0, 0, 0, 1, 0, :], expected)


def test_runtime_single_mode_init_matches_gx_real_phase() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1
        )
    )

    seeded = g0[0, 0, 0, ky_index, 0, :]
    assert np.allclose(seeded.real, 0.25)
    assert np.allclose(seeded.imag, 0.0)


def test_runtime_single_mode_init_applies_gx_kpar_phase() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
            kpar_init=2.0,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1
        )
    )

    z = np.asarray(grid.z, dtype=float)
    z_period = _periodic_zp_from_grid(z)
    expected = 0.25 * np.cos(2.0 * z / z_period)
    seeded = g0[0, 0, 0, ky_index, 0, :]
    assert np.allclose(seeded.real, expected, atol=1.0e-6)
    assert np.allclose(seeded.imag, 0.0)


def test_runtime_phi_init_inverts_adiabatic_zonal_quasineutrality() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4, Ny=4, Nz=16, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        ),
        init=InitializationConfig(
            init_field="phi",
            init_amp=0.25,
            gaussian_init=False,
            init_single=True,
            kpar_init=1.0,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, density=1.0, temperature=1.0, kinetic=True
            ),
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = 0
    kx_index = 1

    g0 = _build_initial_condition(
        grid, geom, cfg, ky_index=ky_index, kx_index=kx_index, Nl=2, Nm=2, nspecies=1
    )
    params = build_runtime_linear_params(cfg, Nm=2, geom=geom)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    fields = compute_fields_cached(jnp.asarray(g0), cache, params, use_custom_vjp=False)

    z = np.asarray(grid.z, dtype=float)
    expected_phi = cfg.init.init_amp * np.cos(
        float(cfg.init.kpar_init) * z / _periodic_zp_from_grid(z)
    )
    recovered_phi = np.asarray(fields.phi[ky_index, kx_index, :])
    seeded_density = np.asarray(g0)[0, 0, 0, ky_index, kx_index, :]

    assert np.allclose(recovered_phi.real, expected_phi, rtol=1.0e-5, atol=1.0e-6)
    assert np.allclose(recovered_phi.imag, 0.0, atol=1.0e-7)
    assert not np.allclose(seeded_density.real, expected_phi)


def test_runtime_phi_gaussian_single_mode_inverts_zonal_profile() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4, Ny=4, Nz=16, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        ),
        init=InitializationConfig(
            init_field="phi",
            init_amp=0.25,
            gaussian_init=True,
            gaussian_width=1.0,
            init_single=True,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, density=1.0, temperature=1.0, kinetic=True
            ),
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    ky_index = 0
    kx_index = 1

    g0 = _build_initial_condition(
        grid, geom, cfg, ky_index=ky_index, kx_index=kx_index, Nl=2, Nm=2, nspecies=1
    )
    params = build_runtime_linear_params(cfg, Nm=2, geom=geom)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    fields = compute_fields_cached(jnp.asarray(g0), cache, params, use_custom_vjp=False)

    z = np.asarray(grid.z, dtype=float)
    expected_phi = cfg.init.init_amp * np.exp(-((z / cfg.init.gaussian_width) ** 2))
    recovered_phi = np.asarray(fields.phi[ky_index, kx_index, :])
    seeded_density = np.asarray(g0)[0, 0, 0, ky_index, kx_index, :]

    assert np.allclose(recovered_phi.real, expected_phi, rtol=1.0e-5, atol=1.0e-6)
    assert np.allclose(recovered_phi.imag, 0.0, atol=1.0e-7)
    assert not np.allclose(seeded_density.real, expected_phi)


def test_runtime_phi_init_rejects_masked_gauge_mode() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4, Ny=4, Nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        ),
        init=InitializationConfig(
            init_field="phi", init_amp=0.25, gaussian_init=False, init_single=True
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)

    with pytest.raises(ValueError, match="gauge mode"):
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=0, Nl=2, Nm=2, nspecies=1
        )


def test_runtime_startup_phi_density_seed_validation_paths() -> None:
    cache = SimpleNamespace(
        Jl=np.ones((1, 1, 1, 1, 3), dtype=float),
        jacobian=np.ones(3, dtype=float),
        mask0=np.zeros((1, 1, 3), dtype=bool),
        ky=np.asarray([0.2]),
    )
    params = LinearParams(
        charge_sign=np.asarray([1.0]),
        density=np.asarray([1.0]),
        tz=np.asarray([1.0]),
        tau_e=0.0,
    )
    phi = np.asarray([1.0, 0.5, 0.25], dtype=np.complex64)

    with pytest.raises(ValueError, match="at least one kinetic"):
        startup._density_moments_for_target_phi(
            phi, cache=cache, params=params, ky_i=0, kx_i=0, species_targets=()
        )
    with pytest.raises(ValueError, match="out of range"):
        startup._density_moments_for_target_phi(
            phi, cache=cache, params=params, ky_i=0, kx_i=0, species_targets=(1,)
        )

    bad_params = replace(params, density=np.asarray([np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        startup._density_moments_for_target_phi(
            phi, cache=cache, params=bad_params, ky_i=0, kx_i=0, species_targets=(0,)
        )

    zero_coupling_cache = SimpleNamespace(
        Jl=np.zeros((1, 1, 1, 1, 3), dtype=float),
        jacobian=np.ones(3, dtype=float),
        mask0=np.zeros((1, 1, 3), dtype=bool),
        ky=np.asarray([0.2]),
    )
    tau_params = replace(params, tau_e=1.0)
    with pytest.raises(ValueError, match="cannot be represented"):
        startup._density_moments_for_target_phi(
            phi,
            cache=zero_coupling_cache,
            params=tau_params,
            ky_i=0,
            kx_i=0,
            species_targets=(0,),
        )

    zero_phi = np.zeros(3, dtype=np.complex64)
    masked_cache = SimpleNamespace(
        Jl=np.ones((1, 1, 1, 1, 3), dtype=float),
        jacobian=np.ones(3, dtype=float),
        mask0=np.ones((1, 1, 3), dtype=bool),
        ky=np.asarray([0.0]),
    )
    seed = startup._density_moments_for_target_phi(
        zero_phi,
        cache=masked_cache,
        params=params,
        ky_i=0,
        kx_i=0,
        species_targets=(0,),
    )
    np.testing.assert_allclose(seed[0], zero_phi)

    assert startup._periodic_zp_from_grid(np.asarray([0.0])) == pytest.approx(1.0)
    assert startup._periodic_zp_from_grid(np.asarray([1.0, 1.0])) == pytest.approx(1.0)
    np.testing.assert_allclose(
        startup._as_runtime_species_array(2.0, 3, "density"), [2.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(
        startup._as_runtime_species_array(np.asarray([1.0, 2.0]), 2, "density"),
        [1.0, 2.0],
    )
    with pytest.raises(ValueError, match="length 2"):
        startup._as_runtime_species_array(np.asarray([1.0, 2.0, 3.0]), 2, "density")

    z = np.linspace(-1.0, 1.0, 5)
    direct_phi = startup._build_single_phi_gaussian_profile(
        z,
        kx=0.1,
        ky=0.2,
        s_hat=0.8,
        width=1.0,
        envelope_constant=1.0,
        envelope_sine=0.0,
    )
    delegated = startup._build_gaussian_profile(
        z,
        kx=0.1,
        ky=0.2,
        s_hat=0.8,
        width=1.0,
        envelope_constant=1.0,
        envelope_sine=0.0,
    )
    np.testing.assert_allclose(direct_phi, delegated)
    assert (
        startup._expand_ky(
            np.zeros((1, 1, 1, 2, 1, 1), dtype=np.complex64), nyc=1
        ).shape[-3]
        == 2
    )
    assert (
        startup._enforce_full_ky_hermitian(
            np.zeros((1, 1, 1, 2, 1, 1), dtype=np.complex64)
        ).shape[-3]
        == 2
    )


def test_runtime_startup_initial_condition_resolution_errors() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4, Ny=4, Nz=8, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        ),
        init=InitializationConfig(
            init_field="qpar", init_amp=0.25, gaussian_init=False, init_single=True
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    with pytest.raises(ValueError, match="moment exceeds"):
        _build_initial_condition(
            grid, geom, cfg, ky_index=1, kx_index=0, Nl=1, Nm=2, nspecies=1
        )

    phi_cfg = replace(cfg, init=replace(cfg.init, init_field="phi"))
    with pytest.raises(ValueError, match="requires at least one"):
        _build_initial_condition(
            grid, geom, phi_cfg, ky_index=1, kx_index=0, Nl=0, Nm=1, nspecies=1
        )


def test_runtime_nonlinear_fixed_mode_returns_frozen_state() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic")
    )
    ky_fixed = float(np.asarray(grid.ky)[1])
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
        expert=RuntimeExpertConfig(fixed_mode=True, iky_fixed=1, ikx_fixed=0),
    )
    res = run_runtime_nonlinear(
        cfg,
        ky_target=ky_fixed,
        Nl=3,
        Nm=4,
        dt=0.01,
        steps=3,
        sample_stride=1,
        return_state=True,
    )

    assert res.state is not None
    initial = _build_initial_condition(
        build_spectral_grid(cfg.grid),
        SAlphaGeometry.from_config(cfg.geometry),
        cfg,
        ky_index=1,
        kx_index=0,
        Nl=3,
        Nm=4,
        nspecies=1,
    )
    assert np.allclose(
        np.asarray(res.state)[..., 1, 0, :], np.asarray(initial)[..., 1, 0, :]
    )


def test_runtime_nonlinear_adaptive_dt() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.2,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=0.02,
            cfl=0.5,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.01, steps=4)
    assert res.diagnostics is not None
    t_arr = np.asarray(res.diagnostics.t)
    assert np.all(np.diff(t_arr) > 0)


def test_runtime_nonlinear_adaptive_default_steps_match_integrator_dt_cap() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.2,
            dt=0.01,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=None,
            cfl=1.0,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    steps_val = _infer_runtime_nonlinear_steps(cfg, dt=0.01, steps=None)

    assert steps_val == 20


def test_runtime_nonlinear_adaptive_default_steps_chunk_until_tmax(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.25,
            dt=0.1,
            method="rk3",
            use_diffrax=False,
            sample_stride=2,
            diagnostics_stride=2,
            fixed_dt=False,
            dt_min=1.0e-5,
            dt_max=None,
            cfl=1.0,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    calls: list[tuple[int, int, int]] = []
    input_means: list[float] = []

    def _fake_integrator(G0, grid, *_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        steps = kwargs["steps"]
        sample_stride = kwargs["sample_stride"]
        diagnostics_stride = kwargs["diagnostics_stride"]
        calls.append((int(steps), int(sample_stride), int(diagnostics_stride)))
        input_means.append(float(np.mean(np.real(np.asarray(G0)))))
        t = np.asarray([0.04, 0.08, 0.12], dtype=float)
        horizon = kwargs.get("time_horizon")
        if horizon is not None:
            t = np.minimum(t, float(horizon))
        dt_t = np.diff(np.insert(t, 0, 0.0))
        gamma_t = np.asarray([1.0, 2.0, 3.0], dtype=float) + 3.0 * (len(calls) - 1)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=dt_t,
            dt_mean=float(np.mean(dt_t)),
            gamma_t=gamma_t,
            omega_t=zeros,
            Wg_t=gamma_t,
            Wphi_t=zeros,
            Wapar_t=zeros,
            heat_flux_t=zeros,
            particle_flux_t=zeros,
            energy_t=gamma_t,
        )
        fields = FieldState(
            phi=np.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64),
            apar=None,
            bpar=None,
        )
        return t, diag, np.asarray(G0) + 1.0, fields

    monkeypatch.setattr(
        "gkx.runtime.integrate_nonlinear_explicit_diagnostics_state",
        _fake_integrator,
    )

    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, dt=0.1, steps=None)

    assert res.diagnostics is not None
    assert calls == [(3, 1, 1), (3, 1, 1), (3, 1, 1)]
    assert input_means[1] > input_means[0] + 0.5
    assert input_means[2] > input_means[1] + 0.5
    assert np.allclose(
        np.asarray(res.diagnostics.t), np.asarray([0.04, 0.12, 0.20, 0.25])
    )
    assert np.allclose(
        np.asarray(res.diagnostics.gamma_t), np.asarray([1.0, 3.0, 5.0, 7.0])
    )
    assert float(np.asarray(res.diagnostics.t)[-1]) == pytest.approx(cfg.time.t_max)


def test_runtime_gaussian_init_populates_multiple_modes_when_not_single() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=6,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=10.0,
            ntheta=16,
            nperiod=1,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=False,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=2,
            Nm=2,
            nspecies=1,
        )
    )
    amp_kykx = np.max(np.abs(g0[0, 0, 0, ...]), axis=-1)
    nonzero = amp_kykx > 0.0
    assert int(np.count_nonzero(nonzero)) > 1


def test_runtime_gaussian_single_moment_uses_configured_parallel_envelope() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=10.0,
            ntheta=16,
            nperiod=1,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=True,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))

    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=1
        )
    )

    seeded = g0[0, 0, 0, ky_index, 0, :]
    z = np.asarray(grid.z, dtype=float)
    expected = np.exp(-((z / cfg.init.gaussian_width) ** 2))
    assert np.allclose(seeded.real, expected)
    assert np.allclose(seeded.imag, expected)


def test_runtime_full_ky_initial_condition_mirrors_negative_ky_block() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=10,
            Ny=22,
            Nz=16,
            Lx=1.25,
            Ly=6.28,
            boundary="linked",
            y0=0.2,
            ntheta=16,
            nperiod=1,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            gaussian_width=0.5,
            init_single=True,
            init_electrons_only=True,
        ),
        species=(
            RuntimeSpeciesConfig(name="ion", charge=1.0, kinetic=True),
            RuntimeSpeciesConfig(name="electron", charge=-1.0, kinetic=True),
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 5.0)))

    g0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=ky_index, kx_index=0, Nl=3, Nm=4, nspecies=2
        )
    )

    neg_ky = g0.shape[3] - ky_index
    kx_neg = np.concatenate(([0], np.arange(g0.shape[4] - 1, 0, -1)))
    expected = np.conj(g0[1, 0, 0, ky_index, :, :])[kx_neg, :]
    assert np.allclose(g0[1, 0, 0, neg_ky, :, :], expected)
    assert np.allclose(g0[0], 0.0)


def test_runtime_nonlinear_dealias_toggle_executes() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.03,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            nonlinear_dealias=False,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )
    res = run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, steps=3)
    assert res.diagnostics is not None
    assert np.all(np.isfinite(res.diagnostics.Wphi_t))


@pytest.mark.parametrize(
    ("configured_cfl_fac", "expected_cfl_fac"),
    [
        pytest.param(None, 1.73, id="rk3-default"),
        pytest.param(1.25, 1.25, id="explicit-override"),
    ],
)
def test_runtime_nonlinear_resolves_cfl_factor(
    monkeypatch: pytest.MonkeyPatch,
    configured_cfl_fac: float | None,
    expected_cfl_fac: float,
) -> None:
    captured: dict[str, float] = {}

    def _fake_integrator(G0, grid, *_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        cfl_fac = kwargs["cfl_fac"]
        captured["cfl_fac"] = float(cfl_fac)
        t = np.asarray([0.1], dtype=float)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(t[0]),
            gamma_t=np.zeros_like(t),
            omega_t=np.zeros_like(t),
            Wg_t=np.zeros_like(t),
            Wphi_t=np.zeros_like(t),
            Wapar_t=np.zeros_like(t),
            heat_flux_t=np.zeros_like(t),
            particle_flux_t=np.zeros_like(t),
            energy_t=np.zeros_like(t),
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr(
        "gkx.runtime.integrate_nonlinear_explicit_diagnostics_state",
        _fake_integrator,
    )

    cfg = replace(
        _base_runtime_cfg(),
        time=TimeConfig(
            t_max=0.1,
            dt=0.1,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=False,
            cfl=1.0,
            cfl_fac=configured_cfl_fac,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    run_runtime_nonlinear(cfg, ky_target=0.2, Nl=3, Nm=4, steps=1)

    assert captured["cfl_fac"] == pytest.approx(expected_cfl_fac)


def test_runtime_init_species_targets_all_vs_electrons_only() -> None:
    cfg_all = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=1, Ny=8, Nz=16, Lx=6.28, Ly=6.28, boundary="periodic"),
        species=(
            RuntimeSpeciesConfig(name="ion", charge=1.0),
            RuntimeSpeciesConfig(name="electron", charge=-1.0, mass=1.0 / 3670.0),
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_single=True,
            init_electrons_only=False,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg_all.geometry)
    grid = build_spectral_grid(cfg_all.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g_all = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg_all,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=2,
        )
    )
    assert np.max(np.abs(g_all[0])) > 0.0
    assert np.max(np.abs(g_all[1])) > 0.0

    cfg_e = replace(
        cfg_all,
        init=replace(cfg_all.init, init_electrons_only=True),
    )
    g_e = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg_e,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=2,
        )
    )
    assert np.max(np.abs(g_e[0])) == 0.0
    assert np.max(np.abs(g_e[1])) > 0.0


def test_runtime_initial_condition_accepts_sampled_geometry_contract() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-8,
            gaussian_init=False,
            init_single=True,
        ),
    )
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )
    assert g0.shape == (1, 3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(g0))


def test_runtime_linear_accepts_gx_netcdf_geometry(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    grid = build_spectral_grid(cfg.grid)
    theta = np.asarray(grid.z, dtype=float)
    path = tmp_path / "geom.out.nc"
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        analytic = SAlphaGeometry.from_config(cfg.geometry)
        sampled = sample_flux_tube_geometry(analytic, grid.z)
        for name, values in {
            "bmag": np.asarray(sampled.bmag_profile),
            "bgrad": np.asarray(sampled.bgrad_profile),
            "gds2": np.asarray(sampled.gds2_profile),
            "gds21": np.asarray(sampled.gds21_profile),
            "gds22": np.asarray(sampled.gds22_profile),
            "cvdrift": np.asarray(sampled.cv_profile),
            "gbdrift": np.asarray(sampled.gb_profile),
            "cvdrift0": np.asarray(sampled.cv0_profile),
            "gbdrift0": np.asarray(sampled.gb0_profile),
            "jacobian": np.asarray(sampled.jacobian_profile),
            "grho": np.asarray(sampled.grho_profile),
        }.items():
            geom.createVariable(name, "f8", ("theta",))[:] = values
        for name, value in {
            "gradpar": sampled.gradpar_value,
            "q": sampled.q,
            "shat": sampled.s_hat,
            "rmaj": sampled.R0,
            "aminor": sampled.epsilon * sampled.R0,
        }.items():
            geom.createVariable(name, "f8", ())[:] = value

    cfg_nc = replace(
        cfg,
        geometry=replace(
            cfg.geometry, model="imported-netcdf", geometry_file=str(path)
        ),
    )
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_root_level_gx_eik_geometry(tmp_path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(path, sampled, Dataset)

    cfg_nc = replace(
        cfg,
        geometry=replace(
            cfg.geometry, model="imported-netcdf", geometry_file=str(path)
        ),
    )
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_explicit_time_accepts_root_level_gx_eik_geometry(
    tmp_path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        time=replace(_base_runtime_cfg().time, dt=0.02, t_max=0.08, sample_stride=1),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(path, sampled, Dataset)

    cfg_nc = replace(
        cfg,
        geometry=replace(
            cfg.geometry, model="imported-netcdf", geometry_file=str(path)
        ),
    )
    out = run_runtime_linear(cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_vmec_model_via_generated_eik(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "generated_geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(path, sampled, Dataset)

    monkeypatch.setattr(
        "gkx.runtime.generate_runtime_vmec_eik", lambda cfg: path
    )

    cfg_vmec = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            model="vmec",
            vmec_file=str(tmp_path / "wout_stub.nc"),
            geometry_file=str(path),
            torflux=0.64,
            npol=1.0,
        ),
    )
    out = run_runtime_linear(cfg_vmec, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_secondary_slab_example_runs() -> None:
    cfg_path = REPO_ROOT / "benchmarks" / "runtime_secondary_slab.toml"
    cfg, _ = load_runtime_from_toml(cfg_path)

    out = run_runtime_linear(cfg, ky_target=0.1, Nl=3, Nm=8, solver="explicit_time")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_linear_accepts_miller_model_via_generated_eik(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = RuntimeConfig(
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=16,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=10.0,
            ntheta=16,
            nperiod=1,
        ),
        time=TimeConfig(
            t_max=0.2,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=True,
        ),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(
            init_field="density", init_amp=1.0e-8, gaussian_init=False
        ),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
        physics=RuntimePhysicsConfig(
            linear=True, nonlinear=False, adiabatic_electrons=True, collisions=False
        ),
    )
    sampled = sample_flux_tube_geometry(
        SAlphaGeometry.from_config(cfg.geometry), build_spectral_grid(cfg.grid).z
    )
    path = tmp_path / "miller.eiknc.nc"
    _write_root_eik_geometry(path, sampled, Dataset)

    monkeypatch.setattr(
        "gkx.runtime.generate_runtime_miller_eik", lambda cfg: path
    )

    cfg_miller = replace(
        cfg,
        geometry=replace(
            cfg.geometry,
            model="miller",
            geometry_file=str(path),
            rhoc=0.5,
            R_geo=cfg.geometry.R0,
        ),
    )
    out = run_runtime_linear(cfg_miller, ky_target=0.2, Nl=4, Nm=6, solver="krylov")

    assert np.isfinite(out.gamma)
    assert np.isfinite(out.omega)


def test_runtime_etg_nonlinear_example_runs_small_smoke() -> None:
    cfg_path = (
        REPO_ROOT
        / "examples"
        / "nonlinear"
        / "axisymmetric"
        / "runtime_etg_nonlinear.toml"
    )
    cfg, _ = load_runtime_from_toml(cfg_path)

    # Keep the shipped ETG pilot contract intact, but reduce the test problem to
    # a CI-sized smoke so GitHub runners do not materialize the production-scale
    # nonlinear diagnostic buffers.
    cfg = replace(
        cfg,
        grid=GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=6.28,
            Ly=6.28,
            boundary="linked",
            y0=0.2,
            ntheta=8,
            nperiod=1,
        ),
        time=replace(
            cfg.time,
            t_max=0.002,
            dt=0.001,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=True,
        ),
    )

    out = run_runtime_nonlinear(
        cfg, ky_target=5.0, kx_target=0.0, steps=2, sample_stride=1, Nl=2, Nm=2
    )

    assert out.diagnostics is not None
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wg_t)))
    assert np.all(np.isfinite(np.asarray(out.diagnostics.Wphi_t)))
    assert np.allclose(np.asarray(out.diagnostics.Wapar_t), 0.0)


def test_runtime_linear_explicit_time_root_level_geometry_matches_analytic_reference(
    tmp_path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.08,
            dt=0.02,
            method="rk4",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0),),
        normalization=RuntimeNormalizationConfig(
            contract="kinetic", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        terms=RuntimeTermsConfig(end_damping=0.0, hypercollisions=0.0),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(
        path, sampled, Dataset, drift_scale=2.0, jacobian=7.0, drhodpsi=1.0
    )

    cfg_nc = replace(
        cfg,
        geometry=replace(
            cfg.geometry, model="imported-netcdf", geometry_file=str(path)
        ),
    )

    analytic_out = run_runtime_linear(
        cfg, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time"
    )
    imported_out = run_runtime_linear(
        cfg_nc, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time"
    )

    assert imported_out.gamma == pytest.approx(
        analytic_out.gamma, rel=2.0e-3, abs=1.0e-6
    )
    assert imported_out.omega == pytest.approx(
        analytic_out.omega, rel=2.0e-3, abs=1.0e-6
    )


def test_runtime_linear_explicit_time_root_level_geometry_matches_analytic_reference_from_toml(
    tmp_path,
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 33)
    path = tmp_path / "geom.eik.nc"
    analytic = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.0)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(
        path, sampled, Dataset, drift_scale=2.0, jacobian=7.0, drhodpsi=1.0
    )

    toml = f"""
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
Ny = 8
Nz = 32
Lx = 62.8
Ly = 62.8
boundary = "linked"
y0 = 10.0
z_min = {-3.0 * np.pi}
z_max = {3.0 * np.pi}

[time]
t_max = 0.08
dt = 0.02
method = "rk4"
use_diffrax = false
sample_stride = 1
fixed_dt = true

[geometry]
model = "imported-netcdf"
geometry_file = "{path}"

[physics]
adiabatic_electrons = true
tau_e = 1.0
electromagnetic = false

[terms]
end_damping = 0.0
hypercollisions = 0.0

[normalization]
contract = "kinetic"
diagnostic_norm = "none"
"""
    cfg_path = tmp_path / "runtime_w7x.toml"
    cfg_path.write_text(toml, encoding="utf-8")

    cfg_loaded, _ = load_runtime_from_toml(cfg_path)
    cfg_ref = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=1,
            Ny=8,
            Nz=32,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.08,
            dt=0.02,
            method="rk4",
            use_diffrax=False,
            sample_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0),),
        normalization=RuntimeNormalizationConfig(
            contract="kinetic", diagnostic_norm="none"
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
        terms=RuntimeTermsConfig(end_damping=0.0, hypercollisions=0.0),
    )

    loaded_out = run_runtime_linear(
        cfg_loaded, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time"
    )
    analytic_out = run_runtime_linear(
        cfg_ref, ky_target=0.2, Nl=4, Nm=6, solver="explicit_time"
    )

    assert loaded_out.gamma == pytest.approx(analytic_out.gamma, rel=2.0e-3, abs=1.0e-6)
    assert loaded_out.omega == pytest.approx(analytic_out.omega, rel=2.0e-3, abs=1.0e-6)


@pytest.mark.parametrize("model", ["vmec-eik", "desc-eik"])
def test_runtime_nonlinear_accepts_imported_eik_geometry_aliases(
    tmp_path, model: str
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=4,
            Ny=8,
            Nz=16,
            Lx=62.8,
            Ly=62.8,
            boundary="linked",
            y0=10.0,
            z_min=-3.0 * np.pi,
            z_max=3.0 * np.pi,
        ),
        time=TimeConfig(
            t_max=0.04,
            dt=0.02,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            fixed_dt=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0, nu=0.01),),
        normalization=RuntimeNormalizationConfig(
            contract="kinetic", diagnostic_norm="rho_star"
        ),
        physics=RuntimePhysicsConfig(
            linear=False,
            nonlinear=True,
            adiabatic_electrons=True,
            tau_e=1.0,
            electrostatic=True,
            electromagnetic=False,
            use_apar=False,
            use_bpar=False,
            collisions=False,
            hypercollisions=True,
        ),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0e-6,
            gaussian_init=False,
            init_single=False,
        ),
        terms=RuntimeTermsConfig(
            end_damping=1.0,
            hypercollisions=1.0,
            hyperdiffusion=1.0,
            nonlinear=1.0,
        ),
    )
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, cfg.grid.Nz + 1)
    path = tmp_path / f"geom_{model}.eik.nc"
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    _write_root_eik_geometry(
        path,
        sampled,
        Dataset,
        drift_scale=2.0,
        jacobian=7.0,
        drhodpsi=1.0,
        nfp=5.0,
    )

    cfg_nc = replace(
        cfg, geometry=replace(cfg.geometry, model=model, geometry_file=str(path))
    )
    out = run_runtime_nonlinear(
        cfg_nc, ky_target=0.2, Nl=3, Nm=4, dt=0.02, steps=2, sample_stride=1
    )

    assert np.all(np.isfinite(out.t))
    assert out.diagnostics is not None
    assert np.isfinite(out.diagnostics.Wg_t[-1])
    assert np.isfinite(out.diagnostics.Wphi_t[-1])
    assert np.isfinite(out.diagnostics.heat_flux_t[-1])
    assert np.isfinite(out.diagnostics.particle_flux_t[-1])


@pytest.mark.parametrize(
    ("grid_config", "init_config"),
    [
        pytest.param(
            None,
            InitializationConfig(
                init_field="all",
                init_amp=1.0,
                gaussian_init=False,
                init_single=True,
            ),
            id="single-mode",
        ),
        pytest.param(
            GridConfig(
                Nx=6,
                Ny=8,
                Nz=16,
                Lx=6.28,
                Ly=6.28,
                boundary="linked",
                y0=10.0,
                ntheta=16,
                nperiod=1,
            ),
            InitializationConfig(
                init_field="all",
                init_amp=1.0,
                gaussian_init=True,
                gaussian_width=0.5,
                init_single=False,
            ),
            id="multimode-gaussian",
        ),
        pytest.param(
            GridConfig(
                Nx=6,
                Ny=8,
                Nz=16,
                Lx=6.28,
                Ly=6.28,
                boundary="periodic",
            ),
            InitializationConfig(
                init_field="all",
                init_amp=1.0,
                gaussian_init=False,
                init_single=False,
                random_seed=7,
            ),
            id="multimode-random",
        ),
    ],
)
def test_runtime_init_all_preserves_hermite_moment_normalization(
    grid_config: GridConfig | None,
    init_config: InitializationConfig,
) -> None:
    overrides = {"init": init_config}
    if grid_config is not None:
        overrides["grid"] = grid_config
    cfg = replace(_base_runtime_cfg(), **overrides)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    ky_index = int(np.argmin(np.abs(np.asarray(grid.ky) - 1.0)))
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=ky_index,
            kx_index=0,
            Nl=3,
            Nm=4,
            nspecies=1,
        )
    )[0]

    amp_density = np.max(np.abs(g0[0, 0, ...]))
    amp_tpar = np.max(np.abs(g0[0, 2, ...]))
    amp_qpar = np.max(np.abs(g0[0, 3, ...]))
    assert amp_density > 0.0
    assert np.isclose(amp_tpar / amp_density, 1.0 / np.sqrt(2.0))
    assert np.isclose(amp_qpar / amp_density, 1.0 / np.sqrt(6.0))


def test_runtime_random_multimode_init_matches_glibc_random_sequence() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=7,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            nspecies=1,
        )
    )[0, 0, 0]

    z = np.asarray(grid.z, dtype=float)
    z_period = _periodic_zp_from_grid(z)
    z_phase = np.cos(float(cfg.init.kpar_init) * z / z_period)
    active_modes = _dealiased_initial_mode_pairs(grid)
    expected = np.zeros_like(g0)
    for (kx_i, ky_i), (ra, rb) in zip(
        active_modes,
        cfg.init.init_amp
        * _glibc_random_pairs_reference(int(cfg.init.random_seed), len(active_modes)),
        strict=True,
    ):
        vals = ((rb + 1j * ra) if kx_i == 0 else (ra + 1j * rb)) * z_phase
        expected[ky_i, kx_i, :] = vals
        if kx_i != 0:
            expected[ky_i, expected.shape[1] - kx_i, :] = (rb + 1j * ra) * z_phase
    ny = expected.shape[0]
    nyc = ny // 2 + 1
    neg_hi = nyc - 1 if (ny % 2) == 0 else nyc
    neg = np.conj(expected[1:neg_hi])[::-1]
    kx_neg = np.concatenate(([0], np.arange(expected.shape[1] - 1, 0, -1)))
    expected[nyc:] = neg[:, kx_neg, :]

    assert np.allclose(g0, expected, atol=1.0e-6)


def test_runtime_dealiased_initial_mode_pairs_match_dealiased_loop_bounds() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=96, Ny=96, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic")
    )
    active_modes = _dealiased_initial_mode_pairs(grid)

    assert active_modes[0] == (0, 1)
    assert active_modes[-1] == (31, 31)
    assert len(active_modes) == 32 * 31
    assert all(0 <= kx_i <= 31 for kx_i, _ in active_modes)
    assert all(1 <= ky_i <= 31 for _, ky_i in active_modes)


def test_runtime_centered_glibc_random_pairs_match_glibc_reference() -> None:
    vals = _centered_glibc_random_pairs(22, 5)
    ref = _glibc_random_pairs_reference(22, 5)

    assert np.allclose(vals, ref)


def test_runtime_periodic_zp_from_grid_uses_discrete_period_not_endpoint_span() -> None:
    cfg, _data = load_runtime_from_toml(
        "examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_short.toml"
    )
    geom = build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    z = np.asarray(grid.z, dtype=float)

    zp = _periodic_zp_from_grid(z)

    assert np.isclose(zp, 1.0, atol=2.0e-4)


def test_runtime_random_multimode_zero_kx_matches_reference_overwrite_order() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=4, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=22,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0 = np.asarray(
        _build_initial_condition(
            grid,
            geom,
            cfg,
            ky_index=1,
            kx_index=0,
            Nl=1,
            Nm=1,
            nspecies=1,
        )
    )[0, 0, 0]

    ra, rb = _glibc_random_pairs_reference(22, 1)[0]
    assert np.allclose(g0[1, 0, :], (rb + 1j * ra) * np.ones_like(g0[1, 0, :]))


def test_runtime_random_multimode_init_does_not_depend_on_diagnostic_ky() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=False,
            init_single=False,
            random_seed=7,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0_ky0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )
    g0_ky1 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=1, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.allclose(g0_ky0, g0_ky1)


def test_runtime_gaussian_multimode_init_does_not_depend_on_diagnostic_ky() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(Nx=6, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="periodic"),
        init=InitializationConfig(
            init_field="density",
            init_amp=1.0,
            gaussian_init=True,
            init_single=False,
            gaussian_width=0.35,
        ),
    )
    geom = SAlphaGeometry.from_config(cfg.geometry)
    grid = build_spectral_grid(cfg.grid)
    g0_ky0 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=0, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )
    g0_ky1 = np.asarray(
        _build_initial_condition(
            grid, geom, cfg, ky_index=1, kx_index=1, Nl=1, Nm=1, nspecies=1
        )
    )

    assert np.allclose(g0_ky0, g0_ky1)


def test_runtime_nonlinear_mode_selection_respects_dealias(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=3,
            Ny=7,
            Nz=16,
            Lx=62.8,
            Ly=62.8,
            boundary="periodic",
            y0=10.0,
        ),
        time=TimeConfig(
            t_max=0.02,
            dt=0.01,
            method="rk2",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            nonlinear_dealias=True,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    captured: dict[str, int] = {}

    def _fake_integrator(G0, grid, *_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        dt = kwargs["dt"]
        omega_ky_index = kwargs["omega_ky_index"]
        omega_kx_index = kwargs["omega_kx_index"]
        captured["omega_ky_index"] = int(omega_ky_index)
        captured["omega_kx_index"] = int(omega_kx_index)
        t = np.asarray([float(dt)], dtype=float)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(dt),
            gamma_t=zeros,
            omega_t=zeros,
            Wg_t=zeros,
            Wphi_t=zeros,
            Wapar_t=zeros,
            heat_flux_t=zeros,
            particle_flux_t=zeros,
            energy_t=zeros,
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr(
        "gkx.runtime.integrate_nonlinear_explicit_diagnostics_state",
        _fake_integrator,
    )
    _res = run_runtime_nonlinear(cfg, ky_target=0.3, Nl=3, Nm=4, steps=1)

    # Ny=7, y0=10 -> ky = [0, 0.1, 0.2, 0.3, -0.3, -0.2, -0.1]
    # 2/3 dealias keeps |n|<=2 => ky=0.3 is excluded; nearest kept mode is ky=0.2 (index 2).
    assert captured["omega_ky_index"] == 2
    # Nx=3 FFT order puts kx=0 at index 0.
    assert captured["omega_kx_index"] == 0


def test_runtime_nonlinear_mode_selection_honors_kx_target(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        grid=GridConfig(
            Nx=5,
            Ny=5,
            Nz=8,
            Lx=12.0,
            Ly=62.8,
            boundary="periodic",
            y0=10.0,
        ),
        time=TimeConfig(
            t_max=0.02,
            dt=0.01,
            method="rk3",
            use_diffrax=False,
            sample_stride=1,
            diagnostics_stride=1,
            nonlinear_dealias=False,
        ),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, nonlinear=True),
        terms=RuntimeTermsConfig(nonlinear=1.0, hypercollisions=0.0, end_damping=0.0),
    )

    captured: dict[str, int] = {}

    def _fake_integrator(G0, grid, *_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        dt = kwargs["dt"]
        omega_ky_index = kwargs["omega_ky_index"]
        omega_kx_index = kwargs["omega_kx_index"]
        captured["omega_ky_index"] = int(omega_ky_index)
        captured["omega_kx_index"] = int(omega_kx_index)
        t = np.asarray([float(dt)], dtype=float)
        zeros = np.zeros_like(t)
        diag = SimulationDiagnostics(
            t=t,
            dt_t=t,
            dt_mean=float(dt),
            gamma_t=zeros,
            omega_t=zeros,
            Wg_t=zeros,
            Wphi_t=zeros,
            Wapar_t=zeros,
            heat_flux_t=zeros,
            particle_flux_t=zeros,
            energy_t=zeros,
        )
        return t, diag, np.asarray(G0), None

    monkeypatch.setattr(
        "gkx.runtime.integrate_nonlinear_explicit_diagnostics_state",
        _fake_integrator,
    )
    _res = run_runtime_nonlinear(
        cfg, ky_target=0.1, kx_target=-1.1, Nl=3, Nm=4, steps=1
    )

    assert captured["omega_ky_index"] == 1
    assert captured["omega_kx_index"] == 3


def test_run_linear_case_uses_toml_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gkx.runtime as runtime

    base = _base_runtime_cfg()
    cfg = replace(base, output=replace(base.output, path=str(tmp_path / "linear_case")))

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.2, "Nl": 4, "Nm": 6, "solver": "krylov"}}

    def fake_run_runtime_linear(*_args, **_kwargs):
        return runtime.RuntimeLinearResult(
            ky=0.2,
            gamma=0.1,
            omega=0.2,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.toml.load_runtime_from_toml",
        fake_load_runtime_from_toml,
    )
    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run_runtime_linear)

    rc = run_linear_case(tmp_path / "dummy.toml", show_progress=False)

    out = capsys.readouterr().out
    assert rc == 0
    assert f"saved {tmp_path / 'linear_case.summary.json'}" in out
    assert (tmp_path / "linear_case.summary.json").exists()


def test_run_linear_case_toml_velocity_auto_reaches_parallel_rhs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gkx.solvers.linear.integrators as linear_integrators

    cfg_path = tmp_path / "velocity_auto.toml"
    cfg_path.write_text(
        """
[grid]
Nx = 1
Ny = 4
Nz = 4
Lx = 6.28
Ly = 62.8318530718
boundary = "periodic"

[time]
t_max = 0.02
dt = 0.01
method = "euler"
use_diffrax = false
sample_stride = 1

[geometry]
model = "s-alpha"
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1.0e-8
gaussian_init = false

[physics]
adiabatic_electrons = true
electrostatic = true
collisions = false
hypercollisions = false
use_apar = false
use_bpar = false
beta = 0.0

[terms]
collisions = 0.0
hypercollisions = 0.0
hyperdiffusion = 0.0
end_damping = 0.0
apar = 0.0
bpar = 0.0

[parallel]
strategy = "velocity"
axis = "hermite"
backend = "auto"
num_devices = 1

[run]
ky = 0.1
Nl = 2
Nm = 3
solver = "time"

[fit]
fit_signal = "phi"
""",
        encoding="utf-8",
    )
    calls: list[str] = []

    def _fake_parallel_rhs(G, cache, params, **kwargs):
        parallel = kwargs["parallel"]
        calls.append(f"{parallel.strategy}:{parallel.axis}:{parallel.backend}")
        return jnp.zeros_like(G), jnp.ones(G.shape[-3:], dtype=G.dtype)

    monkeypatch.setattr(
        linear_integrators, "linear_rhs_parallel_cached", _fake_parallel_rhs
    )

    rc = run_linear_case(cfg_path, show_progress=False)

    out = capsys.readouterr().out
    assert rc == 0
    assert "ky=0.100000" in out
    assert calls
    assert set(calls) == {"velocity:hermite:auto"}


def test_run_nonlinear_case_uses_toml_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gkx.runtime as runtime

    base = _base_runtime_cfg()
    cfg = replace(
        base, output=replace(base.output, path=str(tmp_path / "nonlinear_case"))
    )
    t = np.asarray([0.0, 0.1], dtype=float)
    diag = SimulationDiagnostics(
        t=t,
        dt_t=t + 0.1,
        dt_mean=0.1,
        gamma_t=np.asarray([0.0, 0.1], dtype=float),
        omega_t=np.asarray([0.0, 0.2], dtype=float),
        Wg_t=np.asarray([1.0, 1.1], dtype=float),
        Wphi_t=np.asarray([2.0, 2.1], dtype=float),
        Wapar_t=np.asarray([0.0, 0.0], dtype=float),
        heat_flux_t=np.asarray([4.0, 4.2], dtype=float),
        particle_flux_t=np.asarray([5.0, 5.2], dtype=float),
        energy_t=np.asarray([3.0, 3.2], dtype=float),
    )

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.2, "Nl": 4, "Nm": 6}, "time": {"dt": 0.1}}

    captured: dict[str, object] = {}

    def fake_run_runtime_nonlinear_with_artifacts(*_args, **_kwargs):
        captured.update(_kwargs)
        summary = tmp_path / "nonlinear_case.summary.json"
        diag_path = tmp_path / "nonlinear_case.diagnostics.csv"
        summary.write_text("{}\n", encoding="utf-8")
        diag_path.write_text(
            "t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux\n",
            encoding="utf-8",
        )
        return (
            runtime.RuntimeNonlinearResult(
                t=t,
                diagnostics=diag,
                ky_selected=0.2,
                kx_selected=0.0,
            ),
            {"summary": str(summary), "diagnostics": str(diag_path)},
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.toml.load_runtime_from_toml",
        fake_load_runtime_from_toml,
    )
    monkeypatch.setattr(
        "gkx.workflows.runtime.artifacts.run_runtime_nonlinear_with_artifacts",
        fake_run_runtime_nonlinear_with_artifacts,
    )

    rc = run_nonlinear_case(tmp_path / "dummy.toml", show_progress=False)

    out = capsys.readouterr().out
    assert rc == 0
    assert captured["out"] == str(tmp_path / "nonlinear_case")
    assert f"saved {tmp_path / 'nonlinear_case.summary.json'}" in out
    assert (tmp_path / "nonlinear_case.summary.json").exists()
    assert (tmp_path / "nonlinear_case.diagnostics.csv").exists()


def test_run_linear_case_without_output_path_prints_summary_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gkx.runtime as runtime

    cfg = _base_runtime_cfg()

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.25}}

    def fake_run_runtime_linear(*_args, **_kwargs):
        return runtime.RuntimeLinearResult(
            ky=0.25,
            gamma=0.11,
            omega=0.22,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.toml.load_runtime_from_toml",
        fake_load_runtime_from_toml,
    )
    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run_runtime_linear)

    rc = run_linear_case(tmp_path / "dummy.toml", show_progress=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert "saved" not in out
    assert "ky=0.250000 gamma=0.11000000 omega=0.22000000" in out


def test_run_nonlinear_case_without_output_path_and_without_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gkx.runtime as runtime

    cfg = _base_runtime_cfg()

    def fake_load_runtime_from_toml(_path):
        return cfg, {"run": {"ky": 0.3}, "time": {"dt": 0.1}}

    captured: dict[str, object] = {}

    def fake_run_runtime_nonlinear(*_args, **_kwargs):
        captured.update(_kwargs)
        return runtime.RuntimeNonlinearResult(
            t=np.asarray([0.1]),
            diagnostics=None,
            ky_selected=None,
            kx_selected=0.0,
        )

    monkeypatch.setattr(
        "gkx.workflows.runtime.toml.load_runtime_from_toml",
        fake_load_runtime_from_toml,
    )
    monkeypatch.setattr(runtime, "run_runtime_nonlinear", fake_run_runtime_nonlinear)

    rc = run_nonlinear_case(tmp_path / "dummy.toml", show_progress=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert captured["diagnostics"] is True
    assert "completed without streamed diagnostics" in out


def test_run_runtime_scan_batch_ky_rejects_krylov() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    with pytest.raises(ValueError):
        run_runtime_scan(cfg, ky_values=[0.1, 0.2], solver="krylov", batch_ky=True)


def test_run_runtime_scan_serial_forwards_per_ky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    calls: list[dict[str, object]] = []

    def _fake_run_runtime_linear(_cfg, **kwargs):
        calls.append(kwargs)
        ky = float(kwargs["ky_target"])
        return RuntimeLinearResult(
            ky=ky,
            gamma=ky + 1.0,
            omega=-(ky + 2.0),
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    out = run_runtime_scan(
        cfg,
        ky_values=[0.15, 0.35],
        solver="time",
        method="rk2",
        dt=0.05,
        steps=4,
        sample_stride=2,
        fit_signal="phi",
    )

    assert len(calls) == 2
    assert calls[0]["ky_target"] == pytest.approx(0.15)
    assert calls[1]["ky_target"] == pytest.approx(0.35)
    assert calls[0]["sample_stride"] == 2
    np.testing.assert_allclose(out.gamma, [1.15, 1.35])
    np.testing.assert_allclose(out.omega, [-2.15, -2.35])


def test_run_runtime_scan_independent_workers_preserve_quasilinear_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        return RuntimeLinearResult(
            ky=ky,
            gamma=ky + 1.0,
            omega=-(ky + 2.0),
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            quasilinear={
                "ky": ky,
                "gamma": ky + 1.0,
                "omega": -(ky + 2.0),
                "heat_flux_weight_total": 10.0 * ky,
            },
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    out = run_runtime_scan(
        cfg,
        ky_values=[0.15, 0.35, 0.25],
        solver="time",
        workers=2,
    )

    np.testing.assert_allclose(out.gamma, [1.15, 1.35, 1.25])
    assert out.quasilinear is not None
    np.testing.assert_allclose(
        [row["ky"] for row in out.quasilinear], [0.15, 0.35, 0.25]
    )
    np.testing.assert_allclose(
        [row["heat_flux_weight_total"] for row in out.quasilinear], [1.5, 3.5, 2.5]
    )
    assert out.parallel is not None
    assert out.parallel["requested_workers"] == 2
    assert out.parallel["source"] == "arguments"
    assert out.parallel["quasilinear_state_extraction"] is True


def test_run_runtime_scan_parallel_config_batch_selects_independent_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=3, backend="thread"
        ),
    )
    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        return RuntimeLinearResult(
            ky=ky,
            gamma=ky + 1.0,
            omega=-(ky + 2.0),
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    def _fake_independent_map(fn, values, *, workers=1, executor="thread"):
        items = list(values)
        captured["workers"] = workers
        captured["executor"] = executor
        captured["ky"] = [item["ky"] for item in items]
        return [fn(item) for item in items]

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(runtime, "independent_map", _fake_independent_map)

    out = run_runtime_scan(cfg, ky_values=[0.15, 0.35, 0.25], solver="time")

    assert captured == {
        "workers": 3,
        "executor": "thread",
        "ky": [0.15, 0.35, 0.25],
    }
    np.testing.assert_allclose(out.gamma, [1.15, 1.35, 1.25])
    assert out.parallel is not None
    assert out.parallel["source"] == "runtime_config"
    assert out.parallel["requested_workers"] == 3
    assert out.parallel["effective_workers"] == 3
    assert out.parallel["enabled"] is True


def test_run_runtime_scan_explicit_workers_override_parallel_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        parallel=RuntimeParallelConfig(
            strategy="batch", axis="ky", num_devices=4, backend="process"
        ),
    )
    captured: dict[str, object] = {}

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        return RuntimeLinearResult(
            ky=ky,
            gamma=ky,
            omega=-ky,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        )

    def _fake_independent_map(fn, values, *, workers=1, executor="thread"):
        captured["workers"] = workers
        captured["executor"] = executor
        return [fn(item) for item in values]

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)
    monkeypatch.setattr(runtime, "independent_map", _fake_independent_map)

    out = run_runtime_scan(
        cfg,
        ky_values=[0.15, 0.35, 0.25],
        solver="time",
        workers=2,
        parallel_executor="thread",
    )

    assert captured == {"workers": 2, "executor": "thread"}
    assert out.parallel is not None
    assert out.parallel["source"] == "arguments"
    assert out.parallel["requested_workers"] == 2
    assert out.parallel["effective_workers"] == 2


def test_run_runtime_scan_parallel_config_batch_rejects_non_ky_axis() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        parallel=RuntimeParallelConfig(strategy="batch", axis="kx", num_devices=2),
    )

    with pytest.raises(ValueError, match="axis='ky'"):
        run_runtime_scan(cfg, ky_values=[0.15, 0.35], solver="time")


def test_run_runtime_scan_parallel_config_selects_combined_ky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        parallel=RuntimeParallelConfig(strategy="combined-ky"),
    )
    calls: list[dict[str, object]] = []

    def _fake_run_runtime_scan_batch(_cfg, ky_arr, **kwargs):
        calls.append({"ky_arr": np.asarray(ky_arr), **kwargs})
        return RuntimeLinearScanResult(
            ky=np.asarray(ky_arr, dtype=float),
            gamma=np.asarray(ky_arr, dtype=float) + 1.0,
            omega=-(np.asarray(ky_arr, dtype=float) + 2.0),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(
        runtime, "_run_runtime_scan_batch", _fake_run_runtime_scan_batch
    )

    out = run_runtime_scan(cfg, ky_values=[0.15, 0.35], solver="time", sample_stride=2)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0]["ky_arr"], [0.15, 0.35])
    assert calls[0]["sample_stride"] == 2
    np.testing.assert_allclose(out.gamma, [1.15, 1.35])
    np.testing.assert_allclose(out.omega, [-2.15, -2.35])


def test_run_runtime_scan_batch_empty_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "build_runtime_geometry", lambda _cfg: object())
    monkeypatch.setattr(
        runtime, "apply_geometry_grid_defaults", lambda _geom, grid: grid
    )
    monkeypatch.setattr(
        runtime,
        "build_spectral_grid",
        lambda _grid: SimpleNamespace(
            ky=np.array([0.0, 0.2]), z=np.array([0.0]), kx=np.array([0.0])
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda _cfg, **_kwargs: SimpleNamespace(rho_star=1.0),
    )
    monkeypatch.setattr(runtime, "build_runtime_linear_terms", lambda _cfg: object())

    with pytest.raises(ValueError):
        run_runtime_scan(cfg, ky_values=[], solver="time", batch_ky=True)


def test_runtime_linear_explicit_time_rejects_return_state_before_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(
            contract="cyclone", diagnostic_norm="none"
        ),
    )
    import gkx.runtime as runtime

    monkeypatch.setattr(
        runtime, "build_runtime_geometry", lambda _cfg: SimpleNamespace()
    )
    monkeypatch.setattr(
        runtime, "apply_geometry_grid_defaults", lambda _geom, grid: grid
    )
    monkeypatch.setattr(
        runtime,
        "build_spectral_grid",
        lambda _grid: SimpleNamespace(
            ky=np.array([0.0, 0.2]), kx=np.array([0.0]), z=np.array([-1.0, 0.0, 1.0])
        ),
    )
    monkeypatch.setattr(
        runtime,
        "build_runtime_linear_params",
        lambda _cfg, **_kwargs: SimpleNamespace(rho_star=1.0),
    )
    monkeypatch.setattr(runtime, "build_runtime_linear_terms", lambda _cfg: object())
    monkeypatch.setattr(
        runtime,
        "select_ky_grid",
        lambda grid, _idx: SimpleNamespace(ky=np.array([0.2]), kx=grid.kx, z=grid.z),
    )
    monkeypatch.setattr(
        runtime,
        "_build_initial_condition",
        lambda *args, **kwargs: np.zeros((2, 2, 1, 1, 3), dtype=np.complex64),
    )

    with pytest.raises(ValueError):
        run_runtime_linear(
            cfg, ky_target=0.2, Nl=2, Nm=2, solver="explicit_time", return_state=True
        )


def test_direct_linear_and_nonlinear_integrators_fast_smoke() -> None:
    from gkx.solvers.nonlinear.state_integration import integrate_nonlinear
    from gkx.solvers.linear.integrators import integrate_linear

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    state = (
        jnp.ones((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
        * 1.0e-6
    )

    _, phi_t = integrate_linear(
        state, grid, geom, params, dt=0.1, steps=2, method="rk2"
    )
    _, fields = integrate_nonlinear(
        state, grid, geom, params, dt=0.1, steps=2, method="rk2"
    )

    assert jnp.isfinite(phi_t).all()
    assert jnp.isfinite(fields.phi).all()


def test_runtime_parameter_scan_updates_config_and_continues_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        physics=RuntimePhysicsConfig(beta=0.0),
        species=(RuntimeSpeciesConfig(name="ion"),),
    )
    states = [np.full((1, 1), value) for value in (1.0, 2.0, 3.0)]
    calls: list[tuple[float, dict[str, object]]] = []

    def fake_run(point_cfg, **kwargs):
        index = len(calls)
        calls.append((point_cfg.physics.beta, kwargs))
        return RuntimeLinearResult(
            ky=float(kwargs["ky_target"]),
            gamma=float(point_cfg.physics.beta + 1.0),
            omega=float(-point_cfg.physics.beta),
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=states[index],
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    result = run_runtime_parameter_scan(
        cfg,
        [0.01, 0.02, 0.04],
        parameter_name="beta",
        update_config=lambda base, value, _index: replace(
            base, physics=replace(base.physics, beta=value)
        ),
        ky_target=0.25,
        linear_options={"solver": "krylov"},
        continuation=True,
    )

    assert isinstance(result, RuntimeParameterScanResult)
    np.testing.assert_allclose(result.gamma, [1.01, 1.02, 1.04])
    np.testing.assert_allclose(result.omega, [-0.01, -0.02, -0.04])
    assert "initial_state" not in calls[0][1]
    assert calls[1][1]["initial_state"] is states[0]
    assert calls[2][1]["initial_state"] is states[1]


def test_runtime_parameter_scan_rejects_invalid_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_runtime_cfg()

    def identity(base, _value, _index):
        return base

    with pytest.raises(ValueError, match="parameter_name"):
        run_runtime_parameter_scan(
            cfg, [1.0], parameter_name=" ", update_config=identity
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        run_runtime_parameter_scan(
            cfg, [], parameter_name="beta", update_config=identity
        )
    with pytest.raises(TypeError, match="RuntimeConfig"):
        run_runtime_parameter_scan(
            cfg, [1.0], parameter_name="beta", update_config=lambda *_args: object()
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(
        runtime,
        "run_runtime_linear",
        lambda *_args, **_kwargs: RuntimeLinearResult(
            ky=0.3,
            gamma=0.0,
            omega=0.0,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
        ),
    )
    with pytest.raises(ValueError, match="return state"):
        run_runtime_parameter_scan(
            cfg,
            [1.0, 2.0],
            parameter_name="beta",
            update_config=identity,
            continuation=True,
        )


def test_runtime_linear_rejects_incompatible_initial_state_shape() -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    with pytest.raises(ValueError, match="initial_state shape"):
        run_runtime_linear(
            cfg,
            ky_target=0.3,
            Nl=2,
            Nm=2,
            solver="time",
            steps=1,
            initial_state=np.zeros((1, 2, 2, 1, 1, 1)),
        )


def test_runtime_parameter_scan_selects_one_candidate_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _base_runtime_cfg()
    calls: list[float] = []

    def fake_run(_cfg, **kwargs):
        target = float(kwargs["krylov_cfg"])
        calls.append(target)
        return RuntimeLinearResult(
            ky=0.3,
            gamma=target,
            omega=-target,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=np.asarray([target]),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    result = run_runtime_parameter_scan(
        cfg,
        [0.01, 0.02],
        parameter_name="beta",
        update_config=lambda base, _value, _index: base,
        candidate_options=lambda _value, _index, _previous: (
            {"krylov_cfg": 0.7},
            {"krylov_cfg": 1.5},
        ),
        select_candidate=lambda value, _index, _candidates, _previous: (
            0 if value < 0.015 else 1
        ),
        continuation=True,
    )

    assert calls == [0.7, 1.5, 0.7, 1.5]
    np.testing.assert_allclose(result.gamma, [0.7, 1.5])

    with pytest.raises(IndexError, match="out-of-range"):
        run_runtime_parameter_scan(
            cfg,
            [0.01],
            parameter_name="beta",
            update_config=lambda base, _value, _index: base,
            candidate_options=lambda *_args: ({"krylov_cfg": 0.7},),
            select_candidate=lambda *_args: 2,
        )


def test_runtime_explicit_time_routes_cfl_trajectory(monkeypatch) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        time=replace(
            _base_runtime_cfg().time,
            t_max=1.2,
            dt=0.01,
            fixed_dt=False,
            cfl=0.7,
        ),
    )
    captured: dict[str, object] = {}

    def fake_explicit(_state, grid, _geom, _params, time_cfg, **kwargs):
        captured.update(time_cfg=time_cfg, kwargs=kwargs)
        t = np.linspace(0.01, 1.2, 120)
        signal = np.exp((0.3 - 0.7j) * t)
        phi = signal[:, None, None, None] * np.ones(
            (1, 1, 1, int(grid.z.size)), dtype=complex
        )
        return t, phi

    monkeypatch.setattr(
        "gkx.workflows.linear.integrate_linear_explicit_from_config",
        fake_explicit,
    )
    result = run_runtime_linear(
        cfg,
        ky_target=0.3,
        Nl=2,
        Nm=2,
        solver="explicit_time",
        fit_signal="phi",
        mode_method="z_index",
        auto_window=False,
        tmin=0.2,
        tmax=1.0,
        require_positive=True,
    )

    assert result.gamma == pytest.approx(0.3, rel=1.0e-6)
    assert abs(result.omega) == pytest.approx(0.7, rel=1.0e-6)
    assert captured["time_cfg"].fixed_dt is False
    assert captured["kwargs"]["Nl"] == 2
    assert captured["kwargs"]["Nm"] == 2


def test_runtime_trajectory_can_be_refit_without_integration() -> None:
    t = np.linspace(0.01, 2.0, 200)
    z = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    mode = 1.0 + 0.2 * np.cos(z)
    history = np.exp((0.25 - 0.6j) * t)[:, None, None, None] * mode[None, None, None, :]
    result = RuntimeLinearResult(
        ky=0.3,
        gamma=0.0,
        omega=0.0,
        selection=ModeSelection(ky_index=0, kx_index=0, z_index=4),
        t=t,
        z=z,
        field_history=history,
    )

    refit = refit_runtime_linear_trajectory(
        result, mode_method="svd", auto_window=False, tmin=0.4, tmax=1.8
    )
    assert refit.gamma == pytest.approx(0.25, rel=1.0e-6)
    assert abs(refit.omega) == pytest.approx(0.6, rel=1.0e-6)
    assert refit.field_history is history
    assert refit.eigenfunction is not None

    with pytest.raises(ValueError, match="field_history"):
        refit_runtime_linear_trajectory(
            replace(result, field_history=None), mode_method="project"
        )


def test_run_runtime_scan_warm_start_seeds_neighbours_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warm scan walks ky in monotone order but reports the requested order."""

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    calls: list[dict[str, object]] = []

    def fake_run(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append({"ky": ky, "initial_state": kwargs.get("initial_state")})
        return RuntimeLinearResult(
            ky=ky,
            gamma=ky,
            omega=-ky,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=np.full((2, 2), ky, dtype=np.complex64),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    out = run_runtime_scan(
        cfg, ky_values=[0.5, 0.1, 0.3], solver="krylov", warm_start=True
    )

    assert [call["ky"] for call in calls] == [0.1, 0.3, 0.5]
    assert calls[0]["initial_state"] is None
    # Each later point is seeded from its neighbour, rescaled to unit norm.
    for call, source in zip(calls[1:], (0.1, 0.3)):
        seed = np.asarray(call["initial_state"])
        assert seed.shape == (2, 2)
        np.testing.assert_allclose(seed / np.linalg.norm(seed), 0.5 * np.ones((2, 2)))
        assert np.all(np.isclose(seed.real / source, seed.real[0, 0] / source))
    np.testing.assert_allclose(out.ky, [0.5, 0.1, 0.3])
    np.testing.assert_allclose(out.gamma, [0.5, 0.1, 0.3])
    np.testing.assert_allclose(out.omega, [-0.5, -0.1, -0.3])
    assert out.warm_start == {
        "enabled": True,
        "visit_order": [1, 2, 0],
        "warm_points": 2,
        "cold_points": 1,
    }


def test_run_runtime_scan_warm_start_off_leaves_every_point_cold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    calls: list[dict[str, object]] = []

    def fake_run(_cfg, **kwargs):
        calls.append(dict(kwargs))
        return RuntimeLinearResult(
            ky=float(kwargs["ky_target"]),
            gamma=1.0,
            omega=-1.0,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=np.ones((2, 2), dtype=np.complex64),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    # The default is cold; passing nothing must behave like warm_start=False.
    out = run_runtime_scan(cfg, ky_values=[0.5, 0.1], solver="krylov")

    assert [call["ky_target"] for call in calls] == [0.5, 0.1]
    assert all(call["initial_state"] is None for call in calls)
    assert all(call["return_state"] is False for call in calls)
    assert out.warm_start is None


def test_run_runtime_scan_warm_start_yields_to_independent_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request for real parallelism is not silently serialized by warm start."""

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion"),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
        parallel=RuntimeParallelConfig(strategy="batch", axis="ky", num_devices=2),
    )
    seen: list[object] = []

    def fake_run(_cfg, **kwargs):
        seen.append(kwargs.get("initial_state"))
        return RuntimeLinearResult(
            ky=float(kwargs["ky_target"]),
            gamma=1.0,
            omega=-1.0,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=np.ones((2, 2), dtype=np.complex64),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    out = run_runtime_scan(
        cfg, ky_values=[0.15, 0.35], solver="krylov", warm_start=True
    )

    assert out.warm_start is None
    assert seen == [None, None]
    assert out.parallel["effective_workers"] == 2


def test_run_runtime_scan_warm_start_matches_cold_growth_rates() -> None:
    """The correctness gate: warm and cold must report the same physics."""

    cfg = replace(
        _base_runtime_cfg(),
        species=(RuntimeSpeciesConfig(name="ion", tprim=3.0, fprim=1.0),),
        normalization=RuntimeNormalizationConfig(contract="cyclone"),
    )
    ky_values = [0.2, 0.3, 0.4]
    shared = dict(Nl=2, Nm=4, solver="krylov")

    cold = run_runtime_scan(cfg, ky_values=ky_values, warm_start=False, **shared)
    warm = run_runtime_scan(cfg, ky_values=ky_values, warm_start=True, **shared)

    assert warm.warm_start["warm_points"] == 2
    np.testing.assert_allclose(warm.ky, cold.ky)
    np.testing.assert_allclose(warm.gamma, cold.gamma, rtol=2.0e-3, atol=1.0e-6)
    np.testing.assert_allclose(warm.omega, cold.omega, rtol=2.0e-3, atol=1.0e-6)


def test_runtime_parameter_scan_warm_start_declines_a_large_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opportunistic reuse stops where the parameter stops being a neighbour."""

    cfg = replace(
        _base_runtime_cfg(),
        physics=RuntimePhysicsConfig(beta=0.0),
        species=(RuntimeSpeciesConfig(name="ion"),),
    )
    calls: list[dict[str, object]] = []

    def fake_run(point_cfg, **kwargs):
        calls.append(dict(kwargs))
        return RuntimeLinearResult(
            ky=0.3,
            gamma=float(point_cfg.physics.beta),
            omega=0.0,
            selection=ModeSelection(ky_index=0, kx_index=0, z_index=0),
            state=np.full((2,), float(point_cfg.physics.beta) + 1.0),
        )

    import gkx.runtime as runtime

    monkeypatch.setattr(runtime, "run_runtime_linear", fake_run)
    result = run_runtime_parameter_scan(
        cfg,
        [0.01, 0.011, 1.0],
        parameter_name="beta",
        update_config=lambda base, value, _index: replace(
            base, physics=replace(base.physics, beta=value)
        ),
        warm_start=True,
        warm_start_max_step=0.25,
    )

    assert "initial_state" not in calls[0]
    # 0.010 -> 0.011 is a 9% step, so the state carries.
    assert calls[1]["initial_state"] is not None
    # 0.011 -> 1.0 is a 99% step, so it does not.
    assert "initial_state" not in calls[2]
    assert all(call["return_state"] is True for call in calls)
    np.testing.assert_allclose(result.gamma, [0.01, 0.011, 1.0])
