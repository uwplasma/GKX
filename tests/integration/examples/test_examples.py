"""Fast integration checks for shipped example workflows."""

from __future__ import annotations

import ast
import contextlib
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from support.paths import REPO_ROOT, load_repo_script
from examples.theory_and_demos.autodiff_inverse_growth import (
    run_demo as run_inverse_growth_demo,
)
from examples.theory_and_demos.autodiff_inverse_twomode import (
    run_demo as run_twomode_demo,
)
from examples.theory_and_demos.quasilinear_implicit_sensitivity import (
    run_demo as run_implicit_sensitivity_demo,
)
from gkx.config import CycloneBaseCase, GridConfig, TimeConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry, sample_flux_tube_geometry
from gkx.operators.linear.params import LinearParams
from gkx.solvers_time_runners import (
    integrate_nonlinear_from_config,
)
from gkx.terms.config import TermConfig


_PARALLEL_EXAMPLE = (
    REPO_ROOT / "examples" / "parallelization" / "independent_ky_runtime_batch_scan.py"
)
_PARALLEL_CONFIG = (
    REPO_ROOT / "examples" / "parallelization" / "runtime_batch_ky_scan.toml"
)


def _load_parallel_example_module():
    return load_repo_script(
        _PARALLEL_EXAMPLE.relative_to(REPO_ROOT),
        module_name="independent_ky_runtime_batch_scan",
        write_bytecode=False,
    )


def test_autodiff_inverse_growth_demo_summary(tmp_path: Path) -> None:
    summary = run_inverse_growth_demo(
        outdir=tmp_path,
        steps=24,
        dt=0.05,
        ky_index=1,
        kx_index=0,
        z_index=0,
        tprim_true=2.2,
        fprim_true=0.8,
        tprim_init=1.8,
        fprim_init=1.1,
        gd_steps=4,
        gd_lr=0.5,
        plot=False,
        write_files=False,
    )
    assert max(summary["jac_rel_error"]) < 0.05
    assert summary["loss_final"] >= 0.0
    assert max(summary["observable_abs_error"]) < 5.0e-2
    cov = summary["covariance"]
    assert cov[0][0] > 0.0
    assert cov[1][1] > 0.0
    assert summary["sensitivity_map_rank"] >= 1
    assert summary["jacobian_condition_number"] > 0.0
    assert len(summary["covariance_std"]) == 2
    assert summary["uq_ellipse_area_1sigma"] >= 0.0


def test_autodiff_twomode_demo_summary(tmp_path: Path) -> None:
    summary = run_twomode_demo(
        outdir=tmp_path,
        steps=24,
        dt=0.05,
        ky_indices=(1, 3),
        kx_index=0,
        z_index=0,
        tprim_true=2.2,
        fprim_true=0.8,
        tprim_init=1.8,
        fprim_init=1.1,
        gd_steps=6,
        gd_lr=0.2,
        plot=False,
        write_files=False,
    )
    assert max(summary["jac_rel_error"]) < 0.05
    assert max(summary["parameter_abs_error"]) < 1.0e-2
    assert max(summary["observable_abs_error"]) < 1.0e-4
    cov = summary["covariance"]
    assert cov[0][0] > 0.0
    assert cov[1][1] > 0.0
    assert summary["sensitivity_map_rank"] == 2
    assert summary["jacobian_condition_number"] < 1.0e4
    assert len(summary["covariance_std"]) == 2
    assert summary["uq_ellipse_area_1sigma"] >= 0.0


def test_quasilinear_implicit_sensitivity_demo_summary(tmp_path: Path) -> None:
    summary = run_implicit_sensitivity_demo(
        outdir=tmp_path, plot=False, write_files=False
    )

    assert summary["passed"] is True
    assert summary["branch_isolated"] is True
    assert summary["sensitivity_method"] == "implicit_left_right_eigenpair"
    assert len(summary["observable_labels"]) == 5
    assert len(summary["parameter_labels"]) == 2
    jac_impl = np.asarray(summary["jacobian_implicit"], dtype=float)
    jac_fd = np.asarray(summary["jacobian_fd"], dtype=float)
    np.testing.assert_allclose(jac_impl, jac_fd, rtol=5.0e-2, atol=2.0e-3)


def test_runtime_batch_ky_scan_example_uses_independent_workers(
    monkeypatch,
) -> None:
    import gkx.runtime as runtime

    example = _load_parallel_example_module()
    calls: list[float] = []

    def _unexpected_combined_batch(*_args, **_kwargs):
        raise AssertionError(
            "strategy='batch' example must not use the combined-ky solver path"
        )

    def _fake_run_runtime_linear(_cfg, **kwargs):
        ky = float(kwargs["ky_target"])
        calls.append(ky)
        return SimpleNamespace(gamma=1.0 + ky, omega=-(2.0 + ky), quasilinear=None)

    monkeypatch.setattr(runtime, "_run_runtime_scan_batch", _unexpected_combined_batch)
    monkeypatch.setattr(runtime, "run_runtime_linear", _fake_run_runtime_linear)

    scan = example.run_example(_PARALLEL_CONFIG)

    np.testing.assert_allclose(sorted(calls), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.ky, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(scan.gamma, [1.1, 1.2, 1.3])
    np.testing.assert_allclose(scan.omega, [-2.1, -2.2, -2.3])
    assert scan.parallel is not None
    assert scan.parallel["source"] == "runtime_config"
    assert scan.parallel["requested_workers"] == 2
    assert scan.parallel["effective_workers"] == 2
    assert scan.parallel["executor"] == "thread"
    assert "independent ky workers" in scan.parallel["identity_contract"]


def test_example_smoke_nonlinear_scan() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2")
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    for seed in (0, 1):
        G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
        G = G.at[0, 0, seed, 0, :].set(1.0e-3 + 0.0j)
        _, fields_t = integrate_nonlinear_from_config(
            G,
            grid,
            geom,
            params,
            cfg.time,
            terms=terms,
        )
        assert fields_t.phi.shape[0] == 2


def test_example_smoke_nonlinear_scan_with_sampled_geometry() -> None:
    grid_cfg = GridConfig(Nx=1, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    time_cfg = TimeConfig(t_max=0.2, dt=0.1, method="rk2")
    cfg = CycloneBaseCase(grid=grid_cfg, time=time_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = sample_flux_tube_geometry(SAlphaGeometry.from_config(cfg.geometry), grid.z)
    params = LinearParams()
    terms = TermConfig(nonlinear=1.0)

    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    G = G.at[0, 0, 0, 0, :].set(1.0e-3 + 0.0j)
    _, fields_t = integrate_nonlinear_from_config(
        G,
        grid,
        geom,
        params,
        cfg.time,
        terms=terms,
    )

    assert fields_t.phi.shape[0] == 2


# ---------------------------------------------------------------------------
# Shipped-example execution coverage
#
# The Phase D exit gate asks that every canonical example execute in CI at
# smoke resolution. The examples come in two entrypoint shapes, and confusing
# them is how this coverage goes quietly vacuous: eleven files keep their run
# behind an ``if __name__ == "__main__"`` guard, so merely importing one
# executes no physics, while the other twenty-five run their case in the module
# body. Each example is therefore registered below with the name of the
# callable that does the work, or with ``None`` when the module body *is* the
# run. ``_run_example`` fails loudly when a registered entrypoint has gone
# missing instead of reporting a clean import as success, and
# ``test_every_shipped_example_is_classified`` fails when a new example is
# neither executed here nor skipped with a stated reason.
#
# Examples are turned down through the editable constants they already publish
# near the top of each file (Nl, Nm, dt, steps, t_max, CONFIG); no new smoke
# mechanism is introduced and no example file is edited.
# ---------------------------------------------------------------------------

_EXAMPLES_ROOT = REPO_ROOT / "examples"

# The bundled three-point ky deck: Nx=1, Ny=4, Nz=8, three RK2 steps. It is the
# smallest tracked linear runtime TOML, so it stands in for the CONFIG constant
# of examples whose own deck is too large to run inside a test.
_SMOKE_LINEAR_TOML = _EXAMPLES_ROOT / "parallelization" / "runtime_batch_ky_scan.toml"

# Editable-knob values used to bring a full-resolution deck down to a smoke run.
_SMOKE_NL = 2
_SMOKE_NM = 4
_SMOKE_STEPS = 20
_SMOKE_NONLINEAR_STEPS = 2
_SMOKE_COLLISION_TMAX = 0.05

# relative path -> entrypoint callable, or None when the module body is the run.
_EXAMPLE_ENTRYPOINTS: dict[str, str | None] = {
    "examples/linear/axisymmetric/cyclone_runtime_linear.py": None,
    "examples/linear/axisymmetric/etg_runtime_linear.py": None,
    "examples/linear/axisymmetric/kaw_runtime_linear.py": None,
    "examples/linear/axisymmetric/kbm_runtime_linear.py": None,
    "examples/nonlinear/axisymmetric/cyclone_runtime_nonlinear.py": None,
    "examples/nonlinear/axisymmetric/etg_runtime_nonlinear.py": None,
    "examples/nonlinear/axisymmetric/kbm_runtime_nonlinear.py": None,
    "examples/nonlinear/axisymmetric/miller_nonlinear_runtime.py": None,
    "examples/parallelization/independent_ky_runtime_batch_scan.py": "run_example",
    "examples/theory_and_demos/autodiff_inverse_growth.py": "run_demo",
    "examples/theory_and_demos/autodiff_inverse_twomode.py": "run_demo",
    "examples/theory_and_demos/collision_operator_comparison.py": None,
    "examples/theory_and_demos/cyclone_geometry.py": "main",
    "examples/theory_and_demos/example.py": "main",
    "examples/theory_and_demos/gradB_coupling_hl_1d.py": "main",
    "examples/theory_and_demos/linear_rhs_demo.py": "main",
    "examples/theory_and_demos/quasilinear_implicit_sensitivity.py": "run_demo",
    "examples/theory_and_demos/two_stream_hermite_1d.py": "main",
    "examples/utilities/plot_runtime_outputs.py": None,
    "examples/utilities/runtime_from_toml.py": None,
}

# Underscore-prefixed shared modules; they define helpers and run nothing.
_EXAMPLE_HELPERS = (
    "examples/theory_and_demos/reduced_stellarator_itg/_stellarator_itg_plotting.py",
    "examples/theory_and_demos/reduced_stellarator_itg/_stellarator_itg_workflow.py",
)

# Examples that cannot run in a bounded CI lane, each with the reason. These
# are reported as skips rather than left out of the registry, so the remaining
# gap is visible in the run summary instead of silent.
_EXAMPLE_SKIPS: dict[str, str] = {
    "examples/linear/non-axisymmetric/hsx_linear_imported_geometry.py": (
        "needs the imported flux-tube file hsx_linear.eik.nc, which is not "
        "tracked and is not produced by examples/vmec/generate_wouts.sh (that "
        "script writes wout_*.nc, not *.eik.nc)"
    ),
    "examples/linear/non-axisymmetric/w7x_linear_imported_geometry.py": (
        "needs the imported flux-tube file "
        "itg_w7x_adiabatic_electrons_t2.eik.nc, which is not tracked and is "
        "not produced by examples/vmec/generate_wouts.sh"
    ),
    "examples/nonlinear/non-axisymmetric/hsx_nonlinear_imported_geometry.py": (
        "needs the imported flux-tube file hsx_nonlinear.eik.nc, which is not "
        "tracked; the run itself is a ten-minute CPU nonlinear window"
    ),
    "examples/nonlinear/non-axisymmetric/w7x_nonlinear_imported_geometry.py": (
        "needs the imported flux-tube file w7x_adiabatic_electrons.eik.nc, "
        "which is not tracked; its build_w7x_nonlinear_cfg contract is covered "
        "by tests/integration/runtime/test_runtime_config.py"
    ),
    "examples/nonlinear/non-axisymmetric/hsx_nonlinear_vmec_geometry.py": (
        "needs examples/vmec/wout_NuhrenbergZille_1988_QHS.nc, which is "
        "generated by a full VMEC solve (examples/vmec/generate_wouts.sh, "
        "requires vmex); its cfg builder is covered by "
        "tests/integration/runtime/test_runtime_config.py"
    ),
    "examples/nonlinear/non-axisymmetric/w7x_nonlinear_vmec_geometry.py": (
        "needs examples/vmec/wout_nfp3_QI_fixed_resolution_final.nc from a "
        "full VMEC solve, then samples a 96x96x48 nonlinear flux tube"
    ),
    "examples/optimization/QA_optimization.py": (
        "its VMEX_EXAMPLES_CI=1 smoke path still resolves its seed deck as "
        "<vmex-install>/../examples/data/input.minimal_seed_nfp2, which exists "
        "only in a vmex source checkout, not in an installed vmex"
    ),
    "examples/theory_and_demos/differentiable_geometry_bridge.py": (
        "multi-minute vmex/booz_xform_jax gate campaign that overwrites the "
        "tracked docs/_static release artifacts; its import surface -- the "
        "regression that broke it on main -- is covered by "
        "test_example_first_party_imports_resolve"
    ),
    "examples/theory_and_demos/reduced_stellarator_itg/"
    "compare_stellarator_itg_optimizations.py": (
        "runs three Adam optimization campaigns back to back, about half an "
        "hour on a laptop CPU"
    ),
    "examples/theory_and_demos/reduced_stellarator_itg/"
    "stellarator_itg_growth_optimization.py": (
        "ten-minute Adam optimization campaign with AD/FD gates at every step"
    ),
    "examples/theory_and_demos/reduced_stellarator_itg/"
    "stellarator_itg_nonlinear_heat_flux_optimization.py": (
        "ten-minute Adam campaign over 520-step nonlinear heat-flux windows"
    ),
    "examples/theory_and_demos/reduced_stellarator_itg/"
    "stellarator_itg_portfolio_gate.py": (
        "audits AD against finite differences on 3 surfaces x 2 alphas x 3 ky "
        "for two objectives; minutes on a CPU and its sample grid is a "
        "module-level constant, so it cannot be turned down from outside"
    ),
    "examples/theory_and_demos/reduced_stellarator_itg/"
    "stellarator_itg_quasilinear_flux_optimization.py": (
        "ten-minute Adam optimization campaign"
    ),
    "examples/utilities/strong_scaling_sweep.py": (
        "sweeps 1/2/4/8 devices over 120 RK2 steps on a 128x256 grid and "
        "needs XLA_FLAGS=--xla_force_host_platform_device_count=8, which has "
        "to be set before JAX initialises"
    ),
}


@pytest.fixture(autouse=True, scope="module")
def _headless_matplotlib():
    """Keep example plotting off an interactive backend inside the suite."""

    import matplotlib

    previous = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            matplotlib.use(previous, force=True)


def _shipped_example_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in _EXAMPLES_ROOT.rglob("*.py")
        )
    )


def _run_example(relative: str, *args, **kwargs):
    """Execute a shipped example the way its own entrypoint shape requires."""

    entrypoint = _EXAMPLE_ENTRYPOINTS[relative]
    module = load_repo_script(
        relative,
        module_name=f"shipped_example_{Path(relative).stem}",
        write_bytecode=False,
    )
    if entrypoint is None:
        assert not args and not kwargs, (
            f"{relative} runs in its module body and takes no call arguments"
        )
        return module
    call = getattr(module, entrypoint, None)
    if not callable(call):
        raise AssertionError(
            f"{relative} is registered with entrypoint {entrypoint!r} but "
            f"defines no such callable (module defines "
            f"{sorted(n for n in vars(module) if not n.startswith('_'))}). "
            "Importing it alone runs no physics, so treating the import as "
            "success would pass vacuously; restore the entrypoint or update "
            "_EXAMPLE_ENTRYPOINTS."
        )
    call(*args, **kwargs)
    return module


def _smoke_linear_case(captured: list, *, dt: float):
    """Stand in for ``gkx.run_linear_case`` at the example's own smoke knobs."""

    from gkx.workflows.runtime import commands as runtime_commands

    def _run(config_path, **_editable_knobs):
        deps = runtime_commands.default_runtime_case_deps()

        def _capture(cfg, **run_kwargs):
            result = deps.run_runtime_linear(cfg, **run_kwargs)
            captured.append(result)
            return result

        return runtime_commands.run_linear_case(
            config_path,
            Nl=_SMOKE_NL,
            Nm=_SMOKE_NM,
            solver="explicit_time",
            dt=dt,
            steps=_SMOKE_STEPS,
            # SAMPLE_STRIDE is one of the example's own knobs; the ETG deck
            # stores every 25th step, which would leave a 20-step smoke run
            # with a single sample and nothing to fit.
            sample_stride=1,
            show_progress=False,
            deps=replace(deps, run_runtime_linear=_capture),
        )

    return _run


def _smoke_nonlinear_case(record: dict, *, dt: float, out_root: Path):
    """Stand in for ``gkx.run_nonlinear_case`` at the example's smoke knobs."""

    from gkx.workflows.runtime import commands as runtime_commands

    def _run(config_path, **_editable_knobs):
        deps = runtime_commands.default_runtime_case_deps()

        def _load(path):
            cfg, raw = deps.load_runtime_from_toml(path)
            if cfg.output.path:
                # The ETG deck writes into the checkout's tools_out/; keep the
                # artifacts the example asks for, but under the test's tmp dir.
                cfg = replace(
                    cfg,
                    output=replace(
                        cfg.output,
                        path=str(out_root / Path(cfg.output.path).name),
                    ),
                )
            return cfg, raw

        def _capture(cfg, **run_kwargs):
            result = deps.run_runtime_nonlinear(cfg, **run_kwargs)
            record["results"].append(result)
            return result

        def _capture_with_artifacts(cfg, **run_kwargs):
            result, paths = deps.run_runtime_nonlinear_with_artifacts(cfg, **run_kwargs)
            record["results"].append(result)
            record["paths"].append(paths)
            return result, paths

        return runtime_commands.run_nonlinear_case(
            config_path,
            Nl=_SMOKE_NL,
            Nm=_SMOKE_NM,
            dt=dt,
            steps=_SMOKE_NONLINEAR_STEPS,
            show_progress=False,
            deps=replace(
                deps,
                load_runtime_from_toml=_load,
                run_runtime_nonlinear=_capture,
                run_runtime_nonlinear_with_artifacts=_capture_with_artifacts,
            ),
        )

    return _run


@pytest.mark.parametrize(
    "relative, dt, expected_ky",
    [
        ("examples/linear/axisymmetric/cyclone_runtime_linear.py", 0.01, 0.3),
        ("examples/linear/axisymmetric/etg_runtime_linear.py", 0.001, 15.0),
        ("examples/linear/axisymmetric/kaw_runtime_linear.py", 0.01, 0.01),
        ("examples/linear/axisymmetric/kbm_runtime_linear.py", 0.001, 0.3),
    ],
)
def test_runtime_linear_examples_fit_a_finite_mode(
    monkeypatch, relative: str, dt: float, expected_ky: float
) -> None:
    import gkx

    captured: list = []
    monkeypatch.setattr(gkx, "run_linear_case", _smoke_linear_case(captured, dt=dt))

    _run_example(relative)

    assert len(captured) == 1, f"{relative} did not run its linear case exactly once"
    result = captured[0]
    assert abs(float(result.ky)) == pytest.approx(expected_ky, rel=1.0e-6)
    assert np.isfinite(result.gamma)
    assert np.isfinite(result.omega)
    times = np.asarray(result.t, dtype=float)
    signal = np.asarray(result.signal)
    assert times.size >= 2
    assert times.size == signal.shape[0]
    assert np.all(np.isfinite(signal))
    assert np.abs(signal).max() > 0.0
    summary = result.summary()
    assert summary["ky"] == pytest.approx(float(result.ky))
    assert summary["gamma"] == pytest.approx(float(result.gamma))


@pytest.mark.parametrize(
    "relative, dt",
    [
        ("examples/nonlinear/axisymmetric/cyclone_runtime_nonlinear.py", 0.01),
        ("examples/nonlinear/axisymmetric/etg_runtime_nonlinear.py", 0.01),
        ("examples/nonlinear/axisymmetric/kbm_runtime_nonlinear.py", 0.01),
        ("examples/nonlinear/axisymmetric/miller_nonlinear_runtime.py", 0.01),
    ],
)
def test_runtime_nonlinear_examples_stream_finite_diagnostics(
    monkeypatch, tmp_path: Path, relative: str, dt: float
) -> None:
    import gkx

    record: dict = {"results": [], "paths": []}
    monkeypatch.setattr(
        gkx,
        "run_nonlinear_case",
        _smoke_nonlinear_case(record, dt=dt, out_root=tmp_path),
    )

    _run_example(relative)

    assert len(record["results"]) == 1, f"{relative} did not run its case once"
    result = record["results"][0]
    for paths in record["paths"]:
        # A deck with an [output] block must leave a readable bundle behind.
        assert Path(paths["summary"]).is_file()
    assert result.ky_selected is not None
    diagnostics = result.diagnostics
    assert diagnostics is not None, f"{relative} produced no streamed diagnostics"
    for name in ("Wg_t", "Wphi_t", "heat_flux_t", "particle_flux_t"):
        trace = np.asarray(getattr(diagnostics, name), dtype=float)
        assert trace.size >= 1
        assert np.all(np.isfinite(trace)), f"{relative}: {name} went non-finite"
    # Free energy and the electrostatic field energy are positive definite, so
    # a zero here means the nonlinear step never moved the initial condition.
    assert float(np.asarray(diagnostics.Wg_t)[-1]) > 0.0
    assert float(np.asarray(diagnostics.Wphi_t)[-1]) > 0.0


@pytest.mark.parametrize(
    "relative, owner, attribute",
    [
        (
            "examples/theory_and_demos/example.py",
            "gkx",
            "integrate_linear_from_config",
        ),
        (
            "examples/theory_and_demos/gradB_coupling_hl_1d.py",
            "gkx.solvers_linear_integrators",
            "integrate_linear",
        ),
        (
            "examples/theory_and_demos/linear_rhs_demo.py",
            "gkx.solvers_linear_integrators",
            "integrate_linear",
        ),
        (
            "examples/theory_and_demos/two_stream_hermite_1d.py",
            "gkx.solvers_linear_integrators",
            "integrate_linear",
        ),
    ],
)
def test_linear_demo_examples_drive_the_real_solver(
    monkeypatch, relative: str, owner: str, attribute: str
) -> None:
    import importlib

    module = importlib.import_module(owner)
    real = getattr(module, attribute)
    calls: list = []

    def _record(*args, **kwargs):
        result = real(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(module, attribute, _record)

    _run_example(relative)

    assert len(calls) == 1, f"{relative} did not integrate exactly once"
    state, phi_t = calls[0]
    state = np.asarray(state)
    phi_t = np.asarray(phi_t)
    assert phi_t.ndim == 4
    assert phi_t.shape[0] >= 1
    assert np.all(np.isfinite(phi_t))
    assert np.all(np.isfinite(state))
    # phi itself is identically zero for the ky=0 slab demos (adiabatic
    # quasineutrality), so the distribution is what proves the run advanced.
    norm = float(np.linalg.norm(state))
    assert norm > 0.0, f"{relative} never moved its initial condition"


def test_cyclone_geometry_example_reports_a_varying_kperp2(capsys) -> None:
    _run_example("examples/theory_and_demos/cyclone_geometry.py")

    printed = capsys.readouterr().out
    theta_line = next(
        line for line in printed.splitlines() if line.startswith("theta range")
    )
    extent_line = next(line for line in printed.splitlines() if line.startswith("min="))
    lower, upper = (
        float(value) for value in theta_line.split("[")[1].rstrip("]").split(",")
    )
    kperp2_min = float(extent_line.split("min=")[1].split()[0])
    kperp2_max = float(extent_line.split("max=")[1])
    # nperiod=2 ballooning coordinate: the extended tube covers more than a
    # single 2*pi poloidal turn, and s_hat>0 makes k_perp^2 grow along it.
    assert upper - lower > 2.0 * np.pi
    assert 0.0 < kperp2_min < kperp2_max


def test_collision_operator_comparison_example_tabulates_every_model(
    monkeypatch,
) -> None:
    import gkx.solvers_time_runners as time_runners
    from gkx.operators.linear.params import COLLISION_OPERATOR_NAMES

    real = time_runners.integrate_linear_from_config

    def _short_window(state, grid, geometry, parameters, time_config, *args, **kwargs):
        # ``t_max`` is the example's own editable horizon; everything else --
        # the operators, the moment truncation, the grid -- is untouched.
        return real(
            state,
            grid,
            geometry,
            parameters,
            replace(time_config, t_max=_SMOKE_COLLISION_TMAX),
            *args,
            **kwargs,
        )

    monkeypatch.setattr(time_runners, "integrate_linear_from_config", _short_window)

    module = _run_example("examples/theory_and_demos/collision_operator_comparison.py")

    expected = {name for name in COLLISION_OPERATOR_NAMES if name != "none"}
    rows = module.rows
    assert {row["collision_operator"] for row in rows} == expected
    for row in rows:
        assert np.isfinite(row["growth_rate"]), row["collision_operator"]
        assert row["final_state_norm"] > 0.0, row["collision_operator"]
        assert row["nu"] == pytest.approx(module.NU)


def test_runtime_from_toml_example_solves_every_scan_point(monkeypatch) -> None:
    import gkx.workflows.runtime.toml as runtime_toml

    real_load = runtime_toml.load_runtime_from_toml

    def _smoke_config(_config_path):
        # CONFIG is the example's documented knob ("point CONFIG at any linear
        # runtime TOML"); the bundled three-point deck is the smallest one.
        return real_load(_SMOKE_LINEAR_TOML)

    monkeypatch.setattr(runtime_toml, "load_runtime_from_toml", _smoke_config)

    module = _run_example("examples/utilities/runtime_from_toml.py")

    scan = module.scan
    np.testing.assert_allclose(scan.ky, [0.1, 0.2, 0.3])
    assert np.all(np.isfinite(scan.gamma))
    assert np.all(np.isfinite(scan.omega))


def test_independent_ky_batch_scan_example_solves_every_point_for_real() -> None:
    # The mocked test above pins that the batch strategy never reaches the
    # combined-ky solver; this one pins that the shipped deck actually runs.
    module = load_repo_script(
        _PARALLEL_EXAMPLE.relative_to(REPO_ROOT),
        module_name="shipped_example_independent_ky_runtime_batch_scan",
        write_bytecode=False,
    )
    scan = module.run_example()

    np.testing.assert_allclose(scan.ky, [0.1, 0.2, 0.3])
    assert np.all(np.isfinite(scan.gamma))
    assert np.all(np.isfinite(scan.omega))
    assert scan.parallel["strategy"] == "batch"
    assert scan.parallel["effective_workers"] == 2


def test_plot_runtime_outputs_example_writes_a_figure(tmp_path, monkeypatch) -> None:
    from gkx.runtime import run_runtime_linear
    from gkx.workflows.runtime.artifacts import write_runtime_linear_artifacts
    from gkx.workflows.runtime.commands import RUNTIME_CASE_FIT_KEYS
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    cfg, raw = load_runtime_from_toml(_SMOKE_LINEAR_TOML)
    # This is a single solve, not the deck's ky scan, so the scan-level batch
    # strategy does not apply.
    cfg = replace(cfg, parallel=replace(cfg.parallel, strategy="serial"))
    scan_cfg = raw["scan"]
    result = run_runtime_linear(
        cfg,
        ky_target=0.2,
        Nl=int(scan_cfg["Nl"]),
        Nm=int(scan_cfg["Nm"]),
        solver=str(scan_cfg["solver"]),
        method=scan_cfg["method"],
        steps=int(scan_cfg["steps"]),
        sample_stride=int(scan_cfg["sample_stride"]),
        show_progress=False,
        **{k: v for k, v in raw["fit"].items() if k in RUNTIME_CASE_FIT_KEYS},
    )
    # The example's RUN_PATH is relative, so the completed run has to live at
    # tools_out/cyclone_nonlinear_runtime under the working directory.
    bundle = tmp_path / "tools_out" / "cyclone_nonlinear_runtime"
    written = write_runtime_linear_artifacts(bundle, result)
    assert "summary" in written
    monkeypatch.chdir(tmp_path)

    module = _run_example("examples/utilities/plot_runtime_outputs.py")

    figure = Path(module.out)
    assert figure.is_file()
    assert figure.suffix == ".png"
    assert figure.stat().st_size > 1024


def test_every_shipped_example_is_classified() -> None:
    classified = set(_EXAMPLE_ENTRYPOINTS) | set(_EXAMPLE_SKIPS) | set(_EXAMPLE_HELPERS)
    shipped = set(_shipped_example_paths())
    assert shipped - classified == set(), (
        "new examples are neither executed nor skipped with a reason; add them "
        "to _EXAMPLE_ENTRYPOINTS or _EXAMPLE_SKIPS"
    )
    assert classified - shipped == set(), "registry names examples that no longer exist"


@pytest.mark.parametrize("relative", sorted(_EXAMPLE_ENTRYPOINTS))
def test_executed_example_keeps_its_registered_entrypoint_shape(relative: str) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text(encoding="utf-8"))
    guarded = any(
        isinstance(node, ast.If) and "__main__" in ast.dump(node.test)
        for node in tree.body
    )
    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    entrypoint = _EXAMPLE_ENTRYPOINTS[relative]
    if entrypoint is None:
        assert not guarded, (
            f"{relative} is registered as a module-body script but now hides "
            "its run behind a __main__ guard: importing it would execute no "
            "physics and its test would pass vacuously"
        )
    else:
        assert entrypoint in defined, (
            f"{relative} no longer defines {entrypoint!r}; the test that runs "
            "it would import the module and assert nothing"
        )


@pytest.mark.parametrize("relative", _shipped_example_paths())
def test_example_first_party_imports_resolve(relative: str) -> None:
    """Every ``gkx``/``tools``/sibling name an example imports must exist.

    ``differentiable_geometry_bridge.py`` sat broken on main for days because
    it imported six names that had been deleted, and nothing executed it. This
    is the static half of that guard, and it covers the examples that are too
    expensive to run as well as the ones that are not.
    """

    import importlib

    path = REPO_ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    example_dir = str(path.parent)
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    checked = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in {"gkx", "tools"}:
                    importlib.import_module(alias.name)
                    checked += 1
            continue
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        owner = node.module or ""
        root = owner.split(".")[0]
        if root not in {"gkx", "tools"} and not root.startswith("_stellarator_itg"):
            continue
        module = importlib.import_module(owner)
        for alias in node.names:
            assert hasattr(module, alias.name), (
                f"{relative} imports {alias.name!r} from {owner}, which no "
                "longer provides it"
            )
            checked += 1
    assert checked > 0, (
        f"{relative} resolved no first-party import; the check would pass "
        "without looking at anything"
    )


@pytest.mark.parametrize("relative", sorted(_EXAMPLE_SKIPS))
def test_example_needs_data_or_a_long_run(relative: str) -> None:
    assert (REPO_ROOT / relative).is_file()
    pytest.skip(f"{relative}: {_EXAMPLE_SKIPS[relative]}")
