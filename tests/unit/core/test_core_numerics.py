from __future__ import annotations

from support.helpers import patch_runtime

import io
import subprocess
import sys
from contextlib import redirect_stdout

import jax.numpy as jnp
import numpy as np
import inspect
import pytest

import gkx.config as public_config
import gkx.objectives.vmec_transport as public_vmec_transport
import gkx.operators.linear as public_linear_operators
import gkx.operators.nonlinear.policies as public_nonlinear_policies
from gkx.config import (
    CycloneBaseCase,
    REFERENCE_ELECTRON_MASS,
    GeometryConfig,
    GridConfig,
    KBMBaseCase,
    ModelConfig,
    TimeConfig,
    explicit_method_default_cfl_fac,
    resolve_cfl_fac,
)
from gkx.operators.collision import (
    CollisionContext,
    CollisionOperator,
    SplitCollisionOperator,
)
from gkx.core_grid import (
    SpectralGrid,
    _gyrokinetic_moment_shape,
    build_spectral_grid,
    real_fft_ordered_kx,
    real_fft_unique_ky,
    select_ky_grid,
    select_real_fft_ky_grid,
)
from gkx.core_ky_layout import rows_for_layout
from gkx.callbacks import (
    _PROGRESS_START,
    _emit_progress,
    _format_duration,
    print_callback,
    progress_update_stride,
    should_emit_progress,
)
import gkx
from gkx.config import RuntimeConfig
from gkx.workflows.runtime.results import (
    RuntimeLinearResult,
    RuntimeLinearScanResult,
    RuntimeNonlinearResult,
)
from gkx.diagnostics.modes import ModeSelection
from gkx.api.prepared import PreparedSimulation, prepare_simulation
import jax
from gkx.core_ky_layout import (
    FULL,
    HALF,
    conjugate_kx_order,
    describe,
    half_dealias_mask,
    half_ky_values,
    ky_row_weights,
    negative_ky_block,
    ny_full_candidates,
    nyc_from_ny,
    nyquist_row,
    paired_row_limit,
    self_conjugate_rows,
    source_ky_layout,
    source_ny_full,
    symmetrize_self_conjugate_rows,
    to_full,
    to_half,
)
from scripts.checks._gates.validation_gates import (
    reality_residual,
)
from gkx.core_grid import twothirds_mask
from gkx.core_ky_layout import (
    is_half,
    ky_layout_of,
)
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import TermConfig


@pytest.mark.parametrize(
    ("facade", "required"),
    [
        (public_linear_operators, {"LinearCache", "build_H", "linear_rhs_cached"}),
        (
            public_nonlinear_policies,
            {"NonlinearTimeStepPolicy", "build_nonlinear_time_step_policy"},
        ),
        (
            public_vmec_transport,
            {
                "VMEXTransportObjectiveConfig",
                "vmex_transport_objective_from_state",
            },
        ),
    ],
    ids=(
        "linear-operators",
        "nonlinear-policies",
        "vmec-transport",
    ),
)
def test_solver_facade_exposes_only_supported_public_names(facade, required) -> None:
    assert facade.__all__
    assert not any(name.startswith("_") for name in facade.__all__)
    assert required <= set(facade.__all__)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((2, 3, 4, 5, 6), (2, 3)),
        ((7, 2, 3, 4, 5, 6), (2, 3)),
    ],
)
def test_gyrokinetic_moment_shape_handles_optional_species_axis(
    shape: tuple[int, ...], expected: tuple[int, int]
) -> None:
    assert _gyrokinetic_moment_shape(jnp.empty(shape)) == expected


def test_gyrokinetic_moment_shape_preserves_caller_label() -> None:
    with pytest.raises(ValueError, match="G must have shape"):
        _gyrokinetic_moment_shape(jnp.empty((2, 3, 4, 5)), name="G")


def test_collision_protocols_accept_structural_implementations() -> None:
    class ToyCollision:
        def apply(self, context):
            return context.distribution

    class ToySplitCollision(ToyCollision):
        def split_step(self, context, dt):
            return context.distribution

    context = CollisionContext(
        distribution=jnp.ones(1),
        hamiltonian=2.0 * jnp.ones(1),
        fields={"phi": jnp.zeros(1)},
        cache={"Jl": jnp.ones(1)},
        parameters={"nu": 0.1},
    )
    assert jnp.allclose(ToyCollision().apply(context), context.distribution)
    assert isinstance(ToyCollision(), CollisionOperator)
    assert isinstance(ToySplitCollision(), SplitCollisionOperator)
    assert not isinstance(ToyCollision(), SplitCollisionOperator)


def test_linear_terms_import_has_no_facade_order_dependency() -> None:
    """Low-level term imports must work in a fresh interpreter."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from gkx.operators.linear.dissipation import "
                "collisions_contribution; "
                "from gkx.terms.linear_terms import "
                "conservative_full_f_dougherty_cross_moments; "
                "from gkx.operators.linear import linear_rhs"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nonlinear_operator_facade_resolves_lazy_public_exports() -> None:
    """The public facade must not require eager RHS assembly at import time."""

    from gkx.operators import nonlinear

    assert nonlinear.NonlinearDiagnosticKernels.__module__.endswith("diagnostic_state")
    assert callable(nonlinear.compute_nonlinear_diagnostic_tuple)
    assert callable(nonlinear.make_nonlinear_diagnostic_tuple_fn)
    assert callable(nonlinear.linear_rhs_jit_for_terms_impl)
    assert callable(nonlinear.nonlinear_em_term_cached_impl)
    assert callable(nonlinear.nonlinear_rhs_cached_impl)


def test_velocity_basis_orthonormality_and_validation() -> None:
    from gkx.core_velocity import hermite_ladder_coeffs, laguerre

    xl = jnp.linspace(0.0, 40.0, 8001)
    dxl = xl[1] - xl[0]
    lag = laguerre(xl, 4)
    wl = jnp.exp(-xl)
    gram_l = jnp.einsum("ix,jx,x->ij", lag, lag, wl) * dxl
    assert jnp.allclose(gram_l, jnp.eye(5), atol=2e-2)

    with pytest.raises(ValueError):
        laguerre(jnp.array([0.0]), -1)
    with pytest.raises(ValueError):
        hermite_ladder_coeffs(-1)

    l0 = laguerre(jnp.array([0.0, 1.0]), 0)
    assert l0.shape == (1, 2)


@pytest.mark.parametrize("nl", [4, 8, 16, 32, 64])
def test_laguerre_transform_roundtrip_is_exact_at_high_resolution(nl: int) -> None:
    """The Laguerre pair must invert to machine precision at every resolution.

    This guards a failure that was silent: the earlier construction stored the
    unweighted ``L_ell(x_j)``, which reaches 1e18 at the outermost Gauss node,
    and paired it with Golub-Welsch eigenvector weights whose true value
    ``~exp(-x_j) ~ 1e-54`` sits far below the eigenvector's own accuracy. The
    round-trip degraded from 1e-12 at ``nl=12`` to 1e-3 at 20 and 1e14 at 32,
    with no error and no obvious symptom in the physics, so any run at
    ``Nl >= 20`` was quietly wrong.
    """
    from gkx.core_velocity import laguerre_quadrature_count, laguerre_transform

    to_grid, to_spectral, roots = laguerre_transform(nl)
    nj = laguerre_quadrature_count(nl)
    assert to_grid.shape == (nl, nj)
    assert to_spectral.shape == (nj, nl)

    identity_error = np.abs(to_grid @ to_spectral - np.eye(nl)).max()
    assert identity_error < 1.0e-10, f"nl={nl}: round-trip error {identity_error:.3e}"

    # Bounded entries and conditioning are what make the round trip survivable
    # in float32 too; the unweighted form failed both by two dozen orders.
    assert np.abs(to_grid).max() <= 1.0
    assert np.linalg.cond(to_grid) < 50.0

    to_grid32 = to_grid.astype(np.float32)
    to_spectral32 = to_spectral.astype(np.float32)
    error32 = np.abs(to_grid32 @ to_spectral32 - np.eye(nl, dtype=np.float32)).max()
    assert error32 < 1.0e-5, f"nl={nl}: float32 round-trip error {error32:.3e}"

    # Independent check on the analytic weights: an nj-point Gauss-Laguerre
    # rule integrates x^k exactly against exp(-x) for k <= 2*nj - 1, and
    # int_0^inf x^k exp(-x) dx = k!. to_spectral[j, 0] is exp(-x_j/2) * W_j
    # with W_j = w_j exp(x_j), so w_j = to_spectral[j, 0] * exp(-x_j/2).
    weights = to_spectral[:, 0] * np.exp(-0.5 * roots)
    factorial = 1.0
    for k in range(min(2 * nj, 12)):
        if k > 0:
            factorial *= k
        assert np.isclose(np.sum(weights * roots**k), factorial, rtol=1.0e-10)


def test_laguerre_transform_sign_convention_is_unchanged() -> None:
    """``to_grid[ell, j]`` must stay ``(-1)**ell * exp(-x_j/2) * L_ell(x_j)``.

    The shipped collision tables are keyed to this sign (the
    ``laguerre_convention`` fields under ``gkx/data``), so a flip here would
    corrupt them silently rather than fail loudly.
    """
    from gkx.core_velocity import laguerre_transform

    nl = 12
    to_grid, to_spectral, roots = laguerre_transform(nl)
    reference = np.polynomial.laguerre.lagval(roots, np.eye(nl), tensor=True)
    expected = ((-1.0) ** np.arange(nl))[:, None] * np.exp(-0.5 * roots) * reference
    assert np.abs(to_grid - expected).max() < 1.0e-12

    # to_spectral must carry the identical sign, differing from to_grid only
    # by the per-node weight. That is what makes the projector itself
    # convention-independent, so downstream tables see no change.
    weights = to_spectral[:, 0] / to_grid[0, :]
    assert np.allclose(to_spectral.T, to_grid * weights[None, :], rtol=1.0e-12)


@pytest.mark.parametrize("scale", [1.01, np.nan, np.inf])
def test_laguerre_transform_rejects_a_degraded_pair(scale: float) -> None:
    """The round-trip guard must fire rather than return an unusable transform."""
    import gkx.core_velocity as velocity

    to_grid, to_spectral, _ = velocity.laguerre_transform(8)
    with pytest.raises(ValueError, match="round-trip identity error"):
        velocity._check_laguerre_transform_conditioning(
            to_grid, to_spectral * scale, to_grid.shape[0]
        )


def test_species_builder_core_contracts() -> None:
    from gkx.operators.linear.params import Species, build_linear_params

    ion = Species(
        charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0
    )
    ele = Species(
        charge=-1.0, mass=0.001, density=1.0, temperature=1.0, tprim=2.0, fprim=1.0
    )
    params = build_linear_params([ion, ele], beta=1.0e-4, fapar=1.0)
    assert params.charge_sign.shape == (2,)
    assert params.vth.shape == (2,)
    assert np.isclose(params.vth[0], 1.0)
    assert np.isclose(params.tz[1], -1.0)
    assert params.beta == 1.0e-4


def test_normalization_and_benchmark_public_contracts() -> None:
    import gkx.benchmarking_shared as benchmark_defaults
    import gkx.benchmarking_shared as benchmarks
    from gkx.diagnostics.normalization import (
        apply_diagnostic_normalization,
        get_normalization_contract,
    )

    kin = get_normalization_contract("kinetic")
    assert kin == get_normalization_contract("kinetic_itg")
    with pytest.raises(ValueError, match="Unknown normalization case"):
        get_normalization_contract("not-a-case")

    gamma, omega = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="rho_star"
    )
    assert gamma == pytest.approx(0.1)
    assert omega == pytest.approx(-0.15)
    gamma_none, omega_none = apply_diagnostic_normalization(
        0.2, -0.3, rho_star=0.5, diagnostic_norm="none"
    )
    assert gamma_none == pytest.approx(0.2)
    assert omega_none == pytest.approx(-0.3)

    cyclone = get_normalization_contract("cyclone")
    etg = get_normalization_contract("etg")
    kinetic = get_normalization_contract("kinetic")
    kbm = get_normalization_contract("kbm")
    assert benchmarks.CYCLONE_OMEGA_D_SCALE == pytest.approx(cyclone.omega_d_scale)
    assert benchmarks.CYCLONE_OMEGA_STAR_SCALE == pytest.approx(
        cyclone.omega_star_scale
    )
    assert benchmarks.CYCLONE_RHO_STAR == pytest.approx(cyclone.rho_star)
    assert benchmarks.ETG_OMEGA_D_SCALE == pytest.approx(etg.omega_d_scale)
    assert benchmarks.ETG_OMEGA_STAR_SCALE == pytest.approx(etg.omega_star_scale)
    assert benchmarks.ETG_RHO_STAR == pytest.approx(etg.rho_star)
    assert benchmarks.KINETIC_OMEGA_D_SCALE == pytest.approx(kinetic.omega_d_scale)
    assert benchmarks.KINETIC_OMEGA_STAR_SCALE == pytest.approx(
        kinetic.omega_star_scale
    )
    assert benchmarks.KINETIC_RHO_STAR == pytest.approx(kinetic.rho_star)
    assert benchmarks.KBM_OMEGA_D_SCALE == pytest.approx(kbm.omega_d_scale)
    assert benchmarks.KBM_OMEGA_STAR_SCALE == pytest.approx(kbm.omega_star_scale)
    assert benchmarks.KBM_RHO_STAR == pytest.approx(kbm.rho_star)

    for name in benchmark_defaults.__all__:
        assert getattr(benchmarks, name) is getattr(benchmark_defaults, name)
    assert benchmarks.KINETIC_KRYLOV_REFERENCE_ALIGNED.shift_source == "history"
    assert benchmarks.KINETIC_KRYLOV_DEFAULT.shift_source == "target"
    assert benchmarks.KBM_KRYLOV_DEFAULT.mode_family == "kbm"
    assert benchmarks.KBM_KRYLOV_DEFAULT.omega_sign == 1
    assert benchmarks.ETG_KRYLOV_DEFAULT.omega_sign == -1


def test_public_api_facades_and_lazy_import_contracts() -> None:
    import subprocess
    import sys

    import gkx
    import gkx.api as public_api
    from support.paths import REPO_ROOT

    promoted = [
        "load",
        "solve",
        "scan",
        "plot",
        "prepare",
        "PreparedSimulation",
        "Case",
        "LinearResult",
        "NonlinearResult",
        "ScanResult",
        "flux_tube_geometry_from_mapping",
        "solver_objective_vector_from_geometry",
        "solver_linear_operator_matrix_from_geometry",
        "solver_scalar_objective_from_vector",
        "VMEXTransportObjectiveConfig",
    ]
    assert public_api.__all__ == promoted
    assert gkx.__all__ == ["__version__", *promoted]
    # 353, not 352: PreparedSimulation joins the advertised names, and every
    # advertised name is also a lazy target -- the registry is the loading
    # mechanism for the whole surface, not only the compatibility tail.
    # 349: evicting the parity and sensitivity report builders to
    # scripts/campaigns removed their four advertised names from the registry.
    # 260: ARCH-A contraction 1 deleted the report, gate and prototype modules
    # 86 compatibility names pointed at.
    # 189: ARCH-C moved the gate, window and calibration report modules to
    # scripts/checks/_gates, removing their 28 compatibility names.
    assert len(public_api._EXPORT_TARGETS) == 189
    assert len(public_api.__all__) == len(set(public_api.__all__))
    assert set(gkx.__all__) <= set(dir(gkx))
    # Laziness itself is asserted in the fresh interpreters below, not here:
    # `__getattr__` caches every name it resolves into the module namespace, so
    # in this process `dir(gkx)` reports which earlier tests happened to touch
    # which attribute -- `tests/release` alone imports `LinearParams` off the
    # root package. What is order-independent here is that the compatibility
    # names load through the registry without being advertised.
    assert "LinearParams" in public_api._EXPORT_TARGETS
    wildcard: dict[str, object] = {}
    exec("from gkx import *", wildcard)
    assert set(wildcard) - {"__builtins__"} == set(gkx.__all__)
    from gkx.workflows.runtime.results import plot as plot_owner

    assert gkx.plot is public_api.plot is plot_owner
    assert gkx.prepare is public_api.prepare
    assert gkx.ExplicitTimeConfig.__name__ == "ExplicitTimeConfig"
    assert callable(gkx.integrate_nonlinear_explicit_diagnostics)
    for legacy in (
        "GridConfig",
        "LinearParams",
        "integrate_nonlinear",
        "batch_map",
        "ModeSelection",
        "growth_fit_figure",
        "VMEXGKXTransportObjective",
    ):
        assert legacy not in gkx.__all__
        assert getattr(gkx, legacy) is getattr(public_api, legacy)
    assert "LinearExplicitTimeConfig" not in gkx.__all__
    assert "integrate_nonlinear_diagnostics" not in gkx.__all__

    root_script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import gkx
resolved = sorted(set(vars(gkx)) & set(gkx._EXPORT_TARGETS))
assert resolved == [], resolved
assert len(gkx.__all__) == 16
assert set(gkx.__all__) == {{'__version__', 'Case', 'LinearResult', 'NonlinearResult', 'ScanResult', 'load', 'solve', 'scan', 'plot', 'prepare', 'PreparedSimulation', 'flux_tube_geometry_from_mapping', 'solver_objective_vector_from_geometry', 'solver_linear_operator_matrix_from_geometry', 'solver_scalar_objective_from_vector', 'VMEXTransportObjectiveConfig'}}
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", root_script], check=True)

    api_script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT / "src")!r})
import gkx.api as api
resolved = sorted(set(vars(api)) & set(api._EXPORT_TARGETS))
assert resolved == [], resolved
assert "numpy" not in sys.modules
assert "jax" not in sys.modules
assert "LinearParams" not in api.__all__
assert "LinearParams" in api._EXPORT_TARGETS
"""
    subprocess.run([sys.executable, "-S", "-c", api_script], check=True)


def test_gkx3_product_contracts_preserve_identity_immutability_and_arrays() -> None:
    from dataclasses import FrozenInstanceError, is_dataclass

    import gkx
    from gkx.diagnostics.modes import ModeSelection
    from gkx.config import Case, RuntimeConfig
    from gkx.workflows.runtime.results import (
        LinearResult,
        NonlinearResult,
        RuntimeLinearResult,
        RuntimeLinearScanResult,
        RuntimeNonlinearResult,
        ScanResult,
    )

    assert Case is RuntimeConfig
    assert gkx.Case is RuntimeConfig
    assert is_dataclass(Case)
    case = Case()
    with pytest.raises(FrozenInstanceError):
        case.grid = case.grid  # type: ignore[misc]

    assert LinearResult is RuntimeLinearResult
    assert NonlinearResult is RuntimeNonlinearResult
    assert ScanResult is RuntimeLinearScanResult
    assert gkx.LinearResult is RuntimeLinearResult
    assert gkx.NonlinearResult is RuntimeNonlinearResult
    assert gkx.ScanResult is RuntimeLinearScanResult

    state = np.arange(6.0)
    linear = LinearResult(
        ky=0.2,
        gamma=0.1,
        omega=-0.3,
        selection=ModeSelection(ky_index=0, kx_index=0),
        state=state,
    )
    nonlinear = NonlinearResult(t=state, diagnostics=None, state=state)
    scan = ScanResult(ky=state, gamma=state, omega=state)

    assert linear.state is state
    assert nonlinear.t is state
    assert nonlinear.state is state
    assert scan.ky is state
    assert scan.gamma is state
    assert scan.omega is state
    with pytest.raises(FrozenInstanceError):
        linear.state = state  # type: ignore[misc]


def test_gkx3_workflow_contracts_delegate_to_existing_owners(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    import gkx
    import gkx.runtime as runtime
    from gkx.config import Case

    path = tmp_path / "case.toml"
    path.write_text('[geometry]\ngeometry_file = "relative.eik"\n', encoding="utf-8")
    case = gkx.load(path)
    assert isinstance(case, Case)
    assert case.geometry.geometry_file == str(tmp_path / "relative.eik")
    assert gkx.scan is runtime.run_runtime_scan

    calls: list[tuple[str, RuntimeConfig, dict[str, object]]] = []

    def fake_linear(cfg, **options):
        calls.append(("linear", cfg, options))
        return "linear-result"

    prepared_calls: list[tuple[RuntimeConfig, dict[str, object]]] = []

    def fake_nonlinear(cfg, **options):
        if options.get("prepare_only"):
            prepared_calls.append((cfg, options))
            return "prepared-simulation"
        calls.append(("nonlinear", cfg, options))
        return "nonlinear-result"

    patch_runtime(monkeypatch, "run_runtime_linear", fake_linear)
    patch_runtime(monkeypatch, "run_runtime_nonlinear", fake_nonlinear)
    # gkx.prepare is now the public PreparedSimulation constructor rather than
    # the runtime function it wraps. The delegation this test exists to pin is
    # still real and is asserted directly: preparing a nonlinear case reaches
    # runtime.prepare, so the facade adds a type without adding numerics.
    from gkx.api.prepared import prepare_simulation

    assert gkx.prepare is prepare_simulation
    assert "runtime.prepare" in inspect.getsource(prepare_simulation).replace(
        "_runtime.prepare", "runtime.prepare"
    )
    assert gkx.solve(case, ky_target=0.2) == "linear-result"
    # linear must be cleared as well: a case that declares both is rejected by
    # Case.validate, which is what prepare now runs before it builds anything.
    nonlinear = replace(
        case, physics=replace(case.physics, linear=False, nonlinear=True)
    )
    assert gkx.solve(nonlinear, steps=2) == "nonlinear-result"
    disabled = replace(case, physics=replace(case.physics, linear=False))
    with pytest.raises(ValueError, match="enable linear or nonlinear"):
        gkx.solve(disabled)
    # prepare used to raise for a linear case. It now prepares one and reports
    # that the linear runtime compiles per call rather than at prepare time.
    linear_prepared = gkx.prepare(case)
    assert linear_prepared.kind == "linear"
    assert linear_prepared.summary()["compiled_at_prepare"] is False
    with pytest.raises(ValueError, match="diagnostics=True"):
        gkx.prepare(nonlinear, diagnostics=False)
    nonlinear_prepared = gkx.prepare(nonlinear, steps=2)
    assert nonlinear_prepared.kind == "nonlinear"
    # The wrapper adds a type, not numerics: the object the runtime built is
    # carried through unchanged.
    assert nonlinear_prepared._backend == "prepared-simulation"
    assert prepared_calls[0][0] is nonlinear
    assert prepared_calls[0][1]["diagnostics"] is True
    assert prepared_calls[0][1]["prepare_only"] is True
    assert calls == [
        ("linear", case, {"ky_target": 0.2}),
        ("nonlinear", nonlinear, {"steps": 2}),
    ]


# Progress callback contracts.
def test_emit_progress_reports_one_based_step_once() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_progress(4, 5, 1.0, -2.0, 3.0, 4.0, sim_time=2.0, sim_total=2.5)
    out = buf.getvalue().strip()
    assert out.startswith("[gkx:segment]")
    assert "step=5/5" in out
    assert "progress=100.0%" in out
    assert "t=2/2.5" in out
    assert "elapsed=" in out
    assert "eta=00:00" in out
    assert "step=6/5" not in out


def test_format_duration_clamps_and_rolls_over() -> None:
    assert _format_duration(-2.0) == "00:00"
    assert _format_duration(65.4) == "01:05"
    assert _format_duration(3661.0) == "1:01:01"


def test_progress_update_stride_caps_long_runs() -> None:
    assert progress_update_stride(5) == 1
    assert progress_update_stride(50) == 1
    assert progress_update_stride(51) == 2
    assert progress_update_stride(500) == 10


def test_progress_update_stride_sanitizes_inputs() -> None:
    assert progress_update_stride(0) == 1
    assert progress_update_stride(-10, target_updates=0) == 1
    assert progress_update_stride(9, target_updates=4) == 3


def test_should_emit_progress_reports_first_interval_and_last() -> None:
    assert bool(should_emit_progress(0, 200)) is True
    assert bool(should_emit_progress(3, 200)) is True
    assert bool(should_emit_progress(4, 200)) is False
    assert bool(should_emit_progress(199, 200)) is True


def test_should_emit_progress_sanitizes_steps_and_targets() -> None:
    assert bool(should_emit_progress(0, 0, target_updates=0)) is True
    assert bool(should_emit_progress(1, 9, target_updates=4)) is False
    assert bool(should_emit_progress(2, 9, target_updates=4)) is True


def test_emit_progress_handles_time_variants_and_metric_labels(monkeypatch) -> None:
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr("gkx.callbacks.time.perf_counter", lambda: next(ticks))
    _PROGRESS_START.clear()

    first = io.StringIO()
    with redirect_stdout(first):
        _emit_progress(
            0, 3, 0.1, 0.2, 0.3, 0.4, sim_time=1.25, metric_labels=("A", "B")
        )
    first_out = first.getvalue()
    assert "step=1/3" in first_out
    assert "t=1.25" in first_out
    assert "eta=--:--" in first_out
    assert "A=0.3 B=0.4" in first_out

    second = io.StringIO()
    with redirect_stdout(second):
        _emit_progress(1, 3, 0.1, 0.2, 0.3, 0.4, sim_time=2.0, sim_total=0.0)
    second_out = second.getvalue()
    assert "step=2/3" in second_out
    assert "t=2" in second_out
    assert "/0" not in second_out
    assert "eta=00:01" in second_out


def test_print_callback_returns_state_and_forwards_values(monkeypatch) -> None:
    calls = []

    def fake_callback(fn, *args):
        calls.append(args)
        fn(*args)

    monkeypatch.setattr("gkx.callbacks.jax.debug.callback", fake_callback)

    state = {"unchanged": True}
    buf = io.StringIO()
    with redirect_stdout(buf):
        returned = print_callback(
            state,
            0,
            1,
            1.5,
            -0.5,
            2.0,
            3.0,
            sim_time=None,
            sim_total=None,
            metric_labels=("heat", "free"),
        )

    assert returned is state
    assert calls == [(0, 1, 1.5, -0.5, 2.0, 3.0, None, None)]
    out = buf.getvalue()
    assert "heat=2" in out
    assert "free=3" in out


# Configuration contracts.
def test_config_to_dict():
    """All config dataclasses should serialize to dictionaries."""
    cfg = CycloneBaseCase()
    d = cfg.to_dict()
    assert set(d.keys()) == {
        "grid",
        "time",
        "geometry",
        "model",
        "init",
        "reference_alignment",
    }
    assert d["geometry"]["q"] == cfg.geometry.q
    assert d["grid"]["y0"] == 20.0
    assert d["grid"]["ntheta"] == 32
    assert d["grid"]["nperiod"] == 2
    assert d["reference_alignment"]["enabled"] is True


def test_benchmark_case_presets_keep_stable_public_exports() -> None:
    """Benchmark presets are owned directly by the public config module."""

    for name in (
        "ModelConfig",
        "CycloneBaseCase",
        "KineticElectronModelConfig",
        "KBMBaseCase",
    ):
        assert hasattr(public_config, name)
        assert name in public_config.__all__


def test_config_override():
    """Overrides should propagate into the serialized representation."""
    grid = GridConfig(Nx=12, Ny=10, Nz=8)
    geom = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2)
    model = ModelConfig(tprim_i=7.0, tprim_e=1.0, fprim=2.5)
    time = TimeConfig(t_max=1.0, dt=0.05, compressed_real_fft=False)
    cfg = CycloneBaseCase(grid=grid, time=time, geometry=geom, model=model)
    d = cfg.to_dict()
    assert d["grid"]["Nx"] == 12
    assert d["geometry"]["q"] == 1.7
    assert d["model"]["tprim_e"] == 1.0
    assert d["time"]["dt"] == 0.05
    assert d["time"]["compressed_real_fft"] is False


def test_reference_aligned_mass_ratio_defaults() -> None:
    """Reference-aligned benchmark defaults should use the tracked electron mass."""

    cfg = KBMBaseCase()
    assert (1.0 / cfg.model.mass_ratio) == pytest.approx(REFERENCE_ELECTRON_MASS)


def test_kbm_config_to_dict():
    """KBM configuration should serialize to dictionaries."""
    cfg = KBMBaseCase()
    d = cfg.to_dict()
    assert d["model"]["beta"] == cfg.model.beta


def test_explicit_method_default_cfl_fac_is_method_resolved() -> None:
    assert explicit_method_default_cfl_fac("rk2") == pytest.approx(1.0)
    assert explicit_method_default_cfl_fac("rk3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("sspx3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("rk4") == pytest.approx(2.82)


def test_explicit_method_default_cfl_fac_alias_is_method_resolved() -> None:
    assert explicit_method_default_cfl_fac("rk2") == pytest.approx(1.0)
    assert explicit_method_default_cfl_fac("rk3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("sspx3") == pytest.approx(1.73)
    assert explicit_method_default_cfl_fac("rk4") == pytest.approx(2.82)


def test_resolve_cfl_fac_preserves_explicit_override() -> None:
    assert resolve_cfl_fac("rk3", None) == pytest.approx(1.73)
    assert resolve_cfl_fac("rk4", 1.25) == pytest.approx(1.25)


# Spectral grid contracts.
def test_build_spectral_grid_shapes():
    """Grid arrays should have consistent shapes."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    nky = rows_for_layout(cfg.Ny, cfg.ky_layout)
    assert grid.kx.shape == (cfg.Nx,)
    assert grid.ky.shape == (nky,)
    assert grid.z.shape == (cfg.Nz,)
    assert grid.kx_grid.shape == (nky, cfg.Nx)
    assert grid.ky_grid.shape == (nky, cfg.Nx)
    assert grid.dealias_mask.shape == (nky, cfg.Nx)


def test_build_spectral_grid_spacing():
    """Fourier spacing should match 2*pi/L for each direction."""
    cfg = GridConfig(Nx=8, Ny=6, Nz=4, Lx=2.0, Ly=3.0)
    grid = build_spectral_grid(cfg)
    dkx = grid.kx[1] - grid.kx[0]
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dkx, 2.0 * jnp.pi / cfg.Lx)
    assert jnp.isclose(dky, 2.0 * jnp.pi / cfg.Ly)


def test_spectral_grid_tree_roundtrip():
    """SpectralGrid pytree should round-trip through flatten/unflatten."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=2.0)
    grid = build_spectral_grid(cfg)
    children, aux = grid.tree_flatten()
    grid2 = SpectralGrid.tree_unflatten(aux, children)
    assert jnp.allclose(grid2.kx, grid.kx)
    assert jnp.allclose(grid2.ky, grid.ky)
    assert jnp.allclose(grid2.z, grid.z)


def test_grid_config_y0_and_ntheta():
    """Field-aligned grid inputs should map to expected ky and z spacing."""
    cfg = GridConfig(Nx=4, Ny=12, Nz=4, Lx=2.0, Ly=3.0, y0=20.0, ntheta=8, nperiod=2)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 8 * 3
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 8.0)
    dky = grid.ky[1] - grid.ky[0]
    assert jnp.isclose(dky, 1.0 / 20.0)


def test_grid_config_ntheta_default_zp():
    """ntheta without nperiod should default to Zp=1."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=6)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 6
    assert jnp.isclose(grid.z[0], -jnp.pi)
    dz = grid.z[1] - grid.z[0]
    assert jnp.isclose(dz, 2.0 * jnp.pi / 6.0)


def test_grid_config_explicit_zp():
    """Explicit Zp should override nperiod when provided."""
    cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=2.0, Ly=3.0, ntheta=5, zp=3)
    grid = build_spectral_grid(cfg)
    assert grid.z.shape[0] == 15
    assert jnp.isclose(grid.z[0], -jnp.pi * 3.0)


def test_compressed_real_fft_wavenumbers_match_gx_native_layout():
    """compressed real-FFT helpers should expose positive Nyquist multipliers."""

    # Both helpers exist to compress a two-sided axis, so the grid they read
    # is the two-sided one; on a half axis there is nothing left to compress.
    cfg = GridConfig(Nx=4, Ny=10, Nz=4, Lx=2.0, Ly=20.0, ky_layout="full")
    grid = build_spectral_grid(cfg)
    dkx = 2.0 * jnp.pi / cfg.Lx
    dky = 2.0 * jnp.pi / cfg.Ly
    assert jnp.allclose(
        real_fft_ordered_kx(grid.kx), jnp.asarray([0.0, dkx, 2.0 * dkx, -dkx])
    )
    assert jnp.allclose(
        real_fft_unique_ky(grid.ky),
        jnp.asarray([0.0, dky, 2.0 * dky, 3.0 * dky, 4.0 * dky, 5.0 * dky]),
    )


def test_select_real_fft_ky_grid_uses_explicit_positive_dump_values():
    """GX dump grids should not inherit the negative Nyquist sign from fftfreq order."""

    cfg = GridConfig(Nx=4, Ny=6, Nz=4, Lx=2.0, Ly=6.0)
    grid = build_spectral_grid(cfg)
    gx_ky = jnp.asarray(
        [
            0.0,
            2.0 * jnp.pi / cfg.Ly,
            2.0 * 2.0 * jnp.pi / cfg.Ly,
            3.0 * 2.0 * jnp.pi / cfg.Ly,
        ]
    )
    gx_grid = select_real_fft_ky_grid(grid, gx_ky)

    assert jnp.allclose(gx_grid.ky, gx_ky)
    assert jnp.all(gx_grid.ky >= 0.0)
    assert jnp.allclose(gx_grid.kx, real_fft_ordered_kx(grid.kx))
    assert gx_grid.dealias_mask.shape == (gx_ky.shape[0], cfg.Nx)
    assert jnp.allclose(gx_grid.ky_grid[:, 0], gx_ky)


def test_twothirds_mask_matches_strict_twothirds_cutoff():
    """The nonlinear two-thirds mask excludes the |k| = 1/3 shell."""

    # The GX view is cut from the two-sided axis by ``real_fft_unique_ky``, and
    # the row counts below count the rows of a 96-point two-sided grid.
    cfg = GridConfig(Nx=96, Ny=96, Nz=4, Lx=2.0 * jnp.pi, Ly=96.0, ky_layout="full")
    grid = build_spectral_grid(cfg)
    gx_grid = select_real_fft_ky_grid(grid, real_fft_unique_ky(grid.ky))
    mask = jnp.asarray(gx_grid.dealias_mask)

    # Positive ky rows retained by GX on a 96-point padded grid are 0..31.
    assert int(mask[:, 0].sum()) == 32
    assert bool(mask[31, 0])
    assert not bool(mask[32, 0])

    # Retained kx modes are -31..31 in FFT ordering.
    assert int(mask[0, :].sum()) == 63
    assert bool(mask[0, 31])
    assert not bool(mask[0, 32])


def test_select_ky_grid_disables_nonlinear_dealias_mask_for_linear_slices():
    """Linear ky slices should not zero modes dealiased only for nonlinear products."""

    cfg = GridConfig(Nx=1, Ny=12, Nz=4, Lx=2.0 * jnp.pi, y0=10.0)
    grid = build_spectral_grid(cfg)
    # The strict nonlinear two-thirds mask removes the boundary shell at ky index 4.
    assert not bool(grid.dealias_mask[4, 0])

    sliced = select_ky_grid(grid, [3, 4])

    assert jnp.allclose(sliced.ky, grid.ky[jnp.asarray([3, 4])])
    assert jnp.all(sliced.dealias_mask)


# ---- from test_public_types.py ----
# Contract tests for the public `Case` and result types.
#
# The plan's public-API section requires `Case` to own `replace`, `validate`,
# `to_toml` and `summary`, and each result to own `save`, `plot`,
# `print_summary` and `to_dataset`. Before this suite the types were frozen
# dataclasses with no behaviour, so every one of those verbs lived in a helper a


RESULT_TYPES = (RuntimeLinearResult, RuntimeLinearScanResult, RuntimeNonlinearResult)
CASE_METHODS = ("replace", "validate", "to_toml", "summary")
RESULT_METHODS = ("save", "plot", "print_summary", "to_dataset", "summary")


def _linear() -> RuntimeLinearResult:
    return RuntimeLinearResult(
        ky=0.5,
        gamma=0.14,
        omega=0.06,
        selection=ModeSelection(ky_index=1, kx_index=0, z_index=0),
        t=np.linspace(0.0, 1.0, 5),
        signal=np.ones(5),
        z=np.linspace(-np.pi, np.pi, 4),
        eigenfunction=np.ones(4, dtype=complex),
        fit_settled=True,
    )


@pytest.mark.parametrize("name", CASE_METHODS)
def test_case_advertises_its_contracted_methods(name: str) -> None:
    assert callable(getattr(RuntimeConfig, name))


@pytest.mark.parametrize("result_type", RESULT_TYPES)
@pytest.mark.parametrize("name", RESULT_METHODS)
def test_results_advertise_their_contracted_methods(result_type, name: str) -> None:
    assert callable(getattr(result_type, name))


def test_case_is_frozen_so_replace_is_the_only_way_to_derive_one() -> None:
    case = RuntimeConfig()
    with pytest.raises(Exception):
        case.grid = None  # type: ignore[misc]


def test_replace_returns_a_validated_copy_and_leaves_the_original_alone() -> None:
    case = RuntimeConfig()
    derived = case.replace(time=case.time.__class__(t_max=12.0, dt=0.01))
    assert derived is not case
    assert float(derived.time.t_max) == 12.0
    assert float(case.time.t_max) != 12.0 or float(case.time.dt) != 0.01


def test_replace_rejects_an_invalid_case_at_the_point_it_is_built() -> None:
    """A scan that builds cases in a loop must fail on the case it built."""

    case = RuntimeConfig()
    with pytest.raises(ValueError, match="invalid case"):
        case.replace(time=case.time.__class__(t_max=-1.0))


def test_validate_rejects_simultaneous_linear_and_nonlinear() -> None:
    case = RuntimeConfig()
    physics = case.physics.__class__(linear=True, nonlinear=True)
    with pytest.raises(ValueError, match="cannot both be true"):
        RuntimeConfig(physics=physics).validate()


def test_case_round_trips_through_to_toml_and_load(tmp_path) -> None:
    """What `to_toml` writes is what `gkx.load` reads back, field for field."""

    case = RuntimeConfig()
    written = case.to_toml(tmp_path / "case.toml")
    assert written.exists()
    assert gkx.load(written).to_dict() == case.to_dict()


def test_to_toml_omits_none_rather_than_inventing_a_null() -> None:
    """TOML has no null; an absent key is how the loader spells "default"."""

    from gkx.workflows.runtime.wout import deck_text

    text = deck_text({"grid": {"Nx": 8, "Ny": None}})
    assert "Nx = 8" in text
    assert "Ny" not in text


def test_case_summary_reports_the_identifying_scalars() -> None:
    summary = RuntimeConfig().summary()
    assert summary["n_species"] >= 1
    assert set(summary["grid"]) == {"Nx", "Ny", "Nz", "y0"}


def test_linear_summary_carries_fit_status_beside_the_number() -> None:
    summary = _linear().summary()
    assert summary["kind"] == "linear"
    assert summary["gamma"] == pytest.approx(0.14)
    assert summary["fit_settled"] is True


def test_nonlinear_summary_never_hides_an_unsaturated_verdict() -> None:
    """An unsaturated window must travel with its status, not be inferred."""

    result = RuntimeNonlinearResult(
        t=np.linspace(0.0, 10.0, 3),
        diagnostics=None,
        saturation={"saturated": False, "mean_flux": 4.5, "sem": 0.3},
    )
    summary = result.summary()
    assert summary["saturated"] is False
    assert summary["heat_flux_mean"] == pytest.approx(4.5)


def test_scan_summary_locates_the_growth_peak() -> None:
    scan = RuntimeLinearScanResult(
        ky=np.array([0.1, 0.5, 0.9]),
        gamma=np.array([0.02, 0.20, 0.11]),
        omega=np.array([0.01, 0.05, 0.09]),
    )
    summary = scan.summary()
    assert summary["gamma_peak"] == pytest.approx(0.20)
    assert summary["ky_at_gamma_peak"] == pytest.approx(0.5)


def test_to_dataset_is_xarray_shaped_without_requiring_xarray() -> None:
    """The mapping is exactly what `xarray.Dataset(**payload)` accepts."""

    payload = _linear().to_dataset()
    assert set(payload) == {"coords", "data_vars", "attrs"}
    dims, values = payload["data_vars"]["signal"]
    assert dims == "t"
    assert values.shape == payload["coords"]["t"].shape


def test_to_dataset_drops_absent_arrays_rather_than_emitting_none() -> None:
    scan = RuntimeLinearScanResult(
        ky=np.array([0.1]), gamma=np.array([0.02]), omega=np.array([0.01])
    )
    payload = scan.to_dataset()
    assert set(payload["data_vars"]) == {"gamma", "omega"}


def test_print_summary_writes_every_summary_field(capsys) -> None:
    _linear().print_summary()
    printed = capsys.readouterr().out
    assert "gamma" in printed and "fit_settled" in printed


# ---- the installed environment meets the floor the package declares -------
#
# ``gkx.objectives.core`` opts into ``lax_linalg.eig(..., enable_eigvec_derivs=
# True)``, which first shipped in jax 0.10.1, and ``pyproject.toml`` requires
# it. Nothing enforced that at test time, so running the suite against an older
# jax produced eight unrelated-looking physics failures -- a TypeError deep in
# the objective vector, plus four Hermite-hierarchy gates -- rather than one
# sentence naming the cause. A reviewer who hits that reads it as "the physics
# is broken" instead of "the environment is stale".


def test_installed_jax_meets_the_floor_the_package_declares() -> None:
    import re
    import tomllib
    from pathlib import Path

    import jax

    root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = pyproject["project"]["dependencies"]
    declared = next(r for r in requirements if r.replace(" ", "").startswith("jax>="))
    floor = re.search(r">=\s*([0-9.]+)", declared).group(1)

    def as_tuple(text: str) -> tuple[int, ...]:
        return tuple(int(part) for part in text.split(".") if part.isdigit())

    installed = jax.__version__

    assert as_tuple(installed) >= as_tuple(floor), (
        f"installed jax {installed} is below the {floor} this package requires. "
        "gkx.objectives.core needs eig(..., enable_eigvec_derivs=True), which "
        "older jax rejects with a TypeError, and several physics gates fail as "
        "a downstream consequence. Reinstall into a fresh environment "
        f"(python -m venv env && env/bin/pip install -e '.[dev]') rather than "
        "reading those failures as physics."
    )


# ---- from test_prepared_simulation.py ----
# Contract tests for the public prepared simulation.
#
# `gkx.prepare` used to return the solver's internal
# `PreparedExplicitNonlinearDiagnostics` and to raise for any linear case, so
# the reusable-execution concept covered half the product and leaked a private
# type while doing it. These tests pin the public replacement: both case kinds


PREPARED_METHODS = (
    "solve",
    "scan",
    "value_and_grad",
    "warmup",
    "estimate_memory",
    "summary",
)


def _linear_case() -> RuntimeConfig:
    return RuntimeConfig()


@pytest.mark.parametrize("name", PREPARED_METHODS)
def test_prepared_advertises_its_contracted_methods(name: str) -> None:
    assert callable(getattr(PreparedSimulation, name))


def test_gkx_prepare_returns_the_public_type_not_a_solver_internal() -> None:
    """The exported name must not leak `PreparedExplicitNonlinearDiagnostics`."""

    prepared = gkx.prepare(_linear_case())
    assert isinstance(prepared, PreparedSimulation)


def test_prepare_accepts_a_linear_case() -> None:
    """This raised `prepare currently requires nonlinear physics` before."""

    prepared = prepare_simulation(_linear_case())
    assert prepared.kind == "linear"


def test_prepared_is_frozen_so_topology_cannot_drift_from_its_metadata() -> None:
    prepared = prepare_simulation(_linear_case())
    with pytest.raises(Exception):
        prepared.kind = "nonlinear"  # type: ignore[misc]


def test_state_shape_follows_the_case_grid() -> None:
    case = _linear_case()
    prepared = prepare_simulation(case, Nl=6, Nm=12)
    n_species, n_l, n_m, _n_ky, n_kx, n_z = prepared.state_shape
    assert (n_l, n_m) == (6, 12)
    assert n_kx == int(case.grid.Nx)
    assert n_z == int(case.grid.Nz)
    assert n_species >= 1


def test_estimate_memory_scales_with_the_grid_and_admits_it_is_a_floor() -> None:
    small = prepare_simulation(_linear_case(), Nl=4, Nm=8)
    large = prepare_simulation(_linear_case(), Nl=8, Nm=16)
    assert (
        large.estimate_memory()["state_bytes"] > small.estimate_memory()["state_bytes"]
    )
    assert small.estimate_memory()["is_floor_not_ceiling"] is True


def test_summary_reports_precision_devices_and_cache_state() -> None:
    summary = prepare_simulation(_linear_case()).summary()
    assert summary["precision"] in {"float32", "float64"}
    assert summary["devices"]
    assert "persistent_cache_enabled" in summary
    assert summary["compiled_at_prepare"] is False  # linear compiles per call


def test_solve_refuses_parameters_rather_than_solving_a_different_problem() -> None:
    """A prepared object must not silently change physics."""

    prepared = prepare_simulation(_linear_case())
    with pytest.raises(NotImplementedError, match="value_and_grad"):
        prepared.solve(parameters={"tprim": 3.0})


def test_scan_names_the_parameter_it_cannot_scan() -> None:
    prepared = prepare_simulation(_linear_case())
    with pytest.raises(NotImplementedError, match="ky"):
        prepared.scan("tprim", [1.0, 2.0])


def test_warmup_returns_self_so_timing_code_can_chain() -> None:
    prepared = prepare_simulation(_linear_case())
    assert prepared.warmup() is prepared


def test_value_and_grad_differentiates_a_scalar_objective() -> None:
    prepared = prepare_simulation(_linear_case())
    value, grad = prepared.value_and_grad(lambda x: 3.0 * x**2, 2.0)
    assert float(value) == pytest.approx(12.0)
    assert float(grad) == pytest.approx(12.0)


def test_print_summary_writes_every_field(capsys) -> None:
    prepare_simulation(_linear_case()).print_summary()
    printed = capsys.readouterr().out
    assert "kind" in printed and "precision" in printed


# ---- the prepared topology is the one solve will build --------------------
#
# The deck's [run] table names the velocity resolution. Before it was carried
# on the case, the loader dropped it and prepare_simulation substituted a
# hard-coded (4, 8), so preparing the shipped Cyclone deck -- which asks for
# (16, 48) -- described a calculation nobody had requested. Reporting a
# resolution that solve will not build is worse than reporting none, because a
# recorded result then carries a resolution it was not run at.

_SHIPPED_LINEAR_DECK = "examples/01_linear_tokamak/case_full.toml"


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


def test_prepare_reports_the_resolution_the_shipped_deck_asks_for() -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)
    assert (case.run.Nl, case.run.Nm) == (16, 48), "deck under test changed"

    prepared = prepare_simulation(case)

    assert (prepared.n_laguerre, prepared.n_hermite) == (16, 48), (
        "prepare reported a velocity resolution the deck did not ask for; a "
        "summary that disagrees with the deck misdescribes the calculation"
    )


def test_an_explicit_argument_still_overrides_the_deck() -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)

    prepared = prepare_simulation(case, Nl=6, Nm=10)

    assert (prepared.n_laguerre, prepared.n_hermite) == (6, 10)


def test_a_deck_without_a_run_table_falls_back_to_the_runtime_default() -> None:
    """The fallback is the runtime's own, and it differs by kind.

    ``_CASE_LINEAR_SPECS`` defaults to (12, 24) and ``_CASE_NONLINEAR_SPECS``
    to (4, 8). Preparing must not invent a third pair, so the linear pair is
    read from the runtime's owner rather than repeated here.
    """

    from gkx.workflows.runtime.startup import _RUNTIME_LINEAR_HL_FALLBACK

    case = _linear_case()
    assert case.run.is_empty(), "this fixture is meant to carry no [run] table"

    prepared = prepare_simulation(case)

    assert (prepared.n_laguerre, prepared.n_hermite) == (12, 24)
    assert (prepared.n_laguerre, prepared.n_hermite) == _RUNTIME_LINEAR_HL_FALLBACK


def test_a_deck_with_a_run_table_round_trips_through_toml(tmp_path) -> None:
    case = gkx.load(_repo_root() / _SHIPPED_LINEAR_DECK)

    written = case.to_toml(tmp_path / "resolved.toml")
    reloaded = gkx.load(written)

    assert (reloaded.run.Nl, reloaded.run.Nm) == (16, 48)
    assert reloaded.run.solver == case.run.solver


def test_a_deck_without_a_run_table_does_not_grow_one(tmp_path) -> None:
    """Decks that never had [run] keep writing byte-identical resolved decks."""

    case = _linear_case()

    text = (case.to_toml(tmp_path / "resolved.toml")).read_text(encoding="utf-8")

    assert "[run]" not in text


# ---- warmup moves the compile, instead of claiming to --------------------
#
# Preparing a nonlinear case builds its scan closure but does not compile it:
# XLA compiles when that scan first executes. ``warmup`` documented itself as
# "force compilation now so a later timed call measures execution" and returned
# self without doing anything, and ``summary()`` reported
# ``compiled_at_prepare`` as ``kind == "nonlinear"`` -- claiming a compile that
# had not happened. Measured on the shipped five-step KBM deck in a fresh
# process: prepare 4.3 s, warmup 0.000 s, first solve 19.2 s. Anyone who called
# warmup and then timed solve was timing the compiler.


def _tiny_nonlinear_case() -> RuntimeConfig:
    """The smallest nonlinear case that still exercises a real compile."""

    import dataclasses

    base = RuntimeConfig()
    return base.replace(
        grid=dataclasses.replace(base.grid, Nx=4, Ny=4, Nz=8),
        physics=dataclasses.replace(base.physics, linear=False, nonlinear=True),
        time=dataclasses.replace(base.time, run_to="t_max", t_max=0.01, dt=0.005),
    )


def test_preparing_does_not_claim_a_compile_it_has_not_done() -> None:
    prepared = prepare_simulation(_tiny_nonlinear_case(), Nl=2, Nm=4, steps=2)

    summary = prepared.summary()

    assert summary["compiled_at_prepare"] is False
    assert summary["warmed"] is False


def test_warmup_moves_the_compile_out_of_the_first_solve() -> None:
    """After warmup, a timed solve measures execution rather than compilation."""

    import time

    prepared = prepare_simulation(_tiny_nonlinear_case(), Nl=2, Nm=4, steps=2)

    start = time.perf_counter()
    assert prepared.warmup() is prepared
    warm_seconds = time.perf_counter() - start

    start = time.perf_counter()
    prepared.solve()
    solve_seconds = time.perf_counter() - start

    assert prepared.summary()["warmed"] is True
    # The compile dominates the warmup and is absent from the solve. A factor of
    # ten is far inside the ~4900x measured here, so this fails on a warmup that
    # does nothing without being sensitive to machine speed.
    assert solve_seconds * 10 < warm_seconds, (
        f"warmup took {warm_seconds:.3f} s and the following solve "
        f"{solve_seconds:.3f} s. warmup is supposed to absorb the compile; if "
        "the solve is still paying for it, warmup is not doing its job."
    )


def test_warming_twice_does_not_run_the_case_twice() -> None:
    import time

    prepared = prepare_simulation(_tiny_nonlinear_case(), Nl=2, Nm=4, steps=2)
    prepared.warmup()

    start = time.perf_counter()
    prepared.warmup()
    second_seconds = time.perf_counter() - start

    assert second_seconds < 0.5, (
        f"a second warmup took {second_seconds:.3f} s, so it re-ran the case "
        "instead of returning immediately"
    )
    assert prepared.summary()["warmed"] is True


def test_a_linear_case_reports_that_it_was_not_warmed() -> None:
    """There is nothing to compile ahead of a solver chosen per call."""

    prepared = prepare_simulation(_linear_case())

    assert prepared.warmup() is prepared
    assert prepared.summary()["warmed"] is False


# ---- supplied states on the public entry point (queue row Q23) ------------
#
# The runtime zeroes the off-chain rows of a user ``initial_state`` on a linked
# deck (#247). ``gkx.prepare``'s object is the same door one floor down, and
# ``solve(initial_state=...)`` reaches it directly, so it applies the same rule
# rather than integrating rows the chains never reach -- which the ExB bracket
# would alias straight back onto the chain rows.


def _linked_nonlinear_case() -> RuntimeConfig:
    """A tiny nonlinear case whose linked chains leave kx rows uncovered."""

    import dataclasses

    base = RuntimeConfig()
    return base.replace(
        grid=dataclasses.replace(
            base.grid, Nx=8, Ny=8, Nz=8, Lx=6.28, Ly=6.28, boundary="linked", jtwist=1
        ),
        physics=dataclasses.replace(base.physics, linear=False, nonlinear=True),
        time=dataclasses.replace(base.time, run_to="t_max", t_max=0.01, dt=0.005),
    )


def test_prepared_solve_projects_a_supplied_state_onto_the_linked_cover() -> None:
    import numpy as np

    from gkx.operators.linear.linked import linked_cover_mask_from_cache

    prepared = prepare_simulation(_linked_nonlinear_case(), Nl=2, Nm=4, steps=2)
    backend = prepared._backend
    cover = np.asarray(linked_cover_mask_from_cache(backend.cache))
    assert not cover.all(), "this deck must leave rows outside the chains"

    shape = np.asarray(backend.initial_state).shape
    rng = np.random.default_rng(2302)
    supplied = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.asarray(backend.initial_state).dtype
    )
    supplied = supplied / np.max(np.abs(supplied))
    off = np.broadcast_to(~cover[:, :, None], shape)
    on_cover = np.where(off, 0.0, supplied).astype(supplied.dtype)
    assert np.max(np.abs(supplied[off])) > 0.0

    from_supplied = prepared.solve(initial_state=supplied)
    from_on_cover = prepared.solve(initial_state=on_cover)

    final_supplied = np.asarray(from_supplied[2])
    final_on_cover = np.asarray(from_on_cover[2])
    assert np.max(np.abs(final_supplied[off])) == 0.0
    np.testing.assert_array_equal(final_supplied, final_on_cover)


# ---- from test_core_ky_layout.py ----
# Contract tests for the ``ky`` axis layout (plan 5.3 N3).
#
# GKX evolves the two-sided axis by default -- rebuilding its negative half by the
# reality condition -- and evolves the ``ky >= 0`` rows, as GX, stella and GS2
# do, when a deck opts in with ``[grid] ky_layout = "half"``.  The conversion between the two layouts is one rule
# with one owner instead of five hand-written copies, and the rule is pinned here


EVEN_NY = (2, 4, 8, 16, 32)
ODD_NY = (3, 5, 9, 17)
ALL_NY = EVEN_NY + ODD_NY


def _half_of_real_field(ny: int, nx: int, nz: int, *, seed: int) -> np.ndarray:
    """Return the ``ky >= 0`` spectrum of a random real ``(y, x, z)`` field.

    ``rfft2`` over ``(kx, ky)`` is the bracket's own transform, so this is the
    half block the code actually produces, not an idealization of it.
    """

    rng = np.random.default_rng(seed)
    physical = rng.normal(size=(ny, nx, nz))
    return np.fft.rfft2(physical, axes=(1, 0)).astype(np.complex128)


def _real_field(ny: int, nx: int, nz: int, *, seed: int) -> np.ndarray:
    """Return a two-sided spectrum that satisfies the reality condition exactly.

    Built by widening the half block, so the redundant rows are equal to their
    partners' conjugates bit for bit.  A two-sided ``fft2`` of the same field
    is Hermitian only to roundoff, which would turn every exactness check in
    this file into a tolerance check on numpy's FFT.
    """

    return to_full(_half_of_real_field(ny, nx, nz, seed=seed), ny_full=ny)


def _batched(spectrum: np.ndarray) -> np.ndarray:
    return spectrum[None, None, None, ...]


# --------------------------------------------------------------------------
# Axis arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_nyc_matches_the_rfft_row_count(ny: int) -> None:
    assert nyc_from_ny(ny) == int(np.fft.rfftfreq(ny).size)


@pytest.mark.parametrize("ny", ALL_NY)
def test_nyc_does_not_determine_ny(ny: int) -> None:
    """Both candidates map back to the same ``Nyc``, so widening needs ``Ny``."""

    even, odd = ny_full_candidates(nyc_from_ny(ny))
    assert ny in (even, odd)
    assert even % 2 == 0 and odd % 2 == 1
    assert nyc_from_ny(even) == nyc_from_ny(odd) == nyc_from_ny(ny)


@pytest.mark.parametrize("ny", EVEN_NY)
def test_even_grids_have_a_nyquist_row_at_the_top_of_the_half_axis(ny: int) -> None:
    row = nyquist_row(ny)
    assert row == nyc_from_ny(ny) - 1
    assert self_conjugate_rows(ny) == ((0,) if row == 0 else (0, row))


@pytest.mark.parametrize("ny", ODD_NY)
def test_odd_grids_have_no_nyquist_row(ny: int) -> None:
    assert nyquist_row(ny) is None
    assert self_conjugate_rows(ny) == (0,)
    # Every stored row above zero owns a distinct partner.
    assert paired_row_limit(ny) == nyc_from_ny(ny)


def test_axis_arithmetic_rejects_degenerate_lengths() -> None:
    with pytest.raises(ValueError):
        nyc_from_ny(0)
    with pytest.raises(ValueError):
        ny_full_candidates(0)
    with pytest.raises(ValueError):
        conjugate_kx_order(0)


@pytest.mark.parametrize("nx", (1, 2, 3, 4, 7, 8))
def test_conjugate_kx_order_negates_the_radial_mode_number(nx: int) -> None:
    """``order`` sends each ``kx`` mode to ``-kx`` modulo the box, and is its own
    inverse.  An even grid's Nyquist column is its own image, which is why the
    permutation is written as an index map rather than a sign flip."""

    order = conjugate_kx_order(nx)
    modes = np.rint(np.fft.fftfreq(nx, d=1.0 / nx)).astype(int)
    np.testing.assert_array_equal(modes[order] % nx, (-modes) % nx)
    np.testing.assert_array_equal(order[order], np.arange(nx))


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
@pytest.mark.parametrize("nx", (1, 4, 5))
def test_half_full_round_trip_is_exact_for_a_real_field(ny: int, nx: int) -> None:
    """``to_full(to_half(F)) == F`` bitwise when ``F`` is a real field's spectrum."""

    full = _batched(_real_field(ny, nx, 3, seed=ny * 100 + nx))
    rebuilt = to_full(to_half(full), ny_full=ny)
    assert rebuilt.shape == full.shape
    np.testing.assert_array_equal(rebuilt, full)


@pytest.mark.parametrize("ny", ALL_NY)
def test_to_half_is_idempotent_at_a_boundary_handed_either_layout(ny: int) -> None:
    full = _batched(_real_field(ny, 4, 3, seed=ny))
    half = to_half(full)
    assert half.shape[-3] == nyc_from_ny(ny)
    np.testing.assert_array_equal(to_half(half, ny_full=ny), half)


def test_to_half_refuses_a_row_count_that_is_neither_layout() -> None:
    state = np.zeros((1, 1, 1, 6, 4, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="expected the full axis"):
        to_half(state, ny_full=16)


def test_to_full_refuses_a_half_block_that_does_not_match_ny_full() -> None:
    half = np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="does not match ny_full"):
        to_full(half, ny_full=16)


@pytest.mark.parametrize("ny", (8, 9))
def test_widening_carries_ny_full_rather_than_guessing_it(ny: int) -> None:
    """The same ``Nyc = 5`` block widens to 8 or 9 rows, on request only."""

    half = np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128)
    assert to_full(half, ny_full=ny).shape[-3] == ny


# --------------------------------------------------------------------------
# The reality condition and the self-conjugate rows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_widened_arrays_satisfy_the_reality_condition_row_by_row(ny: int) -> None:
    nx = 6
    half = to_half(_batched(_real_field(ny, nx, 3, seed=ny + 7)))
    full = to_full(half, ny_full=ny)
    order = conjugate_kx_order(nx)
    for row in range(1, paired_row_limit(ny)):
        partner = (-row) % ny
        np.testing.assert_array_equal(
            full[..., partner, :, :], np.conj(full[..., row, order, :])
        )


@pytest.mark.parametrize("ny", ALL_NY)
def test_reality_residual_is_zero_for_a_real_field_and_positive_otherwise(
    ny: int,
) -> None:
    nx = 6
    half = symmetrize_self_conjugate_rows(
        _batched(_half_of_real_field(ny, nx, 3, seed=ny + 11)), ny_full=ny
    )
    full = to_full(half, ny_full=ny)
    assert float(reality_residual(full)) == 0.0
    broken = full.copy()
    broken[..., 0, 1, :] += 1.0  # breaks the ky=0 internal constraint
    assert float(reality_residual(broken)) > 0.0
    # A field's own rfft2 block only satisfies the ky=0 constraint to roundoff.
    raw = to_full(_batched(_half_of_real_field(ny, nx, 3, seed=ny + 11)), ny_full=ny)
    assert float(reality_residual(raw)) < 1e-14


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_half_layout_alone_does_not_enforce_the_self_conjugate_rows(
    ny: int,
) -> None:
    """Storing ``ky >= 0`` leaves ``F[0, kx] = conj(F[0, -kx])`` to be imposed."""

    nx = 6
    half = to_half(_batched(_real_field(ny, nx, 3, seed=ny + 13)))
    polluted = np.array(half)
    polluted[..., 0, 1, :] += 0.5 + 0.25j
    order = conjugate_kx_order(nx)
    assert not np.allclose(polluted[..., 0, :, :], np.conj(polluted[..., 0, order, :]))
    fixed = symmetrize_self_conjugate_rows(polluted, ny_full=ny)
    for row in self_conjugate_rows(ny):
        np.testing.assert_allclose(
            fixed[..., row, :, :], np.conj(fixed[..., row, order, :]), atol=0, rtol=0
        )


@pytest.mark.parametrize("ny", ALL_NY)
def test_symmetrization_is_idempotent_and_leaves_a_real_field_alone(ny: int) -> None:
    half = to_half(_batched(_real_field(ny, 6, 3, seed=ny + 17)))
    once = symmetrize_self_conjugate_rows(half, ny_full=ny)
    np.testing.assert_allclose(once, half, rtol=0, atol=1e-12)
    twice = symmetrize_self_conjugate_rows(once, ny_full=ny)
    np.testing.assert_array_equal(twice, once)


def test_symmetrization_refuses_a_row_count_that_is_neither_layout() -> None:
    with pytest.raises(ValueError, match="expected 16 or 9"):
        symmetrize_self_conjugate_rows(
            np.zeros((1, 1, 1, 6, 4, 2), dtype=np.complex128), ny_full=16
        )


def test_reality_residual_refuses_a_half_spectrum_input() -> None:
    with pytest.raises(ValueError, match="needs the full ky axis"):
        reality_residual(np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128), ny_full=8)


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_row_weights_reproduce_the_full_axis_sum_of_a_symmetric_quantity(
    ny: int,
) -> None:
    """``sum_full |F|^2 == sum(w * |F_half|^2)``: the flux/spectrum identity."""

    full = _real_field(ny, 6, 3, seed=ny + 19)
    power = np.abs(full) ** 2
    weights = ky_row_weights(ny)
    assert weights.shape == (nyc_from_ny(ny),)
    np.testing.assert_allclose(
        float(np.sum(power)),
        float(np.sum(weights[:, None, None] * power[: nyc_from_ny(ny)])),
        rtol=1e-12,
    )


@pytest.mark.parametrize("ny", EVEN_NY)
def test_a_blanket_factor_of_two_over_counts_the_nyquist_row(ny: int) -> None:
    """The reason the weights exist rather than a scalar 2 on ``ky > 0``."""

    weights = ky_row_weights(ny)
    blanket = np.where(np.arange(nyc_from_ny(ny)) == 0, 1.0, 2.0)
    row = nyquist_row(ny)
    assert row is not None
    assert weights[row] == 1.0
    if row != 0:
        assert blanket[row] == 2.0
        assert not np.array_equal(weights, blanket)


@pytest.mark.parametrize("ny", ODD_NY)
def test_odd_grids_weight_every_row_above_zero_by_two(ny: int) -> None:
    weights = ky_row_weights(ny)
    np.testing.assert_array_equal(weights[1:], np.full(weights.size - 1, 2.0))


# --------------------------------------------------------------------------
# Grid views
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_half_ky_values_match_rfftfreq_up_to_the_box_scale(ny: int) -> None:
    ly = 4.0
    ky_full = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    expected = 2.0 * np.pi * np.fft.rfftfreq(ny, d=ly / ny)
    np.testing.assert_allclose(half_ky_values(ky_full), expected, rtol=1e-12)


@pytest.mark.parametrize("ny", ALL_NY)
def test_half_dealias_mask_is_the_top_block_and_is_idempotent(ny: int) -> None:
    mask = np.arange(ny * 4, dtype=float).reshape(ny, 4)
    half = half_dealias_mask(mask, ny_full=ny)
    np.testing.assert_array_equal(half, mask[: nyc_from_ny(ny)])
    np.testing.assert_array_equal(half_dealias_mask(half, ny_full=ny), half)


def test_half_dealias_mask_refuses_a_foreign_row_count() -> None:
    with pytest.raises(ValueError, match="expected 16 or 9"):
        half_dealias_mask(np.zeros((6, 4)), ny_full=16)


def test_describe_reports_the_contract_numbers_for_one_grid() -> None:
    assert describe(8) == {
        "ny_full": 8,
        "nyc": 5,
        "nyquist_row": 4,
        "self_conjugate_rows": (0, 4),
        "paired_rows": (1, 4),
        "ny_full_candidates_for_nyc": (8, 9),
        "row_weights": (1.0, 2.0, 2.0, 2.0, 1.0),
    }


# --------------------------------------------------------------------------
# Backend dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 9))
def test_numpy_input_keeps_numpy_output_so_host_paths_stay_on_the_host(
    ny: int,
) -> None:
    half = to_half(_batched(_real_field(ny, 4, 2, seed=ny)))
    assert isinstance(to_full(half, ny_full=ny), np.ndarray)
    assert isinstance(symmetrize_self_conjugate_rows(half, ny_full=ny), np.ndarray)
    assert isinstance(negative_ky_block(half, ny_full=ny), np.ndarray)


@pytest.mark.parametrize("ny", (8, 9))
def test_jax_and_numpy_widening_agree_and_widening_traces(ny: int) -> None:
    half_np = to_half(_batched(_real_field(ny, 4, 2, seed=ny + 3)))
    half_jax = jnp.asarray(half_np)
    from_numpy = to_full(half_np, ny_full=ny)
    from_jax = to_full(half_jax, ny_full=ny)
    assert isinstance(from_jax, jax.Array)
    np.testing.assert_allclose(np.asarray(from_jax), from_numpy, rtol=0, atol=1e-12)

    traced = jax.jit(lambda x: to_full(x, ny_full=ny))(half_jax)
    np.testing.assert_array_equal(np.asarray(traced), np.asarray(from_jax))


def test_widening_is_differentiable_and_its_vjp_sums_the_conjugate_pair() -> None:
    """The completion is linear over the reals; its adjoint folds both rows back."""

    ny, nx = 8, 4

    def widen(real_imag: jnp.ndarray) -> jnp.ndarray:
        half = real_imag[0] + 1j * real_imag[1]
        return to_full(half, ny_full=ny)

    key = jax.random.PRNGKey(0)
    parts = jax.random.normal(key, (2, 1, 1, 1, nyc_from_ny(ny), nx, 2))
    value, pullback = jax.vjp(widen, parts)
    cotangent = jnp.ones_like(value)
    (grad,) = pullback(cotangent)
    assert grad.shape == parts.shape
    # ky=0 and Nyquist appear once; every paired row appears twice.
    real_grad = np.asarray(grad[0])[0, 0, 0, :, 0, 0]
    np.testing.assert_allclose(real_grad, ky_row_weights(ny), rtol=1e-6)


# --------------------------------------------------------------------------
# The consumers that already sit on the boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 9, 16))
def test_the_bracket_half_core_plus_the_widening_is_the_bracket(ny: int) -> None:
    """The ``ky >= 0`` bracket primitive is the whole kernel; the rest is layout.

    Plan 5.3 N3 has the evolved state call the half core directly.  This pins
    the seam now: the two-sided kernel is exactly the half core followed by the
    contract's widening and the real ``kxfac``, bit for bit.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid
    from gkx.operators.nonlinear.brackets import (
        _complete_hermitian_ky,
        _spectral_bracket_half_core,
        _spectral_bracket_real_fft_core,
    )

    # The seam under test is the widening that turns the half core's output
    # into the two-sided kernel's, so this grid has to carry both halves of the
    # ky axis rather than whichever one the default hands out.
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=ny, Nz=3, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    nx, nz = int(grid.kx.size), int(grid.z.size)
    G = jnp.asarray(_batched(_real_field(ny, nx, nz, seed=ny + 23))[0])
    chi = jnp.asarray(_real_field(ny, nx, nz, seed=ny + 29))
    kwargs = dict(
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
    )
    kxfac = jnp.asarray(1.3)

    half, reported_ny, reported_nx = _spectral_bracket_half_core(
        G, chi, multiple_fields=False, **kwargs
    )
    assert (reported_ny, reported_nx) == (ny, nx)
    assert int(half.shape[-3]) == nyc_from_ny(ny)

    full = _spectral_bracket_real_fft_core(
        G, chi, kxfac=kxfac, multiple_fields=False, **kwargs
    )
    expected = kxfac * _complete_hermitian_ky(half, ny, nx)
    np.testing.assert_array_equal(np.asarray(full), np.asarray(expected))

    # The bracket's own output obeys the reality condition it was widened with.
    assert float(reality_residual(np.asarray(full))) < 1e-6


@pytest.mark.parametrize("ny", (8, 9))
def test_the_nonlinear_projector_is_the_contract_round_trip(ny: int) -> None:
    from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector

    nx, nz = 6, 3
    project = _make_compressed_real_fft_projector(ny_full=ny, nx=nx)
    noise = jnp.asarray(
        _batched(
            np.random.default_rng(ny).normal(size=(ny, nx, nz))
            + 1j * np.random.default_rng(ny + 1).normal(size=(ny, nx, nz))
        )
    )
    projected = project(noise)
    np.testing.assert_array_equal(
        np.asarray(projected), np.asarray(to_full(to_half(noise), ny_full=ny))
    )
    # A projector is idempotent, which is what makes it a projector.
    np.testing.assert_array_equal(np.asarray(project(projected)), np.asarray(projected))


@pytest.mark.parametrize("ny", (8, 9, 3, 2))
def test_the_restart_read_path_widens_with_the_grids_own_ny(ny: int) -> None:
    """``_expand_ky`` used to infer ``Ny = 2*(Nyc-1)``, losing a row on odd grids."""

    from gkx.workflows.runtime.startup import _expand_ky

    nx, nz = 4, 2
    half = _batched(_half_of_real_field(ny, nx, nz, seed=ny + 31))
    widened = _expand_ky(half, ny_full=ny)
    assert widened.shape[-3] == ny
    np.testing.assert_array_equal(widened, to_full(half, ny_full=ny))
    # Already-full input passes through untouched.
    np.testing.assert_array_equal(_expand_ky(widened, ny_full=ny), widened)


@pytest.mark.parametrize("ny", (8, 9))
def test_the_artifact_restart_expansion_round_trips_through_the_contract(
    ny: int,
) -> None:
    from gkx.artifacts.io import _expand_positive_ky_to_full

    nx, nz = 4, 2
    half = _batched(_half_of_real_field(ny, nx, nz, seed=ny + 37))
    full = _expand_positive_ky_to_full(half, ny_full=ny)
    assert isinstance(full, np.ndarray)
    np.testing.assert_array_equal(full, to_full(half, ny_full=ny))
    np.testing.assert_array_equal(to_half(full, ny_full=ny), half)
    with pytest.raises(ValueError, match="does not match ny_full"):
        _expand_positive_ky_to_full(half, ny_full=ny + 4)


@pytest.mark.parametrize("ny", (8, 9))
def test_the_in_place_hermitian_repair_matches_the_contracts_negative_block(
    ny: int,
) -> None:
    from gkx.workflows.runtime.startup import _enforce_full_ky_hermitian

    nx, nz = 4, 2
    rng = np.random.default_rng(ny + 41)
    noisy = (
        rng.normal(size=(1, 1, 1, ny, nx, nz))
        + 1j * rng.normal(size=(1, 1, 1, ny, nx, nz))
    ).astype(np.complex64)
    repaired = _enforce_full_ky_hermitian(noisy.copy())
    nyc = nyc_from_ny(ny)
    np.testing.assert_array_equal(repaired[..., :nyc, :, :], noisy[..., :nyc, :, :])
    np.testing.assert_array_equal(
        repaired[..., nyc:, :, :],
        negative_ky_block(noisy[..., :nyc, :, :], ny_full=ny),
    )


def test_the_grid_reports_the_half_axis_through_the_contract() -> None:
    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid, real_fft_unique_ky

    # ``real_fft_unique_ky`` is the reduction from the two-sided axis to the
    # half one, so it has to be handed a grid that stores the negative rows.
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=10, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    np.testing.assert_allclose(
        np.asarray(real_fft_unique_ky(grid.ky)),
        np.asarray(half_ky_values(grid.ky)),
        rtol=0,
        atol=0,
    )
    assert int(real_fft_unique_ky(grid.ky).size) == nyc_from_ny(10)


def test_prepared_simulation_reports_the_state_it_actually_allocates() -> None:
    """``state_shape`` advertised ``Nyc`` while the runtime allocated ``Ny``."""

    from gkx.api.prepared import prepare_simulation
    from gkx.config import RuntimeConfig
    from gkx.core_grid import build_spectral_grid

    case = RuntimeConfig()
    prepared = prepare_simulation(case, Nl=2, Nm=4)
    grid = build_spectral_grid(case.grid)
    rows = rows_for_layout(case.grid.Ny, case.grid.ky_layout)
    assert prepared.state_shape[-3] == int(grid.ky.size) == rows
    assert prepared.estimate_memory()["elements"] == int(np.prod(prepared.state_shape))


def _grids(ny: int, nx: int = 4):
    """Return the two-sided grid for ``ny`` and its ``ky >= 0`` view.

    Its callers weigh one layout against the other, and the half view is cut
    from the two-sided one, so the parent grid names the two-sided axis instead
    of taking whichever layout the configuration defaults to.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid, select_real_fft_ky_grid

    full_grid = build_spectral_grid(
        GridConfig(Nx=nx, Ny=ny, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    return full_grid, select_real_fft_ky_grid(full_grid, half_ky_values(full_grid.ky))


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_moment_weight_is_the_contract_weight_on_a_half_axis(ny: int) -> None:
    """``_hermitian_mode_weight`` and :func:`ky_row_weights` are one rule.

    Queue row Q24.  The rule used to be written out three times and the copies
    gave an even grid's Nyquist row weight 2, which is the paired weight on a
    row that has no partner.  Nothing shipped reached it -- the only
    half-spectrum grids GKX builds are the GX-comparison views, and the
    two-thirds mask zeroes every row at or above ``Ny/3``, Nyquist included --
    but plan 5.3 N3 moves the evolved state onto this axis, where the
    undealiased weight has to be right.
    """

    from gkx.operators.moments import _hermitian_mode_weight

    _full_grid, half_grid = _grids(ny)
    bare = np.asarray(_hermitian_mode_weight(half_grid, use_dealias=False))[:, 0]
    np.testing.assert_array_equal(bare, ky_row_weights(ny))


@pytest.mark.parametrize("ny", EVEN_NY)
def test_the_half_axis_weight_sums_a_non_zero_nyquist_row_correctly(ny: int) -> None:
    """The case the old rule got wrong, with the Nyquist row carrying power.

    This is the whole point of the fix: with the row zeroed, weight 1 and
    weight 2 agree, so only an undealiased field with power at ``ky = Ny/2``
    tells them apart.  The old blanket-two rule over-counts that row's
    ``|F|^2`` by exactly itself and fails here.
    """

    from gkx.operators.moments import _hermitian_mode_weight

    row = nyquist_row(ny)
    assert row is not None and row != 0

    full = _real_field(ny, 4, 3, seed=ny + 71)
    assert float(np.max(np.abs(full[row]))) > 0.0
    _full_grid, half_grid = _grids(ny, nx=4)

    weight = np.asarray(_hermitian_mode_weight(half_grid, use_dealias=False))
    half = np.abs(full[: nyc_from_ny(ny)]) ** 2
    got = float(np.sum(weight[:, :, None] * half))
    np.testing.assert_allclose(got, float(np.sum(np.abs(full) ** 2)), rtol=1e-12)

    blanket = np.where(np.arange(nyc_from_ny(ny))[:, None] == 0, 1.0, 2.0)
    over = float(np.sum(blanket[:, :, None] * half))
    assert over > got
    np.testing.assert_allclose(
        over - got, float(np.sum(np.abs(full[row]) ** 2)), rtol=1e-12
    )


def test_the_default_deck_builds_the_two_sided_axis() -> None:
    """``[grid] ky_layout`` defaults to ``"full"``; ``"half"`` is an opt-in."""

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid

    assert GridConfig().ky_layout == FULL
    grid = build_spectral_grid(GridConfig(Nx=4, Ny=8, Nz=2))
    assert source_ky_layout(grid) == FULL
    assert int(grid.ky.size) == 8


@pytest.mark.parametrize("ny", (4, 8, 16, 32))
def test_the_half_deck_grid_weights_its_nyquist_row_once(ny: int) -> None:
    """The Nyquist rule, on the grid a deck builds when it names ``"half"``.

    The tests above cut their half axis from a two-sided parent with
    ``select_real_fft_ky_grid``, which is how the GX-comparison views are made.
    A run does not build its grid that way: ``build_spectral_grid`` reads
    ``GridConfig.ky_layout``. Whether the Nyquist row is found on it depends on
    ``ny_full`` reaching the grid from the config, which is a different path
    from the one above and the one a ``ky_layout = "half"`` deck takes. So this
    pins the rule where such a deck runs it: weight 1 on the Nyquist row for
    the Hermitian reductions, 0.5 for the flux representative, and a non-zero
    Nyquist row summed exactly once.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid
    from gkx.operators.moments import _hermitian_mode_weight, _transport_mode_weight

    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=ny, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="half")
    )
    assert source_ky_layout(grid) == HALF
    assert source_ny_full(grid) == ny
    row = nyquist_row(ny)
    assert row == nyc_from_ny(ny) - 1

    hermitian = np.asarray(_hermitian_mode_weight(grid, use_dealias=False))
    transport = np.asarray(_transport_mode_weight(grid, use_dealias=False))
    np.testing.assert_array_equal(hermitian[:, 0], ky_row_weights(ny))
    np.testing.assert_array_equal(hermitian[row], np.ones(4))
    np.testing.assert_array_equal(transport[row], np.full(4, 0.5))
    np.testing.assert_array_equal(transport[0], np.zeros(4))

    full = _real_field(ny, 4, 3, seed=ny + 97)
    assert float(np.max(np.abs(full[row]))) > 0.0
    half = np.abs(full[: nyc_from_ny(ny)]) ** 2
    np.testing.assert_allclose(
        float(np.sum(hermitian[:, :, None] * half)),
        float(np.sum(np.abs(full) ** 2)),
        rtol=1e-12,
    )


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_two_sided_moment_weight_counts_every_stored_row_once(ny: int) -> None:
    """The layout the code evolves today is untouched by the Q24 rule."""

    from gkx.operators.moments import _hermitian_mode_weight

    full_grid, _half_grid = _grids(ny)
    two_sided = np.asarray(_hermitian_mode_weight(full_grid, use_dealias=False))[:, 0]
    np.testing.assert_array_equal(two_sided, np.ones(ny))


@pytest.mark.parametrize("ny", EVEN_NY)
def test_dealiasing_still_zeroes_the_nyquist_row_in_both_weights(ny: int) -> None:
    """Why no shipped number moves: the row the rule changed is masked away."""

    from gkx.operators.moments import _hermitian_mode_weight, _transport_mode_weight

    row = nyquist_row(ny)
    assert row is not None
    _full_grid, half_grid = _grids(ny)
    for weight in (_hermitian_mode_weight, _transport_mode_weight):
        masked = np.asarray(weight(half_grid, use_dealias=True))
        np.testing.assert_array_equal(masked[row], np.zeros(masked.shape[1]))


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_flux_weight_folds_the_pair_and_halves_a_self_conjugate_row(
    ny: int,
) -> None:
    """``_transport_mode_weight`` picks one representative per conjugate pair.

    The flux kernels carry the factor of two themselves, so a row that stands
    for itself and its unstored partner takes weight 1 and a self-conjugate row
    takes 0.5.

    The two-sided convention now says the same thing (plan 5.3 N3, the decision
    Q24 handed to the state switch): an even grid's Nyquist row is a
    representative there too, at the same 0.5, so the two-sided weights are the
    half-axis weights padded with zeros on the rows the half axis does not
    store.  Before the switch that row was dropped from the flux on a two-sided
    axis and counted on a half one, which made the flux depend on the layout.
    """

    from gkx.operators.moments import _transport_mode_weight

    full_grid, half_grid = _grids(ny)
    half = np.asarray(_transport_mode_weight(half_grid, use_dealias=False))[:, 0]
    expected = ky_row_weights(ny) / 2.0
    expected[0] = 0.0
    np.testing.assert_array_equal(half, expected)

    two_sided = np.asarray(_transport_mode_weight(full_grid, use_dealias=False))[:, 0]
    padded = np.zeros(ny, dtype=two_sided.dtype)
    padded[: expected.size] = expected
    np.testing.assert_array_equal(two_sided, padded)


def test_a_selected_subset_of_modes_does_not_claim_a_nyquist_row() -> None:
    """An index into a mode selection would name the wrong row, so it is not used."""

    from gkx.core_grid import select_ky_grid, select_real_fft_ky_grid
    from gkx.operators.moments import _hermitian_mode_weight

    full_grid, half_grid = _grids(8)
    assert full_grid.ny_full == 8 and half_grid.ny_full == 8

    # Fewer rows than the half block: not a complete axis, so no parent length.
    subset = select_ky_grid(full_grid, [0, 1, 2, 3])
    assert subset.ny_full is None
    np.testing.assert_array_equal(
        np.asarray(_hermitian_mode_weight(subset, use_dealias=False))[:, 0],
        np.array([1.0, 2.0, 2.0, 2.0]),
    )

    # As many rows as the half block, but another code's wave numbers rather
    # than this grid's: the value check is what keeps row 4 from being called
    # a Nyquist row it is not.
    dump = select_real_fft_ky_grid(full_grid, np.array([0.0, 1.0, 2.0, 3.0, 5.0]))
    assert dump.ny_full is None
    np.testing.assert_array_equal(
        np.asarray(_hermitian_mode_weight(dump, use_dealias=False))[:, 0],
        np.array([1.0, 2.0, 2.0, 2.0, 2.0]),
    )


def test_the_cached_weight_and_the_quasilinear_weight_are_the_same_rule() -> None:
    """The third and fourth copies of the rule now read from one owner."""

    from types import SimpleNamespace

    from gkx.diagnostics.quasilinear_transport import spectral_phi_weights
    from gkx.operators.moments import _cached_hermitian_mode_weight

    ny, nx = 8, 2
    _full_grid, half_grid = _grids(ny, nx=nx)
    cache = SimpleNamespace(
        ky=half_grid.ky,
        kx=half_grid.kx,
        dealias_mask=half_grid.dealias_mask,
        ny_full=half_grid.ny_full,
    )
    cached = np.asarray(_cached_hermitian_mode_weight(cache, use_dealias=False))
    np.testing.assert_array_equal(cached[:, 0], ky_row_weights(ny))

    nz = 3
    phi = jnp.ones((cached.shape[0], nx, nz), dtype=jnp.complex64)
    vol_fac = jnp.ones((nz,), dtype=jnp.float32) / nz
    weights = np.asarray(spectral_phi_weights(phi, cache, vol_fac, use_dealias=False))
    # ``vol_fac`` sums to one, so the z sum of the quasilinear weight of a
    # unit field is exactly the cached ``(ky, kx)`` weight.
    np.testing.assert_allclose(weights.sum(axis=2), cached, rtol=1e-6, atol=1e-7)


# ---- from test_ky_half_spectrum_state.py ----
# The evolved state on the ``ky >= 0`` half spectrum (plan 5.3 N3).
#
# Stage 1 (#248) wrote the layout contract and split the bracket's ``ky >= 0``
# primitive out as a seam; the state itself stayed two-sided, so none of the
# Hermitian completion was recovered.  This file pins the switch: a spectral grid
# built in :data:`~gkx.core_ky_layout.HALF` carries ``Nyc = 1 + Ny // 2`` rows, an


EVEN_NY_KY_HALF_SPECTRUM_STATE = (8, 12, 16)
ODD_NY_KY_HALF_SPECTRUM_STATE = (9, 15)


# --------------------------------------------------------------------------
# the layout predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ny", EVEN_NY_KY_HALF_SPECTRUM_STATE + ODD_NY_KY_HALF_SPECTRUM_STATE
)
def test_the_layout_of_an_axis_is_read_from_ny_full_not_guessed(ny: int) -> None:
    """``Nyc`` does not determine ``Ny``, so the row count alone cannot decide."""

    nyc = nyc_from_ny(ny)
    assert ky_layout_of(ny, ny) == FULL
    assert ky_layout_of(nyc, ny) == HALF
    assert rows_for_layout(ny, FULL) == ny
    assert rows_for_layout(ny, HALF) == nyc
    # A grid that carries no parent length is a selection of modes and keeps
    # the pre-contract reading, which is what every consumer assumed.
    assert ky_layout_of(nyc, None) == FULL
    assert not is_half(nyc, None)
    with pytest.raises(ValueError, match="neither the full axis"):
        ky_layout_of(nyc + 1, ny)


@pytest.mark.parametrize("ny", (1, 2))
def test_a_degenerate_axis_is_called_full_because_the_layouts_coincide(
    ny: int,
) -> None:
    """``Nyc == Ny`` there, and every half/full branch is the identity."""

    assert nyc_from_ny(ny) == ny
    assert ky_layout_of(ny, ny) == FULL


# --------------------------------------------------------------------------
# the grid
# --------------------------------------------------------------------------


def _grid_cfg(ny: int, nx: int = 8, nz: int = 8) -> GridConfig:
    """A deck for the layout comparisons, with the two-sided axis named.

    Every test below builds both layouts from one config and compares them, so
    neither arm may come from a default: if the default ever changes, a
    "full" arm that silently became half would turn each of these comparisons
    into a tautology that passes.
    """

    return GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=62.8, Ly=62.8, ky_layout=FULL)


@pytest.mark.parametrize(
    "ny", EVEN_NY_KY_HALF_SPECTRUM_STATE + ODD_NY_KY_HALF_SPECTRUM_STATE
)
def test_a_half_grid_stores_nyc_rows_and_still_knows_its_full_length(
    ny: int,
) -> None:
    cfg = _grid_cfg(ny)
    full = build_spectral_grid(cfg)
    half = build_spectral_grid(cfg, ky_layout=HALF)

    assert full.ky_layout == FULL and half.ky_layout == HALF
    assert int(full.ky.shape[0]) == ny
    assert int(half.ky.shape[0]) == nyc_from_ny(ny)
    assert full.ny_full == ny and half.ny_full == ny
    assert source_ny_full(half) == ny

    # The half axis is the magnitudes of the non-negative block, which puts the
    # Nyquist row at +Ny/2 where fftfreq stores it as -Ny/2.
    np.testing.assert_array_equal(
        np.asarray(half.ky), np.abs(np.asarray(full.ky)[: nyc_from_ny(ny)])
    )
    assert np.all(np.asarray(half.ky) >= 0.0)


@pytest.mark.parametrize(
    "ny", EVEN_NY_KY_HALF_SPECTRUM_STATE + ODD_NY_KY_HALF_SPECTRUM_STATE
)
def test_the_half_dealias_mask_is_built_not_sliced_and_matches_the_slice(
    ny: int,
) -> None:
    """The rows agree; stating them is what stops the agreement being luck."""

    nx = 8
    sliced = np.asarray(twothirds_mask(ny, nx))[: nyc_from_ny(ny)]
    built = np.asarray(twothirds_mask(ny, nx, ky_layout=HALF))
    np.testing.assert_array_equal(built, sliced)


@pytest.mark.parametrize("ny", EVEN_NY_KY_HALF_SPECTRUM_STATE)
def test_dealiasing_removes_the_nyquist_row_which_is_why_its_sign_cannot_matter(
    ny: int,
) -> None:
    row = nyquist_row(ny)
    assert row is not None
    mask = np.asarray(twothirds_mask(ny, 8, ky_layout=HALF))
    np.testing.assert_array_equal(mask[row], np.zeros(8, dtype=bool))


# --------------------------------------------------------------------------
# the evolved state
# --------------------------------------------------------------------------


def _geometry() -> SAlphaGeometry:
    return SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)


def _params() -> LinearParams:
    return LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)


def _states(ny: int, nx: int, nz: int, nl: int, nm: int, seed: int):
    """Return ``(half_state, full_state)`` for the same physical field.

    The full state is the widening of the half one, so the pair represents one
    field in two layouts rather than two different fields.
    """

    nyc = nyc_from_ny(ny)
    rng = np.random.default_rng(seed)
    shape = (1, nl, nm, nyc, nx, nz)
    half = jnp.asarray(
        1e-3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)),
        jnp.complex128,
    )
    return half, to_full(half, ny_full=ny)


def _caches(ny: int, nx: int, nz: int, nl: int, nm: int):
    cfg = _grid_cfg(ny, nx=nx, nz=nz)
    geom, params = _geometry(), _params()
    full_grid = build_spectral_grid(cfg)
    half_grid = build_spectral_grid(cfg, ky_layout=HALF)
    return (
        full_grid,
        half_grid,
        build_linear_cache(full_grid, geom, params, nl, nm),
        build_linear_cache(half_grid, geom, params, nl, nm),
        params,
    )


def _dealiased_rows(ny: int) -> int:
    """Rows the two-thirds mask keeps, counting from ``ky = 0`` upward."""

    return 1 + (ny - 1) // 3


@pytest.mark.parametrize("ny", (8, 12, 9))
def test_the_linear_rhs_is_bitwise_between_the_layouts_on_every_dealiased_row(
    ny: int,
) -> None:
    """The linear operator is block diagonal in ``ky``: same row, same numbers.

    Bitwise, not close: no arithmetic changes between the layouts on these
    rows, only how many of them are stored.  The Nyquist row is excluded and
    gets its own test.
    """

    nx, nz, nl, nm = 8, 8, 2, 4
    full_grid, _half_grid, full_cache, half_cache, params = _caches(ny, nx, nz, nl, nm)
    half_state, full_state = _states(ny, nx, nz, nl, nm, seed=20260919)
    terms = TermConfig(nonlinear=0.0)

    out_full = np.asarray(
        assemble_rhs_cached(full_state, full_cache, params, terms=terms)[0]
    )
    out_half = np.asarray(
        assemble_rhs_cached(half_state, half_cache, params, terms=terms)[0]
    )
    assert out_half.shape[-3] == nyc_from_ny(ny)

    reference = np.asarray(to_half(out_full, ny_full=ny))
    keep = _dealiased_rows(ny)
    np.testing.assert_array_equal(
        out_half[..., :keep, :, :], reference[..., :keep, :, :]
    )
    del full_grid


@pytest.mark.parametrize("ny", (8, 12))
def test_only_the_nyquist_row_differs_and_only_by_its_ky_sign(ny: int) -> None:
    """The one row the layouts disagree on, and the reason, stated as a test.

    ``fftfreq`` stores ``|ky| = Ny/2`` as ``-Ny/2`` and the half axis as
    ``+Ny/2``.  The omega-star term carries a factor ``i * ky``, so that row's
    linear RHS flips sign with the convention while every other row is
    untouched.  It never reaches a run: the two-thirds mask zeroes it.
    """

    nx, nz, nl, nm = 8, 8, 2, 4
    full_grid, half_grid, full_cache, half_cache, params = _caches(ny, nx, nz, nl, nm)
    row = nyquist_row(ny)
    assert row is not None
    assert float(np.asarray(full_grid.ky)[row]) < 0.0
    assert float(np.asarray(half_grid.ky)[row]) > 0.0

    half_state, full_state = _states(ny, nx, nz, nl, nm, seed=5)
    terms = TermConfig(nonlinear=0.0)
    out_full = np.asarray(
        assemble_rhs_cached(full_state, full_cache, params, terms=terms)[0]
    )
    out_half = np.asarray(
        assemble_rhs_cached(half_state, half_cache, params, terms=terms)[0]
    )
    reference = np.asarray(to_half(out_full, ny_full=ny))

    # Every row below the dealias cutoff agrees exactly ...
    keep = _dealiased_rows(ny)
    np.testing.assert_array_equal(
        out_half[..., :keep, :, :], reference[..., :keep, :, :]
    )
    # ... and the Nyquist row does not, which is the whole of the difference.
    assert not np.array_equal(out_half[..., row, :, :], reference[..., row, :, :])


@pytest.mark.parametrize("ny", (8, 12, 9))
def test_the_nonlinear_rhs_agrees_to_roundoff_between_the_layouts(ny: int) -> None:
    """The bracket computes on ``ky >= 0`` either way; only the widening moves.

    Not bitwise: a half-spectrum operand changes the shape of the batch handed
    to ``irfft2``/``rfft2``, and the XLA:CPU FFT's reduction order depends on
    it.  The difference is at the level of the transform's own round-off.
    """

    from gkx.solvers_nonlinear_state_integration import nonlinear_rhs_cached

    nx, nz, nl, nm = 8, 8, 2, 4
    _full_grid, _half_grid, full_cache, half_cache, params = _caches(ny, nx, nz, nl, nm)
    half_state, full_state = _states(ny, nx, nz, nl, nm, seed=11)
    terms = TermConfig(nonlinear=1.0)

    out_full = np.asarray(
        nonlinear_rhs_cached(
            full_state, full_cache, params, terms, compressed_real_fft=True
        )[0]
    )
    out_half = np.asarray(
        nonlinear_rhs_cached(
            half_state, half_cache, params, terms, compressed_real_fft=True
        )[0]
    )
    reference = np.asarray(to_half(out_full, ny_full=ny))
    keep = _dealiased_rows(ny)
    scale = np.linalg.norm(reference[..., :keep, :, :])
    delta = np.linalg.norm(out_half[..., :keep, :, :] - reference[..., :keep, :, :])
    assert delta / scale < 1e-13


@pytest.mark.parametrize("ny", (8, 12))
def test_the_bracket_output_is_the_widening_of_its_own_half_block(ny: int) -> None:
    """What the per-stage projector used to restore, the bracket already gives.

    On the two-sided route the bracket computes ``Nyc`` rows and widens them,
    so its output satisfies the reality condition by construction; the
    projector after each stage was restoring a property the bracket had not
    broken.  (The *assembled* RHS is a weaker statement: end damping selects
    ``ky > 0`` and so leaves the stored negative rows undamped, which is why
    this test is on the bracket and not on the total.)
    """

    from gkx.operators.nonlinear.brackets import _spectral_bracket_real_fft

    nyc, nx, nz = nyc_from_ny(ny), 8, 4
    rng = np.random.default_rng(3)
    half = jnp.asarray(
        rng.standard_normal((2, nyc, nx, nz))
        + 1j * rng.standard_normal((2, nyc, nx, nz)),
        jnp.complex128,
    )
    full = to_full(half, ny_full=ny)
    ky_grid = jnp.broadcast_to(jnp.asarray(np.fft.fftfreq(ny))[:, None], (ny, nx))
    kx_grid = jnp.broadcast_to(jnp.asarray(np.fft.fftfreq(nx))[None, :], (ny, nx))
    out = _spectral_bracket_real_fft(
        full,
        full,
        kx_grid=kx_grid,
        ky_grid=ky_grid,
        dealias_mask=twothirds_mask(ny, nx),
        kxfac=jnp.asarray(1.0),
    )
    rebuilt = to_full(to_half(out, ny_full=ny), ny_full=ny)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(rebuilt))


# --------------------------------------------------------------------------
# the per-stage projector
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ny", EVEN_NY_KY_HALF_SPECTRUM_STATE + ODD_NY_KY_HALF_SPECTRUM_STATE
)
def test_the_per_stage_projector_is_the_identity_on_a_half_state(ny: int) -> None:
    """The 41.9%: ``to_full(to_half(G))`` has nothing left to do.

    It is the identity *exactly*, not approximately.  The projector never
    imposed the self-conjugate rows' internal constraint in either layout -- it
    rebuilt the unstored rows and left the stored ones alone -- so on a state
    that stores only the ``ky >= 0`` rows there is no work in it at all.
    """

    from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector

    nyc = nyc_from_ny(ny)
    rng = np.random.default_rng(1234)
    block = jnp.asarray(
        rng.standard_normal((2, nyc, 8, 4)) + 1j * rng.standard_normal((2, nyc, 8, 4)),
        jnp.complex128,
    )
    half_project = _make_compressed_real_fft_projector(ny_full=ny, nx=8, rows=nyc)
    assert half_project(block) is block

    full_project = _make_compressed_real_fft_projector(ny_full=ny, nx=8, rows=ny)
    widened = to_full(block, ny_full=ny)
    np.testing.assert_array_equal(
        np.asarray(full_project(widened)), np.asarray(widened)
    )


# --------------------------------------------------------------------------
# the blockers stage 1 listed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (12, 16, 15))
def test_the_linked_chains_cover_the_same_physical_modes_in_both_layouts(
    ny: int,
) -> None:
    """``naky`` counts dealiased rows of the *two-sided* axis, not stored rows.

    Reading it off the stored count would build chains for ``1 + (Nyc-1)//3``
    rows instead of ``1 + (Ny-1)//3`` and leave the rest of the band with no
    parallel derivative, silently.
    """

    from gkx.operators.linear.linked import _build_linked_fft_maps

    kx = np.asarray(np.fft.fftfreq(8) * 2.0 * np.pi, dtype=float)
    ky_full = np.asarray(np.fft.fftfreq(ny) * 2.0 * np.pi, dtype=float)
    ky_half = np.abs(ky_full[: nyc_from_ny(ny)])

    idx_full, _kz_full = _build_linked_fft_maps(
        kx, ky_full, 1.0, 1, 0.1, 8, np.float64, None, ny
    )
    idx_half, _kz_half = _build_linked_fft_maps(
        kx, ky_half, 1.0, 1, 0.1, 8, np.float64, None, ny
    )
    assert len(idx_full) == len(idx_half)
    for a, b in zip(idx_full, idx_half):
        # Same (ky, kx) pairs; only the flat encoding's modulus differs.
        ky_a, kx_a = np.asarray(a) % ny, np.asarray(a) // ny
        ky_b, kx_b = np.asarray(b) % nyc_from_ny(ny), np.asarray(b) // nyc_from_ny(ny)
        np.testing.assert_array_equal(ky_a, ky_b)
        np.testing.assert_array_equal(kx_a, kx_b)


@pytest.mark.parametrize("ny", (12, 16))
def test_the_conjugate_restore_is_the_identity_on_a_half_state(ny: int) -> None:
    """``(-j) % Nyc`` is a different positive row, not a partner.

    Running the two-sided fill on a half axis would conjugate-mirror one
    physical mode onto another, so the restore has to *know* which layout it is
    on rather than being handed a row count.
    """

    from gkx.operators.linear.streaming import _restore_linked_real_fft_conjugates

    nyc = nyc_from_ny(ny)
    rng = np.random.default_rng(99)
    out = jnp.asarray(
        rng.standard_normal((2, nyc, 8, 4)) + 1j * rng.standard_normal((2, nyc, 8, 4)),
        jnp.complex128,
    )
    covered = jnp.asarray(
        np.arange(nyc) < _dealiased_rows(ny)
    )  # what the chains actually visit
    assert (
        _restore_linked_real_fft_conjugates(out, covered_rows=covered, ny_full=ny)
        is out
    )
    # On the two-sided axis the same call still fills the negative rows.
    widened = to_full(out, ny_full=ny)
    covered_full = jnp.asarray(np.arange(ny) < _dealiased_rows(ny))
    restored = _restore_linked_real_fft_conjugates(
        widened, covered_rows=covered_full, ny_full=ny
    )
    assert restored is not widened


@pytest.mark.parametrize("ny", (12, 16, 15))
def test_the_hyperdiffusion_cutoff_is_the_same_wavenumber_in_both_layouts(
    ny: int,
) -> None:
    """``kperp2_max`` normalizes ``Dfac``; halving it inflates every rate.

    Deriving the cutoff row from the stored count would put it at
    ``(Nyc-1)//3`` -- roughly half the true ``ky`` -- and multiply ``Dfac`` by
    ``4 ** p_hyper_kperp`` with no shape error to show for it.
    """

    from gkx.operators.linear.dissipation import hyperdiffusion_contribution

    nx = 8
    ky_full = jnp.asarray(np.fft.fftfreq(ny) * 2.0 * np.pi)
    ky_half = jnp.abs(ky_full[: nyc_from_ny(ny)])
    kx = jnp.asarray(np.fft.fftfreq(nx) * 2.0 * np.pi)
    rng = np.random.default_rng(4)
    nyc = nyc_from_ny(ny)
    block = jnp.asarray(
        rng.standard_normal((1, 2, 2, nyc, nx, 3))
        + 1j * rng.standard_normal((1, 2, 2, nyc, nx, 3)),
        jnp.complex128,
    )
    kwargs = dict(
        kx=kx,
        D_hyper=jnp.asarray(1.0),
        p_hyper_kperp=jnp.asarray(2.0),
        weight=jnp.asarray(1.0),
    )
    out_half = hyperdiffusion_contribution(
        block,
        ky=ky_half,
        ny_full=ny,
        dealias_mask=twothirds_mask(ny, nx, ky_layout=HALF),
        **kwargs,
    )
    out_full = hyperdiffusion_contribution(
        to_full(block, ny_full=ny),
        ky=ky_full,
        ny_full=ny,
        dealias_mask=twothirds_mask(ny, nx),
        **kwargs,
    )
    np.testing.assert_allclose(
        np.asarray(out_half),
        np.asarray(to_half(out_full, ny_full=ny)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("ny", EVEN_NY_KY_HALF_SPECTRUM_STATE)
def test_the_full_complex_bracket_refuses_a_half_spectrum_operand(ny: int) -> None:
    """It multiplies the whole two-sided spectrum, so it has no half form."""

    from gkx.operators.nonlinear.brackets import _spectral_bracket_full_core

    nyc, nx = nyc_from_ny(ny), 8
    ky_grid = jnp.broadcast_to(
        jnp.abs(jnp.asarray(np.fft.fftfreq(ny)))[:nyc, None], (nyc, nx)
    )
    kx_grid = jnp.broadcast_to(jnp.asarray(np.fft.fftfreq(nx))[None, :], (nyc, nx))
    block = jnp.zeros((2, nyc, nx, 3), jnp.complex128)
    with pytest.raises(ValueError, match="needs the two-sided ky axis"):
        _spectral_bracket_full_core(
            block,
            block,
            kx_grid=kx_grid,
            ky_grid=ky_grid,
            dealias_mask=twothirds_mask(ny, nx, ky_layout=HALF),
            kxfac=jnp.asarray(1.0),
            ny_full=ny,
            multiple_fields=False,
        )


# --------------------------------------------------------------------------
# restart round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (12, 16, 15))
def test_the_restart_block_keeps_the_same_rows_whichever_layout_wrote_it(
    ny: int,
) -> None:
    """The file has always stored the dealiased ``ky >= 0`` block.

    That block is the same index range in both layouts, but its length is a
    property of ``Ny``: taken from the state's own row count on a half state it
    would silently keep about a third of the band.
    """

    from gkx.artifacts.spectral_layout import _restart_to_netcdf_layout

    nx, nz, nl, nm = 8, 6, 2, 3
    nyc = nyc_from_ny(ny)
    rng = np.random.default_rng(77)
    half = (
        rng.standard_normal((1, nl, nm, nyc, nx, nz))
        + 1j * rng.standard_normal((1, nl, nm, nyc, nx, nz))
    ).astype(np.complex64)
    full = np.asarray(to_full(half, ny_full=ny))

    packed_full = _restart_to_netcdf_layout(full, ny_full=ny)
    packed_half = _restart_to_netcdf_layout(half, ny_full=ny)
    assert packed_half.shape == packed_full.shape
    assert packed_half.shape[5] == _dealiased_rows(ny)
    np.testing.assert_array_equal(packed_half, packed_full)

    # Without ny_full a half state is read as if it were the full axis, which
    # is the silent truncation the argument exists to prevent.
    truncated = _restart_to_netcdf_layout(half)
    assert truncated.shape[5] == _dealiased_rows(nyc)
    assert truncated.shape[5] < packed_half.shape[5]


@pytest.mark.parametrize("ny", (12, 16))
def test_a_half_state_survives_the_binary_restart_round_trip(ny: int) -> None:
    """``Nyc`` rows out, the same ``Nyc`` rows back, bitwise."""

    nyc, nx, nz, nl, nm = nyc_from_ny(ny), 8, 6, 2, 3
    rng = np.random.default_rng(31)
    half = (
        rng.standard_normal((1, nl, nm, nyc, nx, nz))
        + 1j * rng.standard_normal((1, nl, nm, nyc, nx, nz))
    ).astype(np.complex64)
    widened = np.asarray(to_full(half, ny_full=ny))
    np.testing.assert_array_equal(np.asarray(to_half(widened, ny_full=ny)), half)


# --------------------------------------------------------------------------
# the real-space output paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 12, 9))
def test_the_real_space_writer_gives_the_same_image_from_either_layout(
    ny: int,
) -> None:
    """``irfft2`` of the stored rows, not ``real(ifft2)`` of ``Nyc`` of them.

    Taking the complex transform of a half-spectrum field would return an
    ``Nyc``-row image against an ``Ny``-long ``y`` axis -- a wrong picture with
    the right dtype.
    """

    from gkx.artifacts.spectral_layout import _spectral_to_xy

    nyc, nx, nz = nyc_from_ny(ny), 8, 4
    rng = np.random.default_rng(6)
    half = (
        rng.standard_normal((nyc, nx, nz)) + 1j * rng.standard_normal((nyc, nx, nz))
    ).astype(np.complex128)
    full = np.asarray(to_full(half, ny_full=ny))

    from_full = _spectral_to_xy(full)
    from_half = _spectral_to_xy(half, ny_full=ny)
    assert from_half.shape == from_full.shape == (ny, nx, nz)
    np.testing.assert_allclose(from_half, from_full, rtol=0, atol=2e-6)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 9))
def test_the_linear_rhs_gradient_matches_between_the_layouts(ny: int) -> None:
    """A cotangent on the stored rows pulls back the same way in either layout.

    The two-sided run's cotangent is the widening of the half one, so the two
    scalars are the same function of the same field; their gradients must agree
    on the rows both layouts store.
    """

    nx, nz, nl, nm = 8, 6, 2, 3
    _fg, _hg, full_cache, half_cache, params = _caches(ny, nx, nz, nl, nm)
    half_state, full_state = _states(ny, nx, nz, nl, nm, seed=808)
    terms = TermConfig(nonlinear=0.0)
    keep = _dealiased_rows(ny)

    def scalar(state, cache):
        rhs = assemble_rhs_cached(state, cache, params, terms=terms)[0]
        return jnp.sum(jnp.abs(rhs[..., :keep, :, :]) ** 2)

    g_half = np.asarray(jax.grad(scalar)(half_state, half_cache))
    g_full = np.asarray(jax.grad(scalar)(full_state, full_cache))
    np.testing.assert_allclose(
        g_half[..., :keep, :, :],
        np.asarray(to_half(g_full, ny_full=ny))[..., :keep, :, :],
        rtol=1e-12,
        atol=1e-18,
    )


# --------------------------------------------------------------------------
# supplied-state intake (#247, #253)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (12, 16))
def test_the_supplied_state_intake_reaches_the_same_modes_in_both_layouts(
    ny: int,
) -> None:
    """The chain cover is the same physical mode set, however it is stored.

    The cover mask is built from the chain membership indexed ``ky + ny * kx``
    and then, on a two-sided axis, widened by the conjugate mirror so a row no
    chain visits is kept when its ``-ky`` partner is.  ``(-j) % Nyc`` names an
    unrelated positive row, so on a half axis the mirror is inapplicable rather
    than merely unnecessary: running it would keep rows that carry no physics
    and zero rows that do.
    """

    from gkx.operators.linear.cache_builder import linked_chain_cover_mask

    cfg = GridConfig(
        Nx=8,
        Ny=ny,
        Nz=8,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        jtwist=1,
        ky_layout=FULL,
    )
    geom, params = _geometry(), _params()
    full_grid = build_spectral_grid(cfg)
    half_grid = build_spectral_grid(cfg, ky_layout=HALF)

    full_mask = linked_chain_cover_mask(full_grid, geom, params)
    half_mask = linked_chain_cover_mask(half_grid, geom, params)
    if full_mask is None:  # periodic or full-cover deck: nothing to compare
        assert half_mask is None
        return
    full_mask = np.asarray(full_mask)
    half_mask = np.asarray(half_mask)
    assert half_mask.shape == (nyc_from_ny(ny), 8)
    # The half cover is the two-sided cover's non-negative rows.
    np.testing.assert_array_equal(half_mask, full_mask[: nyc_from_ny(ny)])


@pytest.mark.parametrize("ny", (12, 16))
def test_masking_a_supplied_half_state_keeps_the_chain_rows_untouched(
    ny: int,
) -> None:
    """Intake is a ``where`` against a fixed mask, so it traces either way."""

    from gkx.operators.linear.cache_builder import (
        linked_chain_cover_mask,
        mask_off_chain_rows,
    )

    cfg = GridConfig(
        Nx=8,
        Ny=ny,
        Nz=8,
        Lx=62.8,
        Ly=62.8,
        boundary="linked",
        jtwist=1,
        ky_layout=FULL,
    )
    geom, params = _geometry(), _params()
    half_grid = build_spectral_grid(cfg, ky_layout=HALF)
    linked = linked_chain_cover_mask(half_grid, geom, params)
    nyc = nyc_from_ny(ny)
    rng = np.random.default_rng(404)
    state = jnp.asarray(
        rng.standard_normal((1, 2, 3, nyc, 8, 8))
        + 1j * rng.standard_normal((1, 2, 3, nyc, 8, 8)),
        jnp.complex128,
    )
    out = np.asarray(mask_off_chain_rows(state, half_grid, geom, params))
    assert out.shape == (1, 2, 3, nyc, 8, 8)
    if linked is None:
        np.testing.assert_array_equal(out, np.asarray(state))
        return
    mask = np.asarray(linked)
    kept = np.where(mask[None, None, None, :, :, None], np.asarray(state), 0.0)
    np.testing.assert_array_equal(out, kept)


# ---- from fix/half-layout-cfl (#306) ----


@pytest.mark.parametrize("ny", (12, 16, 15))
def test_the_explicit_cfl_bound_is_the_same_in_both_layouts(ny: int) -> None:
    """The CFL reads the two-sided extent, not the stored row count.

    Slicing an already-half axis again, or indexing it at ``(Nyc-1)//3``,
    halves ``ky_max`` and lets the adaptive step run past the true CFL.
    """

    from gkx.operators.nonlinear.policies import (
        _build_nonlinear_cfl_bounds,
        _nonlinear_cfl_frequency_components,
    )
    from gkx.solvers_time_explicit_cfl import (
        _grid_frequency_bounds,
        _laguerre_velocity_max,
        _linear_frequency_bound,
    )
    from gkx.terms.config import FieldState

    nx, nz, nl, nm = 8, 4, 2, 2
    full_grid, half_grid, full_cache, half_cache, params = _caches(ny, nx, nz, nl, nm)
    geom = _geometry()
    bf, bh = _grid_frequency_bounds(full_grid), _grid_frequency_bounds(half_grid)
    np.testing.assert_array_equal(bh.ky, bf.ky)
    assert bh.ky_max == bf.ky_max
    np.testing.assert_array_equal(
        _linear_frequency_bound(half_grid, geom, params, nl, nm),
        _linear_frequency_bound(full_grid, geom, params, nl, nm),
    )

    bounds = [
        _build_nonlinear_cfl_bounds(
            grid,
            geom,
            params,
            cache,
            real_dtype=jnp.float64,
            linear_frequency_bound_fn=_linear_frequency_bound,
            laguerre_velocity_max_fn=_laguerre_velocity_max,
        )
        for grid, cache in ((full_grid, full_cache), (half_grid, half_cache))
    ]
    assert float(bounds[1].ky_max) == float(bounds[0].ky_max)
    np.testing.assert_array_equal(bounds[1].linear_omega, bounds[0].linear_omega)

    half, full = _states(ny, nx, nz, 1, 1, seed=7)
    kwargs = dict(
        compressed_real_fft=True,
        kx_max=bounds[0].kx_max,
        ky_max=bounds[0].ky_max,
        kxfac=1.0,
        vpar_max=1.0,
        muB_max=1.0,
    )
    omega_full = _nonlinear_cfl_frequency_components(
        FieldState(phi=full[0, 0, 0], apar=None, bpar=None),
        full_grid,
        full_cache,
        **kwargs,
    )
    omega_half = _nonlinear_cfl_frequency_components(
        FieldState(phi=half[0, 0, 0], apar=None, bpar=None),
        half_grid,
        half_cache,
        **kwargs,
    )
    np.testing.assert_allclose(
        np.asarray(omega_half), np.asarray(omega_full), rtol=1e-12, atol=0.0
    )
