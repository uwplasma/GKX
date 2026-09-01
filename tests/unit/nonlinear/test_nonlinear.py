"""Core nonlinear solver, RHS, and diagnostic-state tests."""

from __future__ import annotations

from dataclasses import replace
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.diagnostics.moments import fieldline_quadrature_weights
from gkx.diagnostics.transport import heat_flux_total
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    compute_nonlinear_diagnostic_tuple,
    make_nonlinear_diagnostic_tuple_fn,
)
from gkx.operators.nonlinear.policies import build_nonlinear_imex_operator
from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector
from gkx.operators.nonlinear.rhs import (
    linear_rhs_jit_for_terms_impl,
    nonlinear_em_term_cached_impl,
    nonlinear_rhs_cached_impl,
)
from gkx.solvers_nonlinear_diagnostic_integration import (
    integrate_nonlinear_explicit_diagnostics,
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.solvers_nonlinear_state_integration import (
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_imex_cached,
    nonlinear_heat_flux_window,
)
from gkx.solvers_time_explicit import _linear_frequency_bound
from gkx.terms.config import FieldState
from gkx.terms.config import TermConfig
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_integrate_nonlinear_checkpoint_runs():
    """Checkpointed nonlinear integration should run on a tiny grid."""
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=1.0)
    _, fields_t = integrate_nonlinear(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk4",
        terms=terms,
        checkpoint=True,
    )
    assert fields_t.phi.shape[0] == 2


def test_explicit_nonlinear_integrator_applies_custom_collision_each_step():
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G0 = jnp.ones((1, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64)

    class DragCollision:
        def apply(self, context):
            return -3.0 * context.distribution

    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.25,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        nonlinear=0.0,
        apar=0.0,
        bpar=0.0,
    )
    G_final = integrate_nonlinear(
        G0,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="euler",
        terms=terms,
        compressed_real_fft=False,
        return_fields=False,
        collision_operator=DragCollision(),
    )
    np.testing.assert_allclose(np.asarray(G_final), 0.925**2, atol=1.0e-6)


def test_nonlinear_imex_reuses_prebuilt_operator():
    """Prebuilt IMEX operator should be reusable for the same state shape."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    terms = TermConfig(nonlinear=0.0)
    op = build_nonlinear_imex_operator(
        G,
        cache,
        params,
        dt=0.05,
        terms=terms,
        implicit_preconditioner="damping",
    )
    G_out, fields_t = integrate_nonlinear_imex_cached(
        G,
        cache,
        params,
        dt=0.05,
        steps=2,
        terms=terms,
        implicit_operator=op,
    )
    assert G_out.shape == G.shape
    assert fields_t.phi.shape[0] == 2


@pytest.mark.parametrize("checkpoint", [False, True])
def test_nonlinear_imex_state_gradient_matches_finite_difference(
    checkpoint: bool,
) -> None:
    """Implicit GMRES VJPs should differentiate the solved system, not iterations."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    direction = jnp.ones(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    ) * jnp.asarray(1.0e-7, dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    terms = TermConfig(nonlinear=0.0)
    operator = build_nonlinear_imex_operator(
        direction,
        cache,
        params,
        dt=0.05,
        terms=terms,
        implicit_preconditioner="damping",
    )

    def final_energy(scale: jnp.ndarray) -> jnp.ndarray:
        final_state, _fields = integrate_nonlinear_imex_cached(
            scale * direction,
            cache,
            params,
            dt=0.05,
            steps=2,
            terms=terms,
            implicit_operator=operator,
            implicit_maxiter=20,
            checkpoint=checkpoint,
        )
        return jnp.real(jnp.vdot(final_state, final_state))

    value, gradient = jax.value_and_grad(final_energy)(jnp.asarray(1.0))
    step = jnp.asarray(1.0e-2)
    centered_fd = (final_energy(1.0 + step) - final_energy(1.0 - step)) / (2 * step)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=5.0e-2, atol=1.0e-16
    )


def test_nonlinear_imex_parameter_gradient_rebuilds_operator() -> None:
    """Implicit VJPs should include parameter dependence of the matrix operator."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    base_params = LinearParams()
    initial_state = jnp.ones(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    ) * jnp.asarray(1.0e-7, dtype=jnp.complex64)
    terms = TermConfig(nonlinear=0.0)

    def final_energy(rlt: jnp.ndarray) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
        operator = build_nonlinear_imex_operator(
            initial_state,
            cache,
            params,
            dt=0.05,
            terms=terms,
            implicit_preconditioner="damping",
        )
        final_state, _fields = integrate_nonlinear_imex_cached(
            initial_state,
            cache,
            params,
            dt=0.05,
            steps=2,
            terms=terms,
            implicit_operator=operator,
            implicit_maxiter=20,
        )
        return jnp.real(jnp.vdot(final_state, final_state))

    value, gradient = jax.value_and_grad(final_energy)(jnp.asarray(6.9))
    step = jnp.asarray(0.05)
    centered_fd = (final_energy(6.9 + step) - final_energy(6.9 - step)) / (2 * step)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-16
    )


def test_nonlinear_imex_heat_flux_gradient_matches_finite_difference() -> None:
    """Implicit VJPs should differentiate a physical endpoint heat flux."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base_params = LinearParams()
    initial_state = jnp.zeros(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    )
    profile = 1.0e-3 * (1.0 + 0.2 * jnp.cos(grid.z))
    initial_state = initial_state.at[0, 0, 1, 0, :].set(
        profile * (1.0 + 0.35j * jnp.sin(grid.z))
    )
    initial_state = initial_state.at[1, 0, 1, 0, :].set(profile * (0.2 + 0.4j))
    terms = TermConfig(nonlinear=1.0)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geom, grid)

    def endpoint_heat_flux(rlt: jnp.ndarray, tolerance: float) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
        operator = build_nonlinear_imex_operator(
            initial_state,
            cache,
            params,
            dt=0.02,
            terms=terms,
            implicit_preconditioner="damping",
        )
        final_state, fields = integrate_nonlinear_imex_cached(
            initial_state,
            cache,
            params,
            dt=0.02,
            steps=3,
            terms=terms,
            implicit_operator=operator,
            implicit_preconditioner="damping",
            implicit_tol=tolerance,
            implicit_maxiter=30,
            compressed_real_fft=False,
        )
        return heat_flux_total(
            final_state,
            fields.phi[-1],
            fields.apar[-1],
            fields.bpar[-1],
            cache,
            grid,
            params,
            flux_factor,
        )

    rlt = jnp.asarray(6.9)
    value, gradient = jax.value_and_grad(endpoint_heat_flux, argnums=0)(rlt, 1.0e-6)
    step = jnp.asarray(0.05)
    centered_fd = (
        endpoint_heat_flux(rlt + step, 1.0e-6) - endpoint_heat_flux(rlt - step, 1.0e-6)
    ) / (2 * step)
    tight_gradient = jax.grad(endpoint_heat_flux, argnums=0)(rlt, 1.0e-7)

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(value) != 0.0
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-13
    )
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(tight_gradient), rtol=2.0e-3, atol=1.0e-13
    )


def test_integrate_nonlinear_explicit_diagnostics_shapes():
    """Nonlinear diagnostics should return time-series arrays."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)
    t, diag = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.1,
        steps=3,
        method="sspx3",
        terms=terms,
    )
    assert t.shape[0] == 3
    assert diag.energy_t.shape[0] == 3
    assert diag.heat_flux_species_t is not None
    assert diag.particle_flux_species_t is not None
    assert np.asarray(diag.heat_flux_species_t).shape == (3, 1)
    assert np.asarray(diag.particle_flux_species_t).shape == (3, 1)
    assert np.isfinite(np.asarray(diag.dt_mean))
    assert np.isfinite(np.asarray(diag.dt_t)).all()


def test_prepared_nonlinear_diagnostics_reuses_compiled_scan():
    """A prepared diagnostic simulation should compile once per state signature."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    state = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    prepared = prepare_nonlinear_explicit_diagnostics(
        state,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )

    # Count traces locally rather than reading ``_run_raw._cache_size()``. That
    # counter is process-wide jit bookkeeping, and in a long test session it
    # reads 0 for a scan that was in fact compiled once and reused, so the
    # assertion turned into a false alarm about whatever else ran first. The
    # wrapped body below runs once per trace and never on a cache hit, which is
    # the reuse claim itself rather than a proxy for it.
    traces = 0
    scan_raw = prepared._run_raw.__wrapped__

    def counting_scan_raw(initial_state):
        nonlocal traces
        traces += 1
        return scan_raw(initial_state)

    prepared = replace(prepared, _run_raw=jax.jit(counting_scan_raw))

    first = prepared.run()
    assert traces == 1
    second = prepared.run()
    # A different state of the same shape and dtype is the same signature.
    reused = prepared.run(jnp.full_like(prepared.initial_state, 1.0e-3))
    direct = integrate_nonlinear_explicit_diagnostics_state(
        state,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )

    assert traces == 1
    # Index 2 is the final state: the reused executable integrated the new
    # initial condition rather than replaying the first one.
    assert bool(jnp.all(jnp.isfinite(reused[2])))
    assert not bool(jnp.allclose(reused[2], first[2]))
    for first_value, second_value in zip(
        first[:1] + first[2:], second[:1] + second[2:]
    ):
        for first_leaf, second_leaf in zip(
            jax.tree_util.tree_leaves(first_value),
            jax.tree_util.tree_leaves(second_value),
        ):
            np.testing.assert_allclose(np.asarray(first_leaf), np.asarray(second_leaf))
    np.testing.assert_allclose(np.asarray(first[0]), np.asarray(direct[0]))
    np.testing.assert_allclose(
        np.asarray(first[1].heat_flux_t), np.asarray(direct[1].heat_flux_t)
    )
    np.testing.assert_allclose(np.asarray(first[2]), np.asarray(direct[2]))


def test_nonlinear_diagnostics_honors_time_horizon():
    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    state = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))

    t, diag, _state, _fields = integrate_nonlinear_explicit_diagnostics_state(
        state,
        grid,
        geom,
        LinearParams(),
        dt=0.1,
        steps=4,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        time_horizon=0.25,
        resolved_diagnostics=False,
    )

    np.testing.assert_allclose(np.asarray(t), [0.1, 0.2, 0.25, 0.25])
    np.testing.assert_allclose(np.asarray(diag.dt_t), [0.1, 0.1, 0.05, 0.0])


def test_prepared_nonlinear_diagnostics_preserves_adaptive_default_path():
    """Default adaptive runs keep static CFL setup outside traced overrides."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    state = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    prepared = prepare_nonlinear_explicit_diagnostics(
        state,
        grid,
        geom,
        params,
        dt=0.01,
        steps=2,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        fixed_dt=False,
        resolved_diagnostics=False,
    )

    final_state, diagnostics, _dt, _fields = prepared.run()
    assert bool(jnp.all(jnp.isfinite(final_state)))
    assert bool(jnp.all(jnp.isfinite(diagnostics.dt_t)))
    with pytest.raises(ValueError, match="require fixed_dt=True"):
        prepared.run_arrays(cache=prepared.cache, params=params)


def test_prepared_nonlinear_arrays_support_reverse_mode_state_gradients():
    """The raw prepared scan should remain differentiable through time stepping."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    direction = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-7
    prepared = prepare_nonlinear_explicit_diagnostics(
        direction,
        grid,
        geom,
        params,
        dt=0.02,
        steps=2,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )

    def final_energy(scale: jnp.ndarray) -> jnp.ndarray:
        final_state, _diagnostics, _fields = prepared.run_arrays(scale * direction)
        return jnp.real(jnp.vdot(final_state, final_state))

    value, gradient = jax.value_and_grad(final_energy)(jnp.asarray(1.0))
    eps = jnp.asarray(1.0e-2)
    centered_fd = (final_energy(1.0 + eps) - final_energy(1.0 - eps)) / (2.0 * eps)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(value) > 0.0
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-16
    )


def test_prepared_nonlinear_arrays_accept_matched_dynamic_cache_and_params():
    """A prepared scan should differentiate through a rebuilt parameter cache."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    base_params = LinearParams()
    state = jnp.ones((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)) * 1.0e-7
    prepared = prepare_nonlinear_explicit_diagnostics(
        state,
        grid,
        geom,
        base_params,
        dt=0.01,
        steps=2,
        method="rk2",
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )

    def final_energy(rlt: jnp.ndarray) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
        final_state, _diagnostics, _fields = prepared.run_arrays(
            cache=cache, params=params
        )
        return jnp.real(jnp.vdot(final_state, final_state))

    value, gradient = jax.value_and_grad(final_energy)(jnp.asarray(6.9))
    step = jnp.asarray(1.0e-2)
    centered_fd = (final_energy(6.9 + step) - final_energy(6.9 - step)) / (2 * step)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=5.0e-2, atol=1.0e-16
    )
    with pytest.raises(ValueError, match="supplied together"):
        prepared.run_arrays(params=base_params)


def test_block_checkpointed_nonlinear_heat_flux_gradient_matches_finite_difference():
    """The bounded-memory adjoint must differentiate the physical heat flux."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base_params = LinearParams()
    state = jnp.zeros(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    )
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 0, :].set(0.25j * profile)
    state = state.at[0, 0, 1, 1, :].set((0.2 - 0.1j) * profile)
    terms = TermConfig(nonlinear=1.0)

    def mean_heat_flux(rlt: jnp.ndarray, checkpoint: bool = True) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        return nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            params,
            dt=0.01,
            steps=11,
            method="rk2",
            tail_steps=7,
            terms=terms,
            checkpoint=checkpoint,
            compressed_real_fft=False,
        )

    point = jnp.asarray(6.9)
    value, gradient = jax.value_and_grad(mean_heat_flux)(point)
    plain = jax.value_and_grad(lambda value: mean_heat_flux(value, False))(point)
    step = jnp.asarray(1.0e-2)
    centered_fd = (mean_heat_flux(point + step) - mean_heat_flux(point - step)) / (
        2 * step
    )

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray((value, gradient)), np.asarray(plain), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=5.0e-2, atol=1.0e-16
    )


@pytest.mark.parametrize("checkpoint", [False, True])
def test_prepared_nonlinear_arrays_differentiate_dynamic_geometry(
    checkpoint: bool,
) -> None:
    """Curvature-profile derivatives should cross the prepared scan boundary."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    analytic_geometry = SAlphaGeometry.from_config(cfg.geometry)
    geometry = ensure_flux_tube_geometry_data(analytic_geometry, grid.z)
    params = LinearParams()
    state = jnp.zeros(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
    )
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 0, :].set(0.25j * profile)
    prepared = prepare_nonlinear_explicit_diagnostics(
        state,
        grid,
        geometry,
        params,
        dt=0.02,
        steps=3,
        method="rk2",
        checkpoint=checkpoint,
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )

    def final_mode_projection(curvature_scale: jnp.ndarray) -> jnp.ndarray:
        dynamic_geometry = replace(
            geometry,
            cv_profile=curvature_scale * geometry.cv_profile,
            gb_profile=curvature_scale * geometry.gb_profile,
            cv0_profile=curvature_scale * geometry.cv0_profile,
            gb0_profile=curvature_scale * geometry.gb0_profile,
        )
        cache = build_linear_cache(grid, dynamic_geometry, params, Nl=2, Nm=2)
        final_state, _diagnostics, _fields = prepared.run_arrays(
            geometry=dynamic_geometry,
            cache=cache,
            params=params,
        )
        return jnp.real(final_state[0, 0, 1, 0, 1]) + 0.37 * jnp.imag(
            final_state[0, 0, 1, 0, 2]
        )

    value, gradient = jax.value_and_grad(final_mode_projection)(jnp.asarray(1.0))
    step = jnp.asarray(1.0e-2)
    centered_fd = (
        final_mode_projection(1.0 + step) - final_mode_projection(1.0 - step)
    ) / (2 * step)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=5.0e-2, atol=1.0e-16
    )
    with pytest.raises(ValueError, match="dynamic geometry requires"):
        prepared.run_arrays(geometry=geometry)


def test_integrate_nonlinear_imex_diagnostics_shapes():
    """IMEX nonlinear diagnostics should return time-series arrays."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)
    t, diag = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=2,
        method="imex",
        terms=terms,
    )
    assert t.shape[0] == 2
    assert diag.energy_t.shape[0] == 2
    assert diag.heat_flux_species_t is not None
    assert diag.particle_flux_species_t is not None
    assert np.asarray(diag.heat_flux_species_t).shape == (2, 1)
    assert np.asarray(diag.particle_flux_species_t).shape == (2, 1)


def test_integrate_nonlinear_collision_split_sts():
    """Collision split with STS scheme should run and remain finite."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0, collisions=1.0)
    _t, diag = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="rk2",
        terms=terms,
        collision_split=True,
        collision_scheme="sts",
    )
    assert np.isfinite(np.asarray(diag.Wg_t)).all()


def test_nonlinear_split_keeps_conserving_collisions_in_explicit_rhs():
    """A diagonal split must not discard collision field-particle corrections."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(nu=0.2)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    G = jnp.asarray(
        np.linspace(
            1.0, 1.0 + 2 * 2 * 2 * 2 * 4 - 1, 2 * 2 * 2 * 2 * 4, dtype=np.float32
        ).reshape(2, 2, 2, 2, 4),
        dtype=jnp.complex64,
    )
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        nonlinear=0.0,
        apar=0.0,
        bpar=0.0,
    )

    _t, _diag, G_final, _fields = integrate_nonlinear_explicit_diagnostics_state(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="rk3",
        cache=cache,
        terms=terms,
        collision_split=True,
        collision_scheme="exp",
    )

    _t, _diag, expected, _fields = integrate_nonlinear_explicit_diagnostics_state(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=1,
        method="rk3",
        cache=cache,
        terms=terms,
        collision_split=False,
    )

    np.testing.assert_allclose(
        np.asarray(G_final), np.asarray(expected), rtol=1.0e-6, atol=1.0e-6
    )


def test_nonlinear_adaptive_default_dt_max_matches_requested_dt():
    """Adaptive nonlinear runtime diagnostics should clamp dt to dt when dt_max is unset."""

    grid_cfg = GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    G = jnp.zeros((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)
    _t, diag = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.05,
        steps=3,
        method="rk3",
        terms=terms,
        fixed_dt=False,
        dt_max=None,
        cfl=10.0,
    )
    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert np.nanmax(dt_t) <= 0.05 + 1.0e-6


def test_nonlinear_adaptive_dt_includes_linear_frequency_cap():
    """Adaptive nonlinear dt should honor the linear CFL estimate even with zero nonlinear drive."""

    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=16, Lx=20.0, Ly=20.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    geom_eff = ensure_flux_tube_geometry_data(geom, grid.z)
    params = LinearParams(tprim=3.0, fprim=1.0)
    G = jnp.zeros((2, 4, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz))
    terms = TermConfig(nonlinear=0.0)

    cfl = 0.5
    cfl_fac = 1.73
    dt0 = 0.1
    cache = build_linear_cache(grid, geom_eff, params, Nl=2, Nm=4)
    linear_omega = _linear_frequency_bound(
        grid,
        geom_eff,
        params,
        nl=int(cache.l.shape[0]),
        nm=int(cache.m.shape[1]),
        include_diamagnetic_drive=False,
    )
    expected_dt = cfl_fac * cfl / float(np.sum(linear_omega))

    _t, diag = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=dt0,
        steps=2,
        method="rk3",
        terms=terms,
        fixed_dt=False,
        dt_max=dt0,
        cfl=cfl,
        cfl_fac=cfl_fac,
    )

    dt_t = np.asarray(diag.dt_t, dtype=float)
    assert dt_t.size > 0
    assert dt_t[0] == pytest.approx(expected_dt, rel=1.0e-5, abs=1.0e-8)
    assert dt_t[0] < dt0


@pytest.mark.parametrize("method", ["rk3", "imex"])
def test_nonlinear_gamma_omega_use_previous_step_not_previous_diagnostic(method: str):
    """Nonlinear gamma/omega should be invariant to diagnostics_stride."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    terms = TermConfig(nonlinear=0.0)

    t_dense, diag_dense = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=4,
        method=method,
        terms=terms,
        sample_stride=1,
        diagnostics_stride=1,
    )
    t_sparse, diag_sparse = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=4,
        method=method,
        terms=terms,
        sample_stride=1,
        diagnostics_stride=2,
    )

    t_dense_arr = np.asarray(t_dense)
    t_sparse_arr = np.asarray(t_sparse)
    gamma_dense = np.asarray(diag_dense.gamma_t)
    omega_dense = np.asarray(diag_dense.omega_t)
    gamma_sparse = np.asarray(diag_sparse.gamma_t)
    omega_sparse = np.asarray(diag_sparse.omega_t)

    stride_indices = list(range(0, len(t_dense_arr), 2))
    forced_final = bool(
        t_sparse_arr[-1] == pytest.approx(t_dense_arr[-1])
        and stride_indices[-1] != len(t_dense_arr) - 1
    )
    compared_sparse = slice(None, -1 if forced_final else None)
    compared_indices = stride_indices[: len(t_sparse_arr[compared_sparse])]

    assert np.allclose(t_dense_arr[compared_indices], t_sparse_arr[compared_sparse])
    assert np.allclose(gamma_dense[compared_indices], gamma_sparse[compared_sparse])
    assert np.allclose(omega_dense[compared_indices], omega_sparse[compared_sparse])
    if forced_final:
        assert gamma_sparse[-1] == pytest.approx(gamma_sparse[-2])
        assert omega_sparse[-1] == pytest.approx(omega_sparse[-2])


def test_nonlinear_imex_diagnostics_match_operator_dtype_under_x64():
    """IMEX runtime diagnostics should keep the scan state dtype aligned with the implicit operator."""

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)

    old_x64 = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        grid = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = LinearParams()
        shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
        base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)

        _t, diag = integrate_nonlinear_explicit_diagnostics(
            G,
            grid,
            geom,
            params,
            dt=0.02,
            steps=2,
            method="imex",
            terms=TermConfig(nonlinear=0.0),
            sample_stride=1,
            diagnostics_stride=1,
        )
    finally:
        jax.config.update("jax_enable_x64", old_x64)

    assert np.isfinite(np.asarray(diag.gamma_t)).all()
    assert np.isfinite(np.asarray(diag.omega_t)).all()


@pytest.mark.parametrize("method", ["rk3", "sspx3"])
def test_nonlinear_state_diagnostics_can_freeze_one_mode(method: str):
    """Fixed-mode projection should preserve a selected Fourier mode exactly."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)

    _t, _diag, G_final, _fields = integrate_nonlinear_explicit_diagnostics_state(
        G,
        grid,
        geom,
        params,
        dt=0.02,
        steps=3,
        method=method,
        terms=TermConfig(nonlinear=1.0, collisions=0.0, hypercollisions=0.0),
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    assert np.allclose(np.asarray(G_final)[..., 1, 0, :], np.asarray(G)[..., 1, 0, :])


def test_linked_boundary_heat_flux_window_gradient_matches_finite_difference() -> None:
    """A twist-shift nonlinear window differentiates through its own cache.

    The window is the shape a turbulent design objective has -- several
    integration chunks, the heat flux read at each chunk end, averaged. Under
    ``boundary="linked"`` the cache is rebuilt inside the trace on every call,
    which used to fail closed. Dissipation is on because the toy grid blows up
    without it, not because the boundary needs it.
    """

    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=8, Lx=6.0, Ly=6.0, boundary="linked")
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base_params = LinearParams(nu=0.05, nu_hyper=0.5)
    terms = TermConfig(nonlinear=1.0)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geom, grid)
    Nl, Nm = 2, 2
    dt, chunk, chunks = 0.02, 8, 2

    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    initial_state = 1.0e-3 * (
        jax.random.normal(jax.random.PRNGKey(0), shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(1), shape)
    )

    def window_heat_flux(rlt: jnp.ndarray) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        state = initial_state
        total = jnp.zeros((), dtype=jnp.result_type(float))
        for _ in range(chunks):
            state, fields = integrate_nonlinear_cached(
                state,
                cache,
                params,
                dt,
                chunk,
                terms=terms,
                compressed_real_fft=False,
            )
            total = total + heat_flux_total(
                state,
                fields.phi[-1],
                fields.apar[-1],
                fields.bpar[-1],
                cache,
                grid,
                params,
                flux_factor,
            )
        return total / float(chunks)

    rlt = jnp.asarray(6.9)
    forward = jax.jit(window_heat_flux)
    value = forward(rlt)
    gradient = jax.jit(jax.grad(window_heat_flux))(rlt)
    step = jnp.asarray(0.05)
    centered_fd = (forward(rlt + step) - forward(rlt - step)) / (2.0 * step)

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-10
    )


@pytest.mark.parametrize("boundary", ["periodic", "linked"])
def test_compressed_real_fft_heat_flux_window_gradient_matches_finite_difference(
    boundary: str,
) -> None:
    """The default production nonlinear path differentiates, not only the full one.

    ``compressed_real_fft=True`` is what the example TOMLs run, and it used to
    refuse ``jit`` outright: the Hermitian projector read the sign pattern off
    ``cache.ky``, which is a traced array whenever the cache is built inside
    the trace. Every gradient test therefore ran the full-complex bracket
    instead. The layout the projector actually needs is grid topology, so both
    boundaries now differentiate on the compressed path as well.
    """

    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=8, Lx=6.0, Ly=6.0, boundary=boundary)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    base_params = LinearParams(nu=0.05, nu_hyper=0.5)
    terms = TermConfig(nonlinear=1.0)
    _volume_factor, flux_factor = fieldline_quadrature_weights(geom, grid)
    Nl, Nm = 2, 2
    dt, chunk, chunks = 0.02, 8, 2

    # The compressed path is only worth differentiating where it does something:
    # this grid's projector really does rebuild the negative-ky half.
    projector = _make_compressed_real_fft_projector(
        ny_full=int(grid.ky.size), nx=int(grid.kx.size)
    )
    probe_shape = (1, 1, grid.ky.size, grid.kx.size, 1)
    probe = (1.0 + 1.0j) * jnp.arange(
        int(np.prod(probe_shape)), dtype=jnp.complex64
    ).reshape(probe_shape)
    assert not np.allclose(np.asarray(projector(probe)), np.asarray(probe))

    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    initial_state = 1.0e-3 * (
        jax.random.normal(jax.random.PRNGKey(0), shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(1), shape)
    )

    def window_heat_flux(rlt: jnp.ndarray) -> jnp.ndarray:
        params = replace(base_params, tprim=rlt)
        cache = build_linear_cache(grid, geom, params, Nl, Nm)
        state = initial_state
        total = jnp.zeros((), dtype=jnp.result_type(float))
        for _ in range(chunks):
            state, fields = integrate_nonlinear_cached(
                state,
                cache,
                params,
                dt,
                chunk,
                terms=terms,
                compressed_real_fft=True,
            )
            total = total + heat_flux_total(
                state,
                fields.phi[-1],
                fields.apar[-1],
                fields.bpar[-1],
                cache,
                grid,
                params,
                flux_factor,
            )
        return total / float(chunks)

    rlt = jnp.asarray(6.9)
    forward = jax.jit(window_heat_flux)
    value = forward(rlt)
    gradient = jax.jit(jax.grad(window_heat_flux))(rlt)
    step = jnp.asarray(0.05)
    centered_fd = (forward(rlt + step) - forward(rlt - step)) / (2.0 * step)

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered_fd), rtol=2.0e-2, atol=1.0e-10
    )


def test_adaptive_time_step_run_compiles_and_matches_the_eager_trajectory() -> None:
    """``fixed_dt = false`` is what the Cyclone inputs run, and it now jits.

    The CFL bound that sets ``dt_max`` lifted the parallel extent out of
    ``grid.z`` with ``jnp`` arithmetic before reading it back on the host.
    Inside a trace that stages out, so the adaptive route refused under ``jit``
    with a bare conversion error -- for a grid, a geometry and a parameter set
    that were all concrete host data, and for a number (how many 2*pi periods
    the tube spans) that has no derivative in the first place.
    """

    from gkx.solvers_nonlinear_state_integration import (
        integrate_nonlinear_sheared_transport,
    )

    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    params = LinearParams(nu=0.05, nu_hyper=0.5)
    terms = TermConfig(nonlinear=1.0)
    Nl, Nm = 2, 2
    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    state = 1.0e-3 * (
        jax.random.normal(jax.random.PRNGKey(0), shape)
        + 1.0j * jax.random.normal(jax.random.PRNGKey(1), shape)
    )

    def run(initial):
        trace = integrate_nonlinear_sheared_transport(
            initial,
            grid,
            geom,
            params,
            0.02,
            3,
            shear_rate=0.0,
            terms=terms,
            fixed_dt=False,
        )
        return jnp.sum(jnp.abs(trace.final_state) ** 2)

    eager = run(state)
    compiled = jax.jit(run)(state)
    assert bool(jnp.isfinite(eager)) and float(eager) > 0.0
    # The adaptive step is chosen from the same host bound either way. The
    # final float32 reduction may reassociate under JIT, so require one relative
    # rounding unit rather than an unattainable float64-scale tolerance.
    tolerance = max(1.0e-12, float(np.finfo(np.asarray(eager).dtype).eps))
    np.testing.assert_allclose(np.asarray(compiled), np.asarray(eager), rtol=tolerance)


# ---- from test_nonlinear_rhs.py ----


def _minimal_cache() -> SimpleNamespace:
    return SimpleNamespace(
        Jl=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        JlB=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32),
        kx_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        ky_grid=jnp.zeros((1, 1), dtype=jnp.float32),
        dealias_mask=jnp.ones((1, 1), dtype=bool),
        kxfac=1.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        laguerre_j0=None,
        laguerre_j1_over_alpha=None,
        b=None,
    )


def test_linear_rhs_jit_for_terms_impl_selects_narrowest_route() -> None:
    electrostatic = object()
    full = object()

    assert (
        linear_rhs_jit_for_terms_impl(
            TermConfig(apar=0.0, bpar=0.0),
            electrostatic_rhs_fn=electrostatic,  # type: ignore[arg-type]
            full_rhs_fn=full,  # type: ignore[arg-type]
            is_static_zero_fn=lambda value: float(value) == 0.0,
        )
        is electrostatic
    )
    assert (
        linear_rhs_jit_for_terms_impl(
            TermConfig(apar=1.0, bpar=0.0),
            electrostatic_rhs_fn=electrostatic,  # type: ignore[arg-type]
            full_rhs_fn=full,  # type: ignore[arg-type]
            is_static_zero_fn=lambda value: float(value) == 0.0,
        )
        is full
    )


def test_nonlinear_rhs_cached_impl_skips_bracket_when_disabled() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64))
    calls = {"linear": 0, "nonlinear": 0}

    def linear_rhs(G_in, *_args, **_kwargs):
        calls["linear"] += 1
        return 2.0 * G_in, fields

    def nonlinear_contribution(*_args, **_kwargs):
        calls["nonlinear"] += 1
        raise AssertionError("nonlinear contribution should not run")

    rhs, rhs_fields = nonlinear_rhs_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.0, apar=0.0, bpar=0.0),
        electrostatic_rhs_fn=linear_rhs,
        full_rhs_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    assert calls == {"linear": 1, "nonlinear": 0}
    np.testing.assert_allclose(np.asarray(rhs), 2.0)
    assert rhs_fields is fields


def test_nonlinear_rhs_cached_impl_accepts_custom_collision_operator() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=2.0 * jnp.ones((1, 1, 2), dtype=jnp.complex64))
    seen: dict[str, object] = {}

    def linear_rhs(G_in, _cache, _params, terms, **_kwargs):
        seen["linear_collision_weight"] = terms.collisions
        seen["hypercollision_weight"] = terms.hypercollisions
        return 2.0 * G_in, fields

    class DragCollision:
        def apply(self, context):
            seen["cache"] = context.cache
            seen["parameters"] = context.parameters
            seen["fields"] = context.fields
            seen["hamiltonian"] = context.hamiltonian
            return -3.0 * context.distribution

    cache = _minimal_cache()
    parameters = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))
    rhs, _ = nonlinear_rhs_cached_impl(
        G,
        cache,
        parameters,
        TermConfig(collisions=0.25, hypercollisions=0.75, nonlinear=0.0),
        collision_operator=DragCollision(),
        electrostatic_rhs_fn=linear_rhs,
        full_rhs_fn=linear_rhs,
    )

    assert seen["linear_collision_weight"] == 0.0
    assert seen["hypercollision_weight"] == 0.75
    assert seen["cache"] is cache
    assert seen["parameters"] is parameters
    assert seen["fields"] is fields
    np.testing.assert_allclose(np.asarray(seen["hamiltonian"]), 3.0)
    np.testing.assert_allclose(np.asarray(rhs), 1.25)


def test_nonlinear_rhs_cached_impl_rejects_invalid_collision_shape() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=jnp.zeros((1, 1, 2), dtype=jnp.complex64))

    class InvalidCollision:
        def apply(self, context):
            return context.distribution[..., 0]

    with pytest.raises(ValueError, match="same state shape"):
        nonlinear_rhs_cached_impl(
            G,
            _minimal_cache(),
            SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
            TermConfig(collisions=1.0, nonlinear=0.0),
            collision_operator=InvalidCollision(),
            electrostatic_rhs_fn=lambda state, *_args, **_kwargs: (
                jnp.zeros_like(state),
                fields,
            ),
            full_rhs_fn=lambda state, *_args, **_kwargs: (
                jnp.zeros_like(state),
                fields,
            ),
        )


def test_nonlinear_rhs_cached_impl_forwards_physical_bracket_payload() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    seen: dict[str, object] = {}

    def linear_rhs(G_in, *_args, **_kwargs):
        return jnp.zeros_like(G_in), fields

    def nonlinear_contribution(G_in, **kwargs):
        seen["shape"] = tuple(G_in.shape)
        seen["apar_is_none"] = kwargs["apar"] is None
        seen["bpar_is_none"] = kwargs["bpar"] is None
        seen["weight"] = float(kwargs["weight"])
        seen["apar_weight"] = kwargs["apar_weight"]
        seen["bpar_weight"] = kwargs["bpar_weight"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return 4.0 * jnp.ones_like(G_in)

    rhs, _rhs_fields = nonlinear_rhs_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.25, apar=1.0, bpar=0.0),
        compressed_real_fft=False,
        laguerre_mode="spectral",
        electrostatic_rhs_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError()
        ),
        full_rhs_fn=linear_rhs,
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    np.testing.assert_allclose(np.asarray(rhs), 4.0)
    assert seen == {
        "shape": tuple(G.shape),
        "apar_is_none": False,
        "bpar_is_none": True,
        "weight": 0.25,
        "apar_weight": 1.0,
        "bpar_weight": 0.0,
        "compressed_real_fft": False,
        "laguerre_mode": "spectral",
    }


def test_nonlinear_em_term_cached_impl_reuses_imex_bracket_payload() -> None:
    G = jnp.ones((1, 1, 1, 1, 1, 2), dtype=jnp.complex64)
    phi = jnp.ones((1, 1, 2), dtype=jnp.complex64)
    fields = FieldState(phi=phi, apar=2.0 * phi, bpar=3.0 * phi)
    seen: dict[str, object] = {"fields": 0, "nonlinear": 0}

    def fields_fn(G_in, *_args, **kwargs):
        seen["fields"] = int(seen["fields"]) + 1
        seen["external_phi"] = kwargs["external_phi"]
        np.testing.assert_allclose(np.asarray(G_in), np.asarray(G))
        return fields

    def nonlinear_contribution(G_in, **kwargs):
        seen["nonlinear"] = int(seen["nonlinear"]) + 1
        seen["shape"] = tuple(G_in.shape)
        seen["apar_is_none"] = kwargs["apar"] is None
        seen["bpar_is_none"] = kwargs["bpar"] is None
        seen["weight"] = float(kwargs["weight"])
        seen["apar_weight"] = kwargs["apar_weight"]
        seen["bpar_weight"] = kwargs["bpar_weight"]
        seen["compressed_real_fft"] = kwargs["compressed_real_fft"]
        seen["laguerre_mode"] = kwargs["laguerre_mode"]
        return 5.0 * jnp.ones_like(G_in)

    zero = nonlinear_em_term_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.0),
        fields_fn=fields_fn,
        nonlinear_contribution_fn=nonlinear_contribution,
    )
    np.testing.assert_allclose(np.asarray(zero), 0.0)
    assert seen == {"fields": 0, "nonlinear": 0}

    out = nonlinear_em_term_cached_impl(
        G,
        _minimal_cache(),
        SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0])),
        TermConfig(nonlinear=0.5, apar=1.0, bpar=1.0),
        external_phi=3.0,
        compressed_real_fft=False,
        laguerre_mode="spectral",
        fields_fn=fields_fn,
        nonlinear_contribution_fn=nonlinear_contribution,
    )

    np.testing.assert_allclose(np.asarray(out), 5.0)
    assert seen == {
        "fields": 1,
        "nonlinear": 1,
        "external_phi": 3.0,
        "shape": tuple(G.shape),
        "apar_is_none": False,
        "bpar_is_none": False,
        "weight": 0.5,
        "apar_weight": 1.0,
        "bpar_weight": 1.0,
        "compressed_real_fft": False,
        "laguerre_mode": "spectral",
    }


# ---- from test_nonlinear_diagnostic_state.py ----


def _array(value: float) -> jnp.ndarray:
    return jnp.asarray([value], dtype=jnp.float32)


def _unused(*_args, **_kwargs):
    raise AssertionError("kernel should not be used in this branch")


def _growth(*_args, **_kwargs):
    return (
        jnp.asarray([[2.0]], dtype=jnp.float32),
        jnp.asarray([[-3.0]], dtype=jnp.float32),
    )


def _minimal_inputs():
    phi = jnp.ones((1, 1, 1), dtype=jnp.complex64)
    return {
        "G_state": jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        "fields_state": FieldState(phi=phi),
        "G_prev_step": jnp.zeros((1, 1, 1, 1, 1, 1), dtype=jnp.complex64),
        "fields_prev_step": FieldState(phi=phi),
        "dt_step": jnp.asarray(0.1, dtype=jnp.float32),
        "grid": SimpleNamespace(),
        "cache": SimpleNamespace(),
        "params": SimpleNamespace(),
        "vol_fac": jnp.ones((1,), dtype=jnp.float32),
        "flux_fac": jnp.asarray(1.0, dtype=jnp.float32),
        "mask": jnp.asarray([[True]]),
        "z_idx": 0,
        "use_dealias": False,
        "real_dtype": jnp.float32,
        "omega_ky_index": None,
        "omega_kx_index": None,
        "flux_scale": 1.0,
        "wphi_scale": 1.0,
    }


def test_compute_nonlinear_diagnostic_tuple_unresolved_uses_scalar_kernels() -> None:
    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=_unused,
        zonal_phi_mode_kxt=_unused,
        zonal_phi_line_kxt=_unused,
        distribution_free_energy=lambda *_args, **_kwargs: _array(10),
        distribution_free_energy_resolved=_unused,
        electrostatic_field_energy=lambda *_args, **_kwargs: _array(20),
        electrostatic_field_energy_resolved=_unused,
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _array(30),
        magnetic_vector_potential_energy_resolved=_unused,
        heat_flux_species=lambda *_args, **_kwargs: jnp.asarray([4.0, 5.0]),
        heat_flux_channel_resolved_species=_unused,
        particle_flux_species=lambda *_args, **_kwargs: jnp.asarray([6.0]),
        particle_flux_channel_resolved_species=_unused,
        turbulent_heating_species=lambda *_args, **_kwargs: jnp.asarray([7.0]),
        turbulent_heating_resolved_species=_unused,
    )

    out = compute_nonlinear_diagnostic_tuple(
        **_minimal_inputs(),
        resolved_diagnostics=False,
        kernels=kernels,
    )

    assert len(out) == 13
    np.testing.assert_allclose(np.asarray(out[0]), 2.0)
    np.testing.assert_allclose(np.asarray(out[1]), -3.0)
    np.testing.assert_allclose(np.asarray(out[2]), [10.0])
    np.testing.assert_allclose(np.asarray(out[5]), 9.0)
    np.testing.assert_allclose(np.asarray(out[6]), 6.0)
    np.testing.assert_allclose(np.asarray(out[7]), 7.0)
    assert out[-1] == ()


def test_make_nonlinear_diagnostic_tuple_fn_preserves_scalar_contract() -> None:
    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=_unused,
        zonal_phi_mode_kxt=_unused,
        zonal_phi_line_kxt=_unused,
        distribution_free_energy=lambda *_args, **_kwargs: _array(10),
        distribution_free_energy_resolved=_unused,
        electrostatic_field_energy=lambda *_args, **_kwargs: _array(20),
        electrostatic_field_energy_resolved=_unused,
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _array(30),
        magnetic_vector_potential_energy_resolved=_unused,
        heat_flux_species=lambda *_args, **_kwargs: jnp.asarray([4.0, 5.0]),
        heat_flux_channel_resolved_species=_unused,
        particle_flux_species=lambda *_args, **_kwargs: jnp.asarray([6.0]),
        particle_flux_channel_resolved_species=_unused,
        turbulent_heating_species=lambda *_args, **_kwargs: jnp.asarray([7.0]),
        turbulent_heating_resolved_species=_unused,
    )
    inputs = _minimal_inputs()
    compute_diag = make_nonlinear_diagnostic_tuple_fn(
        grid=inputs["grid"],
        cache=inputs["cache"],
        params=inputs["params"],
        vol_fac=inputs["vol_fac"],
        flux_fac=inputs["flux_fac"],
        mask=inputs["mask"],
        z_idx=inputs["z_idx"],
        use_dealias=inputs["use_dealias"],
        real_dtype=inputs["real_dtype"],
        omega_ky_index=inputs["omega_ky_index"],
        omega_kx_index=inputs["omega_kx_index"],
        flux_scale=inputs["flux_scale"],
        wphi_scale=inputs["wphi_scale"],
        resolved_diagnostics=False,
        kernels=kernels,
    )

    out = compute_diag(
        inputs["G_state"],
        inputs["fields_state"],
        inputs["G_prev_step"],
        inputs["fields_prev_step"],
        inputs["dt_step"],
    )

    assert len(out) == 13
    np.testing.assert_allclose(np.asarray(out[0]), 2.0)
    np.testing.assert_allclose(np.asarray(out[5]), 9.0)
    assert out[-1] == ()


def test_compute_nonlinear_diagnostic_tuple_resolved_packs_marker_order() -> None:
    calls = {"heat": 0, "particle": 0}

    def _resolved_tuple(start: int, count: int):
        return tuple(_array(start + idx) for idx in range(count))

    def _channel_tuple(name: str, start: int):
        calls[name] += 1
        return (
            _resolved_tuple(start, 5),
            _resolved_tuple(start + 5, 5),
            _resolved_tuple(start + 10, 5),
        )

    kernels = NonlinearDiagnosticKernels(
        instantaneous_growth_rate_step=_growth,
        phi2_resolved=lambda *_args, **_kwargs: _resolved_tuple(100, 8),
        zonal_phi_mode_kxt=lambda *_args, **_kwargs: _array(108),
        zonal_phi_line_kxt=lambda *_args, **_kwargs: _array(109),
        distribution_free_energy=lambda *_args, **_kwargs: _unused(),
        distribution_free_energy_resolved=lambda *_args, **_kwargs: _resolved_tuple(
            110, 6
        ),
        electrostatic_field_energy=lambda *_args, **_kwargs: _unused(),
        electrostatic_field_energy_resolved=lambda *_args, **_kwargs: _resolved_tuple(
            116, 5
        ),
        magnetic_vector_potential_energy=lambda *_args, **_kwargs: _unused(),
        magnetic_vector_potential_energy_resolved=lambda *_args, **_kwargs: (
            _resolved_tuple(121, 5)
        ),
        heat_flux_species=lambda *_args, **_kwargs: _unused(),
        heat_flux_channel_resolved_species=lambda *_args, **_kwargs: _channel_tuple(
            "heat", 131
        ),
        particle_flux_species=lambda *_args, **_kwargs: _unused(),
        particle_flux_channel_resolved_species=lambda *_args, **_kwargs: _channel_tuple(
            "particle", 151
        ),
        turbulent_heating_species=lambda *_args, **_kwargs: _unused(),
        turbulent_heating_resolved_species=lambda *_args, **_kwargs: _resolved_tuple(
            166, 5
        ),
    )

    out = compute_nonlinear_diagnostic_tuple(
        **_minimal_inputs(),
        resolved_diagnostics=True,
        kernels=kernels,
    )

    resolved = out[-1]
    assert len(resolved) == 58
    assert calls == {"heat": 1, "particle": 1}
    np.testing.assert_allclose(np.asarray(out[2]), [110.0])
    np.testing.assert_allclose(np.asarray(out[5]), [408.0])
    np.testing.assert_allclose(np.asarray(resolved[0]), [101.0])
    np.testing.assert_allclose(np.asarray(resolved[7]), [108.0])
    np.testing.assert_allclose(np.asarray(resolved[26]), [132.0])
    np.testing.assert_allclose(np.asarray(resolved[-1]), [170.0])
