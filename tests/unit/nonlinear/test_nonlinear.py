"""Nonlinear integrator tests."""

from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gkx.config import CycloneBaseCase, GridConfig
from gkx.diagnostics.transport import heat_flux_total
from gkx.diagnostics.moments import fieldline_quadrature_weights
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.solvers.time.explicit import _linear_frequency_bound
from gkx.core.grid import build_spectral_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.operators.nonlinear.policies import build_nonlinear_imex_operator
from gkx.solvers.nonlinear.diagnostic_integration import integrate_nonlinear_explicit_diagnostics, integrate_nonlinear_explicit_diagnostics_state, prepare_nonlinear_explicit_diagnostics
from gkx.solvers.nonlinear.state_integration import integrate_nonlinear, integrate_nonlinear_imex_cached
from gkx.terms.config import TermConfig

pytestmark = pytest.mark.integration


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

    first = prepared.run()
    cache_size = prepared._run_raw._cache_size()
    second = prepared.run()
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

    assert cache_size == 1
    assert prepared._run_raw._cache_size() == cache_size
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
