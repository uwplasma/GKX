"""Core nonlinear solver, RHS, and diagnostic-state tests."""

from __future__ import annotations

from dataclasses import replace
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.operators.fluxes import heat_flux_total
from gkx.operators.moments import fieldline_quadrature_weights
from gkx.geometry import SAlphaGeometry, ensure_flux_tube_geometry_data
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.linked import (
    linked_cover_mask_from_cache,
    mask_supplied_state,
)
from gkx.operators.linear.params import LinearParams
from gkx.operators.nonlinear.diagnostic_state import (
    NonlinearDiagnosticKernels,
    compute_nonlinear_diagnostic_tuple,
    make_nonlinear_diagnostic_tuple_fn,
)
from gkx.operators.nonlinear.policies import build_nonlinear_imex_operator
from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector
from gkx.operators.nonlinear.rhs import (
    nonlinear_rhs_cached_impl,
)
from gkx.solvers.nonlinear.diagnostics import (
    integrate_nonlinear_explicit_diagnostics,
    integrate_nonlinear_explicit_diagnostics_state,
    prepare_nonlinear_explicit_diagnostics,
)
from gkx.solvers.nonlinear.diagnostics import (
    ADJOINT_MEMORY_BUDGET_BYTES,
    integrate_nonlinear,
    integrate_nonlinear_cached,
    integrate_nonlinear_imex_cached,
)
from gkx.solvers.nonlinear.transport_window import nonlinear_heat_flux_window
from gkx.solvers.time_explicit import _linear_frequency_bound
from gkx.terms.config import FieldState
from gkx.terms.config import TermConfig
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

requires_jax_07 = pytest.mark.skipif(
    jax.__version_info__ < (0, 7),
    reason=(
        "jax 0.6 does not prune the unused diagnostics from the scan adjoint, "
        "whose NaN derivatives then reach the state gradient"
    ),
)


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
        method="rk2",
        terms=terms,
        compressed_real_fft=False,
        return_fields=False,
        collision_operator=DragCollision(),
    )
    np.testing.assert_allclose(
        np.asarray(G_final), (1 - 0.075 + 0.075**2 / 2) ** 2, atol=1.0e-6
    )


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

    # The integration runs with ``compressed_real_fft=False``, and the
    # full-complex bracket is defined on the two-sided ky axis only.
    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0, ky_layout="full")
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


def test_adaptive_chunk_compile_cache_matches_fresh_compile_bitwise():
    """Chunked adaptive runs reuse one graph; the horizon is an operand."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=2, Ny=2, Nz=4, Lx=6.0, Ly=6.0))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    state = jnp.full((2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz), 1.0e-3 + 0j)
    kwargs = dict(
        dt=0.1,
        steps=3,
        method="rk3",
        fixed_dt=False,
        dt_max=0.1,
        terms=TermConfig(nonlinear=0.0),
        resolved_diagnostics=False,
    )
    cache: dict = {}
    for horizon in (None, 0.25, 0.15):
        fresh = integrate_nonlinear_explicit_diagnostics_state(
            state, grid, geom, LinearParams(), time_horizon=horizon, **kwargs
        )
        cached = integrate_nonlinear_explicit_diagnostics_state(
            state,
            grid,
            geom,
            LinearParams(),
            time_horizon=horizon,
            compile_cache=cache,
            **kwargs,
        )
        t_a, diag_a, g_a, f_a = fresh
        t_b, diag_b, g_b, f_b = cached
        pairs = [(t_a, t_b), (g_a, g_b), (f_a.phi, f_b.phi)]
        pairs += [
            (getattr(diag_a, k), getattr(diag_b, k))
            for k in ("t", "dt_t", "Wg_t", "heat_flux_t")
        ]
        for a, b in pairs:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # One entry without a horizon, one shared by both horizons.
    assert len(cache) == 2


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

    # The window is integrated with ``compressed_real_fft=False``, and that
    # bracket refuses a grid that stores only the ky >= 0 half.
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, ky_layout="full")
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


@pytest.mark.parametrize("ky_layout", ["full", "half"])
def test_block_only_adjoint_schedule_keeps_the_window_gradient(ky_layout: str):
    """Dropping the inner remat changes cost, not the value or either gradient.

    The block-only schedule (default budget) is compared with the nested one
    (budget ``None``) for the value, ``d/d tprim`` and a geometry direction (the
    drift profiles), and both derivatives are checked against centered finite
    differences, on both ``ky`` layouts with the compressed real-FFT bracket
    and rk3, and vmapped over two tubes. The budget fallback is pinned in
    ``tests/unit/solvers/test_time_integrators.py``.
    """

    from gkx.core_ky_layout import FULL, HALF, to_half

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(
        cfg.grid, ky_layout=FULL if ky_layout == "full" else HALF
    )
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    x64 = bool(jax.config.jax_enable_x64)
    state = jnp.zeros(
        (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz),
        dtype=jnp.complex128 if x64 else jnp.complex64,
    )
    profile = 1.0e-4 * (1.0 + 0.2 * jnp.cos(grid.z))
    state = state.at[0, 0, 1, 0, :].set(profile + 0.3j * profile * jnp.sin(grid.z))
    state = state.at[0, 1, 1, 0, :].set(0.25j * profile)
    state = state.at[0, 0, 1, 1, :].set((0.2 - 0.1j) * profile)
    if ky_layout == "half":
        state = to_half(state, ny_full=cfg.grid.Ny)
    terms = TermConfig(nonlinear=1.0)

    def window(rlt, drift_scale, budget):
        scaled = replace(
            geom,
            gb_profile=geom.gb_profile * drift_scale,
            cv_profile=geom.cv_profile * drift_scale,
        )
        return nonlinear_heat_flux_window(
            state,
            grid,
            scaled,
            replace(LinearParams(), tprim=rlt),
            dt=0.01,
            steps=11,
            method="rk3",
            tail_steps=7,
            terms=terms,
            adjoint_memory_budget_bytes=budget,
        )

    point = (jnp.asarray(6.9), jnp.asarray(1.0))

    def value_and_both(budget):
        value, grads = jax.value_and_grad(
            lambda rlt, scale: window(rlt, scale, budget), argnums=(0, 1)
        )(*point)
        return np.asarray([value, *grads])

    block = value_and_both(ADJOINT_MEMORY_BUDGET_BYTES)
    nested = value_and_both(None)
    assert np.all(np.isfinite(block)) and np.all(block[1:] != 0.0)
    np.testing.assert_allclose(block, nested, rtol=1.0e-12 if x64 else 1.0e-6)

    h = 1.0e-2
    fd_tprim = (
        window(point[0] + h, point[1], None) - window(point[0] - h, point[1], None)
    ) / (2 * h)
    fd_geom = (
        window(point[0], point[1] + h, None) - window(point[0], point[1] - h, None)
    ) / (2 * h)
    np.testing.assert_allclose(block[1], float(fd_tprim), rtol=5.0e-2)
    np.testing.assert_allclose(block[2], float(fd_geom), rtol=5.0e-2)

    # Multi-tube objectives (plan F.5 step 2) are a vmap over tube geometry.
    tubes = jax.vmap(
        jax.value_and_grad(
            lambda rlt, scale: window(rlt, scale, ADJOINT_MEMORY_BUDGET_BYTES), (0, 1)
        ),
        in_axes=(None, 0),
    )(point[0], jnp.asarray([1.0, 1.02]))
    np.testing.assert_allclose(
        [tubes[0][0], tubes[1][0][0], tubes[1][1][0]], block, rtol=1.0e-5
    )


@requires_jax_07
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
        (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz), dtype=jnp.complex64
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
    G = jnp.zeros((2, 4, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz))
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
    """Nonlinear gamma/omega should be invariant to diagnostics_stride.

    The invariance is exact in exact arithmetic and holds to 1.7e-15 in
    float64. In float32 it is roundoff-limited, because gamma and omega are
    per-step log-amplitude and phase ratios: a relative perturbation ``e`` of
    the two amplitudes moves gamma by about ``2 e / dt``, so relative to
    ``gamma`` itself by ``2 e / (dt |gamma|)``. Here ``dt = 0.02`` and
    ``|gamma| ~ 0.099``, which turns a few float32 eps into a few 1e-4. The
    first sample is the one that moves, because it is formed from the
    pre-scan diagnostic, which the two strides compile into differently fused
    graphs. This is the same amplification Q14 recorded for the window
    gradients, and it was already true of ``gkx.prepare`` before queue row Q18
    put both nonlinear routes on one graph.
    """

    eps32 = float(np.finfo(np.float32).eps)
    if bool(jax.config.read("jax_enable_x64")):
        ratio_rtol = 1.0e-12
    else:
        ratio_rtol = 4.0 * 2.0 * eps32 / (0.02 * 0.099)

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz)
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

    # Stride 2 over 4 steps retains steps 0 and 2 plus the off-stride final
    # step 3, which must carry the diagnostics of the final state.
    sample_indices = [0, 2, 3]
    np.testing.assert_allclose(t_sparse_arr, t_dense_arr[sample_indices])
    np.testing.assert_allclose(
        gamma_sparse,
        gamma_dense[sample_indices],
        rtol=ratio_rtol,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        omega_sparse,
        omega_dense[sample_indices],
        rtol=ratio_rtol,
        atol=1.0e-8,
    )


@pytest.mark.parametrize("method", ["rk3", "imex"])
@pytest.mark.parametrize(
    ("sample_stride", "diagnostics_stride"), [(1, 3), (3, 1), (3, 2)]
)
def test_nonlinear_off_stride_final_sample_is_the_final_state(
    method: str, sample_stride: int, diagnostics_stride: int
):
    """Every retained row, including the forced final one, is the state at its time.

    Eight steps at retained stride 3 keep steps 0, 3 and 6 plus the off-stride
    final step 7. That row used to reuse the scan carry from step 6, so the
    last two rows had different times but bit-identical diagnostics. With
    ``sample_stride=3, diagnostics_stride=2`` the row at step 3 also used to
    carry step 2's diagnostics.
    """

    grid_cfg = GridConfig(Nx=2, Ny=4, Nz=4, Lx=6.0, Ly=6.0, ky_layout="full")
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()

    shape = (2, 2, cfg.grid.Ny, cfg.grid.Nx, cfg.grid.Nz)
    base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    G = jnp.asarray(base + 1.0j * (base + 1.0), dtype=jnp.complex64)
    options = {"dt": 0.02, "steps": 8, "method": method, "terms": TermConfig()}

    _t_dense, diag_dense = integrate_nonlinear_explicit_diagnostics(
        G, grid, geom, params, sample_stride=1, diagnostics_stride=1, **options
    )
    t_sparse, diag_sparse = integrate_nonlinear_explicit_diagnostics(
        G,
        grid,
        geom,
        params,
        sample_stride=sample_stride,
        diagnostics_stride=diagnostics_stride,
        **options,
    )

    sample_indices = [0, 3, 6, 7]
    np.testing.assert_allclose(
        np.asarray(t_sparse), np.asarray(diag_dense.t)[sample_indices]
    )
    rtol = 1.0e-10 if bool(jax.config.read("jax_enable_x64")) else 1.0e-3
    for name in ("gamma_t", "omega_t", "Wg_t", "Wphi_t", "heat_flux_t"):
        sparse = np.asarray(getattr(diag_sparse, name))
        dense = np.asarray(getattr(diag_dense, name))
        assert not np.array_equal(sparse[-1], sparse[-2]), name
        np.testing.assert_allclose(
            sparse, dense[sample_indices], rtol=rtol, atol=1.0e-12, err_msg=name
        )


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
        shape = (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz)
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

    shape = (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz)
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

    # The window runs with ``compressed_real_fft=False``, and the full-complex
    # bracket only accepts a two-sided ky axis, so the deck declares one.
    grid_cfg = GridConfig(
        Nx=8, Ny=8, Nz=8, Lx=6.0, Ly=6.0, boundary="linked", ky_layout="full"
    )
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
    """Execute f32 CPU AD in isolation; never skip an untested future backend."""
    if bool(jax.config.read("jax_enable_x64")) or jax.default_backend() != "cpu":
        _compressed_real_fft_gradient_check(boundary)
        return

    import os
    import subprocess
    import sys

    # Rank-7 f32 multiply/reduce can SIGSEGV in XLA:CPU YNN (#196).
    # Run the actual application check, not a smaller proxy for its fusions.
    code = (
        "import runpy,sys\n"
        "if sys.platform != 'win32':\n"
        " import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
        "runpy.run_path(sys.argv[1])['_compressed_real_fft_gradient_check'](sys.argv[2])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, boundary],
        env=dict(os.environ, JAX_ENABLE_X64="false", GKX_X64="0"),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("code", [-11, -9, 1])
def test_compressed_gradient_isolation_does_not_hide_unconfirmed_failures(
    monkeypatch, code
):
    import subprocess

    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=code, stdout="", stderr="sentinel failure"
        ),
    )
    with jax.enable_x64(False), pytest.raises(AssertionError, match="sentinel failure"):
        test_compressed_real_fft_heat_flux_window_gradient_matches_finite_difference(
            "periodic"
        )


def _compressed_real_fft_gradient_check(boundary: str) -> None:
    """Check real-FFT projector tracing and nonlinear AD against physical-flux FD."""

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

    from gkx.solvers.nonlinear.transport_window import (
        integrate_nonlinear_sheared_transport,
    )

    # ``integrate_nonlinear_sheared_transport`` evaluates its bracket with full
    # complex transforms, which need the two-sided ky axis. The subject here is
    # the adaptive step under jit, so the layout is pinned rather than varied.
    grid_cfg = GridConfig(Nx=8, Ny=8, Nz=8, Lx=6.0, Ly=6.0, ky_layout="full")
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


def _route_output_leaves(obj, path="", out=None):
    """Flatten a route result into ``(name, array)`` pairs, dataclasses included.

    ``SimulationDiagnostics`` is a plain dataclass, not a PyTree, so
    ``tree_flatten`` would hand back one opaque leaf and hide every diagnostic
    the comparison is about.
    """

    import dataclasses

    if out is None:
        out = []
    if isinstance(obj, (jax.Array, np.ndarray, np.number, int, float, complex, bool)):
        array = np.asarray(obj)
        if array.dtype.kind in "fciub":
            out.append((path, array))
        return out
    if obj is None or isinstance(obj, str):
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _route_output_leaves(getattr(obj, field.name), f"{path}.{field.name}", out)
        return out
    if isinstance(obj, dict):
        for key in sorted(obj):
            _route_output_leaves(obj[key], f"{path}[{key}]", out)
        return out
    if isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _route_output_leaves(value, f"{path}[{index}]", out)
        return out
    return out


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(
        np.array_equal(
            np.ascontiguousarray(left).view(np.uint8).ravel(),
            np.ascontiguousarray(right).view(np.uint8).ravel(),
        )
    )


def _prepared_and_runtime_nonlinear_route_outputs(*, method: str, fixed_dt: bool):
    """Run the same nonlinear diagnostics case through both shipped routes."""

    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    key = jax.random.PRNGKey(20260918)
    state = 1.0e-3 * jax.random.normal(
        key, (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz)
    )
    options = dict(
        dt=0.01,
        steps=3,
        method=method,
        terms=TermConfig(nonlinear=1.0),
        fixed_dt=fixed_dt,
        resolved_diagnostics=True,
    )
    prepared = prepare_nonlinear_explicit_diagnostics(
        state, grid, geom, params, **options
    )
    reference = prepared.run()
    runtime = integrate_nonlinear_explicit_diagnostics_state(
        state, grid, geom, params, **options
    )
    return reference, runtime


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "x64"])
@pytest.mark.parametrize("method", ["rk3", "rk4"])
@pytest.mark.parametrize("fixed_dt", [True, False], ids=["fixed_dt", "adaptive_dt"])
def test_prepared_and_runtime_nonlinear_routes_are_bitwise_identical(
    x64: bool, method: str, fixed_dt: bool
) -> None:
    """``gkx.prepare`` and the runtime entry point are one reference route.

    Queue row Q18 (plan §5.3). Both compile the same jitted diagnostics graph,
    so every output array must match bit for bit, not to a tolerance. Running
    the runtime entry point outside jit -- what it did before the contract --
    fails this in both precisions: the state moved by up to 2.1e-7 relative in
    float32 and two near-cancelling turbulent-heating diagnostics by O(1).
    """

    with jax.enable_x64(x64):
        reference, runtime = _prepared_and_runtime_nonlinear_route_outputs(
            method=method, fixed_dt=fixed_dt
        )

    left = dict(_route_output_leaves(reference))
    right = dict(_route_output_leaves(runtime))
    assert set(left) == set(right)
    assert len(left) > 40

    differing = sorted(
        name for name in left if not _bitwise_equal(left[name], right[name])
    )
    assert differing == [], differing


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "x64"])
def test_nonlinear_route_reference_holds_for_value_and_gradient(x64: bool) -> None:
    """The reference-route contract covers the reverse-mode pass as well.

    The scale is taken in the real dtype of the state on purpose. A wider scale
    promotes ``G0``, and the two entry points then run at different precisions:
    the prepared object casts to the dtype it was prepared with, the function
    entry point adopts the dtype it is handed.
    """

    with jax.enable_x64(x64):
        grid_cfg = GridConfig(Nx=4, Ny=4, Nz=4, Lx=6.0, Ly=6.0)
        cfg = CycloneBaseCase(grid=grid_cfg)
        grid = build_spectral_grid(cfg.grid)
        geom = SAlphaGeometry.from_config(cfg.geometry)
        params = LinearParams()
        direction = 1.0e-3 * jax.random.normal(
            jax.random.PRNGKey(6), (2, 2, int(grid.ky.size), cfg.grid.Nx, cfg.grid.Nz)
        )
        options = dict(
            dt=0.01,
            steps=3,
            method="rk3",
            terms=TermConfig(nonlinear=1.0),
            fixed_dt=True,
            resolved_diagnostics=False,
        )
        prepared = prepare_nonlinear_explicit_diagnostics(
            direction, grid, geom, params, **options
        )
        real_dtype = jnp.zeros((), dtype=direction.dtype).real.dtype

        def prepared_energy(scale: jnp.ndarray) -> jnp.ndarray:
            final_state, _diagnostics, _fields = prepared.run_arrays(scale * direction)
            return jnp.real(jnp.vdot(final_state, final_state))

        def runtime_energy(scale: jnp.ndarray) -> jnp.ndarray:
            _t, _diag, final_state, _fields = (
                integrate_nonlinear_explicit_diagnostics_state(
                    scale * direction, grid, geom, params, **options
                )
            )
            return jnp.real(jnp.vdot(final_state, final_state))

        one = jnp.asarray(1.0, dtype=real_dtype)
        prepared_value, prepared_grad = jax.value_and_grad(prepared_energy)(one)
        runtime_value, runtime_grad = jax.value_and_grad(runtime_energy)(one)

    assert float(prepared_grad) != 0.0
    assert _bitwise_equal(np.asarray(prepared_value), np.asarray(runtime_value))
    assert _bitwise_equal(np.asarray(prepared_grad), np.asarray(runtime_grad))


# ---- supplied states below the runtime (queue row Q23) --------------------
#
# #247 zeroes the off-chain rows of a user ``initial_state`` and of a restart
# file at runtime intake. The entry points underneath took whatever state they
# were handed, and on a linked deck that is not harmless: the ExB bracket takes
# the *whole* state to real space and dealiases only its output, so off-chain
# content aliases back onto the chain rows. Measured on this deck at saturated
# amplitude, one right-hand side's chain rows move 2.37e-02 relative.
#
# The contract these pin is: project, do not reject. A rejection would have to
# read the state's values, and these are differentiable entry points whose
# state is a tracer under ``jit`` and ``grad``; the projection is a mask fixed
# by the deck's topology and survives both.


def _linked_nonlinear_deck(boundary: str = "linked"):
    """A small nonlinear deck whose linked chains leave kx rows uncovered."""

    grid_cfg = GridConfig(
        Nx=8,
        Ny=8,
        Nz=8,
        Lx=6.28,
        Ly=6.28,
        boundary=boundary,
        jtwist=1 if boundary == "linked" else None,
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams()
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)
    return grid, geom, params, cache


def _supplied_and_on_cover(grid, cache, *, seed: int = 2301):
    """A saturated-amplitude broadband state and its projection onto the cover."""

    cover = linked_cover_mask_from_cache(cache)
    shape = (1, 2, 3, int(grid.ky.size), int(grid.kx.size), int(grid.z.size))
    rng = np.random.default_rng(seed)
    supplied = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        np.complex64
    )
    supplied = (supplied / np.max(np.abs(supplied))).astype(np.complex64)
    if cover is None:
        return supplied, supplied, None, np.zeros(shape, dtype=bool)
    cover_np = np.asarray(cover)
    off = np.broadcast_to(~cover_np[:, :, None], shape)
    on_cover = np.where(off, 0.0, supplied).astype(np.complex64)
    return supplied, on_cover, cover_np, off


_Q23_OPTIONS = dict(
    dt=1.0e-3,
    steps=3,
    method="rk3",
    terms=TermConfig(nonlinear=1.0),
    fixed_dt=True,
    resolved_diagnostics=False,
)


def test_prepare_projects_the_state_it_was_given_onto_the_linked_cover() -> None:
    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache)
    assert np.max(np.abs(supplied[off])) > 0.0

    prepared = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(supplied), grid, geom, params, **_Q23_OPTIONS
    )
    from_on_cover = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(on_cover), grid, geom, params, **_Q23_OPTIONS
    )

    assert prepared.cover_mask is not None
    held = np.asarray(prepared.initial_state)
    assert np.max(np.abs(held[off])) == 0.0
    # The state the object holds is the one it would have held had the caller
    # projected first -- bitwise, not to a tolerance.
    np.testing.assert_array_equal(held, np.asarray(from_on_cover.initial_state))


def test_prepared_run_with_a_supplied_state_matches_the_projected_state() -> None:
    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache)
    prepared = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(on_cover), grid, geom, params, **_Q23_OPTIONS
    )

    from_supplied, _diag_s, _fields_s = prepared.run_arrays(jnp.asarray(supplied))
    from_on_cover, _diag_c, _fields_c = prepared.run_arrays(jnp.asarray(on_cover))

    assert np.max(np.abs(np.asarray(from_supplied)[off])) == 0.0
    np.testing.assert_array_equal(np.asarray(from_supplied), np.asarray(from_on_cover))


def test_the_function_entry_point_takes_the_same_view_as_the_prepared_object() -> None:
    """Q18 made these one graph; the supplied-state contract must not split them."""

    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, _on_cover, _cover, _off = _supplied_and_on_cover(grid, cache)
    prepared = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(supplied), grid, geom, params, **_Q23_OPTIONS
    )

    _t, _diag, final_state, _fields = integrate_nonlinear_explicit_diagnostics_state(
        jnp.asarray(supplied), grid, geom, params, **_Q23_OPTIONS
    )
    prepared_final, _pdiag, _pfields = prepared.run_arrays()

    np.testing.assert_array_equal(np.asarray(final_state), np.asarray(prepared_final))


@requires_jax_07
def test_a_supplied_states_off_chain_rows_carry_no_gradient() -> None:
    """The VJP of the projection: exactly zero off the chains, unchanged on them."""

    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache)
    prepared = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(on_cover), grid, geom, params, **_Q23_OPTIONS
    )

    def final_energy(state):
        final_state, _diag, _fields = prepared.run_arrays(state)
        return jnp.sum(jnp.abs(final_state) ** 2)

    cotangent = np.asarray(jax.grad(final_energy)(jnp.asarray(supplied)))
    reference = np.asarray(jax.grad(final_energy)(jnp.asarray(on_cover)))

    assert np.max(np.abs(cotangent[off])) == 0.0
    assert np.max(np.abs(cotangent[~off])) > 0.0
    np.testing.assert_array_equal(cotangent[~off], reference[~off])


def test_the_objective_adjoint_projects_its_saturated_state() -> None:
    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, _off = _supplied_and_on_cover(grid, cache)
    geom_data = ensure_flux_tube_geometry_data(geom, grid.z)

    def window(state, drift_scale):
        scaled = replace(
            geom_data,
            gb_profile=geom_data.gb_profile * drift_scale,
            cv_profile=geom_data.cv_profile * drift_scale,
        )
        return nonlinear_heat_flux_window(
            jnp.asarray(state),
            grid,
            scaled,
            params,
            dt=1.0e-3,
            steps=3,
            method="rk2",
            terms=TermConfig(nonlinear=1.0),
        )

    one = jnp.asarray(1.0)
    value_s, grad_s = jax.value_and_grad(lambda a: window(supplied, a))(one)
    value_c, grad_c = jax.value_and_grad(lambda a: window(on_cover, a))(one)

    assert float(grad_c) != 0.0
    np.testing.assert_array_equal(np.asarray(value_s), np.asarray(value_c))
    np.testing.assert_array_equal(np.asarray(grad_s), np.asarray(grad_c))


def test_a_periodic_deck_is_untouched_by_the_supplied_state_contract() -> None:
    """No cover, no mask, no extra array op: these decks trace exactly as before."""

    grid, geom, params, cache = _linked_nonlinear_deck("periodic")
    supplied, _on_cover, cover, _off = _supplied_and_on_cover(grid, cache)
    assert cover is None
    assert linked_cover_mask_from_cache(cache) is None
    assert mask_supplied_state(supplied, cache) is supplied

    prepared = prepare_nonlinear_explicit_diagnostics(
        jnp.asarray(supplied), grid, geom, params, **_Q23_OPTIONS
    )

    assert prepared.cover_mask is None
    final_state, _diag, _fields = prepared.run_arrays(jnp.asarray(supplied))
    _t, _rdiag, runtime_final, _rfields = (
        integrate_nonlinear_explicit_diagnostics_state(
            jnp.asarray(supplied), grid, geom, params, **_Q23_OPTIONS
        )
    )
    np.testing.assert_array_equal(np.asarray(final_state), np.asarray(runtime_final))


# --- Q29: the raw drivers take the same view of a state they did not build ---

_Q29_RAW_OPTIONS = dict(
    dt=1.0e-3,
    steps=3,
    method="rk3",
    terms=TermConfig(nonlinear=1.0),
)


def _raw_cached(state, cache, params):
    G_final, _fields = integrate_nonlinear_cached(
        jnp.asarray(state), cache, params, **_Q29_RAW_OPTIONS
    )
    return np.asarray(G_final)


def test_integrate_nonlinear_cached_projects_a_supplied_state() -> None:
    """The raw cached driver runs the projected state, not the one handed in."""

    grid, _geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache, seed=2901)
    assert np.max(np.abs(supplied[off])) > 0.0

    from_supplied = _raw_cached(supplied, cache, params)
    from_on_cover = _raw_cached(on_cover, cache, params)

    assert np.max(np.abs(from_supplied[off])) == 0.0
    np.testing.assert_array_equal(from_supplied, from_on_cover)


def test_integrate_nonlinear_projects_a_supplied_state_like_its_cached_driver() -> None:
    """Both raw doors are one contract: the grid door delegates to the cached one."""

    grid, geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache, seed=2902)

    G_grid_door, _fields = integrate_nonlinear(
        jnp.asarray(supplied), grid, geom, params, cache=cache, **_Q29_RAW_OPTIONS
    )
    G_grid_door = np.asarray(G_grid_door)

    assert np.max(np.abs(G_grid_door[off])) == 0.0
    np.testing.assert_array_equal(G_grid_door, _raw_cached(on_cover, cache, params))


def test_a_raw_driver_gives_a_supplied_states_off_chain_rows_no_gradient() -> None:
    """Projection at a raw driver traces and differentiates, as it does above."""

    grid, _geom, params, cache = _linked_nonlinear_deck()
    supplied, on_cover, _cover, off = _supplied_and_on_cover(grid, cache, seed=2903)

    def final_energy(state):
        G_final, _fields = integrate_nonlinear_cached(
            state, cache, params, **_Q29_RAW_OPTIONS
        )
        return jnp.sum(jnp.abs(G_final) ** 2)

    cotangent = np.asarray(jax.grad(final_energy)(jnp.asarray(supplied)))
    reference = np.asarray(jax.grad(final_energy)(jnp.asarray(on_cover)))

    assert np.max(np.abs(cotangent[off])) == 0.0
    assert np.max(np.abs(cotangent[~off])) > 0.0
    np.testing.assert_array_equal(cotangent[~off], reference[~off])


def test_a_periodic_deck_reaches_the_raw_drivers_unprojected() -> None:
    """No cover means the same array object, so the driver's graph is unchanged."""

    grid, geom, params, cache = _linked_nonlinear_deck("periodic")
    supplied, _on_cover, cover, _off = _supplied_and_on_cover(grid, cache, seed=2904)
    assert cover is None
    assert linked_cover_mask_from_cache(cache) is None
    state = jnp.asarray(supplied)
    assert mask_supplied_state(state, cache) is state

    from_cached = _raw_cached(supplied, cache, params)
    G_grid_door, _fields = integrate_nonlinear(
        jnp.asarray(supplied), grid, geom, params, cache=cache, **_Q29_RAW_OPTIONS
    )
    np.testing.assert_array_equal(from_cached, np.asarray(G_grid_door))


# ---- from test_nonlinear_helpers_extra.py ----

from functools import partial
import jax
from gkx import runtime as rt
from gkx.geometry.core import apply_geometry_grid_defaults
from gkx.operators.linear.dissipation import multispecies_collision_invariant_rates
from gkx.operators.nonlinear.policies import measured_streaming_frequency
from gkx.solvers.nonlinear.imex import (
    ARS_TABLEAUX,
    build_chain_implicit_linear,
    collisions_bind,
    refactor_chain_implicit_linear,
)
from gkx.operators.linear.cache_arrays import keep_moments
from gkx.solvers.nonlinear.diagnostics import species_moments, stiff_linear_split
from gkx.solvers.nonlinear.diagnostics import nonlinear_rhs_cached
from gkx.terms.assembly import assemble_rhs_cached, compute_fields_cached
from gkx.terms.config import TermConfig
from gkx.workflows.runtime.toml import load
from support.paths import REPO_ROOT

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields as dataclass_fields
from gkx.diagnostics.metadata import ResolvedDiagnostics, SimulationDiagnostics
from gkx.operators.fluxes import heat_flux_species
from gkx.operators.linear.cache_builder import (
    update_linear_cache_for_sheared_kx,
)
from gkx.operators.linear.params import (
    Species,
    build_linear_params,
)
from gkx.operators.nonlinear import projection as nonlinear_projection
from gkx.operators.nonlinear.collisions import (
    _collision_damping,
)
from gkx.solvers.nonlinear.diagnostics import (
    _sample_indices_with_final,
    finalize_nonlinear_scan_diagnostics,
    run_sampled_explicit_diagnostic_scan,
    sampled_scan_intervals,
    _integrate_nonlinear_explicit_diagnostics_impl,
    integrate_nonlinear_imex_diagnostics,
)
from gkx.operators.nonlinear.policies import (
    _nonlinear_cfl_frequency_components,
)
from gkx.operators.nonlinear.projection import (
    _make_hermitian_projector,
)
from gkx.solvers.nonlinear.diagnostics import (
    DIVERGENCE_KNEE_STEPS,
)
from gkx.solvers.nonlinear.transport_window import (
    integrate_nonlinear_sheared,
    integrate_nonlinear_sheared_transport,
)
from gkx.workflows.runtime.chunks import run_adaptive_runtime_chunk_loop
from pathlib import Path
from scripts.checks._gates.nonlinear_replicates import (
    nonlinear_replicate_spread_report,
)
import gc
import pytest
from support.helpers import only_terms, patch_attrs, spectral_grid, uniform_moments


_SPECIES_PARAMS = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]))


def _raise(message: str):
    def _fail(*args, **kwargs):
        raise AssertionError(message)

    return _fail


def test_sample_indices_with_final_preserves_last_step() -> None:
    np.testing.assert_array_equal(
        _sample_indices_with_final(6, 4), np.asarray([0, 4, 5])
    )
    np.testing.assert_array_equal(_sample_indices_with_final(6, 5), np.asarray([0, 5]))
    assert isinstance(_sample_indices_with_final(6, 1), slice)


def test_sampled_scan_intervals_and_runner_retain_final_step() -> None:
    np.testing.assert_array_equal(
        sampled_scan_intervals(5, 2), np.asarray([1, 2, 2], dtype=np.int32)
    )

    def run(scale, steps=5, stride=2):
        def step_fn(carry, _idx):
            G, _G_prev, fields_prev, _diag_prev, t_prev, dt_prev = carry
            G_new = G * scale + 1
            saved = (G_new * 100, t_prev + dt_prev, dt_prev)
            return (G_new, G_new, fields_prev + 10, *saved), saved

        return run_sampled_explicit_diagnostic_scan(
            step_fn,
            (jnp.asarray(0.0),) * 5 + (jnp.asarray(0.1),),
            steps=steps,
            stride=stride,
        )

    final_carry, diag_out = run(1.0)
    for steps, stride in ((5, 2), (6, 4), (1, 7), (7, 1), (0, 3)):
        value, grad = jax.jit(
            jax.value_and_grad(lambda scale: run(scale, steps, stride)[0][0])
        )(1.0)
        np.testing.assert_allclose(value, steps)
        np.testing.assert_allclose(grad, steps * (steps - 1) / 2)

    G_final, _G_prev, fields_final, diag_final, t_final, dt_final = final_carry
    diag_t, t_t, dt_t = diag_out
    np.testing.assert_allclose(np.asarray(G_final), 5)
    np.testing.assert_allclose(np.asarray(fields_final), 50)
    np.testing.assert_allclose(np.asarray(diag_final), 500)
    np.testing.assert_allclose(np.asarray(t_final), 0.5, rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(dt_final), 0.1, rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(diag_t), [100, 300, 500])
    np.testing.assert_allclose(np.asarray(t_t), [0.1, 0.3, 0.5], rtol=1.0e-6)
    np.testing.assert_allclose(np.asarray(dt_t), [0.1, 0.1, 0.1], rtol=1.0e-6)


def test_finalize_nonlinear_scan_diagnostics_applies_output_sampling() -> None:
    series = tuple(jnp.arange(5, dtype=jnp.float32) + offset for offset in range(12))
    diag = (*series, ())
    t = jnp.linspace(0.1, 0.5, 5, dtype=jnp.float32)
    dt = jnp.ones((5,), dtype=jnp.float32) * 0.1

    sampled, retained = (
        finalize_nonlinear_scan_diagnostics(
            diag,
            t=t,
            dt_series=dt,
            stride=3,
            sampled_scan=sampled_scan,
            resolved_diagnostics=False,
        )
        for sampled_scan in (False, True)
    )

    np.testing.assert_allclose(np.asarray(sampled.t), [0.1, 0.4, 0.5])
    np.testing.assert_allclose(np.asarray(sampled.gamma_t), [0.0, 3.0, 4.0])
    np.testing.assert_allclose(np.asarray(retained.t), np.asarray(t))
    np.testing.assert_allclose(np.asarray(retained.gamma_t), np.arange(5))


def _advance(state, kx, ky, rate, previous_time, time, *, x0=1.0, **kwargs):
    return nonlinear_projection.advance_shearing_coordinates(
        state,
        kx=kx,
        ky=ky,
        x0=x0,
        shear_rate=rate,
        previous_time=previous_time,
        time=time,
        **kwargs,
    )


def test_shearing_coordinates_follow_analytic_wave_and_inverse_remap() -> None:
    kx = jnp.asarray([0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0])
    ky = jnp.asarray([0.0, 1.0])
    state = jnp.zeros((1, 2, 8, 1), dtype=jnp.complex64)
    state = state.at[0, 1, 0, 0].set(2.0 - 0.5j)

    update = _advance(state, kx, ky, 1.0, 0.0, 1.2)

    assert int(update.cumulative_mode_shift[0]) == 0
    assert int(update.cumulative_mode_shift[1]) == -1
    np.testing.assert_allclose(update.state[0, 1, 7, 0], 2.0 - 0.5j)
    np.testing.assert_allclose(update.effective_kx[1, 7], -1.2, atol=2.0e-7)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(update.state)),
        np.linalg.norm(np.asarray(state)),
        atol=2.0e-7,
    )

    restored = _advance(update.state, kx, ky, 1.0, 1.2, 0.0)
    np.testing.assert_allclose(restored.state, state, atol=2.0e-7)


def test_shearing_coordinate_tangent_matches_finite_difference() -> None:
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0])
    ky = jnp.asarray([0.0, 0.75])
    state = jnp.ones((1, 2, 4, 1), dtype=jnp.complex64)

    def observables(rate):
        update = _advance(state, kx, ky, rate, 0.0, 0.2)
        return update.effective_kx[1, 0], update.phase[1, 1]

    rate = jnp.asarray(0.4, dtype=jnp.float32)
    _, tangent = jax.jvp(observables, (rate,), (jnp.ones_like(rate),))
    step = jnp.asarray(1.0e-3, dtype=jnp.float32)
    plus = observables(rate + step)
    minus = observables(rate - step)
    finite_difference = tuple((hi - lo) / (2.0 * step) for hi, lo in zip(plus, minus))
    np.testing.assert_allclose(tangent[0], finite_difference[0], rtol=2.0e-4)
    np.testing.assert_allclose(tangent[1], finite_difference[1], rtol=3.0e-4)

    def radial_scale_observable(x0):
        return _advance(state, kx, ky, rate, 0.0, 0.2, x0=x0).phase[1, 1]

    x0 = jnp.asarray(1.1, dtype=jnp.float32)
    _, x0_tangent = jax.jvp(radial_scale_observable, (x0,), (jnp.ones_like(x0),))
    x0_plus = radial_scale_observable(x0 + step)
    x0_minus = radial_scale_observable(x0 - step)
    x0_finite_difference = (x0_plus - x0_minus) / (2.0 * step)
    np.testing.assert_allclose(x0_tangent, x0_finite_difference, rtol=3.0e-4)


def test_sheared_integrator_zero_shear_identity_and_full_step_remap() -> None:
    # The comparison is between the compressed bracket and the full-complex one
    # the sheared integrator uses by default, and the state is Hermitian-
    # completed, so both sides need the two-sided ky axis.
    grid = spectral_grid(4, 4, 4, ky_layout="full")
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
    state = jnp.zeros((1, 1, 4, 4, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(0.2 + 0.1j)
    project_state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)
    state = project_state(state)
    nonlinear_only = only_terms(nonlinear=1.0)
    run = partial(integrate_nonlinear_sheared, state, grid, geom, params, cache=cache)
    for method in ("rk2", "rk3"):
        reference_state, reference_fields = integrate_nonlinear_cached(
            jnp.asarray(np.asarray(state).copy()),
            cache,
            params,
            dt=0.02,
            steps=2,
            method=method,
            terms=nonlinear_only,
            compressed_real_fft=True,
        )
        options = dict(dt=0.02, steps=2, shear_rate=0.0, method=method)
        sheared_state, sheared_fields = run(terms=nonlinear_only, **options)
        np.testing.assert_allclose(sheared_state, reference_state, atol=2.0e-7)
        np.testing.assert_allclose(sheared_fields.phi, reference_fields.phi, atol=2e-7)
        np.testing.assert_allclose(
            sheared_state, project_state(sheared_state), atol=1e-7
        )
        state_only = run(terms=nonlinear_only, return_fields=False, **options)
        np.testing.assert_allclose(state_only, reference_state, atol=2.0e-7)

    disabled = TermConfig(*([0.0] * 12))
    remapped_state, _ = run(
        dt=0.4, steps=3, shear_rate=1.0, method="rk2", terms=disabled
    )
    expected = nonlinear_projection.advance_shearing_coordinates(
        state,
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=1.2,
        dealias_mask=grid.dealias_mask,
    ).state
    np.testing.assert_allclose(remapped_state, project_state(expected), atol=2.0e-7)

    with pytest.raises(ValueError, match="steps must be at least one"):
        run(dt=0.1, steps=0, shear_rate=1.0)
    with pytest.raises(ValueError, match="method must be"):
        run(dt=0.1, steps=1, shear_rate=1.0, method="rk4")


@pytest.mark.parametrize("method", ["rk2", "rk3", "imex"])
def test_linked_sheared_integrator_has_exact_zero_shear_trajectory_identity(
    method: str,
) -> None:
    # Both trajectories run with ``compressed_real_fft=False``, and that bracket
    # is defined on the two-sided ky axis only.
    grid = spectral_grid(8, 4, 8, boundary="linked", ky_layout="full")
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=0.3, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    state = jnp.zeros((1, 2, 4, 8, 8), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(0.2 + 0.1j)
    state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(state)
    streaming_only = only_terms(streaming=1.0)
    state_host = np.asarray(state)

    reference_state, reference_fields = integrate_nonlinear_cached(
        jnp.asarray(state_host.copy()),
        cache,
        params,
        dt=0.01,
        steps=2,
        method=method,
        terms=streaming_only,
        compressed_real_fft=False,
    )
    sheared_state, sheared_fields = integrate_nonlinear_sheared(
        jnp.asarray(state_host.copy()),
        grid,
        geom,
        params,
        dt=0.01,
        steps=2,
        shear_rate=0.0,
        method=method,
        cache=cache,
        terms=streaming_only,
        compressed_real_fft=False,
    )

    np.testing.assert_allclose(sheared_state, reference_state, atol=2.0e-7)
    np.testing.assert_allclose(sheared_fields.phi, reference_fields.phi, atol=2.0e-7)


def test_linked_sheared_cache_preserves_chains_and_has_correct_tangent() -> None:
    grid = spectral_grid(8, 4, 8, boundary="linked")
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=0.3, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)

    def shifted_cache(displacement):
        effective_kx = cache.kx_grid - cache.ky_grid * displacement
        return update_linear_cache_for_sheared_kx(
            cache, grid, geom, params, effective_kx
        )

    updated = shifted_cache(jnp.asarray(0.17, dtype=cache.kx_grid.dtype))
    np.testing.assert_array_equal(updated.kx_link_plus, cache.kx_link_plus)
    np.testing.assert_array_equal(updated.kx_link_minus, cache.kx_link_minus)
    for ky_index in range(cache.kx_link_plus.shape[0]):
        valid = np.asarray(cache.kx_link_mask_plus[ky_index])
        linked = np.asarray(cache.kx_link_plus[ky_index])[valid]
        source = np.arange(cache.kx_link_plus.shape[1])[valid]
        before = np.asarray(
            cache.kx_grid[ky_index, linked] - cache.kx_grid[ky_index, source]
        )
        after = np.asarray(
            updated.kx_grid[ky_index, linked] - updated.kx_grid[ky_index, source]
        )
        np.testing.assert_allclose(after, before, atol=2.0e-7)

    def objective(displacement):
        return jnp.sum(shifted_cache(displacement).kperp2 ** 2)

    displacement = jnp.asarray(0.17, dtype=cache.kx_grid.dtype)
    _, tangent = jax.jvp(objective, (displacement,), (jnp.ones_like(displacement),))
    step = jnp.asarray(1.0e-3, dtype=displacement.dtype)
    finite_difference = (
        objective(displacement + step) - objective(displacement - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-3, atol=2.0e-4)


def _small_sheared_transport_case():
    # The sheared-transport routes evaluate the bracket with full complex
    # transforms unless asked otherwise, and the state is built by completing
    # the Hermitian partners, so this case lives on the two-sided ky axis.
    grid = spectral_grid(4, 4, 4, ky_layout="full")
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0, tprim=2.0, fprim=0.5
    )
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=4)
    state = jnp.zeros((1, 4, 4, 4, 4), dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, :].set(
        jnp.asarray([0.2 + 0.1j, 0.1 - 0.05j, 0.15 + 0.02j, 0.07 - 0.03j])
    )
    state = state.at[0, 2, 1, 1, :].set(0.03 + 0.02j)
    state = _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(state)
    terms = TermConfig(collisions=0.0, hypercollisions=0.0, end_damping=0.0)
    return grid, geom, params, cache, state, terms


def _transport_runner():
    """The small sheared case and a runner ``run(state, **options)`` bound to it."""
    grid, geom, params, cache, state, terms = _small_sheared_transport_case()

    def run(G, *, route=integrate_nonlinear_sheared_transport, **options):
        options = {"cache": cache, "terms": terms, **options}
        return route(G, grid, geom, params, **options)

    return SimpleNamespace(
        grid=grid, geom=geom, params=params, cache=cache, state=state, terms=terms
    ), run


def test_sheared_imex_rejects_unvalidated_adaptive_and_collision_routes() -> None:
    case, run = _transport_runner()
    options = dict(dt=0.02, steps=2, shear_rate=0.1, method="imex")
    with pytest.raises(ValueError, match="requires fixed_dt=True"):
        run(case.state, fixed_dt=False, **options)
    with pytest.raises(NotImplementedError, match="custom collision operators"):
        run(
            case.state,
            route=integrate_nonlinear_sheared,
            collision_operator=object(),
            **options,
        )


@pytest.mark.parametrize("method", ["rk2", "imex"])
def test_sheared_transport_trace_matches_canonical_final_heat_flux(method: str) -> None:
    case, run = _transport_runner()
    trace = run(case.state, dt=0.02, steps=3, shear_rate=0.0, method=method)

    np.testing.assert_allclose(trace.time, [0.02, 0.04, 0.06], rtol=1.0e-6)
    assert trace.heat_flux.shape == (3, 1)
    _, flux_fac = fieldline_quadrature_weights(case.geom, case.grid)
    _, final_fields = nonlinear_rhs_cached(
        trace.final_state,
        case.cache,
        case.params,
        case.terms,
        compressed_real_fft=False,
    )
    zeros = jnp.zeros_like(final_fields.phi)
    expected = heat_flux_species(
        trace.final_state,
        final_fields.phi,
        zeros,
        zeros,
        case.cache,
        case.grid,
        case.params,
        flux_fac,
    )
    np.testing.assert_allclose(trace.heat_flux[-1], expected, rtol=2.0e-6, atol=2.0e-7)


def test_sheared_transport_compressed_bracket_matches_full_fractional_phase() -> None:
    case, run = _transport_runner()
    options = dict(
        dt=0.017, steps=3, shear_rate=0.37, method="rk3", differentiable=False
    )
    full = run(case.state, compressed_real_fft=False, **options)
    compressed = run(case.state, compressed_real_fft=True, **options)

    np.testing.assert_allclose(compressed.time, full.time, atol=1.0e-7)
    for name in ("final_state", "heat_flux"):
        np.testing.assert_allclose(
            getattr(compressed, name), getattr(full, name), rtol=2.0e-5, atol=2.0e-7
        )


def test_sheared_transport_scale_does_not_change_trajectory() -> None:
    case, run = _transport_runner()
    options = dict(dt=0.02, steps=2, shear_rate=0.2)
    base = run(case.state, **options)
    scaled = run(case.state, flux_scale=3.0, **options)
    fast = run(case.state, differentiable=False, **options)

    np.testing.assert_allclose(scaled.final_state, base.final_state, atol=2.0e-7)
    np.testing.assert_allclose(scaled.heat_flux, 3.0 * base.heat_flux, atol=2.0e-7)
    np.testing.assert_allclose(fast.final_state, base.final_state, atol=2.0e-7)
    np.testing.assert_allclose(fast.heat_flux, base.heat_flux, atol=2.0e-7)


def test_sheared_transport_adaptive_cfl_records_accepted_time_steps() -> None:
    case, run_case = _transport_runner()

    def run(amplitude):
        return run_case(
            amplitude * case.state,
            dt=0.02,
            steps=4,
            shear_rate=0.1,
            method="rk3",
            fixed_dt=False,
            dt_min=1.0e-7,
            dt_max=0.02,
            cfl=0.9,
        )

    amplitude = jnp.asarray(1.0e5, dtype=jnp.float32)
    trace = run(amplitude)

    accepted_dt = np.diff(np.concatenate(([0.0], np.asarray(trace.time))))
    assert np.all(np.isfinite(accepted_dt))
    assert np.all(accepted_dt > 0.0)
    assert np.max(accepted_dt) <= 0.02 + 1.0e-7
    assert np.min(accepted_dt) < 0.019
    assert np.all(np.isfinite(np.asarray(trace.heat_flux)))

    def final_time(scale):
        return run(scale).time[-1]

    _, tangent = jax.jvp(final_time, (amplitude,), (jnp.ones_like(amplitude),))
    step = 100.0
    finite_difference = (
        final_time(amplitude + step) - final_time(amplitude - step)
    ) / (2.0 * step)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1.0e-3, atol=1.0e-12)


def test_sheared_transport_restart_preserves_physical_time_and_state() -> None:
    case, run = _transport_runner()
    options = dict(dt=0.02, shear_rate=0.2, method="rk3", differentiable=False)
    complete = run(case.state, steps=4, **options)
    first = run(case.state, steps=2, **options)
    second = run(
        first.final_state,
        steps=2,
        initial_time=first.time[-1],
        initial_dt=first.time[-1] - first.time[-2],
        **options,
    )

    np.testing.assert_allclose(second.final_state, complete.final_state, atol=3.0e-7)
    np.testing.assert_allclose(
        jnp.concatenate((first.time, second.time)), complete.time, atol=1.0e-7
    )
    np.testing.assert_allclose(
        jnp.concatenate((first.heat_flux, second.heat_flux)),
        complete.heat_flux,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


@pytest.mark.parametrize("method", ["rk2", "imex"])
def test_sheared_transport_gradient_matches_tangent_and_finite_difference(
    method: str,
) -> None:
    jax.clear_caches()
    case, run = _transport_runner()

    def objective(shear_rate):
        trace = run(case.state, dt=0.02, steps=2, shear_rate=shear_rate, method=method)
        return jnp.mean(trace.heat_flux)

    shear_rate = jnp.asarray(0.2, dtype=jnp.float32)
    _, tangent = jax.jvp(objective, (shear_rate,), (jnp.ones_like(shear_rate),))
    gradient = jax.grad(objective)(shear_rate)
    step = jnp.asarray(0.05, dtype=shear_rate.dtype)
    finite_difference = (
        objective(shear_rate + step) - objective(shear_rate - step)
    ) / (2.0 * step)

    assert np.isfinite(float(gradient))
    np.testing.assert_allclose(tangent, gradient, rtol=3.0e-5, atol=1.0e-11)
    np.testing.assert_allclose(gradient, finite_difference, rtol=5.0e-3, atol=1.0e-10)
    jax.clear_caches()
    gc.collect()


@pytest.mark.parametrize(
    ("method", "minimum_order"),
    [("rk2", 1.8), ("rk3", 2.6), ("imex", 0.8)],
)
def test_sheared_runge_kutta_recovers_observed_order_on_physical_rhs(
    method: str,
    minimum_order: float,
) -> None:
    jax.clear_caches()
    # ``integrate_nonlinear_sheared`` brackets with full complex transforms by
    # default, and the initial state is Hermitian-completed, so the convergence
    # study is run on the two-sided ky axis.
    grid = spectral_grid(4, 4, 4, ky_layout="full")
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        rho_star=1.0,
        nu_hyper=0.0,
        nu_hyper_m=0.0,
        tprim=2.0,
        fprim=0.5,
    )
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    initial = np.zeros((1, 2, 4, 4, 4), dtype=np.complex64)
    initial[0, 0, 1, 0, :] = np.asarray(
        [0.2 + 0.1j, 0.1 - 0.05j, 0.15 + 0.02j, 0.07 - 0.03j]
    )
    initial[0, 1, 1, 1, :] = 0.03 + 0.02j
    initial = np.asarray(
        _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(initial)
    )
    drift_drive = only_terms(curvature=1.0, gradb=1.0, diamagnetic=1.0)

    solutions = []
    final_time = 0.08
    with jax.disable_jit():
        for steps in (2, 4, 8):
            state, _ = integrate_nonlinear_sheared(
                jnp.asarray(initial.copy()),
                grid,
                geom,
                params,
                dt=final_time / steps,
                steps=steps,
                shear_rate=0.3,
                method=method,
                cache=cache,
                terms=drift_drive,
            )
            solutions.append(np.asarray(state))

    coarse_difference = np.linalg.norm(solutions[0] - solutions[1])
    fine_difference = np.linalg.norm(solutions[1] - solutions[2])
    observed_order = np.log(coarse_difference / fine_difference) / np.log(2.0)
    assert observed_order > minimum_order


def test_strong_flow_shear_suppresses_linear_itg_amplitude_after_dt_refinement() -> (
    None
):
    # ``integrate_nonlinear_sheared`` brackets with full complex transforms by
    # default, and the seed is Hermitian-completed, so this deck asks for the
    # two-sided ky axis.
    grid = spectral_grid(
        8, 4, 8, lx=2.0 * np.pi / 0.2, ly=2.0 * np.pi / 0.3, ky_layout="full"
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18)
    params = LinearParams(
        rho_star=1.0,
        nu_hyper=0.0,
        nu_hyper_m=0.0,
        tprim=6.9,
        fprim=2.2,
        damp_ends_rate=0.0,
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    rng = np.random.default_rng(10)
    initial = np.zeros((2, 4, 4, 8, 8), dtype=np.complex64)
    initial[:, :, 1, 0, :] = 1.0e-4 * (
        rng.normal(size=(2, 4, 8)) + 1j * rng.normal(size=(2, 4, 8))
    )
    initial = np.asarray(
        _make_hermitian_projector(np.asarray(grid.ky), nx=grid.kx.size)(initial)
    )
    linear_itg = only_terms(
        streaming=1.0, mirror=1.0, curvature=1.0, gradb=1.0, diamagnetic=1.0
    )

    amplitudes: dict[tuple[float, float], float] = {}
    for dt in (0.02, 0.01):
        for shear_rate in (0.0, 0.5):
            _, fields = integrate_nonlinear_sheared(
                jnp.asarray(initial.copy()),
                grid,
                geom,
                params,
                dt=dt,
                steps=int(round(2.0 / dt)),
                shear_rate=shear_rate,
                method="rk2",
                cache=cache,
                terms=linear_itg,
            )
            amplitudes[(dt, shear_rate)] = float(jnp.linalg.norm(fields.phi[-1]))

    for shear_rate in (0.0, 0.5):
        coarse = amplitudes[(0.02, shear_rate)]
        fine = amplitudes[(0.01, shear_rate)]
        assert abs(coarse - fine) / fine < 0.01
    assert amplitudes[(0.01, 0.5)] / amplitudes[(0.01, 0.0)] < 0.8


def test_collision_damping_and_imex_operator_builder(monkeypatch) -> None:
    cache = SimpleNamespace(lb_lam=jnp.ones((2, 2, 1, 1, 1), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.1)
    term_cfg = TermConfig(collisions=0.5, hypercollisions=2.0)
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * 3.0,
    )
    damp = _collision_damping(
        cache, params, term_cfg, jnp.float32, squeeze_species=False
    )
    np.testing.assert_allclose(np.asarray(damp), 2.0 * 3.0)

    cache6 = SimpleNamespace(lb_lam=jnp.ones((1, 2, 2, 1, 1, 1), dtype=jnp.float32))
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype),
    )
    squeezed = _collision_damping(
        cache6,
        SimpleNamespace(nu=0.4),
        TermConfig(collisions=1.0, hypercollisions=1.0),
        jnp.float32,
        squeeze_species=True,
    )
    assert squeezed.shape == (2, 2, 1, 1, 1)

    cache_low_rank = SimpleNamespace(
        lb_lam=jnp.ones((2, 2), dtype=jnp.float32),
        b=jnp.zeros((1, 1, 1, 1), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.operators.nonlinear.collisions.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones((1, 2, 2, 1, 1, 1), dtype=dtype),
    )
    squeezed_low_rank = _collision_damping(
        cache_low_rank,
        SimpleNamespace(nu=jnp.asarray([0.4], dtype=jnp.float32)),
        TermConfig(collisions=1.0, hypercollisions=1.0),
        jnp.float32,
        squeeze_species=True,
    )
    assert squeezed_low_rank.shape == (2, 2, 1, 1, 1)
    np.testing.assert_allclose(np.asarray(squeezed_low_rank), 1.0)

    monkeypatch.setattr(
        "gkx.operators.nonlinear.policies._build_implicit_operator",
        lambda *args, **kwargs: (
            jnp.zeros((1, 2, 2, 1, 1, 1), dtype=jnp.complex64),
            (1, 2, 2, 1, 1, 1),
            4,
            jnp.asarray(0.1, dtype=jnp.float32),
            None,
            lambda x: x,
            True,
        ),
    )
    op = build_nonlinear_imex_operator(
        jnp.zeros((2, 2, 1, 1, 1), dtype=jnp.complex64),
        SimpleNamespace(),
        SimpleNamespace(),
        dt=0.1,
    )
    assert op.shape == (1, 2, 2, 1, 1, 1)
    assert op.squeeze_species is True


def test_nonlinear_cfl_frequency_components_recovers_spectral_gradient_cfl() -> None:
    """CFL estimate should reduce to the pseudo-spectral derivative maximum."""

    ny = nx = 4
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky_grid, kx_grid = jnp.meshgrid(ky, kx, indexing="ij")
    grid = SimpleNamespace(ky=ky, kx=kx)
    cache = SimpleNamespace(kx_grid=kx_grid, ky_grid=ky_grid)

    def _sin_x_hat(amplitude: float) -> jnp.ndarray:
        # Coefficients for amplitude * sin(x); irfft/ifft paths multiply by N.
        field = jnp.zeros((ny, nx, 1), dtype=jnp.complex64)
        field = field.at[0, 1, 0].set(-0.5j * amplitude)
        return field.at[0, -1, 0].set(0.5j * amplitude)

    phi_amp = 1.25
    apar_amp = 0.5
    bpar_amp = 0.25
    vpar_max = 2.0
    muB_max = 3.0
    fields = FieldState(
        phi=_sin_x_hat(phi_amp),
        apar=_sin_x_hat(apar_amp),
        bpar=_sin_x_hat(bpar_amp),
    )

    omega_x, omega_y = _nonlinear_cfl_frequency_components(
        fields,
        grid,
        cache,
        compressed_real_fft=False,
        kx_max=5.0,
        ky_max=3.0,
        kxfac=-2.0,
        vpar_max=vpar_max,
        muB_max=muB_max,
    )

    expected_vmax_y = phi_amp + vpar_max * apar_amp + muB_max * bpar_amp
    assert float(omega_x) == pytest.approx(0.0, abs=1.0e-6)
    assert float(omega_y) == pytest.approx(
        0.5 * 2.0 * 3.0 * expected_vmax_y, rel=1.0e-6
    )


_SI = "gkx.solvers.nonlinear.diagnostics"
_DI = "gkx.solvers.nonlinear.diagnostics"


def _diag_call(fn, G0=None, **kwargs):
    if G0 is None:
        G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    ns3 = (SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    return fn(G0, *ns3, dt=0.1, steps=2, **kwargs)


def test_integrate_nonlinear_builds_cache_and_rejects_bad_shape(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []
    patch_attrs(
        monkeypatch,
        _SI,
        ensure_flux_tube_geometry_data=lambda geom, z: "geom_eff",
        build_linear_cache=lambda grid, geom, params, Nl, Nm: (
            calls.append((Nl, Nm)) or "cache"
        ),
        integrate_nonlinear_cached=lambda *a, **k: ("G_out", "fields_out"),
    )
    grid = SimpleNamespace(z=np.array([-1.0, 0.0, 1.0, 2.0]))
    for shape in ((2, 3, 1, 1, 4), (1, 2, 3, 1, 1, 4)):
        G0 = jnp.zeros(shape, dtype=jnp.complex64)
        out = integrate_nonlinear(G0, grid, object(), object(), dt=0.1, steps=2)
        assert out == ("G_out", "fields_out")
    assert calls == [(2, 3), (2, 3)]

    with pytest.raises(ValueError):
        integrate_nonlinear(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            SimpleNamespace(z=np.array([0.0])),
            object(),
            object(),
            dt=0.1,
            steps=2,
        )


def test_nonlinear_diagnostics_routes_explicit_imex_and_state(monkeypatch) -> None:
    payload = ("t_explicit", "diag_explicit", "G_final", "fields_final")
    patch_attrs(
        monkeypatch,
        _DI,
        integrate_nonlinear_imex_diagnostics=lambda *a, **k: ("t_imex", "diag_imex"),
        _integrate_nonlinear_explicit_diagnostics_impl=lambda *a, **k: payload,
    )
    imex = _diag_call(integrate_nonlinear_explicit_diagnostics, method="imex")
    assert imex == ("t_imex", "diag_imex")
    explicit = _diag_call(integrate_nonlinear_explicit_diagnostics, method="rk3")
    assert explicit == payload[:2]
    state = _diag_call(integrate_nonlinear_explicit_diagnostics_state, method="rk3")
    assert state == payload
    with pytest.raises(ValueError):
        _diag_call(integrate_nonlinear_explicit_diagnostics_state, method="imex")


def test_explicit_diagnostics_impl_rejects_imex_and_bad_state_rank(
    monkeypatch,
) -> None:
    monkeypatch.setattr(f"{_DI}.ensure_flux_tube_geometry_data", lambda geom, z: geom)
    grid = SimpleNamespace(z=np.array([0.0]))
    cases = [
        (
            (1, 1, 1, 1, 1),
            "imex",
            object(),
            "Final-state runtime diagnostics helper only supports explicit methods",
        ),
        ((2, 2), "rk2", None, "G0 must have shape"),
    ]
    for shape, method, cache, match in cases:
        with pytest.raises(ValueError, match=match):
            _integrate_nonlinear_explicit_diagnostics_impl(
                jnp.zeros(shape, dtype=jnp.complex64),
                grid,
                object(),
                object(),
                dt=0.1,
                steps=1,
                method=method,
                cache=cache,
            )


def _ones_tuple(*shapes):
    return tuple(jnp.ones(shape, dtype=jnp.float32) for shape in shapes)


def _diag_grid_cache(ky, kx, **cache_extra):
    grid = SimpleNamespace(
        ky=np.array(ky, dtype=float),
        kx=np.array(kx, dtype=float),
        z=np.array([0.0, 1.0], dtype=float),
        dealias_mask=np.ones((len(ky), len(kx)), dtype=bool),
    )
    cache = SimpleNamespace(
        ky=jnp.asarray(grid.ky),
        kx=jnp.asarray(grid.kx),
        kxfac=1.0,
        l=jnp.asarray([0], dtype=jnp.int32),
        m=jnp.asarray([[0]], dtype=jnp.int32),
        **cache_extra,
    )
    return grid, cache


def _diag_stubs(mask_shape, *, rhs, fields_fn) -> dict[str, object]:
    """The geometry, quadrature, CFL and RHS stubs every diagnostic-impl test uses."""
    return dict(
        ensure_flux_tube_geometry_data=lambda geom, z: geom,
        fieldline_quadrature_weights=lambda geom, grid: (
            jnp.ones((grid.z.size,), dtype=jnp.float32),
            jnp.asarray(1.0),
        ),
        _diagnostic_omega_mode_mask=lambda *a, **k: jnp.ones(mask_shape, dtype=bool),
        _linear_frequency_bound=lambda *a, **k: np.zeros(3, dtype=float),
        _laguerre_velocity_max=lambda nl, **_: 0.0,
        nonlinear_rhs_cached=rhs,
        compute_fields_cached=fields_fn,
    )


def _null_cache() -> SimpleNamespace:
    names = "Jl JlB sqrt_m sqrt_m_p1 kx_grid ky_grid dealias_mask laguerre_to_grid"
    names += " laguerre_to_spectral laguerre_roots laguerre_j0 laguerre_j1_over_alpha b"
    return SimpleNamespace(kxfac=1.0, **dict.fromkeys(names.split()))


def _identity_operator(shape) -> SimpleNamespace:
    return SimpleNamespace(
        shape=shape,
        dt_val=jnp.asarray(0.1, dtype=jnp.float32),
        precond_op=lambda x: x,
        matvec=lambda x: x,
        squeeze_species=False,
        state_dtype=jnp.complex64,
    )


def _zero_phi_fields(nz: int = 2) -> FieldState:
    phi = jnp.zeros((1, 1, nz), dtype=jnp.complex64)
    return FieldState(phi=phi, apar=None, bpar=None)


def test_explicit_diagnostics_impl_applies_fixed_mode_collision_and_stride(
    monkeypatch,
) -> None:
    grid, cache = _diag_grid_cache(
        [0.0, 0.2], [0.0], lb_lam=jnp.ones((1, 1, 1, 2, 1, 2), dtype=jnp.float32)
    )
    params = SimpleNamespace(tz=jnp.asarray([1.0]), vth=jnp.asarray([1.0]), nu=0.2)
    fields = FieldState(
        phi=jnp.ones((2, 1, 2), dtype=jnp.complex64), apar=None, bpar=None
    )
    five = ((1,), (1, 1), (1, 1), (1, 1), (1, 1))

    def _fake_collision_split(G_state, damping, dt_local, scheme):
        assert scheme == "exp"
        return G_state + 5.0

    patch_attrs(
        monkeypatch,
        _DI,
        **_diag_stubs(
            (2, 1),
            rhs=lambda G, *a, **k: (jnp.ones_like(G), fields),
            fields_fn=lambda *a, **k: fields,
        ),
        _instantaneous_growth_rate_step=lambda *a, **k: (
            jnp.ones((2, 1), dtype=jnp.float32) * 2.0,
            jnp.ones((2, 1), dtype=jnp.float32) * -3.0,
        ),
        phi2_resolved=lambda *a, **k: (jnp.asarray(1.0), *_ones_tuple(*[(1,)] * 7)),
        distribution_free_energy_resolved=lambda *a, **k: _ones_tuple(*five, (1, 1)),
        electrostatic_field_energy_resolved=lambda *a, **k: _ones_tuple(*five),
        magnetic_vector_potential_energy_resolved=lambda *a, **k: _ones_tuple(*five),
        heat_flux_channel_resolved_species=lambda *a, **k: (_ones_tuple(*five),) * 3,
        particle_flux_channel_resolved_species=lambda *a, **k: (
            (_ones_tuple(*five),) * 3
        ),
        turbulent_heating_resolved_species=lambda *a, **k: _ones_tuple(*five),
        _collision_damping=lambda *a, **k: jnp.ones((1, 1, 2, 1, 2), jnp.float32),
        _apply_collision_split=_fake_collision_split,
    )

    G0 = jnp.zeros((1, 1, 2, 1, 2), dtype=jnp.complex64)
    G0 = G0.at[..., 1:2, 0:1, :].set(7.0 + 0.0j)
    run = partial(
        _integrate_nonlinear_explicit_diagnostics_impl,
        G0,
        grid,
        SimpleNamespace(),
        params,
        dt=0.1,
        cache=cache,
    )

    for method in ("rk3", "rk4"):
        t_branch, diag_branch, G_branch, _fields_branch = run(
            steps=1,
            method=method,
            terms=TermConfig(),
            sample_stride=1,
            diagnostics_stride=1,
            omega_ky_index=1,
            omega_kx_index=0,
        )
        np.testing.assert_allclose(np.asarray(t_branch), [0.1])
        np.testing.assert_allclose(np.asarray(diag_branch.gamma_t), [2.0])
        assert G_branch.shape == G0.shape

    with pytest.raises(ValueError):
        run(steps=1, method="not-a-method")

    t, diag, G_final, fields_final = run(
        steps=3,
        method="rk2",
        terms=TermConfig(collisions=1.0, hypercollisions=0.0),
        sample_stride=1,
        diagnostics_stride=2,
        collision_split=True,
        collision_scheme="exp",
        fixed_mode_ky_index=1,
        fixed_mode_kx_index=0,
    )

    np.testing.assert_allclose(np.asarray(t), [0.1, 0.3])
    np.testing.assert_allclose(np.asarray(diag.gamma_t), [2.0, 2.0])
    np.testing.assert_allclose(np.asarray(G_final[..., 1:2, 0:1, :]), 7.0)
    assert np.all(np.asarray(G_final[..., 0:1, 0:1, :]) > 0.0)
    assert diag.resolved is not None
    for field_info in dataclass_fields(ResolvedDiagnostics):
        resolved_value = getattr(diag.resolved, field_info.name)
        assert resolved_value is not None
        assert np.asarray(resolved_value).shape[0] == np.asarray(t).shape[0]
    assert fields_final.phi.shape == (2, 1, 2)


def test_integrate_nonlinear_imex_diagnostics_rejects_bad_shape(monkeypatch) -> None:
    monkeypatch.setattr(f"{_DI}.ensure_flux_tube_geometry_data", lambda geom, z: geom)
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_diagnostics(
            jnp.zeros((2, 2), dtype=jnp.complex64),
            SimpleNamespace(z=np.array([0.0])),
            object(),
            object(),
            dt=0.1,
            steps=2,
        )


def test_integrate_nonlinear_imex_cached_shape_mismatch_and_zero_nonlinear(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    with pytest.raises(ValueError):
        integrate_nonlinear_imex_cached(
            G0,
            SimpleNamespace(),
            SimpleNamespace(),
            dt=0.1,
            steps=2,
            implicit_operator=_identity_operator((1, 2, 2, 1, 1, 2)),
        )

    gmres_calls: list[int] = []
    monkeypatch.setattr(
        "gkx.solvers.nonlinear.imex.gmres",
        lambda matvec, rhs, **kwargs: SimpleNamespace(
            x=gmres_calls.append(rhs.size) or rhs,
            converged=True,
            residual_norm=jnp.zeros((), dtype=jnp.float32),
            iterations=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    patch_attrs(
        monkeypatch,
        _SI,
        assemble_rhs_cached_jit=lambda G, *a, **k: (
            jnp.zeros_like(G),
            _zero_phi_fields(),
        ),
        compute_fields_cached=_raise("nonlinear path should stay off"),
    )

    G_out, fields_t = integrate_nonlinear_imex_cached(
        G0,
        _null_cache(),
        _SPECIES_PARAMS,
        dt=0.1,
        steps=2,
        terms=TermConfig(nonlinear=0.0),
        implicit_operator=_identity_operator(G0.shape),
    )

    assert gmres_calls
    assert G_out.shape == G0.shape
    assert fields_t.phi.shape[0] == 2


# ---- from test_adaptive_chunk_memory.py ----
# The adaptive chunk loop must not accumulate what it is about to discard.


CHUNK_SAMPLES = 7


CHUNK_COUNT = 5


DT_CHUNK_MEMORY = 0.5


T_MAX = (CHUNK_SAMPLES - 1) * DT_CHUNK_MEMORY * CHUNK_COUNT


def _chunk(index: int) -> SimulationDiagnostics:
    """A chunk whose payload carries its own global sample id."""

    sample_id = np.arange(CHUNK_SAMPLES, dtype=float) + index * CHUNK_SAMPLES
    zeros = np.zeros(CHUNK_SAMPLES)
    return SimulationDiagnostics(
        t=np.arange(CHUNK_SAMPLES, dtype=float) * DT_CHUNK_MEMORY,
        dt_t=np.full(CHUNK_SAMPLES, DT_CHUNK_MEMORY),
        dt_mean=np.asarray(DT_CHUNK_MEMORY),
        gamma_t=sample_id,
        omega_t=zeros,
        Wg_t=zeros,
        Wphi_t=zeros,
        Wapar_t=zeros,
        heat_flux_t=sample_id,
        particle_flux_t=zeros,
        energy_t=zeros,
        resolved=ResolvedDiagnostics(Phi2_kxt=sample_id.reshape(-1, 1)),
    )


def _run(*, stride: int, spill_dir: Path | None = None):
    remaining = iter(range(CHUNK_COUNT))

    def integrate_chunk(_show_progress, _remaining_time):
        return None, _chunk(next(remaining)), object(), object()

    return run_adaptive_runtime_chunk_loop(
        integrate_chunk=integrate_chunk,
        t_max=T_MAX,
        chunk_steps=CHUNK_SAMPLES,
        label="memory-test",
        diagnostics_stride=stride,
        spill_dir=spill_dir,
    )


def test_disk_spill_is_a_storage_choice_not_a_different_answer(tmp_path) -> None:
    in_ram = _run(stride=3).diagnostics
    on_disk = _run(stride=3, spill_dir=tmp_path / "spill").diagnostics

    assert np.array_equal(np.asarray(in_ram.gamma_t), np.asarray(on_disk.gamma_t)), (
        "spilling chunks to disk changed the retained time series"
    )
    assert in_ram.resolved is not None and on_disk.resolved is not None
    assert np.array_equal(
        np.asarray(in_ram.resolved.Phi2_kxt), np.asarray(on_disk.resolved.Phi2_kxt)
    ), "the resolved payload did not survive the spill round trip"


# ---- from test_nonlinear_replicate_diagnostics.py ----


def _replicate_rows(*rows, **extra) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "late_mean": mean,
            "source_artifact": f"case_{label}_heat_flux_trace.csv",
            **extra,
        }
        for index, (mean, label) in enumerate(rows)
    ]


_READY = dict(passed=True, promotion_ready=True)


def test_replicate_spread_report_identifies_mixed_seed_timestep_spread() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
                "passed": False,
                "statistics": {
                    "ensemble_mean": 10.0,
                    "mean_rel_spread": 0.30,
                    "combined_sem_rel": 0.04,
                },
                "config": {"max_mean_rel_spread": 0.15},
                "rows": _replicate_rows(
                    (10.0, "seed31"),
                    (11.5, "seed32"),
                    (8.5, "dt0p04"),
                    sem=0.2,
                    **_READY,
                ),
            }
        ]
    )

    assert report["passed"] is False
    assert report["summary"]["failed_states"] == ["plus_delta"]
    row = report["state_rows"][0]
    assert row["classification"] == "mixed_seed_timestep_spread"
    assert row["high_variant_axis"] == "seed"
    assert row["low_variant_axis"] == "timestep"
    assert "Do not add same-bracket replicas blindly" in row["recommendation"]


def test_replicate_spread_report_passes_with_small_seed_spread() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_baseline_t900_ensemble",
                "passed": True,
                "statistics": {
                    "ensemble_mean": 10.0,
                    "mean_rel_spread": 0.02,
                    "combined_sem_rel": 0.04,
                },
                "rows": _replicate_rows((9.9, "seed31"), (10.1, "seed32"), **_READY),
            }
        ]
    )

    assert report["passed"] is True
    assert report["state_rows"][0]["classification"] == "passed_replicate_spread_gate"


def test_replicate_spread_report_preserves_joint_seed_timestep_labels() -> None:
    report = nonlinear_replicate_spread_report(
        [
            {
                "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
                "passed": False,
                "statistics": {"ensemble_mean": 10.0, "mean_rel_spread": 0.30},
                "rows": _replicate_rows(
                    (11.5, "seed32_dt0p04"), (8.5, "seed22_dt0p05")
                ),
            }
        ]
    )

    assert report["replicate_rows"][0]["variant_label"] == "seed32_dt0p04"
    assert report["replicate_rows"][0]["variant_axis"] == "seed_timestep"


# ---- from test_nonlinear_window_gradient_matrix.py ----
# AD-vs-finite-difference coverage matrix for the production heat-flux window.


NX, NY, NZ = 16, 16, 8


NL, NM = 2, 4


DT_WINDOW_GRADIENT = 0.004


STEPS = 6


TAIL = 4


ES_TERMS = TermConfig(nonlinear=1.0, apar=0.0, bpar=0.0)


EM_TERMS = TermConfig(nonlinear=1.0, apar=1.0, bpar=1.0)


IONS = Species(
    charge=1.0, mass=1.0, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
)


ELECTRONS = Species(
    charge=-1.0, mass=2.7e-4, density=1.0, temperature=1.0, tprim=2.49, fprim=0.8
)


SECOND_ION = Species(
    charge=1.0, mass=2.0, density=0.2, temperature=1.0, tprim=1.8, fprim=0.5
)


# The steps below were chosen for float64, where they leave the centered
# difference truncation-limited. Under float32 the same difference is
# roundoff-limited, with relative error about eps32 |Q| / (h |dQ/dp|): 3e-3 to
# 1.2 at these steps, because the window is only weakly sensitive to its design
# parameters (|p dQ/dp| / |Q| is 2.7e-4 for the hypercollision rate). A float32
# run therefore keeps its float32 adjoint and checks it against the same centered
# difference evaluated in float64.
X64_WINDOW_RTOL = 1.0e-6


# Float32 roundoff of this six-step window relative to the same window in
# float64, for the value and the reverse-mode gradient alike: at most 5.7e-6
# (48 float32 ulps) across the matrix on XLA:CPU. The bound carries a 4x margin
# for another CPU's fusion choices. Measured in plan/log.md, Q14.
F32_WINDOW_ROUNDOFF = 2.5e-5


def _float64_active() -> bool:
    return bool(jax.config.read("jax_enable_x64"))


def _build_case_grid():
    cfg = CycloneBaseCase(grid=GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=6.0, Ly=6.0))
    grid = build_spectral_grid(cfg.grid)
    geom = ensure_flux_tube_geometry_data(
        SAlphaGeometry.from_config(cfg.geometry), grid.z
    )
    return grid, geom


@pytest.fixture(scope="module")
def case_grid():
    return _build_case_grid()


def _seed(grid, n_species: int, key: int) -> jnp.ndarray:
    """Deterministic low-amplitude state standing in for a saturated one."""

    rng = np.random.default_rng(key)
    shape = (n_species, NL, NM, grid.ky.size, grid.kx.size, grid.z.size)
    draw = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return jnp.asarray(1.0e-4 * draw, dtype=jnp.complex128)


class HermiteLaguerreDrag:
    """Custom collision model whose rate is a traced design parameter.

    ``integrate_nonlinear`` has always accepted ``collision_operator``; the
    differentiable window did not, so a run saturated with a custom operator and
    then differentiated returned the derivative of a *different* model. This
    operator exists to test that the rate reaches the gradient at all.
    """

    def __init__(self, nu):
        self.nu = nu

    def apply(self, context):
        state = context.distribution
        offset = state.ndim - 5
        n_laguerre, n_hermite = state.shape[offset : offset + 2]
        ell = jnp.arange(n_laguerre).reshape((n_laguerre, 1, 1, 1, 1))
        hermite = jnp.arange(n_hermite).reshape((1, n_hermite, 1, 1, 1))
        rate = jnp.asarray(self.nu) * (hermite + 2.0 * ell)
        return -rate * state


def _window(grid, geom, state, params, *, terms, method, collision, checkpoint):
    return nonlinear_heat_flux_window(
        state,
        grid,
        geom,
        params,
        dt=DT_WINDOW_GRADIENT,
        steps=STEPS,
        method=method,
        tail_steps=TAIL,
        terms=terms,
        checkpoint=checkpoint,
        compressed_real_fft=True,
        collision_operator=collision,
    )


def _drive_case(grid, geom, params, terms, method, state, collision=None):
    """Differentiate a uniform multiplier on the per-species ``a/L_T`` drive."""

    def evaluate(scale, checkpoint=True):
        scaled = replace(params, tprim=jnp.asarray(params.tprim) * scale)
        return _window(
            grid,
            geom,
            state,
            scaled,
            terms=terms,
            method=method,
            collision=collision,
            checkpoint=checkpoint,
        )

    return evaluate, jnp.asarray(1.0), 1.0e-4


def _rk2_window_by(grid, geom, state, make_params, terms, collision=None):
    """The rk2 window as a function of one scalar design parameter."""

    def evaluate(value, checkpoint=True):
        return _window(
            grid,
            geom,
            state,
            make_params(value),
            terms=terms,
            method="rk2",
            collision=None if collision is None else collision(value),
            checkpoint=checkpoint,
        )

    return evaluate


def _build_case(name, grid, geom):
    if name in ("baseline_es_rk2", "baseline_es_rk3", "baseline_es_rk4"):
        method = name.rsplit("_", 1)[1]
        return _drive_case(
            grid, geom, LinearParams(), ES_TERMS, method, _seed(grid, 1, 17)
        )

    if name == "multispecies_kinetic_electrons":
        params = build_linear_params([IONS, ELECTRONS], tau_e=1.0)
        return _drive_case(grid, geom, params, ES_TERMS, "rk2", _seed(grid, 2, 19))

    if name == "multispecies_two_ions":
        params = build_linear_params([IONS, SECOND_ION], tau_e=1.0)
        return _drive_case(grid, geom, params, ES_TERMS, "rk2", _seed(grid, 2, 19))

    if name == "electromagnetic_d_beta":
        base = LinearParams(beta=0.02, fapar=1.0)
        by_beta = _rk2_window_by(
            grid, geom, _seed(grid, 1, 23), lambda b: replace(base, beta=b), EM_TERMS
        )
        return by_beta, jnp.asarray(0.02), 1.0e-6

    if name == "electromagnetic_d_drive":
        return _drive_case(
            grid,
            geom,
            LinearParams(beta=0.02, fapar=1.0),
            EM_TERMS,
            "rk2",
            _seed(grid, 1, 23),
        )

    if name == "custom_collisions_d_nu":
        terms = TermConfig(nonlinear=1.0, collisions=1.0, apar=0.0, bpar=0.0)
        by_nu = _rk2_window_by(
            grid,
            geom,
            _seed(grid, 1, 29),
            lambda _nu: LinearParams(),
            terms,
            collision=HermiteLaguerreDrag,
        )
        return by_nu, jnp.asarray(0.3), 1.0e-5

    if name == "hypercollisions_d_nu_hyper_m":
        base = LinearParams(
            nu_hyper_m=0.5, nu_hyper_l=0.5, p_hyper_m=6.0, p_hyper_l=6.0
        )
        terms = TermConfig(nonlinear=1.0, hypercollisions=1.0, apar=0.0, bpar=0.0)
        by_hyper = _rk2_window_by(
            grid, geom, _seed(grid, 1, 31), lambda v: replace(base, nu_hyper_m=v), terms
        )
        # The hypercollisional sensitivity is ~3 decades below the window value,
        # so a 1e-5 step differences two nearly equal numbers; 1e-3 is where the
        # centered difference is truncation- rather than roundoff-limited.
        return by_hyper, jnp.asarray(0.5), 1.0e-3

    if name == "combined_ms_em_coll_hyper_rk3":
        params = build_linear_params(
            [IONS, ELECTRONS],
            tau_e=1.0,
            beta=0.02,
            fapar=1.0,
            nu_hyper_m=0.5,
            nu_hyper_l=0.5,
            p_hyper_m=6.0,
            p_hyper_l=6.0,
        )
        terms = TermConfig(
            nonlinear=1.0, apar=1.0, bpar=1.0, collisions=1.0, hypercollisions=1.0
        )
        return _drive_case(
            grid,
            geom,
            params,
            terms,
            "rk3",
            _seed(grid, 2, 37),
            collision=HermiteLaguerreDrag(0.2),
        )

    raise AssertionError(f"unknown gradient-matrix case {name}")


COVERAGE_MATRIX = (
    "baseline_es_rk2",
    "baseline_es_rk3",
    "baseline_es_rk4",
    "multispecies_kinetic_electrons",
    "multispecies_two_ions",
    "electromagnetic_d_beta",
    "electromagnetic_d_drive",
    "custom_collisions_d_nu",
    "hypercollisions_d_nu_hyper_m",
    "combined_ms_em_coll_hyper_rk3",
)


def _float64_centered_difference(case: str) -> float:
    """The case's centered difference, evaluated in float64 in this process.

    Grid, seed and parameters are rebuilt inside the context, so nothing
    captured at float32 enters the reference.
    """

    with jax.enable_x64(True):
        grid, geom = _build_case_grid()
        evaluate, point, step = _build_case(case, grid, geom)
        upper, lower = evaluate(point + step), evaluate(point - step)
        assert upper.dtype == jnp.float64 and lower.dtype == jnp.float64
        return float((upper - lower) / (2.0 * step))


@pytest.mark.parametrize("case", COVERAGE_MATRIX)
def test_window_gradient_matches_centered_finite_difference(case, case_grid):
    """Every production switch must reach the adjoint of the physical flux."""

    grid, geom = case_grid
    evaluate, point, step = _build_case(case, grid, geom)
    value, gradient = jax.value_and_grad(evaluate)(point)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(gradient))
    assert float(gradient) != 0.0, f"{case} left the design parameter disconnected"

    if _float64_active():
        centered = (evaluate(point + step) - evaluate(point - step)) / (2.0 * step)
        rtol = X64_WINDOW_RTOL
    else:
        # The float64 assertion holds the reference within X64_WINDOW_RTOL; the
        # float32 adjoint adds its own roundoff on top.
        centered = _float64_centered_difference(case)
        rtol = X64_WINDOW_RTOL + F32_WINDOW_ROUNDOFF
    np.testing.assert_allclose(
        np.asarray(gradient), np.asarray(centered), rtol=rtol, atol=0.0
    )


def test_block_checkpointed_window_matches_plain_reverse_pass(case_grid):
    """Block checkpointing must not change the value or the gradient.

    Run on the 6-D kinetic-electron state, where the retained block boundaries
    carry a species axis the existing single-species parity test never saw.
    """

    grid, geom = case_grid
    evaluate, point, _step = _build_case("multispecies_kinetic_electrons", grid, geom)
    blocked = jax.value_and_grad(evaluate)(point)
    plain = jax.value_and_grad(lambda value: evaluate(value, False))(point)
    # Blocked and plain are two compilations of the same arithmetic. Each lies
    # within its precision's roundoff of the exact window, so under float32 they
    # lie within twice F32_WINDOW_ROUNDOFF of each other. Float64 keeps 1e-12.
    rtol = 1.0e-12 if _float64_active() else 2.0 * F32_WINDOW_ROUNDOFF
    np.testing.assert_allclose(
        np.asarray(blocked), np.asarray(plain), rtol=rtol, atol=0.0
    )


def test_custom_collision_operator_changes_the_differentiated_window(case_grid):
    """A window that ignored ``collision_operator`` would differentiate other physics."""

    grid, geom = case_grid
    evaluate, point, _step = _build_case("custom_collisions_d_nu", grid, geom)
    with_operator = float(evaluate(point))
    without_operator = float(
        _window(
            grid,
            geom,
            _seed(grid, 1, 29),
            LinearParams(),
            terms=TermConfig(nonlinear=1.0, collisions=1.0, apar=0.0, bpar=0.0),
            method="rk2",
            collision=None,
            checkpoint=True,
        )
    )
    assert with_operator != without_operator


def test_electromagnetic_window_actually_solves_apar_and_bpar(case_grid):
    """Guard the EM rows: a silently electrostatic field solve would still pass FD."""

    from gkx.operators.linear.cache_builder import build_linear_cache
    from gkx.solvers.nonlinear.diagnostics import nonlinear_rhs_cached

    grid, geom = case_grid
    params = LinearParams(beta=0.02, fapar=1.0)
    state = _seed(grid, 1, 23)
    cache = build_linear_cache(grid, geom, params, Nl=NL, Nm=NM)
    _derivative, fields = nonlinear_rhs_cached(
        state, cache, params, EM_TERMS, differentiable=True
    )
    assert fields.apar is not None and float(jnp.max(jnp.abs(fields.apar))) > 0.0
    assert fields.bpar is not None and float(jnp.max(jnp.abs(fields.bpar))) > 0.0


def test_window_beyond_the_divergence_knee_warns(case_grid):
    """A window past the measured knee must say so before it burns the compute.

    The knee is lowered to one step here so the guard is tested for the price of
    a two-step window rather than the 1025 the shipped default would cost.
    """

    with pytest.warns(RuntimeWarning, match="divergence knee"):
        _short_window(case_grid, divergence_knee_steps=1)


def _short_window(case_grid, **kwargs):
    grid, geom = case_grid
    return nonlinear_heat_flux_window(
        _seed(grid, 1, 17),
        grid,
        geom,
        LinearParams(),
        dt=DT_WINDOW_GRADIENT,
        steps=2,
        terms=ES_TERMS,
        **kwargs,
    )


def test_window_at_or_below_the_knee_is_silent(case_grid, recwarn):
    """QA_optimization runs at exactly the knee; that must not warn."""

    _short_window(case_grid, divergence_knee_steps=2)
    _short_window(case_grid)
    assert not [w for w in recwarn if "divergence knee" in str(w.message)]


def test_shipped_optimization_example_stays_at_or_below_the_knee():
    """The QA example's window is the reason the guard exists; pin it."""

    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[3] / "examples" / "10_vmex_optimization/run.py"
    ).read_text()
    match = re.search(r"WINDOW_STEPS\s*=\s*([\d_]+)", source)
    assert match is not None
    assert int(match.group(1).replace("_", "")) <= DIVERGENCE_KNEE_STEPS


# --- Q29: the sheared IMEX route carries the same solve status ----------------


def _sheared_imex_deck(ky_layout):
    grid = spectral_grid(4, 4, 4, ky_layout=ky_layout)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=1, Nm=2)
    state = jnp.zeros(
        (1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    state = state.at[0, 0, 0, 1, 0, :].set(0.2 + 0.1j)
    return grid, geom, params, cache, state


_Q29_SHEARED = dict(dt=0.02, steps=3, shear_rate=0.5, terms=TermConfig(nonlinear=1.0))
_Q29_STARVED_BUDGET = dict(implicit_tol=1.0e-14, implicit_maxiter=1, implicit_restart=1)
_Q29_GENEROUS_BUDGET = dict(
    implicit_tol=max(1.0e-8, 10 * float(jnp.finfo(jnp.asarray(0.0).dtype).eps)),
    implicit_maxiter=200,
    implicit_restart=20,
)


def _q29_runner(ky_layout, route, method="imex"):
    grid, geom, params, cache, state = _sheared_imex_deck(ky_layout)
    return partial(
        route,
        state,
        grid,
        geom,
        params,
        method=method,
        cache=cache,
        compressed_real_fft=ky_layout == "half",
        **_Q29_SHEARED,
    )


@pytest.mark.parametrize("ky_layout", ["full", "half"])
def test_sheared_imex_carries_unconverged_solves_to_the_host_gate(ky_layout) -> None:
    """A starved inner budget is visible on the sheared IMEX route and refused."""

    from gkx.solvers.linear.implicit import require_converged_implicit_solves

    run_route = _q29_runner(ky_layout, integrate_nonlinear_sheared)

    def run(budget):
        return run_route(return_solve_stats=True, **budget)[2]

    starved = run(_Q29_STARVED_BUDGET)
    assert int(starved.solves) == 3
    assert int(starved.unconverged_solves) == 3
    assert float(starved.max_relative_residual) > 1.0e-14
    with pytest.raises(RuntimeError, match="3 of 3 implicit GMRES solves did not"):
        require_converged_implicit_solves(starved, label="sheared probe")

    summary = require_converged_implicit_solves(
        run(_Q29_GENEROUS_BUDGET), label="sheared probe"
    )
    assert summary.converged and summary.solves == 3
    assert summary.unconverged_solves == 0


@pytest.mark.parametrize("ky_layout", ["full", "half"])
def test_sheared_transport_trace_carries_the_same_status(ky_layout) -> None:
    """The transport door reports through the trace it already returns."""

    from gkx.solvers.linear.implicit import require_converged_implicit_solves

    run = _q29_runner(ky_layout, integrate_nonlinear_sheared_transport)
    trace = run(return_solve_stats=True, **_Q29_STARVED_BUDGET)
    assert trace.solve_stats is not None
    with pytest.raises(RuntimeError, match="3 of 3 implicit GMRES solves did not"):
        require_converged_implicit_solves(trace.solve_stats, label="sheared transport")

    unasked = run(**_Q29_GENEROUS_BUDGET)
    assert unasked.solve_stats is None
    asked = run(return_solve_stats=True, **_Q29_GENEROUS_BUDGET)
    np.testing.assert_array_equal(
        np.asarray(unasked.final_state), np.asarray(asked.final_state)
    )


@contextmanager
def _counting_backend_compiles(name: str = "") -> Iterator[dict[str, int]]:
    """Count compiles of modules named like ``name`` (skip if JAX moves the hook)."""

    compiler = pytest.importorskip("jax._src.compiler")
    original = getattr(compiler, "backend_compile_and_load", None)
    if original is None:
        pytest.skip("jax._src.compiler.backend_compile_and_load is not available")
    counter = {"compiles": 0}

    def counting(backend, module, *args, **kwargs):
        counter["compiles"] += name in str(module.operation.attributes["sym_name"])
        return original(backend, module, *args, **kwargs)

    compiler.backend_compile_and_load = counting
    try:
        yield counter
    finally:
        compiler.backend_compile_and_load = original


def _window_value_and_grad(grid, geom, state, params):
    """Differentiate the window through the drive, the way callers do."""

    def objective(tprim):
        return nonlinear_heat_flux_window(
            state,
            grid,
            geom,
            replace(params, tprim=tprim),
            dt=DT_WINDOW_GRADIENT,
            steps=STEPS,
            terms=ES_TERMS,
            method="rk2",
        )

    return jax.value_and_grad(objective)


def _moved_geometry(grid, *, epsilon: float, alpha: float):
    """Change differentiable shape while preserving linked-chain topology."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=NX, Ny=NY, Nz=NZ, Lx=6.0, Ly=6.0))
    shape = replace(
        SAlphaGeometry.from_config(cfg.geometry), epsilon=epsilon, alpha=alpha
    )
    return ensure_flux_tube_geometry_data(shape, grid.z)


def test_window_adjoint_compiles_one_graph_and_reuses_it(case_grid):
    """Repeated objective evaluations reuse the compiled adjoint window."""

    grid, geom = case_grid
    state = _seed(grid, 1, 17)
    params = LinearParams()
    value_and_grad = _window_value_and_grad(grid, geom, state, params)
    drive = jnp.asarray(params.tprim)
    jax.block_until_ready(value_and_grad(drive))
    with _counting_backend_compiles("heat_flux_window") as counter:
        jax.block_until_ready(value_and_grad(drive))
        jax.block_until_ready(value_and_grad(drive * 1.01))
    assert counter["compiles"] == 0


def test_window_adjoint_takes_the_saturated_state_as_an_operand(case_grid):
    """A new saturated state reuses the compiled window: no state constant."""

    grid, geom = case_grid
    params = LinearParams()
    drive = jnp.asarray(params.tprim)
    jax.block_until_ready(
        _window_value_and_grad(grid, geom, _seed(grid, 1, 17), params)(drive)
    )
    with _counting_backend_compiles() as counter:
        jax.block_until_ready(
            _window_value_and_grad(grid, geom, _seed(grid, 1, 23), params)(drive)
        )
    assert counter["compiles"] == 0


def test_window_adjoint_reuses_its_graph_for_a_new_geometry(case_grid):
    """Differentiable geometry changes do not trigger recompilation."""

    grid, geom = case_grid
    state = _seed(grid, 1, 17)
    params = LinearParams()
    jax.block_until_ready(
        _window_value_and_grad(grid, geom, state, params)(jnp.asarray(params.tprim))
    )
    moved = _moved_geometry(grid, epsilon=0.19, alpha=0.1)
    with _counting_backend_compiles() as counter:
        jax.block_until_ready(
            _window_value_and_grad(grid, moved, state, params)(
                jnp.asarray(params.tprim)
            )
        )
    assert counter["compiles"] == 0


def test_window_adjoint_shared_graph_is_not_stale_for_a_new_geometry(case_grid):
    """A reused graph retains no stale geometry in values or gradients."""

    grid, geom = case_grid
    state = _seed(grid, 1, 17)
    params = LinearParams()
    drive = jnp.asarray(params.tprim)
    moved = _moved_geometry(grid, epsilon=0.19, alpha=0.1)

    jax.clear_caches()
    cold = jax.block_until_ready(
        _window_value_and_grad(grid, moved, state, params)(drive)
    )
    jax.clear_caches()
    jax.block_until_ready(_window_value_and_grad(grid, geom, state, params)(drive))
    warm = jax.block_until_ready(
        _window_value_and_grad(grid, moved, state, params)(drive)
    )

    assert np.asarray(cold[0]).tobytes() == np.asarray(warm[0]).tobytes()
    assert np.asarray(cold[1]).tobytes() == np.asarray(warm[1]).tobytes()


# --- kinetic-electron step: measured streaming bound, per-chain IMEX operator

KE_NL, KE_NM = 2, 3


def _tiny_kinetic_electron_box(
    layout: str = "half",
    nu: float = 0.0,
    nl: int = KE_NL,
    nm: int = KE_NM,
    ion_counts=None,
):
    cfg = uniform_moments(load(REPO_ROOT / "examples/05_kinetic_electrons/case.toml"))
    cfg = replace(
        cfg, grid=replace(cfg.grid, Nx=4, Ny=4, Nz=8, ntheta=8, ky_layout=layout)
    )
    if nu:  # ion rate nu, electron rate nu / sqrt(m_e / m_i)
        ion, electron = cfg.species
        cfg = replace(
            cfg,
            species=(
                replace(ion, nu=nu),
                replace(electron, nu=nu / electron.mass**0.5),
            ),
        )
    if ion_counts:  # the ion keeps fewer moments than the state
        ions, electron = cfg.species
        counts = dict(zip(("Nl", "Nm"), ion_counts))
        cfg = replace(cfg, species=(replace(ions, **counts), electron))
    geom = rt.build_runtime_geometry(cfg)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, cfg.grid))
    params = rt.build_runtime_linear_params(cfg, Nm=nm, geom=geom)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    shape = (2, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    return cfg, params, cache, shape


def _ke_dense(op, shape, dtype):
    n = int(np.prod(shape))
    eye = jnp.eye(n, dtype=dtype).reshape((n, *shape))
    return np.asarray(jax.lax.map(op, eye)).reshape(n, n).T


def test_measured_streaming_frequency_is_the_operator_spectral_radius():
    _, params, cache, shape = _tiny_kinetic_electron_box()
    terms = TermConfig(
        **dict.fromkeys(
            "mirror curvature gradb diamagnetic collisions hypercollisions".split(), 0.0
        ),
        end_damping=0.0,
    )
    dtype = jnp.complex64

    def op(G):
        return assemble_rhs_cached(G, cache, params, terms=terms)[0]

    radius = np.max(np.abs(np.linalg.eigvals(_ke_dense(op, shape, dtype))))
    measured = measured_streaming_frequency(params, cache, nl=KE_NL, nm=KE_NM)
    assert measured > 0.0
    np.testing.assert_allclose(measured, radius, rtol=1e-3)


def test_measured_streaming_frequency_skips_adiabatic_like_decks():
    _, params, cache, _ = _tiny_kinetic_electron_box()
    heavy = replace(params, vth=jnp.ones_like(jnp.asarray(params.vth)))
    assert measured_streaming_frequency(heavy, cache, nl=KE_NL, nm=KE_NM) == 0.0


def test_measured_streaming_frequency_includes_the_amplified_end_absorber():
    """The absorber damps h, so the field solve lifts its radius far above ``nu_end``."""
    _, params, cache, shape = _tiny_kinetic_electron_box()
    params = replace(params, damp_ends_rate=100.0)
    terms = TermConfig(
        **dict.fromkeys(
            "mirror curvature gradb diamagnetic collisions hypercollisions".split(), 0.0
        )
    )

    def op(G):
        return assemble_rhs_cached(G, cache, params, terms=terms)[0]

    radius = np.max(np.abs(np.linalg.eigvals(_ke_dense(op, shape, jnp.complex64))))
    measured = measured_streaming_frequency(
        params, cache, nl=KE_NL, nm=KE_NM, absorber_vs=0.0
    )
    assert measured > 100.0
    np.testing.assert_allclose(measured, radius, rtol=1e-3)
    # a periodic box has no absorber and keeps the streaming-only radius
    quiet = replace(cache, damp_profile=jnp.zeros_like(cache.damp_profile))
    assert (
        measured_streaming_frequency(params, quiet, nl=KE_NL, nm=KE_NM, absorber_vs=0.0)
        < measured
    )


def test_measured_streaming_frequency_skips_the_absorber_below_the_gain_bound():
    _, params, cache, _ = _tiny_kinetic_electron_box()
    params = replace(params, damp_ends_rate=100.0)
    base = measured_streaming_frequency(params, cache, nl=KE_NL, nm=KE_NM)
    # rate * gain (5000, light species) stays below the analytic bound: no absorber
    skipped = measured_streaming_frequency(
        params, cache, nl=KE_NL, nm=KE_NM, absorber_vs=1.0e12
    )
    assert skipped == base
    with_absorber = measured_streaming_frequency(
        params, cache, nl=KE_NL, nm=KE_NM, absorber_vs=0.0
    )
    assert with_absorber > base


def test_measured_streaming_frequency_zero_damp_profile_keeps_streaming_only():
    _, params, cache, _ = _tiny_kinetic_electron_box()
    quiet = replace(cache, damp_profile=jnp.zeros_like(cache.damp_profile))
    base = measured_streaming_frequency(params, quiet, nl=KE_NL, nm=KE_NM)
    assert base > 0.0  # light species: streaming-only measure still runs
    # rate == 0 must not select the absorber even though 0 >= absorber_vs holds
    assert (
        measured_streaming_frequency(params, quiet, nl=KE_NL, nm=KE_NM, absorber_vs=0.0)
        == base
    )


@pytest.mark.parametrize("method", ["rk4", "rk4-pi"])
def test_explicit_step_respects_a_large_end_damping_rate(method):
    """A stiff absorber keeps the adaptive explicit run finite and inside RK4's real axis."""
    import gkx

    cfg, *_ = _tiny_kinetic_electron_box()
    time = replace(
        cfg.time, method=method, damp_ends_rate=600.0, t_max=2.0e-4, run_to="t_max"
    )
    out = gkx.solve(cfg.replace(time=time), ky_target=cfg.run.ky, Nl=KE_NL, Nm=KE_NM)
    d = out.diagnostics
    assert np.isfinite(np.asarray(d.heat_flux_species_t)).all()
    radius = float(np.asarray(d.cfl_scales)[2])  # streaming + absorber, measured
    assert radius > 600.0
    cap = 2.785 * (
        2.0 if method.endswith("pi") else 1.0
    )  # rk4-pi may take 2x the CFL step
    assert float(np.max(np.asarray(d.dt_t))) * radius <= cap


def test_chain_solve_inverts_the_stiff_linear_operator():
    cfg, params, cache, shape = _tiny_kinetic_electron_box()
    terms = rt.build_runtime_term_config(cfg)
    split, fields, mirror, _coll = stiff_linear_split(
        cache, params, terms, compute_fields_cached
    )
    dt = 0.05
    op = build_chain_implicit_linear(
        split,
        fields,
        shape,
        dt,
        modes=np.ones(shape[3:5], bool),
        scheme="imex-ars3",
        mirror_rhs=mirror,
    )
    assert op.spec == ((0,), (1,))  # the electron mirror is implicit
    key = jax.random.PRNGKey(3)
    G = (jax.random.normal(key, shape) + 1j * jax.random.normal(key + 1, shape)).astype(
        jnp.complex64
    )
    gamma = ARS_TABLEAUX["imex-ars3"][1][1][1]
    x = op.solve(G - gamma * dt * op.linear(G))
    assert float(jnp.linalg.norm(x - G) / jnp.linalg.norm(G)) < 2e-3
    block = int(np.prod(shape[:3])) * shape[-1]
    dense = sum(np.size(g.ky) * np.shape(g.ky)[1] * block**2 * 8 for g in op.groups)
    assert op.nbytes < dense
    # A new dt refactors from the stored probes, keeping the pytree structure.
    op2 = refactor_chain_implicit_linear(op, 4 * dt)
    assert jax.tree.structure(op2) == jax.tree.structure(op)
    x = op2.solve(G - gamma * 4 * dt * op.linear(G))
    assert float(jnp.linalg.norm(x - G) / jnp.linalg.norm(G)) < 2e-3


def test_chain_factor_follows_each_species_moment_count():
    """Ions keeping fewer moments: the factor and the collision sub-step shrink, same result."""
    cfg, params, cache, shape = _tiny_kinetic_electron_box(
        nu=0.05, nl=3, nm=6, ion_counts=(2, 4)
    )
    terms = rt.build_runtime_term_config(cfg)
    split, fields, mirror, coll = stiff_linear_split(
        cache, params, terms, compute_fields_cached
    )
    moments = species_moments(cache)
    assert moments == ((2, 4), (3, 6))

    def build(counts):
        return build_chain_implicit_linear(
            split,
            fields,
            shape,
            0.05,
            modes=np.ones(shape[3:5], bool),
            scheme="imex-ars3",
            dtype=jnp.complex128,
            mirror_rhs=mirror,
            collision_rhs=coll,
            moments=counts,
        )

    grouped, padded = build(moments), build(None)
    assert grouped.dims == ((2, 4), (3, 6)) and padded.dims == ((3, 6), (3, 6))
    assert grouped.nbytes < 0.9 * padded.nbytes
    key = jax.random.PRNGKey(3)
    G = jax.random.normal(key, shape) + 1j * jax.random.normal(key + 1, shape)
    G = keep_moments(G, params.moments)
    ref = padded.solve(G)
    assert float(jnp.abs(grouped.solve(G) - ref).max() / jnp.abs(ref).max()) < 1e-12
    again = refactor_chain_implicit_linear(grouped, 0.1)
    assert again.dims == grouped.dims
    cols = grouped.collisions, padded.collisions
    assert (cols[0].cmat.shape[0], cols[1].cmat.shape[0]) == (2 * 4 + 3 * 6, 36)
    ya = cols[0].apply(G, cols[0].factor(5.0), 5.0)
    yb = cols[1].apply(G, cols[1].factor(5.0), 5.0)
    assert float(jnp.abs(ya - yb).max() / jnp.abs(yb).max()) < 1e-12


def test_chain_collisions_are_a_conserving_local_implicit_substep():
    cfg, params, cache, shape = _tiny_kinetic_electron_box(nu=0.05)
    terms = rt.build_runtime_term_config(cfg)
    split, fields, mirror, coll = stiff_linear_split(
        cache, params, terms, compute_fields_cached
    )
    op = build_chain_implicit_linear(
        split,
        fields,
        shape,
        0.05,
        modes=np.ones(shape[3:5], bool),
        scheme="imex-ars3",
        dtype=jnp.complex128,
        mirror_rhs=mirror,
        collision_rhs=coll,
    )
    key = jax.random.PRNGKey(3)
    G = jax.random.normal(key, shape) + 1j * jax.random.normal(key + 1, shape)
    # tau = 100 dt: far beyond the explicit bound, 1 / tau still resolves C.
    tau = 5.0
    y = op.collisions.apply(G, op.collisions.factor(tau), tau)
    # It solves (I - tau C_H) y = G with C_H the operator acting on H(G).
    residual = y - tau * coll(y, fields(y)) - G
    assert float(jnp.linalg.norm(residual) / jnp.linalg.norm(G)) < 1e-12
    # Particle number, total momentum and total energy are untouched.
    rates = multispecies_collision_invariant_rates(
        y - G, density=params.density, mass=params.mass, temperature=params.temp
    )
    for rate in rates:
        assert float(jnp.abs(rate).max()) < 1e-9 * float(jnp.abs(G).max())
    assert float(jnp.linalg.norm(y - G)) > 0.1 * float(jnp.linalg.norm(G))
    # A new dt refactors the sub-step too, keeping the pytree structure.
    again = refactor_chain_implicit_linear(op, 4 * 0.05)
    assert jax.tree.structure(again) == jax.tree.structure(op)
    assert not np.allclose(again.cfac[0], op.cfac[0])


def test_collision_substep_is_used_only_when_explicit_collisions_would_bind():
    for nu, binds in ((0.002, False), (0.5, True)):  # nu_e = nu / sqrt(m_e)
        cfg, params, cache, shape = _tiny_kinetic_electron_box(nu=nu)
        terms = rt.build_runtime_term_config(cfg)
        _split, fields, _mirror, coll = stiff_linear_split(
            cache, params, terms, compute_fields_cached
        )
        assert collisions_bind(coll, fields, shape, jnp.complex128, 0.05) is binds


def test_chain_imex_dt_is_not_bound_by_the_electron_collision_rate():
    """nu_e rho(C) dt = 7: past the explicit bound, finite with the sub-step."""
    import gkx

    deck = uniform_moments(
        gkx.load(REPO_ROOT / "examples/05_kinetic_electrons/case.toml")
    )
    ion, electron = deck.species
    case = deck.replace(
        species=(replace(ion, nu=0.6), replace(electron, nu=36.5)),
        grid=replace(deck.grid, Nx=4, Ny=4, Nz=8, ntheta=8),
        time=replace(
            deck.time,
            method="imex-ars3",
            fixed_dt=True,
            dt=0.05,
            t_max=2.0,
            run_to="t_max",
            # The weak layer the deck used to pin: the default rate (598) damps the
            # explicit-collision blow-up this test needs.
            damp_ends_rate=0.1,
        ),
    )

    def run(scheme):
        timed = replace(case.time, collision_scheme=scheme)
        return gkx.solve(
            case.replace(time=timed), ky_target=case.run.ky, Nl=2, Nm=3
        ).diagnostics

    assert np.isfinite(np.asarray(run("implicit").Wg_t)).all()
    with pytest.raises(RuntimeError, match="non-finite"):
        run("explicit")


def test_second_identical_solve_compiles_nothing_and_repeats_bits():
    """A rebuilt run closure reuses the executable of an identical first run."""

    import gkx

    deck = gkx.load(REPO_ROOT / "examples/03_nonlinear_tokamak/case.toml")
    case = deck.replace(
        grid=replace(deck.grid, Nx=8, Ny=8, Nz=8, ntheta=8),
        time=replace(deck.time, method="rk3", fixed_dt=True, t_max=4 * deck.time.dt),
    )

    def run():
        out = gkx.solve(case, ky_target=case.run.ky, Nl=2, Nm=2).diagnostics
        return np.asarray(out.Wg_t).tobytes()

    first = run()
    with _counting_backend_compiles() as counter:
        second = run()
    assert counter["compiles"] == 0 and first == second


def test_chain_factor_is_reused_for_an_identical_deck_and_dt():
    cfg, params, cache, shape = _tiny_kinetic_electron_box()
    terms = rt.build_runtime_term_config(cfg)
    modes = np.ones(shape[3:5], bool)

    def build(dt):
        split, fields, mirror, _coll = stiff_linear_split(
            cache, params, terms, compute_fields_cached
        )
        return build_chain_implicit_linear(
            split, fields, shape, dt, modes=modes, scheme="imex-ars3", mirror_rhs=mirror
        )

    first = build(0.05)
    assert build(0.05) is first
    assert build(0.04) is not first


@pytest.mark.parametrize(
    "horizon, initial_dt, dt_max, chunk_steps",
    [
        (4.0, 0.05, 1.0, None),
        (0.1, 0.05, 1.0, None),
        (0.1, 0.1, 0.05, None),
        (0.1, 0.05, None, None),
        (0.1, 0.01, 0.01, 3),
    ],
)
def test_adaptive_chain_imex_grows_dt_between_chunks(
    horizon, initial_dt, dt_max, chunk_steps, monkeypatch
):
    """fixed_dt = false: the chain IMEX step adapts dt between chunks."""
    import gkx
    import gkx.workflows.nonlinear as workflow

    original_controller = workflow.ImexDtControl
    controllers = []

    def controller(op, *args, **kwargs):
        controllers.append(op)
        assert float(op.dt) <= kwargs["dt_max"]
        return original_controller(op, *args, **kwargs)

    monkeypatch.setattr(workflow, "ImexDtControl", controller)

    deck = uniform_moments(
        gkx.load(REPO_ROOT / "examples/05_kinetic_electrons/case.toml")
    )
    case = deck.replace(
        grid=replace(deck.grid, Nx=4, Ny=4, Nz=8, ntheta=8),
        time=replace(
            deck.time,
            method="imex-ark3",
            fixed_dt=False,
            dt=initial_dt,
            dt_max=dt_max,
            t_max=horizon,
            chunk_steps=chunk_steps,
            diagnostics_interval=0.025 if chunk_steps is not None else None,
            # The weak layer the deck used to pin: on this 4x4x8 grid the default rate (598)
            # keeps the error-controlled step at or below 0.05 and dt never grows.
            damp_ends_rate=0.1,
        ),
    )
    d = gkx.solve(case, ky_target=case.run.ky, Nl=2, Nm=3).diagnostics
    assert controllers  # Require the IMEX controller to execute.
    if chunk_steps is not None:
        np.testing.assert_allclose(d.t, [0.03, 0.05, 0.08, 0.1], atol=1e-12)
    dt = np.asarray(d.dt_t)
    assert np.isfinite(np.asarray(d.Wg_t)).all()
    if dt_max is not None:
        assert np.all(dt <= dt_max * (1.0 + 1e-12))
    assert dt[0] == pytest.approx(min(initial_dt, dt_max or initial_dt))
    assert np.all(np.diff(np.asarray(d.t)) > 0)
    assert float(d.t[-1]) == pytest.approx(horizon)
    if horizon > 1.0:
        assert dt[-1] > 1.2 * dt[0]  # the linear phase allows far more than 0.05


def test_fixed_step_budget_ending_short_of_t_max_adds_no_empty_chunk():
    """Regression: dt=0.05 over 8000 steps sums to t_max - 1e-10, not t_max.

    The loop then asked for a zero-step chunk and the runaway check raised on
    its empty ``Wg_t`` (a ``diagnostics_interval`` run made the same chunk
    empty). The Cyclone deck is shrunk only in grid and moments.
    """
    import gkx
    from gkx.workflows.nonlinear import _chunk_runaway

    assert _chunk_runaway(np.empty(0)) is False
    deck = gkx.load(REPO_ROOT / "examples/03_nonlinear_tokamak/case_full.toml")
    case = deck.replace(
        grid=replace(deck.grid, Nx=4, Ny=4, Nz=8, ntheta=8),
        time=replace(
            deck.time,
            fixed_dt=True,
            dt=0.05,
            dt_max=0.05,
            t_max=400.0,
            diagnostics_interval=0.5,
            run_to="t_max",
        ),
    )
    d = gkx.solve(case, ky_target=case.run.ky, Nl=2, Nm=3).diagnostics
    assert len(d.t) == 800 and float(d.t[-1]) == pytest.approx(400.0, abs=1e-6)


def test_chunk_loop_horizon_slack_ends_a_run_that_stops_just_short():
    chunk = [0.5, 1.0 - 1e-9]  # each chunk is relative to its own start
    times = iter([chunk, chunk])

    def integrate(_show_progress, remaining):
        t = np.asarray(next(times))
        return t, _cadence_diag(t, t), object(), object()

    kw = dict(t_max=2.0, chunk_steps=2, label="slack")
    done = run_adaptive_runtime_chunk_loop(
        integrate_chunk=integrate, horizon_slack=1e-6, **kw
    )
    assert len(done.diagnostics.t) == 4
    with pytest.raises(StopIteration):  # without slack it asks for a third chunk
        times = iter([chunk, chunk])
        run_adaptive_runtime_chunk_loop(integrate_chunk=integrate, **kw)


def test_explicit_chunk_retries_at_half_cfl_fac_on_non_finite_state(monkeypatch):
    """A non-finite or runaway chunk is redone at half the CFL factor, and it sticks."""
    import gkx
    import gkx.workflows.nonlinear as workflow

    real = workflow.integrate_nonlinear_explicit_diagnostics_state
    seen, steps = [], []

    def flaky(G, *args, **kwargs):
        seen.append(kwargs["cfl_fac"])
        t, diag, G_next, fields = real(G, *args, **kwargs)
        steps.append(float(np.asarray(diag.dt_t)[0]))
        if kwargs["cfl_fac"] > 2.0:  # non-finite, then a finite energy jump past 1e3
            G_next = G_next * jnp.nan
        elif kwargs["cfl_fac"] > 1.0:
            wg = jnp.asarray(diag.Wg_t)
            diag = replace(diag, Wg_t=jnp.concatenate([wg, 1e6 * wg + 1e4]))
        return t, diag, G_next, fields

    monkeypatch.setattr(
        workflow, "integrate_nonlinear_explicit_diagnostics_state", flaky
    )
    deck = gkx.load(REPO_ROOT / "examples/03_nonlinear_tokamak/case.toml")
    case = deck.replace(
        grid=replace(deck.grid, Nx=4, Ny=4, Nz=8, ntheta=8),
        time=replace(
            deck.time,
            method="rk3",
            fixed_dt=False,
            dt_max=10.0,
            cfl_fac=3.0,
            t_max=0.5,
            chunk_steps=2,
            run_to="t_max",
        ),
    )
    d = gkx.solve(case, ky_target=case.run.ky, Nl=2, Nm=3).diagnostics
    assert seen[:3] == [3.0, 1.5, 0.75]
    assert steps[2] == pytest.approx(
        steps[1] / 2
    )  # the smaller factor reaches the step
    assert max(seen[3:]) == 0.75
    assert np.isfinite(np.asarray(d.Wg_t)).all()


def _cadence_diag(times, values=None):
    times = np.asarray(times)
    values = times if values is None else np.asarray(values)
    zeros = np.zeros_like(times)
    return SimulationDiagnostics(
        t=times,
        dt_t=np.full_like(times, 0.1),
        dt_mean=np.asarray(0.1),
        gamma_t=values,
        omega_t=zeros,
        Wg_t=values,
        Wphi_t=values,
        Wapar_t=zeros,
        heat_flux_t=values,
        particle_flux_t=zeros,
        energy_t=2 * values,
        resolved=ResolvedDiagnostics(Phi2_kxt=values[:, None]),
    )


@pytest.mark.parametrize("partition", [1, 3, 7])
def test_physical_cadence_keeps_fresh_samples_across_chunk_partitions(partition):
    times = np.arange(1, 11, dtype=float) / 10
    cursor = 0
    flags, messages, monitoring = [], [], []

    def integrate(show_progress, remaining):
        nonlocal cursor
        stop = min(cursor + partition, times.size)
        offset = 0.0 if cursor == 0 else times[cursor - 1]
        diag = _cadence_diag(times[cursor:stop] - offset, times[cursor:stop])
        flags.append(show_progress)
        cursor = stop
        return diag.t, diag, object(), object()

    def monitor(t, q, final):
        monitoring.append((t.copy(), q.copy()))
        return {"stop": False}

    result = run_adaptive_runtime_chunk_loop(
        integrate_chunk=integrate,
        t_max=1.0,
        chunk_steps=partition,
        label="cadence",
        diagnostics_stride=7,
        diagnostics_interval=0.25,
        stop_condition=monitor,
        show_progress=True,
        status_callback=messages.append,
        time_offset=3.0,
        progress_t_max=4.0,
    )
    expected = np.asarray([0.3, 0.5, 0.8, 1.0])
    np.testing.assert_allclose(result.diagnostics.t, expected, atol=1e-14)
    np.testing.assert_allclose(result.diagnostics.heat_flux_t, expected, atol=1e-14)
    np.testing.assert_allclose(result.diagnostics.resolved.Phi2_kxt[:, 0], expected)
    np.testing.assert_allclose(result.diagnostics.dt_t, 0.1)
    np.testing.assert_allclose(monitoring[-1][0], times)
    np.testing.assert_allclose(monitoring[-1][1], times)
    assert flags and not any(flags)
    assert "from t=3 up to t_max=4" in messages[0]
    assert "t=4/4" in messages[-1] and "elapsed=" in messages[-1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_physical_cadence_deduplicates_crossings_and_keeps_final(dtype):
    from gkx.workflows.runtime.diagnostic_arrays import _time_sample_runtime_diagnostics

    diag = _cadence_diag(np.asarray([0.1, 0.9, 1.0], dtype=dtype))
    selected = _time_sample_runtime_diagnostics(diag, 0.2)
    np.testing.assert_allclose(selected.t, [0.9, 1.0])
    final_only = _time_sample_runtime_diagnostics(diag, 2.0)
    np.testing.assert_allclose(final_only.t, [1.0])
    boundary = _cadence_diag(np.asarray([0.25, 0.25, 0.5], dtype=dtype))
    selected = _time_sample_runtime_diagnostics(boundary, 0.25, previous_time=0.25)
    np.testing.assert_allclose(selected.t, [0.5])
    exact = _cadence_diag(np.asarray([0.1, 0.2, 0.3], dtype=dtype))
    np.testing.assert_allclose(_time_sample_runtime_diagnostics(exact, 0.1).t, exact.t)


def test_physical_cadence_keeps_the_accepted_terminal_row_before_padding():
    from gkx.workflows.runtime.diagnostic_arrays import _time_sample_runtime_diagnostics

    diag = replace(_cadence_diag([0.1, 0.3, 0.3]), dt_t=np.asarray([0.1, 0.2, 0.0]))
    selected = _time_sample_runtime_diagnostics(diag, 0.25)
    np.testing.assert_allclose(selected.t, [0.3])
    np.testing.assert_allclose(selected.dt_t, [0.2])
    diag = _cadence_diag(np.arange(1, 11) / 10)
    selected = _time_sample_runtime_diagnostics(
        diag, 0.25, previous_time=1.13, time_offset=1.13
    )
    np.testing.assert_allclose(selected.t, [0.2, 0.4, 0.7, 0.9, 1.0])


def _lawson_box(nl, nm, n=4):
    import gkx

    case = gkx.load(REPO_ROOT / "examples/03_nonlinear_tokamak/case.toml")
    case = case.replace(grid=replace(case.grid, Nx=n, Ny=n, Nz=n, ntheta=n))
    geom = rt.build_runtime_geometry(case)
    grid = build_spectral_grid(apply_geometry_grid_defaults(geom, case.grid))
    params = rt.build_runtime_linear_params(case, Nm=nm, geom=geom)
    cache = build_linear_cache(
        grid, ensure_flux_tube_geometry_data(geom, grid.z), params, nl, nm
    )
    return case, grid, geom, params, cache


@pytest.mark.parametrize("kron_moments", [0, 128], ids=["separable", "dense"])
def test_lawson_drift_factor_is_the_matrix_exponential_of_the_truncated_operator(
    monkeypatch, kron_moments
):
    """exp(h L_d) from the eigenbasis equals ``expm`` of the production drift operator."""
    import gkx.solvers.nonlinear.lawson as lawson
    from scipy.linalg import expm
    from gkx.terms.linear_terms import drift_contributions

    monkeypatch.setattr(lawson, "_KRON_MOMENTS", kron_moments)
    with jax.enable_x64(True):
        nl, nm = 3, 5
        case, _grid, _geom, params, cache = _lawson_box(nl, nm)
        terms = rt.build_runtime_term_config(case)
        factor = lawson.build_drift_factor(cache, params, terms, 1, jnp.float64)
        shape = (1, nl, nm, *cache.cv_d.shape)
        size = int(np.prod(shape))
        ones, zero = jnp.ones(()), jnp.zeros(())

        def drift(x):  # the production curvature + grad-B terms, mirror off
            _, curv, gradb = drift_contributions(
                x.reshape(shape),
                cache,
                vth=jnp.ones(1),
                tz=jnp.asarray(params.tz).reshape(1),
                omega_d_scale=params.omega_d_scale,
                weights=(zero, ones, ones),
                imag=jnp.asarray(1j),
            )
            return np.asarray(curv + gradb).reshape(-1)

        eye = jnp.eye(size, dtype=jnp.complex128)
        operator = np.stack([drift(eye[i]) for i in range(size)], axis=1)
        h = 40.0 / np.abs(np.linalg.eigvals(operator)).max()  # |h L_d| = 40
        x = np.random.default_rng(0).standard_normal((2, size))
        x = (x[0] + 1j * x[1]).reshape(shape)
        eigen = factor.forward(jnp.asarray(x))
        got = np.asarray(factor.inverse(factor.rotate(eigen, h))).reshape(-1)
        want = expm(h * operator) @ x.reshape(-1)
        assert np.linalg.norm(got - want) < 1e-12 * np.linalg.norm(want)
        np.testing.assert_allclose(np.asarray(factor.inverse(eigen)), x, atol=1e-13)


@pytest.mark.parametrize("method", ["rk3", "rk4"])
def test_lawson_steppers_follow_the_explicit_ones_at_a_small_step(method):
    """At a step far inside both stability limits the two routes integrate one system."""
    import gkx

    case = gkx.load(REPO_ROOT / "examples/03_nonlinear_tokamak/case.toml")
    case = case.replace(
        grid=replace(case.grid, Nx=8, Ny=8, Nz=8, ntheta=8),
        init=replace(case.init, init_amp=0.05),  # the bracket is not negligible
    )
    wg = {}
    for name in (method, f"lawson-{method}"):
        time = replace(
            case.time, method=name, dt=0.01, fixed_dt=True, t_max=0.4, run_to="t_max"
        )
        out = gkx.solve(case.replace(time=time), ky_target=case.run.ky, Nl=3, Nm=6)
        wg[name] = np.asarray(out.diagnostics.Wg_t)
    assert np.ptp(wg[method]) > 0.01 * wg[method][0]
    np.testing.assert_allclose(wg[f"lawson-{method}"], wg[method], rtol=2e-4)


def test_lawson_cfl_bound_leaves_out_the_drift_frequencies():
    from gkx.operators.nonlinear.policies import _build_nonlinear_cfl_bounds
    from gkx.solvers.nonlinear.diagnostics import _laguerre_velocity_max
    from gkx.solvers.time_explicit import _linear_frequency_bound

    _case, grid, geom, params, cache = _lawson_box(2, 4, n=8)
    omega = {}
    for exact in (False, True):
        bounds = _build_nonlinear_cfl_bounds(
            grid,
            ensure_flux_tube_geometry_data(geom, grid.z),
            params,
            cache,
            real_dtype=jnp.float32,
            linear_frequency_bound_fn=_linear_frequency_bound,
            laguerre_velocity_max_fn=_laguerre_velocity_max,
            drift_exact=exact,
        )
        omega[exact] = np.asarray(bounds.linear_omega)
    assert omega[False][0] > 0.0 and omega[False][1] > 0.0
    np.testing.assert_array_equal(omega[True][:2], 0.0)
    assert omega[True][2] == omega[False][2] > 0.0


@pytest.mark.parametrize(
    ("omega", "cap_ratio", "sharded", "method"),
    [
        # Frequency bounds (radial drift, binormal drift, streaming) and step cap over the
        # rk4-pi bound step measured on the decks of the Lawson study (cap dt = 0.1 unless
        # noted), with the faster stepper of each.
        ([21.1, 26.7, 24.7], 2.86, False, "lawson-rk4-pi"),  # Cyclone (8, 16)
        ([19.2, 24.2, 24.7], 2.68, False, "lawson-rk4-pi"),  # Cyclone (4, 16)
        ([62.9, 23.3, 24.7], 4.37, False, "lawson-rk4-pi"),  # Miller (8, 16)
        ([774.3, 4839.2, 1416.4], 2.78, False, "lawson-rk4-pi"),  # ETG, dt_max 0.001
        (
            [10.5, 13.3, 17.5],
            16.3,
            False,
            "lawson-rk4-pi",
        ),  # Cyclone (4, 8), cap lifted
        ([10.5, 13.3, 17.5], 1.63, False, "rk4-pi"),  # Cyclone (4, 8): the cap binds
        ([8.4, 10.6, 15.1], 1.34, False, "rk4-pi"),  # Cyclone (4, 6)
        ([31.4, 11.6, 17.4], 2.38, False, "rk4-pi"),  # Cyclone Miller
        ([15.5, 14.0, 31.6], 2.41, False, "rk4-pi"),  # W7-X QI
        ([8.6, 9.4, 18.8], 1.45, False, "rk4-pi"),  # HSX
        ([17.3, 18.9, 26.6], 2.48, False, "rk4-pi"),  # HSX (8, 16)
        ([4.6, 5.7, 1001.5], 90.0, False, "rk4-pi"),  # kinetic electrons: streaming
        ([21.1, 26.7, 24.7], 2.86, True, "rk4-pi"),  # the sharded routes keep rk4-pi
    ],
)
def test_auto_stepper_follows_the_drift_share_and_the_step_cap(
    omega, cap_ratio, sharded, method
):
    from gkx.config import RuntimeConfig, TimeConfig
    from gkx.workflows.nonlinear import _resolve_auto_stepper

    cfg = RuntimeConfig(time=TimeConfig(method="auto"))
    messages: list[str] = []
    out = _resolve_auto_stepper(
        cfg,
        (0.05, cap_ratio, np.asarray(omega)),
        exact_drift_ok=not sharded,
        status=messages.append,
    )
    assert out.time.method == method
    assert method in messages[0]


def test_auto_stepper_without_a_bound_takes_rk4_pi():
    from gkx.config import RuntimeConfig, TimeConfig
    from gkx.workflows.nonlinear import _resolve_auto_stepper

    cfg = RuntimeConfig(time=TimeConfig(method="auto"))
    out = _resolve_auto_stepper(cfg, None, exact_drift_ok=True, status=lambda _: None)
    assert out.time.method == "rk4-pi"


def test_chunk_runaway_ignores_small_seeds_and_zero_start():
    from gkx.workflows.nonlinear import _chunk_runaway

    assert not _chunk_runaway(np.asarray([1e-12, 1e-6]))  # linear growth, tiny seed
    assert not _chunk_runaway(np.asarray([0.0, 5.0]))  # no starting energy to compare
    assert not _chunk_runaway(np.asarray([-1.0, 1e9]))
    assert _chunk_runaway(np.asarray([1.0, 1e5]))
    assert _chunk_runaway(np.asarray([1.0, np.nan]))
    assert _chunk_runaway(np.asarray([0.0, np.inf]))
