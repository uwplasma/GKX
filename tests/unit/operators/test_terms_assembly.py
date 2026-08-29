from dataclasses import replace

import numpy as np
import jax.numpy as jnp
import pytest

from gkx.config import GridConfig
from gkx.geometry import SAlphaGeometry
from gkx.core.grid import build_spectral_grid, select_ky_grid
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.terms import assembly as assembly_mod
from gkx.terms.assembly import (
    assemble_rhs_cached,
    assemble_rhs_cached_electrostatic_jit,
    assemble_rhs_cached_jit,
    assemble_rhs_terms_cached,
    compute_fields_cached,
)
from gkx.terms.config import FieldState, TermConfig


def test_assemble_rhs_terms_sum_matches_total() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        D_hyper=0.07,
    )
    Nl, Nm = 4, 4
    cache = build_linear_cache(grid, geom, params, Nl, Nm)
    rng = np.random.default_rng(0)
    G0 = rng.normal(
        size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0)
    term_cfg = TermConfig(hyperdiffusion=1.0)
    rhs_total, _fields = assemble_rhs_cached(G0, cache, params, terms=term_cfg)
    rhs_terms, _fields_terms, contrib = assemble_rhs_terms_cached(
        G0, cache, params, terms=term_cfg
    )
    rhs_sum = (
        contrib["streaming"]
        + contrib["mirror"]
        + contrib["curvature"]
        + contrib["gradb"]
        + contrib["diamagnetic"]
        + contrib["collisions"]
        + contrib["hypercollisions"]
        + contrib["hyperdiffusion"]
        + contrib["end_damping"]
    )
    assert np.allclose(
        np.asarray(rhs_terms), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8
    )
    assert np.allclose(
        np.asarray(rhs_sum), np.asarray(rhs_total), rtol=1.0e-6, atol=1.0e-8
    )


def test_assemble_rhs_cached_validates_state_shape_and_species_match() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)

    with pytest.raises(ValueError):
        assemble_rhs_cached(jnp.ones((2, 3, 4, 5), dtype=jnp.complex64), cache, params)

    G_species = jnp.ones(
        (2, 3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    with pytest.raises(ValueError):
        assemble_rhs_cached(G_species, cache, params)


def test_compute_fields_cached_matches_rhs_fields_and_validation() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(2)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0)

    rhs, fields_rhs = assemble_rhs_cached(
        G0, cache, params, use_custom_vjp=False, dt=0.1
    )
    fields_only = compute_fields_cached(G0, cache, params, use_custom_vjp=False)
    assert rhs.shape == G0.shape
    assert np.allclose(
        np.asarray(fields_only.phi),
        np.asarray(fields_rhs.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    with pytest.raises(ValueError):
        compute_fields_cached(
            jnp.ones((2, 3, 4, 5), dtype=jnp.complex64), cache, params
        )


def test_disabled_em_fields_skip_hamiltonian_branches(monkeypatch) -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    G0 = jnp.ones((3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    terms = TermConfig(apar=0.0, bpar=0.0)
    fields = FieldState(
        phi=jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        apar=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
        bpar=jnp.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64),
    )
    apar, bpar, h_apar, h_bpar = assembly_mod._rhs_field_views(fields, terms)
    assert apar.shape == fields.phi.shape
    assert bpar.shape == fields.phi.shape
    assert h_apar is None
    assert h_bpar is None

    seen: dict[str, bool] = {}
    original_build_h = assembly_mod.build_H

    def _record_build_h(*args, **kwargs):
        seen["apar_is_none"] = kwargs.get("apar") is None
        seen["bpar_is_none"] = kwargs.get("bpar") is None
        return original_build_h(*args, **kwargs)

    monkeypatch.setattr(assembly_mod, "build_H", _record_build_h)
    assemble_rhs_cached(G0, cache, params, terms=terms, use_custom_vjp=False)
    assert seen == {"apar_is_none": True, "bpar_is_none": True}


def test_assemble_rhs_cached_jit_accepts_term_config() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    G0 = jnp.ones((3, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    rhs, fields = assemble_rhs_cached_jit(
        G0, cache, params, TermConfig(apar=0.0, bpar=0.0)
    )
    assert rhs.shape == G0.shape
    assert fields.phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)


def test_electrostatic_rhs_jit_matches_generic_zero_em_fields() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.8,
        tprim=2.49,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
        beta=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 4)
    rng = np.random.default_rng(8)
    G0 = rng.normal(
        size=(3, 4, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 4, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    terms = TermConfig(apar=0.0, bpar=0.0)

    rhs_generic, fields_generic = assemble_rhs_cached_jit(G0, cache, params, terms)
    rhs_electrostatic, fields_electrostatic = assemble_rhs_cached_electrostatic_jit(
        G0, cache, params, terms
    )

    np.testing.assert_allclose(
        np.asarray(rhs_electrostatic), np.asarray(rhs_generic), rtol=1.0e-6, atol=1.0e-6
    )
    np.testing.assert_allclose(
        np.asarray(fields_electrostatic.phi),
        np.asarray(fields_generic.phi),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_external_phi_source_shifts_fields_and_rhs() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 2, 2)
    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)

    fields0 = compute_fields_cached(G0, cache, params, use_custom_vjp=False)
    fields_src = compute_fields_cached(
        G0, cache, params, use_custom_vjp=False, external_phi=0.25
    )
    np.testing.assert_allclose(
        np.asarray(fields_src.phi - fields0.phi), 0.25, atol=1.0e-7
    )

    rhs0, _ = assemble_rhs_cached(G0, cache, params, use_custom_vjp=False)
    rhs_src, _ = assemble_rhs_cached(
        G0, cache, params, use_custom_vjp=False, external_phi=0.25
    )
    assert not np.allclose(np.asarray(rhs_src), np.asarray(rhs0))


def test_collision_zero_guard_uses_current_nu_not_cache_build_nu() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(3)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
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
        apar=0.0,
        bpar=0.0,
    )

    rhs_zero, _fields_zero, contrib_zero = assemble_rhs_terms_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    np.testing.assert_allclose(np.asarray(rhs_zero), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(contrib_zero["collisions"]), 0.0, atol=1.0e-7)

    rhs_nonzero, _fields_nonzero, contrib_nonzero = assemble_rhs_terms_cached(
        G0,
        cache,
        replace(params, nu=0.2),
        terms=terms,
        use_custom_vjp=False,
    )
    assert np.linalg.norm(np.asarray(rhs_nonzero)) > 1.0e-5
    assert np.linalg.norm(np.asarray(contrib_nonzero["collisions"])) > 1.0e-5


def test_collision_zero_guard_preserves_preexpanded_collision_operator() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    rng = np.random.default_rng(4)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    cache_with_collision_matrix = replace(
        cache,
        collision_lam=jnp.ones_like(G0[None, ...], dtype=jnp.float32) * 0.2,
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
        apar=0.0,
        bpar=0.0,
    )

    rhs, _fields, contrib = assemble_rhs_terms_cached(
        G0,
        cache_with_collision_matrix,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    assert np.linalg.norm(np.asarray(rhs)) > 1.0e-5
    assert np.linalg.norm(np.asarray(contrib["collisions"])) > 1.0e-5


def test_collision_zero_weight_skips_invalid_preexpanded_operator_shape() -> None:
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=8, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        fprim=0.0,
        tprim=0.0,
        tprim_e=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    cache = build_linear_cache(grid, geom, params, 3, 3)
    cache_with_unused_bad_collision_matrix = replace(
        cache,
        collision_lam=jnp.ones_like(cache.lb_lam, dtype=jnp.float32),
    )
    rng = np.random.default_rng(5)
    G0 = rng.normal(
        size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(3, 3, grid.ky.size, grid.kx.size, grid.z.size))
    G0 = jnp.asarray(G0, dtype=jnp.complex64)
    terms = TermConfig(
        streaming=0.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    rhs, _fields, contrib = assemble_rhs_terms_cached(
        G0,
        cache_with_unused_bad_collision_matrix,
        params,
        terms=terms,
        use_custom_vjp=False,
    )
    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=1.0e-7)
    np.testing.assert_allclose(np.asarray(contrib["collisions"]), 0.0, atol=1.0e-7)


def test_static_zero_switches_stay_static_inside_a_trace() -> None:
    """A term switch the host can see is off stays off under ``jit``.

    ``_is_static_zero`` used to ask whether a ``jnp`` copy of its argument was
    traced. Inside a trace that copy always is, so every switch of every jitted
    run answered "not statically zero" and every fast path this predicate
    selects was silently abandoned -- including the electrostatic RHS for a
    ``TermConfig`` with ``apar = bpar = 0``, which is what the linear
    benchmarks and every electrostatic nonlinear run are.
    """

    import jax

    from gkx.operators.nonlinear.rhs import linear_rhs_jit_for_terms_impl

    seen: dict[str, object] = {}

    def probe(x: jnp.ndarray) -> jnp.ndarray:
        seen["python_zero"] = assembly_mod._is_static_zero(0.0)
        seen["python_one"] = assembly_mod._is_static_zero(1.0)
        seen["host_array"] = assembly_mod._is_static_zero(np.zeros(3))
        seen["underflows"] = assembly_mod._is_static_zero(1.0e-50, jnp.float32)
        seen["traced"] = assembly_mod._is_static_zero(x)
        seen["route"] = linear_rhs_jit_for_terms_impl(TermConfig(apar=0.0, bpar=0.0))
        return x * 2.0

    jax.jit(probe)(jnp.asarray(3.0))
    assert seen["python_zero"] is True
    assert seen["python_one"] is False
    assert seen["host_array"] is True
    # ``dtype`` still reproduces the operator's own cast.
    assert seen["underflows"] is True
    # A value that really is traced stays unknowable, which is the only case
    # the predicate was ever entitled to give up on.
    assert seen["traced"] is False
    assert seen["route"] is assemble_rhs_cached_electrostatic_jit
    assert (
        linear_rhs_jit_for_terms_impl(TermConfig(apar=1.0, bpar=0.0))
        is assemble_rhs_cached_jit
    )
