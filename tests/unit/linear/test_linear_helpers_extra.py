from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.diagnostics.analysis import estimate_observed_order
from gkx.config import GridConfig
from gkx.geometry import FluxTubeGeometryData, SAlphaGeometry
from gkx.core.grid import build_spectral_grid
from gkx.core.velocity import _gyro_bessel_factors, J_l_all
import gkx.operators.linear as linear_cache
import gkx.solvers.linear.integrators as linear_integrators
import gkx.solvers.linear.implicit as linear_implicit
import gkx.operators.linear.dissipation as linear_dissipation
import gkx.terms.linear_terms as linear_terms
from gkx.operators.linear.cache_arrays import hypercollision_damping
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.moments import build_H, lenard_bernstein_eigenvalues
from gkx.operators.linear.params import LinearParams, LinearTerms, linear_terms_to_term_config, term_config_to_linear_terms
from gkx.operators.linear.rhs import linear_rhs, linear_rhs_cached
from gkx.solvers.linear.integrators import integrate_linear, integrate_linear_diagnostics
from gkx.solvers.linear.parallel import linear_rhs_parallel_cached
from gkx.operators.linear.cache_arrays import (
    _build_end_damping_profile_array,
    _build_gyroaverage_cache_arrays,
    _build_low_rank_moment_cache_arrays,
)
from gkx.operators.linear.linked import (
    _build_linked_end_damping_profile,
    _build_linked_fft_maps,
    _signed_to_index,
)
from gkx.operators.linear.params import (
    _as_species_array,
    _check_nonnegative,
    _check_positive,
    _is_tracer,
    _resolve_implicit_preconditioner,
)
from gkx.solvers.linear.implicit import (
    _build_implicit_operator,
    _integrate_linear_implicit_cached,
)
from gkx.solvers.linear.integrators import _integrate_linear_cached_impl
from gkx.solvers.linear.parallel import (
    _is_electrostatic_field_terms,
    _is_electrostatic_slice_terms,
    _is_streaming_only_terms,
    _resolve_parallel_devices,
)
from gkx.terms.config import FieldState, TermConfig


def test_linear_validation_helpers_scalar_and_array() -> None:
    _check_positive(1.0, "x")
    _check_nonnegative(0.0, "x")
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    _check_nonnegative(jnp.asarray([0.0, 2.0]), "arr")

    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_nonnegative(-1.0, "x")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")
    with pytest.raises(ValueError):
        _check_nonnegative(jnp.asarray([0.0, -1.0]), "arr")


def test_linear_validation_helpers_accept_traced_values() -> None:
    @jax.jit
    def _checked(x):
        _check_positive(x, "x")
        _check_nonnegative(x, "x")
        return x

    assert float(_checked(jnp.asarray(1.0))) == pytest.approx(1.0)


def test_linear_validation_helpers_still_check_host_values_inside_a_trace() -> None:
    """A concrete argument is validated under ``jit``, not waved through.

    The guard used to build a ``jnp`` copy of its argument and ask whether
    *that* was traced. Inside a trace it always is, so the answer was yes for
    every argument and the operator stopped validating anything at all under
    ``jit`` -- a negative ``vth`` or a zero ``dz`` written into a jitted run
    went straight into the kernels. Only a genuinely traced value may skip.
    """

    # The guard is called from inside a live trace and its verdict is carried
    # out rather than raised out, so the trace this test opens is also the
    # trace it closes.
    seen: dict[str, str | None] = {}

    def verdict(fn, value, name):
        try:
            fn(value, name)
        except ValueError as exc:
            return str(exc)
        return None

    def probe(x):
        seen["negative_scalar"] = verdict(_check_positive, -1.0, "vth")
        seen["zero_in_array"] = verdict(
            _check_positive, np.asarray([1.0, 0.0]), "dz"
        )
        seen["negative_nonneg"] = verdict(_check_nonnegative, -1.0, "tau_e")
        seen["good_scalar"] = verdict(_check_positive, 1.0, "vth")
        seen["traced"] = verdict(_check_positive, x, "vth")
        return x * 2.0

    jax.jit(probe)(jnp.asarray(-1.0))
    assert seen["negative_scalar"] == "vth must be > 0"
    assert seen["zero_in_array"] == "dz must be > 0"
    assert seen["negative_nonneg"] == "tau_e must be >= 0"
    assert seen["good_scalar"] is None
    # A negative value that is genuinely traced still goes unchecked: its sign
    # is not known until the trace runs, which is the one case the guard may
    # skip.
    assert seen["traced"] is None


def test_as_species_array_and_preconditioner_resolution() -> None:
    np.testing.assert_allclose(
        np.asarray(_as_species_array(2.0, 3, "nu")), [2.0, 2.0, 2.0]
    )
    np.testing.assert_allclose(
        np.asarray(_as_species_array(jnp.asarray([1.0, 2.0]), 2, "nu")), [1.0, 2.0]
    )
    with pytest.raises(ValueError):
        _as_species_array(jnp.asarray([1.0, 2.0]), 3, "nu")

    assert _resolve_implicit_preconditioner(None) == "auto"
    assert _resolve_implicit_preconditioner("  Damping ") == "damping"

    def fn(x):
        return x

    assert _resolve_implicit_preconditioner(fn) is fn


def test_structured_tridiagonal_last_axis_matches_fused_reference_and_jvp() -> None:
    dtype = jnp.complex128 if jax.config.read("jax_enable_x64") else jnp.complex64
    real_dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32
    shape = (2, 3, 7)
    phase = jnp.arange(np.prod(shape), dtype=real_dtype).reshape(shape)
    lower = (0.03 + 0.01j) * jnp.cos(phase).astype(dtype)
    upper = (-0.02 + 0.015j) * jnp.sin(phase + 0.2).astype(dtype)
    lower = lower.at[..., 0].set(0.0)
    upper = upper.at[..., -1].set(0.0)
    diagonal = jnp.full(shape, 2.5 + 0.1j, dtype=dtype)
    rhs = (jnp.cos(phase) + 1j * jnp.sin(0.3 * phase)).astype(dtype)

    solve = jax.jit(
        lambda value: linear_implicit._solve_tridiagonal_last_axis(
            lower, diagonal, upper, value
        )
    )
    actual = solve(rhs)
    reference = jax.lax.linalg.tridiagonal_solve(
        lower, diagonal, upper, rhs[..., None]
    )[..., 0]
    tolerance = 2.0e-6 if dtype == jnp.complex64 else 1.0e-12
    np.testing.assert_allclose(actual, reference, rtol=tolerance, atol=tolerance)

    tangent = jnp.full(shape, 0.2 - 0.1j, dtype=dtype)
    _, tangent_out = jax.jvp(solve, (rhs,), (tangent,))
    np.testing.assert_allclose(
        tangent_out,
        solve(tangent),
        rtol=tolerance,
        atol=tolerance,
    )



def test_linear_dissipation_terms_have_single_canonical_owner() -> None:
    for name in linear_dissipation.__all__:
        assert getattr(linear_terms, name) is getattr(linear_dissipation, name)
        assert inspect.getmodule(getattr(linear_terms, name)) is linear_dissipation


def test_is_tracer_and_lenard_bernstein_eigenvalues() -> None:
    assert _is_tracer(1.0) is False
    traced_flag = jax.jit(
        lambda x: jnp.asarray(1 if _is_tracer(x) else 0, dtype=jnp.int32)
    )(1.0)
    assert int(traced_flag) == 1

    expected = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    got = np.asarray(
        lenard_bernstein_eigenvalues(2, 3, nu_hermite=0.3, nu_laguerre=0.7),
        dtype=np.float32,
    )
    np.testing.assert_allclose(got, expected)


def test_low_rank_moment_and_damping_cache_match_expected_shapes_and_values() -> None:
    params = LinearParams(
        nu_hermite=0.3, nu_laguerre=0.7, p_hyper=2, p_hyper_l=3, p_hyper_m=4
    )
    cache = _build_low_rank_moment_cache_arrays(2, 3, params, jnp.float32)

    expected_lb = np.asarray([[0.0, 0.3, 0.6], [0.7, 1.0, 1.3]], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(cache["lb_lam"]), expected_lb, rtol=1e-6)
    assert cache["lb_lam"].shape == (2, 3)
    assert cache["hyper_ratio"].shape == (2, 3, 1, 1, 1)
    assert cache["sqrt_p"].shape == (1, 1, 3, 1, 1, 1)
    assert cache["mask_const"].dtype == jnp.bool_

    periodic = np.asarray(
        _build_end_damping_profile_array(8, 0.25, "periodic", jnp.float32)
    )
    linked = np.asarray(
        _build_end_damping_profile_array(8, 0.25, "linked", jnp.float32)
    )
    np.testing.assert_allclose(periodic, np.zeros(8, dtype=np.float32))
    assert linked[0] > 0.0
    assert linked[-1] > 0.0


def test_low_rank_moment_cache_keeps_high_order_kz_hypercollision_finite() -> None:
    params = LinearParams(p_hyper_m=20.0)
    cache = _build_low_rank_moment_cache_arrays(24, 128, params, jnp.float32)

    assert np.all(np.isfinite(np.asarray(cache["m_pow"])))
    assert np.isfinite(float(np.asarray(cache["m_norm_kz_factor"])))
    assert float(np.max(np.asarray(cache["m_pow"]))) <= 1.0


def test_hypercollision_damping_preserves_low_moments_and_grows_with_kz() -> None:
    params = LinearParams(
        nu_hyper=0.0,
        nu_hyper_l=0.2,
        nu_hyper_m=0.3,
        nu_hyper_lm=0.4,
        hypercollisions_const=1.0,
        hypercollisions_kz=1.0,
        p_hyper_l=2.0,
        p_hyper_m=4.0,
        p_hyper_lm=2.0,
        vth=1.5,
        kpar_scale=2.0,
    )
    moment_cache = _build_low_rank_moment_cache_arrays(3, 5, params, jnp.float32)
    cache = SimpleNamespace(
        **moment_cache,
        kz=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float32),
    )

    damping = np.asarray(hypercollision_damping(cache, params, jnp.float32))

    assert damping.shape == (1, 3, 5, 1, 1, 3)
    np.testing.assert_allclose(damping[0, 0, 0, 0, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(damping[0, 1, 2, 0, 0], 0.0, atol=0.0)
    assert damping[0, 2, 0, 0, 0, 0] > 0.0
    assert damping[0, 0, 4, 0, 0, 2] > damping[0, 0, 4, 0, 0, 1]
    assert damping[0, 0, 4, 0, 0, 1] > damping[0, 0, 4, 0, 0, 0]


def test_gyroaverage_cache_helper_matches_species_vmap_convention() -> None:
    b = jnp.asarray(
        [
            [[[0.0, 0.2], [0.4, 0.6]]],
            [[[0.1, 0.3], [0.5, 0.7]]],
        ],
        dtype=jnp.float32,
    )
    Jl, JlB = _build_gyroaverage_cache_arrays(b, Nl=3, real_dtype=jnp.float32)
    expected = jax.vmap(lambda bs: J_l_all(bs, l_max=2))(b).astype(jnp.float32)

    np.testing.assert_allclose(np.asarray(Jl), np.asarray(expected), rtol=1e-6)
    assert Jl.shape == (2, 3, 1, 2, 2)
    assert JlB.shape == Jl.shape
    np.testing.assert_allclose(np.asarray(JlB[:, 0]), np.asarray(Jl[:, 0]), rtol=1e-6)


def test_linear_params_and_terms_roundtrip() -> None:
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]), nu=jnp.asarray([0.1, 0.2]), beta=0.3
    )
    leaves, treedef = jax.tree_util.tree_flatten(params)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_allclose(np.asarray(restored.charge_sign), [1.0, -1.0])
    np.testing.assert_allclose(np.asarray(restored.nu), [0.1, 0.2])
    assert float(restored.beta) == pytest.approx(0.3)

    terms = LinearTerms(apar=0.0, bpar=0.0, hyperdiffusion=1.0)
    term_cfg = linear_terms_to_term_config(terms, nonlinear=0.25)
    assert float(term_cfg.nonlinear) == pytest.approx(0.25)
    assert term_config_to_linear_terms(term_cfg) == terms

    assert linear_terms_to_term_config(None) == TermConfig()
    assert term_config_to_linear_terms(None) == LinearTerms()

    custom_cfg = TermConfig(
        streaming=0.2,
        mirror=0.3,
        curvature=0.4,
        gradb=0.5,
        diamagnetic=0.6,
        collisions=0.7,
        hypercollisions=0.8,
        hyperdiffusion=0.9,
        end_damping=0.1,
        apar=0.0,
        bpar=1.0,
        nonlinear=3.0,
    )
    assert term_config_to_linear_terms(custom_cfg) == LinearTerms(
        streaming=0.2,
        mirror=0.3,
        curvature=0.4,
        gradb=0.5,
        diamagnetic=0.6,
        collisions=0.7,
        hypercollisions=0.8,
        hyperdiffusion=0.9,
        end_damping=0.1,
        apar=0.0,
        bpar=1.0,
    )


def test_linear_term_classifiers_and_parallel_device_validation() -> None:
    streaming_only = LinearTerms(
        streaming=1.0,
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
    electrostatic_slices = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    assert _is_streaming_only_terms(streaming_only) is True
    assert _is_streaming_only_terms(LinearTerms()) is False
    assert _is_electrostatic_slice_terms(electrostatic_slices) is True
    assert _is_electrostatic_slice_terms(LinearTerms(collisions=1.0)) is False
    assert _is_electrostatic_field_terms(LinearTerms(apar=0.0, bpar=0.0)) is True
    assert _is_electrostatic_field_terms(LinearTerms(apar=1.0, bpar=0.0)) is False

    devices = ["cpu0", "cpu1"]
    assert _resolve_parallel_devices(devices=devices) == devices
    assert _resolve_parallel_devices(devices=devices, num_devices=2) == devices
    with pytest.raises(ValueError, match="must match"):
        _resolve_parallel_devices(devices=devices, num_devices=1)
    with pytest.raises(ValueError, match="at least one"):
        _resolve_parallel_devices(devices=[])
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_parallel_devices(num_devices=0)
    with pytest.raises(ValueError, match="requested"):
        _resolve_parallel_devices(num_devices=len(jax.devices()) + 1)


def test_signed_to_index_and_linked_end_damping_profile() -> None:
    assert _signed_to_index(0, 3) == 0
    assert _signed_to_index(1, 3) == 1
    assert _signed_to_index(-1, 3) == 2
    assert _signed_to_index(-3, 3) == -1

    linked = (jnp.asarray([[1, 4]], dtype=jnp.int32),)
    profile = _build_linked_end_damping_profile(
        linked_indices=linked,
        ny=3,
        nx=2,
        nz=4,
        widthfrac=0.5,
        ky_mode=np.asarray([0, 1, -1], dtype=np.int32),
    )
    assert profile.shape == (3, 2, 4)
    assert np.max(profile) > 0.0
    assert np.all(profile[0] == 0.0)

    empty = _build_linked_end_damping_profile(
        linked_indices=(),
        ny=2,
        nx=2,
        nz=2,
        widthfrac=0.5,
    )
    assert np.allclose(empty, 0.0)

    with pytest.raises(ValueError):
        _build_linked_end_damping_profile(
            linked_indices=linked,
            ny=3,
            nx=2,
            nz=4,
            widthfrac=0.5,
            ky_mode=np.asarray([0, 1], dtype=np.int32),
        )

    profile_zero_width = _build_linked_end_damping_profile(
        linked_indices=(jnp.asarray([[1]], dtype=jnp.int32),),
        ny=2,
        nx=1,
        nz=4,
        widthfrac=0.01,
    )
    assert np.allclose(profile_zero_width, 0.0)


def test_linked_fft_maps_validate_ky_mode_and_empty_maps() -> None:
    empty_indices, empty_kz = _build_linked_fft_maps(
        ky=np.asarray([]),
        kx=np.asarray([]),
        y0=1.0,
        nz=4,
        dz=0.5,
        jtwist=1,
        real_dtype=jnp.float32,
    )
    assert empty_indices == ()
    assert empty_kz == ()

    indices, kz = _build_linked_fft_maps(
        ky=np.asarray([0.0, 0.1, 0.2]),
        kx=np.asarray([0.0, 0.2, -0.2, 0.4]),
        y0=1.0,
        nz=4,
        dz=0.5,
        jtwist=1,
        real_dtype=jnp.float32,
        ky_mode=np.asarray([0, 1, 2]),
    )
    assert len(indices) == len(kz)
    assert all(idx.ndim == 2 for idx in indices)

    profile = _build_linked_end_damping_profile(
        linked_indices=(jnp.asarray([1, 2], dtype=jnp.int32),),
        ny=3,
        nx=2,
        nz=4,
        widthfrac=0.5,
    )
    assert np.allclose(profile, 0.0)


def test_build_linear_cache_linked_non_twist_contract() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            y0=1.0,
            boundary="linked",
            jtwist=1,
            non_twist=True,
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.25)

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    assert cache.use_twist_shift is True
    assert cache.jtwist == 1
    assert cache.kperp2.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert len(cache.linked_indices) == len(cache.linked_kz)
    assert cache.linked_damp_profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))
    if cache.linked_indices:
        assert cache.linked_use_gather is True
        assert cache.linked_gather_map.shape == (grid.ky.size * grid.kx.size,)


def test_build_linear_cache_y0_default_and_zero_twist_branches() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="linked",
            jtwist=None,
        )
    )
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_widthfrac=0.0)
    geom = SAlphaGeometry(q=1.4, s_hat=0.0, epsilon=0.1)

    cache = build_linear_cache(replace(grid, y0=None), geom, params, Nl=2, Nm=2)

    assert cache.use_twist_shift is True
    assert cache.jtwist == 1
    assert cache.linked_damp_profile.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.allclose(np.asarray(cache.linked_damp_profile), 0.0)
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))


def _sampled_geometry_with_shear(
    theta: jnp.ndarray, s_hat: jnp.ndarray
) -> FluxTubeGeometryData:
    shear = s_hat * theta
    ones = jnp.ones_like(theta)
    zeros = jnp.zeros_like(theta)
    return FluxTubeGeometryData(
        theta=theta,
        gradpar_value=1.0,
        bmag_profile=ones,
        bgrad_profile=zeros,
        gds2_profile=1.0 + shear * shear,
        gds21_profile=-s_hat * shear,
        gds22_profile=s_hat * s_hat * ones,
        cv_profile=jnp.cos(theta) + shear * jnp.sin(theta),
        gb_profile=jnp.cos(theta) + shear * jnp.sin(theta),
        cv0_profile=-s_hat * jnp.sin(theta),
        gb0_profile=-s_hat * jnp.sin(theta),
        jacobian_profile=ones,
        grho_profile=ones,
        q=1.4,
        s_hat=s_hat,
        epsilon=0.1,
        R0=1.0,
        source_model="sampled-test",
    )


def test_build_linear_cache_allows_traced_shear_for_periodic_sampled_geometry() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=2, Ny=4, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        )
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float32)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def objective(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
        return jnp.sum(cache.kperp2)

    grad = jax.grad(objective)(jnp.asarray(0.8, dtype=jnp.float32))

    assert np.isfinite(float(grad))


def test_laguerre_bessel_factors_have_analytic_zero_limit_and_tangent() -> None:
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def factors(alpha2: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack(_gyro_bessel_factors(alpha2))

    value, tangent = jax.jvp(factors, (zero,), (jnp.ones_like(zero),))

    np.testing.assert_allclose(value, np.asarray([1.0, 0.5]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        tangent, np.asarray([-0.25, -0.0625]), rtol=1.0e-12, atol=1.0e-12
    )


def test_laguerre_bessel_cache_has_finite_geometry_tangent_at_zero_mode() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=2, Ny=4, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi, boundary="periodic"
        )
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float64)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def observable(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=2, Nm=1)
        return jnp.sum(cache.laguerre_j0) + jnp.sum(cache.laguerre_j1_over_alpha)

    shear = jnp.asarray(0.8, dtype=jnp.float64)
    tangent = jax.jvp(observable, (shear,), (jnp.ones_like(shear),))[1]

    assert np.isfinite(float(tangent))


def test_build_linear_cache_periodic_non_twist_uses_geometry_shear() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=2,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="periodic",
            non_twist=True,
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    assert cache.use_twist_shift is False
    assert np.all(np.isfinite(np.asarray(cache.kperp2)))
    assert np.all(np.isfinite(np.asarray(cache.cv_d)))
    assert np.all(np.isfinite(np.asarray(cache.gb_d)))


def test_sheared_kx_cache_zero_shear_identity_and_tangent() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=4,
            Ny=4,
            Nz=8,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="periodic",
        )
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(rho_star=1.0, nu_hyper=0.0, nu_hyper_m=0.0)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=3)

    identity = linear_cache.update_linear_cache_for_sheared_kx(
        cache,
        grid,
        geom,
        params,
        cache.kx_grid,
    )
    for name in (
        "kx_grid",
        "kperp2",
        "cv_d",
        "gb_d",
        "omega_d",
        "b",
        "Jl",
        "JlB",
        "laguerre_j0",
        "laguerre_j1_over_alpha",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(identity, name)),
            np.asarray(getattr(cache, name)),
            rtol=2.0e-6,
            atol=2.0e-7,
        )

    G = (
        jnp.arange(2 * 3 * 4 * 4 * 8, dtype=jnp.float32).reshape(2, 3, 4, 4, 8)
        / 1000.0
    ).astype(jnp.complex64)
    rhs_base, phi_base = linear_rhs_cached(G, cache, params, use_jit=False)
    rhs_identity, phi_identity = linear_rhs_cached(
        G,
        identity,
        params,
        use_jit=False,
    )
    np.testing.assert_allclose(rhs_identity, rhs_base, rtol=3.0e-6, atol=3.0e-7)
    np.testing.assert_allclose(phi_identity, phi_base, rtol=3.0e-6, atol=3.0e-7)

    def observable(shift: jnp.ndarray) -> jnp.ndarray:
        effective_kx = cache.kx_grid - shift * cache.ky_grid
        updated = linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom,
            params,
            effective_kx,
        )
        return jnp.sum(updated.kperp2) + 0.1 * jnp.sum(updated.Jl)

    shift = jnp.asarray(0.17, dtype=jnp.float32)
    tangent = jax.jvp(observable, (shift,), (jnp.ones_like(shift),))[1]
    step = jnp.asarray(5.0e-3, dtype=jnp.float32)
    finite_difference = (observable(shift + step) - observable(shift - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(tangent, finite_difference, rtol=5.0e-4)

    non_twist_grid = replace(grid, boundary="linked", non_twist=True)
    with pytest.raises(NotImplementedError, match="periodic or linked standard"):
        linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            non_twist_grid,
            geom,
            params,
            cache.kx_grid,
        )
    with pytest.raises(ValueError, match="shape \(ky, kx\)"):
        linear_cache.update_linear_cache_for_sheared_kx(
            cache,
            grid,
            geom,
            params,
            cache.kx,
        )


def test_hyperdiffusion_accepts_sheared_two_dimensional_kx() -> None:
    G = jnp.ones((1, 1, 4, 4, 2), dtype=jnp.complex64)
    kx = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    ky = jnp.asarray([0.0, 1.0, -2.0, -1.0], dtype=jnp.float32)
    mask = jnp.ones((4, 4), dtype=bool)
    kwargs = {
        "ky": ky,
        "dealias_mask": mask,
        "D_hyper": jnp.asarray(0.1),
        "p_hyper_kperp": jnp.asarray(2.0),
        "weight": jnp.asarray(1.0),
    }
    one_dimensional = linear_dissipation.hyperdiffusion_contribution(
        G,
        kx=kx,
        **kwargs,
    )
    two_dimensional = linear_dissipation.hyperdiffusion_contribution(
        G,
        kx=jnp.broadcast_to(kx[None, :], mask.shape),
        **kwargs,
    )
    np.testing.assert_allclose(two_dimensional, one_dimensional)


def test_build_linear_cache_rejects_traced_shear_for_twist_shift_geometry() -> None:
    grid = build_spectral_grid(
        GridConfig(
            Nx=2,
            Ny=4,
            Nz=4,
            Lx=2.0 * np.pi,
            Ly=2.0 * np.pi,
            boundary="linked",
            jtwist=1,
        )
    )
    theta = jnp.asarray(grid.z, dtype=jnp.float32)
    params = LinearParams(nu_hyper=0.0, nu_hyper_m=0.0)

    def objective(s_hat: jnp.ndarray) -> jnp.ndarray:
        geom = _sampled_geometry_with_shear(theta, s_hat)
        cache = build_linear_cache(grid, geom, params, Nl=1, Nm=1)
        return jnp.sum(cache.kperp2)

    with pytest.raises(
        ValueError,
        match="differentiating with respect to magnetic shear is not supported",
    ):
        jax.grad(objective)(jnp.asarray(0.8, dtype=jnp.float32))


def test_build_H_field_couplings_and_errors() -> None:
    G5 = jnp.zeros((2, 2, 1, 1, 3), dtype=jnp.complex64)
    Jl4 = jnp.ones((2, 1, 1, 3), dtype=jnp.float32)
    phi = jnp.ones((1, 1, 3), dtype=jnp.complex64)
    apar = 0.5 * phi
    bpar = 0.25 * phi

    H = build_H(
        G5,
        Jl4,
        phi,
        tz=jnp.asarray(2.0),
        apar=apar,
        vth=jnp.asarray(3.0),
        bpar=bpar,
        JlB=Jl4,
    )

    assert H.shape == G5.shape
    assert np.max(np.abs(np.asarray(H[0, 0]))) > 0.0
    assert np.max(np.abs(np.asarray(H[0, 1]))) > 0.0
    with pytest.raises(ValueError, match="vth"):
        build_H(G5, Jl4, phi, tz=1.0, apar=apar)
    with pytest.raises(ValueError, match="JlB"):
        build_H(G5, Jl4, phi, tz=1.0, bpar=bpar)


def test_linear_rhs_rejects_invalid_state_rank() -> None:
    with pytest.raises(ValueError, match="G must have shape"):
        linear_rhs(
            jnp.zeros((2, 2), dtype=jnp.complex64), object(), object(), LinearParams()
        )


def test_linear_rhs_accepts_multispecies_state() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=2, Ny=2, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(charge_sign=jnp.asarray([1.0]), nu_hyper=0.0, nu_hyper_m=0.0)
    G = jnp.zeros(
        (1, 2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )

    dG, phi = linear_rhs(G, grid, geom, params, terms=LinearTerms())

    assert dG.shape == G.shape
    assert phi.shape == (grid.ky.size, grid.kx.size, grid.z.size)
    assert np.all(np.isfinite(np.asarray(dG)))


def test_build_implicit_operator_handles_species_squeeze(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
    )
    params = SimpleNamespace(
        nu=0.1,
        tz=1.0,
        vth=1.0,
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    G, shape, size, dt_val, precond_op, matvec, squeeze_species = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner="damping",
        )
    )

    assert squeeze_species is True
    assert shape == (1, 2, 2, 1, 1, 2)
    assert size == 8
    assert G.shape == shape
    assert np.isfinite(np.asarray(precond_op(G))).all()
    assert np.isfinite(np.asarray(matvec(G))).all()
    assert float(dt_val) == pytest.approx(0.2)


def test_build_implicit_operator_preconditioner_aliases_and_errors(monkeypatch) -> None:
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.ones((2,), dtype=jnp.float32),
        sqrt_p=jnp.ones((2,), dtype=jnp.float32),
        kz=jnp.array([0.0, 1.0], dtype=jnp.float32),
        linked_indices=(),
        linked_kz=(),
        use_twist_shift=False,
    )
    params = SimpleNamespace(
        nu=jnp.asarray([0.1], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    size = int(np.prod(np.asarray(G0.shape)))
    x = jnp.ones((size,), dtype=G0.dtype)
    for key in ("pas-coarse", "hermite-line", "hermite-line-coarse", "identity"):
        _G, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = (
            _build_implicit_operator(
                G0,
                cache,
                params,
                dt=0.2,
                terms=LinearTerms(),
                implicit_preconditioner=key,
            )
        )
        y = precond_op(x)
        assert y.shape == x.shape
        assert np.isfinite(np.asarray(y)).all()

    _G, _shape, _size, _dt_val, callable_precond, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner=lambda x: 2.0 * x,
        )
    )
    np.testing.assert_allclose(np.asarray(callable_precond(x)), np.asarray(2.0 * x))

    with pytest.raises(ValueError):
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.2,
            terms=LinearTerms(),
            implicit_preconditioner="not-a-preconditioner",
        )


def test_build_implicit_operator_linked_hermite_line_preconditioner(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((1, 1, 2, 1, 1, 2), dtype=jnp.complex64)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2, d=1.0)
    cache = SimpleNamespace(
        lb_lam=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        l=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        m=jnp.ones((1, 1, 2, 1, 1, 2), dtype=jnp.float32),
        cv_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        gb_d=jnp.ones((1, 1, 2), dtype=jnp.float32),
        bgrad=jnp.ones((2,), dtype=jnp.float32),
        sqrt_m_ladder=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        sqrt_p=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        kz=jnp.asarray([0.0, np.pi], dtype=jnp.float32),
        linked_indices=(jnp.asarray([[0]], dtype=jnp.int32),),
        linked_kz=(kz_link,),
        use_twist_shift=True,
    )
    params = SimpleNamespace(
        nu=jnp.asarray([0.1], dtype=jnp.float32),
        tz=jnp.asarray([1.0], dtype=jnp.float32),
        vth=jnp.asarray([1.0], dtype=jnp.float32),
        omega_d_scale=1.0,
        kpar_scale=1.0,
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (jnp.ones_like(G), None),
    )

    _G, _shape, _size, _dt_val, precond_op, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.1,
            terms=LinearTerms(),
            implicit_preconditioner="hermite-line",
        )
    )
    y = precond_op(jnp.ones((G0.size,), dtype=G0.dtype))

    assert y.shape == (G0.size,)
    assert np.all(np.isfinite(np.asarray(y)))

    _G, _shape, _size, _dt_val, coarse_precond, _matvec, _squeeze = (
        _build_implicit_operator(
            G0,
            cache,
            params,
            dt=0.1,
            terms=LinearTerms(),
            implicit_preconditioner="hermite-line-coarse",
        )
    )
    y_coarse = coarse_precond(jnp.ones((G0.size,), dtype=G0.dtype))
    assert y_coarse.shape == (G0.size,)
    assert np.all(np.isfinite(np.asarray(y_coarse)))


def test_integrate_linear_wrapper_routes_methods(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_cached",
        lambda *args, **kwargs: calls.append(("cached", kwargs["method"]))
        or ("G", "phi"),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_cached_donate",
        lambda *args, **kwargs: calls.append(("donate", kwargs["method"]))
        or ("Gd", "phid"),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_implicit_cached",
        lambda *args, **kwargs: calls.append(("implicit", "implicit"))
        or ("Gi", "phii"),
    )

    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="semi-implicit"
    ) == ("G", "phi")
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", donate=True
    ) == ("Gd", "phid")
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="implicit"
    ) == ("Gi", "phii")
    assert ("cached", "imex") in calls
    assert ("donate", "rk2") in calls
    assert ("implicit", "implicit") in calls

    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0)
    with pytest.raises(ValueError):
        integrate_linear(G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2)
    with pytest.raises(ValueError):
        integrate_linear(jnp.zeros((2, 2)), grid, geom, params, dt=0.1, steps=2)


def test_cached_linear_binding_resolves_impl_at_trace(monkeypatch) -> None:
    bound = linear_integrators._bind_cached_linear_integrator(donate=False)
    calls: list[bool] = []

    def _fake_impl(G0, *args, **kwargs):
        calls.append(True)
        return G0 + 1.0, G0 + 2.0

    monkeypatch.setattr(linear_integrators, "_integrate_linear_cached_impl", _fake_impl)
    G = jnp.zeros((2,), dtype=jnp.float32)
    state, field = bound(G, G, G, 0.1, 1)

    assert calls == [True]
    np.testing.assert_array_equal(state, jnp.ones_like(G))
    np.testing.assert_array_equal(field, 2.0 * jnp.ones_like(G))


def test_integrate_linear_wrapper_routes_nonserial_parallel(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    parallel = SimpleNamespace(strategy="velocity", backend="auto")
    calls: list[object] = []

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_cached_impl",
        lambda *args, **kwargs: calls.append(kwargs["parallel"]) or ("Gp", "phip"),
    )

    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", parallel=parallel
    ) == ("Gp", "phip")
    assert calls == [parallel]

    with pytest.raises(NotImplementedError, match="explicit fixed-step"):
        integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=0.1,
            steps=2,
            method="implicit",
            parallel=parallel,
        )
    with pytest.raises(NotImplementedError, match="donated"):
        integrate_linear(
            G0,
            grid,
            geom,
            params,
            dt=0.1,
            steps=2,
            method="rk2",
            donate=True,
            parallel=parallel,
        )


def test_integrate_linear_wrapper_enables_electrostatic_field_specialization(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )

    def _fake_cached(*args, **kwargs):
        captured["force_electrostatic_fields"] = kwargs["force_electrostatic_fields"]
        return "G", "phi"

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_cached", _fake_cached
    )

    assert integrate_linear(
        G0,
        grid,
        geom,
        params,
        dt=0.1,
        steps=2,
        method="rk2",
        terms=LinearTerms(apar=0.0, bpar=0.0),
    ) == ("G", "phi")
    assert captured["force_electrostatic_fields"] is True


@pytest.mark.parametrize(
    "terms",
    [
        None,
        LinearTerms(apar=1.0, bpar=0.0),
        LinearTerms(apar=0.0, bpar=1.0),
    ],
)
def test_integrate_linear_wrapper_does_not_force_electrostatic_when_em_terms_enabled(
    monkeypatch,
    terms: LinearTerms | None,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    captured: dict[str, bool] = {}

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.build_linear_cache",
        lambda *args, **kwargs: "cache",
    )

    def _fake_cached(*args, **kwargs):
        captured["force_electrostatic_fields"] = kwargs["force_electrostatic_fields"]
        return "G", "phi"

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators._integrate_linear_cached", _fake_cached
    )

    kwargs = {} if terms is None else {"terms": terms}
    assert integrate_linear(
        G0, grid, geom, params, dt=0.1, steps=2, method="rk2", **kwargs
    ) == ("G", "phi")
    assert captured["force_electrostatic_fields"] is False


def test_linear_rhs_cached_uses_generic_jit_unless_electrostatic_is_forced(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_generic(G, cache, params, terms, dt=None, external_phi=None):
        calls.append("generic")
        assert float(terms.apar) == pytest.approx(0.0)
        assert float(terms.bpar) == pytest.approx(0.0)
        assert float(dt) == pytest.approx(0.25)
        return jnp.zeros_like(G), FieldState(
            phi=jnp.zeros(G.shape[-3:], dtype=G.dtype), apar=None, bpar=None
        )

    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_jit", _fake_generic
    )
    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_electrostatic_jit",
        lambda *args, **kwargs: pytest.fail(
            "electrostatic RHS should run only when explicitly forced"
        ),
    )

    rhs, phi = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(apar=0.0, bpar=0.0),
        dt=0.25,
    )

    assert calls == ["generic"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]


def test_linear_rhs_parallel_cached_serial_alias_and_error_branches(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_serial(G, cache, params, **kwargs):
        calls.append("serial")
        assert kwargs["use_jit"] is False
        assert kwargs["use_custom_vjp"] is False
        assert float(kwargs["dt"]) == pytest.approx(0.2)
        return jnp.ones_like(G), jnp.zeros(G.shape[-3:], dtype=G.dtype)

    monkeypatch.setattr(
        "gkx.operators.linear.rhs.linear_rhs_cached", _fake_serial
    )

    rhs, phi = linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(),
        parallel=None,
        use_jit=False,
        use_custom_vjp=False,
        dt=0.2,
    )
    assert calls == ["serial"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]

    with pytest.raises(NotImplementedError, match="Hermite axis"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(apar=0.0, bpar=0.0),
            parallel=SimpleNamespace(
                strategy="velocity", backend="auto", axis="laguerre"
            ),
        )
    with pytest.raises(NotImplementedError, match="backend='auto'"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(),
            parallel=SimpleNamespace(strategy="velocity", backend="auto"),
        )
    with pytest.raises(NotImplementedError, match="currently supports only"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(),
            parallel=SimpleNamespace(strategy="kx", backend="auto"),
        )


def test_linear_rhs_parallel_cached_routes_gated_velocity_backends(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[tuple[str, int | None]] = []

    streaming_only = LinearTerms(
        streaming=1.0,
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
    electrostatic_slices = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )

    def _out(name):
        def _fake(G, cache, params, **kwargs):
            calls.append((name, kwargs.get("num_devices")))
            return jnp.ones_like(G), jnp.zeros(G.shape[-3:], dtype=G.dtype)

        return _fake

    monkeypatch.setattr(
        "gkx.solvers.linear.parallel.linear_rhs_streaming_velocity_sharded",
        _out("streaming"),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.parallel.linear_rhs_streaming_electrostatic_velocity_sharded",
        _out("streaming_es"),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.parallel.linear_rhs_electrostatic_slices_velocity_sharded",
        _out("slices"),
    )

    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=streaming_only,
        parallel=SimpleNamespace(
            strategy="velocity", backend="streaming_only", num_devices=1
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=streaming_only,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="streaming_electrostatic",
            num_devices=2,
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=electrostatic_slices,
        parallel=SimpleNamespace(
            strategy="velocity",
            backend="electrostatic_linear_slices",
            num_devices=3,
        ),
    )
    linear_rhs_parallel_cached(
        G0,
        cache,
        params,
        terms=electrostatic_slices,
        parallel=SimpleNamespace(strategy="velocity", backend="auto", num_devices=4),
    )

    assert calls == [
        ("streaming", 1),
        ("streaming_es", 2),
        ("slices", 3),
        ("slices", 4),
    ]

    with pytest.raises(NotImplementedError, match="streaming-only"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=electrostatic_slices,
            parallel=SimpleNamespace(strategy="velocity", backend="streaming_only"),
        )
    with pytest.raises(NotImplementedError, match="collision/EM"):
        linear_rhs_parallel_cached(
            G0,
            cache,
            params,
            terms=LinearTerms(collisions=1.0, apar=0.0, bpar=0.0),
            parallel=SimpleNamespace(
                strategy="velocity", backend="electrostatic_linear_slices"
            ),
        )


def test_linear_rhs_cached_can_use_electrostatic_specialized_jit(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = params = object()
    calls: list[str] = []

    def _fake_electrostatic(G, cache, params, terms, dt=None, external_phi=None):
        calls.append("electrostatic")
        return jnp.ones_like(G), FieldState(
            phi=jnp.ones(G.shape[-3:], dtype=G.dtype), apar=None, bpar=None
        )

    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_electrostatic_jit",
        _fake_electrostatic,
    )
    monkeypatch.setattr(
        "gkx.terms.assembly.assemble_rhs_cached_jit",
        lambda *args, **kwargs: pytest.fail(
            "generic RHS should not run when electrostatic specialization is forced"
        ),
    )

    rhs, phi = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=LinearTerms(apar=0.0, bpar=0.0),
        force_electrostatic_fields=True,
    )

    assert calls == ["electrostatic"]
    assert rhs.shape == G0.shape
    assert phi.shape == G0.shape[-3:]


def test_linear_rhs_cached_zero_and_near_zero_states_remain_finite() -> None:
    grid = build_spectral_grid(
        GridConfig(Nx=2, Ny=2, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    params = LinearParams(
        nu=0.0, nu_hyper=0.0, nu_hyper_m=0.0, damp_ends_amp=0.0, damp_ends_widthfrac=0.0
    )
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=2)
    terms = LinearTerms(
        collisions=0.0, hypercollisions=0.0, end_damping=0.0, apar=0.0, bpar=0.0
    )
    G0 = jnp.zeros((2, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)

    dG0, phi0 = linear_rhs_cached(
        G0,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    np.testing.assert_allclose(np.asarray(dG0), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(phi0), 0.0, atol=0.0)

    tiny = G0 + jnp.asarray(1.0e-30 + 1.0e-30j, dtype=G0.dtype)
    dG_tiny, phi_tiny = linear_rhs_cached(
        tiny,
        cache,
        params,
        terms=terms,
        use_jit=False,
        use_custom_vjp=False,
        force_electrostatic_fields=True,
    )
    assert np.all(np.isfinite(np.asarray(dG_tiny)))
    assert np.all(np.isfinite(np.asarray(phi_tiny)))


def test_integrate_linear_cached_impl_invalid_and_sampled(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.0)
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.zeros((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    with pytest.raises(ValueError):
        _integrate_linear_cached_impl(G0, cache, params, dt=0.1, steps=2, method="bad")

    G_out, phi_t = _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        method="euler",
        sample_stride=2,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_cached_impl_uses_parallel_rhs(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32))
    params = SimpleNamespace(nu=0.0)
    parallel = SimpleNamespace(strategy="velocity", backend="auto")
    calls: list[object] = []

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.linear_rhs_cached",
        lambda *args, **kwargs: pytest.fail(
            "serial RHS should not be used for nonserial parallel integration"
        ),
    )

    def _fake_parallel_rhs(G, cache, params, **kwargs):
        calls.append(kwargs["parallel"])
        return jnp.ones_like(G), jnp.zeros((1, 1, 2), dtype=jnp.complex64)

    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.linear_rhs_parallel_cached",
        _fake_parallel_rhs,
    )

    G_out, phi_t = _integrate_linear_cached_impl(
        G0,
        cache,
        params,
        dt=0.1,
        steps=2,
        method="euler",
        parallel=parallel,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert calls and all(call is parallel for call in calls)


def test_integrate_linear_implicit_cached_sampled_path(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace()
    params = SimpleNamespace()

    monkeypatch.setattr(
        "gkx.solvers.linear.implicit._build_implicit_operator",
        lambda *args, **kwargs: (
            G0,
            G0.shape,
            G0.size,
            jnp.asarray(0.1, dtype=jnp.float32),
            (lambda x: x),
            (lambda x: x),
            False,
        ),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.gmres",
        lambda matvec, rhs, **kwargs: SimpleNamespace(x=rhs, converged=True),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.implicit.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t = _integrate_linear_implicit_cached(
        G0,
        cache,
        params,
        dt=0.1,
        steps=4,
        sample_stride=2,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2


def test_integrate_linear_diagnostics_validates_and_records_energy(monkeypatch) -> None:
    G0 = jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    grid = geom = params = object()
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((1, 2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    with pytest.raises(ValueError):
        integrate_linear_diagnostics(
            G0, grid, geom, params, dt=0.1, steps=2, sample_stride=0, cache=cache
        )
    with pytest.raises(ValueError):
        integrate_linear_diagnostics(
            G0, grid, geom, params, dt=0.1, steps=3, sample_stride=2, cache=cache
        )

    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        grid,
        geom,
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=4,
        method="rk2",
        cache=cache,
        sample_stride=2,
        species_index=0,
        record_hl_energy=True,
    )
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2
    assert hl_t.shape[0] == 2


@pytest.mark.parametrize("method", ["euler", "rk2", "sspx3", "rk4"])
def test_integrate_linear_diagnostics_explicit_method_branches(
    monkeypatch, method: str
) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method=method,
        cache=cache,
        sample_stride=1,
        species_index=0,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2


def test_integrate_linear_diagnostics_multispecies_density_and_invalid_method(
    monkeypatch,
) -> None:
    G0 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t, density_t, hl_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
        dt=0.1,
        steps=2,
        method="rk4",
        cache=cache,
        sample_stride=1,
        species_index=None,
        record_hl_energy=True,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2
    assert hl_t.shape[0] == 2
    with pytest.raises(ValueError, match="Unsupported method"):
        integrate_linear_diagnostics(
            G0,
            object(),
            object(),
            SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
            dt=0.1,
            steps=1,
            method="bad",
            cache=cache,
        )


def test_integrate_linear_diagnostics_builds_cache_and_uses_imex2(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((1, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((1, 2, 1, 1, 2), dtype=jnp.float32),
    )
    build_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.build_linear_cache",
        lambda grid, geom, params, Nl, Nm: build_calls.append((Nl, Nm)) or cache,
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="imex2",
        cache=None,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )

    assert build_calls == [(2, 2)]
    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape[0] == 2


def test_integrate_linear_diagnostics_imex_sampled_multispecies(monkeypatch) -> None:
    G0 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    cache = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.zeros_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G_out, phi_t, density_t = integrate_linear_diagnostics(
        G0,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.1, 0.2], dtype=jnp.float32)),
        dt=0.1,
        steps=4,
        method="imex",
        cache=cache,
        sample_stride=2,
        species_index=None,
        record_hl_energy=False,
    )

    assert G_out.shape == G0.shape
    assert phi_t.shape[0] == 2
    assert density_t.shape == (2, 1, 1, 2)


def test_integrate_linear_diagnostics_species_none_and_5d_density_paths(
    monkeypatch,
) -> None:
    cache6 = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 2, 1, 1, 2), dtype=jnp.float32),
    )
    cache5 = SimpleNamespace(
        lb_lam=jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.float32),
        Jl=jnp.ones((2, 1, 1, 2), dtype=jnp.float32),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.hypercollision_damping",
        lambda cache, params, dtype: jnp.zeros_like(cache.lb_lam, dtype=dtype),
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrator_diagnostics.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.ones_like(G),
            jnp.ones((1, 1, 2), dtype=jnp.complex64),
        ),
    )

    G6 = jnp.zeros((2, 2, 2, 1, 1, 2), dtype=jnp.complex64)
    G6_out, phi6_t, density6_t = integrate_linear_diagnostics(
        G6,
        object(),
        object(),
        SimpleNamespace(nu=jnp.asarray([0.0, 0.0])),
        dt=0.1,
        steps=2,
        method="sspx3",
        cache=cache6,
        sample_stride=1,
        species_index=None,
        record_hl_energy=False,
    )
    assert G6_out.shape == G6.shape
    assert phi6_t.shape[0] == 2
    assert density6_t.shape[0] == 2

    G5 = jnp.zeros((2, 2, 1, 1, 2), dtype=jnp.complex64)
    G5_out, phi5_t, density5_t = integrate_linear_diagnostics(
        G5,
        object(),
        object(),
        SimpleNamespace(nu=0.0),
        dt=0.1,
        steps=2,
        method="rk4",
        cache=cache5,
        sample_stride=1,
        species_index=0,
        record_hl_energy=False,
    )
    assert G5_out.shape == G5.shape
    assert phi5_t.shape[0] == 2
    assert density5_t.shape[0] == 2


@pytest.mark.parametrize(
    ("method", "rate", "damping", "min_order"),
    [
        ("rk2", -0.7 + 0.3j, 0.0, 1.75),
        ("rk4", -0.7 + 0.3j, 0.0, 3.2),
        ("sspx3", -0.7 + 0.3j, 0.0, 2.6),
        ("imex", -0.7 + 0.3j, 0.8, 0.95),
        ("imex2", -0.7 + 0.3j, 0.0, 1.9),
    ],
)
def test_integrate_linear_cached_impl_observed_order_against_exact_solution(
    monkeypatch,
    method: str,
    rate: complex,
    damping: float,
    min_order: float,
) -> None:
    cache = SimpleNamespace(lb_lam=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32))
    G0 = jnp.asarray([[[[[1.0 + 0.25j]]]]], dtype=jnp.complex64)
    params = LinearParams(nu=0.0)
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.hypercollision_damping",
        lambda cache, params, dtype: jnp.ones_like(cache.lb_lam, dtype=dtype) * damping,
    )
    monkeypatch.setattr(
        "gkx.solvers.linear.integrators.linear_rhs_cached",
        lambda G, cache, params, **kwargs: (
            jnp.asarray(rate * G, dtype=G.dtype),
            jnp.asarray(G[0, 0], dtype=G.dtype),
        ),
    )

    t_final = 0.8
    step_sizes: list[float] = []
    errors: list[float] = []
    exact = np.exp(rate * t_final) * np.asarray(G0)
    for steps in (1, 2, 4, 8):
        dt = t_final / steps
        G_out, _phi_t = _integrate_linear_cached_impl(
            G0,
            cache,
            params,
            dt=dt,
            steps=steps,
            method=method,
        )
        errors.append(float(np.max(np.abs(np.asarray(G_out) - exact))))
        step_sizes.append(dt)

    metrics = estimate_observed_order(np.array(step_sizes), np.array(errors))
    assert metrics.asymptotic_order >= min_order
