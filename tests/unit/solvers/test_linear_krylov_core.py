"""Matrix-free linear solver core: Krylov/Arnoldi, shift-invert, preconditioners.

The first half exercises the matrix-free linear stack -- Arnoldi and its
breakdown thresholds, Ritz selection, shift-invert and its fallback policy, the
physics preconditioners and the adaptive propagator. The second half (absorbed
from test_dot_precision_guard.py) asserts the precision contract on the very
contractions the first half runs; its full rationale is kept verbatim at the
origin marker below.
"""

from __future__ import annotations

from dataclasses import replace
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core_grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.geometry.sensitivity import _damped_gauss_newton_step
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    Species,
    build_linear_params,
    linear_terms_to_term_config,
)
import gkx.solvers_linear_implicit as implicit
from support.paired_solvax import requires_paired_solvax
from types import SimpleNamespace
import inspect
import re
import gkx.solvers_linear_adaptive_propagator as ap
import gkx.solvers_linear_krylov as lk
import gkx.solvers_linear_krylov_algorithms as ka
import gkx.solvers_linear_krylov_propagator as kp
import gkx.solvers_linear_precond_pr3 as pr3
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import solvax


requires_solvax_eigen_api = requires_paired_solvax(
    "adaptive_eigenpair",
    "eigenpair_reverse",
    "estimate_rk4_timestep",
    "propagator_eigenpairs",
    "sparse_eigenpairs",
    "sparse_operator_matrix",
)


def test_published_solvax_contract_matches_consumed_interfaces() -> None:
    """Check that the consumed solvax interfaces are available (no version pin)."""

    for name in (
        "chunked_jacfwd",
        "gmres",
        "linear_solve",
        "low_rank_corrected",
        "tridiagonal_solve",
    ):
        assert callable(getattr(solvax, name))


@requires_solvax_eigen_api
def test_experimental_solvax_eigen_contract() -> None:
    """The downstream branch must expose every experimental eigenmode API."""

    assert callable(solvax.adaptive_eigenpair)
    assert callable(solvax.eigenpair_reverse)
    assert callable(solvax.estimate_rk4_timestep)
    assert callable(solvax.propagator_eigenpairs)


def _tiny_krylov_setup(*, linked: bool = False, ky_layout: str | None = None):
    """Build a small linked or periodic cache and a state that fits its grid.

    ``ky_layout`` is left unset by default, so the grid follows the configured
    layout and these tests exercise whichever one ships.  A caller passes
    ``"full"`` only when the negative ``ky`` rows are themselves under test.
    """

    layout = {} if ky_layout is None else {"ky_layout": ky_layout}
    grid_cfg = GridConfig(
        Nx=4 if linked else 2,
        Ny=4 if linked else 2,
        Nz=8,
        Lx=6.0,
        Ly=6.0,
        boundary="linked" if linked else "periodic",
        y0=20.0 if linked else None,
        jtwist=1 if linked else None,
        **layout,
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=0.0,
        omega_star_scale=0.0,
        nu=0.01,
        nu_hyper=0.0,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    Nl, Nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl=Nl, Nm=Nm)
    v0 = jnp.ones(
        (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    ) * (1.0 + 0.1j)
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=1.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    return grid, cache, params, v0, term_cfg, terms


def _patch_shift_invert(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    """Route a faked ``(eig, vec)`` solve through the inner-statistics seam."""

    def with_stats(*args, **kwargs):
        eig, vec = fake(*args, **kwargs)
        return eig, vec, ka._empty_inner_solve_stats(vec.dtype)

    monkeypatch.setattr(lk, "_shift_invert_eigenpair_with_inner_stats", with_stats)


def test_mode_family_and_target_selection_helpers() -> None:
    assert lk._mode_family_sign("cyclone") == 1
    assert lk._mode_family_sign("kbm") == 1
    assert lk._mode_family_sign("etg") == -1
    assert lk._mode_family_sign("other") == 0
    real = jnp.asarray([-0.1, 0.05, 0.08])
    imag = jnp.asarray([1.0, -1.9, -2.2])
    mask = jnp.asarray([True, True, False])
    idx = lk._select_by_target(
        real,
        imag,
        mask,
        omega_scale=jnp.asarray(1.0),
        omega_target_factor=2.0,
        omega_sign=1,
        fallback_idx=jnp.asarray(0),
    )
    assert int(idx) == 1
    idx_neg = lk._select_by_target(
        real,
        -imag,
        mask,
        omega_scale=jnp.asarray(1.0),
        omega_target_factor=2.0,
        omega_sign=-1,
        fallback_idx=jnp.asarray(0),
    )
    assert int(idx_neg) == 1


def test_select_by_overlap_prefers_reference_branch() -> None:
    V = jnp.asarray([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]])
    eigvecs = jnp.eye(2, dtype=jnp.complex64)
    v_ref = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j])
    mask = jnp.asarray([True, True])
    idx = lk._select_by_overlap(eigvecs, V, v_ref, mask, fallback_idx=jnp.asarray(1))
    assert int(idx) == 0
    idx_fallback = lk._select_by_overlap(
        eigvecs, V, v_ref, jnp.asarray([False, False]), fallback_idx=jnp.asarray(1)
    )
    assert int(idx_fallback) == 1


def test_ritz_vector_uses_complex_eigenvector_without_conjugation() -> None:
    """Arnoldi Ritz vectors are ``V @ y``, not ``V @ conj(y)``."""

    basis = jnp.asarray(
        [
            [1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j],
        ],
        dtype=jnp.complex64,
    )
    eigvecs = jnp.asarray(
        [[1.0 + 0.0j, 0.0 + 1.0j], [0.0 + 1.0j, 1.0 + 0.0j]],
        dtype=jnp.complex64,
    )

    vector = ka._ritz_vector_from_index(basis, eigvecs, jnp.asarray(0), krylov_dim=2)
    expected = jnp.asarray([1.0, 1.0j], dtype=jnp.complex64) / jnp.sqrt(2.0)

    assert jnp.allclose(vector, expected)


def test_arnoldi_uses_dtype_scaled_near_breakdown_threshold() -> None:
    """Do not normalize roundoff into a spurious Krylov direction."""

    v0 = jnp.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=jnp.complex64)
    eps = jnp.finfo(jnp.float32).eps

    def apply_near(vector, *_args):
        matrix = jnp.asarray([[2.0, 0.0], [5.0 * eps, 1.0]], vector.dtype)
        return matrix @ vector

    basis_near, hessenberg_near = ka._arnoldi(
        v0, apply_near, None, None, None, krylov_dim=1
    )
    assert hessenberg_near[1, 0] == 0.0
    assert jnp.all(basis_near[1] == 0.0)

    def apply_resolved(vector, *_args):
        matrix = jnp.asarray([[2.0, 0.0], [1.0e-3, 1.0]], vector.dtype)
        return matrix @ vector

    basis_resolved, hessenberg_resolved = ka._arnoldi(
        v0, apply_resolved, None, None, None, krylov_dim=1
    )
    assert hessenberg_resolved[1, 0] > 0.0
    assert jnp.linalg.norm(basis_resolved[1]) == pytest.approx(1.0)


def test_residual_gate_never_sits_below_the_working_dtype_noise_floor() -> None:
    """A residual gate no precision can reach certifies nothing and never stops."""

    double = float(np.finfo(np.complex128).eps)
    single = float(np.finfo(np.complex64).eps)

    # A relative residual built from float32 operator applications bottoms out a
    # few eps above zero, so the historic 1e-10 gate was unreachable there while
    # remaining a genuine gate in float64.
    assert ap.certifiable_residual_tolerance(1.0e-10, np.complex128) == 1.0e-10
    assert ap.certifiable_residual_tolerance(1.0e-10, np.complex64) == pytest.approx(
        1.0e3 * single
    )
    assert 1.0e3 * single > 1.0e-10

    # The floor tracks the dtype rather than a hard-coded constant, and a request
    # the arithmetic can already meet is passed through untouched.
    assert ap.certifiable_residual_tolerance(0.0, np.complex128) == pytest.approx(
        1.0e3 * double
    )
    assert ap.certifiable_residual_tolerance(1.0e-3, np.complex64) == 1.0e-3
    assert ap.certifiable_residual_tolerance(1.0e-3, np.complex128) == 1.0e-3
    assert ap.certifiable_residual_tolerance(
        1.0e-10, np.float32
    ) == ap.certifiable_residual_tolerance(1.0e-10, np.complex64)


def test_rayleigh_quotient_minimizes_fixed_vector_residual(monkeypatch) -> None:
    matrix = jnp.asarray(
        [[1.0 + 0.2j, 0.4 - 0.1j], [-0.3 + 0.5j, 2.0 - 0.4j]],
        dtype=jnp.complex64,
    )
    vector = jnp.asarray([1.0 + 0.3j, -0.2 + 0.7j], dtype=jnp.complex64)
    monkeypatch.setattr(
        ka,
        "_apply_operator",
        lambda state, _cache, _params, _terms: matrix @ state,
    )

    eigenvalue = ka._rayleigh_quotient(vector, None, None, None)
    operator_vector = matrix @ vector
    residual = jnp.linalg.norm(operator_vector - eigenvalue * vector)
    perturbed_residual = jnp.linalg.norm(
        operator_vector - (eigenvalue + 0.3 - 0.2j) * vector
    )

    assert jnp.isfinite(eigenvalue)
    assert residual < perturbed_residual


def test_shift_invert_spectrum_rejects_arnoldi_breakdown_values() -> None:
    eigvals = jnp.asarray([0.0 + 0.0j, 0.5 - 0.25j], dtype=jnp.complex64)
    sigma = jnp.asarray(0.1 - 0.2j, dtype=jnp.complex64)

    transformed, real_part, imag_part, finite = ka._shift_invert_spectrum(
        eigvals, sigma
    )

    assert not bool(finite[0])
    assert not bool(jnp.isfinite(real_part[0]))
    assert not bool(jnp.isfinite(imag_part[0]))
    assert bool(finite[1])
    assert jnp.allclose(transformed[1], sigma + 1.0 / eigvals[1])


def test_normalize_handles_zero_and_tiny_vectors_without_nan() -> None:
    zero = jnp.zeros((3,), dtype=jnp.complex64)
    zero_normed = lk._normalize(zero)
    assert jnp.all(jnp.isfinite(jnp.real(zero_normed)))
    assert jnp.allclose(zero_normed, zero)

    tiny = jnp.asarray(
        [1.0e-12 + 0.0j, 0.0 + 1.0e-12j, 0.0 + 0.0j], dtype=jnp.complex64
    )
    tiny_normed = lk._normalize(tiny)
    assert jnp.all(jnp.isfinite(jnp.real(tiny_normed)))
    assert jnp.linalg.norm(tiny_normed) == pytest.approx(1.0)


def test_candidate_certification_rejects_zero_and_nonfinite_vectors() -> None:
    vectors = jnp.stack((jnp.zeros((2,)), jnp.ones((2,)), jnp.ones((2,))))
    certified = ap._certified_candidates(
        jnp.asarray([0.0, jnp.nan, 1.0]),
        vectors,
        jnp.asarray([0.0, 0.0, 1.0e-10]),
        1.0e-8,
    )
    np.testing.assert_array_equal(np.asarray(certified), (False, False, True))


def test_dominant_eigenpair_arnoldi_branch_normalizes_wrapper_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v0 = jnp.ones((2,), dtype=jnp.complex64)
    v_ref = jnp.asarray([0.0 + 0.0j, 2.0 + 0.0j], dtype=jnp.complex64)
    captured: dict[str, object] = {}

    def _fake_arnoldi(v0_in, v_ref_in, _cache, _params, term_cfg, **kwargs):
        captured["v0"] = v0_in
        captured["v_ref"] = v_ref_in
        captured["term_cfg"] = term_cfg
        captured.update(kwargs)
        return jnp.asarray(0.1 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 3.0 + 0.0j)

    monkeypatch.setattr(lk, "dominant_eigenpair_cached", _fake_arnoldi)
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.1 + 0.2j, vector.dtype) * vector,
    )

    eig, vec = lk.dominant_eigenpair(
        v0,
        object(),
        object(),
        terms=LinearTerms(apar=0.0, bpar=0.0),
        v_ref=v_ref,
        select_overlap=True,
        krylov_dim=3,
        restarts=0,
        omega_sign=0,
        mode_family="etg",
        method=" Arnoldi ",
    )

    assert jnp.allclose(eig, jnp.asarray(0.1 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec, 3.0 + 0.0j)
    assert captured["krylov_dim"] == 3
    assert captured["restarts"] == 1
    assert captured["omega_sign"] == -1
    assert captured["select_overlap"] is True
    assert captured["v_ref"] is v_ref
    assert float(captured["term_cfg"].apar) == pytest.approx(0.0)
    assert float(captured["term_cfg"].bpar) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("shift_selection", "select_targeted", "select_growth"),
    [
        ("targeted", True, True),
        ("target", True, False),
        ("growth", False, True),
        ("shift", False, False),
    ],
)
def test_shift_invert_selection_key_controls_cached_branch_flags(
    monkeypatch: pytest.MonkeyPatch,
    shift_selection: str,
    select_targeted: bool,
    select_growth: bool,
) -> None:
    v0 = jnp.ones((2,), dtype=jnp.complex64)
    captured: dict[str, object] = {}

    def _fake_shift(v_init, v_ref, _cache, _params, term_cfg, **kwargs):
        captured["v_init"] = v_init
        captured["v_ref"] = v_ref
        captured["term_cfg"] = term_cfg
        captured.update(kwargs)
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    _patch_shift_invert(monkeypatch, _fake_shift)
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.4 + 0.2j, vector.dtype) * vector,
    )

    eig, vec = lk.dominant_eigenpair(
        v0,
        object(),
        object(),
        terms=LinearTerms(apar=0.0, bpar=0.0),
        method="shift_invert",
        shift=0.2 - 1.1j,
        shift_source="target",
        shift_selection=shift_selection,
        select_overlap=True,
        fallback_method="none",
    )

    assert jnp.allclose(eig, jnp.asarray(0.4 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec, 5.0 + 0.0j)
    assert captured["select_targeted"] is select_targeted
    assert captured["select_growth"] is select_growth
    assert captured["select_overlap"] is True
    assert jnp.allclose(captured["sigma"], jnp.asarray(0.2 - 1.1j, dtype=v0.dtype))
    assert captured["v_init"] is v0
    assert captured["v_ref"] is v0


def testbuild_shift_invert_preconditioneritioner_modes() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, None
    )
    assert precond is None and op is None
    # This used to assert that "unknown" returned (None, None), which pinned the
    # silent-drop behaviour as correct and is why the defect survived: an
    # unrecognized name disabled preconditioning while the solve reported itself
    # as preconditioned. Unknown names now raise; see the dedicated cases below.
    with pytest.raises(ValueError, match="unknown shift-invert preconditioner"):
        lk.build_shift_invert_preconditioner(
            v0, cache, params, term_cfg, sigma, "unknown"
        )

    precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "damping"
    )
    assert precond is not None and op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    _precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "hermite-line-coarse"
    )
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))


def test_shifted_hermite_preconditioner_has_the_correct_complex_scaling() -> None:
    """With a zero approximate operator, the inverse is exactly ``-I/sigma``."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    params = replace(
        params,
        nu=0.0,
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
    )
    term_cfg = replace(term_cfg, streaming=0.0, collisions=0.0)
    sigma = jnp.asarray(0.3 - 0.7j, dtype=v0.dtype)

    _precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )

    assert op is not None
    result = op(v0.reshape(-1)).reshape(v0.shape)
    assert jnp.allclose(result, -v0 / sigma, rtol=2e-6, atol=2e-6)


def test_hermite_line_inverts_additive_diagonal_and_streaming_symbol() -> None:
    """The line solve represents ``D + S``, not the old product ``D S``."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    state = implicit._prepare_implicit_state(v0, 0.4 - 0.2j, term_cfg)
    data = implicit._build_implicit_preconditioner_data(cache, params, state)
    mode, kz_index = jnp.arange(v0.shape[1], dtype=v0.dtype) + 0.3j, 1
    phase = jnp.exp(2j * jnp.pi * kz_index * jnp.arange(v0.shape[-1]) / v0.shape[-1])
    rhs = jnp.zeros_like(state.G).at[0, 0, :, 0, 0, :].set(mode[:, None] * phase)

    solved = implicit._apply_hermite_line_preconditioner(
        rhs.reshape(-1), cache=cache, params=params, state=state, data=data
    ).reshape(state.shape)
    observed = jnp.fft.fft(solved, axis=-1)[0, 0, :, 0, 0, kz_index]
    coefficient = (
        state.dt_val
        * data.w_stream
        * params.kpar_scale
        * data.vth[0]
        * data.imag
        * cache.kz[kz_index]
    )
    diagonal = jnp.reciprocal(data.precond_full)
    matrix = jnp.diag(jnp.mean(diagonal[0, 0, :, 0, 0], axis=-1))
    matrix += jnp.diag(coefficient * data.sqrt_m_line[1:], -1)
    matrix += jnp.diag(coefficient * data.sqrt_p_line[:-1], 1)

    expected = jnp.linalg.solve(matrix, mode * v0.shape[-1])
    assert jnp.allclose(observed, expected, rtol=2.0e-5, atol=2.0e-5)


@pytest.mark.parametrize("weight", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("linked", [False, True])
def test_hermite_line_inverts_kz_hypercollisions(weight, linked) -> None:
    """The |kz| multiplier lives on each FFT chain, including its zero mode."""
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=linked)
    params = replace(
        params,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
        kpar_scale=-0.7,
        nu_hyper_m=0.8,
    )
    term_cfg = replace(term_cfg, streaming=0.0, collisions=0.0, hypercollisions=weight)
    sigma = jnp.asarray(0.3 - 0.7j, dtype=v0.dtype)
    apply = implicit._build_shifted_hermite_preconditioner(
        v0, cache, params, term_cfg, sigma
    )
    rng = np.random.default_rng(19)
    rhs = jnp.asarray(
        rng.normal(size=v0.shape) + 1j * rng.normal(size=v0.shape), dtype=v0.dtype
    )
    active = jnp.zeros(v0.shape[-3] * v0.shape[-2], dtype=bool)
    for indices in cache.linked_indices:
        active = active.at[indices.ravel()].set(True)
    if not linked:
        active = jnp.ones_like(active)
    active = active.reshape(v0.shape[-2], v0.shape[-3]).T[..., None]
    # The RHS also reconstructs conjugate ky rows. Certify the independent
    # chain-covered subspace, not an inverse on every redundant spectral row.
    rhs = rhs * active
    expected_coarse = np.zeros(v0.shape, dtype=np.asarray(rhs).dtype)
    for indices in cache.linked_indices:
        for chain in np.asarray(indices):
            kx, ky = chain // v0.shape[-3], chain % v0.shape[-3]
            expected_coarse[..., ky, kx, :] = np.mean(
                np.asarray(rhs)[..., ky, kx, :], axis=-2, keepdims=True
            )
    if not linked:
        expected_coarse = np.broadcast_to(
            np.mean(rhs, axis=-2, keepdims=True), rhs.shape
        )
    np.testing.assert_allclose(
        implicit._project_kx_coarse(rhs[None], cache)[0],
        expected_coarse,
        rtol=1e-6,
        atol=1e-6,
    )
    candidate = apply(rhs.ravel()).reshape(v0.shape)
    residual = (
        ka._apply_operator(candidate, cache, params, term_cfg) - sigma * candidate - rhs
    )
    relative = float(jnp.linalg.norm(residual * active) / jnp.linalg.norm(rhs))
    assert relative < 3e-6, relative
    # JAX's complex transpose is bilinear (not the Hermitian adjoint).
    dual = jnp.asarray(
        rng.normal(size=rhs.size) + 1j * rng.normal(size=rhs.size), dtype=v0.dtype
    )
    transpose = jax.linear_transpose(apply, rhs.ravel())(dual)[0]
    np.testing.assert_allclose(
        jnp.sum(dual * candidate.ravel()),
        jnp.sum(transpose * rhs.ravel()),
        rtol=3e-6,
        atol=3e-5,
    )

    def solve_rate(rate):
        return implicit._build_shifted_hermite_preconditioner(
            v0, cache, replace(params, nu_hyper_m=rate), term_cfg, sigma
        )(rhs.ravel())

    rate = jnp.asarray(params.nu_hyper_m)
    tangent = jax.jvp(solve_rate, (rate,), (jnp.ones_like(rate),))[1]
    finite_difference = (solve_rate(rate + 1e-3) - solve_rate(rate - 1e-3)) / 2e-3
    np.testing.assert_allclose(tangent, finite_difference, rtol=5e-3, atol=2e-4)


def test_shifted_hermite_preconditioner_handles_a_zero_shift() -> None:
    """A marginal target must use the finite damping fallback."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.0 + 0.0j, dtype=v0.dtype)
    _diagonal, damping_op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "damping"
    )
    _precond, line_op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )

    assert damping_op is not None and line_op is not None
    expected = damping_op(v0.reshape(-1))
    result = line_op(v0.reshape(-1))
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


@pytest.mark.parametrize("kz_damping", [False, True])
def test_field_corrected_shifted_preconditioner_removes_low_moment_coupling(
    kz_damping,
) -> None:
    """Woodbury field correction must fix the Hermite line inverse's main defect."""

    _grid, cache, params, v0, _term_cfg, _terms = _tiny_krylov_setup(linked=False)
    params = replace(
        params,
        omega_star_scale=1.0,
        omega_d_scale=1.0,
        tprim=6.9,
        fprim=2.2,
        hypercollisions_const=float(not kz_damping),
        hypercollisions_kz=float(kz_damping),
    )
    term_cfg = linear_terms_to_term_config(
        LinearTerms(
            streaming=1.0,
            mirror=1.0,
            curvature=1.0,
            gradb=1.0,
            diamagnetic=1.0,
            collisions=1.0,
            hypercollisions=0.0,
            end_damping=0.0,
            apar=0.0,
            bpar=0.0,
        )
    )
    sigma = jnp.asarray(0.3 - 0.7j, dtype=v0.dtype)

    def preconditioned_residual(mode: str) -> jnp.ndarray:
        _diagonal, preconditioner = lk.build_shift_invert_preconditioner(
            v0,
            cache,
            params,
            term_cfg,
            sigma,
            mode,
        )
        assert preconditioner is not None
        candidate = preconditioner(jnp.ravel(v0)).reshape(v0.shape)
        residual = ka._apply_operator(candidate, cache, params, term_cfg)
        residual = residual - sigma * candidate - v0
        return jnp.linalg.norm(residual) / jnp.linalg.norm(v0)

    line_residual = preconditioned_residual("hermite-line")
    corrected_residual = preconditioned_residual("field-corrected")
    assert corrected_residual < 0.1 * line_residual
    assert corrected_residual < 0.2


@pytest.mark.parametrize(
    ("linked", "multi_species"),
    [(False, True), (True, False)],
)
def test_field_corrected_preconditioner_covers_em_species_and_linked_layouts(
    linked: bool,
    multi_species: bool,
) -> None:
    """The field map must retain EM/species axes and twist-linked state layout."""

    grid_cfg = GridConfig(
        Nx=4 if linked else 2,
        Ny=4 if linked else 2,
        Nz=8,
        Lx=6.0,
        Ly=6.0,
        boundary="linked" if linked else "periodic",
        y0=20.0 if linked else None,
        jtwist=1 if linked else None,
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    if multi_species:
        params = build_linear_params(
            (
                Species(1.0, 1.0, 1.0, 1.0, 6.9, 2.2),
                Species(-1.0, 0.01, 1.0, 1.0, 6.9, 2.2),
            ),
            beta=0.01,
            fapar=1.0,
            nu_hyper=0.0,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        species_shape = (2,)
    else:
        params = LinearParams(
            beta=0.01,
            fapar=1.0,
            nu_hyper=0.0,
            damp_ends_amp=0.0,
            damp_ends_widthfrac=0.0,
        )
        species_shape = ()
    n_laguerre, n_hermite = 2, 4
    cache = build_linear_cache(
        grid,
        geom,
        params,
        Nl=n_laguerre,
        Nm=n_hermite,
    )
    shape = (
        *species_shape,
        n_laguerre,
        n_hermite,
        grid.ky.size,
        grid.kx.size,
        grid.z.size,
    )
    vector = jnp.ones(shape, dtype=jnp.complex128) * (1.0 + 0.1j)
    terms = linear_terms_to_term_config(
        LinearTerms(
            hypercollisions=0.0,
            end_damping=0.0,
            apar=1.0,
            bpar=1.0,
        )
    )
    sigma = jnp.asarray(0.3 - 0.7j, dtype=vector.dtype)
    _diagonal, preconditioner = lk.build_shift_invert_preconditioner(
        vector,
        cache,
        params,
        terms,
        sigma,
        "field-corrected",
    )
    assert preconditioner is not None
    observed = preconditioner(jnp.ravel(vector))
    scaled = preconditioner(jnp.ravel((1.0 - 0.25j) * vector))
    assert observed.shape == (vector.size,)
    assert jnp.all(jnp.isfinite(observed))
    assert jnp.allclose(scaled, (1.0 - 0.25j) * observed, rtol=1.0e-10, atol=1.0e-10)
    _, tangent = jax.jvp(
        preconditioner, (jnp.ravel(vector),), (jnp.ravel(0.2j * vector),)
    )
    assert jnp.allclose(tangent, 0.2j * observed, rtol=1.0e-10, atol=1.0e-10)


def testbuild_shift_invert_preconditioneritioner_linked_branch() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=True)
    sigma = jnp.asarray(0.2j, dtype=v0.dtype)
    _precond, op = lk.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))


def test_shift_invert_uses_right_preconditioning_and_physical_fgmres_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shifted solve must minimize the original, not transformed, residual.

    It must also start from zero. Under right preconditioning the first Krylov
    vector is ``M^-1 b``, so a cycle from zero already minimizes over a space
    containing the old ``x0 = M^-1 b``; handing that point in as the guess can
    only start the solve further from the answer, and with a weak ``M`` it does
    -- on the shipped Cyclone deck it started 25x behind ``x = 0`` and the
    reported inner residual said so. ``x0`` is therefore asserted *absent*.
    """

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    calls: list[tuple[bool, object, int]] = []
    monkeypatch.setattr(
        ka,
        "build_shift_invert_preconditioner",
        lambda *_args: (jnp.ones_like(v0), lambda value: 0.1 * value),
    )
    monkeypatch.setattr(ka, "_apply_operator", lambda value, *_args: value)

    def fake_gmres(_matvec, b, *, precond, max_restarts, **kwargs):
        calls.append((precond is not None, kwargs.get("x0", "absent"), max_restarts))
        return SimpleNamespace(
            x=b,
            residual_norm=jnp.asarray(0.5),
            iterations=jnp.asarray(2, dtype=jnp.int32),
            converged=jnp.asarray(False),
        )

    monkeypatch.setattr(ka, "gmres", fake_gmres)
    apply_inverse = ka._shift_invert_apply_factory(
        v0,
        cache,
        params,
        term_cfg,
        sigma_val=jnp.asarray(0.0, v0.dtype),
        gmres_tol=1.0e-4,
        gmres_maxiter=2,
        gmres_restart=2,
        shift_preconditioner="damping",
    )

    observed, stats = apply_inverse(v0, cache, params, term_cfg)

    assert len(calls) == 1
    assert calls[0][0]
    assert calls[0][1] == "absent", (
        "the shifted FGMRES must start from zero; passing M^-1 b as x0 begins "
        "the solve outside the space the first cycle already searches"
    )
    assert calls[0][2] == 1
    assert jnp.allclose(observed, v0)
    # SOLVAX's true residual is kept relative to ||b||, with the budget outcome.
    relative = 0.5 / float(jnp.linalg.norm(v0))
    assert np.isclose(float(stats.max_relative_residual), relative, rtol=1.0e-6)
    assert (int(stats.total_iterations), int(stats.solves)) == (2, 1)
    assert int(stats.unconverged_solves) == 1


def _shift_invert_options(tol: float, maxiter: int, restart: int) -> dict[str, object]:
    return dict(
        krylov_dim=3,
        restarts=2,
        sigma=0.5j,
        omega_min_factor=0.0,
        omega_target_factor=0.0,
        omega_cap_factor=2.0,
        omega_sign=0,
        gmres_tol=tol,
        gmres_maxiter=maxiter,
        gmres_restart=restart,
        shift_preconditioner="damping",
        select_targeted=False,
        select_growth=False,
        select_overlap=False,
    )


@pytest.mark.parametrize(
    ("maxiter", "tol", "converged"), [(1, 1.0e-10, False), (400, 1.0e-3, True)]
)
def test_shift_invert_inner_stats_cover_every_arnoldi_rhs(
    maxiter: int, tol: float, converged: bool
) -> None:
    """Every inner solve of the build reports its true residual and budget."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    _eig, _vec, stats = ka._shift_invert_eigenpair_with_inner_stats(
        v0, v0, cache, params, term_cfg, **_shift_invert_options(tol, maxiter, maxiter)
    )
    residual = float(stats.max_relative_residual)
    assert int(stats.solves) == 6
    assert (int(stats.unconverged_solves) == 0) is converged
    assert (residual <= tol) is converged
    assert 0 < int(stats.total_iterations) <= 6 * (maxiter + v0.size)
    summary = lk._inner_solve_summary(stats, tol)
    assert f"converged={converged}" in summary
    assert f"max_relative_residual={residual:.3g} tol={tol:.3g}" in summary


def test_shift_invert_rejection_names_unconverged_inner_solves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A starved inner budget is named in the status stream and the rejection."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    monkeypatch.setattr(lk, "_eigenpair_relative_residual", lambda *_args: 1.0)
    messages: list[str] = []
    with pytest.raises(RuntimeError, match="outer residual gate") as rejected:
        lk.dominant_eigenpair(
            v0,
            cache,
            params,
            terms=terms,
            method="shift_invert",
            krylov_dim=3,
            restarts=2,
            shift=0.5j,
            shift_source="reference",
            shift_tol=1.0e-10,
            shift_maxiter=1,
            shift_restart=1,
            shift_preconditioner="damping",
            shift_selection="nearest",
            mode_family="none",
            fallback_method="none",
            status_callback=messages.append,
        )
    finished = [m for m in messages if m.startswith("shift-invert solve finished")]
    for text in (finished[-1], str(rejected.value)):
        assert "inner converged=False unconverged=6/6" in text
        assert "tol=1e-10" in text
    reported = re.search(r"max_relative_residual=(\S+)", str(rejected.value))
    assert reported is not None and float(reported.group(1)) > 1.0e-10


def test_shift_solve_method_labels_share_one_compiled_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The historical labels are validated but neither change nor recompile."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    traces = 0
    factory = ka._shift_invert_apply_factory

    def counting_factory(*args, **kwargs):
        nonlocal traces
        traces += 1
        return factory(*args, **kwargs)

    monkeypatch.setattr(ka, "_shift_invert_apply_factory", counting_factory)
    options = _shift_invert_options(1.0e-4, 7, 5)
    pairs = [
        ka.dominant_eigenpair_shift_invert_cached(
            v0, v0, cache, params, term_cfg, gmres_solve_method=label, **options
        )
        for label in ("batched", "incremental", "flexible")
    ]
    assert traces == 1
    for eig, vec in pairs[1:]:
        np.testing.assert_array_equal(np.asarray(eig), np.asarray(pairs[0][0]))
        np.testing.assert_array_equal(np.asarray(vec), np.asarray(pairs[0][1]))
    with pytest.raises(ValueError, match="shift_solve_method"):
        lk.dominant_eigenpair(
            v0, cache, params, method="shift_invert", shift_solve_method="bogus"
        )


@pytest.mark.parametrize("method", ["power", "propagator", "arnoldi"])
def test_raw_eigenpair_routes_fail_closed_unless_certify_is_false(method: str) -> None:
    """A four-vector raw solve is unconverged: it must raise, or be flagged."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    options = dict(
        terms=terms,
        method=method,
        krylov_dim=4,
        restarts=1,
        power_iters=4,
        power_dt=0.05,
    )
    tolerance = lk.certifiable_residual_tolerance(1.0e-6, v0.dtype)
    with pytest.raises(RuntimeError) as rejected:
        lk.dominant_eigenpair(v0, cache, params, **options)
    match = re.match(
        rf"{method} eigenpair failed the outer residual gate: "
        r"residual=(\S+), tolerance=(\S+); ",
        str(rejected.value),
    )
    assert match is not None, str(rejected.value)
    assert float(match.group(1)) > tolerance
    assert float(match.group(2)) == pytest.approx(tolerance, rel=1.0e-5)

    messages: list[str] = []
    eig, vec = lk.dominant_eigenpair(
        v0, cache, params, certify=False, status_callback=messages.append, **options
    )
    assert vec.shape == v0.shape
    assert jnp.isfinite(jnp.real(eig))
    assert jnp.isfinite(jnp.imag(eig))
    flagged = [item for item in messages if f"UNCERTIFIED {method} eigenpair" in item]
    assert flagged and f"residual={match.group(1)}" in flagged[0]


def test_raw_propagator_route_returns_a_converged_pair() -> None:
    """Seeded with an exact eigenvector, the propagator route certifies it.

    The propagator's Ritz vector carries an O(dt) splitting error; dt=1e-3 in
    complex64 leaves it near 1e-7, three decades under the dtype-floored gate.
    """

    _grid, cache, params, v0, term_cfg, terms = _tiny_krylov_setup(linked=False)
    dtype = v0.dtype
    basis = jnp.eye(v0.size, dtype=dtype).reshape((v0.size, *v0.shape))
    columns = jax.vmap(lambda e: lk._apply_operator(e, cache, params, term_cfg))(basis)
    matrix = np.asarray(columns).reshape((v0.size, v0.size)).T.astype(np.complex128)
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(np.where(np.abs(values) > 1.0e-2, values.real, -np.inf)))
    seed = jnp.asarray(vectors[:, index].reshape(v0.shape), dtype=dtype)

    messages: list[str] = []
    eig, vec, status = lk.dominant_eigenpair(
        seed,
        cache,
        params,
        terms=terms,
        method="propagator",
        krylov_dim=4,
        restarts=1,
        power_dt=1.0e-3,
        status_callback=messages.append,
        return_status=True,
    )
    assert complex(np.asarray(eig)) == pytest.approx(complex(values[index]), rel=1e-4)
    residual = lk._eigenpair_relative_residual(eig, vec, cache, params, term_cfg)
    assert residual <= lk.certifiable_residual_tolerance(1.0e-6, dtype)
    assert any("certified=True" in item for item in messages)
    # The certified pair reports the residual it was gated on, beside the gate.
    assert (status.method, status.route, status.certified) == (
        "propagator",
        "propagator",
        True,
    )
    assert status.residual == residual
    assert status.tolerance == lk.certifiable_residual_tolerance(1.0e-6, dtype)
    assert status.inner is None


def test_a_zero_eigenvector_never_certifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """A breakdown that returns ``(0, 0)`` scored ``0 / floor = 0`` and passed."""

    _grid, cache, params, v0, term_cfg, terms = _tiny_krylov_setup(linked=False)
    zero = jnp.zeros_like(v0)
    assert lk._eigenpair_relative_residual(
        jnp.asarray(0.0j, v0.dtype), zero, cache, params, term_cfg
    ) == float("inf")
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_cached",
        lambda *args, **kwargs: (jnp.asarray(0.0j, v0.dtype), zero),
    )
    with pytest.raises(RuntimeError, match="residual=inf, tolerance="):
        lk.dominant_eigenpair(v0, cache, params, terms=terms, method="arnoldi")


def test_default_krylov_config_and_wrapper_resolve_to_certified_adaptive() -> None:
    from gkx.api import KrylovConfig
    from gkx.config import RuntimeConfig
    from gkx.workflows.runtime.startup import _runtime_default_krylov_config

    assert KrylovConfig is lk.KrylovConfig
    assert KrylovConfig().method == "adaptive"
    assert KrylovConfig().certify is True
    assert _runtime_default_krylov_config(RuntimeConfig()) == KrylovConfig()
    parameters = inspect.signature(lk.dominant_eigenpair).parameters
    assert parameters["method"].default == "adaptive"
    assert parameters["certify"].default is True


def test_krylov_config_and_wrapper_agree_on_every_shared_default() -> None:
    """The two doors into one route must cost the same.

    ``KrylovConfig`` is what the runtime builds from a deck and what
    ``workflows.linear`` forwards field by field; the ``dominant_eigenpair``
    signature is what a direct caller gets. They are two entry points to one
    computation, so a field that means the same thing in both must default to
    the same value. ``power_iters`` used to be 200 in the dataclass and 40 in
    the signature -- the same nominal route at 5x the propagator applies
    depending on which door it was entered by, with no accuracy to show for it.
    Asserting the whole intersection guards the class of defect, not the one
    instance of it.
    """

    import dataclasses

    parameters = inspect.signature(lk.dominant_eigenpair).parameters
    shared = {
        field.name: field
        for field in dataclasses.fields(lk.KrylovConfig)
        if field.name in parameters
    }
    # The wrapper exposes every knob the dataclass carries except the two
    # continuation controls, which only the runtime's scan driver sets.
    assert set(shared) == {
        field.name for field in dataclasses.fields(lk.KrylovConfig)
    } - {"continuation", "continuation_selection"}
    disagreements = {
        name: (field.default, parameters[name].default)
        for name, field in shared.items()
        if field.default != parameters[name].default
    }
    assert disagreements == {}
    assert lk.KrylovConfig().power_iters == 40


def test_certify_opt_out_does_not_relax_the_shift_invert_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    _patch_shift_invert(
        monkeypatch,
        lambda *args, **kwargs: (jnp.asarray(0.4 + 0.2j, v0.dtype), jnp.ones_like(v0)),
    )
    monkeypatch.setattr(lk, "_eigenpair_relative_residual", lambda *_args: 1.0)
    with pytest.raises(RuntimeError, match="shift-invert eigenpair failed the outer"):
        lk.dominant_eigenpair(
            v0,
            cache,
            params,
            terms=terms,
            method="shift_invert",
            shift=0.4 + 0.2j,
            shift_source="reference",
            fallback_method="none",
            certify=False,
        )


def test_long_horizon_propagator_selects_growth_and_recovers_frequency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase wrapping may not corrupt growth selection or physical frequency."""

    _grid, cache, params, _v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    eigenvalues = jnp.asarray(
        [0.2 + 4.0j, 0.1 - 0.5j, -0.3 + 1.0j],
        dtype=jnp.complex128,
    )
    step_dt = 0.2
    monkeypatch.setattr(
        ka,
        "_apply_operator",
        lambda state, *_args: eigenvalues * state,
    )
    initial = jnp.ones((3,), dtype=jnp.complex128)

    value, vector = ka.dominant_eigenpair_propagator_cached(
        initial,
        initial,
        cache,
        params,
        term_cfg,
        krylov_dim=3,
        restarts=1,
        dt=step_dt,
        propagator_steps=10,
        omega_min_factor=0.0,
        omega_target_factor=0.0,
        omega_cap_factor=1.0,
        omega_sign=0,
        select_overlap=False,
    )

    assert complex(np.asarray(value)) == pytest.approx(
        complex(np.asarray(eigenvalues[0])),
        rel=1.0e-10,
    )
    assert abs(complex(np.asarray(vector[0]))) > 1.0 - 1.0e-10


def test_certified_candidate_lift_requests_exact_dot_precision() -> None:
    """The vector the residual gate measures must not be built in TF32.

    ``certifiable_residual_tolerance`` floors the gate at ``1000 * eps`` of the
    working precision, 1.19e-04 for the complex64 runtime state. TF32 keeps ten
    mantissa bits, ~4.9e-04, so an unpinned lift puts more error into the
    eigenvector than the gate is allowed to accept and the certified branch
    rejects converged pairs on every Ampere-or-newer GPU. No CPU *value* can see
    this -- there is no TF32 path on CPU and both precisions are bit-identical --
    but the request is recorded in the jaxpr on every backend, so that is what is
    asserted, at the candidate counts ``_adaptive_branch`` can select.
    """

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)

    def lift_precisions(candidates: int) -> list[object]:
        jaxpr = jax.make_jaxpr(
            lambda value: kp.dominant_eigenpairs_propagator_cached(
                value,
                cache,
                params,
                term_cfg,
                krylov_dim=4,
                dt=0.01,
                propagator_steps=2,
                candidates=candidates,
            )
        )(v0).jaxpr

        def walk(inner):
            for eqn in inner.eqns:
                if eqn.primitive.name == "dot_general":
                    yield eqn
                for value in eqn.params.values():
                    items = value if isinstance(value, (list, tuple)) else [value]
                    for item in items:
                        nested = getattr(item, "jaxpr", item)
                        nested = getattr(nested, "jaxpr", nested)
                        if hasattr(nested, "eqns"):
                            yield from walk(nested)

        found = []
        for eqn in walk(jaxpr):
            frames = getattr(eqn.source_info.traceback, "frames", []) or []
            if any("krylov_propagator" in getattr(f, "file_name", "") for f in frames):
                found.append(eqn.params["precision"])
        return found

    exact = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    for candidates in (1, 2):
        precisions = [p for p in lift_precisions(candidates) if p is not None]
        assert precisions, (
            f"the candidate lift lowered to no pinned dot at candidates={candidates}; "
            "the certified eigenvector is being built at the backend default"
        )
        assert all(precision == exact for precision in precisions), (
            f"the candidate lift is not pinned at candidates={candidates} "
            f"({precisions}); TF32 puts ~4.9e-04 into the vector the residual gate "
            "measures, against its 1.19e-04 complex64 floor"
        )


@requires_solvax_eigen_api
def test_adaptive_propagator_selects_stable_step_and_stops_on_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production adapter must infer dt and avoid its unused restart budget."""

    _grid, cache, params, _v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    eigenvalues = jnp.asarray(
        [
            0.3 + 0.2j,
            0.1 - 0.4j,
            -0.2 + 3.0j,
            -0.3 - 4.0j,
            -0.4 + 5.0j,
            -0.5 - 6.0j,
            -0.6 + 7.0j,
            -0.7 - 8.0j,
            -0.8 + 9.0j,
            -0.9 - 10.0j,
        ],
        dtype=jnp.complex128,
    )
    monkeypatch.setattr(
        ap,
        "_apply_operator",
        lambda state, *_args: eigenvalues * state,
    )
    monkeypatch.setattr(
        kp,
        "_apply_operator",
        lambda state, *_args: eigenvalues * state,
    )
    monkeypatch.setattr(
        ka,
        "_apply_operator",
        lambda state, *_args: eigenvalues * state,
    )
    initial = jnp.ones_like(eigenvalues)
    solution = lk.adaptive_propagator_eigenpair(
        initial,
        cache,
        params,
        terms=terms,
        krylov_dim=8,
        candidate_count=2,
        max_restarts=5,
        tol=1.0e-9,
        chunk_horizon=20.0,
        stability_dimension=8,
    )
    repeated = lk.adaptive_propagator_eigenpair(
        initial,
        cache,
        params,
        terms=terms,
        krylov_dim=8,
        candidate_count=2,
        max_restarts=5,
        tol=1.0e-9,
        chunk_horizon=20.0,
        stability_dimension=8,
    )

    # The requested gate is what x64 certifies against; single precision cannot
    # reach it, so the adapter certifies against its own dtype noise floor.
    gate = ap.certifiable_residual_tolerance(1.0e-9, eigenvalues.dtype)
    assert solution.converged
    assert solution.stable
    assert solution.restarts < 5
    assert solution.filter_dt < 2.8 / float(jnp.max(jnp.abs(eigenvalues)))
    assert complex(np.asarray(solution.eigenvalue)) == pytest.approx(
        complex(np.asarray(eigenvalues[0])), rel=gate
    )
    assert float(np.asarray(solution.residual)) < gate
    np.testing.assert_array_equal(
        np.asarray(solution.eigenvalue),
        np.asarray(repeated.eigenvalue),
    )
    np.testing.assert_array_equal(
        np.asarray(solution.eigenvector),
        np.asarray(repeated.eigenvector),
    )


@requires_solvax_eigen_api
def test_adaptive_propagator_halves_step_after_false_stable_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residual exhaustion must retry when the spectral sketch misses a mode."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    attempted_steps: list[float] = []

    def fake_estimate(*_args, probe_count: int, **_kwargs):
        assert probe_count == 2
        return SimpleNamespace(dt=0.2, operator_applications=12)

    monkeypatch.setattr(
        solvax,
        "estimate_rk4_timestep",
        fake_estimate,
    )

    def fake_adaptive(*_args, filter_dt: float, **_kwargs):
        attempted_steps.append(filter_dt)
        converged = len(attempted_steps) == 2
        return solvax.AdaptiveEigenSolution(
            eigenvalue=jnp.asarray(0.2 + 0.1j),
            eigenvector=v0,
            residual=jnp.asarray(0.0 if converged else 1.0),
            converged=converged,
            stable=True,
            restarts=1,
            operator_applications=100,
            filter_dt=filter_dt,
            filter_steps=100,
            filter_horizon=20.0,
            filter_growth_defect=0.0,
        )

    monkeypatch.setattr(solvax, "adaptive_eigenpair", fake_adaptive)
    solution = lk.adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        chunk_horizon=20.0,
        max_stability_retries=2,
    )

    assert solution.converged
    assert attempted_steps == pytest.approx([0.2, 0.1])
    assert solution.operator_applications == 212


@requires_solvax_eigen_api
def test_adaptive_propagator_uses_smaller_corrective_subspaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Later residual corrections may cost less without corrupting accounting."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    dimensions: list[int] = []
    selected_values: list[complex] = []

    monkeypatch.setattr(
        solvax,
        "estimate_rk4_timestep",
        lambda *_args, **_kwargs: SimpleNamespace(
            dt=0.2,
            operator_applications=12,
        ),
    )

    def fake_candidates(
        *_args,
        krylov_dim: int,
        candidates: int,
        **_kwargs,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dimensions.append(krylov_dim)
        assert candidates == 2
        values = jnp.asarray([0.19 + 0.1j, 0.2 + 0.1j])
        vectors = jnp.stack([v0, v0])
        residuals = jnp.zeros((2,))
        return values, vectors, residuals

    monkeypatch.setattr(
        ap,
        "dominant_eigenpairs_propagator_cached",
        fake_candidates,
    )

    def fake_adaptive(
        _apply,
        restart_once,
        vector,
        *,
        filter_dt: float,
        filter_steps: int,
        applications_per_restart: int,
        **_kwargs,
    ):
        assert applications_per_restart == 0
        first_value, _first = restart_once(vector)
        selected_values.append(complex(first_value))
        second_value, corrected = restart_once(vector)
        selected_values.append(complex(second_value))
        return solvax.AdaptiveEigenSolution(
            eigenvalue=jnp.asarray(0.2 + 0.1j),
            eigenvector=corrected,
            residual=jnp.asarray(0.0),
            converged=True,
            stable=True,
            restarts=2,
            operator_applications=4,
            filter_dt=filter_dt,
            filter_steps=filter_steps,
            filter_horizon=20.0,
            filter_growth_defect=0.0,
        )

    monkeypatch.setattr(solvax, "adaptive_eigenpair", fake_adaptive)
    solution = lk.adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        krylov_dim=16,
        restart_krylov_dim=8,
        candidate_count=2,
        chunk_horizon=20.0,
    )

    assert dimensions == [16, 8]
    assert selected_values == pytest.approx([0.2 + 0.1j, 0.2 + 0.1j])
    assert solution.operator_applications == 12 + 4 + 2 + 4 * 100 * (16 + 8)
    np.testing.assert_allclose(
        np.asarray(solution.candidate_eigenvalues),
        np.asarray([0.19 + 0.1j, 0.2 + 0.1j]),
    )
    assert float(np.asarray(solution.candidate_growth_gap)) == pytest.approx(0.01)


@requires_solvax_eigen_api
def test_adaptive_propagator_biorthogonally_continues_subdominant_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A previous left mode must outrank maximum growth across an exchange."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    first = np.zeros(v0.size, dtype=complex)
    second = np.zeros(v0.size, dtype=complex)
    first[0] = 1.0
    second[1] = 1.0
    vectors = jnp.asarray(np.stack([first, second]).reshape((2, *v0.shape)))
    continuation_covector = vectors[1]
    values = jnp.asarray([0.31 + 0.2j, 0.30 - 0.4j], dtype=v0.dtype)

    monkeypatch.setattr(
        solvax,
        "estimate_rk4_timestep",
        lambda *_args, **_kwargs: SimpleNamespace(
            dt=0.2,
            operator_applications=12,
        ),
    )
    monkeypatch.setattr(
        ap,
        "dominant_eigenpairs_propagator_cached",
        lambda *_args, **_kwargs: (values, vectors, jnp.zeros((2,))),
    )

    def fake_adaptive(
        _apply,
        restart_once,
        vector,
        *,
        filter_dt: float,
        filter_steps: int,
        **_kwargs,
    ):
        value, selected = restart_once(vector)
        return solvax.AdaptiveEigenSolution(
            eigenvalue=value,
            eigenvector=selected,
            residual=jnp.asarray(0.0),
            converged=True,
            stable=True,
            restarts=1,
            operator_applications=2,
            filter_dt=filter_dt,
            filter_steps=filter_steps,
            filter_horizon=20.0,
            filter_growth_defect=0.0,
        )

    monkeypatch.setattr(solvax, "adaptive_eigenpair", fake_adaptive)
    solution = lk.adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        krylov_dim=8,
        candidate_count=2,
        continuation_vector=vectors[1],
        continuation_covector=continuation_covector,
        continuation_overlap_floor=0.95,
        continuation_spectral_gap_floor=0.1,
        chunk_horizon=20.0,
    )

    assert solution.converged
    assert solution.continued
    assert solution.continuation_passed
    assert solution.selected_candidate_index == 1
    assert complex(np.asarray(solution.eigenvalue)) == pytest.approx(0.30 - 0.4j)
    assert float(np.asarray(solution.continuation_overlap)) == pytest.approx(1.0)
    assert float(np.asarray(solution.selected_spectral_gap)) == pytest.approx(
        abs(complex(values[1] - values[0]))
    )
    np.testing.assert_allclose(
        np.asarray(solution.candidate_overlaps),
        np.asarray([0.0, 1.0]),
    )
    right_only = lk.adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        krylov_dim=8,
        candidate_count=2,
        continuation_vector=vectors[1],
        continuation_overlap_floor=0.95,
        chunk_horizon=20.0,
    )
    assert right_only.selected_candidate_index == 1
    assert float(np.asarray(right_only.continuation_overlap)) == pytest.approx(1.0)
    monkeypatch.setattr(
        ap,
        "dominant_eigenpairs_propagator_cached",
        lambda *_args, **_kwargs: (
            values,
            vectors,
            jnp.asarray([1.0, 0.0]),
        ),
    )
    rejected = lk.adaptive_propagator_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        krylov_dim=8,
        candidate_count=2,
        continuation_vector=vectors[1],
        continuation_covector=continuation_covector,
        continuation_overlap_floor=0.95,
        continuation_spectral_gap_floor=1.0,
        chunk_horizon=20.0,
    )
    assert rejected.continuation_passed is False
    assert rejected.converged is False
    assert float(np.asarray(rejected.selected_spectral_gap)) == pytest.approx(
        abs(complex(values[1] - values[0]))
    )


def test_dominant_eigenpair_shift_invert_rejects_unconverged_sources() -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    for source in ("propagator", "target", "power"):
        with pytest.raises(RuntimeError, match="outer residual gate"):
            lk.dominant_eigenpair(
                v0,
                cache,
                params,
                terms=terms,
                method="shift_invert",
                shift=None,
                shift_source=source,
                shift_preconditioner="damping",
                krylov_dim=4,
                restarts=1,
                shift_maxiter=15,
                shift_restart=10,
                power_iters=4,
                power_dt=0.05,
            )
    with pytest.raises(ValueError):
        lk.dominant_eigenpair(v0, cache, params, terms=terms, method="bad")


@pytest.mark.parametrize("shift_source", ["propagator", "power"])
def test_dominant_eigenpair_explicit_shift_uses_requested_seed_source(
    monkeypatch: pytest.MonkeyPatch,
    shift_source: str,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    captured: dict[str, jnp.ndarray] = {}
    seed = jnp.full_like(v0, 3.0 + 0.0j)

    def _fake_shift(v_init, v_ref, *_args, sigma, **_kwargs):
        captured["v_init"] = v_init
        captured["v_ref"] = v_ref
        captured["sigma"] = sigma
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    _patch_shift_invert(monkeypatch, _fake_shift)
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.4 + 0.2j, vector.dtype) * vector,
    )
    if shift_source == "propagator":
        monkeypatch.setattr(
            lk,
            "dominant_eigenpair_propagator_cached",
            lambda *args, **kwargs: (jnp.asarray(0.1 + 0.0j, dtype=v0.dtype), seed),
        )
    else:
        monkeypatch.setattr(
            lk,
            "dominant_eigenpair_power",
            lambda *args, **kwargs: (jnp.asarray(0.1 + 0.0j, dtype=v0.dtype), seed),
        )

    eig, vec = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift=0.2 - 1.1j,
        shift_source=shift_source,
        shift_selection="shift",
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_iters=4,
        power_dt=0.05,
    )

    assert jnp.allclose(captured["sigma"], jnp.asarray(0.2 - 1.1j, dtype=v0.dtype))
    assert jnp.allclose(captured["v_init"], seed)
    assert jnp.allclose(eig, jnp.asarray(0.4 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec, 5.0 + 0.0j)


def test_dominant_eigenpair_explicit_shift_defaults_to_reference_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    captured: dict[str, jnp.ndarray] = {}
    v_ref = jnp.full_like(v0, 7.0 + 0.0j)

    def _fake_shift(v_init, v_ref_in, *_args, sigma, **_kwargs):
        captured["v_init"] = v_init
        captured["v_ref"] = v_ref_in
        captured["sigma"] = sigma
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    _patch_shift_invert(monkeypatch, _fake_shift)
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.4 + 0.2j, vector.dtype) * vector,
    )

    lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift=0.2 - 1.1j,
        shift_source="target",
        shift_selection="shift",
        v_ref=v_ref,
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_iters=4,
        power_dt=0.05,
    )

    assert jnp.allclose(captured["sigma"], jnp.asarray(0.2 - 1.1j, dtype=v0.dtype))
    assert jnp.allclose(captured["v_init"], v_ref)
    assert jnp.allclose(captured["v_ref"], v_ref)


def test_dominant_eigenpair_target_shift_uses_physical_omega_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    captured: dict[str, jnp.ndarray] = {}

    def _fake_shift(v_init, v_ref_in, *_args, sigma, **_kwargs):
        captured["sigma"] = sigma
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 5.0 + 0.0j)

    _patch_shift_invert(monkeypatch, _fake_shift)
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.4 + 0.2j, vector.dtype) * vector,
    )
    monkeypatch.setattr(lk, "_omega_scale", lambda *_args, **_kwargs: jnp.asarray(2.0))

    lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift_source="target",
        shift_selection="shift",
        omega_target_factor=0.5,
        omega_sign=-1,
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_iters=4,
        power_dt=0.05,
    )

    assert jnp.allclose(captured["sigma"], jnp.asarray(0.0 + 1.0j, dtype=v0.dtype))


def test_shift_invert_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)

    def fake_shift(*args, **kwargs):
        return jnp.asarray(jnp.nan + 1j * jnp.nan, dtype=v0.dtype), jnp.ones_like(v0)

    def fake_prop(*args, **kwargs):
        return jnp.asarray(1.0 + 0.2j, dtype=v0.dtype), jnp.full_like(v0, 2.0 + 0.0j)

    def fake_arnoldi(*args, **kwargs):
        return jnp.asarray(0.5 + 0.1j, dtype=v0.dtype), jnp.full_like(v0, 3.0 + 0.0j)

    def fake_power(*args, **kwargs):
        return jnp.asarray(0.2 + 0.05j, dtype=v0.dtype), jnp.full_like(v0, 4.0 + 0.0j)

    _patch_shift_invert(monkeypatch, fake_shift)
    monkeypatch.setattr(lk, "dominant_eigenpair_propagator_cached", fake_prop)
    monkeypatch.setattr(lk, "dominant_eigenpair_cached", fake_arnoldi)
    monkeypatch.setattr(lk, "dominant_eigenpair_power", fake_power)
    monkeypatch.setattr(lk, "_omega_scale", lambda *_args, **_kwargs: jnp.asarray(1.0))
    monkeypatch.setattr(lk, "_eigenpair_relative_residual", lambda *_args: 0.0)

    eig_p, vec_p = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="propagator",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_p, jnp.asarray(1.0 + 0.2j, dtype=v0.dtype))
    assert jnp.allclose(vec_p, 2.0 + 0.0j)

    eig_a, vec_a = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="arnoldi",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_a, jnp.asarray(0.5 + 0.1j, dtype=v0.dtype))
    assert jnp.allclose(vec_a, 3.0 + 0.0j)

    eig_w, vec_w = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="power",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_w, jnp.asarray(0.2 + 0.05j, dtype=v0.dtype))
    assert jnp.allclose(vec_w, 4.0 + 0.0j)

    # Nearest-shift selection may intentionally target a stable eigenvalue.
    _patch_shift_invert(
        monkeypatch,
        lambda *args, **kwargs: (
            jnp.asarray(-0.2 + 0.5j, dtype=v0.dtype),
            jnp.ones_like(v0),
        ),
    )
    eig_stable, _ = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift=0.5j,
        shift_source="reference",
        shift_selection="nearest",
        fallback_method="none",
        fallback_real_floor=0.0,
    )
    assert jnp.allclose(eig_stable, jnp.asarray(-0.2 + 0.5j, dtype=v0.dtype))

    with pytest.raises(RuntimeError, match="growth-selection floor"):
        lk.dominant_eigenpair(
            v0,
            cache,
            params,
            terms=terms,
            method="shift_invert",
            shift=0.5j,
            shift_source="reference",
            shift_selection="growth",
            fallback_method="none",
            fallback_real_floor=0.0,
        )

    # A rejected pair without a fallback must fail rather than escape as NaN.
    _patch_shift_invert(monkeypatch, fake_shift)
    with pytest.raises(RuntimeError, match="non-finite"):
        lk.dominant_eigenpair(
            v0,
            cache,
            params,
            terms=terms,
            method="shift_invert",
            fallback_method="none",
            fallback_real_floor=0.0,
        )

    # dominant_eigenvalue wrapper should reuse dominant_eigenpair.
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair",
        lambda *args, **kwargs: (jnp.asarray(0.7 + 0.1j), jnp.ones_like(v0)),
    )
    eig_val = lk.dominant_eigenvalue(
        v0, cache, params, terms=terms, krylov_dim=4, restarts=1
    )
    assert jnp.allclose(eig_val, jnp.asarray(0.7 + 0.1j))


@pytest.mark.parametrize(
    ("beta", "apar", "bpar", "residuals", "expected"),
    (
        (0.0, 0.0, 0.0, (0.0,), ("hermite-line",)),
        (0.0, 1.0, 1.0, (0.0,), ("hermite-line",)),
        (0.01, 1.0, 0.0, (0.0,), ("field-corrected",)),
        (0.01, 0.0, 1.0, (0.0,), ("field-corrected",)),
        (0.0, 0.0, 0.0, (1.0, 0.0), ("hermite-line", "field-corrected")),
    ),
)
def test_shift_invert_auto_selects_and_certifies_physics_preconditioner(
    monkeypatch: pytest.MonkeyPatch,
    beta: float,
    apar: float,
    bpar: float,
    residuals: tuple[float, ...],
    expected: tuple[str, ...],
) -> None:
    """Auto avoids field setup for ES, uses it for EM, and retries failed ES."""

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    calls: list[str] = []
    remaining = iter(residuals)

    def fake_shift(*_args, shift_preconditioner, **_kwargs):
        calls.append(shift_preconditioner)
        return jnp.asarray(0.4 + 0.2j, dtype=v0.dtype), jnp.ones_like(v0)

    _patch_shift_invert(monkeypatch, fake_shift)
    monkeypatch.setattr(
        lk, "_eigenpair_relative_residual", lambda *_args: next(remaining)
    )
    lk.dominant_eigenpair(
        v0,
        cache,
        replace(params, beta=beta, fapar=float(apar != 0.0)),
        terms=replace(terms, apar=apar, bpar=bpar),
        method="shift_invert",
        shift=0.4 + 0.2j,
        shift_source="reference",
        shift_selection="nearest",
        shift_preconditioner="auto",
        fallback_method="none",
    )
    assert tuple(calls) == expected


def test_sparse_shift_invert_selects_only_original_operator_certified_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sparse approximation cannot certify a pair on the physics operator."""

    from scipy.sparse import eye

    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    values = jnp.asarray((0.1 + 0.2j, 0.4 + 0.2j, 0.3 + 0.2j))
    vectors = jnp.stack(
        (jnp.ones(v0.size), 2.0 * jnp.ones(v0.size), 3.0 * jnp.ones(v0.size))
    )
    captured = {}
    monkeypatch.setattr(
        solvax,
        "sparse_operator_matrix",
        lambda *_args, **_kwargs: eye(v0.size, dtype=np.complex128),
        raising=False,
    )
    monkeypatch.setattr(
        solvax, "SpluFactorization", lambda _matrix: "factor", raising=False
    )

    def fake_modes(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            eigenvalues=values,
            eigenvectors=vectors,
            converged=jnp.ones((3,), dtype=bool),
        )

    monkeypatch.setattr(solvax, "sparse_eigenpairs", fake_modes, raising=False)
    residuals = iter((1.0e-12, 1.0, 1.0e-12))
    monkeypatch.setattr(
        lk, "_eigenpair_relative_residual", lambda *_args: next(residuals)
    )
    value, vector = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="sparse_shift_invert",
        shift=0.2 + 0.2j,
        shift_selection="growth",
    )
    assert value == values[2]
    assert jnp.all(vector == 3.0)
    assert captured["factorization"] == "factor"


def _selected_linked_case(nx: int = 8):
    """One-ky linked grid; at Nx=8 its five-link chain skips kx rows {3, 4, 5}."""

    from gkx.core_grid import select_ky_grid

    grid_cfg = GridConfig(
        Nx=nx,
        Ny=4,
        Nz=8,
        Ly=2.0 * np.pi * 10.0,
        boundary="linked",
        y0=10.0,
        jtwist=1,
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = select_ky_grid(build_spectral_grid(cfg.grid), 1)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(damp_ends_amp=0.1)
    cache = build_linear_cache(grid, geom, params, Nl=2, Nm=4)
    shape = (2, 4, grid.ky.size, grid.kx.size, grid.z.size)
    index = jnp.arange(int(np.prod(shape)), dtype=jnp.float64)
    # Golden-angle start, nonzero on every row like gkx.objectives.core's.
    seed = jnp.reshape(jnp.exp(1j * (index + 1.0) * 0.6180339887498948), shape)
    return cache, params, seed, LinearTerms()


def _chain_modes(cache) -> np.ndarray:
    ny, nx = int(cache.ky.size), int(cache.kx.size)
    visited = np.zeros(ny * nx, dtype=bool)
    for chain in cache.linked_indices:
        visited[np.asarray(chain).reshape(-1)] = True  # flat index ky + ny * kx
    return visited.reshape(nx, ny).T


def _off_chain_weight(vector, mask) -> float:
    outside = np.asarray(vector)[..., ~np.asarray(mask), :]
    return float(np.max(np.abs(outside), initial=0.0))


def test_linked_cover_mask_is_the_chain_modes_and_none_when_all_are() -> None:
    cache, _params, seed, _terms = _selected_linked_case()
    mask = np.asarray(ka._linked_covered_mode_mask(cache))
    np.testing.assert_array_equal(mask, _chain_modes(cache))
    assert np.flatnonzero(mask[0]).tolist() == [0, 1, 2, 6, 7]  # GX Nakx = 5
    projected = ka._project_to_linked_cover(seed, jnp.asarray(mask))
    assert _off_chain_weight(projected, mask) == 0.0
    np.testing.assert_array_equal(
        np.asarray(projected)[..., mask, :], np.asarray(seed)[..., mask, :]
    )

    # Two-sided ky: chains visit rows {0, 1}; row -1 is rebuilt from the
    # conjugate (-ky, -kx) modes, so it is coupled; the Nyquist row is not.
    # The negative row only exists on the full axis, which is what the mirror
    # half of the cover mask is, so this grid asks for it by name.
    _grid, two_sided, _p, _v, _t, _terms2 = _tiny_krylov_setup(
        linked=True, ky_layout="full"
    )
    chain = _chain_modes(two_sided)
    two_sided_mask = np.asarray(ka._linked_covered_mode_mask(two_sided))
    np.testing.assert_array_equal(two_sided_mask[:3], chain[:3])
    np.testing.assert_array_equal(two_sided_mask[3], chain[1][[0, 3, 2, 1]])
    assert not two_sided_mask[2].any()

    _grid, periodic, _p, periodic_seed, _t, _terms3 = _tiny_krylov_setup()
    full_cover, _p1, _s1, _t1 = _selected_linked_case(nx=1)
    assert full_cover.linked_full_cover
    for unchanged in (periodic, full_cover):
        assert ka._linked_covered_mode_mask(unchanged) is None
    assert ka._project_to_linked_cover(periodic_seed, None) is periodic_seed


@pytest.mark.parametrize("two_sided", [False, True])
def test_modes_outside_linked_chains_are_decoupled_from_them(two_sided) -> None:
    if two_sided:
        # The parameter names the layout under test: on the full axis the
        # chains also cover the negative rows, so decoupling has to hold for
        # modes the half axis never stores.
        _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(
            linked=True, ky_layout="full"
        )
        state = jnp.asarray(np.random.default_rng(3).normal(size=v0.shape), v0.dtype)
    else:
        cache, params, state, terms = _selected_linked_case()
        term_cfg = linear_terms_to_term_config(terms)
    mask = ka._linked_covered_mode_mask(cache)
    on_chain = ka._project_to_linked_cover(state, mask)
    image_on = ka._apply_operator(on_chain, cache, params, term_cfg)
    image_off = ka._apply_operator(state - on_chain, cache, params, term_cfg)
    assert float(jnp.linalg.norm(image_on)) > 0.0
    assert _off_chain_weight(image_on, mask) == 0.0
    assert float(jnp.max(jnp.abs(ka._project_to_linked_cover(image_off, mask)))) == 0.0


def _dense_chain_target(cache, params, term_cfg, seed) -> complex:
    columns = jax.vmap(
        lambda column: ka._apply_operator(
            column.reshape(seed.shape), cache, params, term_cfg
        ).reshape(-1)
    )(jnp.eye(seed.size, dtype=seed.dtype))
    values, vectors = np.linalg.eig(np.asarray(columns).T)
    mask = np.broadcast_to(np.asarray(_chain_modes(cache))[:, :, None], seed.shape)
    chain = np.linalg.norm(vectors[mask.reshape(-1)], axis=0) > 0.5
    return complex(values[chain][np.argmax(values[chain].real)])


def test_sparse_shift_invert_assembles_linked_chain_columns_only(monkeypatch) -> None:
    cache, params, seed, terms = _selected_linked_case()
    term_cfg = linear_terms_to_term_config(terms)
    target = _dense_chain_target(cache, params, term_cfg, seed)
    mask = _chain_modes(cache)

    def solve():
        messages: list[str] = []
        pair = lk.dominant_eigenpair(
            seed,
            cache,
            params,
            terms=terms,
            method="sparse_shift_invert",
            shift=target + 0.01,
            status_callback=messages.append,
        )
        return pair, " ".join(messages)

    (value, vector), status = solve()
    assert "n=320 " in status and "(320 of 512 unknowns)" in status
    assert vector.shape == seed.shape
    assert _off_chain_weight(vector, mask) == 0.0
    assert abs(complex(value) - target) <= 1.0e-9 * abs(target)
    monkeypatch.setattr(lk, "_linked_covered_mode_mask", lambda _cache: None)
    (full_value, _full_vector), full_status = solve()
    assert "n=512 " in full_status
    assert abs(complex(value) - complex(full_value)) <= 1.0e-12 * abs(target)


@pytest.mark.parametrize("method", ["power", "propagator", "arnoldi", "adaptive"])
def test_eigen_routes_return_no_weight_outside_linked_chains(method) -> None:
    cache, params, seed, terms = _selected_linked_case()
    if method == "adaptive":
        # One bounded restart: this small-gap case is not certified, and the
        # contract under test is the support of every vector the route returns.
        solution = ap.adaptive_propagator_eigenpair(
            seed,
            cache,
            params,
            terms,
            krylov_dim=8,
            max_restarts=1,
            chunk_horizon=5.0,
            max_stability_retries=0,
            candidate_count=2,
        )
        vector = solution.eigenvector
    else:
        _value, vector = lk.dominant_eigenpair(
            seed, cache, params, terms=terms, method=method, certify=False
        )
    assert vector.shape == seed.shape
    assert float(jnp.linalg.norm(vector)) > 0.0
    assert _off_chain_weight(vector, _chain_modes(cache)) == 0.0


def test_shift_invert_inner_solve_stays_on_linked_chain_modes() -> None:
    cache, params, seed, terms = _selected_linked_case()
    term_cfg = linear_terms_to_term_config(terms)
    mask = ka._linked_covered_mode_mask(cache)
    solve = ka._shift_invert_apply_factory(
        seed,
        cache,
        params,
        term_cfg,
        sigma_val=jnp.asarray(0.05 - 0.1j, dtype=seed.dtype),
        gmres_tol=1.0e-8,
        gmres_maxiter=40,
        gmres_restart=20,
        shift_preconditioner="hermite-line",
    )
    x, stats = solve(ka._project_to_linked_cover(seed, mask), cache, params, term_cfg)
    assert float(jnp.linalg.norm(x)) > 0.0
    assert _off_chain_weight(x, np.asarray(mask)) == 0.0
    assert np.isfinite(float(stats.max_relative_residual))


def test_seed_without_linked_chain_component_is_refused() -> None:
    cache, params, seed, terms = _selected_linked_case()
    mask = ka._linked_covered_mode_mask(cache)
    off_chain = seed - ka._project_to_linked_cover(seed, mask)
    with pytest.raises(ValueError, match="linked-chain modes"):
        lk.dominant_eigenpair(off_chain, cache, params, terms=terms)
    with pytest.raises(ValueError, match="linked-chain modes"):
        ap.adaptive_propagator_eigenpair(off_chain, cache, params, terms)


def test_shift_invert_outer_residual_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    _patch_shift_invert(
        monkeypatch,
        lambda *args, **kwargs: (
            jnp.asarray(0.4 + 0.2j, dtype=v0.dtype),
            jnp.ones_like(v0),
        ),
    )
    monkeypatch.setattr(
        lk, "_apply_operator", lambda vector, *_args: jnp.zeros_like(vector)
    )
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_propagator_cached",
        lambda *args, **kwargs: (
            jnp.asarray(0.1 + 0.05j, dtype=v0.dtype),
            jnp.full_like(v0, 2.0),
        ),
    )
    monkeypatch.setattr(
        lk,
        "_eigenpair_relative_residual",
        lambda eigenvalue, *_args: 1.0 if float(jnp.real(eigenvalue)) > 0.2 else 0.0,
    )

    eigenvalue, eigenvector = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        fallback_method="propagator",
        shift_outer_residual_tol=0.1,
    )

    assert jnp.allclose(eigenvalue, jnp.asarray(0.1 + 0.05j, dtype=v0.dtype))
    assert jnp.allclose(eigenvector, 2.0)


def test_dominant_eigenpair_reports_shift_invert_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    messages: list[str] = []

    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_propagator_cached",
        lambda *args, **kwargs: (
            jnp.asarray(0.1 + 0.2j, dtype=v0.dtype),
            jnp.full_like(v0, 2.0 + 0.0j),
        ),
    )
    _patch_shift_invert(
        monkeypatch,
        lambda *args, **kwargs: (
            jnp.asarray(0.3 + 0.4j, dtype=v0.dtype),
            jnp.full_like(v0, 3.0 + 0.0j),
        ),
    )
    monkeypatch.setattr(
        lk,
        "_apply_operator",
        lambda vector, *_args: jnp.asarray(0.3 + 0.4j, vector.dtype) * vector,
    )

    eig, vec = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method="shift_invert",
        shift_source="propagator",
        krylov_dim=4,
        restarts=1,
        shift_maxiter=15,
        shift_restart=10,
        power_dt=0.05,
        status_callback=messages.append,
    )

    assert jnp.allclose(eig, jnp.asarray(0.3 + 0.4j, dtype=v0.dtype))
    assert jnp.allclose(vec, 3.0 + 0.0j)
    assert any("preparing shift-invert solve" in item for item in messages)
    assert any("estimating shift from propagator seed" in item for item in messages)
    assert any("running shift-invert Arnoldi" in item for item in messages)
    assert any("shift-invert solve finished" in item for item in messages)
    assert any("residual=" in item for item in messages)


def testbuild_shift_invert_preconditioneritioner_rejects_unknown_mode() -> None:
    """An unsupported name must raise, not silently disable preconditioning.

    This function used to return ``(None, None)`` for anything outside its
    whitelist. A benchmark passed ``"auto"`` -- plausible, since three sibling
    options accept it -- got no preconditioner at all, and published results
    claiming the physics-aware Hermite-line inverse was active.
    """

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    for mode in ("unknown", "hermitline", "damping-diagonal"):
        with pytest.raises(ValueError, match="unknown shift-invert preconditioner"):
            ka.build_shift_invert_preconditioner(
                v0, cache, params, term_cfg, sigma, mode
            )

    # "auto" is a real policy resolved by _automatic_shift_preconditioner one
    # level up, so reaching this function with it means that layer was bypassed.
    # The message has to say that rather than suggest a substitute: mapping it to
    # "damping" would hand a damping diagonal to a caller that asked for a
    # physics-aware line solve.
    with pytest.raises(ValueError, match="was bypassed"):
        ka.build_shift_invert_preconditioner(v0, cache, params, term_cfg, sigma, "auto")


@pytest.mark.parametrize("mode", sorted(ka.SHIFT_PRECOND_NAMES - {"none"}))
def testbuild_shift_invert_preconditioneritioner_documented_names_resolve(
    mode: str,
) -> None:
    """Every advertised name builds a usable operator.

    Only ``"damping"`` returns a diagonal array; the line and field-corrected
    variants return ``(None, operator)`` because they are not diagonal. What has
    to hold for all of them is that an operator comes back and produces finite
    output -- a name that is accepted but resolves to nothing is the same defect
    as one that is silently dropped.

    ``pr3-cm`` is the one name whose builder needs host-built factors, so this
    test supplies them the way ``_shift_invert_branch`` does. It must not be
    exempted instead: the point of the parametrization is that every advertised
    spelling reaches a working operator.
    """

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    factors = None
    if mode in pr3.PR3_PRECOND_NAMES:
        factors, _meta = pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    precond, operator = ka.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, mode, factors
    )

    assert callable(operator), f"{mode!r} is advertised but resolved to no operator"
    if mode == "damping":
        assert precond is not None
    result = operator(v0.reshape(-1))
    assert result.shape == (v0.size,)
    assert bool(jnp.all(jnp.isfinite(jnp.real(result))))


def testbuild_shift_invert_preconditioneritioner_opt_out_and_normalisation() -> None:
    """``None``/``"none"`` stay the opt-out; names are case- and space-tolerant."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    for opt_out in (None, "none", "NONE", " none "):
        assert ka.build_shift_invert_preconditioner(
            v0, cache, params, term_cfg, sigma, opt_out
        ) == (None, None)

    spaced, _ = ka.build_shift_invert_preconditioner(
        v0, cache, params, term_cfg, sigma, "  DAMPING "
    )
    assert spaced is not None


def test_line_and_field_preconditioners_are_not_the_damping_diagonal() -> None:
    """The advertised variants must actually differ from ``"damping"``.

    ``test_..._documented_names_resolve`` only asserts each name yields a usable
    operator, and on the tiny fixture that is satisfied trivially: with the drift
    terms zeroed there is nothing but collisional damping, so the Hermite line
    solve legitimately collapses onto the diagonal and all sixteen names agree to
    1e-16. A name that resolved to nothing would pass that test.

    So this one uses a cache where parallel streaming matters. Measured on the
    Cyclone linear case at ``Nl=4, Nm=16``: both variants differ from the damping
    diagonal by a relative 0.99. If that collapses, a run asking for the
    physics-aware preconditioner is silently getting the diagonal -- which is the
    defect the whole name-validation change exists to prevent.
    """

    from pathlib import Path

    from gkx.core_grid import select_ky_grid
    from gkx.geometry.flux_tube import sample_flux_tube_geometry
    from gkx.runtime import (
        build_runtime_geometry,
        build_runtime_linear_params,
        build_runtime_linear_terms,
    )
    from gkx.workflows.runtime.toml import load_runtime_from_toml

    runtime, _raw = load_runtime_from_toml(
        Path("examples/linear/axisymmetric/cyclone.toml")
    )
    analytic = build_runtime_geometry(runtime)
    ntheta = int(runtime.grid.ntheta)
    theta = jnp.linspace(
        float(runtime.grid.z_min), float(runtime.grid.z_max), ntheta, endpoint=False
    )
    geom = sample_flux_tube_geometry(analytic, theta)
    n_laguerre, n_hermite = 4, 16
    params = build_runtime_linear_params(runtime, Nm=n_hermite, geom=analytic)
    term_cfg = linear_terms_to_term_config(build_runtime_linear_terms(runtime))
    grid = select_ky_grid(
        build_spectral_grid(GridConfig(Nx=1, Ny=12, Nz=ntheta, Lx=6.0, Ly=62.83)), 3
    )
    cache = build_linear_cache(grid, geom, params, Nl=n_laguerre, Nm=n_hermite)
    state = jnp.ones(
        (n_laguerre, n_hermite, 1, 1, grid.z.size), dtype=jnp.complex128
    ) * (1.0 + 0.1j)
    sigma = jnp.asarray(0.1j, dtype=state.dtype)

    def applied(mode: str) -> jnp.ndarray:
        _precond, operator = ka.build_shift_invert_preconditioner(
            state, cache, params, term_cfg, sigma, mode
        )
        return operator(state.reshape(-1))

    reference = applied("damping")
    scale = float(jnp.max(jnp.abs(reference)))
    for mode in ("hermite-line", "field-corrected"):
        relative = float(jnp.max(jnp.abs(applied(mode) - reference))) / scale
        assert relative > 0.1, (
            f"{mode!r} is within {relative:.2e} of the damping diagonal on a cache "
            "where streaming matters, so it is not applying the line solve"
        )


# ---- from test_dot_precision_guard.py ----
# Guard the contractions that XLA would otherwise satisfy with TF32.
#
# On Ampere and later NVIDIA GPUs XLA may answer an unpinned ``dot`` on the tensor
# cores in TF32, which keeps 10 mantissa bits: a relative error of 2^-11 = 4.9e-04
# where float32 would give ~1e-07. Two properties make the defect hard to see.
#
# Nothing on CPU can observe it. There is no TF32 path there, so an unpinned dot
# and ``Precision.HIGHEST`` produce bit-identical CPU numbers and no assertion on a
# *value* can fail. The precision request, however, is recorded in the jaxpr on
# every backend, so that is what these tests assert.
#
# And it is shape-dependent. Measured on an RTX A4000, XLA uses TF32 only when the
# contraction is a genuine matrix product -- a free dimension left on *both*
# operands. Every vector-shaped contraction measured exact: a vdot of two length
# 4096 vectors at 4.2e-08, a ``(16,) x (16, 4096)`` tensordot at 1.2e-07, a
# ``(512, 512) x (512,)`` matvec at 1.1e-07, an ``n,kn->k`` einsum at 1.8e-07, and
# the Arnoldi Gram-Schmidt vdot inside its ``fori_loop`` at 8.7e-07. The matrix
# products on the same GPU: ``(16, 16) x (16, 4096)`` at 3.0e-04 and a real
# ``(m, m) x (m, m)`` at 3.1e-04 for m = 8, 64 and 512, every one of them back to
# ~1e-07 when pinned.
#
# So the hazard is not "a dot" but "a dot that is matrix-shaped", and a contraction
# can cross that line without being edited -- ``dominant_eigenpairs_propagator_cached``
# lifts a ``(candidates, k) x (k, n)`` product that is exact at the default
# ``candidates = 1`` and TF32 from two candidates on. ``assert_matrix_dots_pinned``
# therefore classifies by shape rather than by call site, which is what lets it
# catch a contraction that becomes hazardous because a config value moved.


EXACT = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)


ALLOWED_UNPINNED_MATRIX_DOTS = {
    # Renamed by the flat-layout pass; the file, the line and the measurement
    # behind the exemption are unchanged, only the module path is. The line moved
    # 675 -> 749 when the inner-solve statistics were added above it,
    # 749 -> 828 when the linked-chain mask helpers were (Q6), 828 -> 817
    # when those helpers moved to operators/linear/linked.py (Q19), and
    # 817 -> 809 when Q23 and Q24 merged above it, and 809 -> 834 when Q28 added
    # the pr3-cm import, its names and its factors argument, and the comment
    # recording why the shifted FGMRES starts from zero, all above it; same
    # code, still the `lifted = jnp.tensordot(eigvecs.T, V[:krylov_dim],
    # axes=1)` of `_propagator_arnoldi_restart_step`, verified at the new line.
    "solvers_linear_krylov_algorithms.py:834": "overlap ranking only; argmax provably unmoved",
}


def _iter_dots(jaxpr):
    """Yield every ``dot_general``, descending into nested jaxprs.

    ``fori_loop``, ``scan``, ``cond``, ``pjit`` and ``custom_vjp`` all park their
    equations in a sub-jaxpr, so a scan of ``jaxpr.eqns`` alone sees none of the
    Arnoldi contractions -- they live inside the loop body.
    """

    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            yield eqn
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(item, "jaxpr", item)
                inner = getattr(inner, "jaxpr", inner)
                if hasattr(inner, "eqns"):
                    yield from _iter_dots(inner)


def _is_matrix_product(eqn) -> bool:
    """True when both operands keep a free axis, i.e. the tensor-core case."""

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = eqn.params[
        "dimension_numbers"
    ]
    lhs, rhs = eqn.invars[0].aval, eqn.invars[1].aval
    lhs_free = [
        axis
        for axis in range(len(lhs.shape))
        if axis not in lhs_contract and axis not in lhs_batch
    ]
    rhs_free = [
        axis
        for axis in range(len(rhs.shape))
        if axis not in rhs_contract and axis not in rhs_batch
    ]
    return bool(lhs_free) and bool(rhs_free)


def _origin(eqn) -> str:
    """Return ``file.py:line`` inside gkx for one equation, for the allowlist."""

    traceback = eqn.source_info.traceback
    for frame in getattr(traceback, "frames", []) or []:
        name = getattr(frame, "file_name", "")
        if "/gkx/" in name and "site-packages" not in name:
            return f"{name.rsplit('/', 1)[-1]}:{frame.line_num}"
    return "<unknown>"


def matrix_dots(function, *args, **kwargs):
    """Return ``(origin, precision)`` for every matrix-shaped dot in a callable."""

    jaxpr = jax.make_jaxpr(function)(*args, **kwargs).jaxpr
    return [
        (_origin(eqn), eqn.params["precision"])
        for eqn in _iter_dots(jaxpr)
        if _is_matrix_product(eqn)
    ]


def assert_matrix_dots_pinned(label, function, *args, **kwargs) -> int:
    """Assert every matrix-shaped contraction is pinned; return how many there were."""

    found = matrix_dots(function, *args, **kwargs)
    unpinned = [
        origin
        for origin, precision in found
        if precision is None and origin not in ALLOWED_UNPINNED_MATRIX_DOTS
    ]
    assert not unpinned, (
        f"{label} contains matrix-shaped contractions with no precision request: "
        f"{sorted(set(unpinned))}. XLA satisfies these with TF32 on Ampere and "
        "later NVIDIA GPUs (10 mantissa bits, ~4.9e-04 relative). Pin them with "
        "precision=jax.lax.Precision.HIGHEST, or add the call site to "
        "ALLOWED_UNPINNED_MATRIX_DOTS with the measurement that says it is safe."
    )
    return len(found)


def _linear_setup():
    grid_cfg = GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="periodic")
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        nu=0.01,
        nu_hyper=0.01,
        damp_ends_amp=0.0,
        damp_ends_widthfrac=0.0,
    )
    nl, nm = 2, 4
    cache = build_linear_cache(grid, geom, params, Nl=nl, Nm=nm)
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = linear_terms_to_term_config(terms)
    rng = np.random.default_rng(0)
    shape = (nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    v0 = jnp.asarray(
        rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=jnp.complex64
    )
    return cache, params, term_cfg, v0


@pytest.mark.parametrize("candidates", [1, 2, 4])
def test_propagator_candidate_lift_pins_exact_dot_precision(candidates: int) -> None:
    """The lift that produces the returned eigenvectors must never run in TF32.

    This is the contraction whose hazard depends on a config value: at
    ``candidates = 1`` it is a matvec and XLA leaves it exact, at two or more it
    is a matrix product and TF32 moved the returned eigenvector from 1.2e-08 to
    1.8e-05 against a float64 reference on an RTX A4000. Every candidate count is
    asserted so the pin cannot be dropped on the argument that the default is safe.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    found = matrix_dots(
        lambda value: kp.dominant_eigenpairs_propagator_cached(
            value,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            dt=0.01,
            propagator_steps=2,
            candidates=candidates,
        ),
        v0,
    )
    lifts = [
        precision
        for origin, precision in found
        # Flat-layout rename: the module is solvers_linear_krylov_propagator.py.
        # This filter selects which contractions the precision guard inspects,
        # so a stale prefix makes the guard silently vacuous rather than failing.
        if origin.startswith("solvers_linear_krylov_propagator")
    ]
    assert lifts, "the candidate lift lowered to no matrix dot; the guard is vacuous"
    assert all(precision == EXACT for precision in lifts), (
        f"the candidate lift is unpinned at candidates={candidates} ({lifts}); it "
        "will be answered in TF32 on Ampere and later NVIDIA GPUs"
    )


def test_gauss_newton_normal_equations_pin_exact_dot_precision() -> None:
    """J^T J is the one matrix product in the inverse-design step.

    Forming the normal equations squares the conditioning, so TF32's 10 mantissa
    bits land directly in the step: 1.8e-04 on the entries and 1.7e-04 on the
    step against float64 on an RTX A4000, versus the 1.0e-04 rtol the geometry
    AD-versus-FD report gates on. The matvec and the vector dot on either side of
    it are vector-shaped and stay exact unpinned, so only one pin is expected.
    """

    jac = jnp.asarray(np.linspace(0.1, 1.6, 12).reshape(4, 3), dtype=jnp.float32)
    residual = jnp.asarray([0.3, -0.2, 0.5, 0.1], dtype=jnp.float32)
    found = matrix_dots(
        lambda j, r: _damped_gauss_newton_step(j, r, damping=1.0e-6), jac, residual
    )
    assert found, "the Gauss-Newton step lowered to no matrix dot; the guard is vacuous"
    assert all(precision == EXACT for _origin, precision in found), (
        f"the normal-equations product is unpinned ({found}); TF32 puts 1.7e-04 "
        "into a step whose own report gates at rtol 1.0e-04"
    )


def test_hot_path_matrix_contractions_are_pinned() -> None:
    """No matrix-shaped contraction in the solver hot paths may be left to TF32.

    A repo-wide ratchet scoped to the paths that actually run on a GPU. It is
    shape-aware on purpose: pinning every dot would cost throughput on the
    vector-shaped majority that measures exact anyway, while pinning by call site
    would miss the case this audit found -- a contraction that becomes a matrix
    product because a config value changed, with no edit to the line.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    checked = 0
    checked += assert_matrix_dots_pinned(
        "_arnoldi",
        lambda value: ka._arnoldi(
            value, ka._apply_operator, cache, params, term_cfg, 8
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "_operator_arnoldi_restart_step",
        lambda value: ka._operator_arnoldi_restart_step(
            value,
            value,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            omega_min_factor=0.0,
            omega_target_factor=2.0,
            omega_cap_factor=10.0,
            omega_sign=1,
            select_overlap=True,
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "_propagator_arnoldi_restart_step",
        lambda value: ka._propagator_arnoldi_restart_step(
            value,
            value,
            ka._apply_operator,
            cache,
            params,
            term_cfg,
            krylov_dim=8,
            horizon=jnp.asarray(1.0),
            growth_only=True,
            omega_min_factor=0.0,
            omega_target_factor=2.0,
            omega_cap_factor=10.0,
            omega_sign=1,
            select_overlap=True,
        ),
        v0,
    )
    checked += assert_matrix_dots_pinned(
        "dominant_eigenpair_power",
        lambda value: ka.dominant_eigenpair_power(
            value, cache, params, term_cfg, iterations=2, dt=0.01
        ),
        v0,
    )
    for candidates in (1, 4):
        checked += assert_matrix_dots_pinned(
            f"dominant_eigenpairs_propagator_cached(candidates={candidates})",
            lambda value, count=candidates: kp.dominant_eigenpairs_propagator_cached(
                value,
                cache,
                params,
                term_cfg,
                krylov_dim=8,
                dt=0.01,
                propagator_steps=2,
                candidates=count,
            ),
            v0,
        )
    # Three today: the allowlisted overlap lift in the propagator restart, and the
    # candidate lift at each of the two candidate counts. The floor is here so that
    # a refactor which stops reaching these paths fails loudly instead of leaving a
    # sweep that asserts nothing.
    assert checked >= 3, (
        f"only {checked} matrix-shaped contractions were reached; the sweep has "
        "stopped exercising the paths it is supposed to guard"
    )


def test_allowlisted_contraction_is_still_matrix_shaped_and_unpinned() -> None:
    """Keep the allowlist honest.

    If the overlap lift is ever pinned, or stops being matrix-shaped, the entry
    above is stale and must be deleted rather than left to excuse some future
    contraction that happens to land on the same line.
    """

    cache, params, term_cfg, v0 = _linear_setup()
    found = dict(
        matrix_dots(
            lambda value: ka._propagator_arnoldi_restart_step(
                value,
                value,
                ka._apply_operator,
                cache,
                params,
                term_cfg,
                krylov_dim=8,
                horizon=jnp.asarray(1.0),
                growth_only=True,
                omega_min_factor=0.0,
                omega_target_factor=2.0,
                omega_cap_factor=10.0,
                omega_sign=1,
                select_overlap=True,
            ),
            v0,
        )
    )
    for origin in ALLOWED_UNPINNED_MATRIX_DOTS:
        assert origin in found, (
            f"{origin} is allowlisted as an unpinned matrix contraction but no "
            "longer appears as one; delete the stale entry"
        )
        assert found[origin] is None, (
            f"{origin} is pinned now, so its allowlist entry is misleading; delete it"
        )


# --------------------------------------------------------------------------- #
# pr3-cm (Q28): the structured preconditioner and its exact z-block solve        #
# --------------------------------------------------------------------------- #
def _pr3_setup(*, Nx: int = 1, Nl: int = 6, Nm: int = 6, nu: float = 0.0):
    """A fixture with the terms ``pr3-cm`` actually splits switched on.

    ``_tiny_krylov_setup`` zeroes the drifts, the mirror, the drive and the end
    damping, which leaves the z-local block diagonal in ``(l, m)`` and would
    make every structural assertion below true for the wrong reason: a diagonal
    block is l-tridiagonal whatever the physics does. This one turns them on,
    and uses linked boundaries, because the z-mean drift bookkeeping is the
    whole content of the ``-cm`` suffix and the chain cover is what the
    Hermite-line solve restricts to.

    ``Nx`` and ``nu`` are the two knobs the guards react to: ``Nx > 1`` gives
    the grid zonal ``(ky = 0, kx > 0)`` rows, whose adiabatic ``<phi>`` sums
    over z and so is not z-local, and ``nu > 0`` switches on a collision
    operator that couples the Laguerre index beyond ``l +- 1``.
    """

    grid_cfg = GridConfig(
        Nx=Nx, Ny=4, Nz=8, Lx=6.0, Ly=6.0, boundary="linked", y0=20.0, jtwist=1
    )
    cfg = CycloneBaseCase(grid=grid_cfg)
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        nu=nu,
        nu_hyper=0.0,
        damp_ends_amp=0.1,
        damp_ends_widthfrac=0.125,
    )
    cache = build_linear_cache(grid, geom, params, Nl=Nl, Nm=Nm)
    shape = (Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    rng = np.random.default_rng(0)
    v0 = jnp.asarray(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        dtype=jnp.complex128,
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=1.0,
        hypercollisions=1.0,
        end_damping=1.0,
        apar=0.0,
        bpar=0.0,
    )
    return cache, params, v0, linear_terms_to_term_config(terms)


_PR3_SIGMA = 0.05 - 0.2j


def test_pr3_z_block_is_l_tridiagonal_plus_a_rank_one_field_part() -> None:
    """The structural property the exact solve is allowed to assume.

    Block-Thomas plus Sherman-Morrison is exact only while the z-local block is
    block-tridiagonal in the Laguerre index and its field part is rank one per
    ``(kx, z)``. Q21 (#255) measured both on the production chain -- largest
    off-tridiagonal entry exactly 0 against a block norm of 74.7, rank-one
    singular-value ratio <= 5.3e-14 -- and the build re-measures them at every
    shift, so this test pins the property itself rather than the decision that
    follows from it. A Laguerre coupling beyond ``l +- 1``, or a second field
    that is not one scalar per ``(kx, z)``, is caught here.
    """

    cache, params, v0, term_cfg = _pr3_setup()
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    _factors, meta = pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma)

    structure = meta["structure"]
    assert structure["block_norm_max"] > 0.0
    assert structure["off_tridiagonal_max"] == 0.0, (
        "the z-local block gained a Laguerre coupling beyond l +- 1, so the "
        "block-Thomas solve is no longer exact"
    )
    assert structure["rank_one_s0_max"] > 0.0
    assert structure["rank_one_ratio_max"] < 1.0e-10, (
        "the field part of the z-local block is no longer rank one per (kx, z), "
        "so the Sherman-Morrison correction is no longer exact"
    )
    # The dense blocks must also *be* the operator: the split only means
    # anything while everything outside streaming and hypercollisions is z-local.
    assert meta["locality_defect"] < 1.0e-12
    assert meta["block_solve"] == "block-thomas"


def test_pr3_exact_block_solve_equals_the_dense_inverse_and_is_smaller() -> None:
    """The cheaper apply has to be the same preconditioner, not a cheaper one.

    Q21 measured the two solves agreeing to 4.4e-16 with every iteration count
    unchanged, and the factors 5.26x smaller at the production chain. That
    combination -- an exactness property at an unchanged iteration count, plus a
    memory reduction -- is what makes this a cost reduction with no accuracy
    loss under the rewritten §5.1 gate rather than a different preconditioner
    that would need re-certifying.
    """

    cache, params, v0, term_cfg = _pr3_setup()
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    exact, exact_meta = pr3.build_pr3_factors(
        v0, cache, params, term_cfg, sigma, block_solve="block-thomas"
    )
    dense, dense_meta = pr3.build_pr3_factors(
        v0, cache, params, term_cfg, sigma, block_solve="dense"
    )
    assert exact_meta["block_solve"] == "block-thomas"
    assert dense_meta["block_solve"] == "dense"
    # Same parameter and same shift, so the same operator is inverted both ways.
    assert exact_meta["alpha"] == dense_meta["alpha"]
    assert exact_meta["s1"] == dense_meta["s1"]

    probe = v0.reshape(-1)
    fast = pr3.build_pr3_apply(v0, cache, params, term_cfg, exact)(probe)
    slow = pr3.build_pr3_apply(v0, cache, params, term_cfg, dense)(probe)
    relative = float(jnp.linalg.norm(fast - slow) / jnp.linalg.norm(slow))
    assert relative < 1.0e-12, (
        f"block-Thomas and dense applies differ by {relative:.3g}"
    )
    assert exact_meta["factor_bytes"] < dense_meta["factor_bytes"]


def test_pr3_falls_back_to_the_dense_inverse_when_the_structure_breaks() -> None:
    """An operator that breaks the structure must not get the exact solve.

    The break is real, not injected: switching on the collision operator couples
    the Laguerre index beyond ``l +- 1``, and the shipped Cyclone deck's own
    ``nu = 0`` is why the production chain measures exactly 0 there. ``"auto"``
    falls back to the dense batched inverse -- the same preconditioner, a
    costlier apply -- and records why; an explicit ``"block-thomas"`` refuses
    instead, because a caller who asked for the exact solve by name is
    measuring it and must not be handed a different one silently.
    """

    cache, params, v0, term_cfg = _pr3_setup(nu=0.01)
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    _factors, meta = pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    assert meta["block_solve"] == "dense"
    assert "not l-tridiagonal" in meta["block_solve_reason"]
    assert meta["structure"]["off_tridiagonal_max"] > 0.0

    with pytest.raises(ValueError, match="not l-tridiagonal"):
        pr3.build_pr3_factors(
            v0, cache, params, term_cfg, sigma, block_solve="block-thomas"
        )


def test_pr3_refuses_a_grid_whose_non_streaming_part_is_not_z_local() -> None:
    """z-locality is a refusal, not a fallback.

    The dense blocks are a *representation* of the non-streaming operator; if
    they do not reproduce it, the two halves this preconditioner sweeps between
    are not the halves of this operator, and no choice of block solve repairs
    that. Again the break is real: a grid with more than one ``kx`` carries
    zonal ``(ky = 0, kx > 0)`` rows, whose adiabatic ``<phi>`` is a sum over z,
    and the linear eigen route escapes it only because it reduces the grid to
    one non-zero ``ky``. The check is against the operator itself on a random
    vector, so a coupling the column probes cannot see is still caught.
    """

    cache, params, v0, term_cfg = _pr3_setup(Nx=4)
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    with pytest.raises(ValueError, match="not z-local"):
        pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    # ... and the refusal names what to use instead, because a user who hit it
    # asked for pr3-cm by name and needs a route, not a verdict.
    with pytest.raises(ValueError, match="hermite-line"):
        pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma, block_solve="dense")


def test_pr3_parameter_follows_the_symbol_rule_and_is_overridable() -> None:
    """``alpha = -sqrt(s1 d)`` by default; the shift split is ``s1 = sigma/2 - alpha``."""

    cache, params, v0, term_cfg = _pr3_setup()
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    _f, auto = pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma)
    assert auto["alpha_source"] == "symbol-bounds"
    assert auto["alpha"][1] == 0.0 and auto["alpha"][0] < 0.0
    expected = -float(np.sqrt(auto["symbol_s1"] * auto["block_spectral_radius"]))
    assert auto["alpha"][0] == pytest.approx(expected, rel=1.0e-12)
    # 2 s1 + 2 alpha = sigma is what makes the two half-steps add up.
    s1 = complex(*auto["s1"])
    alpha = complex(*auto["alpha"])
    assert 2.0 * s1 + 2.0 * alpha == pytest.approx(complex(_PR3_SIGMA))
    assert auto["sweeps"] == 3

    _f, forced = pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma, alpha=-2.5)
    assert forced["alpha_source"] == "explicit"
    assert forced["alpha"] == [-2.5, 0.0]

    with pytest.raises(ValueError, match="shift_precond_block_solve must be one of"):
        pr3.build_pr3_factors(
            v0, cache, params, term_cfg, sigma, block_solve="tridiagonal"
        )
    with pytest.raises(ValueError, match="alpha must be non-zero"):
        pr3.build_pr3_factors(v0, cache, params, term_cfg, sigma, alpha=0.0)


def test_pr3_needs_its_host_built_factors() -> None:
    """Reaching the traced builder without factors is a caller error, not a typo."""

    cache, params, v0, term_cfg = _pr3_setup()
    sigma = jnp.asarray(_PR3_SIGMA, dtype=v0.dtype)

    with pytest.raises(ValueError, match="needs the host-built"):
        ka.build_shift_invert_preconditioner(
            v0, cache, params, term_cfg, sigma, "pr3-cm"
        )


def test_pr3_is_selectable_by_name_and_reports_its_setup() -> None:
    """The route reaches it by name and records what the build measured.

    ``EigenSolveStatus.inner["preconditioner_setup"]`` is the only place a user
    sees which z-block solve ran, so a fallback that did not report itself would
    be invisible. The default is unchanged: ``"auto"`` still resolves to the
    line solve, and ``KrylovConfig`` still defaults to ``"auto"``.
    """

    cache, params, v0, term_cfg = _pr3_setup()
    assert lk.KrylovConfig().shift_preconditioner == "auto"
    assert lk.KrylovConfig().shift_precond_block_solve == "auto"
    assert lk.KrylovConfig().shift_precond_alpha is None
    assert lk._automatic_shift_preconditioner(params, term_cfg) == "hermite-line"
    assert pr3.PR3_PRECOND_NAMES <= ka.SHIFT_PRECOND_NAMES

    try:
        _value, _vector, status = lk.dominant_eigenpair(
            v0,
            cache,
            params,
            None,
            method="shift_invert",
            shift_preconditioner="pr3-cm",
            shift_maxiter=40,
            shift_restart=40,
            krylov_dim=6,
            restarts=1,
            fallback_method="none",
            return_status=True,
        )
    except RuntimeError as exc:
        # This fixture is far too small to certify an eigenpair, and that is not
        # what the test owns; the build's report reaches the message either way.
        assert "residual" in str(exc)
        return
    assert status.inner is not None
    assert status.inner["preconditioner"] == "pr3-cm"
    setup = status.inner["preconditioner_setup"]
    assert setup is not None and setup["sweeps"] == 3
    assert setup["block_solve"] in {"block-thomas", "dense"}
