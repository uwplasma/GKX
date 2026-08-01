"""Focused unit/regression tests for matrix-free Krylov utilities."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import solvax

import gkx.solvers.linear.adaptive_propagator as ap
import gkx.solvers.linear.krylov as lk
import gkx.solvers.linear.krylov_algorithms as ka
import gkx.solvers.linear.krylov_propagator as kp
from gkx.config import CycloneBaseCase, GridConfig
from gkx.core.grid import build_spectral_grid
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import (
    LinearParams,
    LinearTerms,
    Species,
    build_linear_params,
    linear_terms_to_term_config,
)
from gkx.solvers.linear import implicit
from support.paired_solvax import requires_paired_solvax

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


def _tiny_krylov_setup(*, linked: bool = False):
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)
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


def test_build_shift_invert_preconditioner_modes() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.1j, dtype=v0.dtype)

    precond, op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, None
    )
    assert precond is None and op is None
    precond, op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "unknown"
    )
    assert precond is None and op is None

    precond, op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "damping"
    )
    assert precond is not None and op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    _precond, op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))

    precond, op = lk._build_shift_invert_precond(
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

    _precond, op = lk._build_shift_invert_precond(
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


def test_shifted_hermite_preconditioner_handles_a_zero_shift() -> None:
    """A marginal target must use the finite damping fallback."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    sigma = jnp.asarray(0.0 + 0.0j, dtype=v0.dtype)
    _diagonal, damping_op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "damping"
    )
    _precond, line_op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )

    assert damping_op is not None and line_op is not None
    expected = damping_op(v0.reshape(-1))
    result = line_op(v0.reshape(-1))
    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, expected)


def test_field_corrected_shifted_preconditioner_removes_low_moment_coupling() -> None:
    """Woodbury field correction must fix the Hermite line inverse's main defect."""

    _grid, cache, params, v0, _term_cfg, _terms = _tiny_krylov_setup(linked=False)
    params = replace(
        params,
        omega_star_scale=1.0,
        omega_d_scale=1.0,
        R_over_LTi=6.9,
        R_over_Ln=2.2,
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
        _diagonal, preconditioner = lk._build_shift_invert_precond(
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
    _diagonal, preconditioner = lk._build_shift_invert_precond(
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


def test_build_shift_invert_preconditioner_linked_branch() -> None:
    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=True)
    sigma = jnp.asarray(0.2j, dtype=v0.dtype)
    _precond, op = lk._build_shift_invert_precond(
        v0, cache, params, term_cfg, sigma, "hermite-line"
    )
    assert op is not None
    y = op(v0.reshape(-1))
    assert y.shape == (v0.size,)
    assert jnp.all(jnp.isfinite(jnp.real(y)))


def test_shift_invert_uses_right_preconditioning_and_physical_fgmres_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shifted solve must minimize the original, not transformed, residual."""

    _grid, cache, params, v0, term_cfg, _terms = _tiny_krylov_setup(linked=False)
    calls: list[tuple[bool, jnp.ndarray, int]] = []
    monkeypatch.setattr(
        ka,
        "_build_shift_invert_precond",
        lambda *_args: (jnp.ones_like(v0), lambda value: 0.1 * value),
    )
    monkeypatch.setattr(ka, "_apply_operator", lambda value, *_args: value)

    def fake_gmres(_matvec, b, *, precond, x0, max_restarts, **_kwargs):
        calls.append((precond is not None, x0, max_restarts))
        return SimpleNamespace(x=b)

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
        gmres_solve_method="batched",
        shift_preconditioner="damping",
    )

    observed = apply_inverse(v0, cache, params, term_cfg)

    assert len(calls) == 1
    assert calls[0][0]
    assert jnp.allclose(calls[0][1], 0.1 * v0.reshape(-1))
    assert calls[0][2] == 1
    assert jnp.allclose(observed, v0)


@pytest.mark.parametrize("method", ["power", "propagator", "arnoldi"])
def test_dominant_eigenpair_methods_produce_finite_values(method: str) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    eig, vec = lk.dominant_eigenpair(
        v0,
        cache,
        params,
        terms=terms,
        method=method,
        krylov_dim=4,
        restarts=1,
        power_iters=4,
        power_dt=0.05,
    )
    assert vec.shape == v0.shape
    assert jnp.isfinite(jnp.real(eig))
    assert jnp.isfinite(jnp.imag(eig))


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

    assert solution.converged
    assert solution.stable
    assert solution.restarts < 5
    assert solution.filter_dt < 2.8 / float(jnp.max(jnp.abs(eigenvalues)))
    assert complex(np.asarray(solution.eigenvalue)) == pytest.approx(
        complex(np.asarray(eigenvalues[0])), rel=1.0e-9
    )
    assert float(np.asarray(solution.residual)) < 1.0e-9
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", _fake_shift)
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", fake_shift)
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
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_shift_invert_cached",
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
    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", fake_shift)
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

    monkeypatch.setattr(lk, "dominant_eigenpair_shift_invert_cached", fake_shift)
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


def test_shift_invert_outer_residual_triggers_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _grid, cache, params, v0, _term_cfg, terms = _tiny_krylov_setup(linked=False)
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_shift_invert_cached",
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
    monkeypatch.setattr(
        lk,
        "dominant_eigenpair_shift_invert_cached",
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
