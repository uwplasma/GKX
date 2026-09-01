"""Unit tests for low-level term operators."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import gkx.terms as term_pkg
from gkx.core_velocity import hermite_ladder_coeffs
from gkx.terms.config import FieldState, TermConfig
from gkx.operators.linear.streaming import (
    _check_positive,
    abs_z_linked_fft,
    apply_hermite_v,
    apply_hermite_v2,
    apply_laguerre_x,
    grad_z_linked_fft,
    grad_z_periodic,
    shift_axis,
    streaming_ladder_term,
)


def test_grad_z_periodic_with_dz_and_kz_match() -> None:
    n = 64
    z = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    f = jnp.sin(3.0 * z)
    dz = z[1] - z[0]
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dz)
    df_dz = grad_z_periodic(f, dz=dz)
    df_kz = grad_z_periodic(f, kz=kz)
    expected = 3.0 * jnp.cos(3.0 * z)
    assert jnp.allclose(df_dz, expected, atol=2.0e-2)
    assert jnp.allclose(df_dz, df_kz, atol=1.0e-10)


def test_fft_z_operators_preserve_complex64_with_float64_wavenumbers() -> None:
    """FFT multipliers must not promote complex64 states under x64/sharding."""

    nz = 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.exp(1j * z).astype(jnp.complex64)
    kz = jnp.asarray(2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz), dtype=jnp.float64)

    out_periodic = grad_z_periodic(f, kz=kz)

    linked_f = jnp.stack([f, 0.5j * f], axis=0)[None, ...]
    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = jnp.asarray(
        2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz), dtype=jnp.float64
    )
    out_linked = grad_z_linked_fft(
        linked_f, dz=dz, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )
    out_abs = abs_z_linked_fft(
        linked_f, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )

    assert out_periodic.dtype == jnp.complex64
    assert out_linked.dtype == jnp.complex64
    assert out_abs.dtype == jnp.complex64


def test_grad_z_periodic_requires_dz_or_kz() -> None:
    with pytest.raises(ValueError):
        grad_z_periodic(jnp.ones((8,)))


def test_grad_z_linked_fft_valid_chain_matches_manual() -> None:
    ny, nx, nz = 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[0, 0, :].set(jnp.exp(1j * z))
    f = f.at[0, 1, :].set(2.0 * jnp.exp(1j * 2.0 * z))

    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    out = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )

    chain = jnp.concatenate([f[0, 0, :], f[0, 1, :]])
    dchain = grad_z_periodic(chain, dz=dz)
    expected = jnp.stack([dchain[:nz], dchain[nz:]], axis=0)[None, ...]
    assert jnp.allclose(out, expected, atol=1.0e-5)


def test_grad_z_linked_fft_with_inverse_permutation_matches_scatter_path() -> None:
    ny, nx, nz = 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[0, 0, :].set(jnp.exp(1j * z))
    f = f.at[0, 1, :].set(2.0 * jnp.exp(1j * 2.0 * z))

    idx_map = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    inv = jnp.asarray([1, 0], dtype=jnp.int32)

    out_scatter = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )
    out_perm = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_inverse_permutation=inv,
        linked_full_cover=True,
    )
    assert jnp.allclose(out_perm, out_scatter, atol=1.0e-5)


def test_linked_fft_gather_paths_match_scatter_for_derivative_and_abs() -> None:
    ny, nx, nz = 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[0, 0, :].set(jnp.exp(1j * z))
    f = f.at[0, 1, :].set((0.5 + 0.25j) * jnp.exp(1j * 2.0 * z))
    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    gather_map = jnp.asarray([0, 1], dtype=jnp.int32)
    gather_mask = jnp.asarray([True, True])

    grad_scatter = grad_z_linked_fft(
        f, dz=dz, linked_indices=(idx_map,), linked_kz=(kz_link,)
    )
    grad_gather = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_gather_map=gather_map,
        linked_gather_mask=gather_mask,
        linked_use_gather=True,
    )
    abs_scatter = abs_z_linked_fft(f, linked_indices=(idx_map,), linked_kz=(kz_link,))
    abs_gather = abs_z_linked_fft(
        f,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
        linked_gather_map=gather_map,
        linked_gather_mask=gather_mask,
        linked_use_gather=True,
    )

    assert jnp.allclose(grad_gather, grad_scatter, atol=1.0e-5)
    assert jnp.allclose(abs_gather, abs_scatter, atol=1.0e-5)


def test_grad_z_linked_fft_restores_negative_ky_rows_by_conjugate_symmetry() -> None:
    ny, nx, nz = 8, 4, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[1, 0, :].set(jnp.exp(1j * z))
    f = f.at[1, 1, :].set((1.0 - 0.5j) * jnp.exp(1j * 2.0 * z))
    kx_neg = jnp.asarray([0, 3, 2, 1], dtype=jnp.int32)
    f = f.at[7, :, :].set(jnp.conj(jnp.take(f[1], kx_neg, axis=0)))

    idx_map = jnp.asarray([[1, 1 + ny]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    out = grad_z_linked_fft(
        f,
        dz=dz,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )

    assert jnp.max(jnp.abs(out[7])) > 0.0
    assert jnp.allclose(out[7], jnp.conj(jnp.take(out[1], kx_neg, axis=0)), atol=1.0e-5)


def test_abs_z_linked_fft_restores_negative_ky_rows_by_conjugate_symmetry() -> None:
    ny, nx, nz = 8, 4, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    f = jnp.zeros((ny, nx, nz), dtype=jnp.complex64)
    f = f.at[1, 0, :].set(jnp.exp(1j * z))
    f = f.at[1, 1, :].set((1.0 + 0.25j) * jnp.exp(1j * 2.0 * z))
    kx_neg = jnp.asarray([0, 3, 2, 1], dtype=jnp.int32)
    f = f.at[7, :, :].set(jnp.conj(jnp.take(f[1], kx_neg, axis=0)))

    idx_map = jnp.asarray([[1, 1 + ny]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=z[1] - z[0])
    out = abs_z_linked_fft(
        f,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )

    assert jnp.max(jnp.abs(out[7])) > 0.0
    assert jnp.allclose(out[7], jnp.conj(jnp.take(out[1], kx_neg, axis=0)), atol=1.0e-5)


def test_linked_fft_validates_inputs() -> None:
    f = jnp.ones((1, 2, 8), dtype=jnp.complex64)
    dz = jnp.asarray(0.1)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(16, d=dz)
    with pytest.raises(ValueError):
        grad_z_linked_fft(f, dz=dz, linked_indices=(), linked_kz=())
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(),
        )
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([0, 1], dtype=jnp.int32),),
            linked_kz=(kz,),
        )
    with pytest.raises(ValueError):
        grad_z_linked_fft(
            f,
            dz=dz,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(kz,),
            linked_full_cover=True,
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(f, linked_indices=(), linked_kz=())
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(),
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([0, 1], dtype=jnp.int32),),
            linked_kz=(kz,),
        )
    with pytest.raises(ValueError):
        abs_z_linked_fft(
            f,
            linked_indices=(jnp.asarray([[0, 1]], dtype=jnp.int32),),
            linked_kz=(kz,),
            linked_full_cover=True,
        )


def test_shift_axis_edge_cases() -> None:
    arr = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    assert jnp.allclose(shift_axis(arr, 0, axis=0), arr)
    assert jnp.allclose(shift_axis(arr, 1, axis=0), jnp.asarray([2.0, 3.0, 4.0, 0.0]))
    assert jnp.allclose(shift_axis(arr, -1, axis=0), jnp.asarray([0.0, 1.0, 2.0, 3.0]))
    assert jnp.allclose(shift_axis(arr, 10, axis=0), jnp.zeros_like(arr))
    assert jnp.allclose(shift_axis(arr, -10, axis=0), jnp.zeros_like(arr))


def test_hermite_laguerre_operators_shapes_and_values() -> None:
    G = jnp.zeros((2, 3, 4, 1, 1, 1))
    G = G.at[0, 1, 2, 0, 0, 0].set(1.0)
    hv = apply_hermite_v(G)
    hv2 = apply_hermite_v2(G)
    lx = apply_laguerre_x(G)
    assert hv.shape == G.shape
    assert hv2.shape == G.shape
    assert lx.shape == G.shape
    assert jnp.isfinite(hv).all()
    assert jnp.isfinite(hv2).all()
    assert jnp.isfinite(lx).all()


def test_streaming_term_periodic_and_linked_paths() -> None:
    ns, nl, nm, ny, nx, nz = 1, 2, 3, 1, 2, 8
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    dz = z[1] - z[0]
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    H = jnp.zeros((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    H = H.at[0, 0, 0, 0, 0, :].set(jnp.exp(1j * z))
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32)

    out_periodic = streaming_ladder_term(
        H, kz=kz, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m
    )
    assert out_periodic.shape == H.shape
    assert jnp.isfinite(out_periodic).all()

    idx_map = jnp.asarray([[0, 1]], dtype=jnp.int32)
    kz_link = 2.0 * jnp.pi * jnp.fft.fftfreq(2 * nz, d=dz)
    out_linked = streaming_ladder_term(
        H,
        kz=kz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        dz=dz,
        use_twist_shift=True,
        linked_indices=(idx_map,),
        linked_kz=(kz_link,),
    )
    assert out_linked.shape == H.shape
    assert jnp.isfinite(out_linked).all()


def test_streaming_term_linked_fd_and_errors() -> None:
    ns, nl, nm, ny, nx, nz = 1, 1, 3, 1, 2, 8
    H = jnp.ones((ns, nl, nm, ny, nx, nz), dtype=jnp.complex64)
    dz = jnp.asarray(0.2)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=dz)
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    vth = jnp.ones((1, 1, 1, 1, 1, 1), dtype=jnp.float32)
    kx_link_plus = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kx_link_minus = jnp.asarray([[1, 0]], dtype=jnp.int32)
    kx_mask = jnp.asarray([[True, True]])
    out = streaming_ladder_term(
        H,
        kz=kz,
        vth=vth,
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
        dz=dz,
        use_twist_shift=True,
        kx_link_plus=kx_link_plus,
        kx_link_minus=kx_link_minus,
        kx_mask_plus=kx_mask,
        kx_mask_minus=kx_mask,
    )
    assert out.shape == H.shape
    assert jnp.isfinite(out).all()

    with pytest.raises(ValueError):
        streaming_ladder_term(
            H, kz=kz, vth=vth, sqrt_p=sqrt_p, sqrt_m=sqrt_m, use_twist_shift=True
        )
    with pytest.raises(ValueError):
        streaming_ladder_term(
            H,
            kz=kz,
            vth=vth,
            sqrt_p=sqrt_p,
            sqrt_m=sqrt_m,
            dz=dz,
            use_twist_shift=True,
        )
    with pytest.raises(ValueError):
        streaming_ladder_term(
            H,
            kz=kz,
            vth=vth,
            sqrt_p=sqrt_p,
            sqrt_m=sqrt_m,
            dz=dz,
            use_twist_shift=True,
            kx_link_plus=kx_link_plus,
            kx_link_minus=kx_link_minus,
        )


def test_terms_positive_validation_checks() -> None:
    _check_positive(1.0, "x")
    _check_positive(jnp.asarray([1.0, 2.0]), "arr")
    with pytest.raises(ValueError):
        _check_positive(0.0, "x")
    with pytest.raises(ValueError):
        _check_positive(jnp.asarray([1.0, 0.0]), "arr")


def test_terms_positive_validation_skips_tracer_runtime_checks() -> None:
    @jax.jit
    def f(x: jnp.ndarray) -> jnp.ndarray:
        _check_positive(x, "x")
        return x + 1.0

    out = f(jnp.asarray(0.0))
    assert float(out) == 1.0


def test_streaming_positive_validation_is_the_shared_guard() -> None:
    """Streaming validates a concrete ``dz`` under ``jit`` like everything else.

    The streaming kernels carried their own copy of the guard, and that copy
    asked whether a ``jnp`` round trip of its argument was traced -- true of
    every argument inside a trace -- so ``dz`` and ``vth`` went unchecked in
    every jitted run. One guard now answers for the whole linear operator.
    """

    from gkx.operators.linear import params as linear_params
    from gkx.operators.linear import streaming as streaming_mod

    assert streaming_mod._check_positive is linear_params._check_positive

    seen: dict[str, str | None] = {}

    def probe(x: jnp.ndarray) -> jnp.ndarray:
        try:
            _check_positive(0.0, "dz")
            seen["dz"] = None
        except ValueError as exc:
            seen["dz"] = str(exc)
        return x

    jax.jit(probe)(jnp.asarray(1.0))
    assert seen["dz"] == "dz must be > 0"


def test_terms_package_lazy_exports() -> None:
    assert callable(term_pkg.assemble_rhs_cached)
    assert callable(term_pkg.assemble_rhs_cached_jit)
    with pytest.raises(AttributeError):
        _ = term_pkg.not_a_real_symbol


def test_terms_config_pytrees_roundtrip() -> None:
    cfg = TermConfig(streaming=0.5, nonlinear=0.25, bpar=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(cfg)
    cfg_rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert cfg_rt == cfg

    state = FieldState(phi=jnp.ones((2, 2)), apar=jnp.zeros((2, 2)), bpar=None)
    leaves_s, tree_s = jax.tree_util.tree_flatten(state)
    state_rt = jax.tree_util.tree_unflatten(tree_s, leaves_s)
    assert jnp.allclose(state_rt.phi, state.phi)
    assert jnp.allclose(state_rt.apar, state.apar)
    assert state_rt.bpar is None
