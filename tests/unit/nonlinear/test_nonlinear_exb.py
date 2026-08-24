import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gkx.config import GridConfig
from gkx.core.velocity import bessel_j0, bessel_j1, laguerre_transform
from gkx.core.grid import build_spectral_grid, real_fft_mesh
from gkx.terms import nonlinear as nonlinear_terms_module
from gkx.operators.nonlinear.brackets import (
    _apply_mask_xy,
    _broadcast_grid,
    _broadcast_mask,
    _broadcast_to_G,
    _spectral_bracket,
    _spectral_bracket_multi,
    _stack_fields,
)
from gkx.operators.nonlinear.projection import advance_shearing_coordinates
from gkx.core.velocity import (
    _laguerre_bpar_correction,
    _laguerre_bpar_correction_precomputed,
    _laguerre_j0_field,
    _laguerre_j0_field_precomputed,
    _laguerre_to_grid,
    _laguerre_to_spectral,
)
from gkx.terms.nonlinear import (
    _apply_flutter,
    exb_nonlinear_contribution,
    nonlinear_em_contribution,
    nonlinear_em_components,
    placeholder_nonlinear_contribution,
)


def test_nonlinear_em_contribution_masks_nonlaguerre_fields_once(monkeypatch):
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    G = jnp.ones(
        (1, 1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    phi = jnp.ones((grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    calls: list[tuple[int, ...]] = []

    def _fake_apply_mask(field, mask):
        calls.append(tuple(field.shape))
        return field

    def _fake_spectral_bracket(G_arg, chi_arg, **kwargs):
        return jnp.zeros_like(G_arg)

    monkeypatch.setattr(nonlinear_terms_module, "_apply_mask_xy", _fake_apply_mask)
    monkeypatch.setattr(
        nonlinear_terms_module, "_spectral_bracket", _fake_spectral_bracket
    )

    out = nonlinear_terms_module.nonlinear_em_contribution(
        G,
        phi=phi,
        apar=None,
        bpar=None,
        Jl=jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32),
        JlB=jnp.ones(
            (1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32
        ),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((1, 1, 1, 1, 1), dtype=jnp.float32),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=0.0,
        bpar_weight=0.0,
        laguerre_to_grid=None,
        laguerre_to_spectral=None,
        laguerre_roots=None,
        b=None,
    )

    assert out.shape == G.shape
    assert calls == [tuple(phi.shape)]


def _ifft2_xy(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.ifft2(x, axes=(-3, -2))


def _finite_diff_periodic(f: np.ndarray, dx: float, axis: int) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * dx)


def _expand_hermitian(pos: np.ndarray, ny_full: int) -> np.ndarray:
    """Expand rFFT-unique ky rows into full Hermitian spectrum."""
    nyc = ny_full // 2 + 1
    if nyc <= 2:
        return pos
    neg_hi = nyc - 1 if (ny_full % 2 == 0) else nyc
    neg = np.conj(pos[..., 1:neg_hi, :, :])
    neg = neg[..., ::-1, :, :]
    nx = pos.shape[-2]
    if nx > 1:
        kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
        neg = neg[..., kx_neg, :]
    return np.concatenate([pos, neg], axis=-3)


def test_laguerre_transform_matches_moveaxis_tensordot_reference():
    rng = np.random.default_rng(42)
    G = rng.normal(size=(2, 3, 4, 5, 6, 2)) + 1j * rng.normal(size=(2, 3, 4, 5, 6, 2))
    to_grid = rng.normal(size=(3, 5))
    G_jax = jnp.asarray(G, dtype=jnp.complex64)
    to_grid_jax = jnp.asarray(to_grid, dtype=jnp.float32)

    # The reference has to be computed at the same precision as the kernel under
    # test. _laguerre_to_grid emits no dot at all, so it keeps full float32; an
    # unpinned tensordot is lowered to TF32 on Ampere and later NVIDIA GPUs, so
    # without the pin the "reference" is the less accurate of the two and the
    # comparison measures TF32 truncation (1.8e-3) rather than agreement of the
    # two formulations.
    g_mu = _laguerre_to_grid(G_jax, to_grid_jax)
    ref_grid = jnp.moveaxis(
        jnp.tensordot(
            jnp.moveaxis(G_jax, 1, -1),
            to_grid_jax,
            axes=([-1], [0]),
            precision=jax.lax.Precision.HIGHEST,
        ),
        -1,
        1,
    )
    np.testing.assert_allclose(
        np.asarray(g_mu), np.asarray(ref_grid), rtol=1.0e-6, atol=1.0e-6
    )

    to_spectral = jnp.asarray(rng.normal(size=(5, 3)), dtype=jnp.float32)
    spectral = _laguerre_to_spectral(g_mu, to_spectral)
    ref_spectral = jnp.moveaxis(
        jnp.tensordot(
            jnp.moveaxis(g_mu, 1, -1),
            to_spectral,
            axes=([-1], [0]),
            precision=jax.lax.Precision.HIGHEST,
        ),
        -1,
        1,
    )
    np.testing.assert_allclose(
        np.asarray(spectral), np.asarray(ref_spectral), rtol=1.0e-6, atol=1.0e-6
    )


def test_velocity_and_shearing_contractions_pin_exact_dot_precision():
    """The Laguerre transforms and the shearing remap must not run in TF32.

    CPU cannot see a missing pin: an unpinned dot and Precision.HIGHEST give
    bit-identical CPU numbers. On Ampere and later NVIDIA GPUs the unpinned form is
    lowered to TF32, which truncates to 10 mantissa bits. That is destructive here
    for a reason worth stating: the shearing remap contracts against a matrix of
    exact zeros and ones, so it should return the state's own values reordered, yet
    in TF32 it rounds the state itself (measured 9.7e-4 on an RTX A4000). The jaxpr
    carries the precision request on every backend, so CPU CI can guard it.

    There are two ways to hold that invariant, and the code uses both. The
    shearing remap is a real contraction, so it pins ``Precision.HIGHEST``. The
    Laguerre transforms contract a single-digit velocity axis against the whole
    perpendicular grid; they are written as broadcast multiply-adds instead,
    which emits no ``dot_general`` at all and so has nothing for a backend to
    lower to TF32. Guarding the second form means asserting the absence of the
    primitive, not the presence of a pin on it.
    """

    def dot_equations(function, *args, **kwargs):
        equations = jax.make_jaxpr(function)(*args, **kwargs).jaxpr.eqns
        return [
            equation.params["precision"]
            for equation in equations
            if equation.primitive.name == "dot_general"
        ]

    exact = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    G = jnp.zeros(
        (1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    dot_free = {
        "laguerre_to_grid": dot_equations(
            _laguerre_to_grid, G, jnp.zeros((2, 2), dtype=jnp.float32)
        ),
        "laguerre_to_spectral": dot_equations(
            _laguerre_to_spectral, G, jnp.zeros((2, 2), dtype=jnp.float32)
        ),
    }
    pinned = {
        "shearing_remap": dot_equations(
            lambda value: advance_shearing_coordinates(
                value,
                kx=grid.kx,
                ky=grid.ky,
                x0=grid.x0,
                shear_rate=1.0,
                previous_time=0.0,
                time=1.0,
                dealias_mask=grid.dealias_mask,
            ).state,
            G,
        ),
    }
    for name, precisions in dot_free.items():
        assert not precisions, (
            f"{name} grew a dot_general {precisions}; either pin it to "
            "Precision.HIGHEST or keep the broadcast multiply-add form, or it "
            "will run in TF32 on Ampere and later NVIDIA GPUs"
        )
    for name, precisions in pinned.items():
        assert precisions, f"{name} lowered to no dot_general; the guard is vacuous"
        assert all(precision == exact for precision in precisions), (
            f"{name} has an unpinned contraction {precisions}; it will run in TF32 "
            "on Ampere and later NVIDIA GPUs"
        )


def test_exb_bracket_zero_mean_mode():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=4, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(0)
    G_real = rng.normal(size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size))
    phi_real = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size))
    G = np.fft.fft2(G_real, axes=(-3, -2))
    phi = np.fft.fft2(phi_real, axes=(0, 1))
    dG = exb_nonlinear_contribution(
        jnp.asarray(G),
        phi=jnp.asarray(phi),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
    )
    mean_mode = np.asarray(dG[..., 0, 0, :])
    assert np.max(np.abs(mean_mode)) < 1.0e-3


def test_exb_bracket_respects_dealias_mask():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(2)
    G = rng.normal(
        size=(2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(2, 2, 2, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    dG_zero = exb_nonlinear_contribution(
        jnp.asarray(G),
        phi=jnp.asarray(phi),
        dealias_mask=jnp.zeros_like(grid.dealias_mask),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    assert np.allclose(np.asarray(dG_zero), 0.0)


def test_exb_bracket_matches_finite_difference():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    phi_hat = np.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128)
    G_hat = np.zeros_like(phi_hat)
    phi_hat[1, 2, 0] = 1.0
    phi_hat[-1, -2, 0] = 1.0
    G_hat[2, 1, 0] = 0.5
    G_hat[-2, -1, 0] = 0.5
    kx = np.asarray(grid.kx_grid)
    ky = np.asarray(grid.ky_grid)
    Nxy = grid.kx.size * grid.ky.size
    dphi_dx = np.fft.ifft2(1j * kx * phi_hat[:, :, 0]) * Nxy
    dphi_dy = np.fft.ifft2(1j * ky * phi_hat[:, :, 0]) * Nxy
    dG_dx = np.fft.ifft2(1j * kx * G_hat[:, :, 0]) * Nxy
    dG_dy = np.fft.ifft2(1j * ky * G_hat[:, :, 0]) * Nxy
    bracket = dG_dx * dphi_dy - dG_dy * dphi_dx
    bracket_ref = np.fft.fft2(bracket) / Nxy
    bracket_spec = exb_nonlinear_contribution(
        jnp.asarray(G_hat[None, None, None, ...]),
        phi=jnp.asarray(phi_hat),
        dealias_mask=jnp.ones_like(grid.dealias_mask),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    bracket_spec_hat = np.asarray(bracket_spec[0, 0, 0, :, :, 0])
    assert np.max(np.abs(bracket_spec_hat - bracket_ref)) < 5.0e-6


def test_spectral_bracket_is_antisymmetric_for_periodic_fields():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    x = np.arange(grid.kx.size) * 2.0 * np.pi / grid.kx.size
    y = np.arange(grid.ky.size) * 2.0 * np.pi / grid.ky.size
    Y, X = np.meshgrid(y, x, indexing="ij")
    f = np.sin(X + 2.0 * Y) + 0.3 * np.cos(2.0 * X - Y)
    g = np.cos(2.0 * X + Y) - 0.2 * np.sin(X - Y)
    nxy = grid.kx.size * grid.ky.size
    f_hat = (np.fft.fft2(f) / nxy)[:, :, None]
    g_hat = (np.fft.fft2(g) / nxy)[:, :, None]
    full_mask = jnp.ones_like(grid.dealias_mask)

    bracket_fg = _spectral_bracket(
        jnp.asarray(f_hat[None, None, None, ...]),
        jnp.asarray(g_hat),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=full_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    bracket_gf = _spectral_bracket(
        jnp.asarray(g_hat[None, None, None, ...]),
        jnp.asarray(f_hat),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=full_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )

    np.testing.assert_allclose(np.asarray(bracket_fg + bracket_gf), 0.0, atol=1.0e-6)


def test_exb_bracket_free_energy_inner_product_vanishes_for_periodic_scalar():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    x = np.arange(grid.kx.size) * 2.0 * np.pi / grid.kx.size
    y = np.arange(grid.ky.size) * 2.0 * np.pi / grid.ky.size
    Y, X = np.meshgrid(y, x, indexing="ij")
    G_real = np.sin(X + 2.0 * Y) + 0.3 * np.cos(2.0 * X - Y)
    phi_real = np.cos(2.0 * X + Y) - 0.2 * np.sin(X - Y)
    nxy = grid.kx.size * grid.ky.size
    G_hat = (np.fft.fft2(G_real) / nxy)[:, :, None]
    phi_hat = (np.fft.fft2(phi_real) / nxy)[:, :, None]
    full_mask = jnp.ones_like(grid.dealias_mask)

    dG_hat = exb_nonlinear_contribution(
        jnp.asarray(G_hat[None, None, None, ...]),
        phi=jnp.asarray(phi_hat),
        dealias_mask=full_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    dG_real = np.fft.ifft2(np.asarray(dG_hat[0, 0, 0, :, :, 0])) * nxy
    free_energy_rate = np.mean(np.real(np.conj(G_real) * dG_real))

    assert np.max(np.abs(dG_real)) > 1.0
    assert abs(float(free_energy_rate)) < 1.0e-6


def test_apar_flutter_hermite_ladder():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(3)
    G = rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size))
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    vth = jnp.asarray([1.2])
    sqrt_m = jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None]
    sqrt_m_p1 = jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[None, :, None, None, None]

    apar_masked = _apply_mask_xy(jnp.asarray(apar), grid.dealias_mask)
    bracket_apar = _spectral_bracket(
        jnp.asarray(G),
        Jl * apar_masked[None, None, ...],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    flutter_expected = _apply_flutter(bracket_apar, vth, sqrt_m, sqrt_m_p1)

    dG = nonlinear_em_contribution(
        jnp.asarray(G),
        phi=jnp.zeros_like(jnp.asarray(apar)),
        apar=apar_masked,
        bpar=None,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=vth,
        sqrt_m=sqrt_m,
        sqrt_m_p1=sqrt_m_p1,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
    )
    assert np.max(np.abs(np.asarray(dG - flutter_expected))) < 1.0e-6


def test_apar_flutter_no_wrap_boundary():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    Nm = 4
    bracket = jnp.zeros(
        (1, 1, Nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64
    )
    bracket = bracket.at[:, :, -1, 0, 0, 0].set(1.0 + 0.0j)
    sqrt_m = jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None]
    sqrt_m_p1 = jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[
        None, :, None, None, None
    ]
    flutter = _apply_flutter(bracket, jnp.asarray([1.0]), sqrt_m, sqrt_m_p1)
    flutter_np = np.asarray(flutter)
    assert np.allclose(flutter_np[:, :, 0, 0, 0, 0], 0.0)
    assert np.allclose(flutter_np[:, :, -1, 0, 0, 0], 0.0)
    expected = -np.sqrt(Nm - 1)
    assert np.allclose(flutter_np[:, :, -2, 0, 0, 0], expected)


def test_em_weights_are_toggles():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(5)
    G = rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size))
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    common = dict(
        phi=jnp.zeros_like(jnp.asarray(apar)),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    dG_on = nonlinear_em_contribution(
        jnp.asarray(G),
        apar_weight=1.0,
        bpar_weight=1.0,
        weight=jnp.asarray(1.0),
        **common,
    )
    dG_off = nonlinear_em_contribution(
        jnp.asarray(G),
        apar_weight=1.0,
        bpar_weight=1.0,
        weight=jnp.asarray(0.0),
        **common,
    )
    assert np.max(np.abs(np.asarray(dG_off))) < 1.0e-8
    assert np.max(np.abs(np.asarray(dG_on))) > 0.0


def test_compressed_real_fft_bracket_match():
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(42)
    G_nyc = rng.normal(size=(1, 1, 1, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 1, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )

    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(phi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )

    Nxy = nx * ny
    _, _, kx_nyc, ky_nyc = real_fft_mesh(grid.kx_grid, grid.ky_grid)
    kx_b = kx_nyc[None, None, None, :, :, None]
    ky_b = ky_nyc[None, None, None, :, :, None]
    kx_c = kx_nyc[None, None, :, :, None]
    ky_c = ky_nyc[None, None, :, :, None]

    dG_dx = np.fft.irfft2(1j * kx_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dG_dy = np.fft.irfft2(1j * ky_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dx = np.fft.irfft2(1j * kx_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dy = np.fft.irfft2(1j * ky_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    bracket = dG_dx * dphi_dy[:, :, None, ...] - dG_dy * dphi_dx[:, :, None, ...]
    bracket_hat_nyc = np.fft.rfft2(bracket, axes=(-2, -3)) / Nxy
    mask_nyc = np.asarray(grid.dealias_mask)[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * mask_nyc[None, None, None, :, :, None]
    bracket_hat_ref = _expand_hermitian(bracket_hat_nyc, ny)

    assert np.allclose(
        np.asarray(bracket_hat),
        bracket_hat_ref,
        rtol=1.0e-4,
        atol=2.0e-5,
    )


def test_compressed_real_fft_toggle_matches_full_fft_for_hermitian():
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    ny = grid.ky.size
    nx = grid.kx.size
    rng = np.random.default_rng(123)
    normalization = float(ny * nx)
    G_full = (
        np.fft.fft2(
            rng.normal(size=(1, 1, 1, ny, nx, grid.z.size)),
            axes=(-3, -2),
        )
        / normalization
    )
    chi_full = (
        np.fft.fft2(
            rng.normal(size=(ny, nx, grid.z.size)),
            axes=(-3, -2),
        )
        / normalization
    )
    mask = np.asarray(grid.dealias_mask)
    G_full *= mask[None, None, None, :, :, None]
    chi_full *= mask[:, :, None]

    bracket_compressed = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(chi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )
    bracket_full = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(chi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    np.testing.assert_allclose(
        bracket_compressed, bracket_full, rtol=2.0e-5, atol=2.0e-6
    )

    # At an integer radial shift the residual phase is unity. Both
    # representations must therefore produce the same post-remap bracket.
    state_update = advance_shearing_coordinates(
        jnp.asarray(G_full),
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=1.0,
        dealias_mask=grid.dealias_mask,
    )
    field_update = advance_shearing_coordinates(
        jnp.asarray(chi_full),
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=1.0,
        dealias_mask=grid.dealias_mask,
    )
    np.testing.assert_allclose(state_update.phase, 1.0, atol=2.0e-6)
    bracket_remapped_compressed = _spectral_bracket(
        state_update.state,
        field_update.state,
        kx_grid=state_update.effective_kx,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )
    bracket_remapped_full = _spectral_bracket(
        state_update.state,
        field_update.state,
        kx_grid=state_update.effective_kx,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        radial_phase=state_update.phase,
        compressed_real_fft=False,
    )
    np.testing.assert_allclose(
        bracket_remapped_compressed,
        bracket_remapped_full,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    # At a fractional shift, the full kernel carries the residual radial phase
    # explicitly. The compressed kernel evaluates the equivalent canonical
    # shearing-coordinate bracket with the base radial wavenumbers.
    state_fractional = advance_shearing_coordinates(
        jnp.asarray(G_full),
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=0.23,
        dealias_mask=grid.dealias_mask,
    )
    field_fractional = advance_shearing_coordinates(
        jnp.asarray(chi_full),
        kx=grid.kx,
        ky=grid.ky,
        x0=grid.x0,
        shear_rate=1.0,
        previous_time=0.0,
        time=0.23,
        dealias_mask=grid.dealias_mask,
    )
    assert float(jnp.max(jnp.abs(state_fractional.phase - 1.0))) > 0.1
    bracket_fractional_full = _spectral_bracket(
        state_fractional.state,
        field_fractional.state,
        kx_grid=state_fractional.effective_kx,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        radial_phase=state_fractional.phase,
        compressed_real_fft=False,
    )
    bracket_fractional_compressed = _spectral_bracket(
        state_fractional.state,
        field_fractional.state,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )
    np.testing.assert_allclose(
        bracket_fractional_compressed,
        bracket_fractional_full,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    def bracket_norm(scale):
        bracket = _spectral_bracket(
            scale * state_fractional.state,
            field_fractional.state,
            kx_grid=state_fractional.effective_kx,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            radial_phase=state_fractional.phase,
            compressed_real_fft=True,
        )
        return jnp.real(jnp.vdot(bracket, bracket))

    scale = jnp.asarray(1.0, dtype=jnp.float32)
    _, tangent = jax.jvp(bracket_norm, (scale,), (jnp.ones_like(scale),))
    step = jnp.asarray(1.0e-3, dtype=scale.dtype)
    finite_difference = (bracket_norm(scale + step) - bracket_norm(scale - step)) / (
        2.0 * step
    )
    np.testing.assert_allclose(tangent, finite_difference, rtol=2.0e-4)


def test_full_fft_shearing_phase_preserves_poisson_bracket_coordinates():
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=6, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(401)
    G = rng.normal(size=(1, 1, 1, 6, 6, 1)) + 1j * rng.normal(size=(1, 1, 1, 6, 6, 1))
    chi = rng.normal(size=(6, 6, 1)) + 1j * rng.normal(size=(6, 6, 1))
    mask = np.asarray(grid.dealias_mask)
    G *= mask[None, None, None, :, :, None]
    chi *= mask[:, :, None]

    reference = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    radial_x = 2.0 * jnp.pi * grid.x0 * jnp.arange(6) / 6.0
    shear_displacement = jnp.asarray(0.23)
    residual_kx = -grid.ky * shear_displacement
    phase = jnp.exp(1j * residual_kx[:, None] * radial_x[None, :])
    effective_kx = grid.kx_grid + residual_kx[:, None]
    sheared = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi),
        kx_grid=effective_kx,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        radial_phase=phase,
        compressed_real_fft=False,
    )
    np.testing.assert_allclose(sheared, reference, rtol=2.0e-5, atol=2.0e-5)

    with pytest.raises(ValueError, match="radial_phase must have shape"):
        _spectral_bracket(
            jnp.asarray(G),
            jnp.asarray(chi),
            kx_grid=effective_kx,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            radial_phase=phase[:, :-1],
            compressed_real_fft=True,
        )


def test_compressed_real_fft_bracket_preserves_full_hermitian_symmetry():
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=8, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(7)
    G_nyc = rng.normal(size=(1, 1, 2, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 2, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )
    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = np.asarray(
        _spectral_bracket(
            jnp.asarray(G_full),
            jnp.asarray(phi_full),
            kx_grid=grid.kx_grid,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            compressed_real_fft=True,
        )
    )

    neg_hi = nyc - 1 if (ny % 2 == 0) else nyc
    kx_neg = np.concatenate(([0], np.arange(nx - 1, 0, -1)))
    neg_ref = np.conj(bracket_hat[..., 1:neg_hi, :, :])[..., ::-1, :, :]
    neg_ref = neg_ref[..., kx_neg, :]
    assert np.allclose(bracket_hat[..., nyc:, :, :], neg_ref, rtol=1.0e-6, atol=1.0e-6)


def test_compressed_real_fft_bracket_match_odd_ny():
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=7, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    ny = grid.ky.size
    nx = grid.kx.size
    nyc = ny // 2 + 1
    rng = np.random.default_rng(321)
    G_nyc = rng.normal(size=(1, 1, 1, nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(1, 1, 1, nyc, nx, grid.z.size)
    )
    phi_nyc = rng.normal(size=(nyc, nx, grid.z.size)) + 1j * rng.normal(
        size=(nyc, nx, grid.z.size)
    )
    G_full = _expand_hermitian(G_nyc, ny)
    phi_full = _expand_hermitian(phi_nyc[None, ...], ny)[0]

    bracket_hat = _spectral_bracket(
        jnp.asarray(G_full),
        jnp.asarray(phi_full),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )

    Nxy = nx * ny
    _, _, kx_nyc, ky_nyc = real_fft_mesh(grid.kx_grid, grid.ky_grid)
    kx_b = kx_nyc[None, None, None, :, :, None]
    ky_b = ky_nyc[None, None, None, :, :, None]
    kx_c = kx_nyc[None, None, :, :, None]
    ky_c = ky_nyc[None, None, :, :, None]

    dG_dx = np.fft.irfft2(1j * kx_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dG_dy = np.fft.irfft2(1j * ky_b * G_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dx = np.fft.irfft2(1j * kx_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    dphi_dy = np.fft.irfft2(1j * ky_c * phi_nyc, s=(nx, ny), axes=(-2, -3)) * Nxy
    bracket = dG_dx * dphi_dy[:, :, None, ...] - dG_dy * dphi_dx[:, :, None, ...]
    bracket_hat_nyc = np.fft.rfft2(bracket, axes=(-2, -3)) / Nxy
    mask_nyc = np.asarray(grid.dealias_mask)[:nyc, :]
    bracket_hat_nyc = bracket_hat_nyc * mask_nyc[None, None, None, :, :, None]
    bracket_hat_ref = _expand_hermitian(bracket_hat_nyc, ny)

    assert np.allclose(
        np.asarray(bracket_hat),
        bracket_hat_ref,
        rtol=1.0e-4,
        atol=2.0e-5,
    )


def test_bpar_contributes_to_chi():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(4)
    G = rng.normal(
        size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = 2.0 * jnp.ones_like(Jl)
    phi_masked = _apply_mask_xy(jnp.asarray(phi), grid.dealias_mask)
    bpar_masked = _apply_mask_xy(jnp.asarray(bpar), grid.dealias_mask)
    chi = Jl * phi_masked[None, None, ...] + JlB * bpar_masked[None, None, ...]
    bracket_expected = _spectral_bracket(
        jnp.asarray(G),
        chi,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
    )
    dG = nonlinear_em_contribution(
        jnp.asarray(G),
        phi=phi_masked,
        apar=None,
        bpar=bpar_masked,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(2, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 3, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
    )
    assert np.max(np.abs(np.asarray(dG - bracket_expected))) < 1.0e-5


def test_bracket_multi_matches_separate():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(11)
    G = rng.normal(
        size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = 2.0 * jnp.ones_like(Jl)
    chi_phi = Jl * phi[None, None, ...]
    chi_bpar = JlB * bpar[None, None, ...]
    chi_stack = _stack_fields(
        jnp.asarray(G), [jnp.asarray(chi_phi), jnp.asarray(chi_bpar)]
    )
    brackets = _spectral_bracket_multi(
        jnp.asarray(G),
        chi_stack,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    bracket_phi = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi_phi),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    bracket_bpar = _spectral_bracket(
        jnp.asarray(G),
        jnp.asarray(chi_bpar),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    assert np.allclose(
        np.asarray(brackets[0]), np.asarray(bracket_phi), rtol=1.0e-6, atol=1.0e-6
    )
    assert np.allclose(
        np.asarray(brackets[1]), np.asarray(bracket_bpar), rtol=1.0e-6, atol=1.0e-6
    )


def test_single_field_bracket_fast_path_matches_multi_reference():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=6, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(41)
    G = rng.normal(
        size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 2, 3, grid.ky.size, grid.kx.size, grid.z.size))
    chi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    chi_stack = _stack_fields(jnp.asarray(G), [jnp.asarray(chi)])

    for compressed_real_fft in (False, True):
        single = _spectral_bracket(
            jnp.asarray(G),
            jnp.asarray(chi),
            kx_grid=grid.kx_grid,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            compressed_real_fft=compressed_real_fft,
        )
        multi = _spectral_bracket_multi(
            jnp.asarray(G),
            chi_stack,
            kx_grid=grid.kx_grid,
            ky_grid=grid.ky_grid,
            dealias_mask=grid.dealias_mask,
            kxfac=jnp.asarray(1.0),
            compressed_real_fft=compressed_real_fft,
        )[0]
        assert np.allclose(
            np.asarray(single), np.asarray(multi), rtol=1.0e-6, atol=1.0e-6
        )


def test_laguerre_precompute_matches_direct():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(21)
    Ns, Nl, Nm = 1, 2, 2
    G = rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    bpar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    b = np.full((Ns, grid.ky.size, grid.kx.size, grid.z.size), 0.05, dtype=np.float64)
    lag_to_grid, lag_to_spec, lag_roots = laguerre_transform(Nl)
    laguerre_to_grid = jnp.asarray(lag_to_grid, dtype=jnp.float64)
    laguerre_to_spectral = jnp.asarray(lag_to_spec, dtype=jnp.float64)
    laguerre_roots = jnp.asarray(lag_roots, dtype=jnp.float64)
    alpha = jnp.sqrt(
        jnp.maximum(
            0.0,
            2.0
            * laguerre_roots[None, :, None, None, None]
            * jnp.asarray(b)[:, None, ...],
        )
    )
    j0 = bessel_j0(alpha)
    j1 = bessel_j1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)

    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        b=jnp.asarray(b),
        compressed_real_fft=False,
    )
    dG_direct = nonlinear_em_contribution(jnp.asarray(G), **common)
    dG_cached = nonlinear_em_contribution(
        jnp.asarray(G),
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        **common,
    )
    assert np.allclose(
        np.asarray(dG_direct), np.asarray(dG_cached), rtol=1.0e-6, atol=1.0e-6
    )


def test_laguerre_mode_spectral_matches_identity_grid():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(12)
    Ns, Nl, Nm = 1, 1, 1
    G = rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    laguerre_to_grid = jnp.asarray([[1.0]], dtype=jnp.float64)
    laguerre_to_spectral = jnp.asarray([[1.0]], dtype=jnp.float64)
    laguerre_roots = jnp.asarray([0.0], dtype=jnp.float64)
    b = jnp.zeros((Ns, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j0 = jnp.ones((Ns, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j1_over_alpha = jnp.full_like(j0, 0.5)

    common = dict(
        phi=jnp.asarray(phi),
        apar=None,
        bpar=None,
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.zeros((Ns, Nm, 1, 1, 1), dtype=jnp.float32),
        sqrt_m_p1=jnp.ones((Ns, Nm, 1, 1, 1), dtype=jnp.float32),
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=0.0,
        bpar_weight=0.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        b=b,
        compressed_real_fft=False,
    )

    dG_grid = nonlinear_em_contribution(jnp.asarray(G), laguerre_mode="grid", **common)
    dG_spec = nonlinear_em_contribution(
        jnp.asarray(G), laguerre_mode="spectral", **common
    )
    assert np.allclose(
        np.asarray(dG_grid), np.asarray(dG_spec), rtol=1.0e-6, atol=1.0e-6
    )


def test_laguerre_grid_electrostatic_fast_path_matches_component_reference():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(42)
    Ns, Nl, Nm = 1, 2, 2
    G = rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    lag_to_grid, lag_to_spec, lag_roots = laguerre_transform(Nl)
    b = jnp.full((Ns, grid.ky.size, grid.kx.size, grid.z.size), 0.04, dtype=jnp.float32)
    laguerre_roots = jnp.asarray(lag_roots, dtype=jnp.float32)
    alpha = jnp.sqrt(
        jnp.maximum(
            0.0,
            2.0 * laguerre_roots[None, :, None, None, None] * b[:, None, ...],
        )
    )
    j0 = bessel_j0(alpha)

    common = dict(
        phi=jnp.asarray(phi),
        apar=None,
        bpar=None,
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=0.0,
        bpar_weight=0.0,
        laguerre_to_grid=jnp.asarray(lag_to_grid, dtype=jnp.float32),
        laguerre_to_spectral=jnp.asarray(lag_to_spec, dtype=jnp.float32),
        laguerre_roots=laguerre_roots,
        laguerre_j0=j0,
        laguerre_j1_over_alpha=None,
        b=b,
        compressed_real_fft=False,
    )

    contribution = nonlinear_em_contribution(
        jnp.asarray(G), laguerre_mode="grid", **common
    )
    components = nonlinear_em_components(jnp.asarray(G), laguerre_mode="grid", **common)
    assert np.allclose(
        np.asarray(contribution),
        np.asarray(components["total"]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_nonlinear_components_total_matches_contribution():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(21)
    G = rng.normal(
        size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 3, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    Jl = jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    JlB = jnp.ones_like(Jl)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=None,
        Jl=Jl,
        JlB=JlB,
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(3, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 4, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=0.0,
        compressed_real_fft=False,
    )
    dG = nonlinear_em_contribution(jnp.asarray(G), **common)
    comps = nonlinear_em_components(jnp.asarray(G), laguerre_mode="spectral", **common)
    assert np.allclose(
        np.asarray(dG), np.asarray(comps["total"]), rtol=1.0e-6, atol=1.0e-6
    )


def test_laguerre_components_total_matches_contribution():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(22)
    Ns, Nl, Nm = 1, 1, 2
    G = rng.normal(
        size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(Ns, Nl, Nm, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    laguerre_to_grid = jnp.asarray([[1.0], [1.0]], dtype=jnp.float64).T
    laguerre_to_spectral = jnp.asarray([[1.0], [1.0]], dtype=jnp.float64)
    laguerre_roots = jnp.asarray([0.0], dtype=jnp.float64)
    b = jnp.zeros((Ns, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j0 = jnp.ones((Ns, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float64)
    j1_over_alpha = jnp.full_like(j0, 0.5)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=None,
        Jl=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        JlB=jnp.ones((Ns, Nl, grid.ky.size, grid.kx.size, grid.z.size)),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(Nm, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, Nm + 1, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=0.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=laguerre_roots,
        laguerre_j0=j0,
        laguerre_j1_over_alpha=j1_over_alpha,
        b=b,
        compressed_real_fft=False,
    )
    dG = nonlinear_em_contribution(jnp.asarray(G), laguerre_mode="grid", **common)
    comps = nonlinear_em_components(jnp.asarray(G), laguerre_mode="grid", **common)
    assert np.allclose(
        np.asarray(dG), np.asarray(comps["total"]), rtol=1.0e-6, atol=1.0e-6
    )


def test_exb_bracket_energy_conserves():
    grid = build_spectral_grid(
        GridConfig(Nx=8, Ny=8, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    G_hat = np.zeros(
        (1, 1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128
    )
    phi_hat = np.zeros((grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex128)
    G_hat[0, 0, 0, 1, 2, 0] = 1.0
    phi_hat[1, 2, 0] = 2.0
    dG = exb_nonlinear_contribution(
        jnp.asarray(G_hat),
        phi=jnp.asarray(phi_hat),
        dealias_mask=grid.dealias_mask,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        weight=jnp.asarray(1.0),
        compressed_real_fft=False,
    )
    assert np.max(np.abs(np.asarray(dG))) < 1.0e-6


def test_nonlinear_broadcast_helpers_cover_shape_contracts():
    mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
    grid = jnp.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=jnp.float32)
    field = jnp.ones((2, 2, 3), dtype=jnp.complex64)
    G = jnp.zeros((1, 2, 3, 2, 2, 3), dtype=jnp.complex64)

    mask_b = _broadcast_mask(mask, 5)
    grid_b = _broadcast_grid(grid, 5)
    assert mask_b.shape == (1, 1, 2, 2, 1)
    assert grid_b.shape == (1, 1, 2, 2, 1)

    masked = _apply_mask_xy(field, mask)
    assert masked.shape == field.shape
    assert np.allclose(np.asarray(masked[0, 1, :]), 0.0)
    assert np.allclose(np.asarray(_apply_mask_xy(field, None)), np.asarray(field))

    same = _broadcast_to_G(G, G)
    assert same.shape == G.shape
    assert _broadcast_to_G(field, G).shape == (1, 1, 1, 2, 2, 3)
    assert _broadcast_to_G(jnp.ones((2, 3), dtype=jnp.float32), G).shape == (
        1,
        1,
        1,
        1,
        2,
        3,
    )


def test_laguerre_transform_helpers_roundtrip():
    rng = np.random.default_rng(1234)
    G = rng.normal(size=(1, 2, 3, 2, 2, 1)) + 1j * rng.normal(size=(1, 2, 3, 2, 2, 1))
    laguerre_to_grid = np.asarray([[1.0, 0.25], [0.0, 1.0]], dtype=np.float64)
    laguerre_to_spectral = np.linalg.inv(laguerre_to_grid)

    g_mu = _laguerre_to_grid(jnp.asarray(G), jnp.asarray(laguerre_to_grid))
    restored = _laguerre_to_spectral(g_mu, jnp.asarray(laguerre_to_spectral))
    assert g_mu.shape == (1, 2, 3, 2, 2, 1)
    assert np.allclose(np.asarray(restored), G, rtol=1.0e-6, atol=1.0e-6)


def test_precomputed_bessel_helpers_match_direct_evaluation():
    field = jnp.asarray([[[1.0 + 0.5j]]], dtype=jnp.complex64)
    b = jnp.asarray([[[0.05]]], dtype=jnp.float32)
    roots = jnp.asarray([0.0, 1.5], dtype=jnp.float32)
    tz = jnp.asarray([1.0], dtype=jnp.float32)

    direct_j0 = _laguerre_j0_field(field, b, roots, 1.0)
    b_species = b[None, ...]
    alpha = jnp.sqrt(
        jnp.maximum(
            0.0, 2.0 * roots[None, :, None, None, None] * b_species[:, None, ...]
        )
    )
    from gkx.core.velocity import bessel_j0 as _b0, bessel_j1 as _b1

    j0 = _b0(alpha)
    j1 = _b1(alpha)
    j1_over_alpha = jnp.where(alpha < 1.0e-8, 0.5, j1 / alpha)
    cached_j0 = _laguerre_j0_field_precomputed(field, j0, 1.0)
    assert np.allclose(
        np.asarray(direct_j0), np.asarray(cached_j0), rtol=1.0e-6, atol=1.0e-6
    )

    bpar = jnp.asarray([[[0.3 + 0.1j]]], dtype=jnp.complex64)
    direct_bpar = _laguerre_bpar_correction(bpar, b, roots, tz, 1.0)
    cached_bpar = _laguerre_bpar_correction_precomputed(
        bpar, j1_over_alpha, roots, tz, 1.0
    )
    assert np.allclose(
        np.asarray(direct_bpar), np.asarray(cached_bpar), rtol=1.0e-6, atol=1.0e-6
    )
    cached_scalar_tz = _laguerre_bpar_correction_precomputed(
        bpar,
        j1_over_alpha,
        roots,
        jnp.asarray(1.0, dtype=jnp.float32),
        1.0,
    )
    assert cached_scalar_tz.shape == cached_bpar.shape

    scalar_j0 = _laguerre_j0_field(field, b, jnp.asarray(0.0, dtype=jnp.float32), 1.0)
    scalar_bpar = _laguerre_bpar_correction(
        bpar,
        b,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        1.0,
    )
    assert scalar_j0.shape == (1, 1, 1, 1, 1)
    assert scalar_bpar.shape == (1, 1, 1, 1, 1)

    flutter_scalar_vth = _apply_flutter(
        jnp.ones((1, 1, 2, 1, 1, 1), dtype=jnp.complex64),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.sqrt(jnp.arange(2, dtype=jnp.float32))[None, :, None, None, None],
        jnp.sqrt(jnp.arange(1, 3, dtype=jnp.float32))[None, :, None, None, None],
    )
    assert flutter_scalar_vth.shape == (1, 1, 2, 1, 1, 1)


def test_stack_fields_and_placeholder_output_contract():
    G = jnp.zeros((1, 2, 3, 4, 5, 1), dtype=jnp.complex64)
    phi = jnp.ones((4, 5, 1), dtype=jnp.complex64)
    apar = 2.0 * jnp.ones((1, 1, 4, 5, 1), dtype=jnp.complex64)

    stacked = _stack_fields(G, [phi, apar])
    assert stacked.shape == (2, 1, 1, 1, 4, 5, 1)
    assert np.allclose(np.asarray(stacked[0, 0, 0, 0]), np.asarray(phi))
    assert np.allclose(np.asarray(stacked[1, 0, 0, 0]), 2.0)

    out = placeholder_nonlinear_contribution(G, weight=jnp.asarray(3.0))
    assert out.shape == G.shape
    assert np.allclose(np.asarray(out), 0.0)


def test_spectral_bracket_explicit_fft_norm_and_single_ky_branch():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(33)
    G = rng.normal(
        size=(1, 1, 1, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 1, grid.ky.size, grid.kx.size, grid.z.size))
    chi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    chi_stack = _stack_fields(jnp.asarray(G), [jnp.asarray(chi)])
    norm = float(grid.ky.size * grid.kx.size)

    bracket_gx = _spectral_bracket_multi(
        jnp.asarray(G),
        chi_stack,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        fft_norm=norm,
        compressed_real_fft=True,
    )
    bracket_full = _spectral_bracket_multi(
        jnp.asarray(G),
        chi_stack,
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        fft_norm=norm,
        compressed_real_fft=False,
    )
    assert bracket_gx.shape == bracket_full.shape == (1,) + G.shape

    single_G = jnp.ones((1, 1, 1, 1, 2, 1), dtype=jnp.complex64)
    single_chi = _stack_fields(single_G, [jnp.ones((1, 2, 1), dtype=jnp.complex64)])
    single_out = _spectral_bracket_multi(
        single_G,
        single_chi,
        kx_grid=jnp.zeros((1, 2), dtype=jnp.float32),
        ky_grid=jnp.zeros((1, 2), dtype=jnp.float32),
        dealias_mask=jnp.ones((1, 2), dtype=bool),
        kxfac=jnp.asarray(1.0),
        compressed_real_fft=True,
    )
    assert single_out.shape == (1, 1, 1, 1, 1, 2, 1)


def test_squeezed_em_components_match_contribution_with_4d_gyroaverages():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(34)
    G = rng.normal(
        size=(1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 2, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    bpar = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=jnp.ones((1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32),
        JlB=jnp.ones((1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(2, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 3, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
        compressed_real_fft=False,
    )

    contribution = nonlinear_em_contribution(jnp.asarray(G), **common)
    components = nonlinear_em_components(
        jnp.asarray(G), laguerre_mode="spectral", **common
    )
    assert contribution.shape == G.shape
    assert components["total"].shape == G.shape
    assert np.allclose(
        np.asarray(contribution),
        np.asarray(components["total"]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_laguerre_components_without_precomputed_bessel_include_bpar_and_apar():
    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=4, Nz=1, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    rng = np.random.default_rng(35)
    G = rng.normal(
        size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size)
    ) + 1j * rng.normal(size=(1, 1, 2, grid.ky.size, grid.kx.size, grid.z.size))
    phi = rng.normal(size=(grid.ky.size, grid.kx.size, grid.z.size)) + 1j * rng.normal(
        size=(grid.ky.size, grid.kx.size, grid.z.size)
    )
    apar = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    bpar = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    laguerre_to_grid = jnp.asarray([[1.0, 0.25]], dtype=jnp.float32)
    laguerre_to_spectral = jnp.asarray([[1.0], [0.0]], dtype=jnp.float32)
    common = dict(
        phi=jnp.asarray(phi),
        apar=jnp.asarray(apar),
        bpar=jnp.asarray(bpar),
        Jl=jnp.ones((1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32),
        JlB=jnp.ones(
            (1, 1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32
        ),
        tz=jnp.asarray([1.0]),
        vth=jnp.asarray([1.0]),
        sqrt_m=jnp.sqrt(jnp.arange(2, dtype=jnp.float32))[None, :, None, None, None],
        sqrt_m_p1=jnp.sqrt(jnp.arange(1, 3, dtype=jnp.float32))[
            None, :, None, None, None
        ],
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
        kxfac=jnp.asarray(1.0),
        weight=jnp.asarray(1.0),
        apar_weight=1.0,
        bpar_weight=1.0,
        laguerre_to_grid=laguerre_to_grid,
        laguerre_to_spectral=laguerre_to_spectral,
        laguerre_roots=jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        b=jnp.zeros((1, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32),
        compressed_real_fft=False,
    )

    contribution = nonlinear_em_contribution(
        jnp.asarray(G), laguerre_mode="grid", **common
    )
    components = nonlinear_em_components(jnp.asarray(G), laguerre_mode="grid", **common)
    assert components["exb_bpar"].shape == G.shape
    assert components["flutter"].shape == G.shape
    assert np.allclose(
        np.asarray(contribution),
        np.asarray(components["total"]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )

    precomputed = nonlinear_em_components(
        jnp.asarray(G),
        laguerre_mode="grid",
        laguerre_j0=jnp.ones(
            (1, 2, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.float32
        ),
        laguerre_j1_over_alpha=jnp.full(
            (1, 2, grid.ky.size, grid.kx.size, grid.z.size), 0.5, dtype=jnp.float32
        ),
        **common,
    )
    assert precomputed["exb_bpar"].shape == G.shape
