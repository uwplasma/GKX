"""The evolved state on the ``ky >= 0`` half spectrum (plan 5.3 N3).

Stage 1 (#248) wrote the layout contract and split the bracket's ``ky >= 0``
primitive out as a seam; the state itself stayed two-sided, so none of the
Hermitian completion was recovered.  This file pins the switch: a spectral grid
built in :data:`~gkx.core_ky_layout.HALF` carries ``Nyc = 1 + Ny // 2`` rows, an
evolved state on it is the same physics as the two-sided one, and the
per-stage completion and the bracket's widening are both gone rather than
cheaper.

Two facts organise the whole file.

*The layouts agree on every dealiased row.*  The linear operator is block
diagonal in ``ky`` and the bracket's kernel is the same function of the
non-negative rows either way, so the half run reproduces the two-sided run row
for row, to round-off on the nonlinear path and bitwise on the linear one.

*They disagree on an even grid's Nyquist row, by the sign of its ``ky``.*
``fftfreq`` stores ``|ky| = Ny/2`` once, as ``-Ny/2``; the half axis stores the
same row as ``+Ny/2``.  Both are legitimate representations of the same real
field -- the row is its own conjugate partner -- but they are different arrays,
and any term carrying an odd power of ``ky`` sees the difference.  Nothing in a
run does: the two-thirds mask zeroes every row at or above ``Ny/3`` and
``Ny/2`` is always above it.  The tests say both halves of that out loud.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.config import GridConfig
from gkx.core_grid import build_spectral_grid, twothirds_mask
from gkx.core_ky_layout import (
    FULL,
    HALF,
    is_half,
    ky_layout_of,
    nyc_from_ny,
    nyquist_row,
    rows_for_layout,
    source_ny_full,
    to_full,
    to_half,
)
from gkx.geometry import SAlphaGeometry
from gkx.operators.linear.cache_builder import build_linear_cache
from gkx.operators.linear.params import LinearParams
from gkx.terms.assembly import assemble_rhs_cached
from gkx.terms.config import TermConfig

EVEN_NY = (8, 12, 16)
ODD_NY = (9, 15)


# --------------------------------------------------------------------------
# the layout predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", EVEN_NY + ODD_NY)
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
    return GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=62.8, Ly=62.8)


@pytest.mark.parametrize("ny", EVEN_NY + ODD_NY)
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


@pytest.mark.parametrize("ny", EVEN_NY + ODD_NY)
def test_the_half_dealias_mask_is_built_not_sliced_and_matches_the_slice(
    ny: int,
) -> None:
    """The rows agree; stating them is what stops the agreement being luck."""

    nx = 8
    sliced = np.asarray(twothirds_mask(ny, nx))[: nyc_from_ny(ny)]
    built = np.asarray(twothirds_mask(ny, nx, ky_layout=HALF))
    np.testing.assert_array_equal(built, sliced)


@pytest.mark.parametrize("ny", EVEN_NY)
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


@pytest.mark.parametrize("ny", EVEN_NY + ODD_NY)
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


@pytest.mark.parametrize("ny", EVEN_NY)
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

    cfg = GridConfig(Nx=8, Ny=ny, Nz=8, Lx=62.8, Ly=62.8, boundary="linked", jtwist=1)
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

    cfg = GridConfig(Nx=8, Ny=ny, Nz=8, Lx=62.8, Ly=62.8, boundary="linked", jtwist=1)
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
