"""Contract tests for the ``ky`` axis layout (plan 5.3 N3).

GKX evolves the ``ky >= 0`` rows by default, as GX, stella and GS2 do, and keeps
the two-sided axis -- which rebuilds its negative half by the reality condition
-- as an explicit opt-out.  The conversion between the two layouts is one rule
with one owner instead of five hand-written copies, and the rule is pinned here
where it is easy to get wrong: at ``ky = 0``, at the Nyquist row, and on an odd
``Ny`` where ``Nyc`` does not determine ``Ny``.

Most tests here pin the contract itself and name the layout they build.
``test_the_default_grid_weights_its_nyquist_row_once`` is the exception: it
builds its grid with no layout named, so it checks the rule on the axis a run
actually takes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gkx.core_ky_layout import (
    HALF,
    conjugate_kx_order,
    describe,
    half_dealias_mask,
    half_ky_values,
    ky_row_weights,
    negative_ky_block,
    ny_full_candidates,
    nyc_from_ny,
    nyquist_row,
    paired_row_limit,
    reality_residual,
    rows_for_layout,
    self_conjugate_rows,
    source_ky_layout,
    source_ny_full,
    symmetrize_self_conjugate_rows,
    to_full,
    to_half,
)

EVEN_NY = (2, 4, 8, 16, 32)
ODD_NY = (3, 5, 9, 17)
ALL_NY = EVEN_NY + ODD_NY


def _half_of_real_field(ny: int, nx: int, nz: int, *, seed: int) -> np.ndarray:
    """Return the ``ky >= 0`` spectrum of a random real ``(y, x, z)`` field.

    ``rfft2`` over ``(kx, ky)`` is the bracket's own transform, so this is the
    half block the code actually produces, not an idealization of it.
    """

    rng = np.random.default_rng(seed)
    physical = rng.normal(size=(ny, nx, nz))
    return np.fft.rfft2(physical, axes=(1, 0)).astype(np.complex128)


def _real_field(ny: int, nx: int, nz: int, *, seed: int) -> np.ndarray:
    """Return a two-sided spectrum that satisfies the reality condition exactly.

    Built by widening the half block, so the redundant rows are equal to their
    partners' conjugates bit for bit.  A two-sided ``fft2`` of the same field
    is Hermitian only to roundoff, which would turn every exactness check in
    this file into a tolerance check on numpy's FFT.
    """

    return to_full(_half_of_real_field(ny, nx, nz, seed=seed), ny_full=ny)


def _batched(spectrum: np.ndarray) -> np.ndarray:
    return spectrum[None, None, None, ...]


# --------------------------------------------------------------------------
# Axis arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_nyc_matches_the_rfft_row_count(ny: int) -> None:
    assert nyc_from_ny(ny) == int(np.fft.rfftfreq(ny).size)


@pytest.mark.parametrize("ny", ALL_NY)
def test_nyc_does_not_determine_ny(ny: int) -> None:
    """Both candidates map back to the same ``Nyc``, so widening needs ``Ny``."""

    even, odd = ny_full_candidates(nyc_from_ny(ny))
    assert ny in (even, odd)
    assert even % 2 == 0 and odd % 2 == 1
    assert nyc_from_ny(even) == nyc_from_ny(odd) == nyc_from_ny(ny)


@pytest.mark.parametrize("ny", EVEN_NY)
def test_even_grids_have_a_nyquist_row_at_the_top_of_the_half_axis(ny: int) -> None:
    row = nyquist_row(ny)
    assert row == nyc_from_ny(ny) - 1
    assert self_conjugate_rows(ny) == ((0,) if row == 0 else (0, row))


@pytest.mark.parametrize("ny", ODD_NY)
def test_odd_grids_have_no_nyquist_row(ny: int) -> None:
    assert nyquist_row(ny) is None
    assert self_conjugate_rows(ny) == (0,)
    # Every stored row above zero owns a distinct partner.
    assert paired_row_limit(ny) == nyc_from_ny(ny)


def test_axis_arithmetic_rejects_degenerate_lengths() -> None:
    with pytest.raises(ValueError):
        nyc_from_ny(0)
    with pytest.raises(ValueError):
        ny_full_candidates(0)
    with pytest.raises(ValueError):
        conjugate_kx_order(0)


@pytest.mark.parametrize("nx", (1, 2, 3, 4, 7, 8))
def test_conjugate_kx_order_negates_the_radial_mode_number(nx: int) -> None:
    """``order`` sends each ``kx`` mode to ``-kx`` modulo the box, and is its own
    inverse.  An even grid's Nyquist column is its own image, which is why the
    permutation is written as an index map rather than a sign flip."""

    order = conjugate_kx_order(nx)
    modes = np.rint(np.fft.fftfreq(nx, d=1.0 / nx)).astype(int)
    np.testing.assert_array_equal(modes[order] % nx, (-modes) % nx)
    np.testing.assert_array_equal(order[order], np.arange(nx))


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
@pytest.mark.parametrize("nx", (1, 4, 5))
def test_half_full_round_trip_is_exact_for_a_real_field(ny: int, nx: int) -> None:
    """``to_full(to_half(F)) == F`` bitwise when ``F`` is a real field's spectrum."""

    full = _batched(_real_field(ny, nx, 3, seed=ny * 100 + nx))
    rebuilt = to_full(to_half(full), ny_full=ny)
    assert rebuilt.shape == full.shape
    np.testing.assert_array_equal(rebuilt, full)


@pytest.mark.parametrize("ny", ALL_NY)
def test_to_half_is_idempotent_at_a_boundary_handed_either_layout(ny: int) -> None:
    full = _batched(_real_field(ny, 4, 3, seed=ny))
    half = to_half(full)
    assert half.shape[-3] == nyc_from_ny(ny)
    np.testing.assert_array_equal(to_half(half, ny_full=ny), half)


def test_to_half_refuses_a_row_count_that_is_neither_layout() -> None:
    state = np.zeros((1, 1, 1, 6, 4, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="expected the full axis"):
        to_half(state, ny_full=16)


def test_to_full_refuses_a_half_block_that_does_not_match_ny_full() -> None:
    half = np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="does not match ny_full"):
        to_full(half, ny_full=16)


@pytest.mark.parametrize("ny", (8, 9))
def test_widening_carries_ny_full_rather_than_guessing_it(ny: int) -> None:
    """The same ``Nyc = 5`` block widens to 8 or 9 rows, on request only."""

    half = np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128)
    assert to_full(half, ny_full=ny).shape[-3] == ny


# --------------------------------------------------------------------------
# The reality condition and the self-conjugate rows
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_widened_arrays_satisfy_the_reality_condition_row_by_row(ny: int) -> None:
    nx = 6
    half = to_half(_batched(_real_field(ny, nx, 3, seed=ny + 7)))
    full = to_full(half, ny_full=ny)
    order = conjugate_kx_order(nx)
    for row in range(1, paired_row_limit(ny)):
        partner = (-row) % ny
        np.testing.assert_array_equal(
            full[..., partner, :, :], np.conj(full[..., row, order, :])
        )


@pytest.mark.parametrize("ny", ALL_NY)
def test_reality_residual_is_zero_for_a_real_field_and_positive_otherwise(
    ny: int,
) -> None:
    nx = 6
    half = symmetrize_self_conjugate_rows(
        _batched(_half_of_real_field(ny, nx, 3, seed=ny + 11)), ny_full=ny
    )
    full = to_full(half, ny_full=ny)
    assert float(reality_residual(full)) == 0.0
    broken = full.copy()
    broken[..., 0, 1, :] += 1.0  # breaks the ky=0 internal constraint
    assert float(reality_residual(broken)) > 0.0
    # A field's own rfft2 block only satisfies the ky=0 constraint to roundoff.
    raw = to_full(_batched(_half_of_real_field(ny, nx, 3, seed=ny + 11)), ny_full=ny)
    assert float(reality_residual(raw)) < 1e-14


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_half_layout_alone_does_not_enforce_the_self_conjugate_rows(
    ny: int,
) -> None:
    """Storing ``ky >= 0`` leaves ``F[0, kx] = conj(F[0, -kx])`` to be imposed."""

    nx = 6
    half = to_half(_batched(_real_field(ny, nx, 3, seed=ny + 13)))
    polluted = np.array(half)
    polluted[..., 0, 1, :] += 0.5 + 0.25j
    order = conjugate_kx_order(nx)
    assert not np.allclose(polluted[..., 0, :, :], np.conj(polluted[..., 0, order, :]))
    fixed = symmetrize_self_conjugate_rows(polluted, ny_full=ny)
    for row in self_conjugate_rows(ny):
        np.testing.assert_allclose(
            fixed[..., row, :, :], np.conj(fixed[..., row, order, :]), atol=0, rtol=0
        )


@pytest.mark.parametrize("ny", ALL_NY)
def test_symmetrization_is_idempotent_and_leaves_a_real_field_alone(ny: int) -> None:
    half = to_half(_batched(_real_field(ny, 6, 3, seed=ny + 17)))
    once = symmetrize_self_conjugate_rows(half, ny_full=ny)
    np.testing.assert_allclose(once, half, rtol=0, atol=1e-12)
    twice = symmetrize_self_conjugate_rows(once, ny_full=ny)
    np.testing.assert_array_equal(twice, once)


def test_symmetrization_refuses_a_row_count_that_is_neither_layout() -> None:
    with pytest.raises(ValueError, match="expected 16 or 9"):
        symmetrize_self_conjugate_rows(
            np.zeros((1, 1, 1, 6, 4, 2), dtype=np.complex128), ny_full=16
        )


def test_reality_residual_refuses_a_half_spectrum_input() -> None:
    with pytest.raises(ValueError, match="needs the full ky axis"):
        reality_residual(np.zeros((1, 1, 1, 5, 4, 2), dtype=np.complex128), ny_full=8)


# --------------------------------------------------------------------------
# Reductions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_row_weights_reproduce_the_full_axis_sum_of_a_symmetric_quantity(
    ny: int,
) -> None:
    """``sum_full |F|^2 == sum(w * |F_half|^2)``: the flux/spectrum identity."""

    full = _real_field(ny, 6, 3, seed=ny + 19)
    power = np.abs(full) ** 2
    weights = ky_row_weights(ny)
    assert weights.shape == (nyc_from_ny(ny),)
    np.testing.assert_allclose(
        float(np.sum(power)),
        float(np.sum(weights[:, None, None] * power[: nyc_from_ny(ny)])),
        rtol=1e-12,
    )


@pytest.mark.parametrize("ny", EVEN_NY)
def test_a_blanket_factor_of_two_over_counts_the_nyquist_row(ny: int) -> None:
    """The reason the weights exist rather than a scalar 2 on ``ky > 0``."""

    weights = ky_row_weights(ny)
    blanket = np.where(np.arange(nyc_from_ny(ny)) == 0, 1.0, 2.0)
    row = nyquist_row(ny)
    assert row is not None
    assert weights[row] == 1.0
    if row != 0:
        assert blanket[row] == 2.0
        assert not np.array_equal(weights, blanket)


@pytest.mark.parametrize("ny", ODD_NY)
def test_odd_grids_weight_every_row_above_zero_by_two(ny: int) -> None:
    weights = ky_row_weights(ny)
    np.testing.assert_array_equal(weights[1:], np.full(weights.size - 1, 2.0))


# --------------------------------------------------------------------------
# Grid views
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", ALL_NY)
def test_half_ky_values_match_rfftfreq_up_to_the_box_scale(ny: int) -> None:
    ly = 4.0
    ky_full = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    expected = 2.0 * np.pi * np.fft.rfftfreq(ny, d=ly / ny)
    np.testing.assert_allclose(half_ky_values(ky_full), expected, rtol=1e-12)


@pytest.mark.parametrize("ny", ALL_NY)
def test_half_dealias_mask_is_the_top_block_and_is_idempotent(ny: int) -> None:
    mask = np.arange(ny * 4, dtype=float).reshape(ny, 4)
    half = half_dealias_mask(mask, ny_full=ny)
    np.testing.assert_array_equal(half, mask[: nyc_from_ny(ny)])
    np.testing.assert_array_equal(half_dealias_mask(half, ny_full=ny), half)


def test_half_dealias_mask_refuses_a_foreign_row_count() -> None:
    with pytest.raises(ValueError, match="expected 16 or 9"):
        half_dealias_mask(np.zeros((6, 4)), ny_full=16)


def test_describe_reports_the_contract_numbers_for_one_grid() -> None:
    assert describe(8) == {
        "ny_full": 8,
        "nyc": 5,
        "nyquist_row": 4,
        "self_conjugate_rows": (0, 4),
        "paired_rows": (1, 4),
        "ny_full_candidates_for_nyc": (8, 9),
        "row_weights": (1.0, 2.0, 2.0, 2.0, 1.0),
    }


# --------------------------------------------------------------------------
# Backend dispatch
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 9))
def test_numpy_input_keeps_numpy_output_so_host_paths_stay_on_the_host(
    ny: int,
) -> None:
    half = to_half(_batched(_real_field(ny, 4, 2, seed=ny)))
    assert isinstance(to_full(half, ny_full=ny), np.ndarray)
    assert isinstance(symmetrize_self_conjugate_rows(half, ny_full=ny), np.ndarray)
    assert isinstance(negative_ky_block(half, ny_full=ny), np.ndarray)


@pytest.mark.parametrize("ny", (8, 9))
def test_jax_and_numpy_widening_agree_and_widening_traces(ny: int) -> None:
    half_np = to_half(_batched(_real_field(ny, 4, 2, seed=ny + 3)))
    half_jax = jnp.asarray(half_np)
    from_numpy = to_full(half_np, ny_full=ny)
    from_jax = to_full(half_jax, ny_full=ny)
    assert isinstance(from_jax, jax.Array)
    np.testing.assert_allclose(np.asarray(from_jax), from_numpy, rtol=0, atol=1e-12)

    traced = jax.jit(lambda x: to_full(x, ny_full=ny))(half_jax)
    np.testing.assert_array_equal(np.asarray(traced), np.asarray(from_jax))


def test_widening_is_differentiable_and_its_vjp_sums_the_conjugate_pair() -> None:
    """The completion is linear over the reals; its adjoint folds both rows back."""

    ny, nx = 8, 4

    def widen(real_imag: jnp.ndarray) -> jnp.ndarray:
        half = real_imag[0] + 1j * real_imag[1]
        return to_full(half, ny_full=ny)

    key = jax.random.PRNGKey(0)
    parts = jax.random.normal(key, (2, 1, 1, 1, nyc_from_ny(ny), nx, 2))
    value, pullback = jax.vjp(widen, parts)
    cotangent = jnp.ones_like(value)
    (grad,) = pullback(cotangent)
    assert grad.shape == parts.shape
    # ky=0 and Nyquist appear once; every paired row appears twice.
    real_grad = np.asarray(grad[0])[0, 0, 0, :, 0, 0]
    np.testing.assert_allclose(real_grad, ky_row_weights(ny), rtol=1e-6)


# --------------------------------------------------------------------------
# The consumers that already sit on the boundary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ny", (8, 9, 16))
def test_the_bracket_half_core_plus_the_widening_is_the_bracket(ny: int) -> None:
    """The ``ky >= 0`` bracket primitive is the whole kernel; the rest is layout.

    Plan 5.3 N3 has the evolved state call the half core directly.  This pins
    the seam now: the two-sided kernel is exactly the half core followed by the
    contract's widening and the real ``kxfac``, bit for bit.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid
    from gkx.operators.nonlinear.brackets import (
        _complete_hermitian_ky,
        _spectral_bracket_half_core,
        _spectral_bracket_real_fft_core,
    )

    # The seam under test is the widening that turns the half core's output
    # into the two-sided kernel's, so this grid has to carry both halves of the
    # ky axis rather than whichever one the default hands out.
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=ny, Nz=3, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    nx, nz = int(grid.kx.size), int(grid.z.size)
    G = jnp.asarray(_batched(_real_field(ny, nx, nz, seed=ny + 23))[0])
    chi = jnp.asarray(_real_field(ny, nx, nz, seed=ny + 29))
    kwargs = dict(
        kx_grid=grid.kx_grid,
        ky_grid=grid.ky_grid,
        dealias_mask=grid.dealias_mask,
    )
    kxfac = jnp.asarray(1.3)

    half, reported_ny, reported_nx = _spectral_bracket_half_core(
        G, chi, multiple_fields=False, **kwargs
    )
    assert (reported_ny, reported_nx) == (ny, nx)
    assert int(half.shape[-3]) == nyc_from_ny(ny)

    full = _spectral_bracket_real_fft_core(
        G, chi, kxfac=kxfac, multiple_fields=False, **kwargs
    )
    expected = kxfac * _complete_hermitian_ky(half, ny, nx)
    np.testing.assert_array_equal(np.asarray(full), np.asarray(expected))

    # The bracket's own output obeys the reality condition it was widened with.
    assert float(reality_residual(np.asarray(full))) < 1e-6


@pytest.mark.parametrize("ny", (8, 9))
def test_the_nonlinear_projector_is_the_contract_round_trip(ny: int) -> None:
    from gkx.operators.nonlinear.projection import _make_compressed_real_fft_projector

    nx, nz = 6, 3
    project = _make_compressed_real_fft_projector(ny_full=ny, nx=nx)
    noise = jnp.asarray(
        _batched(
            np.random.default_rng(ny).normal(size=(ny, nx, nz))
            + 1j * np.random.default_rng(ny + 1).normal(size=(ny, nx, nz))
        )
    )
    projected = project(noise)
    np.testing.assert_array_equal(
        np.asarray(projected), np.asarray(to_full(to_half(noise), ny_full=ny))
    )
    # A projector is idempotent, which is what makes it a projector.
    np.testing.assert_array_equal(np.asarray(project(projected)), np.asarray(projected))


@pytest.mark.parametrize("ny", (8, 9, 3, 2))
def test_the_restart_read_path_widens_with_the_grids_own_ny(ny: int) -> None:
    """``_expand_ky`` used to infer ``Ny = 2*(Nyc-1)``, losing a row on odd grids."""

    from gkx.workflows.runtime.startup import _expand_ky

    nx, nz = 4, 2
    half = _batched(_half_of_real_field(ny, nx, nz, seed=ny + 31))
    widened = _expand_ky(half, ny_full=ny)
    assert widened.shape[-3] == ny
    np.testing.assert_array_equal(widened, to_full(half, ny_full=ny))
    # Already-full input passes through untouched.
    np.testing.assert_array_equal(_expand_ky(widened, ny_full=ny), widened)


@pytest.mark.parametrize("ny", (8, 9))
def test_the_artifact_restart_expansion_round_trips_through_the_contract(
    ny: int,
) -> None:
    from gkx.artifacts.io import _expand_positive_ky_to_full

    nx, nz = 4, 2
    half = _batched(_half_of_real_field(ny, nx, nz, seed=ny + 37))
    full = _expand_positive_ky_to_full(half, ny_full=ny)
    assert isinstance(full, np.ndarray)
    np.testing.assert_array_equal(full, to_full(half, ny_full=ny))
    np.testing.assert_array_equal(to_half(full, ny_full=ny), half)
    with pytest.raises(ValueError, match="does not match ny_full"):
        _expand_positive_ky_to_full(half, ny_full=ny + 4)


@pytest.mark.parametrize("ny", (8, 9))
def test_the_in_place_hermitian_repair_matches_the_contracts_negative_block(
    ny: int,
) -> None:
    from gkx.workflows.runtime.startup import _enforce_full_ky_hermitian

    nx, nz = 4, 2
    rng = np.random.default_rng(ny + 41)
    noisy = (
        rng.normal(size=(1, 1, 1, ny, nx, nz))
        + 1j * rng.normal(size=(1, 1, 1, ny, nx, nz))
    ).astype(np.complex64)
    repaired = _enforce_full_ky_hermitian(noisy.copy())
    nyc = nyc_from_ny(ny)
    np.testing.assert_array_equal(repaired[..., :nyc, :, :], noisy[..., :nyc, :, :])
    np.testing.assert_array_equal(
        repaired[..., nyc:, :, :],
        negative_ky_block(noisy[..., :nyc, :, :], ny_full=ny),
    )


def test_the_grid_reports_the_half_axis_through_the_contract() -> None:
    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid, real_fft_unique_ky

    # ``real_fft_unique_ky`` is the reduction from the two-sided axis to the
    # half one, so it has to be handed a grid that stores the negative rows.
    grid = build_spectral_grid(
        GridConfig(Nx=6, Ny=10, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    np.testing.assert_allclose(
        np.asarray(real_fft_unique_ky(grid.ky)),
        np.asarray(half_ky_values(grid.ky)),
        rtol=0,
        atol=0,
    )
    assert int(real_fft_unique_ky(grid.ky).size) == nyc_from_ny(10)


def test_prepared_simulation_reports_the_state_it_actually_allocates() -> None:
    """``state_shape`` advertised ``Nyc`` while the runtime allocated ``Ny``."""

    from gkx.api.prepared import prepare_simulation
    from gkx.config import RuntimeConfig
    from gkx.core_grid import build_spectral_grid

    case = RuntimeConfig()
    prepared = prepare_simulation(case, Nl=2, Nm=4)
    grid = build_spectral_grid(case.grid)
    rows = rows_for_layout(case.grid.Ny, case.grid.ky_layout)
    assert prepared.state_shape[-3] == int(grid.ky.size) == rows
    assert prepared.estimate_memory()["elements"] == int(np.prod(prepared.state_shape))


def _grids(ny: int, nx: int = 4):
    """Return the two-sided grid for ``ny`` and its ``ky >= 0`` view.

    Its callers weigh one layout against the other, and the half view is cut
    from the two-sided one, so the parent grid names the two-sided axis instead
    of taking whichever layout the configuration defaults to.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid, select_real_fft_ky_grid

    full_grid = build_spectral_grid(
        GridConfig(Nx=nx, Ny=ny, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi, ky_layout="full")
    )
    return full_grid, select_real_fft_ky_grid(full_grid, half_ky_values(full_grid.ky))


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_moment_weight_is_the_contract_weight_on_a_half_axis(ny: int) -> None:
    """``_hermitian_mode_weight`` and :func:`ky_row_weights` are one rule.

    Queue row Q24.  The rule used to be written out three times and the copies
    gave an even grid's Nyquist row weight 2, which is the paired weight on a
    row that has no partner.  Nothing shipped reached it -- the only
    half-spectrum grids GKX builds are the GX-comparison views, and the
    two-thirds mask zeroes every row at or above ``Ny/3``, Nyquist included --
    but plan 5.3 N3 moves the evolved state onto this axis, where the
    undealiased weight has to be right.
    """

    from gkx.operators.moments import _hermitian_mode_weight

    _full_grid, half_grid = _grids(ny)
    bare = np.asarray(_hermitian_mode_weight(half_grid, use_dealias=False))[:, 0]
    np.testing.assert_array_equal(bare, ky_row_weights(ny))


@pytest.mark.parametrize("ny", EVEN_NY)
def test_the_half_axis_weight_sums_a_non_zero_nyquist_row_correctly(ny: int) -> None:
    """The case the old rule got wrong, with the Nyquist row carrying power.

    This is the whole point of the fix: with the row zeroed, weight 1 and
    weight 2 agree, so only an undealiased field with power at ``ky = Ny/2``
    tells them apart.  The old blanket-two rule over-counts that row's
    ``|F|^2`` by exactly itself and fails here.
    """

    from gkx.operators.moments import _hermitian_mode_weight

    row = nyquist_row(ny)
    assert row is not None and row != 0

    full = _real_field(ny, 4, 3, seed=ny + 71)
    assert float(np.max(np.abs(full[row]))) > 0.0
    _full_grid, half_grid = _grids(ny, nx=4)

    weight = np.asarray(_hermitian_mode_weight(half_grid, use_dealias=False))
    half = np.abs(full[: nyc_from_ny(ny)]) ** 2
    got = float(np.sum(weight[:, :, None] * half))
    np.testing.assert_allclose(got, float(np.sum(np.abs(full) ** 2)), rtol=1e-12)

    blanket = np.where(np.arange(nyc_from_ny(ny))[:, None] == 0, 1.0, 2.0)
    over = float(np.sum(blanket[:, :, None] * half))
    assert over > got
    np.testing.assert_allclose(
        over - got, float(np.sum(np.abs(full[row]) ** 2)), rtol=1e-12
    )


@pytest.mark.parametrize("ny", (4, 8, 16, 32))
def test_the_default_grid_weights_its_nyquist_row_once(ny: int) -> None:
    """The Nyquist rule, on the grid a deck builds when it names no layout.

    The tests above cut their half axis from a two-sided parent with
    ``select_real_fft_ky_grid``, which is how the GX-comparison views are made.
    A run does not build its grid that way: ``build_spectral_grid`` reads
    ``GridConfig.ky_layout``, and since the default flip that is the half axis.
    Whether the Nyquist row is found on it depends on ``ny_full`` reaching the
    grid from the config, which is a different path from the one above and the
    one the default takes. So this pins the rule where the default runs it:
    weight 1 on the Nyquist row for the Hermitian reductions, 0.5 for the flux
    representative, and a non-zero Nyquist row summed exactly once.
    """

    from gkx.config import GridConfig
    from gkx.core_grid import build_spectral_grid
    from gkx.operators.moments import _hermitian_mode_weight, _transport_mode_weight

    grid = build_spectral_grid(
        GridConfig(Nx=4, Ny=ny, Nz=2, Lx=2.0 * np.pi, Ly=2.0 * np.pi)
    )
    assert source_ky_layout(grid) == HALF
    assert source_ny_full(grid) == ny
    row = nyquist_row(ny)
    assert row == nyc_from_ny(ny) - 1

    hermitian = np.asarray(_hermitian_mode_weight(grid, use_dealias=False))
    transport = np.asarray(_transport_mode_weight(grid, use_dealias=False))
    np.testing.assert_array_equal(hermitian[:, 0], ky_row_weights(ny))
    np.testing.assert_array_equal(hermitian[row], np.ones(4))
    np.testing.assert_array_equal(transport[row], np.full(4, 0.5))
    np.testing.assert_array_equal(transport[0], np.zeros(4))

    full = _real_field(ny, 4, 3, seed=ny + 97)
    assert float(np.max(np.abs(full[row]))) > 0.0
    half = np.abs(full[: nyc_from_ny(ny)]) ** 2
    np.testing.assert_allclose(
        float(np.sum(hermitian[:, :, None] * half)),
        float(np.sum(np.abs(full) ** 2)),
        rtol=1e-12,
    )


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_two_sided_moment_weight_counts_every_stored_row_once(ny: int) -> None:
    """The layout the code evolves today is untouched by the Q24 rule."""

    from gkx.operators.moments import _hermitian_mode_weight

    full_grid, _half_grid = _grids(ny)
    two_sided = np.asarray(_hermitian_mode_weight(full_grid, use_dealias=False))[:, 0]
    np.testing.assert_array_equal(two_sided, np.ones(ny))


@pytest.mark.parametrize("ny", EVEN_NY)
def test_dealiasing_still_zeroes_the_nyquist_row_in_both_weights(ny: int) -> None:
    """Why no shipped number moves: the row the rule changed is masked away."""

    from gkx.operators.moments import _hermitian_mode_weight, _transport_mode_weight

    row = nyquist_row(ny)
    assert row is not None
    _full_grid, half_grid = _grids(ny)
    for weight in (_hermitian_mode_weight, _transport_mode_weight):
        masked = np.asarray(weight(half_grid, use_dealias=True))
        np.testing.assert_array_equal(masked[row], np.zeros(masked.shape[1]))


@pytest.mark.parametrize("ny", ALL_NY)
def test_the_flux_weight_folds_the_pair_and_halves_a_self_conjugate_row(
    ny: int,
) -> None:
    """``_transport_mode_weight`` picks one representative per conjugate pair.

    The flux kernels carry the factor of two themselves, so a row that stands
    for itself and its unstored partner takes weight 1 and a self-conjugate row
    takes 0.5.

    The two-sided convention now says the same thing (plan 5.3 N3, the decision
    Q24 handed to the state switch): an even grid's Nyquist row is a
    representative there too, at the same 0.5, so the two-sided weights are the
    half-axis weights padded with zeros on the rows the half axis does not
    store.  Before the switch that row was dropped from the flux on a two-sided
    axis and counted on a half one, which made the flux depend on the layout.
    """

    from gkx.operators.moments import _transport_mode_weight

    full_grid, half_grid = _grids(ny)
    half = np.asarray(_transport_mode_weight(half_grid, use_dealias=False))[:, 0]
    expected = ky_row_weights(ny) / 2.0
    expected[0] = 0.0
    np.testing.assert_array_equal(half, expected)

    two_sided = np.asarray(_transport_mode_weight(full_grid, use_dealias=False))[:, 0]
    padded = np.zeros(ny, dtype=two_sided.dtype)
    padded[: expected.size] = expected
    np.testing.assert_array_equal(two_sided, padded)


def test_a_selected_subset_of_modes_does_not_claim_a_nyquist_row() -> None:
    """An index into a mode selection would name the wrong row, so it is not used."""

    from gkx.core_grid import select_ky_grid, select_real_fft_ky_grid
    from gkx.operators.moments import _hermitian_mode_weight

    full_grid, half_grid = _grids(8)
    assert full_grid.ny_full == 8 and half_grid.ny_full == 8

    # Fewer rows than the half block: not a complete axis, so no parent length.
    subset = select_ky_grid(full_grid, [0, 1, 2, 3])
    assert subset.ny_full is None
    np.testing.assert_array_equal(
        np.asarray(_hermitian_mode_weight(subset, use_dealias=False))[:, 0],
        np.array([1.0, 2.0, 2.0, 2.0]),
    )

    # As many rows as the half block, but another code's wave numbers rather
    # than this grid's: the value check is what keeps row 4 from being called
    # a Nyquist row it is not.
    dump = select_real_fft_ky_grid(full_grid, np.array([0.0, 1.0, 2.0, 3.0, 5.0]))
    assert dump.ny_full is None
    np.testing.assert_array_equal(
        np.asarray(_hermitian_mode_weight(dump, use_dealias=False))[:, 0],
        np.array([1.0, 2.0, 2.0, 2.0, 2.0]),
    )


def test_the_cached_weight_and_the_quasilinear_weight_are_the_same_rule() -> None:
    """The third and fourth copies of the rule now read from one owner."""

    from types import SimpleNamespace

    from gkx.diagnostics.quasilinear_transport import spectral_phi_weights
    from gkx.operators.moments import _cached_hermitian_mode_weight

    ny, nx = 8, 2
    _full_grid, half_grid = _grids(ny, nx=nx)
    cache = SimpleNamespace(
        ky=half_grid.ky,
        kx=half_grid.kx,
        dealias_mask=half_grid.dealias_mask,
        ny_full=half_grid.ny_full,
    )
    cached = np.asarray(_cached_hermitian_mode_weight(cache, use_dealias=False))
    np.testing.assert_array_equal(cached[:, 0], ky_row_weights(ny))

    nz = 3
    phi = jnp.ones((cached.shape[0], nx, nz), dtype=jnp.complex64)
    vol_fac = jnp.ones((nz,), dtype=jnp.float32) / nz
    weights = np.asarray(spectral_phi_weights(phi, cache, vol_fac, use_dealias=False))
    # ``vol_fac`` sums to one, so the z sum of the quasilinear weight of a
    # unit field is exactly the cached ``(ky, kx)`` weight.
    np.testing.assert_allclose(weights.sum(axis=2), cached, rtol=1e-6, atol=1e-7)
