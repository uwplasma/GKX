"""The ``ky`` axis layout contract for spectral arrays.

GKX represents a real perpendicular field by its Fourier coefficients on a
``(..., ky, kx, z)`` array whose ``ky`` axis is :data:`KY_AXIS`.  Two layouts of
that axis exist in the code base, and every conversion between them goes
through this module.

:data:`FULL` -- ``Nky == Ny``
    The two-sided FFT order ``[0, 1, ..., Ny//2 - 1, -Ny//2, ..., -1]`` that
    ``jnp.fft.fftfreq`` produces.  This is the layout the evolved state uses
    today.  Half of it is redundant: a real field satisfies the reality
    condition ``F[-ky, -kx, z] = conj(F[ky, kx, z])``.

:data:`HALF` -- ``Nky == Nyc == 1 + Ny // 2``
    The non-negative rows ``ky_j = j * 2*pi/Ly`` for ``j = 0 .. Ny // 2``, in
    ``jnp.fft.rfftfreq`` order.  The reality condition holds by construction
    because the negative rows are not stored.  GKX's restart files, NetCDF
    output and real-space snapshots already use this layout, as do the
    flux-tube codes the plan compares against (docs ``numerics``, plan 2.4).

Three facts about the contract are easy to get wrong and are therefore stated
here rather than rederived at each call site.

**Nyc does not determine Ny.**  ``1 + Ny // 2`` maps both ``Ny = 2*(Nyc - 1)``
and ``Ny = 2*Nyc - 1`` to the same ``Nyc``, so a half-spectrum array cannot say
how long its own full axis is.  Every function here that widens an axis takes
``ny_full`` explicitly, and :func:`ny_full_candidates` shows a caller that
believes otherwise both answers.

**The self-conjugate rows are not made real by the layout.**  Row ``j = 0``
always, and row ``j = Ny//2`` when ``Ny`` is even, are their own conjugate
partners.  On them the reality condition collapses to a constraint *within* the
row, ``F[j, kx] = conj(F[j, -kx])``, which storing only ``ky >= 0`` does not
enforce.  A real inverse transform imposes it silently by discarding the
anti-symmetric part; an operator that writes those rows directly must call
:func:`symmetrize_self_conjugate_rows` instead.

**Reductions need row weights, not a factor of two.**  A quantity that obeys
``Q[-ky] = Q[ky]`` -- ``abs(F)**2``, ``Re(F * conj(G))``, every flux -- sums
over the full axis to the :func:`ky_row_weights`-weighted sum over the half
axis.  The weight is 2 on the paired rows and 1 on the self-conjugate rows, so
a blanket factor of two over-counts the Nyquist row of an even-``Ny`` grid.
"""

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp
import numpy as np

__all__ = [
    "FULL",
    "HALF",
    "KY_AXIS",
    "KyLayout",
    "conjugate_kx_order",
    "describe",
    "half_dealias_mask",
    "half_ky_values",
    "ky_row_weights",
    "negative_ky_block",
    "ny_full_candidates",
    "nyc_from_ny",
    "nyquist_row",
    "reality_residual",
    "self_conjugate_rows",
    "symmetrize_self_conjugate_rows",
    "to_full",
    "to_half",
]

#: Axis of the ``ky`` rows in a ``(..., ky, kx, z)`` spectral array.
KY_AXIS: int = -3

KyLayout = Literal["full", "half"]

#: Two-sided ``fftfreq`` order, length ``Ny``.
FULL: KyLayout = "full"

#: Non-negative ``rfftfreq`` order, length ``Nyc = 1 + Ny // 2``.
HALF: KyLayout = "half"


def _xp(array: Any) -> Any:
    """Return the array module that owns ``array``.

    ``numpy`` inputs keep numpy outputs, so the host-side restart and artifact
    paths do not silently acquire device arrays; everything else, tracers
    included, goes through ``jax.numpy``.
    """

    if isinstance(array, (np.ndarray, np.generic)):
        return np
    return jnp


def nyc_from_ny(ny_full: int) -> int:
    """Return the half-spectrum row count ``Nyc = 1 + Ny // 2``."""

    ny = int(ny_full)
    if ny < 1:
        raise ValueError(f"ny_full must be >= 1, got {ny}")
    return 1 + ny // 2


def ny_full_candidates(nyc: int) -> tuple[int, int]:
    """Return the two full lengths that share this ``Nyc``, as ``(even, odd)``."""

    rows = int(nyc)
    if rows < 1:
        raise ValueError(f"nyc must be >= 1, got {rows}")
    return (2 * (rows - 1), 2 * rows - 1)


def nyquist_row(ny_full: int) -> int | None:
    """Return the Nyquist row index, or ``None`` when the grid has none.

    Only an even ``Ny >= 2`` has one, at ``Nyc - 1``.  An odd grid's highest
    stored row still has a distinct negative partner.
    """

    ny = int(ny_full)
    if ny < 2 or ny % 2 != 0:
        return None
    return nyc_from_ny(ny) - 1


def self_conjugate_rows(ny_full: int) -> tuple[int, ...]:
    """Return the half-spectrum rows that are their own conjugate partner."""

    nyquist = nyquist_row(ny_full)
    if nyquist is None or nyquist == 0:
        return (0,)
    return (0, nyquist)


def paired_row_limit(ny_full: int) -> int:
    """Return the exclusive end of the rows that own a distinct partner."""

    nyc = nyc_from_ny(ny_full)
    return nyc - 1 if int(ny_full) % 2 == 0 else nyc


def conjugate_kx_order(nx: int) -> np.ndarray:
    """Return the ``kx -> -kx`` permutation ``[0, nx-1, ..., 1]``.

    Kept on the host as a numpy array.  Projectors built from it are cached and
    reused across traces, and a device constant materialized here would belong
    to whichever trace happened to build it first.  The ordering is an exact
    index reversal, so numpy holds it without loss and each trace materializes
    its own constant at use.
    """

    width = int(nx)
    if width < 1:
        raise ValueError(f"nx must be >= 1, got {width}")
    if width == 1:
        return np.zeros((1,), dtype=np.int32)
    return np.concatenate(([0], np.arange(width - 1, 0, -1))).astype(np.int32)


def to_half(state: Any, *, ny_full: int | None = None) -> Any:
    """Return the ``ky >= 0`` rows of a spectral array.

    An array already in :data:`HALF` is returned unchanged, so the call is
    idempotent at a boundary that may be handed either layout.
    """

    rows = int(state.shape[KY_AXIS])
    ny = rows if ny_full is None else int(ny_full)
    nyc = nyc_from_ny(ny)
    if rows == nyc:
        return state
    if rows != ny:
        raise ValueError(
            f"spectral array has {rows} ky rows; expected the full axis ({ny}) "
            f"or its half-spectrum block ({nyc})"
        )
    return state[..., :nyc, :, :]


def negative_ky_block(
    positive_ky: Any,
    *,
    ny_full: int,
    nx: int | None = None,
    conjugate_kx: Any | None = None,
) -> Any:
    """Return the negative-``ky`` rows implied by ``positive_ky``.

    This is the reality condition written out: ``conj`` of the paired rows,
    reversed in ``ky`` and permuted in ``kx``.  The Nyquist row of an even grid
    is excluded because it has no distinct partner.
    """

    xp = _xp(positive_ky)
    negative = xp.conj(positive_ky[..., 1 : paired_row_limit(ny_full), :, :])
    negative = negative[..., ::-1, :, :]
    width = int(positive_ky.shape[-2]) if nx is None else int(nx)
    if width > 1:
        order = conjugate_kx_order(width) if conjugate_kx is None else conjugate_kx
        negative = negative[..., order, :]
    return negative


def to_full(
    positive_ky: Any,
    *,
    ny_full: int,
    nx: int | None = None,
    conjugate_kx: Any | None = None,
) -> Any:
    """Widen a ``ky >= 0`` array to the two-sided axis by the reality condition.

    ``ny_full`` is required: see the module docstring on why ``Nyc`` alone
    cannot supply it.  ``conjugate_kx`` overrides :func:`conjugate_kx_order` for
    callers that cache the permutation.
    """

    ny = int(ny_full)
    if ny <= 1:
        return positive_ky
    nyc = int(positive_ky.shape[KY_AXIS])
    if nyc != nyc_from_ny(ny):
        raise ValueError(
            f"half-spectrum array has {nyc} ky rows, which does not match "
            f"ny_full={ny} (expected {nyc_from_ny(ny)})"
        )
    negative = negative_ky_block(
        positive_ky, ny_full=ny, nx=nx, conjugate_kx=conjugate_kx
    )
    return _xp(positive_ky).concatenate([positive_ky, negative], axis=KY_AXIS)


def _set_ky_row(state: Any, row: int, block: Any) -> Any:
    if _xp(state) is np:
        out = np.array(state, copy=True)
        out[..., row : row + 1, :, :] = block
        return out
    return state.at[..., row : row + 1, :, :].set(block)


def symmetrize_self_conjugate_rows(state: Any, *, ny_full: int) -> Any:
    """Impose ``F[j, kx] = conj(F[j, -kx])`` on the self-conjugate rows.

    Applies to either layout.  Averaging the pair rather than keeping one side
    keeps the operation self-adjoint and idempotent.  It is also the published
    fix for a packed-transform gyroaverage, which without it left a ``ky = 0``
    asymmetry of up to 14 per cent (plan 5.3 N4 and its cited source).
    """

    xp = _xp(state)
    ny = int(ny_full)
    rows = int(state.shape[KY_AXIS])
    if rows not in (ny, nyc_from_ny(ny)):
        raise ValueError(
            f"spectral array has {rows} ky rows; expected {ny} or {nyc_from_ny(ny)}"
        )
    width = int(state.shape[-2])
    if width <= 1:
        return state
    order = conjugate_kx_order(width)
    out = state
    for row in self_conjugate_rows(ny):
        block = out[..., row : row + 1, :, :]
        mirrored = xp.conj(block[..., order, :])
        out = _set_ky_row(out, row, 0.5 * (block + mirrored))
    return out


def reality_residual(state: Any, *, ny_full: int | None = None) -> Any:
    """Return ``max|F - H(F)| / max|F|`` for a two-sided array.

    ``H`` rebuilds the array from its own ``ky >= 0`` rows, so the residual is
    zero exactly when the stored negative rows agree with the reality
    condition.  The self-conjugate rows are included: their internal constraint
    is part of ``H``.
    """

    xp = _xp(state)
    rows = int(state.shape[KY_AXIS])
    ny = rows if ny_full is None else int(ny_full)
    if rows != ny:
        raise ValueError(
            f"reality_residual needs the full ky axis; got {rows} rows for ny_full={ny}"
        )
    rebuilt = to_full(
        symmetrize_self_conjugate_rows(to_half(state, ny_full=ny), ny_full=ny),
        ny_full=ny,
    )
    scale = xp.max(xp.abs(state))
    denominator = xp.where(scale > 0, scale, xp.ones_like(scale))
    return xp.max(xp.abs(state - rebuilt)) / denominator


def ky_row_weights(ny_full: int, *, dtype: Any = float) -> np.ndarray:
    """Return the half-spectrum reduction weights, length ``Nyc``.

    For any ``Q`` with ``Q[-ky] = Q[ky]``, ``sum_full(Q)`` equals
    ``sum(ky_row_weights(Ny) * Q_half)``.  The weight is 1 on the
    self-conjugate rows and 2 elsewhere.
    """

    weights = np.full((nyc_from_ny(ny_full),), 2.0, dtype=dtype)
    for row in self_conjugate_rows(ny_full):
        weights[row] = 1.0
    return weights


def half_ky_values(ky_full: Any) -> Any:
    """Return the non-negative ``ky`` magnitudes of a two-sided axis."""

    xp = _xp(ky_full)
    return xp.abs(ky_full[: nyc_from_ny(int(ky_full.shape[0]))])


def half_dealias_mask(mask: Any, *, ny_full: int | None = None) -> Any:
    """Return the ``ky >= 0`` rows of a ``(ky, kx)`` dealias mask."""

    rows = int(mask.shape[0])
    ny = rows if ny_full is None else int(ny_full)
    nyc = nyc_from_ny(ny)
    if rows == nyc:
        return mask
    if rows != ny:
        raise ValueError(f"dealias mask has {rows} ky rows; expected {ny} or {nyc}")
    return mask[:nyc, :]


def describe(ny_full: int) -> dict[str, Any]:
    """Return the contract's concrete numbers for one ``Ny``."""

    ny = int(ny_full)
    return {
        "ny_full": ny,
        "nyc": nyc_from_ny(ny),
        "nyquist_row": nyquist_row(ny),
        "self_conjugate_rows": self_conjugate_rows(ny),
        "paired_rows": (1, paired_row_limit(ny)),
        "ny_full_candidates_for_nyc": ny_full_candidates(nyc_from_ny(ny)),
        "row_weights": tuple(float(w) for w in ky_row_weights(ny)),
    }
