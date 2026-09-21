"""The ``pr3-cm`` structured preconditioner for shift-invert Krylov solves.

Q7 (#236) measured this design on the exact assembled operator and adopted it as
the §5.1 L4 preconditioner; Q21 (#255) measured an exact solve for its z-local
block; Q26 (#257) recorded that neither had ever been written into ``src``.
This module is that code.

The design
----------
``pr3-cm`` is a right preconditioner for ``B = A - sigma I``. It splits the
linear operator into two halves that are each invertible in closed form and
sweeps between them:

``S~``  streaming and hypercollisions, **plus the z-mean of the drift
        diagonal**. This is exactly what :func:`gkx.solvers_linear_implicit.
        _build_shifted_hermite_preconditioner` inverts when it is handed the
        deck's terms with collisions switched off: its Hermite line solve is
        spectral in the parallel coordinate and tridiagonal in the Hermite
        index, and it already replaces the z-dependent drift diagonal by its
        z-mean so that the principal symbol stays separable. Moving that mean
        into this half is what the ``-cm`` suffix names.
``D~``  everything else at one z: drift with its exact ``omega_d(z)``, mirror,
        diamagnetic drive, end damping, collisions and the local field
        response, minus the z-mean drift diagonal that ``S~`` took. This half
        is z-local, so it is a batch of dense ``(l, m)`` blocks, one per
        ``(species, ky, kx, z)``.

With ``s1 = sigma/2 - alpha`` (so ``2 s1 + 2 alpha = sigma``) one
Peaceman-Rachford double sweep of ``(S~ - s1)`` and ``(D~ - s1)`` costs one
solve of each half and no operator application at all; ``pr3-cm`` is three such
sweeps started from zero. Q7 measured 30/45/88/164 GMRES iterations to 1e-5
from the pilot to ``(Nz, Nl, Nm) = (96, 8, 24)``, where the shipped
``hermite-line`` preconditioner no longer converges at all.

The z-local block
-----------------
Q21 established, and :func:`check_block_structure` re-checks at every build,
that the z-local block is **exactly** block-tridiagonal in the Laguerre index
with ``Nm x Nm`` blocks once the field term is removed, and that the field term
itself is **exactly** rank one per ``(kx, z)``. So ``D~ - s1`` is solved exactly
by block-Thomas in the Laguerre index plus one Sherman-Morrison correction,
which is the same preconditioner to the last bit -- Q21 measured the two solves
agreeing to 4.4e-16, with every iteration count unchanged.

The Laguerre couplings are themselves banded in the Hermite index, because the
mirror term is the only thing that moves ``l`` and it moves ``(l, m)`` to
``(l +- 1, m +- 1)``. Measured on this operator the half-width is 1, and the
elimination therefore stores ``Nl Nm^2`` Schur inverses plus ``2 Nl (2p+1) Nm``
coupling diagonals rather than the ``3 Nl Nm^2`` of a dense-band factorization
-- 1.5x smaller at ``Nl Nm = 36`` rising to 2.7x at 768, where the dense
``(Nl Nm)^2`` inverse is 3.0x and 14.2x larger. The half-width is measured, never
assumed; a coupling too wide for the band to pay for itself sends the build to
the dense fallback with the measured width recorded.

:func:`solvax.block_thomas_factor_ops` owns the Schur elimination: that is the
numerically delicate half, and its operator-coupling contract is exactly the
banded form above. Only the substitution is written out here, unrolled over the
Laguerre index; :func:`solvax.block_thomas_solve_ops` runs the same two
recurrences as ``lax.scan``s, which is what keeps *its* solve transposable on
JAX before 0.10, and the tests pin the two against each other bitwise. See
:func:`_block_thomas_substitute` for why GKX does not simply call it.

That exactness is a property of the operator, not a law, so it is measured and
not assumed. :func:`build_pr3_factors` verifies it and falls back to the dense
batched inverse -- the same preconditioner, a costlier apply -- whenever it does
not hold, recording why in the returned metadata. The z-locality of the block
*representation* is checked too, against the operator itself on a random vector;
that one is not a fallback but a refusal, because a z-coupled ``D~`` is not the
operator this preconditioner splits.

Setup and apply
---------------
Building the blocks costs ``Nl * Nm`` probes of the z-local operator and a host
factorization, so it happens once per shift, on the host, in
:func:`build_pr3_factors`. The factors are handed to the jitted solve as an
argument rather than captured as a constant, so the compiled graph does not
carry them as literals. :func:`build_pr3_apply` is the traced side.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from gkx.operators.linear.cache_model import LinearCache
from gkx.operators.linear.params import LinearParams
from gkx.terms.config import TermConfig

__all__ = [
    "PR3_PRECOND_NAMES",
    "PR3_BLOCK_SOLVE_CHOICES",
    "Pr3BlockThomasFactors",
    "Pr3DenseFactors",
    "Pr3Factors",
    "build_pr3_apply",
    "build_pr3_factors",
    "check_block_structure",
    "pr3_alpha_from_symbol_bounds",
]

# Accepted spellings of this preconditioner, kept next to the implementation so
# the shift-invert dispatcher has one source for both validation and routing.
PR3_PRECOND_NAMES: frozenset[str] = frozenset(
    {"pr3-cm", "pr3_cm", "pr3", "peaceman-rachford", "peaceman_rachford"}
)
PR3_BLOCK_SOLVE_CHOICES: tuple[str, ...] = ("auto", "block-thomas", "dense")

# Number of Peaceman-Rachford double sweeps. Q7 measured one, two and three and
# adopted three: two costs 1.5x the iterations of three at every rung it ran
# (66 against 45 at (32,8,16), 245 against 164 at (96,8,24)) for 2/3 of the
# apply, and one ("adi-cm") does not converge at the production chain at all.
# It is a constant rather than a config field because no measurement supports a
# fourth sweep and the name of the preconditioner states the number.
PR3_SWEEPS: int = 3

# The exact solve is used only when the block really is l-tridiagonal plus rank
# one. Q21 measured the off-tridiagonal entries at exactly 0.0 against a block
# norm of 74.7, and the rank-one singular-value ratio at <= 5.3e-14; these
# thresholds are far above that and far below anything a genuinely dense or
# genuinely rank-two block could produce.
_TRIDIAGONAL_RELATIVE_TOL: float = 1.0e-10
_RANK_ONE_RATIO_TOL: float = 1.0e-8
# The block representation must reproduce the z-local operator itself. This is
# a refusal threshold, not a fallback one: a defect above it means the operator
# is not z-local and the split this module implements does not describe it.
_LOCALITY_DEFECT_TOL: float = 1.0e-8
# A Laguerre coupling entry counts as present above this fraction of the block
# norm. The mirror term puts every one of them on the Hermite diagonals 0 and
# +-1, so the measured half-width is 1 and the next occupied diagonal is at
# exactly 0.0; this threshold has three orders of magnitude of daylight either
# side and only has to separate "structurally absent" from "small".
_COUPLING_BAND_TOL: float = 1.0e-13


class Pr3BlockThomasFactors(NamedTuple):
    """Exact ``(D~ - s1)^-1``: SOLVAX block-Thomas in ``l``, plus Sherman-Morrison.

    ``schur`` is :class:`solvax.OperatorBlockTridiagFactors` under
    ``store="inverse"``: the Schur inverses ``Delta_a^-1`` stacked
    ``(nblocks, Nl, Nm, Nm)``, carrying the Laguerre couplings as Hermite
    diagonals in its ``params``. ``w`` and ``tu`` are the Sherman-Morrison row
    and the solved rank-one column, and ``denom`` is ``1 + w . tu`` per block.
    """

    schur: Any
    w: jnp.ndarray
    tu: jnp.ndarray
    denom: jnp.ndarray


class Pr3DenseFactors(NamedTuple):
    """Dense batched ``(D~ - s1)^-1``, one ``(Nl Nm, Nl Nm)`` inverse per block."""

    inverse: jnp.ndarray


class Pr3Factors(NamedTuple):
    """Everything :func:`build_pr3_apply` needs, as one pytree argument.

    ``blocks`` is either :class:`Pr3BlockThomasFactors` or
    :class:`Pr3DenseFactors`; which one it is decides the traced apply, and the
    choice is static because a pytree's structure is static under ``jit``.
    """

    alpha: jnp.ndarray
    s1: jnp.ndarray
    blocks: Pr3BlockThomasFactors | Pr3DenseFactors


# --------------------------------------------------------------------------- #
# block layout                                                                  #
# --------------------------------------------------------------------------- #
def _to_blocks(x: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    """``(ns, Nl, Nm, Ny, Nx, Nz)`` state -> ``(nblocks, Nl*Nm)`` z-local rows."""

    ns, nl, nm, ny, nx, nz = shape
    return jnp.transpose(x.reshape(shape), (0, 3, 4, 5, 1, 2)).reshape(
        ns * ny * nx * nz, nl * nm
    )


def _from_blocks(y: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
    """Inverse of :func:`_to_blocks`, returning a flat state vector."""

    ns, nl, nm, ny, nx, nz = shape
    return jnp.transpose(y.reshape(ns, ny, nx, nz, nl, nm), (0, 4, 5, 1, 2, 3)).reshape(
        -1
    )


def _block_shape(state_shape: tuple[int, ...]) -> tuple[int, ...]:
    """The six-axis ``(species, Nl, Nm, Ny, Nx, Nz)`` view of a state's shape.

    A cache with one species stores the state without the species axis, and the
    operator is built for exactly that layout. The block bookkeeping here wants
    the axis present; adding a length-one leading axis changes no element and no
    memory order, so the two views share one flat vector.
    """

    if len(state_shape) == 6:
        return state_shape
    return (1, *state_shape)


def _z_local_terms(term_cfg: TermConfig) -> TermConfig:
    """The ``D~`` half: every term except streaming and hypercollisions."""

    return replace(term_cfg, streaming=0.0, hypercollisions=0.0)


def _line_terms(term_cfg: TermConfig) -> TermConfig:
    """The ``S~`` half's terms for the Hermite-line builder.

    The builder reads only ``collisions``, ``hypercollisions`` and the drift
    switches: it inverts the streaming ladder against a diagonal built from the
    collision damping and the z-mean drift. Collisions belong to ``D~`` here, so
    they are switched off; the drift switches stay on, which is what puts the
    z-mean drift diagonal in this half.
    """

    return replace(term_cfg, collisions=0.0)


# --------------------------------------------------------------------------- #
# host-side construction                                                        #
# --------------------------------------------------------------------------- #
def _probe_z_local_blocks(
    apply: Callable[[jnp.ndarray], jnp.ndarray],
    shape: tuple[int, ...],
    *,
    batch: int,
) -> np.ndarray:
    """Dense ``(l, m)`` blocks of a z-local operator by ``Nl*Nm`` column probes.

    One probe sets a single ``(l, m)`` component to one at *every*
    ``(ky, kx, z)`` and reads the whole column of every block at once, which is
    only valid because the operator is z-local. The caller checks that.

    ``shape`` is always the six-axis block shape; ``apply`` receives whatever
    axis count the operator itself uses, which is one fewer when the cache
    carries a single implicit species.
    """

    ns, nl, nm, ny, nx, nz = shape
    block_size, nblocks = nl * nm, ns * ny * nx * nz
    batched = jax.jit(jax.vmap(apply))
    out = np.empty((nblocks, block_size, block_size), dtype=np.complex128)
    for start in range(0, block_size, batch):
        cols = np.arange(start, min(block_size, start + batch))
        probe = np.zeros((cols.size, *shape), dtype=np.complex128)
        probe[np.arange(cols.size), :, cols // nm, cols % nm] = 1.0
        image = np.asarray(batched(jnp.asarray(probe))).reshape(cols.size, *shape)
        image = image.transpose(0, 1, 4, 5, 6, 2, 3).reshape(cols.size, nblocks, -1)
        out[:, :, cols] = np.transpose(image, (1, 2, 0))
    return out


def _z_mean_drift_diagonal(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> np.ndarray:
    """The drift diagonal's z-mean, exactly as the Hermite line solve uses it.

    Read off the analytic diagonal the implicit preconditioner already builds
    rather than off an assembled matrix, so the quantity subtracted from ``D~``
    is bit-for-bit the quantity ``S~`` adds. With collisions and hypercollisions
    both off the builder's ``precond_full`` is ``1 / (1 - drift_diagonal)`` at
    ``dt = 1``, which inverts to the drift diagonal alone.
    """

    from gkx.solvers_linear_implicit import (
        _build_implicit_preconditioner_data,
        _prepare_implicit_state,
    )

    drift_only = replace(term_cfg, collisions=0.0, hypercollisions=0.0)
    state = _prepare_implicit_state(v0, 1.0, drift_only)
    data = _build_implicit_preconditioner_data(cache, params, state)
    drift = np.asarray(1.0 - jnp.reciprocal(data.precond_full))
    mean = np.broadcast_to(drift.mean(axis=-1, keepdims=True), drift.shape)
    ns, nl, nm, ny, nx, nz = (int(v) for v in drift.shape)
    return (
        np.transpose(np.asarray(mean), (0, 3, 4, 5, 1, 2))
        .reshape(ns * ny * nx * nz, nl * nm)
        .astype(np.complex128)
    )


def check_block_structure(
    with_phi: np.ndarray, no_phi: np.ndarray, nl: int, nm: int
) -> dict[str, float]:
    """Measure the two properties the exact z-block solve relies on.

    ``off_tridiagonal_max`` is the largest entry of the field-free block outside
    its Laguerre tridiagonal, and ``rank_one_ratio_max`` the largest
    second-to-first singular-value ratio of the field part. Both are reported
    against ``block_norm_max`` so a caller can judge them relatively.

    ``coupling_halfwidth`` is the Hermite half-width of the two Laguerre
    couplings: how far off its own diagonal the ``l -> l +- 1`` block reaches in
    the Hermite index. The banded storage pays for itself while ``2p+1 <= Nm``,
    so this is a third fallback condition and not only a diagnostic.
    """

    view = no_phi.reshape(no_phi.shape[0], nl, nm, nl, nm)
    off = 0.0
    for a in range(nl):
        for b in range(nl):
            if abs(a - b) > 1:
                off = max(off, float(np.abs(view[:, a, :, b, :]).max()))
    field = with_phi - no_phi
    singular = np.linalg.svd(field, compute_uv=False)
    norm = float(np.abs(no_phi).max())
    _diag, lower, upper = _laguerre_bands(no_phi, nl, nm, 0.0)
    return {
        "off_tridiagonal_max": off,
        "block_norm_max": norm,
        "rank_one_s0_max": float(singular[:, 0].max()),
        "rank_one_s1_max": float(singular[:, 1].max()),
        "rank_one_ratio_max": float(
            (singular[:, 1] / np.maximum(singular[:, 0], 1.0e-300)).max()
        ),
        "coupling_halfwidth": float(
            _coupling_halfwidth(lower, upper, max(norm, 1.0e-300))
        ),
    }


def _rank_one_factors(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``field[b] = u[b] w[b]^T`` taken from each block's largest-modulus pivot.

    A block whose field part is identically zero has no pivot; it is not a
    degenerate case but the ordinary one for every row the field solve masks
    out (the dealiased band, and the zonal rows of a grid that carries them).
    Such a block gets ``u = w = 0``, which makes the Sherman-Morrison
    denominator exactly 1 and the correction exactly nothing -- the right
    answer, where dividing by the absent pivot would produce a NaN and send the
    whole build to the dense fallback.
    """

    nblocks = field.shape[0]
    pivot = np.abs(field).reshape(nblocks, -1).argmax(axis=1)
    rows, cols = np.unravel_index(pivot, field.shape[1:])
    index = np.arange(nblocks)
    peak = field[index, rows, cols]
    safe = np.where(peak == 0.0, 1.0, peak)
    u = field[index, :, cols]
    w = field[index, rows, :] / safe[:, None]
    return u, w


def _shift(values: jnp.ndarray, offset: int, axis: int) -> jnp.ndarray:
    """``v[i] -> v[i + offset]`` along ``axis``, zero filled at both ends."""

    if offset == 0:
        return values
    index: list[Any] = [slice(None)] * values.ndim
    index[axis] = slice(offset, None) if offset > 0 else slice(None, offset)
    kept = values[tuple(index)]
    pad_shape = list(values.shape)
    pad_shape[axis] = abs(offset)
    pad = jnp.zeros(pad_shape, values.dtype)
    parts = [kept, pad] if offset > 0 else [pad, kept]
    return jnp.concatenate(parts, axis=axis)


@functools.lru_cache(maxsize=None)
def _coupling_action(halfwidth: int) -> Callable[..., jnp.ndarray]:
    """SOLVAX's coupling contract for Hermite-banded Laguerre couplings.

    ``couple(params, a, Z, which=..., transpose=...)`` returns ``L_a Z`` or
    ``U_a Z`` for a 2-D ``Z``, reading the coupling as ``2p+1`` Hermite
    diagonals: ``(C Z)[i] = sum_o band[p+o, i] Z[i+o]``, and transposed
    ``(C^T Z)[i] = sum_o band[p-o, i+o] Z[i+o]``.

    Cached on the half-width so that repeated builds hand ``jit`` the *same*
    function object. SOLVAX carries ``couple`` as static pytree metadata, so a
    fresh closure per shift would recompile the factorization every time.
    """

    def couple(
        bands: tuple[jnp.ndarray, jnp.ndarray],
        index: jnp.ndarray,
        operand: jnp.ndarray,
        *,
        which: str,
        transpose: bool,
    ) -> jnp.ndarray:
        band = bands[0] if which == "lower" else bands[1]
        row = band[index]
        out = jnp.zeros_like(operand)
        for offset in range(-halfwidth, halfwidth + 1):
            weight = (
                _shift(row[halfwidth - offset], offset, 0)
                if transpose
                else row[halfwidth + offset]
            )
            out = out + weight[:, None] * _shift(operand, offset, 0)
        return out

    return couple


def _laguerre_bands(
    no_phi: np.ndarray, nl: int, nm: int, shift: complex
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``T - shift I`` as diagonal blocks and the two Laguerre couplings."""

    nblocks = no_phi.shape[0]
    view = no_phi.reshape(nblocks, nl, nm, nl, nm)
    eye = np.eye(nm, dtype=np.complex128)
    diag = np.stack([view[:, a, :, a, :] for a in range(nl)], axis=1) - shift * eye
    lower = np.zeros((nblocks, nl, nm, nm), dtype=np.complex128)
    upper = np.zeros_like(lower)
    for a in range(1, nl):
        lower[:, a] = view[:, a, :, a - 1, :]
    for a in range(nl - 1):
        upper[:, a] = view[:, a, :, a + 1, :]
    return diag, lower, upper


def _coupling_halfwidth(lower: np.ndarray, upper: np.ndarray, scale: float) -> int:
    """Smallest ``p`` holding every coupling entry inside ``|i - j| <= p``."""

    nm = lower.shape[-1]
    out = 0
    for offset in range(-(nm - 1), nm):
        for band in (lower, upper):
            entry = float(
                np.abs(np.diagonal(band, offset=offset, axis1=-2, axis2=-1)).max()
            )
            if entry > _COUPLING_BAND_TOL * scale:
                out = max(out, abs(offset))
    return out


def _hermite_band(blocks: np.ndarray, halfwidth: int) -> np.ndarray:
    """``(..., Nm, Nm)`` -> ``(..., 2p+1, Nm)`` with ``band[p+o, i] = C[i, i+o]``."""

    nm = blocks.shape[-1]
    out = np.zeros((*blocks.shape[:-2], 2 * halfwidth + 1, nm), dtype=np.complex128)
    rows = np.arange(nm)
    for offset in range(-halfwidth, halfwidth + 1):
        cols = rows + offset
        inside = (cols >= 0) & (cols < nm)
        out[..., halfwidth + offset, rows[inside]] = blocks[
            ..., rows[inside], cols[inside]
        ]
    return out


def _block_thomas_factors(
    no_phi: np.ndarray, nl: int, nm: int, shift: complex, halfwidth: int
) -> Any:
    """SOLVAX's Schur elimination of the l-tridiagonal part, one shift.

    ``store="inverse"`` is the storage that matches what this preconditioner
    does with the factors: every application is a matrix product against
    ``Delta_a^-1`` rather than a pair of triangular solves, which is what the
    superseded host factorization stored and measurably the cheaper of the two
    here. It costs SOLVAX a transposed coupling action per elimination step,
    which the banded action supplies.
    """

    diag, lower, upper = _laguerre_bands(no_phi, nl, nm, shift)
    couple = _coupling_action(halfwidth)

    def factor(blocks: jnp.ndarray, bands: tuple[jnp.ndarray, ...]) -> Any:
        return solvax.block_thomas_factor_ops(blocks, couple, bands, store="inverse")

    # SOLVAX eliminates in the precision of the arrays it is handed, and a GKX
    # runtime that has not switched on x64 canonicalizes complex128 down to
    # complex64 at the ``jnp.asarray`` boundary. The superseded host
    # factorization was complex128 whatever the runtime precision was, so the
    # elimination is pinned to it here and its result is re-canonicalized on
    # the way out: the stored factors keep exactly the dtype they had before,
    # and only the arithmetic that produces them is protected.
    with jax.enable_x64(True):
        factors = jax.jit(jax.vmap(factor))(
            jnp.asarray(diag),
            (
                jnp.asarray(_hermite_band(lower, halfwidth)),
                jnp.asarray(_hermite_band(upper, halfwidth)),
            ),
        )
        factors = jax.tree.map(np.asarray, factors)
    factors = jax.tree.map(jnp.asarray, factors)
    return replace(factors, work_dtype=factors.blocks.dtype)


def _block_thomas_substitute(
    factors: Any, nl: int, nm: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """``x -> (T - shift)^-1 x``, SOLVAX's substitution unrolled over ``l``.

    These are exactly the two recurrences :func:`solvax.block_thomas_solve_ops`
    documents and runs -- a downward sweep forming
    ``sigma_a = b_a - U_a Delta_{a+1}^-1 sigma_{a+1}`` and an upward one forming
    ``x_a = Delta_a^-1 (sigma_a - L_a x_{a-1})`` -- against the same factors, and
    the two agree bitwise, which a test pins.

    SOLVAX runs them as ``lax.scan``s because that is what keeps *its* solve
    transposable on JAX releases before 0.10. GKX only ever applies this
    preconditioner forwards and never differentiates through it, so it does not
    need that property, and at the Laguerre block counts this operator has it
    costs real time: a scan of this recurrence is 1.6x to 1.9x slower than the
    unrolled form at ``Nl`` of 6 to 16 on CPU, which is why SOLVAX's own
    stored-band solve unrolls for block counts in this range as well.
    """

    inverses = factors.blocks  # (nblocks, Nl, Nm, Nm), Delta_a^-1
    lower, upper = factors.params  # (nblocks, Nl, 2p+1, Nm)
    halfwidth = (lower.shape[-2] - 1) // 2

    def coupled(band: jnp.ndarray, a: int, rows: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(rows)
        for offset in range(-halfwidth, halfwidth + 1):
            out = out + band[:, a, halfwidth + offset] * _shift(rows, offset, -1)
        return out

    def solve(x: jnp.ndarray) -> jnp.ndarray:
        rows = x.reshape(x.shape[0], nl, nm)
        sigma = [rows[:, nl - 1]]
        for a in range(nl - 2, -1, -1):
            forward = jnp.einsum("bij,bj->bi", inverses[:, a + 1], sigma[-1])
            sigma.append(rows[:, a] - coupled(upper, a, forward))
        sigma.reverse()
        out = [jnp.einsum("bij,bj->bi", inverses[:, 0], sigma[0])]
        for a in range(1, nl):
            out.append(
                jnp.einsum(
                    "bij,bj->bi", inverses[:, a], sigma[a] - coupled(lower, a, out[-1])
                )
            )
        return jnp.stack(out, axis=1).reshape(x.shape[0], nl * nm)

    return solve


def pr3_alpha_from_symbol_bounds(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    blocks: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Q7's scalar rule ``alpha = -sqrt(s1 d)`` from the two halves' symbols.

    ``s1`` is the streaming-plus-hypercollision symbol bound at the smallest
    non-zero ``|kz|`` and ``d`` the spectral radius of the ``D~`` blocks. Q7
    measured this scalar as good as a per-rung scan at ``Nz = 96`` and 1.2x
    worse at ``(64, 8, 32)``, and rejected per-``kz`` parameters outright.
    """

    from gkx.solvers_linear_implicit import (
        _build_implicit_preconditioner_data,
        _prepare_implicit_state,
    )

    state = _prepare_implicit_state(v0, 1.0, _line_terms(term_cfg))
    data = _build_implicit_preconditioner_data(cache, params, state)
    sqrt_m = np.asarray(data.sqrt_m_line)
    sqrt_p = np.asarray(data.sqrt_p_line).copy()
    sqrt_p[-1] = 0.0
    ladder = np.diag(sqrt_m[1:], -1) + np.diag(sqrt_p[:-1], 1)
    x_max = float(np.abs(np.linalg.eigvals(ladder)).max())
    coefficient = abs(
        float(np.asarray(data.w_stream))
        * float(np.asarray(params.kpar_scale))
        * float(np.asarray(data.vth).reshape(-1)[0])
    )
    hyper_max = float(np.asarray(data.hyper_kz).max())
    kz = np.abs(np.asarray(cache.kz))
    symbol = coefficient * kz * x_max + hyper_max * kz
    unique = np.sort(np.unique(symbol))
    s1_bound = float(unique[1] if unique.size > 1 else unique[0])
    d_rho = _spectral_radius(blocks)
    alpha = -float(np.sqrt(max(s1_bound * d_rho, 0.0)))
    if not np.isfinite(alpha) or alpha == 0.0:
        alpha = -1.0
    return alpha, {
        "symbol_s1": s1_bound,
        "symbol_s_max": float(symbol.max()),
        "block_spectral_radius": d_rho,
        "hermite_ladder_radius": x_max,
    }


def _spectral_radius(blocks: np.ndarray, iterations: int = 60) -> float:
    """Power iteration on the stacked blocks; the largest per-block Rayleigh."""

    rng = np.random.default_rng(0)
    v = rng.standard_normal(blocks.shape[:2]) + 0j
    for _ in range(iterations):
        v = np.einsum("bij,bj->bi", blocks, v)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(norm > 0.0, norm, 1.0)
    return float(np.abs(np.einsum("bi,bij,bj->b", v.conj(), blocks, v)).max())


def build_pr3_factors(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    sigma: Any,
    *,
    alpha: float | complex | None = None,
    block_solve: str = "auto",
    probe_batch: int = 16,
) -> tuple[Pr3Factors, dict[str, Any]]:
    """Build ``pr3-cm``'s host-side factors for one shift.

    Returns the factor pytree and a metadata dictionary recording the structural
    measurements, which solve was chosen and why, and the factors' size.
    """

    if block_solve not in PR3_BLOCK_SOLVE_CHOICES:
        raise ValueError(
            f"shift_precond_block_solve must be one of "
            f"{', '.join(PR3_BLOCK_SOLVE_CHOICES)}, got {block_solve!r}"
        )
    if v0.ndim not in (5, 6):
        raise ValueError(
            "the pr3-cm preconditioner needs a (species, Nl, Nm, Ny, Nx, Nz) "
            "state, or the same without the species axis; got shape "
            f"{tuple(v0.shape)}"
        )
    from gkx.solvers_linear_krylov_algorithms import _apply_operator

    state_shape = tuple(int(s) for s in v0.shape)
    shape = _block_shape(state_shape)
    ns, nl, nm, ny, nx, nz = shape
    nblocks = ns * ny * nx * nz
    seed = v0
    local_cfg = _z_local_terms(term_cfg)

    def local_apply(state: jnp.ndarray) -> jnp.ndarray:
        return _apply_operator(state.reshape(state_shape), cache, params, local_cfg)

    with_phi = _probe_z_local_blocks(local_apply, shape, batch=probe_batch)
    locality = _block_representation_defect(local_apply, with_phi, shape)
    if not (locality <= _LOCALITY_DEFECT_TOL):
        raise ValueError(
            "the pr3-cm preconditioner splits the operator into a spectral "
            "streaming half and a z-local half, and this operator's non-"
            "streaming part is not z-local: a dense block representation "
            f"reproduces it only to relative {locality:.3e} against a "
            f"{_LOCALITY_DEFECT_TOL:.0e} tolerance. Use shift_preconditioner="
            "'hermite-line' or 'field-corrected' instead."
        )
    mean_drift = _z_mean_drift_diagonal(seed, cache, params, term_cfg)
    diagonal = np.arange(nl * nm)
    with_phi[:, diagonal, diagonal] -= mean_drift

    alpha_meta: dict[str, float] = {}
    alpha_value: complex
    if alpha is None:
        scalar, alpha_meta = pr3_alpha_from_symbol_bounds(
            seed, cache, params, term_cfg, with_phi
        )
        alpha_value = complex(scalar)
    else:
        alpha_value = complex(alpha)
    if alpha_value == 0:
        raise ValueError("the pr3-cm parameter alpha must be non-zero")
    sigma_value = complex(np.asarray(sigma))
    s1 = sigma_value / 2.0 - alpha_value

    structure: dict[str, float] | None = None
    blocks_pytree: Pr3BlockThomasFactors | Pr3DenseFactors | None = None
    reason = "dense batched inverse requested"
    if block_solve != "dense":
        no_phi = _probe_z_local_blocks(
            lambda state: _apply_operator_without_fields(
                state.reshape(state_shape), cache, params, local_cfg
            ),
            shape,
            batch=probe_batch,
        )
        no_phi[:, diagonal, diagonal] -= mean_drift
        structure = check_block_structure(with_phi, no_phi, nl, nm)
        scale = max(structure["block_norm_max"], 1.0e-300)
        halfwidth = int(structure["coupling_halfwidth"])
        exact = (
            structure["off_tridiagonal_max"] <= _TRIDIAGONAL_RELATIVE_TOL * scale
            and structure["rank_one_ratio_max"] <= _RANK_ONE_RATIO_TOL
            and 2 * halfwidth + 1 <= nm
        )
        broken = (
            "the z-local block is not l-tridiagonal in the Laguerre index plus "
            f"a rank-one field part with Hermite-banded couplings: largest "
            f"off-tridiagonal entry {structure['off_tridiagonal_max']:.3e} "
            f"against block norm {structure['block_norm_max']:.3e}, largest "
            f"rank-one singular-value ratio {structure['rank_one_ratio_max']:.3e}"
            f", Laguerre coupling Hermite half-width {halfwidth} against Nm={nm}"
        )
        if not exact and block_solve == "block-thomas":
            raise ValueError(
                f"shift_precond_block_solve='block-thomas' was requested but "
                f"{broken}. Use 'auto', which falls back to the dense batched "
                "inverse and reports that it did."
            )
        if exact:
            u, w = _rank_one_factors(with_phi - no_phi)
            schur = _block_thomas_factors(no_phi, nl, nm, s1, halfwidth)
            solve = jax.jit(_block_thomas_substitute(schur, nl, nm))
            tu = np.asarray(solve(jnp.asarray(u)))
            denom = 1.0 + np.einsum("bi,bi->b", w, tu)
            if np.all(np.isfinite(denom)) and float(np.abs(denom).min()) > 0.0:
                reason = "block is l-tridiagonal plus rank one"
                blocks_pytree = Pr3BlockThomasFactors(
                    schur=schur,
                    w=jnp.asarray(w),
                    tu=jnp.asarray(tu),
                    denom=jnp.asarray(denom),
                )
            else:
                reason = (
                    "dense batched inverse: the Sherman-Morrison denominator "
                    "of the rank-one field part is singular"
                )
        else:
            reason = f"dense batched inverse: {broken}"
        del no_phi
    if blocks_pytree is None:
        eye = np.eye(nl * nm, dtype=np.complex128)
        blocks_pytree = Pr3DenseFactors(
            inverse=jnp.asarray(np.linalg.inv(with_phi - s1 * eye))
        )
    del with_phi

    factors = Pr3Factors(
        alpha=jnp.asarray(alpha_value, dtype=jnp.complex128),
        s1=jnp.asarray(s1, dtype=jnp.complex128),
        blocks=blocks_pytree,
    )
    nbytes = int(sum(int(np.asarray(a).size) * 16 for a in jax.tree.leaves(factors)))
    meta: dict[str, Any] = {
        "block_solve": (
            "block-thomas"
            if isinstance(blocks_pytree, Pr3BlockThomasFactors)
            else "dense"
        ),
        "block_solve_reason": reason,
        "alpha": [alpha_value.real, alpha_value.imag],
        "alpha_source": "symbol-bounds" if alpha is None else "explicit",
        "s1": [s1.real, s1.imag],
        "sweeps": PR3_SWEEPS,
        "blocks": nblocks,
        "block_size": nl * nm,
        "locality_defect": locality,
        "factor_bytes": nbytes,
        "structure": structure,
    }
    meta.update(alpha_meta)
    return factors, meta


def _apply_operator_without_fields(
    state: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
) -> jnp.ndarray:
    """``term_cfg`` applied with the self-consistent field response cancelled.

    Feeding ``external_phi = -phi(G)`` leaves the assembler's total field
    exactly zero, so what comes back is the field-free part of the operator and
    the difference against the full apply is the field term alone.
    """

    from gkx.terms.assembly import assemble_rhs_cached, compute_fields_cached

    fields = compute_fields_cached(
        state, cache, params, terms=term_cfg, use_custom_vjp=False
    )
    image, _fields = assemble_rhs_cached(
        state,
        cache,
        params,
        terms=term_cfg,
        use_custom_vjp=False,
        external_phi=-fields.phi,
    )
    return image


def _block_representation_defect(
    apply: Callable[[jnp.ndarray], jnp.ndarray],
    blocks: np.ndarray,
    shape: tuple[int, ...],
) -> float:
    """Relative error of the dense z-local blocks against the operator itself."""

    rng = np.random.default_rng(0)
    size = int(np.prod(np.asarray(shape)))
    probe = (rng.standard_normal(size) + 1j * rng.standard_normal(size)).astype(
        np.complex128
    )
    exact = np.asarray(apply(jnp.asarray(probe).reshape(shape))).reshape(-1)
    rows = np.asarray(_to_blocks(jnp.asarray(probe), shape))
    modelled = np.asarray(
        _from_blocks(
            jnp.asarray(np.einsum("bij,bj->bi", blocks, rows)),
            shape,
        )
    )
    scale = float(np.linalg.norm(exact))
    return float(np.linalg.norm(modelled - exact) / max(scale, 1.0e-300))


# --------------------------------------------------------------------------- #
# traced apply                                                                  #
# --------------------------------------------------------------------------- #
def _block_solver(
    blocks: Pr3BlockThomasFactors | Pr3DenseFactors, nl: int, nm: int
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """``(D~ - s1)^-1`` on stacked ``(nblocks, Nl*Nm)`` rows."""

    if isinstance(blocks, Pr3DenseFactors):
        inverse = blocks.inverse
        return lambda rows: jnp.einsum("bij,bj->bi", inverse, rows)

    tridiagonal = _block_thomas_substitute(blocks.schur, nl, nm)
    w, tu, denom = blocks.w, blocks.tu, blocks.denom

    def solve(rows: jnp.ndarray) -> jnp.ndarray:
        y = tridiagonal(rows)
        weight = jnp.einsum("bi,bi->b", w, y) / denom
        return y - weight[:, None] * tu

    return solve


def build_pr3_apply(
    v0: jnp.ndarray,
    cache: LinearCache,
    params: LinearParams,
    term_cfg: TermConfig,
    factors: Pr3Factors,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Three Peaceman-Rachford double sweeps, as a flat-vector right inverse.

    One sweep costs one Hermite-line solve and one z-block solve and applies the
    operator zero times; the recurrence carries ``(D~ - alpha) y`` forward so
    that no residual has to be recomputed.
    """

    from gkx.solvers_linear_implicit import _build_shifted_hermite_preconditioner

    shape = _block_shape(tuple(int(s) for s in v0.shape))
    nl, nm = shape[1], shape[2]
    line = _build_shifted_hermite_preconditioner(
        v0, cache, params, _line_terms(term_cfg), factors.s1
    )
    block = _block_solver(factors.blocks, nl, nm)
    two_alpha = 2.0 * factors.alpha

    def dsolve(x_flat: jnp.ndarray) -> jnp.ndarray:
        return _from_blocks(block(_to_blocks(x_flat, shape)), shape).reshape(
            x_flat.shape
        )

    def apply(x_flat: jnp.ndarray) -> jnp.ndarray:
        x = x_flat.reshape(-1).astype(jnp.result_type(x_flat.dtype, factors.s1.dtype))
        carry = jnp.zeros_like(x)  # (D~ - alpha) y of the previous sweep
        y = carry
        for _ in range(PR3_SWEEPS):
            half = line(x - carry)
            residual = carry + two_alpha * half
            y = dsolve(residual)
            carry = residual - two_alpha * y
        return y.reshape(x_flat.shape).astype(x_flat.dtype)

    return apply
