"""Q21: the Laguerre block structure of the z-local D block, shared by the arms.

``Dc`` with the field term removed is exactly block-tridiagonal in the Laguerre
index with ``Nm x Nm`` blocks, and the field term is exactly rank one per
(kx, z) -- both checked by :func:`check_structure` wherever this is used. So
``Dc - s`` can be solved exactly by block-Thomas on the tridiagonal part plus a
Sherman-Morrison correction, at three ``Nm x Nm`` matvecs per Laguerre index and
``3 Nl Nm^2`` stored entries instead of the dense ``(Nl Nm)^2``.

:mod:`blockthomas` measures that apply against the dense inverse; :mod:`q21`
uses :func:`build_solver` to run a whole arm with it.
"""

import jax
import jax.numpy as jnp
import numpy as np


def block_view(blocks: np.ndarray, nl: int, nm: int) -> np.ndarray:
    return blocks.reshape(blocks.shape[0], nl, nm, nl, nm)


def check_structure(no_phi: np.ndarray, with_phi: np.ndarray, nl: int, nm: int) -> dict:
    """Exact tridiagonality without the field term, and rank one with it."""

    T = block_view(no_phi, nl, nm)
    off = 0.0
    for a in range(nl):
        for b in range(nl):
            if abs(a - b) > 1:
                off = max(off, float(np.abs(T[:, a, :, b, :]).max()))
    R = with_phi - no_phi
    sv = np.linalg.svd(R, compute_uv=False)
    return {
        "off_tridiagonal_max": off,
        "no_phi_max": float(np.abs(no_phi).max()),
        "rank1_s0_max": float(sv[:, 0].max()),
        "rank1_s1_max": float(sv[:, 1].max()),
        "rank1_ratio_max": float((sv[:, 1] / np.maximum(sv[:, 0], 1e-300)).max()),
    }


def rank_one_factors(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``R[b] = u[b] w[b]^T`` from the pivot of largest modulus (R is rank one)."""

    nb = R.shape[0]
    flat = np.abs(R).reshape(nb, -1).argmax(axis=1)
    rows, cols = np.unravel_index(flat, R.shape[1:])
    idx = np.arange(nb)
    u = R[idx, :, cols]
    w = R[idx, rows, :] / R[idx, rows, cols][:, None]
    return u, w


def factorize(T: np.ndarray, nl: int, nm: int, shift: complex) -> dict:
    """Block-Thomas factors of ``T - shift I``: Dinv, F = Dinv A, Q = Dinv C."""

    V = block_view(T, nl, nm).copy()
    eye = np.eye(nm, dtype=np.complex128)
    for a in range(nl):
        V[:, a, :, a, :] -= shift * eye
    nb = T.shape[0]
    Dinv = np.zeros((nl, nb, nm, nm), dtype=np.complex128)
    F = np.zeros((nl, nb, nm, nm), dtype=np.complex128)
    Q = np.zeros((nl, nb, nm, nm), dtype=np.complex128)
    D = V[:, 0, :, 0, :].copy()
    Dinv[0] = np.linalg.inv(D)
    for a in range(1, nl):
        A = V[:, a, :, a - 1, :]
        C = V[:, a - 1, :, a, :]
        Q[a - 1] = np.einsum("bij,bjk->bik", Dinv[a - 1], C, optimize=True)
        D = V[:, a, :, a, :] - np.einsum("bij,bjk->bik", A, Q[a - 1], optimize=True)
        Dinv[a] = np.linalg.inv(D)
        F[a] = np.einsum("bij,bjk->bik", Dinv[a], A, optimize=True)
    return {"Dinv": jnp.asarray(Dinv), "F": jnp.asarray(F), "Q": jnp.asarray(Q)}


def thomas_solver(fac: dict, nl: int, nm: int):
    """``x -> (T - shift)^-1 x`` on stacked blocks ``(nb, nl*nm)``."""

    Dinv, F, Q = fac["Dinv"], fac["F"], fac["Q"]

    def solve(x):
        xb = x.reshape(x.shape[0], nl, nm)
        y = [jnp.einsum("bij,bj->bi", Dinv[0], xb[:, 0])]
        for a in range(1, nl):
            y.append(
                jnp.einsum("bij,bj->bi", Dinv[a], xb[:, a])
                - jnp.einsum("bij,bj->bi", F[a], y[a - 1])
            )
        z = [None] * nl
        z[nl - 1] = y[nl - 1]
        for a in range(nl - 2, -1, -1):
            z[a] = y[a] - jnp.einsum("bij,bj->bi", Q[a], z[a + 1])
        return jnp.stack(z, axis=1).reshape(x.shape[0], nl * nm)

    return solve


def build_solver(with_phi, no_phi, nl, nm, shift):
    """Exact ``(Dc - shift)^-1`` on stacked blocks: factors plus a solver maker.

    Returns ``(factors, make)`` where ``factors`` is a pytree to hand to ``jit``
    as an argument (so the graph does not capture it as a constant, as in Q7's
    harness) and ``make(factors)`` returns the ``(nb, nl*nm) -> (nb, nl*nm)``
    solve.
    """

    u, w = rank_one_factors(with_phi - no_phi)
    fac = factorize(no_phi, nl, nm, shift)
    tsolve = thomas_solver(fac, nl, nm)
    Tu = np.asarray(jax.jit(tsolve)(jnp.asarray(u)))
    denom = 1.0 + np.einsum("bi,bi->b", w, Tu)
    factors = (
        fac["Dinv"],
        fac["F"],
        fac["Q"],
        jnp.asarray(w),
        jnp.asarray(Tu),
        jnp.asarray(denom),
    )

    def make(f):
        Dinv, F, Q, w_j, Tu_j, den_j = f
        solve = thomas_solver({"Dinv": Dinv, "F": F, "Q": Q}, nl, nm)

        def apply(xb):
            y = solve(xb)
            s = jnp.einsum("bi,bi->b", w_j, y) / den_j
            return y - s[:, None] * Tu_j

        return apply

    nbytes = int(sum(int(np.asarray(v).size) * 16 for v in factors))
    return (
        factors,
        make,
        {"bytes": nbytes, "denominator_min_abs": float(np.abs(denom).min())},
    )
