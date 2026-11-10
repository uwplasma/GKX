"""Integrating-factor (Lawson) explicit steppers: the magnetic drift is advanced exactly.

The curvature and grad-B drift is diagonal in ``(species, ky, kx, z)`` and acts on the
moments as ``-i (a C (x) I + b I (x) B)``, with ``a = tz * omega_d * w_curv * cv_d`` and
``b`` likewise from ``gb_d`` (both carry the ``ky`` and ``kx`` parts), ``C`` the truncated
Hermite curvature matrix and ``B`` the truncated Laguerre grad-B matrix.  ``C`` and ``B`` are
real symmetric, so ``C = V_c diag(c) V_c^T`` and ``B = V_b diag(beta) V_b^T`` are exact for the
truncation and ``exp(-i h (a C + b B))`` is a diagonal phase ``exp(-i h (a c_m + b beta_l))``
between two orthogonal moment transforms.  Every stage of an explicit Runge-Kutta
method then sees only ``N(G) = F(G) - L_d G``, with ``F`` the full right-hand side, so the
drift leaves the explicit stability limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
import operator
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from solvax.ode import RK4_2

from gkx.terms.assembly import _species_arrays, _term_weights
from gkx.terms.linear_terms import curvature_term, gradb_term


class Tableau(NamedTuple):
    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    e: tuple[float, ...] = ()


# Heun's third-order method, the tableau of the explicit ``rk3`` step.
_HEUN3 = Tableau(
    a=((1.0 / 3.0,), (0.0, 2.0 / 3.0)),
    b=(0.25, 0.0, 0.75),
    c=(0.0, 1.0 / 3.0, 2.0 / 3.0),
)
_TABLEAUX = {
    "rk3": _HEUN3,
    "rk4": Tableau(RK4_2.a, RK4_2.b, RK4_2.c),
    "rk4-pi": Tableau(RK4_2.a, RK4_2.b, RK4_2.c, RK4_2.e),
}


_HI = (
    jax.lax.Precision.HIGHEST
)  # the default GPU matmul (TF32) loses 1e-3 per transform
# Up to this many moments one dense (Nl Nm) x (Nl Nm) transform beats the two small ones.
_KRON_MOMENTS = 128


@dataclass(frozen=True)
class DriftFactor:
    """The drift exponential: moment transforms and the per-mode drift frequencies."""

    vc: jnp.ndarray  # (Nm, Nm) eigenvectors of the truncated Hermite curvature matrix
    vb: jnp.ndarray  # (Nl, Nl) eigenvectors of the truncated Laguerre grad-B matrix
    wc: jnp.ndarray  # (S, 1, Nm, ky, kx, z) a * eigenvalue of C
    wb: jnp.ndarray  # (S, Nl, 1, ky, kx, z) b * eigenvalue of B

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        """Moments to the eigenbasis of the drift."""
        return self._transform(x, False)

    def inverse(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._transform(x, True)

    def _transform(self, x: jnp.ndarray, inverse: bool) -> jnp.ndarray:
        vc, vb = (self.vc, self.vb) if inverse else (self.vc.T, self.vb.T)
        if x.shape[1] * x.shape[2] <= _KRON_MOMENTS:
            flat = x.reshape(x.shape[0], x.shape[1] * x.shape[2], -1)
            return jnp.einsum(
                "ab,sbn->san", jnp.kron(vb, vc), flat, precision=_HI
            ).reshape(x.shape)
        x = jnp.einsum("mj,slmyxz->sljyxz", vc.T, x, precision=_HI)
        return jnp.einsum("lk,sljyxz->skjyxz", vb.T, x, precision=_HI)

    def phases(self, h: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        """``exp(-i h omega)`` as the product of its Hermite and Laguerre factors."""
        return jnp.exp(-1j * h * self.wc), jnp.exp(-1j * h * self.wb)


def _moment_matrix(
    term: Callable[..., Any], coord: Any, n: int, axis: int
) -> np.ndarray:
    """Matrix of a moment-axis drift term, read off the production routine itself."""
    eye = jnp.eye(n).reshape((1, n, n, 1, 1) if axis == 1 else (n, 1, n, 1, 1))
    out = np.asarray(term(eye, coord), dtype=float)
    out = out[0, :, :, 0, 0] if axis == 1 else out[:, 0, :, 0, 0]
    return 0.5 * (out + out.T)


def build_drift_factor(
    cache: Any, params: Any, terms: Any, ns: int, real_dtype: Any
) -> DriftFactor:
    """Diagonalize the truncated drift operator of ``cache`` for ``terms``."""
    if getattr(cache, "mu_grid", False) or getattr(cache, "groups", None) is not None:
        raise NotImplementedError(
            "Lawson steppers need one moment layout and the Laguerre-coefficient basis"
        )
    nl, nm = int(cache.l.shape[0]), int(cache.m.shape[1])
    lam_c, vc = np.linalg.eigh(_moment_matrix(curvature_term, cache.m, nm, 1))
    lam_b, vb = np.linalg.eigh(_moment_matrix(gradb_term, cache.l, nl, 0))
    weights = _term_weights(params, terms, real_dtype)
    scale = (
        _species_arrays(params, ns, real_dtype).tz
        * jnp.asarray(params.omega_d_scale, dtype=real_dtype)
    )[:, None, None, None, None, None]
    rate_c = scale * weights.curvature * jnp.asarray(cache.cv_d, real_dtype)
    rate_b = scale * weights.gradb * jnp.asarray(cache.gb_d, real_dtype)
    return DriftFactor(
        vc=jnp.asarray(vc, real_dtype),
        vb=jnp.asarray(vb, real_dtype),
        wc=rate_c * jnp.asarray(lam_c, real_dtype)[None, None, :, None, None, None],
        wb=rate_b * jnp.asarray(lam_b, real_dtype)[None, :, None, None, None, None],
    )


def attach_drift_factor(
    rhs_fn: Callable[..., Any], prepared: Any, params: Any, terms: Any
):
    """Return ``rhs_fn`` carrying the drift exponential the ``lawson-*`` steppers read."""
    G0 = jnp.asarray(prepared.G0)
    ns = 1 if G0.ndim == 5 else G0.shape[0]
    factor = build_drift_factor(prepared.cache, params, terms, ns, prepared.real_dtype)

    def rhs_with_drift(state: jnp.ndarray) -> Any:
        return rhs_fn(state)

    rhs_with_drift.drift_factor = factor  # type: ignore[attr-defined]
    return rhs_with_drift


def lawson_step(
    G: jnp.ndarray,
    dG: jnp.ndarray,
    h: jnp.ndarray,
    *,
    method: str,
    factor: DriftFactor,
    rhs_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, object]],
    project_state: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One Lawson step from ``G`` with first-stage slope ``dG``: ``(G_new, err)``.

    Stage ``i`` is ``E^{c_i} w + h sum_j a_ij E^{c_i - c_j} N_j`` and the update
    ``E w + h sum_j b_j E^{1 - c_j} N_j``, in the eigenbasis ``w`` of the drift, where
    ``E = exp(h L_d)`` and ``N_j = F_j - L_d Y_j``.  ``err`` is the same sum with the
    embedded weights ``e`` (zero when the tableau has none), also in the eigenbasis, which
    is an orthogonal image of the moments.
    """
    if factor is None:
        raise ValueError(
            f"method '{method}' needs the explicit diagnostic runtime route"
        )
    tab = _TABLEAUX[method.removeprefix("lawson-")]
    lifted = G.ndim == 5
    up = (lambda x: x[None]) if lifted else (lambda x: x)
    down = (lambda x: x[0]) if lifted else (lambda x: x)
    phase_cache: dict[float, tuple[jnp.ndarray, jnp.ndarray]] = {}

    def move(x: jnp.ndarray, d: float) -> jnp.ndarray:
        """``exp(d h L_d) x``."""
        if d <= 1e-12:
            return x
        pc, pb = phase_cache.setdefault(round(d, 9), factor.phases(d * h))
        return x * pc * pb

    def slope(y: jnp.ndarray, dy: jnp.ndarray) -> jnp.ndarray:
        """``N = F - L_d Y`` in the eigenbasis."""
        return factor.forward(up(dy)) + 1j * (factor.wc + factor.wb) * y

    w = factor.forward(up(G))
    w_at: dict[float, jnp.ndarray] = {}
    ns = [slope(w, dG)]
    for row, ci in zip(tab.a, tab.c[1:]):
        if round(ci, 9) not in w_at:
            w_at[round(ci, 9)] = move(w, ci)
        y = w_at[round(ci, 9)]
        for j, aij in enumerate(row):
            if aij:
                y = y + (h * aij) * move(ns[j], ci - tab.c[j])
        dy, _ = rhs_fn(project_state(down(factor.inverse(y))))
        ns.append(slope(y, dy))

    def combine(weights: tuple[float, ...]) -> jnp.ndarray:
        return reduce(
            operator.add,
            [
                (h * wj) * move(ns[j], 1.0 - tab.c[j])
                for j, wj in enumerate(weights)
                if wj
            ],
        )

    w_new = move(w, 1.0) + combine(tab.b)
    err = combine(tab.e) if tab.e else jnp.zeros_like(w_new)
    return down(factor.inverse(w_new)), err
